# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Design: Megatron FSDP v2 Implementation

---

## File Map

| File | Role in overlap |
|---|---|
| `fully_shard.py` | Public `fully_shard()` API and allocator selection |
| `fsdp_module.py` | `FSDPModule`, `_FSDPRootContext`, `_FSDPState`, `unshard()`, `reshard()`, `reduce_grad()` |
| `hooks.py` | Forward/backward hook registration and final callback |
| `param_group.py` | `ParameterGroup.unshard()`, `reduce_grad()`, `release_grad_buffer()`, `_init_buffers()` (memory optimization) |
| `dp_buffer.py` | Placement-shaped flat-buffer views and one-axis redistribution collectives |
| `allocator.py` | `BucketAllocator` hierarchy: `TemporaryBucketAllocator`, `StorageFreeingBucketAllocator`, `TracePoolAllocator` — pooled memory for unsharded parameter and gradient buffers |
| `mcore_fsdp_adapter.py` | `FullyShardedDataParallel.stop_communication()` — synchronizes ag_stream and rs_stream into main stream |
| `utils.py` | V2-to-v1 compatibility proxy used by the existing EP-overlap schedule |

The generic `combined_1f1b` schedule remains unchanged. `find_megatron_fsdp()`
returns the native v1 wrapper or a cached v2 compatibility proxy implementing
the same narrow schedule-facing interface.

---

## `_FSDPRootContext` — Shared Coordination Object

One instance is created by the root `FSDPModule` at `_init_fsdp_state()` time and stored as
`_fsdp_root_context` on **every** FSDP module (root and all children).

```python
@dataclass
class _FSDPRootContext:
    # --- CUDA streams ---
    ag_stream: torch.cuda.Stream   # all-gather (unshard) side stream
    rs_stream: torch.cuda.Stream   # reduce-scatter side stream
    # When the corresponding feature flag is False, these are set to
    # torch.cuda.current_stream() so stream-context switches become no-ops.

    # --- Temporary bucket allocator ---
    bucket_allocator: BucketAllocator
    # One allocator handles all temporary buckets. Allocation keys include
    # both parameter-group identity and buffer role.

    # --- Static execution order (set at init, never mutated) ---
    forward_order: List[FSDPModule]
    # Populated as: [m for m in root.modules() if isinstance(m, FSDPModule)]

    # --- Unshard prefetch tracking ---
    unshard_done_events: Dict[int, Optional[torch.cuda.Event]]
    # module_id -> Event: signals when that module's all-gather is complete.
    # None means "not yet launched" or "resharded"; the event persists after waits.

    # --- Reduce-scatter grad overlap tracking ---
    reduce_grad_buckets: Dict[int, List[Tuple[torch.cuda.Event, ParameterGroup]]]
    # module_id -> [(event, param_group), ...]
    # Each entry: event signals RS complete; param_group holds the grad buffer.

    # --- Feature flags ---
    enable_unshard_prefetch: bool
    enable_async_reduce_grad: bool

    # --- Activation recompute support ---
    backward_phase: bool = False
    # True from the root backward pre-hook until the final callback.

    backward_module: Optional[int] = None
    # ``id(module)`` of the FSDP module whose backward is pending next.
    # Derived from ``_reversed_order`` and ``backward_done_modules`` — NOT
    # set by any hook directly.  Updated by ``_advance_backward_module()``.

    backward_done_modules: set = field(default_factory=set)
    # Set of ``id(module)`` for FSDP modules whose backward has completed.
    # Populated in ``post_backward``, cleared in the root backward pre-hook.

    _reversed_order: List[FSDPModule] = field(default_factory=list)
    # ``list(reversed(forward_order))`` — precomputed backward processing order.

    def _advance_backward_module(self) -> None:
        """Set ``backward_module`` to the first module in ``_reversed_order``
        that is NOT in ``backward_done_modules``."""
        for m in self._reversed_order:
            if id(m) not in self.backward_done_modules:
                self.backward_module = id(m)
                return
        self.backward_module = None
```

### Initialization in `_init_fsdp_state()`

```python
bucket_allocator = StorageFreeingBucketAllocator()
module._init_named_param_groups(...)

forward_order = [child for child in self.modules() if isinstance(child, FSDPModule)]
root_context = _FSDPRootContext(
    ag_streams=resolve_axis_streams(all_gather_streams, enable_unshard_prefetch),
    rs_streams=resolve_axis_streams(reduce_scatter_streams, enable_async_reduce_grad),
    bucket_allocator=bucket_allocator,
    forward_order=forward_order,
    reduce_grad_buckets={id(m): [] for m in forward_order},
    unshard_done_events={id(m): None for m in forward_order},
    enable_unshard_prefetch=enable_unshard_prefetch,
    enable_async_reduce_grad=enable_async_reduce_grad,
)
# Root and children share one context and one bucket allocator:
for module in forward_order:
    for param_group in module._fsdp_param_groups:
        param_group.set_allocator(root_context.bucket_allocator)
for child in self.modules():
    if child is not self and isinstance(child, FSDPModule):
        child._fsdp_state._is_root = False
        setattr(child, "_fsdp_root_context", root_context)
```

`forward_order` is **static** (module tree topology, computed once). There is no first-pass
dynamic recording phase.

**Safety constraint.** `_init_fsdp_state()` must be called **before** any forward/backward pass
runs.  The method includes a runtime guard that rejects re-initialization if any child
FSDPModule is still unsharded (`unshard_done_events` live) or has pending reduce-scatter
operations (`reduce_grad_buckets` non-empty).  Violating this constraint would overwrite a
running module's `_fsdp_root_context` while its hooks are still firing, causing undefined
behavior.

### Meta-device materialization and initialization order

`_materialize_meta_module()` walks `named_modules()` in reverse order, materializing and
resetting leaf modules before their parents. Each `_apply` is non-recursive so a tensor is
materialized exactly once, while a parent `reset_parameters()` hook may safely inspect or
derive state from already initialized descendants. Buffer-only lazy modules remain untouched,
matching the v1 behavior.

---

## Feature 1: Unshard Prefetch

### Hook entry points

For callable hook contracts and Q&A, see [`hooks_api.md`](hooks_api.md). This
section owns the hook lifecycle summary.

| Phase | Hook API | Lifecycle responsibility |
|---|---|---|
| Forward pre-hook | `mfsdp_forward_pre_hook` | Enter forward phase, unshard parameters, and release stale gradient storage. |
| Forward post-hook | `mfsdp_post_forward_hook` | Record CUDA-graph outputs when applicable, then reshard after forward. |
| Backward pre-hook | `mfsdp_pre_backward_setup` | Enter backward phase, unshard parameters for backward, and enqueue or defer the final callback. |
| Backward post-hook | `mfsdp_post_backward_hook` | Reshard completed modules, reduce gradients, and advance backward-order tracking. |
| Final callback | `mfsdp_post_backward_final_callback` | Finish skipped post-backward work, drain async reductions, reset microbatch state, and finalize allocator/CUDA-graph transitions. |

```python
# _register_forward_pre_hook:
module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=False)

# _register_backward_pre_hook (called inside register_multi_grad_hook):
module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)
```

### `FSDPModule.unshard(async_op, bwd_pass)`

```python
caller_stream = torch.cuda.current_stream()
prefetch = ctx.get_prefetch_modules(self, bwd_pass=bwd_pass) if async_op else []
for module in [self] + prefetch:
    if all(pg.weights_are_unsharded() for pg in module._fsdp_param_groups):
        continue
    streams = (
        ctx.ag_streams
        if async_op
        else (caller_stream,) * module._fsdp_param_groups[0].mesh.ndim
    )
    ParameterGroup.unshard_weights(
        module._fsdp_param_groups, streams=streams, async_op=async_op
    )
    if async_op:
        ctx.unshard_done_events[id(module)] = ctx.ag_streams[-1].record_event()

if ctx.unshard_done_events[id(self)] is not None:
    ctx.unshard_done_events[id(self)].wait()

for param_names, param_group in self._named_param_groups:
    for name, param in zip(param_names, param_group.params):
        _replace_module_parameter(self, name, param)
```

**Stream ownership and buffer lifetime.** Each mesh-axis redistribution runs on its
configured stream. `ParameterGroup.unshard_weights()` binds the output and runs
mixed-precision post-processing on the terminal axis stream. The caller waits for the
module event before installing full parameters. A skipped prefetched module waits for
that event in `reshard()` before releasing its temporary full-weight lease.

**Stream ordering barrier.** When `async_op=True`, the caller stream is captured before
any stream switch inside `redistribute_buffers()`. The batch allocates final output
buffers and completes shard preparation first, then inserts
`ag_stream.wait_stream(caller_stream)` immediately before launching the all-gathers.
This ensures caller-stream allocations and writes are visible to the collective without
making `ag_stream` wait for future compute. The edge also makes `ag_stream` join a
full-iteration CUDA graph capture before the capture stream waits on the recorded
unshard event; otherwise CUDA reports `cudaErrorStreamCaptureIsolation` at the first
captured async unshard.

**NVTX profiling.** `unshard()`, `reshard()`, and `reduce_grad()` each push/pop a
`torch.cuda.nvtx` range (`"MFSDP unshard"`, `"MFSDP reshard"`, `"MFSDP reduce_grad"`)
for profiling visibility in tools like Nsight Systems.

**All-gather coalescing.** `FSDPModule.unshard()` delegates the module's ordered
parameter groups to `ParameterGroup.unshard_weights()`. The parameter-group operation
supplies their full replicated target to `DataParallelBuffer.redistribute_buffers()`.
The buffer planner derives mesh-axis order and groups buffers with the same process group,
dtype, device, and source placement. It completes the outer dimension before the inner
dimension, and each dimension uses one grouped launch when it contains multiple
compatible buffers and its process group has more than one rank. Buffer state determines
whether a placement needs a collective. With `async_ops=True`, the coalescing manager
owns the resulting `Work`; the async path calls
`coalescing_event.wait()` while `ag_stream` is current before advancing to the next
dimension or recording the module event.

### `get_prefetch_modules(module, bwd_pass, num_prefetch, require_outer_weight_all_gather)`

```python
order = list(reversed(ctx.forward_order)) if bwd_pass else ctx.forward_order
i = order.index(module)
candidates = order[i + 1 :]
if require_outer_weight_all_gather:
    candidates = [module for module in candidates if _uses_outer_weight_all_gather(module)]
return candidates[:num_prefetch]
```

The generic full-unshard path requests one next module. The HSDP outer-stage path
requests the configured number of eligible modules and only materializes their
persistent `[R, S]` weight placement.

### `FSDPModule.reshard()`

```python
unshard_event = ctx.unshard_done_events[id(self)]
if unshard_event is not None:
    unshard_event.wait()  # skipped prefetch: join AG before freeing its buffer

for param_names, param_group in self._named_param_groups:
    param_group.reshard_weight()                    # unbinds and releases full weights
    for name, dist_param in zip(param_names, param_group.optimizer_params):
        _replace_module_parameter(self, name, dist_param)   # reinstall sharded DTensor
ctx.unshard_done_events[id(self)] = None    # reset so next iteration can prefetch again
pending_post.clear()                         # discard any unused prefetched post phase
```

The conditional wait handles prefetched modules that are skipped by model control flow.
Their caller-owned output buffer cannot be freed until the side-stream all-gather has
completed. Modules that reached their pre-hook already consumed the pending phase, so the
normal compute-to-free path does not add a redundant wait.

---

## Feature 2: Reduce-Scatter Grad Overlap

### Hook entry point

Inside the `post_backward` closure registered by `_register_backward_hook`:

```python
module.reshard()
module.reduce_grad(async_op=ctx.enable_async_reduce_grad)
module.post_backward_issued = True
```

### `FSDPModule.reduce_grad(async_op)`

```python
def reduce_grad(self, async_op: bool = False):
    stream = ctx.rs_stream if async_op else torch.cuda.current_stream()

    # --- Step 1: Sliding drain — free grad buffers 2 positions back in backward order ---
    if async_op:
        backward_order = list(reversed(ctx.forward_order))
        for i, module in enumerate(backward_order):
            if i - 2 >= 0:
                for event, param_group in drain(ctx.reduce_grad_buckets[id(backward_order[i-2])]):
                    event.wait()
                    param_group.release_grad_buffer()
                    #   → deletes param.main_grad views (prevents TE grad-accum-fusion leak)
                    #   → releases the ParameterGroup-owned full-grad lease
            if module is self: break

    # --- Step 2: Stage .grad → main_grad_buffer ---
    for param_names, param_group in self._named_param_groups:
        if not param_group.requires_grad: continue

        accumulate_full_grad = param_group.full_grad_has_value
        stage_tensors = []
        stage_sources = []
        zero_tensors = []
        params_with_grad = []
        for param in param_group.params:
            grad = param.grad
            if grad is not None:
                params_with_grad.append(param)
            if getattr(param, "grad_added_to_main_grad", False):
                continue
            if grad is None:
                if not accumulate_full_grad:
                    zero_tensors.append(param.get_main_grad())
            else:
                stage_tensors.append(param.get_main_grad())
                stage_sources.append(grad.detach())

        stage_on_rs_stream = async_op and getattr(
            self._fsdp_state, "enable_full_iteration_cuda_graph", False
        )
        if stage_on_rs_stream:
            stream.wait_stream(torch.cuda.current_stream())
            for source in stage_sources:
                source.record_stream(stream)
            with torch.cuda.stream(stream):
                if stage_tensors:
                    op = torch._foreach_add_ if accumulate_full_grad else torch._foreach_copy_
                    op(stage_tensors, stage_sources)
                if zero_tensors:
                    torch._foreach_zero_(zero_tensors)
        else:
            if stage_tensors:
                op = torch._foreach_add_ if accumulate_full_grad else torch._foreach_copy_
                op(stage_tensors, stage_sources)
            if zero_tensors:
                torch._foreach_zero_(zero_tensors)

        for param in params_with_grad:
            del param.grad

        # --- Step 3: Reduce-scatter on rs_stream ---
        if async_op:
            param_group.reduce_grad(stream=stream)
            #   → ParameterGroup owns full-grad/output/workspace leases
            #   → DataParallelBuffer.redistribute() runs the selected collective
            #   → ParameterGroup commits or accumulates the result
            event = stream.record_event()
            ctx.reduce_grad_buckets[id(self)].append((event, param_group))
            # param_group.release_grad_buffer() is NOT called here; deferred until drain/final CB
        else:
            param_group.reduce_grad()
            param_group.release_grad_buffer()

        # --- Step 4: Install dist_grad on dist_param (runs in stream context) ---
        for name, param, dist_param, dist_grad in zip(
            param_names, param_group.params, param_group.optimizer_params, param_group.optimizer_grads
        ):
            if param.requires_grad and dist_grad is not None:
                with torch.cuda.stream(stream):
                    dist_grad = dist_grad.to(dist_param.dtype)  # dtype cast on rs_stream
                setattr(dist_param, "grad", dist_grad)          # Python ref, no GPU dependency
```

**Key design point — `DataParallelBuffer.redistribute()` has no asynchronous-work
handle.** The primitive executes synchronously within the current stream, while its
caller owns stream ordering and tensor lifetime. Eager, per-module CUDA graph, and
synchronous-reduction paths stage gradients on the caller stream; then
`ParameterGroup.reduce_grad(stream=...)` inserts one wait before preprocessing and
redistribution on `rs_stream`.

Full-iteration CUDA graphs instead dispatch ordinary async gradient add/copy/zero
staging to `rs_stream` immediately before reduction. The stream first waits for
backward, and detached `.grad` sources record `rs_stream` so their storage remains live
after Python references are deleted. This preserves overlap with the next module's
backward compute. Temporary buffers remain owned by `ParameterGroup` until the module's
reduction event completes.

**`grad_added_to_main_grad` and `overwrite_main_grad` flags:**
When TransformerEngine's `gradient_accumulation_fusion` is active, the backward kernel writes
directly into `param.main_grad` (bypassing `.grad`). Two flags coordinate this:

- **`grad_added_to_main_grad`**: Set to `False` in `pre_backward_hook` before each backward
  pass; the kernel sets it to `True` after writing. In `reduce_grad`, the `zero_()` call is
  skipped when `True` to preserve the fused-gradient value.

- **`overwrite_main_grad`**: Derived from gradient state in `pre_backward_hook`.
  ZeRO-2/FSDP/HSDP use fresh full-gradient scratch and therefore overwrite on
  every microbatch. DDP and ZeRO-1 overwrite while the phase is `EMPTY`, then
  let TE accumulate into persistent full-gradient storage while the phase is
  `ACCUMULATING`.

### Sliding Drain: The `i-2` Rule

The drain loop ensures at most **2 modules' gradient buffers** are live at any time:

```
Backward processing order (reversed forward):
  layer[N]   ← current (i=0): i-2=-2  → no drain
  layer[N-1] ← current (i=1): i-2=-1  → no drain
  layer[N-2] ← current (i=2): i-2=0   → drain layer[N]    (i-2=0)
  layer[N-3] ← current (i=3): i-2=1   → drain layer[N-1]  (i-2=1)
  ...
```

By the time RS for `layer[N-2]` starts, `layer[N]`'s RS event is expected to be done
(two backward steps of compute have elapsed). `event.wait()` makes this explicit and safe
even if the timing estimate is wrong.

### `_post_backward_final_callback`

Registered on the root by `_register_post_backward_final_callback()` via
`Variable._execution_engine.queue_callback`. Fires after all autograd ops complete.

```python
def _post_backward_final_callback(root_state, root_module):
    ctx = root_module._fsdp_root_context
    stream = ctx.rs_stream

    # Handle modules whose post_backward hook was never triggered
    # (e.g. modules with no grad-requiring inputs on this micro-batch)
    for module in reversed(ctx.forward_order):
        if module.post_backward_issued:
            continue
        module.reshard()
        module.reduce_grad(async_op=ctx.enable_async_reduce_grad)

    # Drain ALL remaining buckets (anything not drained by the sliding rule above)
    for buckets in ctx.reduce_grad_buckets.values():
        while buckets:
            event, param_group = buckets.pop()
            event.wait()
            param_group.release_grad_buffer()

    # Ensure main stream sees all rs_stream work before optimizer step
    torch.cuda.current_stream().wait_stream(stream)

    root_state._post_backward_callback_queued = False
```

---

## ZeRO-1 and ZeRO-2 Workflow

### No-Shard (`no_shard`)

1. Forward and backward read replicated `model_weight_buffer`; no parameter
   all-gather is needed beyond the normal buffer rebind.
2. Backward accumulates local full gradients across micro-batches. Post-backward
   reduction skips `no_shard`, matching ZeRO-1's delayed-sync behavior.
3. `finish_grad_sync()` performs one delayed full-buffer all-reduce for each
   `no_shard` grad buffer. The optimizer consumes replicated DTensor grads.

`optim` and `optim_grads` keep compute weights replicated but still expose
optimizer-facing DTensor shards through `optimizer_params`.

### ZeRO-1 (`optim`)

1. Forward and backward read replicated `model_weight_buffer`; no parameter
   all-gather is needed in the steady state.
2. Backward writes local gradients into the replicated `main_grad_buffer`.
   Post-backward reduce-scatter skips `optim` groups, so gradients remain full
   replicas across local gradient accumulation.
3. `finish_grad_sync()` performs one delayed reduce-scatter for each `optim`
   grad buffer. The reduce-scatter output is written into this rank's virtual
   shard, which is what the optimizer consumes through `optimizer_grads`.
4. The optimizer updates this rank's sharded `main_weight_buffer` view. After
   `copy_main_weights_to_model_weights()`, the next forward refreshes the
   replicated compute weights from those updated shards.

### ZeRO-2 (`optim_grads`)

1. Forward and backward also read replicated `model_weight_buffer`; no parameter
   all-gather is needed in the steady state.
2. Backward writes gradients into the full grad buffer acquired by
   `ParameterGroup`; persistent replicated storage is reused when it already
   contains the full value.
3. The post-backward hook reduce-scatters that temporary full buffer and
   accumulates the result into the persistent sharded `main_grad_buffer.data`.
   With overlap enabled, this reduce-scatter is launched on `ctx.rs_stream` and
   the normal sliding drain/final callback releases the temporary buffer after
   its event completes.
4. `finish_grad_sync()` only waits for outstanding `rs_stream` work for
   `optim_grads`; it does not launch another reduce-scatter.
5. The optimizer updates this rank's sharded `main_weight_buffer` view. The next
   forward refreshes replicated compute weights the same way as ZeRO-1.

### Replicated Weight Refresh

For ZeRO-1/2, `copy_main_weights_to_model_weights()` marks the replicated
`DataParallelBuffer` in a shard view when `main_weight_buffer` is sharded and
`model_weight_buffer` is replicated. The next normal unshard for that buffer
asks `ParameterGroup.unshard_weights()` to refresh any replicated
buffer before compute:

1. Non-FP8 weights copy this rank's updated main-weight shard into the matching
   slice of the replicated model-weight buffer.
2. FP8 weights quantize the local FP32 main-weight shard into the local FP8
   model-weight shard first; MXFP8 selects the transpose-buffer shard view as well.
3. `ParameterGroup.unshard_weights()` privately selects the shard buffer,
   asks `DataParallelBuffer.redistribute_buffers()` to gather the updated shards
   into the full replicated compute buffer on every rank, and binds the result
   for the current compute phase.

The rowwise/model buffer is refreshed on forward unshard. For MXFP8, the
transpose buffer is refreshed on backward unshard, where the mixed-precision
policy privately selects the backward representation.

The final backward callback arms `ctx.model_weight_refresh_pending` only when
`is_last_backward` is true. An optimizer integration that explicitly calls
`_copy_main_weights_to_model_weights()` consumes this flag immediately. For a
plain PyTorch optimizer, the next non-recompute root pre-forward hook consumes
the flag and installs the optimized weights before launching any parameter
unshard or prefetch. This keeps weight installation out of `zero_grad()`, skips
intermediate gradient-accumulation micro-batches, and prevents activation
checkpoint recompute from consuming the pending optimizer boundary.

---

## Feature 3: Activation Recomputation (Gradient Checkpointing)

### Problem

When activation checkpointing re-runs a forward pass during backward, the FSDP
forward hooks fire again. Without mitigation this causes two problems:

1. **Redundant all-gather**: `forward_pre_hook` → `unshard()` launches a second
   all-gather even though parameters are already unsharded.
2. **Premature reshard**: `forward_hook` → `reshard()` releases the unsharded
   parameter buffer before backward gradient computation has consumed it.

The baseline Megatron-FSDP addresses this by switching submodules into a
pre-backward mode before backprop (`megatron_fsdp.py:900-938`).

### Solution Overview

Two mechanisms:

| Mechanism | Effect |
|---|---|
| **Derived `backward_module`** | `_advance_backward_module()` scans `_reversed_order` for the first module **not** in `backward_done_modules`. This identifies the pending module even when activation recompute fires **before** any layer's `pre_backward_hook` (which is always the case — the checkpoint wrapper triggers recompute, then backward flows through the recomputed graph). |
| **Buffer readiness check** | `model_weights_are_unsharded(bwd_pass=...)` skips redundant all-gathers for the representations required by the current forward/backward phase. |

The `backward_phase` flag gates the forward post-hook check; `backward_done_modules`
drives both the derived pointer and the prefetch guard.

### Hook Entry Points

```python
# _register_forward_hook → reshard_param_groups:
if ctx.backward_phase and id(module) == ctx.backward_module:
    return                              # skip reshard — this is the pending module

# _register_backward_pre_hook → pre_backward_hook (root only):
ctx.backward_done_modules.clear()
ctx.backward_phase = True
ctx._advance_backward_module()          # picks first non-done in _reversed_order

# _register_backward_hook → post_backward:
ctx.backward_done_modules.add(id(module))
ctx._advance_backward_module()          # advances to next pending module
module.reshard()

# _register_post_backward_final_callback:
ctx.backward_phase = False
ctx.backward_module = None
ctx.backward_done_modules.clear()
```

### Prefetch Constraint

During backward, `unshard(bwd_pass=True)` prefetches the next module in
`_reversed_order`.  An extra guard skips modules whose backward is already done:

```python
# fsdp_module.py — unshard()
if bwd_pass and id(module) in ctx.backward_done_modules:
    continue        # backward already done — skip prefetch
```

### Timeline

Consider two FSDP-wrapped layers L1, L2 checkpointed together.
`forward_order = [root, L1, L2]`, `_reversed_order = [L2, L1, root]`.

```
----- FORWARD (normal) ----------------------------------
L1: pre → unshard(L1) → forward → reshard(L1)
L2: pre → unshard(L2) → forward → reshard(L2)
      (checkpoint drops intermediates)

----- BACKWARD (root enters phase) ----------------------
root pre_backward:
  clear done_modules, backward_phase = True
  _advance → backward_module = L2    (first not done)
  unshard(root)

----- ACTIVATION RECOMPUTE (L1→L2, inside checkpoint backward) --
L1 pre → unshard(L1) → forward
L1 post: L1 ≠ backward_module(L2) → reshard(L1)
L2 pre → unshard(L2)                (event[L2] set, persistent)
L2 post: L2 == backward_module → skip reshard

----- L2 BACKWARD ----------------------------------------
L2 pre_backward → unshard(L2)       (event set → skip)
L2 backward compute
L2 post_backward:
  done_modules.add(L2), _advance → L1, reshard

----- L1 BACKWARD ----------------------------------------
L1 pre_backward → unshard(L1)       (re-allocates, all-gathers)
L1 backward (gradients already computed → copies .grad)
L1 post_backward:
  done_modules.add(L1), _advance → root, reshard

----- FINAL CALLBACK --------------------------------------
backward_phase = False
backward_module = None
done_modules.clear()
```

### Key Design Decisions

1. **`backward_module` is derived, not set by hooks.**  Activation recompute
   always fires before any layer's `pre_backward_hook`.  Deriving from the done
   set + `_reversed_order` correctly identifies the pending module regardless
   of timing.

2. **`_advance_backward_module()` is called at exactly two points:** root
   `pre_backward_hook` (after clearing the done set) and `post_backward`
   (after adding a done module).  These are the only mutations to `backward_done_modules`.

3. **`backward_done_modules` serves dual purpose:** drives the derived pointer
   AND gates the prefetch guard in `unshard()`.

4. **Event persists between `unshard()` and `reshard()`.**  `unshard()` no
   longer clears its own event.  Prevents redundant all-gathers.

### Edge Cases

- **Sync mode (`enable_unshard_prefetch=False`):** No event is recorded,
  so the persistent-event mechanism does not apply.  `backward_module` still
  prevents premature resharding.
- **Module not reached by backward:** The final callback runs `reshard()`
  for untouched modules.
- **Multiple micro-batches:** All state is reset in the final callback.

---

## Complete Timeline

```
FORWARD PASS (enable_unshard_prefetch=True)
---------------------------------------------------------
main stream:  |← compute L[0] →|← compute L[1] →|← compute L[2] →|
ag_stream:    |AG(L[0])  AG(L[1])|        AG(L[2])|                |

pre-hook L[0]: async unshard L[0] + prefetch L[1] on ag_stream
               event[L[0]].wait() → main stream unblocks
               post_unshard(L[0]) on terminal AG stream
               _replace_module_parameter(L[0])

pre-hook L[1]: event[L[1]] already set → wait (likely done)
               post_unshard(L[1]) on terminal AG stream
               _replace_module_parameter(L[1])
               async prefetch L[2] on ag_stream

pre-hook L[2]: event[L[2]].wait() → main stream unblocks
               post_unshard(L[2]) on terminal AG stream
               _replace_module_parameter(L[2])

BACKWARD PASS (enable_async_reduce_grad=True, full-iteration CUDA graph)
---------------------------------------------------------
main stream:  |bwd L[2]-----------|bwd L[1]-----------|bwd L[0]-----------|
ag_stream:    |AG(L[1]) prefetch    |AG(L[0]) prefetch     |                      |
rs_stream:    |stage+RS(L[2]) ------|stage+RS(L[1]) ------|stage+RS(L[0])---------|

post-bwd L[2]: reshard, rs_stream.wait(main), stage grad[2], RS(L[2]), event[2]
post-bwd L[1]: drain event[2-2]? (i=1, no drain), stage grad[1], RS(L[1]), event[1]
post-bwd L[0]: drain event[L[2]], stage grad[0], RS(L[0]), event[0]

final_callback:
  drain event[L[1]], event[L[0]]
  main_stream.wait_stream(rs_stream)
  ← optimizer step safe →
```

---

## Gradient Redistribution — Implementation Note

No `async_op` parameter is needed. The method is purely synchronous within the calling stream:

`ParameterGroup.reduce_grad()` acquires the full-gradient lease and allocates one
communication owner only when the communication dtype differs. It converts and
scales once before redistribution. Gradient reduction then has an explicit inner-FSDP
stage and, on the last HSDP backward, an explicit outer-HSDP stage. A stage writes
directly to persistent storage only when the destination is empty and has the
communication dtype; otherwise it uses a temporary output and assigns or accumulates
afterward. The buffer does not accept raw communication tensors, scaling policy, or
accumulation policy.

The caller (`FSDPModule.reduce_grad`) provides the reduction stream.
`ParameterGroup.reduce_grad()` waits for its caller stream once, then performs
preprocessing, temporary allocation, both reduction stages, commit, and temporary
release inside the reduction-stream context. The helpers therefore require no
additional waits or tensor stream recording.

---

## Pitfall: Zero-Numel Gradient Shards and Fused Optimizers

**Problem.** When a parameter is sharded across DP ranks, its local shard on a given rank
may contain **zero elements** (e.g., a small bias or embedding table on a high-DP-count setup).
Materializing a `DTensor` gradient for such a shard creates a tensor with `numel() == 0`.

Fused multi-tensor optimizers (e.g., TransformerEngine `FusedAdam`) operate on **all**
gradients in a parameter group in a single fused kernel launch. Passing a zero-numel
tensor into these fused ops can silently corrupt the weight updates for **neighboring
non-empty parameters** in the same group. The optimizer does not crash or raise an error
— it produces numerically incorrect steps that manifest only as **convergence divergence**,
making this extremely difficult to attribute and debug.

**Symptoms (hard to diagnose):**
- Training loss diverges or fails to converge despite correct hyperparameters
- No NaN or Inf in gradients — the corruption is a numerical perturbation
- Occurs only at certain DP-world-size / model-size combinations where sharding produces empty local slices
- Bisecting the codebase is unhelpful because the optimizer runs without error

**Fix in `param_group.py`:**
```python
# DO NOT REMOVE THIS CHECK:
if p.requires_grad and grad_data.numel() > 0:
    self.optimizer_grads.append(make_uneven_dtensor(...))
else:
    self.optimizer_grads.append(None)  # zero-numel shard → no DTensor grad
```

By recording `None` for zero-numel shards instead of a DTensor with an empty local tensor,
the fused optimizer never receives the empty tensor and cannot corrupt neighboring updates.
The optimizer already handles `None` grads correctly (parameters without a grad are
simply skipped during the fused update).

**Additional safeguard in `_scale_gradients`:**
```python
for dist_grad in param_group.optimizer_grads:
    if dist_grad is None:
        continue   # skip zero-numel shards
    dist_grad._local_tensor.mul_(scaling_factor)
```

---

## Pitfall: Attribute Propagation from Original Params to DTensor Dist Params

**Problem.**  `_init_buffers()` in `ParameterGroup` creates DTensor views (`optimizer_params`) into
sharded buffers and `_replace_module_parameter` registers these DTensors on the module.
However, critical metadata set on the **original** `nn.Parameter` objects by upstream layers
(e.g. TE linear layers from `transformer_engine.py`) is **not** automatically transferred to
the new DTensor wrappers.

The adapter (`mcore_fsdp_adapter.py:310-330`) copies a fixed list of attributes from original
params to optimizer_params.  If an attribute is missing from this list, downstream consumers that
inspect the registered module parameters will see the wrong metadata.

**Affected attributes and their consumers:**

| Attribute | Set by | Consumer | Failure mode |
|-----------|--------|----------|-------------|
| `allreduce` | `transformer_engine.py:841` — set to `False` on expert MLP weights | `_get_param_groups` (`optimizer/__init__.py:348`) — classifies `is_expert_parallel` | Expert params misclassified as non-expert, causing wrong gradient scaling, clipping group assignment, and optimizer partition placement |
| `is_embedding_parameter` | Various embedding layers | `_get_param_groups` — controls weight decay exclusion | Embeddings incorrectly decayed → convergence divergence |
| `is_embedding_or_output_parameter` | Embedding / output layers | Same as above | Same |
| `sequence_parallel` | TE layers | `parallel_state` / loss computation | Incorrect SP semantics |
| `tensor_model_parallel`, `partition_dim`, `partition_stride` | TE layers | Distributed checkpointing / state dict | Incorrect checkpoint sharding |
| `requires_grad` | All layers | Optimizer | Frozen params may receive updates |

**Fix.**  When adding a new metadata attribute to TE layers or custom modules that are
consumed by downstream code (optimizer, checkpointing, mixed precision), add its name to
the `attr_name` list in `mcore_fsdp_adapter.py` to ensure it propagates to the DTensor
optimizer_params.

**Debugging.**  Misattributed params can be detected by dumping
`model._log_parameter_groups()` output and verifying that expert params appear in the
`is_expert_parallel` group.  NaN after a single step with `gradient_accumulation_fusion`
is a strong indicator of missing `allreduce` propagation.

---

## Memory Optimization: Freeing Original Parameter Storage

After `ParameterGroup._init_buffers()` copies parameter data into the internal weight buffers
(`model_weight_buffer` and optionally `main_weight_buffer`), the original full parameter tensors
are freed via `_free_storage(p.data)`. The module holds DTensor shard views and `unshard()`
rebinds `.data` to the all-gathered buffer, so the original storage is dead and freeing it
reduces peak memory during model construction.

---

## Configuration

```python
fully_shard(
    module,
    mesh=mesh,
    enable_unshard_prefetch=True,   # pipeline AG on ag_stream while current module computes
    enable_async_reduce_grad=True,  # pipeline RS on rs_stream while later modules compute bwd
)
```

Setting either flag to `False` assigns `torch.cuda.current_stream()` to the corresponding
stream variable, making all `with torch.cuda.stream(stream)` blocks no-ops — zero overhead,
identical to baseline.

---

## `stop_communication()` — Main Stream Synchronization

The `FullyShardedDataParallel.stop_communication()` method (in `mcore_fsdp_adapter.py`) ensures
all pending FSDP communication is complete and visible to the main CUDA stream. This is called
before the optimizer step to guarantee that gradient reductions and parameter updates are
synchronized.

For the `fully_shard` path, the implementation was previously `NotImplementedError`. It now
calls:

```python
torch.cuda.current_stream().wait_stream(ctx.ag_stream)  # finish all-gather work
torch.cuda.current_stream().wait_stream(ctx.rs_stream)   # finish reduce-scatter work
```

This brings both communication streams into the main stream, ensuring the optimizer sees
fully-synchronized parameters and gradients.

---

## Known Gaps / Recommended Follow-ups

1. **Full compute-weight prefetch remains single-module.** HSDP outer-stage lookahead
   is configurable, but the generic path still fully unshards at most one future
   module. A size-aware policy may improve networks with many small modules.

---

## Bucket Allocator Hierarchy

`allocator.py` provides a polymorphic allocator family via the `BucketAllocator`
interface, letting callers swap allocation strategies without changing
`DataParallelBuffer`. `ParameterGroup` is the sole allocation-policy owner.

```
BucketAllocator  (interface)
|-- TemporaryBucketAllocator        — legacy: allocates per key, frees + deletes
|-- StorageFreeingBucketAllocator   — allocates per key, frees storage but keeps bucket
|                                     (same tensor object reused on next allocation)
\-- TracePoolAllocator             — two-phase: trace → plan → static pool
```

### `TracePoolAllocator`

**Purpose.**  During parameter unshard and gradient reduction the FSDP
framework allocates and frees temporary flat buffers (all-gather input/output,
gradient accumulation) in a deterministic, repeatable order.  `TracePoolAllocator`
replaces per-call `torch.empty` + `_free_storage` with a one-time planned pool
that eliminates allocation overhead and fragmentation.

**Design — three phases.**

| Phase | Behaviour |
|---|---|
| **Trace** (``plan()`` not yet called) | Records alloc/free pairs via an ``_active_keys`` set: the first ``allocate`` per key records a trace event and marks the key active; duplicate allocs (key still active) are no-ops.  The first ``free`` per key records a trace event and marks it inactive; double-frees and free-before-alloc are ignored.  A key that is freed then re-allocated generates a new pair of events.  Buckets are created with ``torch.empty`` and **never deleted** — on re-alloc the same tensor object is resurrected via ``_alloc_storage``, keeping outstanding views (NVFP4 ``_rowwise_data`` references) live. |
| **Plan** (``plan()``) | Replays the trace to extract per-key live intervals ``(alloc_seq, free_seq)``, groups keys by ``(dtype, device)``, and runs a **conflict-graph coloring** algorithm per group.  Two keys are connected if any of their live intervals overlap; the graph is then colored greedily (largest-size-first, best-fit bin packing).  Each color is a **slot** backed by its own ``torch.empty()`` tensor (per-slot allocation).  Each key maps to exactly one fixed slot via ``_key_to_slot``, with views pre-computed in ``_key_to_view`` for O(1) access. |
| **Optimized** (after ``plan()``) | ``allocate`` returns a ``Bucket`` with a pre-computed view of the per-slot tensor.  ``free`` marks the slot as unused.  Both are O(1) dict lookups — no allocations, storage resizes, slot-lists, or cursor management.  Addresses are fixed per-key across all micro-batches. |

**Conflict-graph coloring vs. left-edge.**  The previous design used greedy
left-edge interval coloring on a monolithic pool tensor, requiring per-key
slot *lists* and *cursors* because the same key could appear in multiple
intervals (pre-merged into a "super-interval" to guarantee one address per
key).  The current design builds an explicit interval-overlap graph and
colors it with greedy graph coloring.  Each key maps to exactly one slot
(``_key_to_slot``) backed by a separate tensor (per-slot allocation).  This:

* Eliminates the need for slot lists, cursors, and ``reset_cursor()``.
* Produces the **theoretical minimum** slot count (optimal coloring of the
  interval-overlap graph).
* Reduces CUDA caching-allocator fragmentation by replacing one giant
  contiguous block with many smaller, independently-placed tensors.
* Still guarantees fixed per-key addresses (CUDA graph safe).

**Properties.**

- **Optimal slot count:** Conflict-graph coloring yields the theoretical
  minimum (peak simultaneous live memory).
- **Repeatable trace required:** The FSDP framework must execute the same
  alloc/free calls in the same order per micro-batch.
- **Fully idempotent:** ``allocate`` and ``free`` are always safe to call
  multiple times.  `allocate` → `allocate` is a no-op (returns existing).
  `free` → `free` is a no-op (ignored).  `free` without a prior `allocate`
  is a no-op.  Both trace and optimized phases share this guarantee.
- **Stable tensor objects:** Buckets are never deleted — the same Python
  tensor object is reused across alloc/free cycles, preventing dangling
  views (e.g., NVFP4 parameter ``_rowwise_data``).
- **Per-slot tensors:** Each slot is a separate ``torch.empty`` allocation,
  not a slice of a monolithic pool.  Slots are independently placed by the
  CUDA caching allocator, avoiding fragmentation from giant contiguous
  blocks.

**API.**

```python
allocator = TracePoolAllocator()
# … run one iteration (trace phase) …
pool_elems = allocator.plan()          # returns total element count
# … subsequent micro-batches use the pre-allocated slots …
print(allocator.total_pool_bytes)       # bytes across all slots
allocator.reset()                       # back to trace phase
```

**Lifecycle diagram for one allocation key across two micro-batches.**

```
Trace phase                                Optimized phase
-----------                                ---------------
allocate(key) → torch.empty  --.             allocate(key) → slot_tensor view  (fixed addr)
free(key)     → _free_storage  | plan(A)     free(key)     → slot free
allocate(key) → torch.empty  --'             allocate(key) → slot_tensor view  (same addr)
free(key)     → _free_storage               free(key)     → slot free
                                             -- next micro-batch --
                                             allocate(key) → slot_tensor view  (same addr)
                                             free(key)     → slot free
                                             ...
```

No ``torch.empty`` or storage resizing occurs in the optimized phase — each
slot owns its own tensor, and buckets are lightweight views into them.
