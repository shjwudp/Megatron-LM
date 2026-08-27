# 1F1B EP Overlap — MFSDP v2 Integration Design

This document describes how the experimental Megatron FSDP v2
(`megatron_fsdp.experimental`) integrates with the 1F1B EP-overlap schedule
(`combined_1f1b`). The v1 implementation (`megatron_fsdp`) defines the
schedule-facing contract; this PR implements the same contract for v2.

---

## 0. Background — Why the Overlap Schedule Needs Special FSDP Handling

### Normal FSDP flow

FSDP hooks fire on `TransformerLayer`:

```
TransformerLayer.forward()
  → pre_forward hook:  unshard_parameters()
  → actual compute
  → post_forward hook: reshard_parameters()
backward()
  → pre_backward hook:  unshard_parameters()
  → compute grads
  → post_backward hook: reshard_parameters() + reduce_grad()
```

### EP overlap flow

The combined 1F1B schedule calls **sub-modules directly**, bypassing
`TransformerLayer.forward()`:

```
combined_forward_backward_step()
  → f_layer.attn.forward()     ← no TransformerLayer hook fires
  → b_layer.mlp.backward()     ← no TransformerLayer hook fires
  → f_layer.moe_dispatch.forward()
  → ...
```

Because the schedule invokes sub-modules (`attn`, `mlp`, `moe_dispatch`,
`moe_combine`) directly, FSDP hooks registered on `TransformerLayer` are
**never triggered**. FSDP must expose manual management APIs that the schedule
calls explicitly at the right moments.

### Key flags

| Flag | What it does |
|---|---|
| `--overlap-moe-expert-parallel-comm` | Enables the combined 1F1B schedule and its explicit FSDP lifecycle callbacks. |
| `--delay-wgrad-compute` | TE layers postpone weight gradient computation until `backward_dw()`. FSDP must defer `reduce_grad()` until after `backward_dw()`. |

### Why `delay_wgrad_compute` matters for FSDP

`post_accumulate_grad_hook` observes execution of the parameter's autograd
`AccumulateGrad` node; it is not a general observer of later writes to
`parameter.grad`. With `delay_wgrad_compute=True`, that node runs during the
main backward before TE's later `backward_dw()` call writes the delayed weight
gradient. `backward_dw()` does not execute `AccumulateGrad` a second time, so
reducing from the hook would miss the delayed contribution. FSDP v2 therefore
disables that hook for this schedule and lets the explicit
`post_backward_release_module()` callback reduce after `backward_dw()`.

---

## 1. Schedule Integration Contract

The schedule in `combined_1f1b.py` calls into MFSDP v2 at the root and layer
boundaries described below.

### 1.1 Discovery — `find_megatron_fsdp(model)`

**File:** `megatron_fsdp/utils.py`

`find_megatron_fsdp()` walks the model wrapper chain and returns either a v1
`MegatronFSDP` wrapper or a `FullyShardedDataParallelV2` adapter wrapping an
`FsdpModule`. Returning the adapter is important because the schedule-facing
callbacks live there, not on the bare `FsdpModule`.

### 1.2 Pre-schedule setup — `_replace_param_with_raw_if_needed()`

**File:** `combined_1f1b.py` → `FullyShardedDataParallelV2._replace_param_with_raw_if_needed()`

The schedule calls this once before running. In v1 it swaps optimizer-facing
DTensor parameters back to raw `nn.Parameter`s. MFSDP v2 always stores raw
tensors backed by `DBuffer` on the module, so **no parameter swap is needed**;
the adapter finalizes the root `FsdpContext` instead, ensuring a child FSDP
unit cannot mistake itself for the root when it executes first:

```python
# In FullyShardedDataParallelV2._setup_1f1b_overlap_interface (mcore_fsdp_adapter.py)
def _replace_param_with_raw_if_needed() -> None:
    self.module.context.ensure_finalized()
```

### 1.3 Root lifecycle setup — `pre_forward()` / `pre_backward()`

**File:** `combined_1f1b.py` → adapter callbacks

The first forward-only microbatch calls `pre_forward()` to materialize
root-owned parameters such as embeddings and the output layer. Layer-owned
parameters are managed by the callbacks in §1.4.

Before a combined segment that has backward work, the schedule instead calls
`pre_backward()`. MFSDP v2:

- transitions this module to `Phase.BACKWARD`,
- if root: forks the reduce-scatter stream from the current stream so later
  allocations stay legal under CUDA-graph capture; the schedule passes
  `register_final_callback=False` because it finalizes explicitly,
- unshards parameters (all-gather) and waits for them,
- prefetches the next module in `backward_order` (static-order prefetch).

### 1.4 Per-layer lifecycle — `set_fsdp_lifecycle_hooks()`

**File:** `combined_1f1b.py` → adapter callbacks

The schedule attaches four callables per `TransformerLayerSchedulePlan`:

- **`pre_forward_module(module)`** — called inside the first forward node's
  CUDA stream; unshards the layer and its nested FSDP units.
- **`post_forward_release_module(module)`** — called after the last forward
  node of a layer; reshard only (no gradient reduction).
- **`pre_backward_module(module)`** — called inside the first backward node's
  CUDA stream; enters backward and materializes the layer and its nested FSDP
  units.
- **`post_backward_release_module(module)`** — called after the last backward
  node; runs the module's `post_backward()` (reshard + gradient reduction).

The callbacks execute in the schedule node's stream so the parameter
all-gather/reduce-scatter dependencies are ordered with the compute that uses
them. The adapter applies each callback recursively because routed experts can
be a nested `FsdpModule` on a different expert-DP mesh while their containing
`TransformerLayer` owns the dense parameters.

### 1.5 Root backward finalization — `post_backward()`

**File:** `combined_1f1b.py` → `FsdpModule.post_backward()`

Called once after each overlapped run completes. `post_backward()` is a
**no-op unless this module is in the BACKWARD phase** (idempotent — the
schedule may call it on a module already released). MFSDP v2:

- walks the module subtree (excluding itself) and finalizes (reduce +
  reshard) any submodule `FsdpModule` that is **still in the BACKWARD
  phase** — i.e. the 1F1B schedule skipped its per-module release,
- then reduces and reshards this module itself and returns it to `RESTING`.

The v2 adapter binds `self.post_backward = self.module.post_backward` (no
arguments).

### 1.6 Gradient sync suppression — `no_sync_func()`

**File:** `combined_1f1b.py` → `FullyShardedDataParallelV2.no_sync()`

A context manager that suppresses gradient finalization for non-final
microbatches so gradients accumulate across inner microbatches:

```python
@contextmanager
def no_sync(self):
    self.module.context.ensure_finalized()
    context = self.module.context
    previous_state = context.is_last_microbatch
    context.is_last_microbatch = False
    try:
        yield
    finally:
        context.is_last_microbatch = previous_state
```

This wires into `config.no_sync_func` via the existing training-loop contract.

---

## 2. Explicit Schedule-Owned Lifecycle

MFSDP v2 keeps its normal hooks only on each `FsdpModule`. It does **not** add
hooks or parent references to every descendant module. The combined schedule
cannot rely on the normal hooks because it invokes attention, dispatch, expert,
and combine callables directly, so it owns these boundaries explicitly:

| Schedule boundary | MFSDP v2 action |
|---|---|
| First forward node | `pre_forward_module(layer)` |
| Last forward node | `post_forward_release_module(layer)` |
| First backward node | `pre_backward_module(layer)` |
| Last backward node, after delayed `backward_dw()` | `post_backward_release_module(layer)` |

The first forward node is `pre_dispatch_computation`; the last is `mlp` for a
dense layer and `moe_combine` for an MoE layer. Backward uses the same nodes in
reverse order. `TransformerLayerNode` invokes the callbacks inside its stream
acquisition context, which establishes the required CUDA ordering without
arbitrary submodule hooks.

Forward preparation deliberately uses the unshard primitive without changing
the FSDP phase. In a combined 1F1B segment, forward and backward can use the
same logical layer concurrently. Backward preparation owns the phase transition
through `pre_backward(register_final_callback=False)`.

---

## 3. `delay_wgrad_compute` — Skipping the Autograd Backward Callback

### Problem

The normal backward path uses per-param `post_accumulate_grad_hook`
(registered in `FsdpModule._register_hooks()`) to detect when all parameter
gradients are ready, then calls `post_backward()` which reshards and reduces.
With `delay_wgrad_compute=True`, this hook fires during `backward()`
(activation gradients only), but weight gradients are written later in
`backward_dw()`, so reduction would see partial gradients.

### Solution

When `skip_backward_callback=True`, `_register_hooks()` does **not** register
per-param `post_accumulate_grad_hook`; reduction relies entirely on the
schedule's explicit `post_backward_release_module()` call, which fires after
`backward_dw()`:

```python
# In FsdpModule._register_hooks()
if skip_backward_callback:
    return
for group in self._parameter_groups:
    if not group.requires_grad:
        continue
    for fsdp_parameter in group.fsdp_parameters:
        fsdp_parameter.unsharded.register_post_accumulate_grad_hook(self._make_grad_hook())
```

The adapter wires it as `skip_backward_cb = overlap_enabled and
ddp_config.delay_wgrad_compute` (the flag only matters on the explicit overlap
path, where the schedule drives reduction).

---

## 4. Public API Surface

### Modified public methods on `FsdpModule`

| Method | Signature | Description |
|---|---|---|
| `pre_backward()` | `(register_final_callback: bool = True) -> None` | Idempotent backward-phase setup; the explicit schedule disables the autograd final callback. |
| `post_backward()` | `() -> None` | Finalize backward: no-op unless BACKWARD phase; reduce+reshard this module, and any submodule `FsdpModule` still in the BACKWARD phase. |

### New parameter on `fully_shard()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skip_backward_callback` | `bool` | `False` | Skip per-param `post_accumulate_grad_hook` for `delay_wgrad_compute`. |

### New methods on `FullyShardedDataParallelV2` (adapter)

The schedule-facing surface is assembled in `_setup_1f1b_overlap_interface`,
which binds closures that operate on a passed `FsdpModule`:

| Attribute | Signature | Description |
|---|---|---|
| `_replace_param_with_raw_if_needed` | `() -> None` | Finalize the root context; no parameter swap is needed. |
| `pre_forward` | `() -> None` | Materialize root-owned parameters for the first forward-only segment. |
| `pre_forward_module` | `(module) -> None` | Materialize a layer and nested FSDP units before forward. |
| `pre_backward_module` | `(module) -> None` | Enter backward for a layer and nested FSDP units. |
| `post_forward_release_module` | `(module) -> None` | Release forward-pass params (reshard only). |
| `post_backward_release_module` | `(module) -> None` | Release backward-pass params (reshard + reduce). |
| `no_sync()` | `() -> contextmanager` | Suppress gradient finalization for non-final microbatches. |
| `_setup_1f1b_overlap_interface()` | `() -> None` | Bind the schedule-facing callbacks above. |

### Changes to `find_megatron_fsdp()` (`megatron_fsdp/utils.py`)

Now also detects `FullyShardedDataParallelV2` adapters (via an `FsdpModule`
wrapped by an object with `ddp_config`) alongside v1 `MegatronFSDP`.

---

## 5. Activation Recomputation Guard

Activation recomputation runs forward computation inside backward. If forward
preparation prefetched the next module in forward order, that module's backward
may already be complete, so no later backward boundary would reshard it. MFSDP
v2 detects recomputation via the module phase and the active autograd GraphTask:

```python
is_recomputing = self.phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
...
if not is_recomputing:
    next_module = context.forward_order.next_item(self)
    if next_module is not None:
        next_module._unshard_parameter_groups()
```

Recomputed forwards still unshard the current module (its parameters are
needed for the recomputed GEMMs) but never prefetch a successor.

---

## 6. Files Touched

| File | Change |
|---|---|
| `experimental/module.py` | Public lifecycle APIs, `skip_backward_callback`, recompute guard, phase transitions |
| `experimental/fully_shard.py` | `skip_backward_callback` parameter |
| `mcore_fsdp_adapter.py` | Add explicit root/layer lifecycle callbacks, `no_sync()`, and `_setup_1f1b_overlap_interface()` |
| `models/common/utils.py` | Invoke explicit lifecycle callbacks inside schedule-node CUDA streams |
| `models/common/model_chunk_schedule_plan.py` | Bind lifecycle callbacks to the first/last layer nodes |
| `pipeline_parallel/combined_1f1b.py` | Drive the root lifecycle and wire per-layer lifecycle callbacks |
| `megatron_fsdp/utils.py` | Extend `find_megatron_fsdp()` |
| `tests/unit_tests/distributed/mfsdp_v2/test_context.py` | Direct-submodule explicit-lifecycle test |
| `tests/unit_tests/distributed/mfsdp_v2/test_mcore_nd_parallel.py` | End-to-end EP-overlap parity test |
| `tests/unit_tests/distributed/mfsdp_v1/utils.py` | Shared GPT overlap-test construction and schedule-plan forward step |

---

## 7. Edge Cases

### 7.1 Activation recomputation with overlap

Handled by the phase/GraphTask detection in §5: recomputed forwards never
prefetch a successor and never reshard the current module mid-recompute.

### 7.2 Pipeline and virtual-pipeline parallelism

PP/VPP support is outside this integration. The adapter continues to reject
pipeline parallelism, and the interleaved combined schedule continues to
reject FSDP model chunks. Cross-chunk context and optimizer support belong in
follow-up changes.

---

## 8. Reference — v1 Contract

For context, the v1 API contract that the schedule calls:

| Call Site (combined_1f1b.py) | v1 Attribute/Method | Type |
|---|---|---|
| Discovery | `find_megatron_fsdp(model)` | Discovers `MegatronFSDP` instance |
| Pre-schedule | `fsdp_wrapper._replace_param_with_raw_if_needed()` | Method (root-context init for v2) |
| Microbatch loop | `no_sync_func()` → `model.no_sync()` | Context manager on adapter |
| First forward-only run | `fsdp_wrapper.pre_forward()` | Optional v2 adapter callback |
| Pre-backward run | `fsdp_wrapper.pre_backward()` | `partial(_root_pre_backward, module=None, skip_backward_hook=True)` |
| Layer plan | `forward_fsdp_wrapper.pre_forward_module` | Optional v2 adapter callback |
| Layer plan | `forward_fsdp_wrapper.pre_backward_module` | Optional v2 adapter callback |
| Layer plan | `forward_fsdp_wrapper.post_forward_release_module` | `partial(_post_forward, input=None, output=None)` |
| Layer plan | `forward_fsdp_wrapper.post_backward_release_module` | `_post_backward_release_module` |
| Post-run | `fsdp_wrapper.post_backward()` | `_root_post_backward` |
