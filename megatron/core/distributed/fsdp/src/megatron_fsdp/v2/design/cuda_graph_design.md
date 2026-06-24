# Silent CUDA Graph inside Megatron FSDP v2 — Design

> **Experimental** — CUDA graph support in Megatron FSDP v2 is an experimental
> feature.  The API and behaviour may change in future releases without notice.

## 1. Motivation

mcore's CUDA graph system (`cuda_graph_impl="local"`, `cuda_graphs.py`) was
designed for DDP's memory model — each layer receives freshly-allocated tensor
inputs/outputs.  Megatron FSDP v2 shares pool-backed buffers across layers,
and FSDP hooks (unshard/reshard) are not part of the graphed region.

This doc describes a CUDA graph system built INTO Megatron FSDP v2, using
separate forward and backward `CUDAGraph` objects with a custom
`autograd.Function` for replay.  The user enables it with a single flag —
everything else is automatic.

## 2. One knob

```python
fully_shard(module, enable_cuda_graph=True)
```

No `--cuda-graph-warmup-steps`, no `--cuda-graph-scope`, no coordination with
the pipeline schedule.  The system automatically captures and replays on the
first optimized microbatch.

## 3. Why TracePoolAllocator is the enabler

CUDA graphs require **stable buffer addresses**.  After `plan()` allocates
the pool tensor, every slot has a fixed offset.  The returned views' addresses
are deterministic every micro-batch.

During graph replay, the graphed CUDA kernels operate on the exact addresses
that were recorded during capture.  The allocator is **not called** for
graphed modules during replay — the graph uses the addresses directly.  Pool
slots are pre-allocated by `plan()`, so no dynamic allocation occurs inside
the graph region.

## 4. Architecture — split fwd/bwd with custom autograd.Function

### 4.1 Why split fwd/bwd?

The shared memory pool requires capture and replay orders to match.  With
a single interleaved fwd+bwd graph per module, the replay order would be
`fwd1 fwd2 fwd3 bwd3 bwd2 bwd1`, which is correct (driven by FSDP hooks).
But a `make_graphed_callables`-based approach captures `fwd1 bwd1 fwd2 bwd2`,
which mismatches the pool layout during replay.

Instead, this system captures two **separate** `CUDAGraph` objects per module:

- **`fwd_graph`** — captures `module.forward` only
- **`bwd_graph`** — captures `torch.autograd.grad` (backward) only

Capture happens lazily, driven by FSDP hook order during the first
microbatch, so capture order naturally matches runtime order.

### 4.2 Custom autograd.Function

```python
class _CudaGraphFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, *flat_inputs):
        # 1. Copy live inputs → static pool buffers
        # 2. Replay fwd_graph
        # 3. save_for_backward(live_inputs)
        # 4. Return clones of static_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # First backward: capture bwd_graph, then replay
        # Subsequent: copy live grads, replay bwd_graph
```

`install()` replaces `module.forward` with a wrapper that flattens args/kwargs
into positional tensors and calls `_CudaGraphFunction.apply(runner, *flat)`.

### 4.3 Lazy backward capture

The backward graph is captured on the **first** backward call.  Subsequent
backwards replay the captured graph.  This is essential because the backward
graph is captured during the first iteration's backward pass, and the capture
must include the exact backward CUDA ops (including any autograd engine
internals).

## 5. Forward capture design

### 5.1 Captured with grad enabled

The forward graph is captured **with autograd enabled** (no `torch.no_grad()`).
This preserves the autograd tape so the backward graph can call
`torch.autograd.grad(static_outputs, ...)` directly — **no forward recompute**
in the backward graph.

### 5.2 Output structure handling

Module outputs may contain `None` entries (e.g., `(hidden_states, None, rotary_pos_emb)`).
The system uses three primitives:

| Method | Purpose |
|--------|---------|
| `_record_output_structure(out)` | Snapshots `_output_is_tuple` (single tensor vs tuple) and `_none_mask` (which positions are `None`) |
| `_flatten_output(out)` | Returns flat tuple of only `torch.Tensor`s, drops `None`s |
| `_unflatten_output(flat)` | Restores original shape: single tensor if `_output_is_tuple=False`, otherwise tuple with `None`s at recorded positions |

The forward graph operates on flattened output tensors.  Before returning to
the caller, `_CudaGraphFunction.forward` calls `_unflatten_output` to restore
the original output shape.  The backward path is unaffected because
`grad_outputs` from autograd always match the flattened tensor count.

## 6. Generator / RNG state

Both `fwd_graph` and `bwd_graph` register the default CUDA generator state
via `register_generator_state`.  This ensures that dropout (and other RNG
operations) produce **different** masks each iteration:

- **Capture**: gen at state S0 → dropout consumes N values → gen at S0+N
- **Replay**: gen restored to S0 → replay consumes N values → gen at S0+N;
  delta N is applied to the real generator → real gen advances monotonically

Without `register_generator_state`, dropout masks are baked into the graph at
capture time and reused identically every iteration, causing convergence
degradation.

The `_ensure_generator_graph_safe()` helper fixes inference-mode tensors in
the generator state before registration.

## 7. Capture stream & shared pool

All runners within the same root FSDP context share:

- **One `graph_pool_handle`** — via `torch.cuda.graph_pool_handle()`, all
  `CUDAGraph` objects share the same backing memory.  The CUDA driver reuses
  scratch memory (cuDNN/cuBLAS workspaces) across layers instead of
  duplicating it N times.

- **One capture stream** — `ctx.cuda_graph_stream`, passed as
  `capture_stream=` to every `FSDPCudaGraphRunner`.  This serializes graph
  captures so they don't race within the shared pool.

```python
# hooks.py
ctx.cuda_graph_stream = torch.cuda.Stream()
ctx.cuda_graph_pool = torch.cuda.graph_pool_handle()

# Each runner:
FSDPCudaGraphRunner(module,
    graph_pool=ctx.cuda_graph_pool,
    capture_stream=ctx.cuda_graph_stream,
)
```

## 8. Lifecycle

```
╔═══════════════════════════════════════════════════════════════╗
║ Stage 1 — Microbatch 0 (trace)                                ║
║                                                                ║
║  forward:   _trace_allocate / _trace_free  → trace recorded   ║
║  backward:  _trace_allocate / _trace_free  → trace continues  ║
║  plan():    pool built, _seq_ops populated, phase="optimized"  ║
║  snapshot_slots():  freeze slot state for replay              ║
╚═══════════════════════════════════════════════════════════════╝
                          │
╔═══════════════════════════════════════════════════════════════╗
║ Stage 2 — Microbatch 1 (capture)                               ║
║                                                                ║
║  root_forward_pre_hook:                                        ║
║    Creates shared graph_pool_handle + capture_stream            ║
║    reset_cursor(); restore_slots()                             ║
║                                                                ║
║  forward (per graphed FSDPModule):                             ║
║    forward_pre_hook: unshards params as usual                  ║
║    Creates FSDPCudaGraphRunner(module, pool, stream)            ║
║    → capture_forward():                                        ║
║        1. Introspects module.forward signature                 ║
║        2. Builds static_inputs in graph pool                   ║
║        3. Warmup: 3 eager fwd+bwd passes                       ║
║        4. Records output structure (None mask)                  ║
║        5. Captures fwd_graph (with grad, saved_tensors_hooks)  ║
║        6. register_generator_state(fwd_graph)                  ║
║    → install(): patches module.forward → _CudaGraphFunction    ║
║    Module runs _CudaGraphFunction.forward (first replay)       ║
║                                                                ║
║  backward (per graphed FSDPModule):                            ║
║    _CudaGraphFunction.backward fires (first time)              ║
║    → _capture_backward():                                      ║
║        1. Builds static_grad_outputs buffers                   ║
║        2. Captures bwd_graph: torch.autograd.grad(             ║
║              static_outputs, inputs=static_inputs+params)      ║
║        3. register_generator_state(bwd_graph)                  ║
║        4. Replays bwd_graph (first backward)                   ║
║    Post-backward hooks: reshard, reduce_grad                   ║
╚═══════════════════════════════════════════════════════════════╝
                          │
╔═══════════════════════════════════════════════════════════════╗
║ Stage 3 — Microbatch 2+ (replay)                               ║
║                                                                ║
║  root_forward_pre_hook:                                        ║
║    reset_cursor(); restore_slots()                             ║
║                                                                ║
║  forward (per graphed FSDPModule):                             ║
║    forward_pre_hook: unshards; runner.exists → skips capture   ║
║    patched forward → _CudaGraphFunction.forward:               ║
║      copy_ live inputs → static_inputs                         ║
║      fwd_graph.replay()                                        ║
║      return _unflatten_output(clones_of_static_outputs)        ║
║                                                                ║
║  backward:                                                      ║
║    _CudaGraphFunction.backward: bwd_graph exists → replay:     ║
║      copy_ live grads → static_grad_outputs                    ║
║      restore param.grad/main_grad buffers                      ║
║      bwd_graph.replay()                                        ║
║    Post-backward hooks: reshard, reduce_grad                   ║
╚═══════════════════════════════════════════════════════════════╝
```

## 9. Parameter gradient handling

Module parameters are passed through `_CudaGraphFunction.apply(runner, *user_args, *module_params)`.
This follows `make_graphed_callables` — params participate in the autograd graph so their
gradients flow through normal FSDP hooks.

- **Forward**: only user args are staged into pool buffers; params stay at their stable addresses.
  `fwd_graph.replay()` runs with `static_inputs` (user args) as inputs.
- **Backward capture**: ``torch.autograd.grad(inputs=user_args + module_params, ...)``
  computes gradients for both.  ``_static_grad_inputs`` records references to all
  computed gradient tensors.
- **Backward replay**: ``backward()`` returns ``(None_for_runner, *user_grads, *param_grads)``.
  Autograd sets ``param.grad`` from the param grads; FSDP's post-backward hook
  (``reduce_grad``) then consumes them normally via ``param.main_grad``.

No manual ``buf.copy_()`` or grad buffer management is needed — the standard
autograd + FSDP hook lifecycle handles everything.

## 10. Re-entrant capture prevention

During `capture_forward`, the warmup loop calls `self._run_forward()` which
executes the module's original `forward`.  This may trigger FSDP hooks on
sub-modules.  To prevent those hooks from attempting to capture the same
target again, `hooks.py` sets `target._fsdp_cg_runner = cg_runner` **before**
calling `capture_forward`:

```python
cg_runner = FSDPCudaGraphRunner(target, ...)
target._fsdp_cg_runner = cg_runner  # block re-entrant captures
try:
    cg_runner.capture_forward(*args, **kwargs)
    cg_runner.install()
except Exception:
    del target._fsdp_cg_runner
    raise
```

The hook's condition `not hasattr(target, "_fsdp_cg_runner")` then returns
`False` for re-entrant calls during warmup.

## 11. Per-FSDPModule selectivity

```python
# Only specific leaf layers graphed
for layer in model.layers:
    fully_shard(layer, enable_cuda_graph=True)
fully_shard(model, enable_cuda_graph=False)
```

Each `FSDPModule` carries a flag in `_FSDPState`:

```python
class _FSDPState:
    enable_cuda_graph: bool = False
```

All modules share the same `TracePoolAllocator` — slot assignments are fixed
by `plan()` regardless of which modules are graphed.

### Nesting limitation

A parent FSDP module that contains other FSDP modules as children **cannot**
use `enable_cuda_graph=True`.  Only leaf FSDP modules (those without FSDP
children) are eligible.

## 12. Allocator interface

```python
class TracePoolAllocator:
    _captured_slot_state: List[bool]

    def snapshot_slots(self):
        self._captured_slot_state = [s.in_use for s in self._slots]

    def restore_slots(self):
        for i, in_use in enumerate(self._captured_slot_state):
            self._slots[i].in_use = in_use
```

`snapshot_slots()` is called after `plan()`; `restore_slots()` is called
before each forward during replay to reset slot state.

## 13. Known issues & fixes

| Issue | Root cause | Fix | Date |
|-------|-----------|-----|------|
| Convergence degradation with CG | Missing `register_generator_state` on fwd_graph and bwd_graph — dropout masks baked at capture time | Added `register_generator_state(_ensure_generator_graph_safe())` on both graphs | 2026-06 |
| `TypeError: multiple values for argument` during capture | `sig.bind(self._module, *args, **kwargs)` injected `self` on bound methods, shifting positional args | `_bind_forward_args` handles both bound and unbound signatures | 2026-06 |
| `'NoneType' object has no attribute 'requires_grad'` | `_flatten_output` returned `tuple(out)` which preserved `None` entries; warmup iterated over them | `_flatten_output` now filters to `isinstance(t, torch.Tensor)` only | 2026-06 |
| Re-entrant capture during warmup | Warmup calls module forward which triggers hooks on sub-modules with same target | Set `target._fsdp_cg_runner` before `capture_forward`, clean up on failure | 2026-06 |
| None entries in output lost | Module output like `(hidden, None, emb)` was flattened but Nones not restored in autograd Function return | `_record_output_structure` + `_unflatten_output` restore original output shape | 2026-06 |
| Each runner created its own capture stream | `capture_stream` not passed from hooks.py to runner, causing parallel captures on different streams in shared pool | Pass `capture_stream=ctx.cuda_graph_stream` to all runners | 2026-06 |

## 14. Files

| File | Role |
|------|------|
| `cuda_graph_runner.py` | `FSDPCudaGraphRunner` — split fwd/bwd graph capture, `_CudaGraphFunction` autograd wrapper, output structure primitives, RNG state registration |
| `fsdp_module.py` | `_FSDPRootContext.cuda_graph_stream` / `cuda_graph_pool` shared state, `cuda_graph_compatible` property |
| `hooks.py` | Capture trigger in forward pre-hook, shared pool+stream creation, re-entrant guard, slot restore |
| `allocator.py` | `TracePoolAllocator.snapshot_slots()` / `restore_slots()` |
| `fully_shard.py` | Accepts `enable_cuda_graph` kwarg |

## 15. No user-visible knobs

| What happens | User action |
|---|---|
| Trace collection | Automatic (MB0 forward) |
| Pool planning | Automatic (MB0 post-backward) |
| Slot snapshot | Automatic (after plan) |
| Graph capture | Automatic (first optimized MB forward + backward) |
| Graph replay | Automatic (subsequent MBs, via patched forward + _CudaGraphFunction) |
| Slot state management | Automatic (snapshot/restore around replay) |
| RNG state management | Automatic (registered on both fwd_graph and bwd_graph) |
