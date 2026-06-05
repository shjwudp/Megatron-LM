# Silent CUDA Graph inside Megatron FSDP v2 — Design

## 1. Motivation

mcore's CUDA graph system (`cuda_graph_impl="local"`, `cuda_graphs.py`) was
designed for DDP's memory model — each layer receives freshly-allocated tensor
inputs/outputs.  FSDP v2 shares the same pool-backed buffers across layers,
and the FSDP hooks (unshard/reshard) are not captured in the graph in the
same way.

This doc describes a CUDA graph system built INTO FSDP v2, using
`TracePoolAllocator` as the stable-memory foundation.  The user enables it
with a single flag — everything else is automatic.

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

## 4. Lifecycle

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
║ Stage 2 — Microbatch 1+ (capture + eager forward)             ║
║                                                                ║
║  root_forward_pre_hook:                                        ║
║    reset_cursor()                                              ║
║    restore_slots()  (restore capture-time slot state)          ║
║    ctx.cuda_graph_active = True                                ║
║  forward (per graphed FSDPModule):                             ║
║    forward_pre_hook detects enable_cuda_graph=True:            ║
║      1. Creates FSDPCudaGraphRunner(module)                    ║
║      2. Pops FSDP hooks from the module                        ║
║      3. Calls torch.cuda.make_graphed_callables(shim, ...)    ║
║         → warms up, captures forward+backward graph            ║
║         → hooks are restored after capture                     ║
║      4. install() patches module.forward to graph replay       ║
║    Module then runs _orig_forward eagerly (this call)          ║
║  backward:                                                      ║
║    post_backward hooks fire (not captured yet)                  ║
║    reshard / reduce_grad run normally                           ║
║  post-bwd:                                                      ║
║    enable_flexible_mode()                                       ║
║    ctx.cuda_graph_active = False                                ║
╚═══════════════════════════════════════════════════════════════╝
                          │
╔═══════════════════════════════════════════════════════════════╗
║ Stage 3 — Microbatch 2+ (replay)                               ║
║                                                                ║
║  root_forward_pre_hook:                                        ║
║    reset_cursor()                                              ║
║    restore_slots()                                             ║
║    ctx.cuda_graph_active = True                                ║
║  forward (per graphed FSDPModule):                             ║
║    forward_pre_hook: runner already exists → skip capture      ║
║    patched forward: graphed(*flat) → replays captured graph    ║
║    → NO Python FSDP hooks fire                                 ║
║    → allocator NOT called                                      ║
║  forward (per non-graphed FSDPModule):                         ║
║    normal eager: hooks fire → alloc calls                     ║
║  backward:                                                      ║
║    post_backward: reshard deferred if cuda_graph_active        ║
║    reduce_grad runs with current stream                        ║
║  post-bwd (_post_backward_final_callback):                     ║
║    deferred reshard for all skipped modules                    ║
║    ctx.cuda_graph_active = False                                ║
║    enable_flexible_mode()                                       ║
╚═══════════════════════════════════════════════════════════════╝
```

## 5. FSDPCudaGraphRunner

The runner encapsulates per-module CUDA graph capture and replay:

```python
class FSDPCudaGraphRunner:
    def capture_forward(self, *sample_args, **sample_kwargs):
        # 1. Build a _ForwardShim that freezes non-tensor kwargs
        # 2. Disable side-stream collectives on the root context
        # 3. Pop FSDP hooks from the module (save/restore)
        # 4. Call torch.cuda.make_graphed_callables(shim, ...)
        #    → handles warmup (3 iters) + capture internally
        # 5. Restore hooks and side-stream settings

    def install(self):
        # Patches module.forward to a wrapper that flattens
        # args/kwargs → positional tensors and calls graphed(*flat)
```

**Why `make_graphed_callables`?** PyTorch's built-in utility handles:
- Warmup iterations to settle cuDNN/cuBLAS auto-tuning and TE FP8 scales
- Memory pool allocation for captured tensors
- AccumulateGrad node creation on the capture stream (avoids
  `cudaErrorStreamCaptureImplicit` by keeping `.grad` tensors alive from
  warmup through capture)

**Why pop hooks?** The graph records the module's `forward()` only — no
FSDP all-gather or reduce-scatter collectives.  Hooks are restored
immediately after capture, before the eager forward call.

## 6. Per-FSDPModule selectivity

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
children) are eligible.  This is enforced in `_init_fsdp_state` with a
`RuntimeError`.

The reason: CUDA graph capture runs the module's forward without FSDP hooks
(the hooks are popped).  For a parent module, the forward calls child
FSDP-module forwards, which may themselves have CUDA graph logic or require
FSDP unshard/reshard.  Limiting CUDA graph to leaf modules avoids this
complexity.

## 7. Capture stream invariants

During CUDA graph capture and replay, side-stream collectives must be
disabled so every CUDA operation lands on the default (capture) stream.
The root context provides:

```python
@dataclass
class _FSDPRootContext:
    cuda_graph_active: bool = False
    """True during capture or replay.
    Suppresses side-stream vs default-stream mismatches and defers reshard."""

    @property
    def cuda_graph_compatible(self) -> bool:
        """Return True when both side-stream collectives are disabled."""
        return (not self.enable_unshard_prefetch) and (not self.enable_async_reduce_grad)
```

`FSDPModule.cuda_graph_compatible` adds an additional check that the
allocator is in `"optimized"` phase:

```python
@property
def cuda_graph_compatible(self) -> bool:
    ctx = self._fsdp_root_context
    if not isinstance(ctx.bucket_allocator, TracePoolAllocator):
        return False
    if ctx.bucket_allocator.phase != "optimized":
        return False
    return ctx.cuda_graph_compatible
```

## 8. Hook changes

### 8.1 Forward pre-hook — capture trigger

```python
# hooks.py — _register_forward_pre_hook
def unshard_param_groups(fsdp_module, args, kwargs):
    # ... unshard as normal ...
    if fsdp_module._fsdp_state.enable_cuda_graph and (
        not hasattr(fsdp_module, "_fsdp_cg_runner")
    ):
        cg_runner = FSDPCudaGraphRunner(fsdp_module)
        cg_runner.capture_forward(*args, **kwargs)
        cg_runner.install()
        fsdp_module._fsdp_cg_runner = cg_runner
```

Capture happens exactly once — the first time the module is called in
an optimized forward pass.  Subsequent calls skip because `_fsdp_cg_runner`
already exists.

### 8.2 Defer post-forward reshard

```python
# hooks.py — _register_forward_hook → reshard_param_groups
def reshard_param_groups(module, *unused):
    ctx = module._fsdp_root_context
    if ctx.backward_phase and id(module) == ctx.backward_module:
        return
    if ctx.cuda_graph_active:
        return  # deferred to post-backward final callback
    module.reshard()
```

### 8.3 Suppress reshard in post_backward

```python
# hooks.py — _register_backward_hook → post_backward
def post_backward(module):
    ...
    if not ctx.cuda_graph_active:
        module.reshard()
    ...
```

### 8.4 Final callback — deferred cleanup

```python
# hooks.py — _post_backward_final_callback
def _post_backward_final_callback(root_state, root_module):
    ctx = root_module._fsdp_root_context
    was_cuda_graph = ctx.cuda_graph_active
    ...
    if was_cuda_graph:
        # Reshard all modules now that the graph region is complete.
        for module in reversed(ctx.forward_order):
            if not getattr(module, "post_backward_issued", False):
                module.reshard()
    ctx.cuda_graph_active = False
```

### 8.5 Root forward pre-hook — activate graph mode

```python
# hooks.py — _register_root_forward_pre_hook
def root_forward_pre_hook(_hook_module, *args, **kwargs):
    ctx = fsdp_module._fsdp_root_context
    ...
    if isinstance(ba, TracePoolAllocator) and ba.phase == "optimized":
        ba.disable_flexible_mode()
        ba.reset_cursor()
        if any(getattr(m._fsdp_state, "enable_cuda_graph", False)
               for m in ctx.forward_order):
            ba.restore_slots()
            ctx.cuda_graph_active = True
```

## 9. Allocator interface

The allocator requires three methods to support CUDA graph replay:

```python
class TracePoolAllocator:
    _captured_slot_state: List[bool]  # snapshot of slot.in_use at capture end

    def snapshot_slots(self):
        """Freeze current slot in_use state after all graphs are captured."""
        self._captured_slot_state = [s.in_use for s in self._slots]

    def restore_slots(self):
        """Restore slot state to capture-time snapshot before each replay."""
        for i, in_use in enumerate(self._captured_slot_state):
            self._slots[i].in_use = in_use
```

`snapshot_slots()` is called in the post-backward callback after `plan()`
(trace→optimized transition).  `restore_slots()` is called in the root
forward pre-hook before each replay to ensure the same slot availability
as during capture.

## 10. Slot state management

After the trace phase and `plan()`, `snapshot_slots()` freezes the
allocator's slot `in_use` state.  Before each forward pass during replay,
`restore_slots()` resets slots to that snapshot.  This ensures:

- Flexible-mode allocs between microbatches see correct slot availability.
- Non-graphed modules during replay advance `in_use` flags correctly.
- The next replay starts from the same clean state.

## 11. Risks

| Risk | Mitigation |
|------|-----------|
| Side-stream all-gather invisible to graph → corrupt param buffers | Force `cuda_graph_active=True` in root forward pre-hook; hooks suppress side streams; `cuda_graph_compatible` assertion in `FSDPCudaGraphRunner` |
| Post-forward reshard frees buffers too early | Defer reshard until `_post_backward_final_callback` when `cuda_graph_active=True` |
| AccumulateGrad nodes carry stale stream → `cudaErrorStreamCaptureImplicit` | `make_graphed_callables` handles warmup + capture on the same stream, keeping `.grad` tensors alive through warmup |
| Non-deterministic FP8/RNG across warmup iterations | `make_graphed_callables` runs 3 warmup iterations before capture — FP8 scales settle, RNG advances |
| Slot `in_use` flags stale after replay | `restore_slots()` resets to capture-end snapshot before each forward |
| Capture triggered for nested FSDP modules | `_init_fsdp_state` rejects `enable_cuda_graph=True` on modules with FSDP children |
| Graph capture OOM (pool too large) | Pool is pre-allocated by `plan()`; CUDA graph mempool uses same pattern |

## 12. Files

| File | Role |
|------|------|
| `cuda_graph_runner.py` | `FSDPCudaGraphRunner` — capture via `make_graphed_callables`, install/uninstall forward patch |
| `fsdp_module.py` | `_FSDPRootContext.cuda_graph_active` flag, `cuda_graph_compatible` property, nested FSDP check in `_init_fsdp_state` |
| `hooks.py` | Capture trigger in forward pre-hook, reshard deferral in forward/backward hooks, slot restore in root forward pre-hook, cleanup in final callback |
| `allocator.py` | `TracePoolAllocator.snapshot_slots()` / `restore_slots()` for stable slot state |
| `fully_shard.py` | Accept `enable_cuda_graph` kwarg; passes to `_enable_cuda_graph()` |

## 13. No user-visible knobs

| What happens | User action |
|---|---|
| Trace collection | Automatic (MB0 forward) |
| Pool planning | Automatic (MB0 post-backward) |
| Slot snapshot | Automatic (after plan) |
| Graph capture | Automatic (first optimized MB forward, via `FSDPCudaGraphRunner`) |
| Graph replay | Automatic (subsequent MBs, via patched forward) |
| Slot state management | Automatic (snapshot/restore around replay) |
| Side-stream suppression | Automatic (`cuda_graph_active` flag in hooks) |
| Flexible mode toggling | Automatic (existing hook lifecycle) |
