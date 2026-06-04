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
the pipeline schedule.  The system automatically progresses through three
stages across the first three microbatches.

## 3. Why TracePoolAllocator is the enabler

CUDA graphs require **stable buffer addresses**.  After `plan()` allocates
the pool tensor, every slot has a fixed offset.  `_pool_allocate` returns
`pool[offset : offset + size]` — the same `data_ptr()` every time.

During graph capture, the returned views' addresses are recorded.  During
graph replay, `fwd_graph.replay()` replays the captured kernels operating on
those exact addresses.  The allocator is **not called** during replay — the
graph uses the addresses directly.

## 4. Three stages

```
╔═══════════════════════════════════════════════════════════════╗
║ Stage 1 — Microbatch 0 (trace)                                ║
║                                                                ║
║  forward:   _trace_allocate / _trace_free  → trace recorded   ║
║  backward:  _trace_allocate / _trace_free  → trace continues  ║
║  plan():    pool built, _seq_ops populated, _stage = 1        ║
║  → entry condition:  _stage == 0  (default)                   ║
║  → exit condition:   plan() runs, _stage = 1                  ║
╚═══════════════════════════════════════════════════════════════╝
                          │
╔═══════════════════════════════════════════════════════════════╗
║ Stage 2 — Microbatch 1 (warmup + capture)                     ║
║                                                                ║
║  forward:                                                      ║
║    reset_cursor()                                              ║
║    for each graphed FSDPModule:                                ║
║      warmup: run fwd eagerly × 1  (settle FP8 / RNG / caches) ║
║      capture:  torch.cuda.graph(pool=global_mempool):          ║
║        nn.Module.__call__ → hooks fire → _pool_allocate/free  ║
║        → all-gather with async_op=False (capture stream)      ║
║        → post-forward reshard SKIPPED (deferred to bwd end)   ║
║  backward:                                                     ║
║    for each graphed FSDPModule:                                ║
║      warmup: run bwd eagerly × 1                              ║
║      capture:  torch.cuda.graph(pool=global_mempool):          ║
║        → post_backward hook fires (captured)                   ║
║        → reduce-scatter with async_op=False                   ║
║    reshard + reduce_grad  (eager, runs after bwd graph)       ║
║  post-bwd:  snapshot slot state → _stage = 2                  ║
║  → entry condition:  _stage == 1                              ║
║  → exit condition:   both graphs captured, _stage = 2         ║
╚═══════════════════════════════════════════════════════════════╝
                          │
╔═══════════════════════════════════════════════════════════════╗
║ Stage 3 — Microbatch 2+ (replay)                               ║
║                                                                ║
║  forward:                                                      ║
║    restore slot state from capture snapshot                    ║
║    for each graphed FSDPModule:                                ║
║      fwd_graph.replay()  → NO Python hooks, allocator idle    ║
║    for each non-graphed FSDPModule:                            ║
║      normal eager: hooks fire → _pool_allocate/free           ║
║  backward:                                                     ║
║    for each graphed FSDPModule:                                ║
║      bwd_graph.replay()  → NO Python hooks, allocator idle    ║
║    for each non-graphed FSDPModule:                            ║
║      normal eager backward hooks fire                         ║
║  post-bwd:  enable_flexible_mode()                             ║
║  → entry condition:  _stage == 2                              ║
║  → exit condition:   never — stays in replay                  ║
╚═══════════════════════════════════════════════════════════════╝
```

## 5. Why warmup?

`torch.cuda.graph()` records every kernel launch, memory operation, and
stream event inside it.  Libraries like cuDNN, cuBLAS, and TE FP8 perform
**auto-tuning** or **lazy initialization** on their first call — producing
different kernel choices or allocating internal buffers.  Running the
computation once eagerly (warmup) settles all of these.  The second run
(capture) sees the final, stable kernel configuration.

For FSDP v2, **`warmup_steps = 1`** is sufficient because the pool tensor is
pre-allocated — there is no caching-allocator fragmentation to settle.

## 6. Per-FSDPModule selectivity

```python
# All FSDP units graphed (default)
fully_shard(model, enable_cuda_graph=True)

# Only specific layers graphed
fully_shard(layer_1, enable_cuda_graph=True)
fully_shard(layer_2, enable_cuda_graph=True)
fully_shard(layer_3, enable_cuda_graph=False)  # eager for this one
```

Each FSDPModule carries a flag:

```python
class _FSDPState:
    enable_cuda_graph: bool = False
```

During stage 2, only flagged modules are captured.  During stage 3, only
flagged modules replay.  Non-graphed modules traverse the normal eager path
(hooks fire, `_pool_allocate`/`_pool_free` advance `_seq`).

All modules share the same `TracePoolAllocator` — slot assignments are fixed
by `plan()` regardless of which modules are graphed.

## 7. Allocator changes

The allocator only needs three lightweight additions:

```python
class TracePoolAllocator:

    _graph_stage: int = 0             # 0=trace, 1=capture, 2=replay
    _captured_slot_state: List[bool]  # snapshot of slot.in_use at capture end

    # ---- Stage transitions ----

    def advance_graph_stage(self):
        """trace→capture or capture→replay."""
        self._graph_stage += 1

    # ---- Slot snapshot for replay ----

    def snapshot_slots(self):
        """Freeze current slot in_use state after all graphs are captured."""
        self._captured_slot_state = [s.in_use for s in self._slots]

    def restore_slots(self):
        """Restore slot state to capture-time snapshot before each replay."""
        for i, in_use in enumerate(self._captured_slot_state):
            self._slots[i].in_use = in_use

    # ---- allocate/free unchanged ----
    # _pool_allocate and _pool_free are IDENTICAL to current.
    # During replay, the graph replays captured CUDA kernels directly —
    # the allocator is not called from graphed modules.
```

**Why no memoization is needed.**  During `torch.cuda.graph()`, `_pool_allocate`
returns `pool[offset:offset+size]` — a view with a deterministic `data_ptr()`.
The CUDA graph captures this address.  During `fwd_graph.replay()`, the graph
replays the captured kernels operating on that exact memory.  `allocate()` is
never called from graphed modules during replay — the graph handles everything.

For **non-graphed** modules during replay, `_pool_allocate`/`_pool_free` run
normally, advancing `_seq` as in stage 1/2.

## 8. Hook changes

### 8.1 Suppress async side streams during graph mode

When `_graph_stage == 1` (capture), FSDP must use the default/capture stream.
Side-stream operations (all-gather on `ag_stream`, reduce-scatter on
`rs_stream`) are invisible to `torch.cuda.graph()` and would create
unfilled-buffer bugs.

```python
# fsdp_module.py — FSDPModule.unshard()
def unshard(self, async_op=False, bwd_pass=False):
    ctx = self._fsdp_root_context
    if ctx.cuda_graph_active:  # True only during graph capture / replay
        async_op = False
    stream = ctx.ag_stream if async_op else torch.cuda.current_stream()
    ...
```

```python
# fsdp_module.py — FSDPModule.reduce_grad()
def reduce_grad(self, async_op=False):
    ctx = self._fsdp_root_context
    if ctx.cuda_graph_active:
        async_op = False
    stream = ctx.rs_stream if async_op else torch.cuda.current_stream()
    ...
```

### 8.2 Defer post-forward reshard

During capture, the unsharded param buffers are captured as graph inputs.
If `reshard()` frees them immediately after forward, the backward graph has
nothing to read.  Reshard must be deferred until backward completes.

```python
# hooks.py — _register_forward_hook → reshard_param_groups
def reshard_param_groups(module, *unused):
    ctx = module._fsdp_root_context
    if ctx.backward_phase and id(module) == ctx.backward_module:
        return
    if ctx.cuda_graph_active:
        return  # Buffer survives until post-backward cleanup
    module.reshard()
```

The deferred reshard runs in `_post_backward_final_callback`, which already
calls `module.reshard()` for every module.

### 8.3 Toggle `cuda_graph_active` flag

```python
# fsdp_module.py — _FSDPRootContext
@dataclass
class _FSDPRootContext:
    ...
    cuda_graph_active: bool = False
    """True when FSDP is inside a CUDA graph capture or about to replay.
    Suppresses side-stream vs default-stream mismatches and defers reshard."""
```

Set to `True` when entering stage 2 (capture), stays `True` through stage 3
(replay).

## 9. Lifecycle hooks

The root module hooks orchestrate stage transitions:

```python
# hooks.py — _register_root_forward_pre_hook (existing, extended)
def root_forward_pre_hook(_hook_module, *unused):
    ctx = fsdp_module._fsdp_root_context
    if not fsdp_module._fsdp_state._is_root:
        return
    ctx.forward_phase = True
    ctx.backward_phase = False

    ba = ctx.bucket_allocator
    if isinstance(ba, TracePoolAllocator) and ba.phase == "optimized":
        ba.reset_cursor()
        if ba._graph_stage == 0:
            ba.advance_graph_stage()  # trace → capture
        if ba._graph_stage >= 1:
            ba.restore_slots()
            ctx.cuda_graph_active = True
        if ba._graph_stage >= 2:
            pass  # replay — graph handles everything
        ba.disable_flexible_mode()
```

```python
# hooks.py — pre_backward_hook (existing, extended)
def pre_backward_hook(module, grads):
    ...
    if module._fsdp_state._is_root:
        ctx.forward_phase = False
        ctx.backward_phase = True
        # Mark fwd trace end for stage 0 (trace phase)
        ba = ctx.bucket_allocator
        if isinstance(ba, TracePoolAllocator) and ba.phase == "trace":
            ba.mark_fwd_trace_end()
```

```python
# hooks.py — _post_backward_final_callback (existing, extended)
def _post_backward_final_callback(root_state, root_module):
    ...
    ctx.backward_phase = False
    ...
    if isinstance(ctx.bucket_allocator, TracePoolAllocator):
        ba = ctx.bucket_allocator
        if ba.phase == "trace":
            ba.plan()
            ba.reset_cursor()
        elif ba.phase == "optimized":
            if ba._graph_stage == 1:
                ba.snapshot_slots()
                ba.advance_graph_stage()  # capture → replay
            ba.enable_flexible_mode()
```

## 10. Selective per-FSDPModule

```
Graphed:    TransformerLayer  →  FSDP hooks fire inside torch.cuda.graph()
                                → unshard/reshard captured in the graph
                                → replay: fwd_graph.replay(), no Python

Non-graphed: Embedding / OutputHead →  normal eager: hooks fire each pass
                                       → _pool_allocate/free advance _seq
```

The allocator's `_seq` cursor is shared — non-graphed modules still need it
to find their planned slots.  The cursor is reset between forward/backward
via `reset_cursor_fwd()` / `reset_cursor_bwd()`.

## 11. Slot state management

After capture, slots have `in_use` flags reflecting the state at capture end.
Before replay, `restore_slots()` resets them to that snapshot.  This ensures
that:
- Flexible-mode allocs between microbatches see correct slot availability.
- Non-graphed modules during replay advance `in_use` flags correctly.
- The next replay starts from the same clean state.

## 12. Integration with `flexible_mode`

Flexible mode is toggled by the existing hooks:
- Root pre-forward: `disable_flexible_mode()` (enter seq-driven mode)
- Root post-backward: `enable_flexible_mode()` (enter flexible mode for
  inter-microbatch auxiliary allocs)

During CUDA graph capture, allocs go through seq-driven `_pool_allocate`
(flexible is off).  During replay, allocs are not called from graphed
modules.  Non-graphed modules and inter-microbatch allocs use whichever
mode is currently active — unchanged from today.

## 13. Multi-microbatch

```
Microbatch 0:
  forward:  trace
  backward: trace
  post-bwd: plan() → _graph_stage = 1

Microbatch 1:
  forward:  warmup + capture (eager then torch.cuda.graph)
  backward: warmup + capture
  post-bwd: snapshot_slots() → _graph_stage = 2

Microbatch 2:
  forward:  fwd_graph.replay()
  backward: bwd_graph.replay()
  post-bwd: enable_flexible_mode()

Microbatch 3+:
  same as microbatch 2
```

## 14. Risks

| Risk | Mitigation |
|------|-----------|
| Side-stream all-gather invisible to graph → corrupt param buffers | Force `async_op=False` during graph capture; all-gather on default stream |
| Post-forward reshard frees buffers too early | Defer reshard until `_post_backward_final_callback` |
| Non-deterministic FP8/RNG across warmup iterations | Single warmup pass (`warmup_steps=1`) — FP8 scales settle, RNG advances once |
| Slot `in_use` flags stale after replay | `restore_slots()` resets to capture-end snapshot before each forward |
| Flexible allocs overlap with graph-replayed buffers | `restore_slots()` ensures correct `in_use` state; overlap check in `_flex_allocate` catches conflicts |
| Non-graphed modules' `_seq` cursor desynced after replay | Separate `reset_cursor_fwd()` / `reset_cursor_bwd()` for non-graphed path |
| Graph capture OOM (pool too large) | Pool is pre-allocated by `plan()`; CUDA graph mempool uses same pattern — no surprise allocations |

## 15. Files

| File | Changes |
|------|---------|
| `allocator.py` | Add `_graph_stage`, `snapshot_slots()`, `restore_slots()`, `advance_graph_stage()` |
| `fsdp_module.py` | Add `cuda_graph_active` to `_FSDPRootContext`; `async_op=False` in `unshard()`/`reduce_grad()` when active |
| `hooks.py` | Defer `reshard()` in `reshard_param_groups` when graph active; extend `root_forward_pre_hook`, `pre_backward_hook`, `_post_backward_final_callback` for stage transitions |
| `fully_shard.py` | Accept `enable_cuda_graph` kwarg; store in `_FSDPState` |

## 16. No user-visible knobs

| What happens | User action |
|---|---|
| Trace collection | Automatic (MB0 forward) |
| Pool planning | Automatic (MB0 post-backward) |
| Warmup (settle libs) | Automatic (MB1, before capture) |
| Graph capture | Automatic (MB1, inside `torch.cuda.graph()`) |
| Graph replay | Automatic (MB2+, `fwd_graph.replay()`) |
| Slot state management | Automatic (snapshot/restore around replay) |
| Side-stream suppression | Automatic (hooks check `cuda_graph_active`) |
| Flexible mode toggling | Automatic (existing hook lifecycle) |
