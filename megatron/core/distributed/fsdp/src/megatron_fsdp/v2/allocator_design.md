# TracePoolAllocator — v3 design (static key→slot plan)

## Background: Megatron FSDP v2 buffer allocation

Megatron FSDP v2 shards model parameters across GPUs. Before a layer computes
forward/backward, it must **all-gather** (unshard) the parameters — collecting
shards from all ranks into a temporary full-sized buffer. After compute, it
**reshards** — freeing that temporary buffer. The same pattern applies to
gradient reduction.

These temporary buffers are managed by a `BucketAllocator` that provides
`allocate(key, size, dtype, device) → Bucket` and `free(key)`. Each FSDP
parameter group gets its own key — e.g. `(pg_id, "model_weight")` for
parameter all-gather and `(pg_id, "main_grad")` for gradient reduction.

```
Forward:   allocate(model_weight)  →  [compute]  →  free(model_weight)
Backward:  allocate(main_grad)     →  [reduce]   →  free(main_grad)
```

## Why CUDA graph needs stable addresses

CUDA graphs capture a sequence of GPU kernel launches into a single replayable
object. During capture, PyTorch records every kernel launch and the exact GPU
memory addresses those kernels read/write. During replay, the graph replays
those exact operations — it cannot handle the addresses changing.

So if a layer's forward reads from address `0x7f00`, that same layer's replay
must also read from `0x7f00`. Every buffer that a CUDA graph touches must have
a **fixed, deterministic memory address** across all micro-batches.

## Why the current TracePoolAllocator breaks

The current allocator uses a **seq-driven replay schedule**:

1. **Trace** (micro-batch 0): Record every alloc/free call as `(seq, op, key)`.
2. **Plan** (`plan()`): Build intervals, color them, produce a `seq → (op, key, slot)` schedule (`_seq_ops`).
3. **Replay** (subsequent micro-batches): Walk `_seq_ops` linearly — each
   `allocate`/`free` call must arrive at the exact seq position in the exact
   order the trace produced.

**The failure**: When CUDA graph is introduced, `capture_forward` pops all
Python hooks from the module and runs `make_graphed_callables`. During this
warmup+capture, the module's forward+backward runs without hooks — so **no
allocator calls happen** inside the graph region. When hooks are restored,
the `_seq` counter is misaligned with `_seq_ops`. The next `allocate` call
hits a `RuntimeError` because the seq-driven schedule expects a different call.

**The root cause**: The allocator assumes alloc/free call order is invariant,
but CUDA graph replay silently skips hook-triggered calls.

## Design goal

> Keep `TracePoolAllocator` as a single class. Replace the seq-driven replay
> schedule (`_seq_ops`) with a **static key→slot plan** (`_key_to_slot`).
> Every `alloc_key` maps to one fixed memory address — derived once from the
> trace — that works regardless of whether, when, or in what order hooks fire.

## Approach: `plan()` builds a static key→slot mapping

Instead of using the trace as a replay script, use it as **input data for
memory planning**. The trace tells us which allocations overlap in time.
`plan()` runs a greedy interval-coloring algorithm to build a static
`alloc_key → slot` map. Runtime is just a dict lookup:

```python
def allocate(key, ...):
    slot = _key_to_slot[key]                     # dict lookup, O(1)
    return pool[slot.offset : slot.offset + size]  # always same address
```

No `_seq` counter. No `_seq_ops` schedule. No dependency on call order.

## What stays, what changes, what's removed

| Stays unchanged | Changes | Removed |
|---|---|---|
| Trace phase (`_trace_allocate`, `_trace_free`, `_TraceEvent`) | `plan()` builds `_key_to_slot` instead of `_seq_ops` | `_seq`, `_seq_ops` |
| Interval construction from alloc/free pairs | `allocate()` / `free()` dispatch to key→slot lookup instead of seq walk | `_pool_allocate`, `_pool_free` |
| Left-edge interval coloring (`_color_group`) | `reset()` clears `in_use` flags only (replaces `reset_cursor`) | `reset_cursor()` |
| Per-(dtype, device) pool tensors | `enable_flexible_mode` / `disable_flexible_mode` not needed | `_flexible`, `_flex_key_to_slot` |
| `Bucket` dataclass, `BucketAllocator` interface | `_phase` transitions: `"trace"` → `"plan"` → `"optimized"` | `snapshot_slots`, `restore_slots` |
| `_key_to_slot: Dict[alloc_key, slot_idx]` | | `dump_trace()` simplified |

## `plan()` — the core method

```python
def plan(self) -> int:
    """Build a static key→slot plan from the recorded trace.

    1. Replay trace events to pair alloc↔free into intervals.
    2. Group intervals by (dtype, device), color each group with
       greedy left-edge algorithm.
    3. Allocate one flat pool tensor per group.
    4. Build _key_to_slot: every alloc_key → exactly one slot_idx.
    """
```

The left-edge coloring is identical to current `_color_group`. The only
difference: after coloring, we validate that every key maps to exactly
ONE slot (same key across multiple non-overlapping intervals must get
the same slot), and store that in `_key_to_slot`. No `_seq_ops` is built.

## Runtime: allocate / free / reset

```python
def allocate(key, size, dtype, device):
    slot_idx = _key_to_slot[key]
    slot = _slots[slot_idx]
    # Guard: slot free or already owned by this key
    if slot.in_use and key not in _active_keys:
        raise RuntimeError("Slot collision")
    assert size <= slot.size
    if key in _active_keys:
        return Bucket(pool[slot.offset : slot.offset + size])  # re-entrant
    slot.in_use = True
    _active_keys.add(key)
    return Bucket(pool[slot.offset : slot.offset + size])

def free(key):
    if key not in _active_keys:
        return  # double-free or never-allocated
    _slots[_key_to_slot[key]].in_use = False
    _active_keys.discard(key)

def reset():
    """Between micro-batches: clear in_use flags, keep pool and key→slot map."""
    assert _phase == "optimized"
    for slot in _slots:
        slot.in_use = False
    _active_keys.clear()
```

## Visual

```
                    ┌─────────────────────────────┐
                    │ Micro-batch 0 (trace)        │
                    │ Phase: "trace"               │
                    │ Forward+backward runs        │
                    │ All alloc/free calls logged  │
                    └──────────────┬──────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │ plan()              │
                         │ Phase: "plan" →     │
                         │        "optimized"  │
                         │                     │
                         │ 1. Build intervals  │
                         │    from trace       │
                         │ 2. Interval         │
                         │    coloring → slots │
                         │ 3. Allocate pool    │
                         │ 4. Build static     │
                         │    key → slot map   │
                         └─────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼──────┐  ┌─────────▼──────┐  ┌─────────▼──────┐
    │ Micro-batch 1  │  │ Micro-batch 2  │  │ Micro-batch N  │
    │ (capture)      │  │ (replay)       │  │ (replay)       │
    │ Phase:         │  │ Phase:         │  │ Phase:         │
    │ "optimized"    │  │ "optimized"    │  │ "optimized"    │
    │                │  │                │  │                │
    │ alloc(key) →   │  │ alloc(key) →   │  │ alloc(key) →   │
    │   same address │  │   same address │  │   same address │
    └────────────────┘  └────────────────┘  └────────────────┘
```

## Key properties

| Property | How it's achieved |
|---|---|
| Fixed address per key | Pool tensors allocated once in `plan()`, never resized. Each key maps to one slot at a fixed offset. `pool[offset:offset+size]` returns identical view every time. |
| Memory efficiency | Left-edge interval coloring reuses slots for non-overlapping allocations — same compression as current design. |
| CUDA graph compatible | No `_seq` counter, no `_seq_ops` schedule. Graph can skip hooks without desyncing anything. `reset()` at micro-batch boundary. |
| Single class, same name | `TracePoolAllocator` — trace phase unchanged, plan phase builds static map instead of schedule, runtime is key→slot lookup. |
| Same trace mechanism | Phase 1 (trace) is identical. Only the plan output and runtime dispatch change. |
