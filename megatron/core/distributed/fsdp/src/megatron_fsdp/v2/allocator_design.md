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

**The failure**: When CUDA graph is introduced, the seq-driven replay cannot
adapt. Hook calls during capture warmup, capture, and replay may arrive in
different orders or at different relative positions from the trace. The
`_seq` counter drifts from `_seq_ops`, and future `allocate`/`free` calls
hit `RuntimeError` because the seq-driven schedule expects a different call
at the current position.

**The root cause**: The allocator assumes a fixed, repeatable alloc/free call
order and wraps it into a position-indexed schedule (`_seq_ops`). CUDA graph
capture and replay break this assumption — calls can be reordered, omitted,
or duplicated relative to the trace.

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
| Per-(dtype, device) pool tensors | `enable_flexible_mode` / `disable_flexible_mode` not needed | `_flexible`, `_flex_key_to_slot` |
| `Bucket` dataclass, `BucketAllocator` interface | `_phase` transitions: `"trace"` → `"optimized"` (no intermediate `"plan"` phase held) | `snapshot_slots`, `restore_slots` |
| `_key_to_slot: Dict[alloc_key, slot_idx]` | | `reset_cursor()` |

### Why flexible mode is removed

In the current design, `enable_flexible_mode` / `disable_flexible_mode` provide
key→slot lookup for auxiliary allocations between micro-batches (e.g. weight
quantisation). In v3, **every** `allocate()` is already a key→slot lookup — the
flexible-mode path is the **default** path. A separate toggle is unnecessary.

Between micro-batches, `reset_batch()` clears `in_use` and `_active_keys` but
preserves `_key_to_slot` and `_pools`. An auxiliary `allocate(quant_key)` will
find the slot free → works exactly as before.

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

### The coloring algorithm (with same-key→same-slot enforcement)

The left-edge algorithm must enforce that **the same key always maps to
exactly one slot**, even when the key produces multiple non-overlapping
intervals. This is NOT a post-hoc validation — it is enforced **during**
coloring.

```python
def _color_group(intervals, dtype, device) -> int:
    sorted_intervals = sorted(intervals, key=lambda iv: iv.alloc_seq)
    free_slots = []           # (local_slot_idx, free_seq)
    group_slots = []          # Slot objects for this group
    local_to_global = {}      # local → global slot index
    key_to_slot = {}          # key → assigned global slot_idx (algo-internal)

    for iv in sorted_intervals:
        # ── same-key constraint: force reuse of the key's assigned slot ──
        assigned_global = key_to_slot.get(iv.key)
        if assigned_global is not None:
            # Find the local slot corresponding to this global index
            assigned_local = ... # lookup from local_to_global
            # The slot must be free at this interval's start
            assert assigned_local is free at iv.alloc_seq, (
                f"key {iv.key} has overlapping intervals — this should be "
                f"impossible in FSDP (same key never alloc'd twice without free)"
            )
            # Resize if this interval needs more capacity
            if iv.size > group_slots[assigned_local].size:
                group_slots[assigned_local].size = iv.size
            # Update free time of this slot to this interval's free_seq
            update_free_slots(free_slots, assigned_local, iv.free_seq)
            # Skip the allocation pool scan; slot is already reserved
            continue

        # ── normal left-edge: reuse an existing free slot, or create new ──
        assigned_local = None
        for local_idx, slot_free_seq in free_slots:
            if slot_free_seq < iv.alloc_seq:
                slot = group_slots[local_idx]
                if iv.size > slot.size:
                    slot.size = iv.size
                # Update this slot's free time to this interval's end
                free_slots[...] = (local_idx, iv.free_seq)
                assigned_local = local_idx
                break

        if assigned_local is None:
            # Need a new slot
            assigned_local = len(group_slots)
            global_idx = len(self._slots)
            local_to_global[assigned_local] = global_idx
            slot = Slot(offset=0, size=iv.size, dtype=dtype, device=device)
            group_slots.append(slot)
            self._slots.append(slot)
            free_slots.append((assigned_local, iv.free_seq))

        # Record key→slot assignment (first-seen wins; subsequent hits the
        # same-key constraint branch above)
        global_idx = local_to_global.get(assigned_local,
                       key_to_slot.get(iv.key))
        key_to_slot[iv.key] = global_idx
        # No _seq_ops entry; slot assignment is stored later in self._key_to_slot

    # Lay out slots contiguously with alignment
    offset = 0
    alignment = _get_alignment(device, dtype)  # see §Memory alignment
    for slot in group_slots:
        offset = (offset + alignment - 1) // alignment * alignment
        slot.offset = offset
        offset += slot.size

    if offset > 0:
        self._pools[(dtype, device)] = torch.empty(offset, dtype=dtype, device=device)

    for key, global_idx in key_to_slot.items():
        self._key_to_slot[key] = global_idx

    return offset
```

### Memory alignment

Each slot's offset is aligned to a device- and dtype-aware minimum. This is
critical for NVFP4 (sub-byte) types and CUDA kernel alignment requirements
(e.g. 256-byte base alignment from `cudaMalloc`):

```python
def _get_alignment(device, dtype):
    """Return the minimum alignment (in elements) for the given device/dtype."""
    element_bytes = torch.empty(0, dtype=dtype, device=device).element_size()
    # At minimum, align to the element size itself
    align_bytes = max(
        element_bytes,
        torch.cuda.get_device_properties(device).texture_alignment
        if device.type == "cuda" else 1,
    )
    return align_bytes // element_bytes
```

### Coloring algorithm — known limitations and future improvements

The current greedy left-edge algorithm with same-key→same-slot enforcement
meets the basic requirements but has room for improvement. Items to review
and discuss:

1. **Per-group isolation**: Coloring runs independently per `(dtype, device)`
   group. Groups on the same device could share a single pool with a unified
   coloring pass, potentially reducing total memory. This requires handling
   different element sizes in the same pool (offset arithmetic must account
   for dtype-specific strides).

2. **Sub-optimal slot reuse**: When a multi-interval key forces slot reuse,
   the coloring for unrelated intervals is impacted — the forced slot may
   spend more time "occupied" (because the key's first interval starts early
   and its last interval ends late), preventing reuse by other keys that
   could fit between the key's intervals. The current algorithm pins a slot
   to a key for its entire traced lifetime, which is correct for correctness
   but may over-reserve.

3. **Alternatives worth exploring**:
   - **Minimum slot count via ILP**: The interval-graph coloring problem
     (minimum chromatic number) has polynomial-time solutions that guarantee
     optimality. This could reduce slot count at the cost of implementation
     complexity.
   - **Slot merging pass**: After coloring, adjacent slots with compatible
     dtypes could be merged if their intervals never overlap and no alignment
     constraints are violated.
   - **Profiling-guided sizing**: The trace gives exact `size` values, but
     if sizes vary slightly across micro-batches (e.g. dynamic shapes), the
     plan should either over-allocate conservatively or support a resizing
     fallback.
   - **Size-class bucketing**: Group intervals by size class to reduce
     internal fragmentation when a small interval forces a large slot.

4. **Evaluation criteria**:
   - Total pool bytes vs. the per-key `torch.empty` baseline.
   - Slot count (fewer slots = less metadata overhead).
   - Internal fragmentation (unused bytes within each slot, `slot.size - max_iv_size`).
   - Alignment waste (padding bytes between slots).

**Plan**: Ship with the current algorithm and instrument it with pool-size
and fragmentation metrics. Collect traces from real workloads (large models,
varied parallelism configs). Use the data to guide which improvements are
worth the complexity.

## Runtime: allocate / free / reset

```python
def allocate(key, size, dtype, device):
    slot_idx = _key_to_slot[key]    # raises KeyError if key never traced
    slot = _slots[slot_idx]
    # Guard: slot free or already owned by this key
    if slot.in_use and key not in _active_keys:
        raise RuntimeError(
            f"Slot collision at slot[{slot_idx}]: key={key} "
            f"but slot is held by active key(s)"
        )
    assert size <= slot.size, (
        f"requested {size} > slot capacity {slot.size} (key={key})"
    )
    if key in _active_keys:
        # Re-entrant: key already allocated this micro-batch
        # (e.g. double-allocate within same iteration — idempotent)
        return Bucket(data=pool[slot.offset : slot.offset + size])
    slot.in_use = True
    _active_keys.add(key)
    return Bucket(data=pool[slot.offset : slot.offset + size])

def free(key):
    if key not in _active_keys:
        return  # double-free or never-allocated → silent no-op
    _slots[_key_to_slot[key]].in_use = False
    _active_keys.discard(key)

def reset_batch():
    """Between micro-batches: clear in_use flags and active keys.

    Does NOT discard _key_to_slot or _pools — the slot→address mapping
    is immutable once planned.

    Called at root pre-forward of each micro-batch after plan().
    """
    assert _phase == "optimized"
    for slot in _slots:
        slot.in_use = False
    _active_keys.clear()

def reset():
    """Full teardown: discard pool, plan, and trace; return to "trace" phase.

    Used for model re-initialization or full training restart.
    """
    self._phase = "trace"
    self._seq = 0            # (present only in trace phase)
    self._trace.clear()
    self._trace_meta.clear()
    self._buckets.clear()
    self._active_keys.clear()
    self._pools.clear()
    self._key_to_slot.clear()
    self._slots.clear()
```

### Error handling for unknown keys

If `allocate(key)` is called with a key never seen during trace, `_key_to_slot`
raises `KeyError`. The caller must re-trace (call `reset()`, re-run micro-batch
0, then `plan()`) to pick up new allocation patterns. This guarantees that all
keys used during CUDA graph replay were planned, keeping addresses stable.

No on-the-fly slot allocation is supported — it would change pool tensor sizes
and invalidate all captured CUDA graphs.

## CUDA graph integration lifecycle

The full lifecycle spanning trace, plan, capture, and replay:

```
                    ┌─────────────────────────────────────────┐
                    │ Micro-batch 0 (trace)                    │
                    │ Phase: "trace"                           │
                    │                                          │
                    │ Pre-forward:  _active_keys clear         │
                    │ Forward:      alloc/free → _trace_*      │
                    │               records; individual        │
                    │               torch.empty tensors        │
                    │               (NO CUDA graph — pool      │
                    │                does not exist yet)       │
                    │ Backward:     alloc/free → _trace_*      │
                    │               records                    │
                    │ Post-backward: plan() builds             │
                    │                _key_to_slot, allocates   │
                    │                pool tensors              │
                    │                Phase → "optimized"       │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼──────────┐   ┌────────▼───────────┐   ┌───────▼───────────┐
    │ Micro-batch 1      │   │ Micro-batch 2       │   │ Micro-batch N     │
    │ (capture)          │   │ (replay)            │   │ (replay)          │
    │ Phase: "optimized" │   │ Phase: "optimized"  │   │ Phase: "optimized"│
    │                    │   │                     │   │                   │
    │ Pre-forward:       │   │ Pre-forward:        │   │ Pre-forward:      │
    │   reset_batch()    │   │   reset_batch()     │   │   reset_batch()   │
    │                    │   │                     │   │                   │
    │ Per-module         │   │                     │   │                   │
    │ pre-forward:       │   │   graphed() call    │   │   graphed() call  │
    │   alloc param      │   │   (replay capture)  │   │   (replay capture)│
    │     → key→slot     │   │                     │   │                   │
    │   alloc main_grad  │   │   pool tensors      │   │   pool tensors    │
    │     → key→slot     │   │   at SAME addr      │   │   at SAME addr    │
    │                    │   │   as MB 1           │   │   as MB 1         │
    │   make_graphed_    │   │                     │   │                   │
    │   callables:       │   │                     │   │                   │
    │   • 3 warmup iters │   │                     │   │                   │
    │   • pop hooks      │   │                     │   │                   │
    │   • capture fwd    │   │                     │   │                   │
    │     into graph     │   │                     │   │                   │
    │   • restore hooks  │   │                     │   │                   │
    │                    │   │                     │   │                   │
    │ Post-bwd hooks     │   │ Post-bwd hooks      │   │ Post-bwd hooks    │
    │   fire (eager)     │   │   fire              │   │   fire            │
    │                    │   │                     │   │                   │
    │ Post-backward:     │   │ Post-backward:      │   │ Post-backward:    │
    │   (hooks clean up) │   │   (hooks clean up)  │   │   (hooks clean up)│
    └────────────────────┘   └─────────────────────┘   └───────────────────┘
```

### Per-module CUDA graph capture detail (MB 1)

FSDP captures one CUDA graph per compatible leaf module (not the whole model).
The capture is triggered in the module's forward pre-hook
(`unshard_param_groups`):

1. **Pre-forward hook allocates param**: The hook calls
   `allocate(pg_id, "model_weight")` → key→slot lookup → pool view at fixed
   address. FSDP all-gathers shards into that view. The param buffer is now
   resident at a deterministic address for the rest of the pool's lifetime.

2. **main_grad is manually allocated**: Before capture, the gradient buffer
   is pre-allocated via `allocate(pg_id, "main_grad")` to ensure both forward
   and backward passes see fixed addresses.

3. **Warmup**: `make_graphed_callables` runs 3 warmup iterations. Hooks fire
   during warmup, so `allocate`/`free` calls happen normally. Key→slot returns
   the same addresses every iteration — cuDNN/cuBLAS auto-tune settles,
   TE FP8 scales converge.

4. **Actual capture**: `_pop_hooks` removes FSDP hooks → only the user's
   `forward()` is captured as a CUDA graph. The graph records GPU kernels
   reading/writing the pool tensor views at their fixed addresses (both param
   and main_grad). Hooks are restored immediately after.

5. **Replay**: The patched forward calls `graphed(*flat)`. Since the graph
   captured the fixed pool addresses, replay reads/writes the **same**
   addresses every time — no re-allocation, no address change.

### Key safety property

During capture, both param and grad buffers are allocated from the pool at
their planned slots before `make_graphed_callables` runs. The CUDA graph then
records kernel operations against those specific memory addresses. Because the
pool tensors are allocated **once** in `plan()` and never resized, the same
`key` will always resolve to the same address — regardless of call ordering
variations across micro-batches.

## Visual

```
                      ┌───────────────────────────────┐
                      │ Micro-batch 0 (trace)          │
                      │ Phase: "trace"                 │
                      │ Forward + backward runs        │
                      │ All alloc/free calls logged    │
                      └──────────────┬────────────────┘
                                     │
                            ┌────────▼──────────┐
                            │ plan()              │
                            │ Phase → "optimized" │
                            │                     │
                            │ 1. Build intervals  │
                            │    from trace       │
                            │ 2. Interval coloring│
                            │    → slots          │
                            │    (enforce same-   │
                            │     key→same-slot)  │
                            │ 3. Align & allocate │
                            │    pool tensors     │
                            │ 4. Build static     │
                            │    key → slot map   │
                            └─────────┬───────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼───────┐     ┌────────▼────────┐     ┌───────▼─────────┐
    │ Micro-batch 1   │     │ Micro-batch 2    │     │ Micro-batch N   │
    │ (capture)       │     │ (replay)         │     │ (replay)        │
    │ Phase:          │     │ Phase:           │     │ Phase:          │
    │ "optimized"     │     │ "optimized"      │     │ "optimized"     │
    │                 │     │                  │     │                 │
    │ alloc(key) →    │     │ alloc(key) →     │     │ alloc(key) →    │
    │   same address  │     │   same address   │     │   same address  │
    └─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Hooks integration (updated for v3)

```
Micro-batch 0
┌───────────────────────────────────────────────────────────────────────────┐
│  root pre-forward     forward_phase = True                                 │
│    forward (trace)                                                         │
│  root pre-backward    forward_phase = False , backward_phase = True        │
│    backward (trace)                                                        │
│  root post-backward   backward_phase = False                               │
│                       plan() → "optimized"                                 │
│                       (no enable_flexible_mode needed)                     │
└───────────────────────────────────────────────────────────────────────────┘

Micro-batch 1+
┌───────────────────────────────────────────────────────────────────────────┐
│  root pre-forward     forward_phase = True                                 │
│                       reset_batch()          ← clears in_use               │
│  per-module pre-forward:                                                   │
│    allocate(param) → key→slot lookup                                       │
│    allocate(main_grad) → key→slot lookup (manual, before capture)         │
│    → if enable_cuda_graph and not yet captured:                            │
│        FSDPCudaGraphRunner.capture_forward()                               │
│    forward (optimized, key→slot lookup)                                    │
│  root pre-backward    forward_phase = False , backward_phase = True        │
│    backward (optimized, key→slot lookup)                                   │
│  root post-backward   backward_phase = False                               │
│                       (no enable_flexible_mode needed —                    │
│                        between-batch allocs work natively)                 │
└───────────────────────────────────────────────────────────────────────────┘
```

## Debug: `dump_trace()`

Updated for v3 to show the full static key→slot mapping (not just active keys):

```python
def dump_trace(self) -> str:
    lines = [f"=== TracePoolAllocator (phase={self._phase}) ==="]
    # ... trace events (unchanged) ...

    if self._phase == "optimized":
        lines.append(f"\nslots: {len(self._slots)}")
        for i, slot in enumerate(self._slots):
            lines.append(
                f"  slot[{i}]: offset={slot.offset} size={slot.size} "
                f"dtype={slot.dtype} device={slot.device} "
                f"{'in_use' if slot.in_use else 'free'}"
            )
        total_bytes = sum(
            s.size * torch.empty(0, dtype=s.dtype).element_size()
            for s in self._slots
        )
        lines.append(f"\ntotal pool: {len(self._slots)} slots, {total_bytes} bytes")
        lines.append(f"\nkey_to_slot ({len(self._key_to_slot)} entries):")
        for key, slot_idx in sorted(self._key_to_slot.items(), key=lambda x: str(x[0])):
            slot = self._slots[slot_idx]
            lines.append(
                f"  {key} -> slot[{slot_idx}] "
                f"(offset={slot.offset}, size={slot.size}, "
                f"address=0x{pool[slot.offset].data_ptr():x})"
            )
        lines.append(f"\nactive_keys ({len(self._active_keys)}):")
        for key in self._active_keys:
            lines.append(f"  {key}")
    return "\n".join(lines)
```

## Key properties

| Property | How it's achieved |
|---|---|
| Fixed address per key | Pool tensors allocated once in `plan()`, never resized. Each key maps to one slot at a fixed, aligned offset. `pool[offset:offset+size]` returns identical view every time. |
| Memory efficiency | Left-edge interval coloring reuses slots for non-overlapping allocations. Same-key→same-slot enforcement shares a single slot across multiple non-overlapping intervals of the same key. |
| CUDA graph compatible | No `_seq` counter, no `_seq_ops` schedule. Key→slot lookup works regardless of call order, duplication, or omission relative to the trace. `reset_batch()` at micro-batch boundary. |
| No fragmentation within pool | Slots are laid out contiguously with alignment padding. No gaps between slots (only alignment-padding gaps). No dynamic allocation/deallocation — pool is a single `torch.empty`. |
| Simple implementation | `allocate()` is a dict lookup + guard. `free()` is a flag clear. `reset_batch()` clears flags and active set. No seq walking, no fast-forward logic. |
| Same trace mechanism | Phase 1 (trace) is identical. Only the plan output and runtime dispatch change. |
| Backward compatible for non-CUDA-graph users | The key→slot runtime also works for regular eager execution — it's strictly simpler than the seq-driven approach. |
