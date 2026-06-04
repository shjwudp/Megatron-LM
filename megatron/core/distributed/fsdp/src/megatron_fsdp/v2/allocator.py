# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Dict, Hashable, List, Optional, Set, Tuple

import torch

AllocatorKey = Hashable


def _resolve_key(key: Optional[AllocatorKey], param_group_id: Optional[AllocatorKey]):
    if key is not None:
        return key
    # Backward-compatible alias for older callers that keyed only by param group.
    assert param_group_id is not None, "allocator key is required"
    return param_group_id


@dataclasses.dataclass
class Bucket:
    """Lightweight container for a temporary allocated tensor buffer."""

    data: torch.Tensor


class BucketAllocator:
    """Interface for allocating and freeing temporary buckets."""

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        """Allocate a bucket for the given key."""
        raise NotImplementedError

    def free(
        self, key: Optional[AllocatorKey] = None, *, param_group_id: Optional[AllocatorKey] = None
    ) -> None:
        """Free the bucket associated with the given key."""
        raise NotImplementedError


class TemporaryBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by a caller-provided key.

    Used by DataParallelBuffer for unshard (all-gather) and gradient
    reduction (reduce-scatter) operations.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        key = _resolve_key(key, param_group_id)
        assert dtype is not None and device is not None
        if key not in self.buckets:
            self.buckets[key] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        return self.buckets[key]

    def free(
        self, key: Optional[AllocatorKey] = None, *, param_group_id: Optional[AllocatorKey] = None
    ) -> None:
        key = _resolve_key(key, param_group_id)
        if key in self.buckets:
            _free_storage(self.buckets[key].data)
            del self.buckets[key]


class StorageFreeingBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by caller-provided allocation key.

    Freeing releases the underlying storage without deleting the bucket entry,
    so the same tensor object can be reused on the next allocation.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        key = _resolve_key(key, param_group_id)
        assert dtype is not None and device is not None
        if key not in self.buckets:
            self.buckets[key] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
            return self.buckets[key]
        _alloc_storage(self.buckets[key].data, torch.Size([size]))
        return self.buckets[key]

    def free(
        self, key: Optional[AllocatorKey] = None, *, param_group_id: Optional[AllocatorKey] = None
    ) -> None:
        key = _resolve_key(key, param_group_id)
        if key in self.buckets:
            _free_storage(self.buckets[key].data)


class TracePoolAllocator(BucketAllocator):
    """Two-phase bucket allocator that eliminates per-call ``torch.empty`` overhead.

    **Design**

    The FSDP framework allocates and frees temporary flat buffers (for
    all-gather input/output and gradient accumulation) in a deterministic,
    repeatable order across micro-batches.  ``TracePoolAllocator`` exploits
    this by profiling one pass and then serving all subsequent passes from
    a pre-allocated pool.

    **Phase 1 — Trace** (``plan()`` not yet called)

    Behaves like ``TemporaryBucketAllocator``: ``allocate`` creates a
    ``torch.empty`` bucket on first use, ``free`` releases its storage.
    Additionally, every alloc/free call is recorded as a ``_TraceEvent``
    with a monotonic ``seq`` number, and metadata ``(size, dtype, device)``
    is stored per allocation key for later planning.

    **Phase 2 — Plan** (``plan()``)

    The trace is replayed to extract *intervals*: for each alloc/free pair
    an ``_Interval(alloc_seq, free_seq, size)`` is built.  Intervals are
    grouped by ``(dtype, device)`` and then colored with a greedy
    left-edge algorithm:

    1. Sort intervals by ``alloc_seq``.
    2. For each interval, try to reuse a *slot* whose previous occupant
       freed before this interval starts (``slot_free_seq < alloc_seq``).
    3. If no slot is free, allocate a new one.
    4. Grow the slot's capacity to ``max(size, current)``.
    5. Record the assignment: ``_seq_ops[alloc_seq] = ("alloc", key, slot)``
       and ``_seq_ops[free_seq] = ("free", key, None)``.

    After coloring, slots are laid out contiguously and a single
    ``torch.empty`` per ``(dtype, device)`` group is allocated. If the
    trace is empty, planning is a no-op and later cursor resets also no-op.

    **Phase 3 — Optimized** (after ``plan()``)

    ``allocate`` and ``free`` dispatch via ``_seq_ops``, a unified
    seq→action map.  When a planned entry does not match the live call
    (e.g., the planned key is cached from a previous micro-batch),
    ``_seq`` fast-forwards past it automatically.  Call
    ``reset_cursor()`` at the start of each micro-batch to rewind
    ``_seq`` to 0 (``_key_to_slot`` survives for cached keys).

    The trace pattern must be **repeatable** — the same alloc/free call
    sequence is expected every micro-batch.

    **Flexible mode** (``enable_flexible_mode`` / ``disable_flexible_mode``)

    When enabled, ``allocate`` and ``free`` bypass the seq-driven replay
    entirely.  Each key maps directly to its first planned slot (built once
    from ``_seq_ops``).  An overlap check catches a key whose slot is
    occupied by a different live key.  This is intended for auxiliary
    allocations (e.g. weight quantisation buffers) that occur between
    micro-batches while the allocator is idle.

    **Hook-coordinated lifecycle** (see ``megatron_fsdp.v2.hooks``)::

        Micro-batch 0
        ┌──────────────────────────────────────────────────────────
        │  root pre-forward     forward_phase = True
        │    forward (trace)
        │  root pre-backward    forward_phase = False , backward_phase = True
        │    backward (trace)
        │  root post-backward   backward_phase = False
        │                       plan() → optimized
        │                       enable_flexible_mode()   ← flexible ON
        └──────────────────────────────────────────────────────────

        Micro-batch 1+
        ┌──────────────────────────────────────────────────────────
        │  root pre-forward     forward_phase = True
        │                       disable_flexible_mode()  ← flexible OFF
        │                       reset_cursor()
        │    forward (optimized, seq-driven)
        │  root pre-backward    forward_phase = False , backward_phase = True
        │    backward (optimized, seq-driven)
        │  root post-backward   backward_phase = False
        │                       enable_flexible_mode()   ← flexible ON (idle)
        └──────────────────────────────────────────────────────────
    """

    # -- Inner types ---------------------------------------------------- #

    class _Slot:
        """A contiguous slice of the pool tensor assigned to one or more
        non-overlapping intervals."""

        __slots__ = ("offset", "size", "dtype", "device", "in_use")

        def __init__(self, offset: int, size: int, dtype: torch.dtype, device: torch.device):
            self.offset = offset
            self.size = size
            self.dtype = dtype
            self.device = device
            self.in_use = False

    @dataclasses.dataclass
    class _TraceEvent:
        """A single alloc or free recorded during the trace phase."""

        seq: int
        op: str  # "alloc" | "free"
        key: AllocatorKey

    @dataclasses.dataclass
    class _Interval:
        """An allocation's lifetime: from alloc_seq to free_seq with a given size."""

        key: AllocatorKey
        size: int
        alloc_seq: int
        free_seq: int

    # -- Init ----------------------------------------------------------- #

    def __init__(self) -> None:
        super().__init__()
        # Phase bookkeeping
        self._phase: str = "trace"  # "trace" | "optimized"
        self._seq: int = 0  # monotonic alloc/free counter

        # Trace state
        self._trace: List["TracePoolAllocator._TraceEvent"] = []
        self._trace_meta: Dict[AllocatorKey, Tuple[int, torch.dtype, torch.device]] = {}
        self._buckets: Dict[AllocatorKey, Bucket] = {}  # only used in trace phase
        self._active_keys: Set[AllocatorKey] = set()  # keys currently allocated

        # Pool state — populated by plan(), used in optimized phase
        self._pools: Dict[Tuple[torch.dtype, torch.device], torch.Tensor] = {}
        self._slots: List["TracePoolAllocator._Slot"] = []
        # seq-driven schedule: seq -> ("alloc", key, slot_idx) | ("free", key, None)
        self._seq_ops: Dict[int, Tuple[str, AllocatorKey, Optional[int]]] = {}
        self._key_to_slot: Dict[AllocatorKey, int] = {}  # active key -> slot_idx

        # Flexible mode — dispenses with seq-driven replay; each key maps
        # directly to its first planned slot.
        self._flexible: bool = False
        self._flex_key_to_slot: Dict[AllocatorKey, int] = {}

        # CUDA graph slot-state snapshot for replay restoration
        self._captured_in_use: List[bool] = []

    # -- Phase 1: trace -------------------------------------------------- #

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        """Dispatch to trace, pool, or flexible path depending on phase."""
        key = _resolve_key(key, param_group_id)
        assert dtype is not None and device is not None
        if self._phase != "optimized":
            return self._trace_allocate(key, size, dtype, device)
        if self._flexible:
            return self._flex_allocate(key, size, dtype, device)
        return self._pool_allocate(key, size, dtype, device)

    def free(
        self, key: Optional[AllocatorKey] = None, *, param_group_id: Optional[AllocatorKey] = None
    ) -> None:
        """Dispatch to trace, pool, or flexible path depending on phase."""
        key = _resolve_key(key, param_group_id)
        if self._phase != "optimized":
            self._trace_free(key)
        elif self._flexible:
            self._flex_free(key)
        else:
            self._pool_free(key)

    def _trace_allocate(
        self, key: AllocatorKey, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Trace-phase allocate — idempotent.

        - First alloc for a key: records trace event, creates bucket.
        - Duplicate alloc (key still active): no-op, returns existing bucket.
        - Re-alloc after free: resurrects storage, records new trace event.
        """
        if key in self._active_keys:
            return self._buckets[key]

        if key not in self._buckets:
            self._trace.append(self._TraceEvent(seq=self._seq, op="alloc", key=key))
            self._seq += 1
            self._trace_meta[key] = (size, dtype, device)
            self._buckets[key] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        else:
            # Key was freed — resurrect the same tensor object.
            self._trace.append(self._TraceEvent(seq=self._seq, op="alloc", key=key))
            self._seq += 1
            _alloc_storage(self._buckets[key].data, torch.Size([size]))

        self._active_keys.add(key)
        return self._buckets[key]

    def _trace_free(self, key: AllocatorKey) -> None:
        """Trace-phase free — idempotent.

        - Free of an active key: records trace event, releases storage.
        - Free of an inactive key (double-free or never-allocated): no-op.
        """
        if key not in self._active_keys:
            return
        self._trace.append(self._TraceEvent(seq=self._seq, op="free", key=key))
        self._seq += 1
        if key in self._buckets:
            _free_storage(self._buckets[key].data)
        self._active_keys.discard(key)

    # -- Phase 2: plan --------------------------------------------------- #

    def plan(self) -> int:
        """Build the static pool from the recorded trace.

        1. Replay the trace to pair alloc/free events into ``_Interval`` objects.
        2. Pair any un-freed allocs with sentinel free sequences so
           persistent keys get coloured and reserved slots.
        3. Group intervals by ``(dtype, device)``.
        4. Color each group with the greedy left-edge algorithm.
        5. Allocate one flat pool tensor per group.
        6. Pre-populate ``_key_to_slot`` with persistent key slots.

        Returns:
            Total pool size in **elements** (sum across all groups).
            Multiply by ``element_size(dtype)`` for bytes.
        """
        assert self._phase == "trace", "plan() can only be called in trace phase"
        if len(self._trace) == 0:
            self._phase = "optimized"
            return 0

        # ---- step 1: build intervals from alloc/free pairs ----
        alloc_stack: Dict[AllocatorKey, List[int]] = {}  # key -> [alloc_seq, ...]
        intervals: List["TracePoolAllocator._Interval"] = []

        for ev in self._trace:
            if ev.op == "alloc":
                alloc_stack.setdefault(ev.key, []).append(ev.seq)
            else:  # "free"
                if ev.key in alloc_stack and alloc_stack[ev.key]:
                    alloc_seq = alloc_stack[ev.key].pop(0)
                    meta = self._trace_meta.get(ev.key)
                    if meta is not None:
                        size, dtype, device = meta
                        intervals.append(
                            self._Interval(
                                key=ev.key, size=size, alloc_seq=alloc_seq, free_seq=ev.seq
                            )
                        )

        # Keys that were allocated but not freed by the end of the trace
        # persist across micro-batches. Pair them with a sentinel free_seq
        # so they get coloured and their slots are reserved for the lifetime
        # of the pool.
        _SENTINEL_FREE_SEQ = 1 << 60
        sentinel_seq = _SENTINEL_FREE_SEQ
        for key, alloc_seqs in alloc_stack.items():
            for alloc_seq in alloc_seqs:
                meta = self._trace_meta.get(key)
                if meta is not None:
                    size, dtype, device = meta
                    intervals.append(
                        self._Interval(
                            key=key, size=size, alloc_seq=alloc_seq,
                            free_seq=sentinel_seq,
                        )
                    )
                    sentinel_seq += 1

        if len(intervals) == 0:
            self._phase = "optimized"
            return 0

        # ---- step 2 & 3: color and allocate ----
        total_elems = self._assign_pool(intervals)

        # Transfer persistent (unfreed) keys into _key_to_slot so the
        # optimized phase recognises them as cached from the start.
        for key, alloc_seqs in alloc_stack.items():
            for alloc_seq in alloc_seqs:
                op = self._seq_ops.get(alloc_seq)
                if op is not None and op[0] == "alloc":
                    self._key_to_slot[key] = op[2]

        self._phase = "optimized"
        return total_elems

    def _assign_pool(self, intervals: List["TracePoolAllocator._Interval"]) -> int:
        """Group intervals by (dtype, device), color each group, sum sizes."""
        groups: Dict[Tuple[torch.dtype, torch.device], List["TracePoolAllocator._Interval"]] = {}
        for iv in intervals:
            meta = self._trace_meta[iv.key]
            dtype_device = (meta[1], meta[2])
            groups.setdefault(dtype_device, []).append(iv)

        # Clear any previous plan state before rebuilding
        self._slots.clear()
        self._pools.clear()
        self._seq_ops.clear()
        self._key_to_slot.clear()

        total_elems = 0
        for (dtype, device), group in groups.items():
            total_elems += self._color_group(group, dtype, device)

        return total_elems

    def _color_group(
        self,
        intervals: List["TracePoolAllocator._Interval"],
        dtype: torch.dtype,
        device: torch.device,
    ) -> int:
        """Greedy left-edge interval coloring for one (dtype, device) group.

        Records the schedule in ``_seq_ops`` as ``("alloc", key, slot)``
        and ``("free", key, None)`` entries so the optimized phase can
        replay without per-key cursor tracking.
        """
        sorted_intervals = sorted(intervals, key=lambda iv: iv.alloc_seq)

        free_slots: List[Tuple[int, int]] = []  # (local_slot_index, free_seq)
        group_slots: List["TracePoolAllocator._Slot"] = []
        local_to_global: Dict[int, int] = {}

        for iv in sorted_intervals:
            assigned = False
            for i, (slot_idx, slot_free_seq) in enumerate(free_slots):
                if slot_free_seq < iv.alloc_seq:
                    slot = group_slots[slot_idx]
                    if iv.size > slot.size:
                        slot.size = iv.size
                    free_slots[i] = (slot_idx, iv.free_seq)
                    self._seq_ops[iv.alloc_seq] = ("alloc", iv.key, local_to_global[slot_idx])
                    self._seq_ops[iv.free_seq] = ("free", iv.key, None)
                    assigned = True
                    break

            if not assigned:
                local_idx = len(group_slots)
                global_idx = len(self._slots)
                local_to_global[local_idx] = global_idx
                slot = self._Slot(offset=0, size=iv.size, dtype=dtype, device=device)
                group_slots.append(slot)
                self._slots.append(slot)
                free_slots.append((local_idx, iv.free_seq))
                self._seq_ops[iv.alloc_seq] = ("alloc", iv.key, global_idx)
                self._seq_ops[iv.free_seq] = ("free", iv.key, None)

        # Lay out slots contiguously within the group pool
        offset = 0
        for slot in group_slots:
            slot.offset = offset
            offset += slot.size

        if offset > 0:
            self._pools[(dtype, device)] = torch.empty(offset, dtype=dtype, device=device)
        return offset

    # -- Phase 3: optimized runtime ------------------------------------- #
    #
    # ``_seq_ops`` maps each ``alloc_seq``/``free_seq`` to the planned
    # action. During replay the live alloc/free sequence must repeat the
    # trace order. When a planned entry targets a key that is already
    # in ``_key_to_slot`` (cached from a previous micro-batch), the seq
    # counter fast-forwards past that entry automatically.
    # Persistent keys are pre-seeded into ``_key_to_slot`` by ``plan()``.

    def _pool_allocate(
        self, key: AllocatorKey, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Allocate from the pre-built schedule.

        Walks ``_seq_ops`` starting at ``_seq`` looking for the planned
        alloc entry for ``key``.  Planned entries (alloc or free) whose
        target key is already in ``_key_to_slot`` are fast-forwarded past
        as cached remnants from a prior micro-batch.
        Returns the cached slot directly if the key is already active.
        """
        while True:
            op = self._seq_ops.get(self._seq)
            if op is not None and op[0] == "alloc" and op[1] == key:
                slot_idx = op[2]
                slot = self._slots[slot_idx]
                assert size <= slot.size, (
                    f"requested {size} > slot capacity {slot.size} (key={key})"
                )
                slot.in_use = True
                self._key_to_slot[key] = slot_idx
                self._seq += 1
                pool = self._pools[(slot.dtype, slot.device)]
                return Bucket(data=pool[slot.offset : slot.offset + size])

            if op is not None and op[1] in self._key_to_slot:
                # planned key is cached — skip past its alloc entry
                self._seq += 1
                continue

            if key in self._key_to_slot:
                slot_idx = self._key_to_slot[key]
                slot = self._slots[slot_idx]
                pool = self._pools[(slot.dtype, slot.device)]
                return Bucket(data=pool[slot.offset : slot.offset + size])

            raise RuntimeError(
                f"unexpected alloc at seq={self._seq} for key={key}; "
                f"planned={op}"
            )

    def _pool_free(self, key: AllocatorKey) -> None:
        """Free the slot scheduled at the current sequence position.

        Walks ``_seq_ops`` the same way ``_pool_allocate`` does, matching
        the planned free entry for ``key`` and fast-forwarding past any
        alloc or free entries whose key is cached.  Double-free and
        free-before-alloc are silent no-ops.
        """
        while True:
            op = self._seq_ops.get(self._seq)
            if op is not None and op[0] == "free" and op[1] == key:
                slot_idx = self._key_to_slot.pop(key, None)
                if slot_idx is not None:
                    self._slots[slot_idx].in_use = False
                self._seq += 1
                return

            if op is not None and op[1] in self._key_to_slot:
                # planned key is cached — skip past its free entry
                self._seq += 1
                continue

            if key not in self._key_to_slot:
                return  # double-free or never allocated → no-op

            raise RuntimeError(
                f"unexpected free at seq={self._seq} for key={key}; "
                f"planned={op}"
            )

    # -- Flexible-mode allocate / free ---------------------------------- #
    #
    # When enabled, allocate and free skip the seq-driven replay entirely.
    # Each key maps directly to its first planned slot (built once by
    # ``enable_flexible_mode``).  An overlap check catches keys whose
    # slots are occupied by a different live key.

    def enable_flexible_mode(self) -> None:
        """Enable flexible allocate/free that does not require replaying
        the trace sequence.

        After this call, every ``allocate(key)`` returns the first slot
        associated with ``key`` in the plan, regardless of ``_seq``.
        An overlap error is raised if the target slot is already in use
        by a different key.
        """
        assert self._phase == "optimized", "flexible mode requires an existing plan"
        self._flexible = True
        self._flex_key_to_slot.clear()
        for op in self._seq_ops.values():
            if op[0] == "alloc" and op[1] not in self._flex_key_to_slot:
                self._flex_key_to_slot[op[1]] = op[2]

    def disable_flexible_mode(self) -> None:
        """Disable flexible mode and return to seq-driven replay."""
        self._flexible = False

    def _flex_allocate(
        self, key: AllocatorKey, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Flexible allocate — key → first planned slot."""
        slot_idx = self._flex_key_to_slot[key]
        slot = self._slots[slot_idx]
        assert size <= slot.size, (
            f"requested {size} > slot capacity {slot.size} (key={key})"
        )

        if slot.in_use and key not in self._key_to_slot:
            raise RuntimeError(
                f"flexible alloc overlap: slot[{slot_idx}] is in use "
                f"but {key} does not own it"
            )

        if key in self._key_to_slot:
            pool = self._pools[(slot.dtype, slot.device)]
            return Bucket(data=pool[slot.offset : slot.offset + size])

        slot.in_use = True
        self._key_to_slot[key] = slot_idx
        pool = self._pools[(slot.dtype, slot.device)]
        return Bucket(data=pool[slot.offset : slot.offset + size])

    def _flex_free(self, key: AllocatorKey) -> None:
        """Flexible free — release the slot associated with the key."""
        slot_idx = self._key_to_slot.pop(key, None)
        if slot_idx is not None:
            self._slots[slot_idx].in_use = False

    # -- Debug ---------------------------------------------------------- #

    def dump_trace(self) -> str:
        """Return a human-readable dump of the trace and pool plan."""
        lines = []
        lines.append(f"=== TracePoolAllocator (phase={self._phase}, seq={self._seq}) ===")
        lines.append(f"trace events: {len(self._trace)}")
        for ev in self._trace:
            meta = self._trace_meta.get(ev.key)
            size_str = f"size={meta[0]}" if meta else "size=?"
            dtype_str = f"dtype={meta[1]}" if meta else "dtype=?"
            device_str = f"device={meta[2]}" if meta else "device=?"
            lines.append(
                f"  seq={ev.seq:>4}  {ev.op:>5}  key={ev.key}  "
                f"{size_str}  {dtype_str}  {device_str}"
            )

        if self._phase == "optimized":
            lines.append(f"\nslots: {len(self._slots)}")
            for i, slot in enumerate(self._slots):
                lines.append(
                    f"  slot[{i}]: offset={slot.offset} size={slot.size} "
                    f"dtype={slot.dtype} device={slot.device} {'in_use' if slot.in_use else 'free'}"
                )
            lines.append(f"\nseq_ops ({len(self._seq_ops)} entries):")
            for seq in sorted(self._seq_ops.keys()):
                op_type, key, slot_idx = self._seq_ops[seq]
                lines.append(f"  seq={seq:>4}  {op_type:>5}  key={key}  slot={slot_idx}")
            lines.append(f"\nkey_to_slot (active):")
            for key, slot_idx in self._key_to_slot.items():
                lines.append(f"  {key} -> slot[{slot_idx}]")

        return "\n".join(lines)

    # -- Lifecycle ------------------------------------------------------- #

    def reset_cursor(self) -> None:
        """Reset sequence counter for the next micro-batch.

        ``_key_to_slot`` is NOT cleared so that keys cached across
        micro-batches survive and automatically fast-forward past
        their planned entries on the next replay.
        """
        for slot in self._slots:
            slot.in_use = False
        self._seq = 0

    # -- CUDA graph slot snapshot / restore ---------------------------- #

    def snapshot_slots(self) -> None:
        """Freeze current ``slot.in_use`` state for replay restoration."""
        self._captured_in_use = [s.in_use for s in self._slots]

    def restore_slots(self) -> None:
        """Restore ``slot.in_use`` to the capture-time snapshot."""
        if not self._captured_in_use:
            return
        for i, in_use in enumerate(self._captured_in_use):
            self._slots[i].in_use = in_use

    def reset(self) -> None:
        """Reset to trace phase, discarding the pool and all recorded state."""
        self._phase = "trace"
        self._seq = 0
        self._trace.clear()
        self._trace_meta.clear()
        self._buckets.clear()
        self._active_keys.clear()
        self._pools.clear()
        self._seq_ops.clear()
        self._key_to_slot.clear()
        self._slots.clear()
        self._flexible = False
        self._flex_key_to_slot.clear()
        self._captured_in_use.clear()

    @property
    def phase(self) -> str:
        """Current allocator phase: ``"trace"`` or ``"optimized"``."""
        return self._phase

    @property
    def total_pool_bytes(self) -> int:
        """Total pool size in bytes across all dtype/device groups."""
        total = 0
        for (dtype, _), pool in self._pools.items():
            total += pool.numel() * pool.element_size()
        return total


def _free_storage(tensor: torch.Tensor) -> None:
    """Free the underlying storage of ``tensor`` by resizing it to 0."""
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                assert tensor.storage_offset() == 0, (
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}"
                )
                tensor._typed_storage()._resize_(0)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """Re-allocate storage for ``tensor`` to the given ``size``.

    Requires that the tensor's storage has been freed (resized to 0)
    before calling.  The caller must ensure ``size`` matches the tensor's
    existing shape.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                assert tensor_storage_size == 0, (
                    "Tensor storage should have been resized to 0 but got "
                    f"{tensor_storage_size} (shape={tensor.shape})"
                )
                tensor._typed_storage()._resize_(size.numel())
