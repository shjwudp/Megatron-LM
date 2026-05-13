import dataclasses
from typing import Dict, List, Optional, Tuple

import torch

from .utils import ParamGroupIdx


@dataclasses.dataclass
class Bucket:
    data: torch.Tensor


class BucketAllocator:
    """Interface for allocating and freeing temporary buckets."""

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Allocate a bucket for the given param group."""
        raise NotImplementedError

    def free(self, param_group_id: ParamGroupIdx) -> None:
        """Free the bucket associated with the given param group."""
        raise NotImplementedError


class TemporaryBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by param_group_id.

    Used by DataParallelBuffer for unshard (all-gather) and gradient
    reduction (reduce-scatter) operations.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self.buckets[param_group_id]

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if param_group_id in self.buckets:
            _free_storage(self.buckets[param_group_id].data)
            del self.buckets[param_group_id]


class StorageFreeingBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by param_group_id, and frees the
    underlying storage after use without deleting the bucket entry, so the
    same tensor object can be reused on the next allocation.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
            return self.buckets[param_group_id]
        _alloc_storage(self.buckets[param_group_id].data, torch.Size([size]))
        return self.buckets[param_group_id]

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if param_group_id in self.buckets:
            _free_storage(self.buckets[param_group_id].data)


class TracePoolAllocator(BucketAllocator):
    """Two-phase memory-pool bucket allocator.

    **Phase 1 — Trace** (``plan()`` not yet called):
    Behaves like ``TemporaryBucketAllocator`` while recording every
    ``allocate`` / ``free`` event with a monotonic sequence number and
    the associated ``(param_group_id, size, dtype, device)`` tuple.

    **Phase 2 — Optimized** (after ``plan()``):
    Analyzes the recorded trace with a greedy interval-coloring algorithm
    to build a single flat pool tensor for each ``(dtype, device)`` group.
    Subsequent ``allocate`` calls return views into the pool; ``free``
    merely marks the slot as unused (no storage is released).

    The trace must be **repeatable** — the same sequence of allocate/free
    calls is expected across iterations / micro-batches.  ``plan()``
    guarantees that with the same call pattern no slot conflict occurs.
    """

    class _Slot:
        __slots__ = ("offset", "size", "dtype", "device", "in_use")

        def __init__(self, offset: int, size: int, dtype: torch.dtype, device: torch.device):
            self.offset = offset
            self.size = size
            self.dtype = dtype
            self.device = device
            self.in_use = False

    @dataclasses.dataclass
    class _TraceEvent:
        seq: int
        op: str  # "alloc" | "free"
        param_group_id: ParamGroupIdx

    @dataclasses.dataclass
    class _Interval:
        param_group_id: ParamGroupIdx
        size: int
        alloc_seq: int
        free_seq: int

    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "trace"
        self._seq: int = 0
        self._trace: List["TracePoolAllocator._TraceEvent"] = []
        self._trace_meta: Dict[ParamGroupIdx, Tuple[int, torch.dtype, torch.device]] = {}
        self._buckets: Dict[ParamGroupIdx, Bucket] = {}

        # Pool state (populated by plan())
        self._pools: Dict[Tuple[torch.dtype, torch.device], torch.Tensor] = {}
        self._slot_map: Dict[ParamGroupIdx, List[int]] = {}  # pg_id -> [slot indices]
        self._slot_cursors: Dict[ParamGroupIdx, int] = {}    # pg_id -> next index
        self._slots: List["TracePoolAllocator._Slot"] = []

    # -- Phase 1: trace -------------------------------------------------- #

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if self._phase != "optimized":
            return self._trace_allocate(param_group_id, size, dtype, device)
        return self._pool_allocate(param_group_id, size, dtype, device)

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if self._phase != "optimized":
            self._trace_free(param_group_id)
        else:
            self._pool_free(param_group_id)

    def _trace_allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self._buckets:
            self._trace.append(
                self._TraceEvent(seq=self._seq, op="alloc", param_group_id=param_group_id)
            )
            self._seq += 1
            self._trace_meta[param_group_id] = (size, dtype, device)
            self._buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self._buckets[param_group_id]

    def _trace_free(self, param_group_id: ParamGroupIdx) -> None:
        self._trace.append(
            self._TraceEvent(seq=self._seq, op="free", param_group_id=param_group_id)
        )
        self._seq += 1
        if param_group_id in self._buckets:
            _free_storage(self._buckets[param_group_id].data)
            del self._buckets[param_group_id]

    # -- Phase 2: plan --------------------------------------------------- #

    def plan(self) -> int:
        """Analyze the trace and build the static memory pool.

        Returns:
            Total pool size in **elements** (sum across all dtype/device
            groups).  Multiply by ``element_size(dtype)`` for bytes.
        """
        assert self._phase == "trace", "plan() can only be called in trace phase"
        assert len(self._trace) > 0, "empty trace — nothing to plan"

        # Replay the trace to build alloc/free intervals.
        alloc_stack: Dict[ParamGroupIdx, List[int]] = {}
        intervals: List["TracePoolAllocator._Interval"] = []

        for ev in self._trace:
            pg_id = ev.param_group_id
            if ev.op == "alloc":
                alloc_stack.setdefault(pg_id, []).append(ev.seq)
            else:  # "free"
                if pg_id in alloc_stack and alloc_stack[pg_id]:
                    alloc_seq = alloc_stack[pg_id].pop(0)
                    meta = self._trace_meta.get(pg_id)
                    if meta is not None:
                        size, dtype, device = meta
                        intervals.append(
                            self._Interval(
                                param_group_id=pg_id,
                                size=size,
                                alloc_seq=alloc_seq,
                                free_seq=ev.seq,
                            )
                        )

        assert len(intervals) > 0, "no paired alloc/free intervals found in trace"
        return self._assign_pool(intervals)

    def _assign_pool(self, intervals: List["TracePoolAllocator._Interval"]) -> int:
        """Greedy left-edge interval coloring + slot layout."""
        groups: Dict[Tuple[torch.dtype, torch.device], List["TracePoolAllocator._Interval"]] = {}
        for iv in intervals:
            meta = self._trace_meta[iv.param_group_id]
            key = (meta[1], meta[2])  # (dtype, device)
            groups.setdefault(key, []).append(iv)

        self._slot_map.clear()
        self._slot_cursors.clear()
        self._slots.clear()
        self._pools.clear()

        total_elems = 0
        for (dtype, device), group in groups.items():
            total_elems += self._color_group(group, dtype, device)

        self._phase = "optimized"
        return total_elems

    def _color_group(
        self,
        intervals: List["TracePoolAllocator._Interval"],
        dtype: torch.dtype,
        device: torch.device,
    ) -> int:
        """Greedy interval coloring for one (dtype, device) group.

        Sorts intervals by ``alloc_seq`` then assigns each to the first
        free slot via a linear scan of the free-list.  The scan is O(n²)
        worst-case, but FSDP workloads typically have tens to low hundreds
        of param groups per dtype/device so this is negligible.
        """
        intervals = sorted(intervals, key=lambda iv: iv.alloc_seq)

        # (local_slot_index, free_seq)
        free_slots: List[Tuple[int, int]] = []
        group_slots: List["TracePoolAllocator._Slot"] = []
        local_to_global: Dict[int, int] = {}

        for iv in intervals:
            assigned = False
            for i, (slot_idx, slot_free_seq) in enumerate(free_slots):
                if slot_free_seq < iv.alloc_seq:
                    slot = group_slots[slot_idx]
                    if iv.size > slot.size:
                        slot.size = iv.size
                    free_slots[i] = (slot_idx, iv.free_seq)
                    self._slot_map.setdefault(iv.param_group_id, []).append(
                        local_to_global[slot_idx]
                    )
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
                self._slot_map.setdefault(iv.param_group_id, []).append(global_idx)

        offset = 0
        for slot in group_slots:
            slot.offset = offset
            offset += slot.size

        if offset > 0:
            self._pools[(dtype, device)] = torch.empty(offset, dtype=dtype, device=device)
        return offset

    # -- Phase 2 runtime ------------------------------------------------- #

    def _pool_allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        slot_list = self._slot_map[param_group_id]
        cursor = self._slot_cursors.get(param_group_id, 0)
        assert cursor < len(slot_list), (
            f"no slot available for pg={param_group_id} "
            f"(cursor={cursor}, slots={slot_list})"
        )
        slot_idx = slot_list[cursor]
        self._slot_cursors[param_group_id] = cursor + 1

        slot = self._slots[slot_idx]
        assert not slot.in_use, (
            f"slot {slot_idx} already in use (pg={param_group_id}, "
            f"seq={self._seq}, cursor={cursor}, slot_list={slot_list})"
        )
        assert size <= slot.size, (
            f"requested {size} > slot capacity {slot.size} (pg={param_group_id})"
        )
        pool = self._pools[(slot.dtype, slot.device)]
        slot.in_use = True
        self._seq += 1
        return Bucket(data=pool[slot.offset : slot.offset + size])

    def _pool_free(self, param_group_id: ParamGroupIdx) -> None:
        slot_idx = self._slot_map[param_group_id][
            self._slot_cursors.get(param_group_id, 1) - 1
        ]
        slot = self._slots[slot_idx]
        assert slot.in_use, (
            f"slot {slot_idx} already free (pg={param_group_id}, seq={self._seq})"
        )
        slot.in_use = False
        self._seq += 1

    # -- Lifecycle ------------------------------------------------------- #

    def dump_trace(self) -> str:
        """Return a human-readable dump of the recorded trace and pool plan.

        Useful for debugging slot-conflict errors.
        """
        lines = []
        lines.append(f"=== TracePoolAllocator trace (phase={self._phase}, seq={self._seq}) ===")
        lines.append(f"trace events: {len(self._trace)}")
        for ev in self._trace:
            meta = self._trace_meta.get(ev.param_group_id)
            size_str = f"size={meta[0]}" if meta else "size=?"
            dtype_str = f"dtype={meta[1]}" if meta else "dtype=?"
            device_str = f"device={meta[2]}" if meta else "device=?"
            lines.append(
                f"  seq={ev.seq:>4}  {ev.op:>5}  pg={ev.param_group_id}  "
                f"{size_str}  {dtype_str}  {device_str}"
            )

        if self._phase == "optimized":
            lines.append(f"\nslots: {len(self._slots)}")
            for i, slot in enumerate(self._slots):
                lines.append(
                    f"  slot[{i}]: offset={slot.offset} size={slot.size} "
                    f"dtype={slot.dtype} device={slot.device}"
                )
            lines.append(f"\nslot_map (pg_id -> [slot indices]):")
            for pg_id, slot_list in self._slot_map.items():
                cursor = self._slot_cursors.get(pg_id, 0)
                lines.append(f"  {pg_id} -> {slot_list}  cursor={cursor}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset to trace phase, discarding the pool and trace."""
        self._phase = "trace"
        self._seq = 0
        self._trace.clear()
        self._trace_meta.clear()
        self._buckets.clear()
        self._pools.clear()
        self._slot_map.clear()
        self._slot_cursors.clear()
        self._slots.clear()

    @property
    def phase(self) -> str:
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
