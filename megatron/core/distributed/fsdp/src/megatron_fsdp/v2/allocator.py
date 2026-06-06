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
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
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
            self.buckets[key] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self.buckets[key]

    def free(
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
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
            self.buckets[key] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
            return self.buckets[key]
        _alloc_storage(self.buckets[key].data, torch.Size([size]))
        return self.buckets[key]

    def free(
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> None:
        key = _resolve_key(key, param_group_id)
        if key in self.buckets:
            _free_storage(self.buckets[key].data)


class TracePoolAllocator(BucketAllocator):
    """Two-phase bucket allocator with a static key→slot plan for CUDA graph compatibility.

    **Design**

    The FSDP framework allocates and frees temporary flat buffers (for
    all-gather input/output and gradient accumulation) in a deterministic
    but not necessarily position-invariant order across micro-batches.
    ``TracePoolAllocator`` profiles one pass and then serves all subsequent
    passes from a pre-allocated pool via a static key→slot map.

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
    left-edge algorithm that enforces **same-key → same-slot**:

    1. Sort intervals by ``alloc_seq``.
    2. For each interval, if the key was already assigned a slot, force reuse
       of that slot (the slot must be free — overlapping intervals for the
       same key would be a programming error).
    3. Otherwise, try to reuse a *slot* whose previous occupant freed before
       this interval starts (``slot_free_seq < alloc_seq``).
    4. If no slot is free, allocate a new one.
    5. Grow the slot's capacity to ``max(size, current)``.

    After coloring, slots are laid out contiguously with device-/dtype-aware
    alignment and a single ``torch.empty`` per ``(dtype, device)`` group is
    allocated.  The resulting ``_key_to_slot`` dict maps every allocation
    key to exactly one fixed slot index — no seq-driven schedule is built.

    **Phase 3 — Optimized** (after ``plan()``)

    ``allocate`` and ``free`` use a simple dict lookup on ``_key_to_slot``
    to return a pool-tensor view.  Because the pool tensors are allocated
    once and never resized, the same key always resolves to the same memory
    address — essential for CUDA graph compatibility.  ``reset_batch()``
    clears slot ``in_use`` flags between micro-batches.

    **Hook-coordinated lifecycle** (see ``megatron_fsdp.v2.hooks``)::

        Micro-batch 0
        ┌──────────────────────────────────────────────────────────
        │  root pre-forward     forward_phase = True
        │    forward (trace)
        │  root pre-backward    forward_phase = False , backward_phase = True
        │    backward (trace)
        │  root post-backward   backward_phase = False
        │                       plan() → optimized
        └──────────────────────────────────────────────────────────

        Micro-batch 1+
        ┌──────────────────────────────────────────────────────────
        │  root pre-forward     forward_phase = True
        │                       reset_batch()
        │    forward (optimized, key→slot lookup)
        │  root pre-backward    forward_phase = False , backward_phase = True
        │    backward (optimized, key→slot lookup)
        │  root post-backward   backward_phase = False
        └──────────────────────────────────────────────────────────
    """

    # -- Inner types ---------------------------------------------------- #

    class _Slot:
        """A contiguous slice of the pool tensor assigned to one or more
        non-overlapping intervals."""

        __slots__ = ("offset", "size", "dtype", "device", "in_use")

        def __init__(
            self, offset: int, size: int, dtype: torch.dtype, device: torch.device
        ):
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
        self._phase: str = "trace"  # "trace" | "optimized"

        # Trace state
        self._seq: int = 0  # monotonic alloc/free counter (trace phase only)
        self._trace: List["TracePoolAllocator._TraceEvent"] = []
        self._trace_meta: Dict[AllocatorKey, Tuple[int, torch.dtype, torch.device]] = {}
        self._buckets: Dict[AllocatorKey, Bucket] = {}  # only used in trace phase
        self._active_keys: Set[AllocatorKey] = set()  # keys currently allocated

        # Pool state — populated by plan(), used in optimized phase
        self._pools: Dict[Tuple[torch.dtype, torch.device], torch.Tensor] = {}
        self._slots: List["TracePoolAllocator._Slot"] = []
        # Static key → slot mapping (immutable once planned)
        self._key_to_slot: Dict[AllocatorKey, int] = {}

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
        """Dispatch to trace or optimized path depending on phase."""
        key = _resolve_key(key, param_group_id)
        assert dtype is not None and device is not None
        if self._phase != "optimized":
            return self._trace_allocate(key, size, dtype, device)
        return self._optimized_allocate(key, size, dtype, device)

    def free(
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> None:
        """Dispatch to trace or optimized path depending on phase."""
        key = _resolve_key(key, param_group_id)
        if self._phase != "optimized":
            self._trace_free(key)
        else:
            self._optimized_free(key)

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
            self._buckets[key] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        else:
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
        """Build the static key→slot plan from the recorded trace.

        1. Replay the trace to pair alloc/free events into ``_Interval`` objects.
        2. Pair any un-freed allocs with sentinel free sequences so
           persistent keys get coloured and reserved slots.
        3. Group intervals by ``(dtype, device)``.
        4. Color each group with the greedy left-edge algorithm (with
           same-key→same-slot enforcement and alignment).
        5. Allocate one flat pool tensor per group.
        6. ``_key_to_slot`` is populated directly by ``_color_group``.

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
                                key=ev.key,
                                size=size,
                                alloc_seq=alloc_seq,
                                free_seq=ev.seq,
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
                            key=key,
                            size=size,
                            alloc_seq=alloc_seq,
                            free_seq=sentinel_seq,
                        )
                    )
                    sentinel_seq += 1

        if len(intervals) == 0:
            self._phase = "optimized"
            return 0

        # ---- step 2 & 3: color and allocate ----
        total_elems = self._assign_pool(intervals)

        self._phase = "optimized"
        return total_elems

    def _assign_pool(self, intervals: List["TracePoolAllocator._Interval"]) -> int:
        """Group intervals by (dtype, device), color each group, sum sizes."""
        groups: Dict[
            Tuple[torch.dtype, torch.device], List["TracePoolAllocator._Interval"]
        ] = {}
        for iv in intervals:
            meta = self._trace_meta[iv.key]
            dtype_device = (meta[1], meta[2])
            groups.setdefault(dtype_device, []).append(iv)

        self._slots.clear()
        self._pools.clear()
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

        Enforces same-key→same-slot: if a key already has an assigned slot
        from an earlier interval, all subsequent intervals of that key must
        reuse the same slot.  The slot must be free at the interval's start
        (overlapping intervals for the same key would be a programming error).
        """
        sorted_intervals = sorted(intervals, key=lambda iv: iv.alloc_seq)

        free_slots: List[Tuple[int, int]] = []  # (local_slot_index, free_seq)
        group_slots: List["TracePoolAllocator._Slot"] = []
        local_to_global: Dict[int, int] = {}

        # Map key → global slot index for same-key enforcement
        key_assigned_global: Dict[AllocatorKey, int] = {}
        # Map key → local slot index for fast slot lookup
        global_to_local: Dict[int, int] = {}

        for iv in sorted_intervals:
            # --- same-key constraint: force reuse of the key's assigned slot ---
            prev_global = key_assigned_global.get(iv.key)
            if prev_global is not None:
                local_idx = global_to_local[prev_global]
                slot = group_slots[local_idx]
                # Verify the slot is free (non-overlapping same-key intervals)
                for _, (sl, sf) in enumerate(free_slots):
                    if sl == local_idx:
                        assert sf < iv.alloc_seq, (
                            f"key {iv.key!r} has overlapping intervals — this should be "
                            f"impossible in FSDP (same key never alloc'd twice without free)"
                        )
                        break
                if iv.size > slot.size:
                    slot.size = iv.size
                # Update this slot's free time
                for i, (sl, _) in enumerate(free_slots):
                    if sl == local_idx:
                        free_slots[i] = (local_idx, iv.free_seq)
                        break
                continue

            # --- normal left-edge: reuse an existing free slot, or create new ---
            assigned_local = None
            for i, (local_idx, slot_free_seq) in enumerate(free_slots):
                if slot_free_seq < iv.alloc_seq:
                    slot = group_slots[local_idx]
                    if iv.size > slot.size:
                        slot.size = iv.size
                    free_slots[i] = (local_idx, iv.free_seq)
                    assigned_local = local_idx
                    break

            if assigned_local is None:
                assigned_local = len(group_slots)
                global_idx = len(self._slots)
                local_to_global[assigned_local] = global_idx
                global_to_local[global_idx] = assigned_local
                slot = self._Slot(offset=0, size=iv.size, dtype=dtype, device=device)
                group_slots.append(slot)
                self._slots.append(slot)
                free_slots.append((assigned_local, iv.free_seq))
            else:
                global_idx = local_to_global[assigned_local]

            key_assigned_global[iv.key] = global_idx

        # Lay out slots contiguously with alignment
        offset = 0
        alignment = self._get_alignment(device, dtype)
        for slot in group_slots:
            offset = (offset + alignment - 1) // alignment * alignment
            slot.offset = offset
            offset += slot.size

        if offset > 0:
            self._pools[(dtype, device)] = torch.empty(
                offset, dtype=dtype, device=device
            )

        # Populate the static key→slot map
        for key, global_idx in key_assigned_global.items():
            self._key_to_slot[key] = global_idx

        return offset

    @staticmethod
    def _get_alignment(device: torch.device, dtype: torch.dtype) -> int:
        """Return the minimum alignment (in elements) for the given device/dtype.

        Aligns to at least the element size and, on CUDA, to the device's
        texture alignment.  Critical for NVFP4 sub-byte types and CUDA
        kernel alignment requirements.
        """
        element_bytes = torch.empty(0, dtype=dtype, device=device).element_size()
        if device.type == "cuda":
            try:
                texture_alignment = torch.cuda.get_device_properties(
                    device
                ).texture_alignment
                align_bytes = max(element_bytes, texture_alignment)
            except Exception:
                align_bytes = element_bytes
        else:
            align_bytes = element_bytes
        return max(1, align_bytes // element_bytes)

    # -- Phase 3: optimized runtime ------------------------------------- #
    #
    # ``_key_to_slot`` maps each allocation key to a fixed slot index.
    # Runtime dispatch is a dict lookup — no seq counter, no schedule
    # walk.  The same key always resolves to the same pool-tensor view.

    def _optimized_allocate(
        self, key: AllocatorKey, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Allocate from the static key→slot plan.

        Returns the pool-tensor view at the key's fixed address.  Raises
        ``KeyError`` if ``key`` was never seen during trace.
        """
        slot_idx = self._key_to_slot[key]
        slot = self._slots[slot_idx]
        if slot.in_use and key not in self._active_keys:
            raise RuntimeError(
                f"Slot collision at slot[{slot_idx}]: key={key!r} "
                f"but slot is held by active key(s)"
            )
        assert size <= slot.size, (
            f"requested {size} > slot capacity {slot.size} (key={key!r})"
        )
        if key in self._active_keys:
            # Re-entrant: key already allocated this micro-batch — idempotent
            pool = self._pools[(slot.dtype, slot.device)]
            return Bucket(data=pool[slot.offset : slot.offset + size])
        slot.in_use = True
        self._active_keys.add(key)
        pool = self._pools[(slot.dtype, slot.device)]
        return Bucket(data=pool[slot.offset : slot.offset + size])

    def _optimized_free(self, key: AllocatorKey) -> None:
        """Free the slot associated with the key — idempotent."""
        if key not in self._active_keys:
            return  # double-free or never-allocated → silent no-op
        self._slots[self._key_to_slot[key]].in_use = False
        self._active_keys.discard(key)

    # -- Debug ---------------------------------------------------------- #

    def dump_trace(self) -> str:
        """Return a human-readable dump of the trace and pool plan."""
        lines = []
        lines.append(f"=== TracePoolAllocator (phase={self._phase}) ===")
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
                    f"dtype={slot.dtype} device={slot.device} "
                    f"{'in_use' if slot.in_use else 'free'}"
                )
            total_bytes = sum(
                s.size * torch.empty(0, dtype=s.dtype).element_size()
                for s in self._slots
            )
            lines.append(f"\ntotal pool: {len(self._slots)} slots, {total_bytes} bytes")
            lines.append(f"\nkey_to_slot ({len(self._key_to_slot)} entries):")
            for key, slot_idx in sorted(
                self._key_to_slot.items(), key=lambda x: str(x[0])
            ):
                slot = self._slots[slot_idx]
                pool = self._pools.get((slot.dtype, slot.device))
                addr_str = ""
                if pool is not None:
                    addr_str = f" address=0x{pool[slot.offset].data_ptr():x}"
                lines.append(
                    f"  {key!r} -> slot[{slot_idx}] "
                    f"(offset={slot.offset}, size={slot.size}, "
                    f"dtype={slot.dtype}{addr_str})"
                )
            lines.append(f"\nactive_keys ({len(self._active_keys)}):")
            for key in self._active_keys:
                lines.append(f"  {key!r}")

        return "\n".join(lines)

    # -- Lifecycle ------------------------------------------------------- #

    def reset_batch(self) -> None:
        """Reset slot state for the next micro-batch.

        Clears ``in_use`` flags on all slots and resets the active-key
        set.  Does NOT discard ``_key_to_slot`` or ``_pools`` — the
        slot→address mapping is immutable once planned.

        Called at root pre-forward of each micro-batch after ``plan()``.
        """
        assert self._phase == "optimized", "reset_batch requires an existing plan"
        for slot in self._slots:
            slot.in_use = False
        self._active_keys.clear()

    def reset(self) -> None:
        """Full teardown: discard pool, plan, and trace; return to "trace" phase.

        Used for model re-initialization or full training restart.
        """
        self._phase = "trace"
        self._seq = 0
        self._trace.clear()
        self._trace_meta.clear()
        self._buckets.clear()
        self._active_keys.clear()
        self._pools.clear()
        self._key_to_slot.clear()
        self._slots.clear()

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
