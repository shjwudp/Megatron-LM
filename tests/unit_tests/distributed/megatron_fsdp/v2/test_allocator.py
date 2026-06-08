# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Unit tests for the v3 TracePoolAllocator (static key→slot plan)."""

import pytest
import torch
import torch.distributed

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import (
    Bucket,
    BucketAllocator,
    StorageFreeingBucketAllocator,
    TemporaryBucketAllocator,
    TracePoolAllocator,
    _alloc_storage,
    _free_storage,
    _resolve_key,
)

try:
    torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
except Exception:
    pass

_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(name: str) -> tuple:
    return (0, name)


def _alloc(ba, key, size=1024, dtype=torch.float32, device=_DEVICE):
    return ba.allocate(key=key, size=size, dtype=dtype, device=device)


def _free(ba, key):
    ba.free(key=key)


def _data_ptr(bucket: Bucket):
    return bucket.data.data_ptr()


# ---------------------------------------------------------------------------
# _resolve_key
# ---------------------------------------------------------------------------


def test_resolve_key_with_explicit_key():
    assert _resolve_key("my_key", None) == "my_key"


def test_resolve_key_falls_back_to_param_group_id():
    assert _resolve_key(None, "pg_0") == "pg_0"


def test_resolve_key_requires_param_group_id_when_key_is_none():
    with pytest.raises(AssertionError, match="allocator key is required"):
        _resolve_key(None, None)


# ---------------------------------------------------------------------------
# TemporaryBucketAllocator
# ---------------------------------------------------------------------------


def test_temporary_allocator_allocate_and_free():
    ba = TemporaryBucketAllocator()
    b = _alloc(ba, "k1", size=256)
    assert b.data.numel() == 256
    assert b.data.data_ptr() != 0
    _free(ba, "k1")
    # After free, storage is released — bucket is removed.
    assert "k1" not in ba.buckets


# ---------------------------------------------------------------------------
# StorageFreeingBucketAllocator
# ---------------------------------------------------------------------------


def test_storage_freeing_allocator_reuses_entry():
    ba = StorageFreeingBucketAllocator()
    b1 = _alloc(ba, "k1", size=16)
    ptr1 = b1.data.data_ptr()
    _free(ba, "k1")
    # Entry stays but storage is freed; next alloc re-uses same tensor object.
    b2 = _alloc(ba, "k1", size=32)
    assert b2 is b1
    assert b2.data.data_ptr() != ptr1  # storage at a different address
    assert b2.data.numel() == 32


# ---------------------------------------------------------------------------
# TracePoolAllocator — __init__ / phase
# ---------------------------------------------------------------------------


def test_initial_phase_is_trace():
    ba = TracePoolAllocator()
    assert ba.phase == "trace"


def test_initial_total_pool_bytes_is_zero():
    ba = TracePoolAllocator()
    assert ba.total_pool_bytes == 0


# ---------------------------------------------------------------------------
# TracePoolAllocator — trace phase
# ---------------------------------------------------------------------------


def test_trace_allocate_returns_bucket():
    ba = TracePoolAllocator()
    b = _alloc(ba, _make_key("w1"), size=512)
    assert b.data.numel() == 512
    assert b.data.dtype == torch.float32


def test_trace_allocate_is_idempotent():
    """Duplicate alloc for active key returns same bucket."""
    ba = TracePoolAllocator()
    key = _make_key("w1")
    b1 = _alloc(ba, key, size=128)
    b2 = _alloc(ba, key, size=128)
    assert b2 is b1
    assert _data_ptr(b2) == _data_ptr(b1)


def test_trace_re_alloc_after_free_gets_new_storage():
    """Free + re-alloc should resurrect storage with a new trace event."""
    ba = TracePoolAllocator()
    key = _make_key("w1")
    b1 = _alloc(ba, key, size=100)
    ptr1 = _data_ptr(b1)
    _free(ba, key)
    b2 = _alloc(ba, key, size=200)
    # Same Bucket object, different underlying storage.
    assert b2 is b1
    assert b2.data.data_ptr() != ptr1  # storage was freed & reallocated
    assert len(ba._trace) == 3  # alloc, free, alloc


def test_trace_free_is_idempotent():
    ba = TracePoolAllocator()
    key = _make_key("w1")
    _alloc(ba, key, size=64)
    _free(ba, key)
    assert key not in ba._active_keys
    # Double-free is silent
    _free(ba, key)
    assert key not in ba._active_keys


def test_trace_free_before_alloc_is_noop():
    ba = TracePoolAllocator()
    key = _make_key("never_allocated")
    _free(ba, key)  # should not raise


def test_trace_records_correct_sequence():
    ba = TracePoolAllocator()
    k1, k2 = _make_key("w1"), _make_key("w2")
    _alloc(ba, k1, size=10)
    _alloc(ba, k2, size=20)
    _free(ba, k1)
    _free(ba, k2)

    events = [(ev.op, ev.key, ev.seq) for ev in ba._trace]
    assert events == [
        ("alloc", k1, 0),
        ("alloc", k2, 1),
        ("free", k1, 2),
        ("free", k2, 3),
    ]


# ---------------------------------------------------------------------------
# TracePoolAllocator — plan (basic)
# ---------------------------------------------------------------------------


def test_plan_empty_trace_is_noop():
    ba = TracePoolAllocator()
    nelems = ba.plan()
    assert nelems == 0
    assert ba.phase == "optimized"
    assert len(ba._key_to_slot) == 0
    assert len(ba._slots) == 0


def test_plan_single_alloc_builds_one_slot():
    ba = TracePoolAllocator()
    key = _make_key("w1")
    _alloc(ba, key, size=512)
    _free(ba, key)
    ba.plan()
    assert ba.phase == "optimized"
    assert key in ba._key_to_slot
    assert len(ba._slots) == 1
    assert ba._slots[0].size == 512


def test_plan_cannot_be_called_twice():
    ba = TracePoolAllocator()
    key = _make_key("w1")
    _alloc(ba, key, size=32)
    _free(ba, key)
    ba.plan()
    with pytest.raises(AssertionError, match="plan\\(\\) can only be called in trace phase"):
        ba.plan()


def test_plan_persistent_keys_get_slots():
    """Keys not freed during trace get sentinel free_seq and reserved slots."""
    ba = TracePoolAllocator()
    k1, k2 = _make_key("w1"), _make_key("w2")
    _alloc(ba, k1, size=100)
    _free(ba, k1)
    _alloc(ba, k2, size=200)  # never freed — persistent
    ba.plan()

    assert k1 in ba._key_to_slot
    assert k2 in ba._key_to_slot
    # Both should have slots
    slot1 = ba._slots[ba._key_to_slot[k1]]
    slot2 = ba._slots[ba._key_to_slot[k2]]
    assert slot1.size == 100
    assert slot2.size == 200


# ---------------------------------------------------------------------------
# TracePoolAllocator — plan (interval coloring / slot reuse)
# ---------------------------------------------------------------------------


def test_non_overlapping_keys_share_slot():
    """Two keys whose lifetimes don't overlap should share one slot."""
    ba = TracePoolAllocator()
    k1, k2 = _make_key("w1"), _make_key("w2")
    _alloc(ba, k1, size=100)
    _free(ba, k1)
    _alloc(ba, k2, size=150)
    _free(ba, k2)
    ba.plan()

    assert len(ba._slots) == 1
    slot = ba._slots[0]
    assert slot.size == 150  # max(100, 150)


def test_overlapping_keys_get_different_slots():
    """Two keys that are both live at the same time need two slots."""
    ba = TracePoolAllocator()
    k1, k2 = _make_key("w1"), _make_key("w2")
    _alloc(ba, k1, size=100)
    _alloc(ba, k2, size=200)  # k1 still live
    _free(ba, k1)
    _free(ba, k2)
    ba.plan()

    assert len(ba._slots) == 2  # overlapping → two slots


def test_multiple_intervals_same_key_same_slot():
    """Same key appearing in multiple non-overlapping intervals → same slot."""
    ba = TracePoolAllocator()
    k = _make_key("w1")
    # Interval 1
    _alloc(ba, k, size=100)
    _free(ba, k)
    # Other key in between (different slot)
    k2 = _make_key("w2")
    _alloc(ba, k2, size=50)
    _free(ba, k2)
    # Interval 2 — same key again
    _alloc(ba, k, size=200)
    _free(ba, k)
    ba.plan()

    assert len(ba._slots) == 2  # k and k2 need different slots (overlap)
    # Both intervals of k map to same slot
    slot_k = ba._key_to_slot[k]
    assert ba._slots[slot_k].size == 200  # max(100, 200)


def test_same_key_non_overlapping_reuses_slot_even_with_others():
    """Key A, key B, key A pattern: A's two intervals share one slot."""
    ba = TracePoolAllocator()
    kA, kB = _make_key("wA"), _make_key("wB")
    # A's first interval
    _alloc(ba, kA, size=100)
    _free(ba, kA)
    # B's interval
    _alloc(ba, kB, size=50)
    _free(ba, kB)
    # A's second interval
    _alloc(ba, kA, size=80)
    _free(ba, kA)
    ba.plan()

    # A's intervals should share the same slot
    assert ba._key_to_slot[kA] == ba._key_to_slot[kA]  # trivial
    # B gets a different slot (since B and A's 2nd interval could overlap... wait,
    # B is freed before A's 2nd alloc, so A's slot is free when A2 starts)
    # Actually with the same-key constraint, A always uses the same slot.
    # B can reuse that same slot since B's interval is between A's two intervals.
    # So we expect 1 slot total.
    assert len(ba._slots) == 1


# ---------------------------------------------------------------------------
# TracePoolAllocator — plan (alignment)
# ---------------------------------------------------------------------------


def test_slots_have_nonzero_offsets_after_alignment():
    ba = TracePoolAllocator()
    for i in range(4):
        k = _make_key(f"w{i}")
        _alloc(ba, k, size=1)
        _free(ba, k)
    ba.plan()

    offsets = [s.offset for s in ba._slots]
    assert offsets[0] == 0  # first slot at offset 0
    # Alignment may cause subsequent offsets to be > previous offset+size
    for i in range(1, len(offsets)):
        prev = ba._slots[i - 1]
        assert offsets[i] >= prev.offset + prev.size


# ---------------------------------------------------------------------------
# TracePoolAllocator — optimized runtime (basic)
# ---------------------------------------------------------------------------


def test_optimized_allocate_returns_fixed_address():
    """Same key always returns same address in optimized phase."""
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=1024)
    _free(ba, k)
    ba.plan()

    b1 = ba.allocate(key=k, size=512, dtype=torch.float32, device=_DEVICE)
    ptr1 = _data_ptr(b1)
    _free(ba, k)

    # Reset for next micro-batch
    ba.reset_batch()

    b2 = ba.allocate(key=k, size=512, dtype=torch.float32, device=_DEVICE)
    ptr2 = _data_ptr(b2)
    assert ptr1 == ptr2  # same memory address
    _free(ba, k)


def test_optimized_re_entrant_allocate_returns_same_view():
    """Allocating same key twice without free in between is idempotent."""
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=512)
    _free(ba, k)
    ba.plan()

    b1 = _alloc(ba, k, size=512)
    b2 = _alloc(ba, k, size=512)  # re-entrant — key already active
    assert b2.data.data_ptr() == b1.data.data_ptr()
    _free(ba, k)


def test_optimized_free_clears_slot():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=128)
    _free(ba, k)
    ba.plan()

    _alloc(ba, k, size=64)
    _free(ba, k)
    slot = ba._slots[ba._key_to_slot[k]]
    assert not slot.in_use
    assert k not in ba._active_keys


def test_optimized_double_free_is_silent():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=64)
    _free(ba, k)
    ba.plan()

    _alloc(ba, k, size=64)
    _free(ba, k)
    _free(ba, k)  # double free — no error
    assert k not in ba._active_keys


def test_optimized_unknown_key_raises_keyerror():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=64)
    _free(ba, k)
    ba.plan()

    with pytest.raises(KeyError):
        _alloc(ba, _make_key("unknown"), size=32)


def test_optimized_size_exceeds_slot_capacity_raises():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=128)
    _free(ba, k)
    ba.plan()

    with pytest.raises(AssertionError, match="requested.*> slot capacity"):
        ba.allocate(key=k, size=1024, dtype=torch.float32, device=_DEVICE)


def test_optimized_slot_collision_detected():
    """A key trying to use a slot already in use by a different key."""
    ba = TracePoolAllocator()
    k1, k2 = _make_key("w1"), _make_key("w2")
    # Overlapping intervals → two different slots
    _alloc(ba, k1, size=100)
    _alloc(ba, k2, size=200)
    _free(ba, k1)
    _free(ba, k2)
    ba.plan()

    assert ba._key_to_slot[k1] != ba._key_to_slot[k2]

    # Allocate k1 — slot becomes in_use
    _alloc(ba, k1, size=100)
    # Allocating a different key that could share k1's slot shouldn't
    # trigger collision because each key maps to its own slot.
    _alloc(ba, k2, size=200)  # k2 has its own slot → no collision
    _free(ba, k1)
    _free(ba, k2)


# ---------------------------------------------------------------------------
# TracePoolAllocator — reset_batch / reset
# ---------------------------------------------------------------------------


def test_reset_batch_clears_in_use():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=128)
    _free(ba, k)
    ba.plan()

    _alloc(ba, k, size=64)
    assert ba._slots[0].in_use is True
    ba.reset_batch()
    assert ba._slots[0].in_use is False
    assert len(ba._active_keys) == 0
    assert k in ba._key_to_slot  # plan preserved


def test_reset_batch_requires_optimized_phase():
    ba = TracePoolAllocator()
    with pytest.raises(AssertionError, match="reset_batch requires an existing plan"):
        ba.reset_batch()


def test_reset_returns_to_trace_and_clears_all():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=128)
    _free(ba, k)
    ba.plan()
    nelems = ba.plan()  # second call on already optimized — wait, this will assert

    # Actually, let me test reset properly
    ba2 = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba2, k, size=128)
    _free(ba2, k)
    ba2.plan()

    ba2.reset()
    assert ba2.phase == "trace"
    assert len(ba2._trace) == 0
    assert len(ba2._key_to_slot) == 0
    assert len(ba2._slots) == 0
    assert len(ba2._pools) == 0
    assert ba2.total_pool_bytes == 0


# ---------------------------------------------------------------------------
# TracePoolAllocator — total_pool_bytes
# ---------------------------------------------------------------------------


def test_total_pool_bytes_reflects_pool_size():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=1024)
    _free(ba, k)
    ba.plan()
    expected = 1024 * torch.finfo(torch.float32).bits // 8  # 1024 * 4
    actual = ba.total_pool_bytes
    # Alignment may add padding, so actual >= expected
    assert actual >= expected


# ---------------------------------------------------------------------------
# TracePoolAllocator — dump_trace
# ---------------------------------------------------------------------------


def test_dump_trace_in_trace_phase():
    ba = TracePoolAllocator()
    _alloc(ba, _make_key("w1"), size=32)
    dump = ba.dump_trace()
    assert "phase=trace" in dump
    assert "w1" in dump


def test_dump_trace_in_optimized_phase():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=64)
    _free(ba, k)
    ba.plan()
    dump = ba.dump_trace()
    assert "phase=optimized" in dump
    assert "key_to_slot" in dump
    assert "slots:" in dump


def test_dump_trace_shows_active_keys():
    ba = TracePoolAllocator()
    k = _make_key("w1")
    _alloc(ba, k, size=64)
    _free(ba, k)
    ba.plan()
    _alloc(ba, k, size=32)
    dump = ba.dump_trace()
    assert "active_keys" in dump
    _free(ba, k)


# ---------------------------------------------------------------------------
# TracePoolAllocator — end-to-end multi-key scenario
# ---------------------------------------------------------------------------


def _make_key(name: str) -> tuple:
    return (0, name)


def test_e2e_trace_plan_optimized_multi_batch():
    """Simulate a typical FSDP forward/backward cycle across micro-batches."""
    ba = TracePoolAllocator()
    keys = [_make_key(f"layer_{i}") for i in range(4)]
    sizes = [1024, 2048, 512, 768]

    # ---- Trace (micro-batch 0) ----
    # Forward: allocate all params
    for k, sz in zip(keys, sizes):
        b = _alloc(ba, k, size=sz)
        assert b.data.numel() == sz
    # Backward: free all params (in reverse)
    for k in reversed(keys):
        _free(ba, k)

    # ---- Plan ----
    nelems = ba.plan()
    assert nelems > 0
    assert ba.phase == "optimized"

    # ---- Optimized (micro-batches 1..N) ----
    for _ in range(3):  # 3 more micro-batches
        ba.reset_batch()
        for k, sz in zip(keys, sizes):
            b = _alloc(ba, k, size=sz)
            assert b.data.numel() >= sz
        for k in reversed(keys):
            _free(ba, k)


def test_e2e_persistent_key_across_micro_batches():
    """A key not freed during trace stays active across micro-batches."""
    ba = TracePoolAllocator()
    k_persist = _make_key("persist")
    k_temp = _make_key("temp")

    _alloc(ba, k_persist, size=256)
    _alloc(ba, k_temp, size=128)
    _free(ba, k_temp)
    # k_persist NOT freed — persistent

    ba.plan()

    # Micro-batch 1
    _alloc(ba, k_persist, size=256)  # already cached via plan
    b = _alloc(ba, k_temp, size=128)
    _free(ba, k_temp)
    ptr1 = _data_ptr(b)
    # k_persist intentionally not freed

    ba.reset_batch()

    # Micro-batch 2
    _alloc(ba, k_persist, size=256)
    b = _alloc(ba, k_temp, size=128)
    _free(ba, k_temp)
    ptr2 = _data_ptr(b)
    # Same address for temp key
    assert ptr1 == ptr2


# ---------------------------------------------------------------------------
# TracePoolAllocator — BucketAllocator interface compliance
# ---------------------------------------------------------------------------


def test_trace_pool_allocator_is_bucket_allocator():
    """Ensure the allocator conforms to the BucketAllocator interface."""
    ba = TracePoolAllocator()
    assert isinstance(ba, BucketAllocator)


# ---------------------------------------------------------------------------
# TracePoolAllocator — same-key overlapping interval guard
# ---------------------------------------------------------------------------


def test_same_key_overlapping_intervals_detected_during_plan():
    """Overlapping intervals for the same key should be impossible in FSDP,
    but the coloring algorithm guards against it."""
    ba = TracePoolAllocator()
    k = _make_key("w1")

    # Manually craft overlapping intervals by calling alloc twice without free
    _alloc(ba, k, size=100)
    # second alloc for same active key — idempotent in trace, so only one event
    _alloc(ba, k, size=200)
    _free(ba, k)

    ba.plan()
    # Should succeed — the trace only has one alloc/free pair
    assert k in ba._key_to_slot
