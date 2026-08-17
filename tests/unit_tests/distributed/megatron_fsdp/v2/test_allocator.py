# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for TemporaryBucketAllocator. Pure CPU, no torch.distributed."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import (
    Bucket,
    TemporaryBucketAllocator,
    TracePoolAllocator,
)


def _run_allocator_tests(allocator: TemporaryBucketAllocator) -> None:
    """Three-phase test covering allocate, free, and re-allocate for any allocator.

    Phase 1 -- allocate:
      - Returned bucket has correct size, dtype, and is tracked internally.
      - Duplicate allocate on the same id returns the same object (no realloc).
      - Different ids produce independent buckets.

    Phase 2 -- free:
      - Freed id is removed from internal tracking.
      - Underlying tensor storage is resized to 0.
      - Freeing a non-existent id is silently ignored.
      - Freeing one id does not affect others.

    Phase 3 -- re-allocate after free:
      - Re-allocating a freed id returns a usable bucket with correct size.
    """

    # ---- Phase 1: allocate ----
    b0 = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0.data.numel() == 1024
    assert b0.data.dtype == torch.float32
    assert 0 in allocator.buckets

    b0_again = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0_again is b0
    assert b0_again.data.data_ptr() == b0.data.data_ptr()

    b1 = allocator.allocate(
        param_group_id=1, size=512, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert b1 is not b0
    assert b1.data.numel() == 512
    assert b1.data.dtype == torch.bfloat16

    # ---- Phase 2: free ----
    tensor_ref = b0.data
    allocator.free(0)
    assert 0 not in allocator.buckets
    assert tensor_ref._typed_storage()._size() == 0

    assert 1 in allocator.buckets
    assert b1.data.numel() == 512

    allocator.free(999)

    # ---- Phase 3: re-allocate ----
    b0_new = allocator.allocate(
        param_group_id=0, size=1024, dtype=torch.float32, device=torch.device("cpu")
    )
    assert b0_new.data.numel() == 1024
    assert 0 in allocator.buckets

    # cleanup
    allocator.free(0)
    allocator.free(1)


class TestTemporaryBucketAllocator:

    def test_full_lifecycle(self):
        _run_allocator_tests(TemporaryBucketAllocator())


class TestTracePoolAllocator:

    def test_trace_phase_reuses_free_slot_without_releasing_storage(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        first = allocator.allocate(key="first", size=16, dtype=torch.float32, device=device)
        first_ptr = first.data.data_ptr()
        allocator.free("first")

        # A logical free remains owned by the online trace pool instead of
        # being returned to the framework allocator.
        assert first.data._typed_storage()._size() == 16

        second = allocator.allocate(key="second", size=8, dtype=torch.float32, device=device)
        assert second.data.data_ptr() == first_ptr
        assert len(allocator._trace_slots) == 1

    def test_trace_phase_uses_best_fit_free_slot(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        large = allocator.allocate(key="large", size=16, dtype=torch.float32, device=device)
        small = allocator.allocate(key="small", size=8, dtype=torch.float32, device=device)
        large_ptr = large.data.data_ptr()
        small_ptr = small.data.data_ptr()
        allocator.free("large")
        allocator.free("small")

        chosen = allocator.allocate(key="chosen", size=7, dtype=torch.float32, device=device)
        assert chosen.data.data_ptr() == small_ptr
        assert chosen.data.data_ptr() != large_ptr

    def test_trace_phase_does_not_reuse_overlapping_or_incompatible_slots(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        first = allocator.allocate(key="first", size=16, dtype=torch.float32, device=device)
        overlapping = allocator.allocate(
            key="overlapping", size=8, dtype=torch.float32, device=device
        )
        allocator.free("overlapping")
        incompatible = allocator.allocate(
            key="incompatible", size=8, dtype=torch.bfloat16, device=device
        )

        assert first.data.data_ptr() != overlapping.data.data_ptr()
        assert incompatible.data.data_ptr() != overlapping.data.data_ptr()
        assert len(allocator._trace_slots) == 3

    def test_trace_phase_rebinds_existing_bucket_tensor_in_place(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        original = allocator.allocate(key="key", size=8, dtype=torch.float32, device=device)
        tensor_object = original.data
        allocator.free("key")

        reallocated = allocator.allocate(key="key", size=8, dtype=torch.float32, device=device)
        assert reallocated is original
        assert reallocated.data is tensor_object

    def test_plan_retires_online_trace_slots(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        traced = allocator.allocate(key="traced", size=8, dtype=torch.float32, device=device)
        trace_storage = traced.data._typed_storage()
        allocator.free("traced")
        allocator.plan()

        assert allocator.phase == "optimized"
        assert allocator._trace_slots == []
        assert trace_storage._size() == 0

    def test_plan_rejects_live_trace_allocations(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        allocator.allocate(key="live", size=8, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError, match="all trace allocations to be freed"):
            allocator.plan()

    def test_optimized_allocate_rejects_live_slot_collision(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        allocator.allocate(key="first", size=8, dtype=torch.float32, device=device)
        allocator.free("first")
        allocator.allocate(key="second", size=8, dtype=torch.float32, device=device)
        allocator.free("second")
        allocator.plan()

        assert allocator._key_to_slot["first"] == allocator._key_to_slot["second"]
        allocator.allocate(key="first", size=8, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError, match="slot collision"):
            allocator.allocate(key="second", size=8, dtype=torch.float32, device=device)

    def test_late_optimized_key_gets_dedicated_slot(self):
        allocator = TracePoolAllocator()
        device = torch.device("cpu")

        allocator.allocate(key="traced", size=8, dtype=torch.float32, device=device)
        allocator.free("traced")
        allocator.plan()
        assert allocator.phase == "optimized"

        bucket = allocator.allocate(key="late", size=4, dtype=torch.float32, device=device)
        assert bucket.data.numel() == 4
        assert bucket.data.dtype == torch.float32
        assert "late" in allocator._key_to_slot
        allocator.free("late")

        allocator.release()
        assert allocator.phase == "released"
        bucket = allocator.allocate(key="late", size=4, dtype=torch.float32, device=device)
        assert allocator.phase == "optimized"
        assert bucket.data.numel() == 4
        assert bucket.data.dtype == torch.float32
        allocator.free("late")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
