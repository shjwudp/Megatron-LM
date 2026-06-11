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

"""Unit tests for allocators. Pure CPU, no torch.distributed."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Import from source file directly — avoids megatron.core import chain which
# requires a GPU environment.
_SRC_DIR = Path(__file__).resolve().parents[5] / "megatron" / "core" / "distributed" / "fsdp" / "src"
_ALLOCATOR_PATH = _SRC_DIR / "megatron_fsdp" / "v2" / "allocator.py"
assert _ALLOCATOR_PATH.exists(), f"allocator.py not found at {_ALLOCATOR_PATH}"

spec = importlib.util.spec_from_file_location(
    "mfsdp_v2_allocator", str(_ALLOCATOR_PATH)
)
_allocator_mod = importlib.util.module_from_spec(spec)
sys.modules["mfsdp_v2_allocator"] = _allocator_mod
spec.loader.exec_module(_allocator_mod)

TemporaryBucketAllocator = _allocator_mod.TemporaryBucketAllocator
TracePoolAllocator = _allocator_mod.TracePoolAllocator


def _run_allocator_tests(allocator) -> None:
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


# ------------------------------------------------------------------
# Helper: run a simple trace → plan → optimized cycle
# ------------------------------------------------------------------

def _trace_plan_cycle(allocator: TracePoolAllocator) -> int:
    """Run a minimal trace→plan cycle and return pool element count.

    Simulates a realistic FSDP forward/backward pattern with
    overlapping allocations (A and B active simultaneously, then A freed,
    then C allocated while B is still active).
    """
    dtype = torch.float32
    device = torch.device("cpu")

    # Forward: allocate two buffers
    allocator.allocate(key="A", size=100, dtype=dtype, device=device)
    allocator.allocate(key="B", size=200, dtype=dtype, device=device)

    # Free A after its forward pass
    allocator.free(key="A")

    # Backward: allocate C while B is still active
    allocator.allocate(key="C", size=150, dtype=dtype, device=device)

    # Free remaining
    allocator.free(key="B")
    allocator.free(key="C")

    return allocator.plan()


# ------------------------------------------------------------------
# TracePoolAllocator tests
# ------------------------------------------------------------------

class TestTracePoolAllocator:

    def test_full_lifecycle(self):
        """Trace → plan → optimized: allocate/free returns fixed-address views."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)
        assert allocator.phase == "optimized"

        dtype = torch.float32
        device = torch.device("cpu")

        # First micro-batch: allocate A
        b0 = allocator.allocate(key="A", size=100, dtype=dtype, device=device)
        assert b0.data.numel() >= 100
        addr0 = b0.data.data_ptr()

        # Second micro-batch: allocate A again — same address
        allocator.free(key="A")
        allocator.free(key="B")  # B wasn't allocated yet; no-op
        b0b = allocator.allocate(key="A", size=100, dtype=dtype, device=device)
        assert b0b.data.data_ptr() == addr0

        allocator.free(key="A")

        # Allocate B+C overlapping
        b1 = allocator.allocate(key="B", size=200, dtype=dtype, device=device)
        b2 = allocator.allocate(key="C", size=150, dtype=dtype, device=device)
        assert b1.data.data_ptr() != b2.data.data_ptr()
        allocator.free(key="B")
        allocator.free(key="C")

    def test_release_in_trace_phase(self):
        """release() in trace phase resets to clean trace state."""
        allocator = TracePoolAllocator()
        assert allocator.phase == "trace"

        allocator.allocate(key="A", size=100, dtype=torch.float32,
                           device=torch.device("cpu"))
        allocator.allocate(key="B", size=200, dtype=torch.float32,
                           device=torch.device("cpu"))

        assert len(allocator._trace) > 0
        assert len(allocator._trace_meta) > 0
        assert "A" in allocator._buckets

        allocator.release()

        assert allocator.phase == "trace"
        assert len(allocator._trace) == 0
        assert len(allocator._trace_meta) == 0
        assert len(allocator._buckets) == 0

    def test_release_in_optimized_phase(self):
        """release() in optimized phase frees tensors but preserves plan."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)
        assert allocator.phase == "optimized"

        n_slots_before = len(allocator._slots)
        n_keys = len(allocator._key_to_slot)
        assert n_slots_before > 0
        assert n_keys > 0

        # Verify slots have real memory
        for slot in allocator._slots:
            assert slot.tensor.numel() > 0

        allocator.release()
        assert allocator.phase == "released"

        # Plan metadata preserved
        assert len(allocator._slots) == n_slots_before
        assert len(allocator._key_to_slot) == n_keys
        assert len(allocator._key_to_view) == n_keys

        # Slots have zero-sized tensors (memory freed)
        for slot in allocator._slots:
            assert slot.tensor.numel() == 0
            assert slot.in_use is False

    def test_auto_resume_on_allocate(self):
        """First allocate after release auto-resumes slots."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)

        # Capture addresses before release
        dtype = torch.float32
        device = torch.device("cpu")
        b_pre = allocator.allocate(key="A", size=100, dtype=dtype, device=device)
        addr_pre = b_pre.data.data_ptr()
        allocator.free(key="A")

        allocator.release()
        assert allocator.phase == "released"

        # allocate() triggers auto-resume
        b_post = allocator.allocate(key="A", size=100, dtype=dtype, device=device)
        assert allocator.phase == "optimized"
        assert b_post.data.numel() >= 100
        assert b_post.data.data_ptr() != 0

        allocator.free(key="A")

    def test_auto_resume_on_free(self):
        """First free after release auto-resumes slots."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)

        dtype = torch.float32
        device = torch.device("cpu")
        allocator.allocate(key="A", size=100, dtype=dtype, device=device)
        allocator.free(key="A")

        allocator.release()
        assert allocator.phase == "released"

        # free() triggers auto-resume (no-op since A is already freed)
        allocator.free(key="A")
        assert allocator.phase == "optimized"

    def test_multiple_release_resume_cycles(self):
        """Multiple release → allocate (auto-resume) cycles work."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)
        assert allocator.phase == "optimized"

        dtype = torch.float32
        device = torch.device("cpu")

        for _ in range(3):
            b = allocator.allocate(key="A", size=100, dtype=dtype, device=device)
            allocator.free(key="A")

            allocator.release()
            assert allocator.phase == "released"

            # Auto-resume via free
            allocator.free(key="B")
            assert allocator.phase == "optimized"

            # Verify slots are re-allocated (non-zero)
            for slot in allocator._slots:
                assert slot.tensor.numel() > 0

    def test_idempotent_allocate_after_release(self):
        """Double allocate after release is idempotent."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)
        allocator.release()

        dtype = torch.float32
        device = torch.device("cpu")

        b1 = allocator.allocate(key="A", size=100, dtype=dtype, device=device)
        b2 = allocator.allocate(key="A", size=100, dtype=dtype, device=device)
        assert b1.data.data_ptr() == b2.data.data_ptr()
        allocator.free(key="A")

    def test_unknown_key_raises_keyerror(self):
        """Allocating an key not seen during trace raises KeyError."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)

        with pytest.raises(KeyError):
            allocator.allocate(key="UNKNOWN", size=100, dtype=torch.float32,
                               device=torch.device("cpu"))

    def test_reset_clears_all(self):
        """reset() discards everything and returns to trace."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)
        assert allocator.phase == "optimized"

        allocator.reset()
        assert allocator.phase == "trace"
        assert len(allocator._trace) == 0
        assert len(allocator._trace_meta) == 0
        assert len(allocator._buckets) == 0
        assert len(allocator._slots) == 0
        assert len(allocator._key_to_slot) == 0
        assert len(allocator._key_to_view) == 0

    def test_dump_trace_covers_phases(self):
        """dump_trace() works in trace, optimized, and released phases."""
        allocator = TracePoolAllocator()
        s = allocator.dump_trace()
        assert "phase=trace" in s

        allocator.allocate(key="A", size=100, dtype=torch.float32,
                           device=torch.device("cpu"))
        allocator.free(key="A")
        allocator.plan()
        s = allocator.dump_trace()
        assert "phase=optimized" in s
        assert "slots:" in s

        allocator.release()
        s = allocator.dump_trace()
        assert "phase=released" in s
        assert "<released>" in s

    def test_total_pool_bytes(self):
        """total_pool_bytes returns positive value after plan()."""
        allocator = TracePoolAllocator()
        _trace_plan_cycle(allocator)
        assert allocator.total_pool_bytes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
