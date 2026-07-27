# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for CUDA Graph runtime capture helpers."""

import contextlib
import gc
from unittest.mock import patch

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph import (
    _activation_recompute_capture_schedule,
    _activation_recompute_region_groups,
    _get_tracked_cuda_generators,
    _graph_context_wrapper,
    _none_grad_context_wrapper,
    _registered_buffer_slot_signature,
    _registered_buffer_slots,
)


def test_activation_recompute_region_schedule():
    """Group serial regions and capture each region in RF-then-B order."""
    groups = _activation_recompute_region_groups((0, 0, 1), 3)
    assert groups == ((0, 1), (2,))
    assert _activation_recompute_capture_schedule(groups) == (
        ("recompute", 2, True),
        ("backward", 2, True),
        ("recompute", 0, False),
        ("recompute", 1, True),
        ("backward", 1, True),
        ("backward", 0, False),
    )

    with pytest.raises(ValueError, match="contiguous and numbered"):
        _activation_recompute_region_groups((0, 1, 0), 3)


def test_registered_buffer_slots_detect_metadata_and_replacement():
    """Describe cached direct slots without a recursive replay-time walk."""
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module[0].register_buffer("scale", torch.ones(2))
    slots = _registered_buffer_slots(module)
    assert tuple(slot[0] for slot in slots) == ("0.scale",)

    original = _registered_buffer_slot_signature(slots[0])
    module[0].scale.requires_grad_(True)
    assert _registered_buffer_slot_signature(slots[0]) != original
    module[0].scale = module[0].scale.detach().clone()
    assert _registered_buffer_slot_signature(slots[0]) != original

    def fail_recursive_walk(*args, **kwargs):
        """Reject an unexpected recursive walk after slot capture."""
        del args, kwargs
        raise AssertionError("replay walked named_modules")

    module.named_modules = fail_recursive_walk
    _registered_buffer_slot_signature(slots[0])


def test_none_grad_context_restores_leaf_grad_after_exception():
    """Restore direct-runtime leaf gradients when capture fails."""
    parameter = torch.nn.Parameter(torch.ones(4))
    original_grad = torch.full_like(parameter, 3)
    parameter.grad = original_grad

    with pytest.raises(RuntimeError, match="capture failed"):
        with _none_grad_context_wrapper((parameter,)):
            assert parameter.grad is None
            raise RuntimeError("capture failed")

    assert parameter.grad is original_grad


def test_tracked_generator_discovery_supports_legacy_fallback():
    """Deduplicate generators and preserve the legacy tensor-state fallback."""
    generator = torch.Generator().manual_seed(123)
    with patch(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph."
        "get_all_rng_states",
        return_value={"first": generator, "alias": generator},
    ):
        assert _get_tracked_cuda_generators() == (generator,)

    with patch(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph."
        "get_all_rng_states",
        return_value={"legacy": torch.get_rng_state()},
    ):
        assert _get_tracked_cuda_generators(require_generators=False) is None
        with pytest.raises(RuntimeError, match="Legacy tensor RNG"):
            _get_tracked_cuda_generators()


def test_graph_context_restores_gc_after_capture_failure():
    """Restore garbage collection when the CUDA graph context raises."""

    @contextlib.contextmanager
    def failing_graph(*args, **kwargs):
        """Raise from a stand-in CUDA graph context."""
        del args, kwargs
        raise RuntimeError("capture failed")
        yield

    gc.enable()
    with (
        patch("torch.cuda.graph", failing_graph),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        with _graph_context_wrapper(object()):
            pass
    assert gc.isenabled()
