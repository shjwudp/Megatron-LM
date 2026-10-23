# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Single-iteration capture, scheduler replay, and weak module ownership."""

import gc
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    iteration,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    IterationOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.models.common import mfsdp_iteration
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    _make_unshard_backward_hook,
    _make_unshard_forward_hook,
)
from megatron.core.models.common.mfsdp_iteration import with_mfsdp_iteration


def _demand(context, module, phase, budget=0):
    order = FsdpContext._advance_order(context, module, phase)
    if order is not None:
        FsdpModule._prefetch_iteration_parameter_groups(module, order, budget)


class _Unit(nn.Module):

    def __init__(self, context):
        super().__init__()
        self.context = context
        self.config = SimpleNamespace(
            cuda_graph_impl='none', virtual_pipeline_model_parallel_size=2
        )
        self.num_parameter_elements = 1
        self._unshard_parameter_groups = Mock()
        event = object()
        self._unshard_events = {Phase.FORWARD: event, Phase.BACKWARD: event}
        self.phase = Phase.RESTING
        self._schedule_policy = SimpleNamespace(
            forward_prefetch_size=None, backward_prefetch_size=None
        )

    def _nvtx_range(self, name):
        return nullcontext()

    def unshard(self, phase=None, *, prefetch=None):
        if isinstance(phase, str):
            self.phase = Phase.FORWARD if phase == "forward" else Phase.BACKWARD
            phase = Phase.BACKWARD if phase == "backward" else Phase.FORWARD
        elif phase is not None:
            self.phase = phase
        return FsdpModule.unshard(
            self, phase, prefetch=phase is not None if prefetch is None else prefetch
        )

    _prefetch_parameter_groups = FsdpModule._prefetch_parameter_groups
    _prefetch_iteration_parameter_groups = FsdpModule._prefetch_iteration_parameter_groups


@pytest.fixture
def context(monkeypatch):
    # Exercise the production schedule wrapper without allocating CUDA resources.
    monkeypatch.setattr(mfsdp_iteration, "FsdpModule", _Unit)
    context = Mock(
        ensure_finalized=Mock(),
        iteration_order=IterationOrder(),
        allgather_stream=Mock(),
        current_stream=Mock(),
    )
    context._advance_order = lambda module, phase: FsdpContext._advance_order(
        context, module, phase
    )
    return context


def test_schedule_capture_replay_and_validation(context):
    """One training order survives validation; replay resets its execution cursor."""
    a, b = _Unit(context), _Unit(context)
    events = [
        (a, "forward"),
        (a, "forward"),
        (b, "forward"),
        (b, "backward"),
        (a, "recompute"),
        (a, "backward"),
    ]
    gathers = Mock()
    gathers.attach_mock(a._unshard_parameter_groups, "a")
    gathers.attach_mock(b._unshard_parameter_groups, "b")
    gathers.attach_mock(context.current_stream.return_value.wait_event, "wait")
    prefetched = []

    @with_mfsdp_iteration
    def run(model, forward_only=False, budget=None):
        order = context.iteration_order
        if forward_only:
            assert order.mode == "demand"
            before = b._unshard_parameter_groups.call_count
            b.unshard("forward")
            assert b._unshard_parameter_groups.call_count == before + 1
            return
        assert order.mode == "record_replay"
        for module, phase in events:
            module._schedule_policy.forward_prefetch_size = budget
            module._schedule_policy.backward_prefetch_size = budget
            before = len(gathers.mock_calls)
            module.unshard(phase)
            calls = gathers.mock_calls[before:]
            assert calls[0][0] == ("a" if module is a else "b")
            assert calls[1][0] == "wait"
            prefetched.extend(call[0] for call in calls[2:])

    run(a, forward_only=True)
    assert not context.iteration_order.captured
    nested = with_mfsdp_iteration(run)
    for schedule, replay in ((run, False), (nested, True), (run, True)):
        gathers.reset_mock()
        prefetched.clear()
        schedule([a, b])
        assert prefetched == (["a", "b", "b", "a", "a"] if replay else [])
        order = context.iteration_order
        assert order.captured and not order.active
        assert order.position == len(events)
        _, phase = order[4]
        assert phase == "recompute"
        run([a, b], forward_only=True)
        assert context.iteration_order is order and order.position == len(events)
        assert order.mode == "record_replay"
        # The construction order was replaced; out-of-scope calls must not
        # consume the saved training trace (e.g. ordinary validation).
        before = len(gathers.mock_calls)
        b.unshard("forward")
        assert [call[0] for call in gathers.mock_calls[before:]] == ["b", "wait"]
        assert order.position == len(events)
    prefetched.clear()
    run(a, budget=0)
    assert not prefetched
    run(a, budget=2)
    # Each demand scans its own window, including overlapping occurrences.
    assert prefetched == ["a", "b", "b", "b", "b", "a", "a", "a", "a"]


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine", "full_iteration"])
def test_cuda_graph_demands_preserve_order_outside_scheduler(context, cuda_graph_impl):
    """Graph warmups may revisit units outside the scheduler's iteration boundary."""
    a, b = _Unit(context), _Unit(context)
    for module in (a, b):
        module.config.cuda_graph_impl = cuda_graph_impl
    order = context.iteration_order
    for module, phase in ((a, "forward"), (b, "forward"), (b, "backward"), (a, "backward")):
        order.append((ref(module), phase))
    order.backward_start = 2
    original = list(order), order.position
    events = [(a, "forward")] * 2 + [(b, "forward")] * 2
    events += [(b, "backward")] * 2 + [(a, "backward")] * 2
    gathers = Mock()
    gathers.attach_mock(a._unshard_parameter_groups, "a")
    gathers.attach_mock(b._unshard_parameter_groups, "b")

    @with_mfsdp_iteration
    def run(model):
        for module, phase in events:
            module.unshard(phase)
        assert order.mode == "demand" and not order.active

    for _ in range(2):
        gathers.reset_mock()
        run([a, b])
        assert [call[0] for call in gathers.mock_calls] == [
            "a" if module is a else "b" for module, _ in events
        ]
        # Local/TE graph creation also calls units after the scheduler returns.
        b.unshard("forward")
        b.unshard("backward")
        assert (list(order), order.position) == original
        assert not order.captured and not order.active

    # Combined hooks retain their root stream dependency without an active scope.
    a.is_root = Mock(return_value=True)
    context.allgather_stream.wait_stream.reset_mock()
    _make_unshard_forward_hook(a)(None, (), {})
    context.allgather_stream.wait_stream.assert_called_once_with(context.current_stream())


def test_nested_graph_scheduler_preserves_explicit_iteration(context):
    """A nested wrapper must not change the strategy of an already active scope."""
    module = _Unit(context)
    module.config.cuda_graph_impl = "full_iteration"

    @with_mfsdp_iteration
    def run(model):
        assert context.iteration_order.mode == "record_replay"
        assert context.iteration_order.active
        model.unshard("forward")

    with iteration(context):
        run(module)
    assert context.iteration_order.captured


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("active", [False, True], ids=["static", "capturing"])
def test_combined_hooks_only_consume_active_order(context, phase, active):
    """Fine-grained repeated owner demands require an iteration scope."""
    owner, other = _Unit(context), _Unit(context)
    owner.is_root = Mock(return_value=True)
    order = context.iteration_order
    if active:
        order.mode = "record_replay"
        order.active = True
    else:
        order.append((ref(owner), phase))
        order.append((ref(other), phase))
    before = list(order), order.position
    if phase == "forward":
        hook = _make_unshard_forward_hook(owner)
        for _ in range(2):
            hook(None, (), {})
    else:
        hook = _make_unshard_backward_hook(owner)
        for _ in range(2):
            hook(None, ())
    assert owner._unshard_parameter_groups.call_count == 2
    if active:
        assert [(module_ref(), event_phase) for module_ref, event_phase in order] == [
            (owner, phase),
            (owner, phase),
        ]
        assert order.position == 2
    else:
        assert (list(order), order.position) == before
        other._unshard_parameter_groups.assert_not_called()
