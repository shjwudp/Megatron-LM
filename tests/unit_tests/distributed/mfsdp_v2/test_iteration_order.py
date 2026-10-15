# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prototype coverage for full-iteration recording and VPP replay."""

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.countdown import Countdown
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.schedule import SchedulePolicy
from megatron.core.pipeline_parallel.schedules import _mfsdp_iteration, get_schedule_table


def test_vpp_iteration_records_then_replays_across_validation(monkeypatch):
    """One context records actual grouped chunk calls and preserves train/eval traces."""
    monkeypatch.setattr(torch.cuda, 'Stream', lambda *_args: Mock())
    monkeypatch.setattr(torch.cuda.nvtx, 'range_push', Mock())
    monkeypatch.setattr(torch.cuda.nvtx, 'range_pop', Mock())
    context = FsdpContext(torch.device('cuda'), caller_managed_grad_sync=True)
    context.current_stream = Mock()

    class HookModule(nn.Module, FsdpModule):
        """Use production hooks with real CPU autograd and mocked communication."""

        def __init__(self, child=None):
            nn.Module.__init__(self)
            self.child = child
            self._context = context
            self._phase = FsdpModule.Phase.RESTING
            self.weight = nn.Parameter(torch.tensor(2.0))
            self._parameter_groups = (
                SimpleNamespace(
                    requires_grad=True,
                    fsdp_parameters=[SimpleNamespace(unsharded=self.weight, fqns=('weight',))],
                ),
            )
            self._trainable_parameter_countdown = Countdown(1)
            self._schedule_policy = SchedulePolicy()
            self._unshard_event = None
            self._unshard_parameter_groups = Mock(
                side_effect=lambda: setattr(self, '_unshard_event', object())
            )
            self.reshard = Mock(side_effect=lambda: setattr(self, '_unshard_event', None))
            self._reduce_gradient_groups = Mock()
            self._prefetch_parameter_groups = Mock(wraps=self._prefetch_parameter_groups)
            self._register_hooks()
            context.register_module(self)

        def forward(self, inputs):
            if self.child is None:
                return inputs * self.weight
            # A shared layer is called twice inside each chunk.
            return self.child(inputs) + self.child(inputs + 1) + self.weight

    chunks = [HookModule(HookModule()) for _ in range(2)]
    context.finalize()
    assert context.forward_order is None
    assert context.backward_order is None
    config = SimpleNamespace(cuda_graph_impl='none', microbatch_group_size_per_vp_stage=2)
    wrappers = [
        SimpleNamespace(
            module=chunk,
            config=config,
            ddp_config=SimpleNamespace(use_megatron_fsdp=True, megatron_fsdp_version=2),
        )
        for chunk in chunks
    ]
    # get_model_config only unwraps attributes; avoid needing a TransformerConfig here.
    monkeypatch.setitem(
        _mfsdp_iteration.__globals__, 'get_model_config', lambda model: model.config
    )

    @_mfsdp_iteration
    def run_schedule(*, model, num_microbatches, forward_only=False, **_kwargs):
        schedule = get_schedule_table(
            num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
        )
        outputs = [[], []]
        backward_calls = iter(schedule)

        def backward_step():
            _, forward_chunk_id = next(backward_calls)
            outputs[len(model) - forward_chunk_id - 1].pop(0).sum().backward()

        # Simulate rank 0 of PP=2: warmup, interleaved 1F1B, and cooldown.
        warmup = min(len(schedule), 2 + config.microbatch_group_size_per_vp_stage)
        with torch.no_grad() if forward_only else contextlib.nullcontext():
            for step, (_, chunk_id) in enumerate(schedule):
                outputs[chunk_id].append(model[chunk_id].module(torch.ones(1, requires_grad=True)))
                if not forward_only and step >= warmup:
                    backward_step()
        if not forward_only:
            for _ in range(warmup):
                backward_step()

    run_schedule(model=wrappers, num_microbatches=6)
    training_order = context.forward_order
    assert list(training_order) == [
        module
        for _, chunk in get_schedule_table(6, 2, 2)
        for module in (chunks[chunk], chunks[chunk].child, chunks[chunk].child)
    ]
    assert len(list(context.backward_order)) == len(list(training_order))
    for chunk in chunks:
        chunk._prefetch_parameter_groups.assert_not_called()
        chunk.child._prefetch_parameter_groups.assert_not_called()
        assert chunk.weight.grad.item() == 6
        assert chunk.child.weight.grad.item() == 18
        chunk.zero_grad(set_to_none=True)

    run_schedule(model=wrappers, num_microbatches=6)
    assert context.forward_order is training_order
    assert training_order.is_complete()
    assert context.backward_order.is_complete()
    assert chunks[0]._prefetch_parameter_groups.call_count > 0
    for chunk in chunks:
        chunk.eval()
    run_schedule(model=wrappers, num_microbatches=2, forward_only=True)
    assert context.backward_order is None
    for chunk in chunks:
        chunk.train()
    run_schedule(model=wrappers, num_microbatches=6)
    assert context.forward_order is training_order

    # A new grouping invalidates the old training trace, even at the same batch size.
    config.microbatch_group_size_per_vp_stage = 3
    run_schedule(model=wrappers, num_microbatches=6)
    assert context.forward_order is not training_order
    new_order = context.forward_order
    run_schedule(model=wrappers, num_microbatches=6)
    assert context.forward_order is new_order
    assert new_order.is_complete()
    assert torch.cuda.nvtx.range_push.call_count == torch.cuda.nvtx.range_pop.call_count
    assert all(
        module.phase is FsdpModule.Phase.RESTING
        for chunk in chunks
        for module in (chunk, chunk.child)
    )


def test_failed_and_incomplete_iterations_do_not_install_traces(monkeypatch):
    """Failed capture, truncated replay, and mismatches cannot become valid traces."""
    monkeypatch.setattr(torch.cuda, 'Stream', Mock())
    context = FsdpContext(torch.device('cuda'))
    context.finalize()
    first, second = nn.Module(), nn.Module()
    with pytest.raises(RuntimeError, match='aborted'):
        with context.iteration():
            context.execution_order(first)
            raise RuntimeError('aborted')
    assert context.forward_order is None
    assert context.backward_order is None
    with context.iteration():
        context.execution_order(first)
        context.execution_order(second, backward=True)
    recorded = context.forward_order
    with pytest.raises(RuntimeError, match='before.*consumed'):
        with context.iteration():
            context.execution_order(first)
    with pytest.raises(RuntimeError, match='diverged'):
        with context.iteration():
            context.execution_order(second)
    with context.iteration():
        context.execution_order(first)
        context.execution_order(second, backward=True)
    assert context.forward_order is recorded
    assert recorded.is_complete()


def test_forward_only_and_nested_iteration_guards(monkeypatch):
    """Forward-only capture never records backward; a shared scope cannot be nested."""
    monkeypatch.setattr(torch.cuda, 'Stream', Mock())
    context = FsdpContext(torch.device('cuda'))
    context.finalize()
    module = nn.Module()
    with pytest.raises(RuntimeError, match='already active'):
        with context.iteration(forward_only=True):
            with context.iteration(forward_only=True):
                pass
    with pytest.raises(RuntimeError, match='forward-only'):
        with context.iteration(forward_only=True):
            context.execution_order(module)
            context.execution_order(module, backward=True)
    with context.iteration(forward_only=True):
        context.execution_order(module)
    with context.iteration(forward_only=True):
        context.execution_order(module)
    assert context.forward_order.is_complete()
    assert context.backward_order is None
