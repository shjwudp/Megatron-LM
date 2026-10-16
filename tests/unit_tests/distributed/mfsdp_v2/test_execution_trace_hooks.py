# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Trace both automatic FSDP hooks and combined 1F1B's custom demand hooks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.countdown import Countdown
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.schedule import SchedulePolicy
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    register_combined_1f1b_hooks,
    reshard_fsdp_module,
)


@pytest.mark.parametrize('manual', [False, True], ids=['automatic', 'combined_hooks'])
def test_hooks_share_trace_and_preserve_gradients(monkeypatch, manual):
    """Exercise production hooks with CPU autograd; mock only CUDA communication."""
    monkeypatch.setattr(torch.cuda, 'Stream', Mock())
    monkeypatch.setattr(torch.cuda.nvtx, 'range_push', Mock())
    monkeypatch.setattr(torch.cuda.nvtx, 'range_pop', Mock())
    context = FsdpContext(torch.device('cuda'), caller_managed_grad_sync=True)
    context.current_stream = Mock()
    gathers = []
    demands = []
    prefetched = []

    class Unit(nn.Module, FsdpModule):
        def __init__(self, child=None):
            nn.Module.__init__(self)
            self.child = child
            self.weight = nn.Parameter(torch.tensor(2.0))
            self._context = context
            self._phase = FsdpModule.Phase.RESTING
            self._parameter_groups = (
                SimpleNamespace(
                    requires_grad=True,
                    fsdp_parameters=[SimpleNamespace(unsharded=self.weight, fqns=('weight',))],
                ),
            )
            self._trainable_parameter_countdown = Countdown(1)
            self._schedule_policy = SchedulePolicy()
            self._unshard_event = None
            self._reduce_gradient_groups = Mock()
            if not manual:
                self._register_hooks()
            context.register_module(self)

        def unshard(self, *args, **kwargs):
            demands.append(self)
            super().unshard(*args, **kwargs)

        def _unshard_parameter_groups(self):
            if self._unshard_event is None:
                gathers.append(self)
                if demands[-1] is not self:
                    prefetched.append(self)
                self._unshard_event = object()

        def _reshard_parameter_groups(self):
            self._unshard_event = None

        def forward(self, inputs):
            if self.child is None:
                return inputs * self.weight
            # Shared ownership: several child calls demand the same parameter group.
            return self.child(inputs) + self.child(inputs + 1) + self.weight

    chunks = [Unit(Unit()), Unit(Unit())]
    if manual:
        for chunk in chunks:
            register_combined_1f1b_hooks(chunk)
    context.finalize()
    trace = context.execution_trace

    def run_schedule():
        outputs = [[], []]
        # Group chunk-0 forwards, then interleave forward and backward across roots.
        for phase, chunk_id in [
            ('f', 0),
            ('f', 0),
            ('f', 1),
            ('b', 0),
            ('f', 1),
            ('b', 1),
            ('b', 0),
            ('b', 1),
        ]:
            chunk = chunks[chunk_id]
            if phase == 'f':
                outputs[chunk_id].append(chunk(torch.ones(1, requires_grad=True)))
                if manual:
                    # Equivalent to the combined scheduler's explicit release callbacks.
                    reshard_fsdp_module(chunk.child)
                    reshard_fsdp_module(chunk)
            else:
                outputs[chunk_id].pop(0).sum().backward()

    with trace.iteration(key=('train', manual)):
        run_schedule()
    recorded = trace.events
    recorded_gathers = len(gathers)
    assert not prefetched
    assert any(event.phase == 'forward' for event in recorded)
    assert any(event.phase == 'backward' for event in recorded)
    expected_gradients = [
        parameter.grad.clone() for chunk in chunks for parameter in chunk.parameters()
    ]
    for chunk in chunks:
        chunk.zero_grad(set_to_none=True)
    with trace.iteration(key=('train', manual)):
        run_schedule()
    assert trace.events is recorded
    assert prefetched
    assert len(gathers) == 2 * recorded_gathers
    actual_gradients = [parameter.grad for chunk in chunks for parameter in chunk.parameters()]
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    assert all(module._unshard_event is None for chunk in chunks for module in (chunk, chunk.child))
    assert torch.cuda.nvtx.range_push.call_count == torch.cuda.nvtx.range_pop.call_count
    # Direct/custom unshard calls were also captured; no FSDP forward hook is required.
    assert sum(event.operation == 'unshard' for event in recorded) == 24
