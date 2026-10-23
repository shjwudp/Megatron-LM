# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Single-iteration capture, scheduler replay, and weak module ownership."""

import gc
from contextlib import contextmanager, nullcontext
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
    ExecutionOrder,
    PrefetchMode,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    _make_unshard_backward_hook,
    _make_unshard_forward_hook,
)


@contextmanager
def _validation(context):
    previous_mode = context.prefetch_mode
    context.prefetch_mode = PrefetchMode.DEMAND
    try:
        yield
    finally:
        context.prefetch_mode = previous_mode


def _demand(context, module, phase, budget=0):
    order = FsdpContext._advance_order(context, module, phase)
    if order is not None:
        FsdpModule._prefetch_parameter_groups(module, order, budget)


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
        if phase is not None:
            self.phase = Phase.BACKWARD if phase is Phase.RECOMPUTE else phase
            if phase is Phase.RECOMPUTE:
                phase = Phase.FORWARD
        return FsdpModule.unshard(
            self, phase, prefetch=phase is not None if prefetch is None else prefetch
        )

    _prefetch_parameter_groups = FsdpModule._prefetch_parameter_groups


@pytest.fixture
def context():
    class Context(SimpleNamespace):
        iteration_recording = FsdpContext.iteration_recording

    context = Context(
        ensure_finalized=Mock(),
        iteration_order=ExecutionOrder(),
        prefetch_mode=PrefetchMode.PRESET,
        iteration_captured=False,
        allgather_stream=Mock(),
        current_stream=Mock(),
    )
    context._advance_order = lambda module, phase: FsdpContext._advance_order(
        context, module, phase
    )
    return context


def test_unscoped_calls_keep_static_prefetch(context):
    """Repeated chunk calls retain construction-order lookup until explicitly captured."""
    a, b = _Unit(context), _Unit(context)
    context.iteration_order = ExecutionOrder(
        [
            (ref(a), Phase.FORWARD),
            (ref(b), Phase.FORWARD),
            (ref(b), Phase.BACKWARD),
            (ref(a), Phase.BACKWARD),
        ]
    )
    for phase, events in ((Phase.FORWARD, (a, a, b, b)), (Phase.BACKWARD, (b, b, a, a))):
        for module in events:
            a._unshard_parameter_groups.reset_mock()
            b._unshard_parameter_groups.reset_mock()
            module.unshard(phase)
            module._unshard_parameter_groups.assert_called_once_with(
                Phase.FORWARD if phase == Phase.FORWARD else Phase.BACKWARD
            )
            successor = b if phase == Phase.FORWARD else a
            if module is not successor:
                successor._unshard_parameter_groups.assert_called_once_with(
                    Phase.FORWARD if phase == Phase.FORWARD else Phase.BACKWARD
                )
            else:
                other = a if successor is b else b
                other._unshard_parameter_groups.assert_not_called()
        assert context.prefetch_mode is PrefetchMode.PRESET
        assert len(context.iteration_order) == 4
    # Preset recomputation must neither consume the order nor prefetch successors.
    position = context.iteration_order.position
    a._unshard_parameter_groups.reset_mock()
    b._unshard_parameter_groups.reset_mock()
    a.unshard(Phase.RECOMPUTE)
    a._unshard_parameter_groups.assert_called_once_with(Phase.FORWARD)
    b._unshard_parameter_groups.assert_not_called()
    assert context.iteration_order.position == position


def test_capture_replay_and_validation(context):
    """One training order survives validation; replay resets its execution cursor."""
    a, b = (_Unit(context), _Unit(context))
    events = [
        (a, Phase.FORWARD),
        (a, Phase.FORWARD),
        (b, Phase.FORWARD),
        (b, Phase.BACKWARD),
        (a, Phase.RECOMPUTE),
        (a, Phase.BACKWARD),
    ]
    gathers = Mock()
    gathers.attach_mock(a._unshard_parameter_groups, 'a')
    gathers.attach_mock(b._unshard_parameter_groups, 'b')
    gathers.attach_mock(context.current_stream.return_value.wait_event, 'wait')
    prefetched = []

    def run(model, forward_only=False, budget=None):
        with _validation(context) if forward_only else iteration(context):
            order = context.iteration_order
            if forward_only:
                assert context.prefetch_mode is PrefetchMode.DEMAND
                before = b._unshard_parameter_groups.call_count
                b.unshard(Phase.FORWARD)
                assert b._unshard_parameter_groups.call_count == before + 1
                return
            assert context.prefetch_mode is PrefetchMode.RECORD_REPLAY
            for module, phase in events:
                module._schedule_policy.forward_prefetch_size = budget
                module._schedule_policy.backward_prefetch_size = budget
                before = len(gathers.mock_calls)
                module.unshard(phase)
                calls = gathers.mock_calls[before:]
                assert calls[0][0] == ('a' if module is a else 'b')
                assert calls[1][0] == 'wait'
                prefetched.extend((call[0] for call in calls[2:]))

    run(a, forward_only=True)
    assert not context.iteration_captured
    for schedule, replay in ((run, False), (run, True), (run, True)):
        gathers.reset_mock()
        prefetched.clear()
        schedule([a, b])
        assert prefetched == (['a', 'b', 'b', 'a', 'a'] if replay else [])
        order = context.iteration_order
        assert context.iteration_captured and context.prefetch_mode is PrefetchMode.DEMAND
        expected_position = len(events) if replay else 0
        assert order.position == expected_position
        _, phase = order[4]
        assert phase == Phase.RECOMPUTE
        run([a, b], forward_only=True)
        assert context.iteration_order is order and order.position == expected_position
        assert context.prefetch_mode is PrefetchMode.DEMAND
        before = len(gathers.mock_calls)
        b.unshard(Phase.FORWARD)
        assert [call[0] for call in gathers.mock_calls[before:]] == ['b', 'wait']
        assert order.position == expected_position
    prefetched.clear()
    run(a, budget=0)
    assert not prefetched
    run(a, budget=2)
    assert prefetched == ['a', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a']


@pytest.mark.parametrize('captured', [False, True], ids=['capturing', 'replaying'])
def test_manual_unshard_preserves_iteration_order(context, captured):
    """Parameter inspection must neither record demands nor consume the saved trace."""
    a, b = (_Unit(context), _Unit(context))
    if captured:
        with iteration(context):
            a.unshard(Phase.FORWARD)
            b.unshard(Phase.BACKWARD)
    with iteration(context):
        order = context.iteration_order
        before = (list(order), order.position)
        a.unshard()
        assert (list(order), order.position) == before
        a.unshard(Phase.FORWARD)
        b.unshard(Phase.BACKWARD)


def test_failed_or_changed_iterations_require_capture(context):
    """Mismatched, incomplete, extra, and failed calls invalidate the single order."""
    a, b = (_Unit(context), _Unit(context))
    events = [(a, Phase.FORWARD)]

    def run(model, fail=False):
        with iteration(context):
            for module, phase in events:
                _demand(context, module, phase)
            if fail:
                raise ValueError('failed compute')

    for changed in (
        [(b, Phase.FORWARD)],
        [(a, Phase.BACKWARD)],
        [],
        [(a, Phase.FORWARD), (a, Phase.FORWARD)],
    ):
        events = [(a, Phase.FORWARD)]
        with iteration(context):
            _demand(context, a, Phase.FORWARD)
        events = changed
        with pytest.raises(RuntimeError):
            run(a)
        assert context.prefetch_mode is PrefetchMode.DEMAND and not context.iteration_captured
    with pytest.raises(ValueError):
        run(a, fail=True)
    assert not context.iteration_captured
    with iteration(context):
        with pytest.raises(RuntimeError, match='active'):
            with iteration(context):
                pass
        _demand(context, b, Phase.FORWARD)
    events = [(b, Phase.FORWARD)]
    run(a)
    assert context.iteration_captured
    with pytest.raises(ValueError, match='failed replay'):
        with iteration(context):
            raise ValueError('failed replay')
    assert context.prefetch_mode is PrefetchMode.DEMAND and not context.iteration_captured


@pytest.mark.parametrize("budget", [None, 0], ids=["prefetch", "demand"])
def test_capture_does_not_retain_modules(context, budget):
    """Captured occurrences keep weak references and reject a destroyed owner."""
    a, b = (_Unit(context), _Unit(context))
    with iteration(context):
        _demand(context, a, Phase.FORWARD)
        _demand(context, b, Phase.BACKWARD)
    b_ref = ref(b)
    del b
    gc.collect()
    assert b_ref() is None

    def run(model):
        with iteration(context):
            _demand(context, model, Phase.FORWARD, budget=budget)
            _demand(context, model, Phase.BACKWARD)

    with pytest.raises(RuntimeError, match='destroyed' if budget is None else 'diverged'):
        run(a)
    assert context.prefetch_mode is PrefetchMode.DEMAND and not context.iteration_captured


@pytest.mark.parametrize(
    'mode, captured',
    [
        (PrefetchMode.PRESET, False),
        (PrefetchMode.RECORD_REPLAY, False),
        (PrefetchMode.RECORD_REPLAY, True),
        (PrefetchMode.DEMAND, True),
    ],
)
def test_repeated_backward_keeps_static_guard(mode, captured, monkeypatch):
    """Active scopes observe repeated backward calls; static mode keeps its guard."""
    context = Mock(iteration_order=ExecutionOrder())
    context.prefetch_mode = mode
    context.iteration_captured = captured
    module = Mock(spec=FsdpModule, context=context, phase=Phase.BACKWARD)
    module.is_root.return_value = False
    push = Mock()
    monkeypatch.setattr(torch.cuda.nvtx, 'range_push', push)
    FsdpModule.pre_backward(module)
    if mode is not PrefetchMode.RECORD_REPLAY:
        module.unshard.assert_not_called()
        context.current_stream.assert_not_called()
    else:
        module.unshard.assert_called_once_with(Phase.BACKWARD, prefetch=True)
    push.assert_not_called()


@pytest.mark.parametrize('reentrant', [False, True])
@pytest.mark.parametrize('record_iteration', [False, True], ids=['static', 'recorded'])
def test_checkpoint_demand_recording(distributed_setup, reentrant, record_iteration):
    """Static and recorded prefetch preserve SGD with both checkpoint variants."""

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            x = checkpoint(self.fc1, x, use_reentrant=reentrant)
            if not record_iteration and isinstance(self.fc2, FsdpModule):
                assert Phase.FORWARD in self.fc2._unshard_events
            return checkpoint(self.fc2, x.relu(), use_reentrant=reentrant)

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(1234)
    baseline, model = (Model().to(device), Model().to(device))
    model.load_state_dict(baseline.state_dict())
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=device) as context:
        fully_shard(model.fc1, mesh=mesh, placements=placements)
        fully_shard(model.fc2, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
    ref_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    torch.manual_seed(4321 + distributed_setup.rank)
    x = torch.randn(2, 8, device=device)

    def step(model, forward_only=False):
        if forward_only:
            return model(x)
        loss = model(x.detach().requires_grad_()).square().mean()
        loss.backward()
        return loss

    def run(model, forward_only=False):
        scope = (
            (_validation(context) if forward_only else iteration(context))
            if record_iteration
            else nullcontext()
        )
        with scope:
            return step(model, forward_only=forward_only)

    for _ in range(3):
        ref_optimizer.zero_grad()
        expected = baseline(x.detach().requires_grad_()).square().mean()
        expected.backward()
        for parameter in baseline.parameters():
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(distributed_setup.world_size)
        ref_optimizer.step()
        optimizer.zero_grad()
        actual = run(model)
        optimizer.step()
        torch.testing.assert_close(actual, expected)
    for _ in range(2):
        with torch.no_grad():
            torch.testing.assert_close(run(model, forward_only=True), baseline(x))
    if record_iteration:
        demands = context.iteration_order
        assert Phase.RECOMPUTE in [phase for _, phase in demands]
    else:
        assert not context.iteration_captured
    with _validation(context):
        for unit in (model, model.fc1, model.fc2):
            unit.unshard()
        try:
            for expected, actual in zip(baseline.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected)
        finally:
            for unit in (model.fc2, model.fc1, model):
                unit.reshard()


def test_replay_prefetch_uses_successor_compute_phase(context):
    """Backward and recompute successors need different materialization phases."""
    a, b = _Unit(context), _Unit(context)
    for _ in range(2):
        with iteration(context):
            b._unshard_parameter_groups.reset_mock()
            a.unshard(Phase.FORWARD)
            if context.iteration_captured:
                b._unshard_parameter_groups.assert_called_once_with(Phase.BACKWARD)
            b.unshard(Phase.BACKWARD)
            if context.iteration_captured:
                a._unshard_parameter_groups.assert_called_with(Phase.FORWARD)
            a.unshard(Phase.RECOMPUTE)


@pytest.mark.parametrize("active", [False, True])
def test_combined_hooks_record_phase_tagged_demands(context, active):
    """Combined 1F1B hooks participate in capture only inside an iteration scope."""
    module = _Unit(context)
    module.is_root = lambda: True
    forward = _make_unshard_forward_hook(module)
    backward = _make_unshard_backward_hook(module)
    with iteration(context) if active else nullcontext():
        for _ in range(2):
            forward(module, (), {})
            backward(module, ())
    assert list(context.iteration_order) == (
        [(ref(module), Phase.FORWARD), (ref(module), Phase.BACKWARD)] * 2 if active else []
    )
    assert context.allgather_stream.wait_stream.call_count == 2


def test_invalid_prefetch_mode_does_not_consume_order(context):
    """Invalid modes report their value without changing execution progress."""
    module = _Unit(context)
    context.prefetch_mode = "invalid"
    with pytest.raises(ValueError, match="Unsupported FSDP prefetch mode: 'invalid'"):
        _demand(context, module, Phase.FORWARD)
    assert context.iteration_order.position == 0
    assert len(context.iteration_order) == 0
