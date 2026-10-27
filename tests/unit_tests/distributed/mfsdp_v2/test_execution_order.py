# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import gc
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import capture_execution_order
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    _make_unshard_backward_hook,
    _make_unshard_forward_hook,
)
from megatron.core.models.common.mfsdp_iteration import mfsdp_execution_order


@pytest.fixture
def context():
    return SimpleNamespace(
        ensure_finalized=Mock(),
        forward_order=ExecutionOrder(),
        backward_order=ExecutionOrder(),
        _recording_execution_order=False,
        _execution_order_captured=False,
        allgather_stream=Mock(),
        current_stream=Mock(),
    )


@pytest.mark.parametrize(
    'budget, expected',
    [
        (None, [[1], [0], [0], [2], []]),
        (0, [[], [], [], [], []]),
        (2, [[1, 0], [0, 0], [0, 2], [2], []]),
    ],
)
def test_prefetch_replays_occurrences(context, budget, expected):
    """Reused modules have occurrence-specific successors under every prefetch budget."""
    modules = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    prefetched = []
    for module in modules:
        module.context = context
        module._unshard_parameter_groups.side_effect = lambda target=module: prefetched.append(
            target
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    backward = list(reversed(sequence))
    with capture_execution_order(context):
        for order, calls in ((context.forward_order, sequence), (context.backward_order, backward)):
            for module in calls:
                FsdpModule._prefetch_parameter_groups(module, order, budget)
    assert list(context.forward_order) == sequence
    assert list(context.backward_order) == backward
    assert not prefetched
    order = context.forward_order
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        for module, targets, successor in zip(sequence, expected, sequence[1:] + [None]):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.next_item() is successor

    for module, successor in zip(backward, backward[1:] + [None]):
        context.backward_order.advance(module)
        assert context.backward_order.next_item() is successor


@pytest.mark.parametrize('captured', [False, True], ids=['first-capture', 'recapture'])
def test_capture_restores_orders_on_failure(context, captured):
    """Nested captures are rejected; failed capture preserves the previous orders."""
    context._execution_order_captured = captured
    forward, backward = context.forward_order, context.backward_order
    with pytest.raises(ValueError, match='failed compute'):
        with capture_execution_order(context):
            with pytest.raises(RuntimeError, match='nest'):
                with capture_execution_order(context):
                    pass
            assert context._recording_execution_order
            raise ValueError('failed compute')
    assert context.forward_order is forward and context.backward_order is backward
    assert not context._recording_execution_order
    assert context._execution_order_captured is captured


def test_unique_calls_resynchronize_and_repeated_calls_validate():
    """Unique VPP chunk calls select their position; ambiguous repeats stay strict."""
    chunk0, layer0, chunk1, layer1 = (nn.Module() for _ in range(4))
    order = ExecutionOrder([chunk0, layer0, chunk1, layer1])
    # Grouped VPP repeats chunk 0 before switching to chunk 1, in one shared order.
    for chunk in (chunk0, chunk0, chunk0, chunk1, chunk1, chunk0):
        layer = layer0 if chunk is chunk0 else layer1
        order.advance(chunk)
        assert order.next_item() is layer
        order.advance(layer)
        assert order.next_item() is (chunk1 if chunk is chunk0 else None)

    order = ExecutionOrder([chunk0, layer0, chunk0])
    order.advance(chunk0)
    assert order.next_item() is layer0
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(chunk0)
    # A rejected repeated call must not consume the expected occurrence.
    order.advance(layer0)
    assert order.next_item() is chunk0
    order.advance(chunk0)
    assert order.next_item() is None
    # Unique calls remain valid anchors even when the sequence also has repeats.
    order.advance(layer0)
    order.advance(chunk0)
    assert order.next_item() is None

    empty = ExecutionOrder()
    assert not empty
    assert empty.next_item() is None
    with pytest.raises(RuntimeError, match='diverged'):
        empty.advance(chunk0)


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder([first, second, first, second])
    assert list(order) == [first, second, first, second]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert list(order) == [first, first]
    order.advance(first)
    assert order.next_item() is None
    # Traversal skips dead references, but the demand cursor must not skip a call.
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(first)


@pytest.mark.parametrize('combined', [False, True], ids=['vpp', 'vpp-combined-1f1b'])
@pytest.mark.parametrize(
    'events, expected',
    [
        (
            [
                ('forward', 0),
                ('forward', 1),
                ('forward', 0),
                ('forward', 1),
                ('forward', 2),
                ('forward', 3),
                ('backward', 3),
                ('backward', 2),
                ('forward', 2),
                ('forward', 3),
                ('backward', 3),
                ('backward', 2),
                ('backward', 1),
                ('backward', 0),
                ('backward', 1),
                ('backward', 0),
            ],
            [1, 0, 1, 2, 3, 2, 2, 3, 3, 2, 1, 0, 1, 0],
        )
    ],
)
def test_pipeline_capture_replay_and_validation(context, combined, events, expected):
    """Two VPP roots replay independently by direction; validation preserves both cursors."""
    units = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(4)]
    gathers = []
    for unit in units:
        unit.context = context
        unit.is_root.return_value = False
        unit._unshard_event = object()
        unit._nvtx_range.return_value = nullcontext()
        unit._schedule_policy = SimpleNamespace(
            forward_prefetch_size=None, backward_prefetch_size=None
        )
        unit._unshard_parameter_groups.side_effect = lambda owner=unit: gathers.append(owner)
        unit._prefetch_parameter_groups.side_effect = (
            lambda order, budget, owner=unit: FsdpModule._prefetch_parameter_groups(
                owner, order, budget
            )
        )
        unit.unshard.side_effect = lambda prefetch='none', owner=unit: FsdpModule.unshard(
            owner, prefetch
        )
    chunks = [SimpleNamespace(module=units[0]), SimpleNamespace(module=units[2])]

    @mfsdp_execution_order
    def schedule(*, model, forward_only=False, fail=False):
        context.allgather_stream.wait_stream.assert_called_with(context.current_stream.return_value)
        calls = [('forward', i) for i in range(4)] if forward_only else events
        prefetched = []
        for direction, index in calls:
            unit = units[index]
            before = len(gathers)
            if combined and direction == 'forward':
                _make_unshard_forward_hook(unit)(unit, (), {})
            elif combined:
                _make_unshard_backward_hook(unit)(unit, ())
            else:
                unit.unshard(prefetch=direction)
            assert gathers[before] is unit
            prefetched.extend(gathers[before + 1 :])
            if fail:
                raise ValueError('failed validation')
        return prefetched

    with capture_execution_order(context):
        assert schedule(model=chunks) == []
    assert context._execution_order_captured
    assert list(context.forward_order) == [
        units[i] for direction, i in events if direction == 'forward'
    ]
    assert list(context.backward_order) == [
        units[i] for direction, i in events if direction == 'backward'
    ]
    orders = context.forward_order, context.backward_order
    for fail in (False, True):
        assert schedule(model=chunks) == [units[i] for i in expected]
        if fail:
            with pytest.raises(ValueError, match='failed validation'):
                schedule(model=chunks, forward_only=True, fail=True)
        else:
            assert schedule(model=chunks, forward_only=True) == []
        assert context.forward_order is orders[0] and context.backward_order is orders[1]
        assert schedule(model=chunks) == [units[i] for i in expected]
