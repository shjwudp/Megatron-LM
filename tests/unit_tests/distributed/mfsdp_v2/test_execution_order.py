# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import gc
from unittest.mock import Mock
from weakref import WeakKeyDictionary, ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import (
    FsdpContext,
    FsdpModule,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


@pytest.mark.parametrize(
    'budget, expected',
    [
        (None, [[1], [0], [0], [2], []]),
        (0, [[], [], [], [], []]),
        (2, [[1, 0], [0, 0], [0, 2], [2], []]),
    ],
)
def test_prefetch_replays_occurrences(budget, expected):
    """Reused modules have occurrence-specific successors under every prefetch budget."""
    modules = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    prefetched = []
    for module in modules:
        module.phase = Phase.FORWARD
        # Prefetch passes the successor's compute phase; record only which module it targets.
        module._unshard_parameter_groups.side_effect = (
            lambda _phase, target=module: prefetched.append(target)
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = ExecutionOrder()
    for module in sequence:
        order.append((module, Phase.FORWARD))
    # A new pass must explicitly reset before replaying repeated occurrences.
    for _ in range(2):
        order.reset_cursor()
        for index, (module, targets) in enumerate(zip(sequence, expected)):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.position == index + 1


def test_demand_calls_validate_module_and_phase():
    """A rejected demand leaves the expected occurrence available for retry."""
    first, second = [Mock(spec=FsdpModule, phase=Phase.FORWARD) for _ in range(2)]
    order = ExecutionOrder([(first, Phase.FORWARD), (second, Phase.BACKWARD)])
    with pytest.raises(RuntimeError, match='diverged'):
        FsdpModule._prefetch_parameter_groups(second, order, 0)
    assert order.position == 0
    FsdpModule._prefetch_parameter_groups(first, order, 0)
    with pytest.raises(RuntimeError, match='diverged'):
        FsdpModule._prefetch_parameter_groups(second, order, 0)
    assert order.position == 1
    second.phase = Phase.BACKWARD
    FsdpModule._prefetch_parameter_groups(second, order, 0)
    with pytest.raises(RuntimeError, match='exceeded'):
        FsdpModule._prefetch_parameter_groups(second, order, 0)


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder((module, Phase.FORWARD) for module in [first, second, first, second])
    assert [module for module, _ in order] == [first, second, first, second]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    order.advance()
    with pytest.raises(RuntimeError, match="destroyed"):
        order[order.position]
    order.reset_cursor()
    first.phase = Phase.FORWARD
    with pytest.raises(RuntimeError, match="destroyed"):
        FsdpModule._prefetch_parameter_groups(first, order, None)
    assert order.position == 1


def test_explicit_reset_and_exhaustion():
    """Only reset starts another pass; lookahead and iteration never advance it."""
    module = nn.Module()
    a, b = (module, Phase.FORWARD), (module, Phase.BACKWARD)
    order = ExecutionOrder([a, b, a])
    for _ in range(2):
        order.reset_cursor()
        assert order.advance() == a
        assert order[order.position] == b
        assert order.position == 1
        assert list(order) == [a, b, a]
        assert order.advance() == b
        assert order.advance() == a
        assert list(order)[order.position :] == []
        with pytest.raises(RuntimeError, match='exceeded'):
            order.advance()
    order.reset_cursor(1)
    assert order.advance() == b
    empty = ExecutionOrder()
    assert list(empty) == []
    with pytest.raises(RuntimeError, match='exceeded'):
        empty.advance()


@pytest.mark.parametrize('roots', [(0, 0, 0), (0, 0, 1, 0, 1)])
def test_root_hooks_reset_preset_cursor(monkeypatch, roots):
    """Root entries restart each chunk's pass, including validation and recompute."""
    monkeypatch.setattr('torch.cuda.nvtx.range_push', lambda _: None)
    monkeypatch.setitem(FsdpModule.pre_forward.__globals__, '_is_in_backward', lambda: False)
    context = object.__new__(FsdpContext)
    chunks = [nn.Sequential(nn.Module()) for _ in range(2)]
    context._is_finalized = False
    context._registered_modules = [unit for root in chunks for unit in (root, root[0])]
    context._roots = WeakKeyDictionary()
    context._module_names = WeakKeyDictionary()
    context.execution_order = ExecutionOrder()
    context.allgather_stream = Mock()
    context.reduce_scatter_stream = Mock()
    context.current_stream = Mock()
    context.register_post_backward_hook = Mock()
    context.finalize()

    def unshard(unit, phase, prefetch=False):
        if prefetch:
            FsdpModule._prefetch_parameter_groups(unit, context.execution_order, None)

    for root in chunks:
        for unit in (root, root[0]):
            unit.context = context
            unit.is_root = Mock(return_value=unit is root)
            unit._nvtx_label = Mock(return_value='test')
            unit.num_parameter_elements = 1
            unit._unshard_parameter_groups = Mock()
            unit.unshard = lambda phase, prefetch=False, unit=unit: unshard(unit, phase, prefetch)

    # Grouped VPP calls and F/B interleaving both restart at a chunk's own offset.
    events = [(index, Phase.FORWARD) for index in roots]
    events += [
        (index, phase) for index in reversed(roots) for phase in (Phase.BACKWARD, Phase.FORWARD)
    ]
    for index, phase in events:
        root = chunks[index]
        for unit in (root, root[0]):
            unit.phase = Phase.RESTING
            if phase is Phase.FORWARD:
                FsdpModule.pre_forward(unit)
            else:
                FsdpModule.pre_backward(unit)
        position = context._roots[root][phase] + 2
        assert context.execution_order.position == position
        if phase is Phase.BACKWARD:
            # Checkpoint recomputation must leave the interrupted backward cursor alone.
            FsdpModule.pre_forward(root)
            assert context.execution_order.position == position
