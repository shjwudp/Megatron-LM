# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import gc
from unittest.mock import Mock
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


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
        module._unshard_parameter_groups.side_effect = lambda target=module: prefetched.append(
            target
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = ExecutionOrder()
    for module in sequence:
        order.append(module)
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        order.reset()
        for module, targets, successor in zip(sequence, expected, sequence[1:] + [None]):
            prefetched.clear()
            assert order.advance() is module
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.next_item() is successor


def test_cursor_is_explicit_and_checks_completion():
    """The caller owns iteration resets; exhausted replay never silently wraps."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder([first, second, first])
    assert not order.is_complete()
    assert order.advance() is first
    assert order.next_item() is second
    assert order.next_item(2) is first
    assert order.advance() is second
    assert order.advance() is first
    assert order.is_complete()
    assert order.next_item() is None
    with pytest.raises(RuntimeError, match='exhausted'):
        order.advance()
    order.reset()
    assert order.advance() is first


def test_order_does_not_retain_modules():
    """Inspection skips dead references; replay cannot silently skip an occurrence."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder([first, second, first])
    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert list(order) == [first, first]
    assert order.advance() is first
    with pytest.raises(RuntimeError, match='deleted'):
        order.advance()
