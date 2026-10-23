# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import gc
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
    PrefetchMode,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
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
        module.context = SimpleNamespace(prefetch_mode=PrefetchMode.RECORD_REPLAY)
        # Prefetch passes the successor's compute phase; record only which module it targets.
        module._unshard_parameter_groups.side_effect = (
            lambda _phase, target=module: prefetched.append(target)
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = ExecutionOrder()
    for module in sequence:
        order.append((ref(module), Phase.FORWARD))
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        for index, (module, targets) in enumerate(zip(sequence, expected)):
            prefetched.clear()
            order.advance((ref(module), Phase.FORWARD))
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.position == index + 1


def test_unique_calls_resynchronize_and_repeated_calls_validate():
    """Unique VPP chunk calls select their position; ambiguous repeats stay strict."""
    chunk0, layer0, chunk1, layer1 = (nn.Module() for _ in range(4))
    order = ExecutionOrder(map(ref, [chunk0, layer0, chunk1, layer1]))
    # Grouped VPP repeats chunk 0 before switching to chunk 1, in one shared order.
    for chunk in (chunk0, chunk0, chunk0, chunk1, chunk1, chunk0):
        layer = layer0 if chunk is chunk0 else layer1
        assert order.advance(ref(chunk)) == ref(chunk)
        assert order[order.position]() is layer
        order.advance(ref(layer))
        if chunk is chunk0:
            assert order[order.position]() is chunk1
        else:
            assert order.position == len(order)

    order = ExecutionOrder(map(ref, [chunk0, layer0, chunk0]))
    assert order.advance(ref(chunk0)) == ref(chunk0)
    assert order[order.position]() is layer0
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(ref(chunk0))
    # A rejected repeated call must not consume the expected occurrence.
    order.advance(ref(layer0))
    assert order[order.position]() is chunk0
    assert order.advance(ref(chunk0)) == ref(chunk0)
    assert order.position == len(order)
    # Unique calls remain valid anchors even when the sequence also has repeats.
    order.advance(ref(layer0))
    assert order.advance(ref(chunk0)) == ref(chunk0)
    assert order.position == len(order)

    empty = ExecutionOrder()
    assert len(empty) == 0
    with pytest.raises(RuntimeError, match='diverged'):
        empty.advance(ref(chunk0))


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder(map(ref, [first, second, first, second]))
    assert [module_ref() for module_ref in order] == [first, second, first, second]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert [module_ref() for module_ref in order] == [first, None, first, None]
    order.advance(ref(first))
    assert order[order.position]() is None
    first.context = SimpleNamespace(prefetch_mode=PrefetchMode.RECORD_REPLAY)
    prefetch_order = ExecutionOrder((module_ref, Phase.FORWARD) for module_ref in order)
    prefetch_order.advance((ref(first), Phase.FORWARD))
    with pytest.raises(RuntimeError, match="destroyed"):
        FsdpModule._prefetch_parameter_groups(first, prefetch_order, None)
    assert prefetch_order.position == 1
    # A destroyed owner must not silently skip its occurrence.
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(ref(first))
