# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generic occurrence traversal independent of FSDP module lookup."""

import gc
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)


def test_explicit_reset_and_exhaustion():
    """Only reset starts another pass; lookahead and iteration never advance it."""
    a, b = nn.Module(), nn.Module()
    order = ExecutionOrder([a, b, a])
    for _ in range(2):
        order.reset_cursor()
        assert order.advance() is a
        assert order[order.position] is b
        assert order[order.position] is b
        assert list(order) == [a, b, a]
        assert order.advance() is b
        assert order.advance() is a
        assert list(order)[order.position :] == []
        with pytest.raises(RuntimeError, match='exceeded'):
            order.advance()
    order.reset_cursor(1)
    assert order.advance() is b
    empty = ExecutionOrder()
    assert list(empty) == []
    with pytest.raises(RuntimeError, match='exceeded'):
        empty.advance()


def test_iteration_and_weak_references():
    """Tuple payloads preserve occurrences while callers keep module ownership weak."""
    first, second = nn.Module(), nn.Module()
    a, b = (ref(first), 'forward'), (ref(second), 'backward')
    order = ExecutionOrder([a, b, a, b])
    assert list(order) == [a, b, a, b]
    assert order.advance() is a
    assert order[order.position] is b
    assert order[order.position + 1] is a

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert order.advance() is b
    module_ref, _ = b
    assert module_ref() is None
    assert order.position == 2 and len(order) == 4
