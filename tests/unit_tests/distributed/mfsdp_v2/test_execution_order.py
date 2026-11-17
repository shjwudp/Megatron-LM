# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for the weak event source and independent filtered snapshots."""

import gc
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionAction,
    ExecutionEvent,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


def test_shared_source_and_independent_filtered_snapshots():
    """Repeated events keep source IDs while snapshot consumers move independently."""
    module, following = nn.Module(), nn.Module()
    order = ExecutionOrder()
    repeated = ExecutionEvent(module, Phase.FORWARD, buffer_bytes=0, event_id=900)
    assert order.append(repeated) == 0
    assert order.append(ExecutionEvent(module, None, ExecutionAction.GRAD_ALLOC, 0)) == 1
    assert order.append(repeated) == 2
    assert order.append(ExecutionEvent(following, Phase.BACKWARD)) == 3
    view = order.view(actions=[ExecutionAction.UNSHARD])
    independent = order.view(actions=[ExecutionAction.UNSHARD])
    assert view.source is independent.source is order
    assert order.append(ExecutionEvent(module, Phase.BACKWARD)) == 4
    assert [event.event_id for event in view] == [0, 2, 3]
    assert list(view) == list(independent)
    assert [event.event_id for event in order] == [0, 1, 2, 3, 4]
    assert order.seek(0).buffer_bytes == 0

    assert view.seek(ExecutionEvent(module, Phase.FORWARD, buffer_bytes=999))
    view.advance()
    assert view.seek(repeated) and view.peek().event_id == 2
    assert independent.position == 0 and independent.peek().event_id == 0
    assert view.peek(-1).event_id == 0
    assert [event.event_id for event in view.successors()] == [3]
    view.advance()
    assert view.peek(1) is None
    view.advance()
    assert view.position == 0
    assert not view.seek(ExecutionEvent(following, Phase.FORWARD))
    assert view.position == 0


def test_weak_source_preserves_expired_slots_and_empty_view_errors():
    """Expired modules leave stable ID slots, and empty views cannot advance."""
    first, middle, last = nn.Module(), nn.Module(), nn.Module()
    order = ExecutionOrder(
        [ExecutionEvent(module, Phase.FORWARD) for module in (first, middle, last)]
    )
    empty = order.view(actions=[])
    assert empty.peek() is None
    with pytest.raises(RuntimeError, match="empty"):
        empty.advance()
    view = order.view(actions=[ExecutionAction.UNSHARD])
    middle_ref = ref(middle)
    del middle
    gc.collect()
    assert middle_ref() is None
    assert len(order) == len(view) == 3
    assert order.seek(1) is None and order.seek(2).module is last
    assert [event.event_id for event in order] == [0, 2]
    assert [event.event_id for event in view] == [0, 2]
    assert list(view.successors()) == []
    view.advance()
    assert view.peek() is None
    view.advance()
    assert view.peek().event_id == 2
    assert order.append(ExecutionEvent(first, Phase.BACKWARD)) == 3
