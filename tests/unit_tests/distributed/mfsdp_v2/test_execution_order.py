# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for the weak event source and independent filtered execution cursors."""

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


def test_cursor_wraps_across_phases():
    """One cursor follows the complete trace, including adjacent repeated calls."""
    first, second = nn.Module(), nn.Module()
    events = [
        ExecutionEvent(first, Phase.FORWARD),
        ExecutionEvent(first, Phase.FORWARD),
        ExecutionEvent(second, Phase.FORWARD),
        ExecutionEvent(first, Phase.BACKWARD),
        ExecutionEvent(second, Phase.BACKWARD),
        ExecutionEvent(first, Phase.BACKWARD),
    ]
    order = ExecutionOrder(events)
    view = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    for _ in range(2):
        for index, event in enumerate(events):
            assert view.peek() == event
            assert list(view.successors()) == events[index + 1 :]
            assert view.peek() == event
            view.advance()
        assert view.position == 0


def test_empty_order_can_be_populated():
    """Recording does not move the cursor, so replay starts at the first event."""
    order = ExecutionOrder()
    view = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    assert view.peek() is None
    with pytest.raises(RuntimeError, match="empty"):
        view.advance()
    event = ExecutionEvent(nn.Module(), Phase.FORWARD)
    order.append(event)
    assert view.peek() == event
    for _ in range(2):
        assert list(view.successors()) == []
        view.advance()
        assert view.peek() == event


def test_order_does_not_keep_modules_alive():
    """The sequence holds weak references and iteration skips dead modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder(
        [ExecutionEvent(first, Phase.FORWARD), ExecutionEvent(second, Phase.FORWARD)]
    )
    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert list(order) == [ExecutionEvent(first, Phase.FORWARD)]
    assert list(order.view(actions=[ExecutionAction.MODULE_ENTRY]).successors()) == []


def test_global_ids_are_append_positions_including_repeated_events():
    """An incoming ID cannot renumber an occurrence or its filtered view."""
    module = nn.Module()
    order = ExecutionOrder()
    repeated = ExecutionEvent(module, Phase.FORWARD, event_id=900)
    assert order.append(repeated) == 0
    assert order.append(ExecutionEvent(module, None, ExecutionAction.GRAD_ALLOC)) == 1
    assert order.append(repeated) == 2
    assert repeated.event_id == 900
    assert [event.event_id for event in order] == [0, 1, 2]
    assert order.seek(0) == repeated == order.seek(2)
    assert [order.seek(index).event_id for index in (2, 0, 1)] == [2, 0, 1]
    assert order.seek(-1) is None
    assert order.seek(3) is None
    assert order.view(actions=[ExecutionAction.MODULE_ENTRY]).event_ids == (0, 2)


def test_views_filter_actions_without_sharing_cursors():
    """Each consumer keeps local positions while resolving the same global IDs."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder(
        [
            ExecutionEvent(first, Phase.FORWARD),
            ExecutionEvent(first, Phase.BACKWARD, ExecutionAction.GRAD_COPY_IN, 0),
            ExecutionEvent(second, Phase.FORWARD),
            ExecutionEvent(first, Phase.BACKWARD, ExecutionAction.GRAD_REDUCE, 0),
            ExecutionEvent(first, Phase.BACKWARD),
        ]
    )
    prefetch = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    other_prefetch = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    gradients = order.view(actions=[ExecutionAction.GRAD_COPY_IN, ExecutionAction.GRAD_REDUCE])
    assert prefetch.source is other_prefetch.source is gradients.source is order
    assert prefetch.event_ids == (0, 2, 4)
    assert gradients.event_ids == (1, 3)
    prefetch.advance()
    gradients.advance()
    assert order.seek(0).event_id == 0
    assert prefetch.position == 1 and prefetch.peek().event_id == 2
    assert other_prefetch.position == 0 and other_prefetch.peek().event_id == 0
    assert gradients.position == 1 and gradients.peek().event_id == 3
    assert [event.event_id for event in prefetch.successors()] == [4]
    assert [event.event_id for event in prefetch] == [0, 2, 4]
    assert prefetch.position == 1


def test_empty_view_tracks_later_appends_without_consuming_them():
    """A view created before recording sees newly appended matching occurrences."""
    module = nn.Module()
    order = ExecutionOrder()
    view = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    assert view.event_ids == () and view.peek() is None
    with pytest.raises(RuntimeError, match="empty"):
        view.advance()
    order.append(ExecutionEvent(module, None, ExecutionAction.GRAD_ALLOC))
    assert len(view) == 0
    order.append(ExecutionEvent(module, Phase.FORWARD))
    assert view.event_ids == (1,) and view.peek().event_id == 1
    order.append(ExecutionEvent(module, Phase.FORWARD, ExecutionAction.RESHARD))
    order.append(ExecutionEvent(module, Phase.BACKWARD))
    assert view.event_ids == (1, 3)
    assert view.position == 0
    view.advance()
    assert view.peek().event_id == 3


def test_action_filter_is_snapshotted_and_empty_selection_stays_empty():
    """A view keeps its action filter when the caller changes the original list."""
    module = nn.Module()
    order = ExecutionOrder()
    actions = [ExecutionAction.UNSHARD, ExecutionAction.RESHARD]
    weights = order.view(actions=actions)
    empty = order.view(actions=[])
    actions.clear()
    for action in ExecutionAction:
        order.append(ExecutionEvent(module, Phase.FORWARD, action))
    assert [event.action for event in weights] == [ExecutionAction.UNSHARD, ExecutionAction.RESHARD]
    assert len(empty) == 0 and empty.event_ids == () and empty.peek() is None
    assert not empty.seek(ExecutionEvent(module, Phase.FORWARD))


def test_filtered_seek_preserves_occurrences_and_wraps_once():
    """Demand lookup starts at the cursor and wraps without consuming a match."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder(
        [
            ExecutionEvent(first, Phase.FORWARD),
            ExecutionEvent(first, Phase.FORWARD, ExecutionAction.UNSHARD),
            ExecutionEvent(second, Phase.FORWARD),
            ExecutionEvent(first, Phase.FORWARD),
            ExecutionEvent(first, Phase.BACKWARD),
        ]
    )
    view = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    target = ExecutionEvent(first, Phase.FORWARD)
    assert view.seek(target) and view.position == 0
    view.advance()
    assert view.seek(target) and view.position == 2 and view.peek().event_id == 3
    assert view.seek(target) and view.position == 2
    view.advance()
    assert view.seek(target) and view.position == 0
    assert not view.seek(ExecutionEvent(nn.Module(), Phase.FORWARD))
    assert view.position == 0


def test_seek_distinguishes_action_phase_and_owner_local_group():
    """Seek matches module, phase, action and group, ignoring byte counts and IDs."""
    module = nn.Module()
    order = ExecutionOrder(
        [
            ExecutionEvent(module, Phase.BACKWARD, ExecutionAction.GRAD_COPY_IN, 0),
            ExecutionEvent(module, Phase.BACKWARD, ExecutionAction.GRAD_COPY_IN, 1),
            ExecutionEvent(module, Phase.BACKWARD, ExecutionAction.GRAD_REDUCE, 0),
            ExecutionEvent(module, None, ExecutionAction.GRAD_REDUCE, 0),
        ]
    )
    view = order.view(actions=[ExecutionAction.GRAD_COPY_IN, ExecutionAction.GRAD_REDUCE])
    for expected_id, (phase, action, group_index) in enumerate(
        [
            (Phase.BACKWARD, ExecutionAction.GRAD_COPY_IN, 0),
            (Phase.BACKWARD, ExecutionAction.GRAD_COPY_IN, 1),
            (Phase.BACKWARD, ExecutionAction.GRAD_REDUCE, 0),
            (None, ExecutionAction.GRAD_REDUCE, 0),
        ]
    ):
        query = ExecutionEvent(
            module, phase, action, group_index, buffer_bytes=999, allocated_bytes=888, event_id=777
        )
        assert view.seek(query)
        assert view.peek().event_id == expected_id
    assert not view.seek(ExecutionEvent(module, None, ExecutionAction.GRAD_REDUCE, group_index=1))
    assert view.position == 3


def test_filtered_peek_and_successors_do_not_wrap():
    """Only advance wraps; lookahead respects the filtered sequence boundary."""
    module = nn.Module()
    order = ExecutionOrder(
        [
            ExecutionEvent(module, Phase.FORWARD),
            ExecutionEvent(module, None, ExecutionAction.GRAD_ALLOC),
            ExecutionEvent(module, Phase.BACKWARD),
        ]
    )
    view = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    assert view.peek(-1) is None and view.peek(2) is None
    assert [event.event_id for event in view.successors()] == [2]
    view.advance()
    assert view.peek(-1).event_id == 0
    assert view.peek(1) is None and list(view.successors()) == []
    view.advance()
    assert view.position == 0 and view.peek().event_id == 0


def test_dead_middle_event_keeps_ids_and_stops_prefetch_lookahead():
    """Views retain no modules and cannot prefetch past an expired occurrence."""
    first, middle, last = nn.Module(), nn.Module(), nn.Module()
    order = ExecutionOrder(
        [ExecutionEvent(module, Phase.FORWARD) for module in (first, middle, last)]
    )
    view = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    middle_ref = ref(middle)
    del middle
    gc.collect()
    assert middle_ref() is None
    assert len(order) == len(view) == 3
    assert view.event_ids == (0, 1, 2)
    assert order.seek(1) is None and order.seek(2).module is last
    assert [event.event_id for event in order] == [0, 2]
    assert [event.event_id for event in view] == [0, 2]
    assert list(view.successors()) == []
    view.advance()
    assert view.peek() is None
    view.advance()
    assert view.peek().event_id == 2
    assert order.append(ExecutionEvent(first, Phase.BACKWARD)) == 3
    assert view.event_ids == (0, 1, 2, 3)


def test_sealing_freezes_the_source_but_not_view_cursors():
    """Sealing blocks appends while each view keeps its own cursor."""
    module = nn.Module()
    order = ExecutionOrder(
        [ExecutionEvent(module, Phase.FORWARD), ExecutionEvent(module, Phase.BACKWARD)]
    )
    view = order.view(actions=[ExecutionAction.MODULE_ENTRY])
    assert not order.is_sealed
    order.seal()
    assert order.is_sealed
    with pytest.raises(RuntimeError):
        order.append(ExecutionEvent(module, Phase.FORWARD))
    assert len(order) == 2 and view.event_ids == (0, 1)
    view.advance()
    assert view.peek().event_id == 1
    assert view.seek(ExecutionEvent(module, Phase.FORWARD)) and view.position == 0
    assert order.view(actions=[ExecutionAction.MODULE_ENTRY]).event_ids == (0, 1)


@pytest.mark.parametrize("value", [None, 0, 2**60])
def test_optional_observations_preserve_missing_zero_and_large_values(value):
    """Missing samples remain distinct from real zero and retain integer precision."""
    module = nn.Module()
    order = ExecutionOrder()
    order.append(
        ExecutionEvent(
            module, None, ExecutionAction.GRAD_ALLOC, buffer_bytes=value, allocated_bytes=value
        )
    )
    event = order.seek(0)
    assert event.buffer_bytes == value and event.allocated_bytes == value
    assert event == ExecutionEvent(module, None, ExecutionAction.GRAD_ALLOC)
    with pytest.raises(AttributeError):
        event.allocated_bytes = 1


@pytest.mark.parametrize("field", ["group_index", "buffer_bytes", "allocated_bytes"])
@pytest.mark.parametrize("value", [-1, True, False])
def test_invalid_metadata_does_not_append_or_consume_an_id(field, value):
    """Rejected observations cannot leave holes in the source event numbering."""
    module = nn.Module()
    order = ExecutionOrder()
    with pytest.raises((TypeError, ValueError)):
        order.append(ExecutionEvent(module, None, ExecutionAction.GRAD_ALLOC, **{field: value}))
    assert len(order) == 0 and order.seek(0) is None
    assert order.append(ExecutionEvent(module, None, ExecutionAction.GRAD_ALLOC)) == 0
