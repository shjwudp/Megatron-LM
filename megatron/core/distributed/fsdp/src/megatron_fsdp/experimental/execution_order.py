# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One weak lifecycle event source with independent filtered cursors."""

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple
from weakref import ref

from torch import nn

from .phase import Phase


class ExecutionAction(enum.Enum):
    """Weight requests and weight/gradient lifecycle boundaries.

    UNSHARD records a weight request, including reuse of already gathered weights.

    These describe the unoptimized schedule. A future plan may omit a boundary,
    but must keep that decision separate from the recorded event source.
    """

    UNSHARD = enum.auto()
    RESHARD = enum.auto()
    GRAD_ALLOC = enum.auto()
    GRAD_COPY_IN = enum.auto()
    GRAD_REDUCE = enum.auto()
    GRAD_RELEASE = enum.auto()


@dataclass(frozen=True)
class ExecutionEvent:
    """An event with optional byte observations.

    ``event_id`` is assigned by the owning ExecutionOrder, not a microbatch or
    module ID. Byte observations and IDs do not affect event matching. Callers
    supply the observations. Missing measurements stay None, distinct from a real zero.
    """

    module: nn.Module
    phase: Phase | None
    action: ExecutionAction = ExecutionAction.UNSHARD
    buffer_bytes: int | None = field(default=None, compare=False)
    allocated_bytes: int | None = field(default=None, compare=False)
    event_id: int | None = field(default=None, compare=False)


class _StoredEvent(NamedTuple):
    module: ref[nn.Module]
    phase: Phase | None
    action: ExecutionAction
    buffer_bytes: int | None
    allocated_bytes: int | None


class ExecutionOrder:
    """Own an append-only weak event source, without a consumer cursor.

    IDs are zero-based append indices local to this source. Expired modules do
    not remove or renumber events. Plans refer to IDs without editing the source.
    """

    def __init__(self, items: Iterable[ExecutionEvent] = ()) -> None:
        """Create a source that records repeated events separately."""
        self._items: list[_StoredEvent] = []
        for item in items:
            self.append(item)

    def append(self, item: ExecutionEvent) -> int:
        """Append an event and return its assigned ID without advancing any view."""
        for name in ("buffer_bytes", "allocated_bytes"):
            value = getattr(item, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None.")
        event_id = len(self._items)
        self._items.append(
            _StoredEvent(
                ref(item.module), item.phase, item.action, item.buffer_bytes, item.allocated_bytes
            )
        )
        return event_id

    def __len__(self) -> int:
        """Number of events, including those whose modules have expired."""
        return len(self._items)

    def __iter__(self) -> Iterator[ExecutionEvent]:
        """Iterate over live events without changing any view cursor."""
        for event_id in range(len(self._items)):
            if (event := self.seek(event_id)) is not None:
                yield event

    def seek(self, event_id: int) -> ExecutionEvent | None:
        """Read an event by ID without moving a cursor.

        Return None if the ID is absent or its module has expired. The returned
        event holds a strong reference to its module.
        """
        if not 0 <= event_id < len(self._items):
            return None
        item = self._items[event_id]
        module = item.module()
        if module is None:
            return None
        return ExecutionEvent(
            module, item.phase, item.action, item.buffer_bytes, item.allocated_bytes, event_id
        )

    def view(self, *, actions: Iterable[ExecutionAction]) -> "ExecutionOrderView":
        """Select actions without copying event records or sharing cursor state."""
        return ExecutionOrderView(self, actions=actions)


class ExecutionOrderView:
    """An action-filtered index into one source, with its own local cursor.

    Views select events present at creation. Cursor positions count only selected
    events; returned events keep their IDs from the source. Only IDs are stored,
    so neither source nor view keeps modules alive.
    """

    def __init__(self, source: ExecutionOrder, *, actions: Iterable[ExecutionAction]) -> None:
        """Create a view of ``source`` with its own cursor."""
        self._source = source
        actions = frozenset(actions)
        self._event_ids = tuple(
            event_id for event_id, item in enumerate(source._items) if item.action in actions
        )
        self._position = 0

    @property
    def source(self) -> ExecutionOrder:
        """The sole owner of this view's event records."""
        return self._source

    @property
    def position(self) -> int:
        """Local index of the next event to consume, distinct from its event ID."""
        return self._position

    def __len__(self) -> int:
        """Number of selected events, including those with expired modules."""
        return len(self._event_ids)

    def __iter__(self) -> Iterator[ExecutionEvent]:
        """Iterate over live selected events without advancing this cursor."""
        for index in range(len(self)):
            if (event := self._source.seek(self._event_ids[index])) is not None:
                yield event

    def peek(self, offset: int = 0) -> ExecutionEvent | None:
        """Read relative to this cursor without wrapping or advancing."""
        index = self._position + offset
        return self._source.seek(self._event_ids[index]) if 0 <= index < len(self) else None

    def successors(self) -> Iterator[ExecutionEvent]:
        """Look ahead without wrapping, stopping at an expired module reference."""
        offset = 1
        while (event := self.peek(offset)) is not None:
            yield event
            offset += 1

    def advance(self) -> None:
        """Consume one occurrence and automatically wrap at the end of the trace."""
        if not self._event_ids:
            raise RuntimeError("Cannot advance an empty FSDP execution order.")
        self._position = (self._position + 1) % len(self._event_ids)

    def seek(self, event: ExecutionEvent) -> bool:
        """Move the cursor to the next matching module/phase/action.

        Search from the current position and wrap at most once. Repeated events
        are kept. IDs and byte observations do not affect matching, so demand
        calls can recover the cursor position in an interleaved schedule.
        """
        size = len(self)
        for offset in range(size):
            index = (self._position + offset) % size
            candidate = self._source._items[self._event_ids[index]]
            if (
                candidate.module() is event.module
                and candidate.phase is event.phase
                and candidate.action is event.action
            ):
                self._position = index
                return True
        return False
