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

"""Weakly held construction or recorded module execution sequences."""

import enum
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Generic, TypeVar
from weakref import ref

if TYPE_CHECKING:
    from .module import FsdpModule

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Look up unique modules and validate repeated module-call occurrences.

    A module appearing once can unambiguously select its position, preserving
    construction-order prefetch for reordered or repeated demand calls. Modules
    appearing more than once must match the next occurrence at the cursor.
    Prefetch lookahead never advances that cursor.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[T] = []
        # An index identifies a unique item; None marks an ambiguous repeated item.
        self._unique_index: dict[T, int | None] = {}
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:
        """Append an occurrence and mark previously seen items as repeated."""
        self._unique_index[item] = None if item in self._unique_index else len(self._items)
        self._items.append(item)

    def __iter__(self) -> Iterator[T]:
        """Iterate over occurrences in execution order."""
        return iter(self._items)

    def reset_cursor(self, position: int = 0) -> None:
        """Reset traversal without changing the recorded sequence."""
        self._position = position

    def __len__(self) -> int:
        return len(self._items)

    @property
    def position(self) -> int:
        """Index of the next occurrence; len(order) means traversal is exhausted."""
        return self._position

    def __getitem__(self, index: int) -> T:
        """Read an occurrence without advancing the cursor."""
        return self._items[index]

    def advance(self, item: T | None = None) -> T:
        """Locate a unique demand call or consume the next expected repeated call.

        With no item, consume the next occurrence without lookup.
        """
        if item is not None:
            index = self._unique_index.get(item)
            if index is None:
                if not self._items or self._items[self._position % len(self._items)] != item:
                    raise RuntimeError(
                        "FSDP module calls diverged from the expected prefetch order."
                    )
                index = self._position % len(self._items)
            self._position = index
        if self._position >= len(self._items):
            raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
        result = self._items[self._position]
        self._position += 1
        return result


class ExecutionPhase(enum.Enum):
    """Compute represented by an occurrence in the execution order."""

    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    RECOMPUTE = enum.auto()


class PrefetchMode(enum.Enum):
    """Source of the parameter prefetch order."""

    STATIC = enum.auto()
    RECORD_REPLAY = enum.auto()
    DEMAND = enum.auto()


ExecutionEvent = tuple[ref["FsdpModule"], ExecutionPhase]


class IterationOrder(ExecutionOrder[ExecutionEvent]):
    """Select static, recorded, or demand-only prefetch for a shared context.

    ``mode`` selects the order source. In ``PrefetchMode.RECORD_REPLAY``, ``captured``
    distinguishes initial recording from replay of the completed sequence.
    ``active`` tracks the iteration scope independently of the selected strategy.
    State lives on FsdpContext because autograd hooks can run on another thread.
    The scheduler owns iteration boundaries and must explicitly recapture when
    the training schedule changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured = False
        self.active = False
        self.mode = PrefetchMode.STATIC

    def append(self, item: ExecutionEvent) -> None:
        """Append an occurrence before the recorded sequence is complete."""
        assert not self.captured, "Cannot append to an already captured order."
        super().append(item)

    @property
    def recording(self) -> bool:
        return self.active and self.mode is PrefetchMode.RECORD_REPLAY and not self.captured
