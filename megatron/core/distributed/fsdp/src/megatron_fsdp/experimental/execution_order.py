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

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Generic, Literal, TypeVar
from weakref import WeakKeyDictionary, ref

if TYPE_CHECKING:
    from .module import FsdpModule

T = TypeVar("T")


class ModuleExecutionOrder(Generic[T]):
    """Look up unique modules and validate repeated module-call occurrences.

    A module appearing once can unambiguously select its position, preserving
    construction-order prefetch for reordered or repeated demand calls. Modules
    appearing more than once must match the next occurrence at the cursor.
    Prefetch lookahead never advances that cursor.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create a weak execution sequence, preserving repeated occurrences."""
        self._items: list[ref[T]] = []
        # An index identifies a unique item; None marks an ambiguous repeated item.
        self._unique_index: WeakKeyDictionary[T, int | None] = WeakKeyDictionary()
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:
        """Append an occurrence and mark previously seen items as repeated."""
        self._unique_index[item] = None if item in self._unique_index else len(self._items)
        self._items.append(ref(item))

    def __iter__(self) -> Iterator[T]:
        """Iterate over live occurrences in execution order."""
        for item_ref in self._items:
            item = item_ref()
            if item is not None:
                yield item

    def advance(self, item: T) -> None:
        """Locate a unique demand call or consume the next expected repeated call."""
        index = self._unique_index.get(item)
        if index is None:
            if not self._items or self._items[self._position]() is not item:
                raise RuntimeError("FSDP module calls diverged from the expected prefetch order.")
            index = self._position
        self._position = (index + 1) % len(self._items)

    def next_item(self, offset: int = 1) -> T | None:
        """Look ahead from the last consumed occurrence without moving the cursor."""
        if not self._items:
            return None
        index = (self._position - 1) % len(self._items)
        next_index = index + offset
        if next_index >= len(self._items):
            return None
        return self._items[next_index]()


class ExecutionOrder(Generic[T]):
    """Consume ordered occurrences with an explicitly reset execution cursor.

    Repeated items are ordinary occurrences. The caller validates each item
    returned by advance(); lookahead never changes execution progress. Callers
    own item lifetimes and can store weak references in their payloads.
    """

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[T] = list(items)
        self._position = 0

    def append(self, item: T) -> None:
        """Append one occurrence."""
        self._items.append(item)

    def reset_cursor(self, position: int = 0) -> None:
        """Reset traversal position without changing the recorded sequence."""
        self._position = position

    def __len__(self) -> int:
        return len(self._items)

    @property
    def position(self) -> int:
        """Number of occurrences consumed in the current traversal."""
        return self._position

    def __iter__(self) -> Iterator[T]:
        """Iterate over all occurrences without changing execution progress."""
        return iter(self._items)

    def advance(self) -> T:
        """Consume the next occurrence without lookup or automatic wrapping."""
        if self._position >= len(self._items):
            raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
        item = self._items[self._position]
        self._position += 1
        return item

    def __getitem__(self, index: int) -> T:
        """Read an occurrence without changing execution progress."""
        return self._items[index]


ExecutionPhase = Literal["forward", "backward", "recompute"]
ExecutionEvent = tuple[ref["FsdpModule"], ExecutionPhase]


class IterationOrder(ExecutionOrder[ExecutionEvent]):
    """Select static, recorded, or demand-only prefetch for a shared context.

    ``mode`` selects the order source. In ``record_replay`` mode, ``captured``
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
        self.mode: Literal["static", "record_replay", "demand"] = "static"

    def append(self, item: ExecutionEvent) -> None:
        """Append an occurrence before the recorded sequence is complete."""
        assert not self.captured, "Cannot append to an already captured order."
        super().append(item)

    @property
    def recording(self) -> bool:
        return self.active and self.mode == "record_replay" and not self.captured
