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
from typing import Generic, TypeVar, cast, overload
from weakref import WeakKeyDictionary, ref

T = TypeVar("T")
_NO_ITEM = object()


class ExecutionOrder(Generic[T]):
    """Traverse occurrences while preserving the original weak module lookup API.

    Construction orders keep ``module_lookup=True`` and use ``advance(module)``.
    Recorded orders set it to False, own their payload lifetimes, and consume
    occurrences with ``advance()``. Lookahead never advances either cursor.
    """

    def __init__(self, items: Iterable[T] = (), *, module_lookup: bool = True) -> None:
        """Create a weak module order or a generic occurrence sequence."""
        self._items: list[T | ref[T]] = []
        # Keep the existing lookup until all callers have iteration boundaries.
        # None selects generic payload storage without an item index.
        self._unique_index: WeakKeyDictionary[T, int | None] | None = (
            WeakKeyDictionary() if module_lookup else None
        )
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:
        """Append an occurrence, preserving weak ownership for module orders."""
        if self._unique_index is None:
            self._items.append(item)
        else:
            self._unique_index[item] = None if item in self._unique_index else len(self._items)
            self._items.append(ref(item))

    def __iter__(self) -> Iterator[T]:
        """Visit generic occurrences or live modules without moving the cursor."""
        if self._unique_index is None:
            yield from cast(list[T], self._items)
        else:
            for item_ref in cast(list[ref[T]], self._items):
                item = item_ref()
                if item is not None:
                    yield item

    def reset_cursor(self, position: int = 0) -> None:
        """Reset traversal position without changing the recorded sequence."""
        self._position = position

    def __len__(self) -> int:
        return len(self._items)

    @property
    def position(self) -> int:
        """Index of the next occurrence; len(order) means traversal is exhausted."""
        return self._position

    def __getitem__(self, index: int) -> T:
        """Read an occurrence without advancing or skipping a destroyed module."""
        item = self._items[index]
        if self._unique_index is None:
            return cast(T, item)
        owner = cast(ref[T], item)()
        if owner is None:
            raise RuntimeError("A module in the FSDP execution order was destroyed.")
        return owner

    @overload
    def advance(self) -> T: ...

    @overload
    def advance(self, item: T) -> None: ...

    def advance(self, item: T | object = _NO_ITEM) -> T | None:
        """Consume the next occurrence, or locate a module for existing callers."""
        if item is _NO_ITEM:
            if self._position >= len(self._items):
                raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
            result = self[self._position]
            self._position += 1
            return result
        if self._unique_index is None:
            raise TypeError("Item lookup is unavailable for generic occurrence orders.")
        index = self._unique_index.get(cast(T, item))
        if index is None:
            if (
                not self._items
                or cast(ref[T], self._items[self._position % len(self._items)])() is not item
            ):
                raise RuntimeError("FSDP module calls diverged from the expected prefetch order.")
            index = self._position % len(self._items)
        self._position = index + 1
        return None
