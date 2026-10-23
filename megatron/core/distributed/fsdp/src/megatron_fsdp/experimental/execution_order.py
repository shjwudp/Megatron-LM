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

"""Generic execution occurrences; callers own payload lifetimes."""

from collections.abc import Hashable, Iterable, Iterator
from typing import Generic, TypeVar, cast, overload

T = TypeVar("T", bound=Hashable)
_NO_ITEM = object()


class ExecutionOrder(Generic[T]):
    """Traverse hashable occurrences without advancing during indexed lookahead."""

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Preserve every occurrence, including repeated items."""
        self._items: list[T] = []
        # Static callers still locate items until iteration scopes replace lookup.
        self._unique_index: dict[T, int | None] = {}
        self._position = 0
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:
        """Append an occurrence and mark repeated items as ambiguous."""
        self._unique_index[item] = None if item in self._unique_index else len(self._items)
        self._items.append(item)

    def __iter__(self) -> Iterator[T]:
        """Iterate over stored values without moving the cursor."""
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

    @overload
    def advance(self) -> T: ...

    @overload
    def advance(self, item: T) -> None: ...

    def advance(self, item: T | object = _NO_ITEM) -> T | None:
        """Consume the next occurrence, or locate an item for static callers."""
        if item is _NO_ITEM:
            if self._position >= len(self._items):
                raise RuntimeError("Calls exceeded the execution order; reset its cursor first.")
            result = self._items[self._position]
            self._position += 1
            return result
        index = self._unique_index.get(cast(T, item))
        if index is None:
            if not self._items or self._items[self._position % len(self._items)] != item:
                raise RuntimeError("FSDP module calls diverged from the expected prefetch order.")
            index = self._position % len(self._items)
        self._position = index + 1
        return None
