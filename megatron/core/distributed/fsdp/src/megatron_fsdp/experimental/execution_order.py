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

"""Weakly held execution occurrences with a pure replay cursor."""

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar
from weakref import ref

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Record occurrences and replay them without identity lookup or automatic wrapping."""

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create an execution sequence, preserving repeated occurrences."""
        self._items: list[ref[T]] = [ref(item) for item in items]
        self._position = 0

    def append(self, item: T) -> None:
        """Append one observed execution occurrence."""
        self._items.append(ref(item))

    def __iter__(self) -> Iterator[T]:
        """Iterate over live recorded occurrences."""
        for item_ref in self._items:
            item = item_ref()
            if item is not None:
                yield item

    def reset(self) -> None:
        """Restart replay at the iteration boundary."""
        self._position = 0

    def is_complete(self) -> bool:
        """Return whether replay consumed every recorded occurrence."""
        return self._position == len(self._items)

    def advance(self) -> T:
        """Consume and return the next occurrence without searching by module identity."""
        if self.is_complete():
            raise RuntimeError("FSDP execution order is exhausted.")
        item = self._items[self._position]()
        if item is None:
            raise RuntimeError("A recorded FSDP module has been deleted.")
        self._position += 1
        return item

    def next_item(self, offset: int = 1) -> T | None:
        """Look ahead from the last consumed occurrence without moving the cursor."""
        next_index = self._position + offset - 1
        if next_index >= len(self._items):
            return None
        return self._items[next_index]()
