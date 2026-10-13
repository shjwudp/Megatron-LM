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

"""Weakly held module occurrences in runtime execution order."""

from collections.abc import Iterator
from typing import Generic, TypeVar
from weakref import ref

T = TypeVar("T")


class ExecutionOrder(Generic[T]):
    """Record repeated calls and look ahead from the current occurrence.

    Unlike an identity index, the demand-call cursor distinguishes repeated
    uses of the same module. Prefetch lookahead never advances that cursor.
    """

    def __init__(self) -> None:
        """Create an empty execution sequence."""
        self._items: list[ref[T]] = []
        self._position = 0

    def append(self, item: T) -> None:
        """Record one call, including repeated calls to the same item."""
        self._items.append(ref(item))

    def clear(self) -> None:
        """Discard an interrupted recording and reset the demand-call cursor."""
        self._items.clear()
        self._position = 0

    def __iter__(self) -> Iterator[T]:
        """Iterate over live occurrences in execution order."""
        for item_ref in self._items:
            item = item_ref()
            if item is not None:
                yield item

    def advance(self, item: T) -> None:
        """Consume a demand call, restarting after the sequence is consumed."""
        if not self._items or self._items[self._position]() is not item:
            raise RuntimeError("FSDP module calls diverged from the recorded prefetch order.")
        self._position = (self._position + 1) % len(self._items)

    def next_item(self, item: T, offset: int = 1) -> T | None:
        """Look ahead from the last consumed occurrence without moving the cursor."""
        index = (self._position - 1) % len(self._items)
        next_index = index + offset
        if next_index >= len(self._items):
            return None
        return self._items[next_index]()
