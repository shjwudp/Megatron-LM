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

"""Gradient-completion bookkeeping for a self-resetting backward window."""

from collections.abc import Hashable, Iterable


class Countdown:
    """Countdown that automatically re-arms after reaching zero.

    Completion is "the ``initial_value``-th callback arrived" and nothing more, so
    the countdown is a valid end-of-backward signal only when a module's backward
    fires exactly one callback per trainable parameter. That holds for the automatic
    single-graph path (``register_hooks=True``); it does not hold for the
    combined/fine-grained 1F1B path, which uses :class:`GradientReadiness`.
    """

    def __init__(self, initial_value: int) -> None:
        """Create a countdown starting at ``initial_value``."""
        if initial_value < 0:
            raise ValueError(f"Countdown initial_value must be non-negative, got {initial_value}.")
        self._initial_value = initial_value
        self._value = initial_value

    @property
    def initial_value(self) -> int:
        """Return the number of decrements in one countdown cycle."""
        return self._initial_value

    def decrement(self) -> bool:
        """Decrement and return whether the countdown completed this call."""
        self._value -= 1
        completed = self._value == 0
        if completed:
            self._value = self._initial_value
        return completed


class GradientReadiness:
    """Track which of a module's trainable parameters produced a gradient.

    A module's backward window is delimited by the schedule, not by a callback
    count. ``Countdown`` infers "backward is over" from the number of callbacks,
    which is wrong whenever a trainable parameter is consumed by a different number
    of autograd graphs than the module assumed: the combined/fine-grained 1F1B
    schedule runs one ``run_backward`` per schedule node, on detached node inputs,
    so a shared parameter (for example the embedding, consumed by both the
    pre-process node and each MTP pre-dispatch node) fires its callback once per
    consuming node and drives the countdown to zero before the module's last
    parameter has a gradient.

    This tracker records *which* parameters fired instead, so a repeated callback is
    idempotent and a missing one is observable. The owning module closes the window
    when the schedule declares its backward over; at that point :meth:`missing` is
    exactly the set of parameters that were unused in the window.
    """

    def __init__(self, keys: Iterable[Hashable]) -> None:
        """Track completion of the parameters identified by ``keys``."""
        self._keys = frozenset(keys)
        self._marked: set[Hashable] = set()

    @property
    def initial_value(self) -> int:
        """Return the number of parameters in one completion window."""
        return len(self._keys)

    @property
    def marked(self) -> int:
        """Return the number of distinct parameters marked in this window."""
        return len(self._marked)

    @property
    def has_pending_marks(self) -> bool:
        """Return whether this window holds any marked, not yet consumed parameter."""
        return bool(self._marked)

    def mark(self, key: Hashable) -> None:
        """Record that ``key`` produced a gradient; repeated marks are idempotent."""
        if key in self._keys:
            self._marked.add(key)

    def missing(self) -> frozenset[Hashable]:
        """Return the parameters without a gradient in the current window."""
        return self._keys - self._marked

    def is_complete(self) -> bool:
        """Return whether every parameter produced a gradient in this window."""
        return not self.missing()

    def close(self) -> bool:
        """End this window, returning whether it completed, and re-arm for the next."""
        complete = self.is_complete()
        self._marked.clear()
        return complete
