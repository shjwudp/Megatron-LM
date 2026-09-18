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

"""Gradient-completion bookkeeping for a self-resetting backward window.

``MultiplicityReadiness`` is the production completion signal: it is an exact
per-parameter accounting against a schedule-declared multiplicity, so the
automatic single-graph backward and the combined/fine-grained 1F1B backward share
one hook set and differ only in who declares the expected counts.

``Countdown`` and ``GradientReadiness`` are its predecessors. They are kept
because the mechanism tests and the design study exercise them directly, but the
MFSDP v2 production path no longer uses either one for completion:

* ``Countdown`` infers completion from a callback count whose period is the number
  of trainable parameters. That is valid only when every parameter contributes
  exactly once per window, which the combined/fine-grained 1F1B backward does not
  guarantee.
* ``GradientReadiness`` records *which* parameters fired, which fixes the
  over-fire but not the loss of information: an idempotent mark cannot tell
  "nothing left to do" from "something never ran", and an edge that arrives before
  the last application leaks a mark into the next window.
"""

from collections.abc import Hashable, Iterable, Mapping

OverFireRecord = tuple[Hashable, int, int]


class Countdown:
    """Countdown that automatically re-arms after reaching zero.

    Completion is "the ``initial_value``-th callback arrived" and nothing more, so
    the countdown is a valid end-of-backward signal only when a module's backward
    fires exactly one callback per trainable parameter. That holds for the automatic
    single-graph path (``register_hooks=True``); it does not hold for the
    combined/fine-grained 1F1B path, which is why that path now uses
    :class:`MultiplicityReadiness`. Retained for the mechanism tests and the
    design study.
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
    idempotent and a missing one is observable. It is superseded by
    :class:`MultiplicityReadiness`, which counts and therefore can also name an
    over-fire; retained for the mechanism tests and the design study.
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


class MultiplicityReadiness:
    """Exact per-parameter accounting against a schedule-declared multiplicity.

    ``expected[key]`` is the number of autograd GraphTasks that will accumulate
    ``key``'s gradient in one iteration. The schedule is the only component that
    knows that number, so it supplies it; the automatic single-graph path simply
    supplies 1 for every trainable parameter, which makes this class degenerate to
    the previous count-based semantics.

    The asymmetry between the two failure directions is what makes the accounting
    safe:

    * declaring too *few* means the window closes before a surplus contribution
      arrives, and that contribution is then charged to the next window -- i.e. the
      original defect. This shows up as an **over-fire** (an observed count above
      the declared one) and must be reported loudly, once per unit rather than once
      per callback.
    * declaring too *many* only means waiting. It shows up as an **under-fire** at
      the close edge (a count below the declared one) and must also be reported
      loudly, because the other explanation for an under-fire is a contribution
      that never arrived at all -- the silent dropped reduce-scatter case.
    """

    def __init__(self, expected: Mapping[Hashable, int]) -> None:
        """Track completion against ``expected``, the per-key multiplicity.

        Every declared multiplicity must be at least 1: a parameter owned by this
        unit contributes at least once per iteration, and a 0 would silently make
        the window complete without ever observing that parameter.
        """
        for key, count in expected.items():
            if count < 1:
                raise ValueError(f"Multiplicity for {key!r} must be at least 1, got {count}.")
        self._expected = dict(expected)
        self._counts: dict[Hashable, int] = {key: 0 for key in expected}
        self._over_fired: list[OverFireRecord] = []
        self._window_open = False

    @property
    def expected_total(self) -> int:
        """Return the total number of marks a correct window must produce."""
        return sum(self._expected.values())

    @property
    def marked_total(self) -> int:
        """Return the number of marks observed so far in this window."""
        return sum(self._counts.values())

    @property
    def expected(self) -> Mapping[Hashable, int]:
        """Return the declared multiplicity of every key, for diagnostics."""
        return dict(self._expected)

    @property
    def has_pending_marks(self) -> bool:
        """Return whether any mark has arrived since the last :meth:`close`."""
        return self._window_open

    def count(self, key: Hashable) -> int:
        """Return how many times ``key`` has been marked in this window."""
        return self._counts[key]

    def mark(self, key: Hashable) -> None:
        """Record one contribution of ``key``.

        An unknown key is a programming error in the caller's key space, not
        something to ignore: silently dropping it would hide a real contribution
        from the accounting. A count above the declared multiplicity is recorded as
        an over-fire, which is the signal that the multiplicity was under-declared.
        """
        if key not in self._counts:
            raise KeyError(f"{key!r} is not a known parameter of this unit.")
        self._counts[key] += 1
        self._window_open = True
        if self._counts[key] > self._expected[key]:
            self._over_fired.append((key, self._counts[key], self._expected[key]))

    def missing(self) -> frozenset[Hashable]:
        """Return the keys that have contributed fewer times than expected."""
        return frozenset(key for key, count in self._counts.items() if count < self._expected[key])

    def is_complete(self) -> bool:
        """Return whether every key reached its declared multiplicity."""
        return not self.missing()

    def over_fired(self) -> tuple[OverFireRecord, ...]:
        """Return ``(key, observed, expected)`` for every over-fire in this window."""
        return tuple(self._over_fired)

    def close(self) -> bool:
        """End this window, return whether it completed, and re-arm for the next.

        The return value is the verdict for the window that is ending. The
        over-fire record is cleared with the counts, so a caller that wants to
        report it must read :meth:`over_fired` (and :meth:`missing`) before
        closing.
        """
        complete = self.is_complete()
        for key in self._counts:
            self._counts[key] = 0
        self._over_fired.clear()
        self._window_open = False
        return complete
