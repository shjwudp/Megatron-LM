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

"""Completion-based scheduling for the minimal Megatron-FSDP path.

:class:`TraceAndReplayScheduler` implements the combined-1F1B fine-grained
execution model: fine-grained FSDP units are unsharded, resharded, and reduced
in an occurrence-based order that the static ``forward_order`` / ``backward_order``
sequences cannot express. The scheduler therefore traces the *real* op stream
for one global-batch step and, at the end of that step, compiles it into an
optimized plan that the next step replays.

Each op in the live schedule gets a standalone op number. The replay step sees
the same calls in the same order (same op numbers), but the plan can change what
an op does — e.g. append an all-gather prefetch at a ``wait_unshard`` op, or skip
a reshard that is immediately undone by a same-module re-unshard.

The scheduler owns execution: it drives the real FSDP lifecycle on each module
(built with ``register_hooks=False`` when the combined scheduler is active), so
it replicates the root stream-sync that ``pre_forward`` otherwise performs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .module import FsdpModule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchedulePolicy:
    """Control communication scheduling for one FSDP module.

    ``None`` prefetches one successor, preserving the default behavior. ``0``
    disables prefetching. Positive values specify parameter-element budgets.
    """

    forward_prefetch_size: int | None = None
    backward_prefetch_size: int | None = None

    def __post_init__(self) -> None:
        """Validate non-negative prefetch budgets."""
        if self.forward_prefetch_size is not None and self.forward_prefetch_size < 0:
            raise ValueError(
                "forward_prefetch_size must be non-negative, " f"got {self.forward_prefetch_size}."
            )
        if self.backward_prefetch_size is not None and self.backward_prefetch_size < 0:
            raise ValueError(
                "backward_prefetch_size must be non-negative, "
                f"got {self.backward_prefetch_size}."
            )


class OpKind(Enum):
    """Kind of a fine-grained execution op the scheduler drives or observes.

    These mirror the lifecycle entry points that the combined-1F1B hooks call:
    unshard (materialize parameters), wait (let compute consume them), reshard
    (release unsharded storage), and issue-reduce-gradients (reduce-scatter).
    """

    ISSUE_UNSHARD = auto()
    WAIT_UNSHARD = auto()
    RESHARD = auto()
    ISSUE_REDUCE_GRADIENTS = auto()


class _Mode(Enum):
    """Lifecycle phase of a :class:`TraceAndReplayScheduler`."""

    OFF = auto()      # Scheduler disabled: passthrough, no tracing/optimization.
    TRACING = auto()  # Recording the live op stream while executing unoptimized.
    REPLAYING = auto()  # Executing the compiled plan (prefetch/skip applied).


@dataclass(frozen=True)
class TraceEvent:
    """One recorded op from the live schedule.

    Attributes:
        op_index: Standalone, monotonically increasing op number within the
            iteration. Replay sees the same calls at the same indices.
        kind: Which lifecycle entry point produced this op.
        module: The FSDP module the op applies to.
        orientation: Payload orientation (``"rowwise"`` forward / ``"colwise"``
            backward) if known; ``None`` when the hook does not carry it.
    """

    op_index: int
    kind: OpKind
    module: FsdpModule
    orientation: str | None = None


@dataclass
class PlanOp:
    """One op in the compiled plan, possibly transformed by optimization.

    The plan is parallel to the traced events (same op numbers). Replay executes
    ``PlanOp``\ s instead of the raw events:

    - ``skip``: drop the op's real work (e.g. a reshard immediately undone by a
      same-module re-unshard; the all-gather storage stays resident).
    - ``prefetch_after``: after handling this op, issue an all-gather for the
      given ``(module, orientation)`` so its unshard overlaps the current module's
      compute.
    """

    trigger_event: TraceEvent
    skip: bool = False
    prefetch_after: tuple[FsdpModule, str | None] | None = None


class TraceAndReplayScheduler:
    """Trace fine-grained FSDP ops and replay an optimized plan.

    The scheduler is owned by an :class:`FsdpContext` and created in
    ``fully_shard_context`` when trace-and-replay prefetch is enabled. It never
    decides compute order — it observes the live schedule (and drives execution)
    and, at the iteration boundary, compiles the observed op stream into a plan
    that the next iteration replays.

    State machine::

        OFF --(enabled)--> TRACING --(end_iteration)--> REPLAYING
                               ^                            |
                               +--------(divergence)---------+
    """

    _mode: _Mode
    _events: list[TraceEvent]
    _plan: list[PlanOp]
    _op_index: int
    _enabled: bool

    def __init__(self, context: FsdpContext) -> None:
        """Create a scheduler.

        Args:
            context: The owning :class:`FsdpContext`, used for streams and to
                validate the enabled flag.
        """
        self._context = context
        self._mode = _Mode.OFF
        self._events = []
        self._plan = []
        self._op_index = 0
        # Diagnostics.
        self._divergences = 0

    # ------------------------------------------------------------------
    # Lifecycle: iteration boundaries
    # ------------------------------------------------------------------

    def begin_iteration(self) -> None:
        """Start a new global-batch step.

        The first call after (re-)enabling starts tracing; subsequent calls keep
        the prior phase so a compiled plan continues to replay across steps until
        it diverges. Resets the per-iteration op cursor.
        """
        if self._mode is _Mode.OFF:
            self._mode = _Mode.TRACING
            logger.debug("TraceAndReplayScheduler: beginning a tracing step.")
        self._op_index = 0

    def end_iteration(self) -> None:
        """Compile the traced op stream into a plan, or validate the replay.

        After a tracing step, builds the optimized ``_plan`` and switches to
        ``REPLAYING``. After a replaying step, verifies every op was consumed;
        a shortfall means the schedule diverged, so it falls back to tracing.
        """
        if self._mode is _Mode.OFF:
            return

        if self._mode is _Mode.TRACING:
            if self._events:
                self._build_plan()
                self._mode = _Mode.REPLAYING
                logger.info(
                    "TraceAndReplayScheduler: compiled %d-op plan; replaying next step.",
                    len(self._plan),
                )
            else:
                # Nothing was traced (e.g. a no-op step); keep tracing.
                self._mode = _Mode.TRACING
        elif self._mode is _Mode.REPLAYING:
            if self._op_index != len(self._plan):
                self._divergences += 1
                logger.warning(
                    "TraceAndReplayScheduler: replay ended after %d of %d ops; "
                    "re-tracing next step (divergence #%d).",
                    self._op_index,
                    len(self._plan),
                    self._divergences,
                )
                self._mode = _Mode.TRACING
                self._events = []
                self._plan = []
        self._op_index = 0

    def report(self) -> None:
        """Log one-line scheduler statistics (called periodically by the loop)."""
        logger.info(
            "TraceAndReplayScheduler: mode=%s ops=%d plan=%d divergences=%d",
            self._mode.name,
            len(self._events) if self._mode is _Mode.TRACING else self._op_index,
            len(self._plan),
            self._divergences,
        )

    # ------------------------------------------------------------------
    # Entry points driven by the combined-1F1B hooks (executor)
    # ------------------------------------------------------------------

    def issue_unshard(self, module: FsdpModule) -> None:
        """Materialize ``module``'s parameters and record/validate the op.

        For a root, syncs the all-gather stream with the current stream first,
        as an external scheduler must (``module.unshard()`` requires it for a
        root). The materialization is idempotent: if a preceding prefetch (or a
        skipped reshard) already left the storage resident, this is a no-op.
        """
        self._record(OpKind.ISSUE_UNSHARD, module, None)
        module.unshard()

    def wait_unshard(self, module: FsdpModule) -> None:
        """Make compute wait on ``module``'s all-gather, then issue its prefetch.

        During replay, after the wait, issues the plan's appended prefetch (if
        any): an all-gather for the next module that overlaps this module's
        compute. During tracing this is a plain wait (no prefetch yet).
        """
        plan_op = self._record(OpKind.WAIT_UNSHARD, module, None)
        module.wait_unshard()
        if plan_op is not None and plan_op.prefetch_after is not None:
            target, orientation = plan_op.prefetch_after
            self._prefetch(target, orientation)

    def reshard(self, module: FsdpModule) -> None:
        """Release ``module``'s unsharded storage, unless the plan skips it.

        A skipped reshard keeps the storage resident for an immediately-following
        same-module re-unshard (the "reshard + unshard pair"), turning that
        re-unshard into a no-op.
        """
        plan_op = self._record(OpKind.RESHARD, module, None)
        if plan_op is not None and plan_op.skip:
            return
        module.reshard()

    def issue_reduce_gradients(self, module: FsdpModule) -> None:
        """Launch ``module``'s reduce-scatters."""
        self._record(OpKind.ISSUE_REDUCE_GRADIENTS, module, None)
        module._reduce_gradient_groups()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record(self, kind: OpKind, module: FsdpModule, orientation: str | None) -> PlanOp | None:
        """Record (tracing) or validate-and-advance (replaying) one op.

        Returns the ``PlanOp`` that replay should honor for this op, or ``None``
        while tracing/disabled (replay then executes the op plainly).
        """
        if self._mode is _Mode.OFF:
            return None

        if self._mode is _Mode.TRACING:
            self._events.append(TraceEvent(self._op_index, kind, module, orientation))
            self._op_index += 1
            return None

        # Replaying.
        if self._op_index >= len(self._plan):
            self._divergences += 1
            logger.warning(
                "TraceAndReplayScheduler: %s(%s) past the %d-op plan; re-tracing.",
                kind.name,
                getattr(module, "name", type(module).__name__),
                len(self._plan),
            )
            self._retrace(kind, module, orientation)
            return None

        plan_op = self._plan[self._op_index]
        trigger_event = self._events[self._op_index]
        if trigger_event.kind is not kind or trigger_event.module is not module:
            self._divergences += 1
            logger.warning(
                "TraceAndReplayScheduler: replay divergence at op %d (expected %s(%s), got "
                "%s(%s)); re-tracing.",
                self._op_index,
                trigger_event.kind.name,
                getattr(trigger_event.module, "name", type(trigger_event.module).__name__),
                kind.name,
                getattr(module, "name", type(module).__name__),
            )
            self._retrace(kind, module, orientation)
            return None

        self._op_index += 1
        return plan_op

    def _prefetch(self, module: FsdpModule, orientation: str | None) -> None:
        """Issue ``module``'s all-gather without consuming an op number.

        ``orientation`` is accepted for the plan builder's ``prefetch_after`` but
        is not yet acted on: the hooks do not currently carry a payload
        orientation, so a module is always materialized in its recorded layout.
        """
        module.unshard()

    def _retrace(self, kind: OpKind, module: FsdpModule, orientation: str | None) -> None:
        """Reset to tracing and seed the new trace with the current op."""
        self._mode = _Mode.TRACING
        self._events = []
        self._plan = []
        self._op_index = 0
        self._events.append(TraceEvent(self._op_index, kind, module, orientation))
        self._op_index += 1

    def _build_plan(self) -> None:
        """Compile ``_events`` into an optimized ``_plan``.

        Two transformations are applied to the raw trace:

        1. **Skip redundant reshard.** A ``RESHARD(M)`` immediately followed by an
           ``ISSUE_UNSHARD(M)`` is pure waste — the storage is released then
           immediately re-materialized. Mark the reshard ``skip`` so its storage
           stays resident and the re-unshard becomes a no-op. (Same-module
           adjacency is the current heuristic; orientation-aware refinement is
           deferred.)
        2. **Append prefetch.** At each ``WAIT_UNSHARD`` op, prefetch the next
           module that will be unsharded (skipping reshard events), so its
           all-gather overlaps the current module's compute.
        """
        events = self._events
        plan = [PlanOp(trigger_event=e) for e in events]
        n = len(plan)

        # 1) Skip a reshard immediately undone by a same-module re-unshard.
        for i in range(n - 1):
            cur, nxt = events[i], events[i + 1]
            if (
                cur.kind is OpKind.RESHARD
                and nxt.kind is OpKind.ISSUE_UNSHARD
                and nxt.module is cur.module
            ):
                plan[i].skip = True

        # 2) Prefetch the next unshard (skipping reshard events) after each wait.
        for i in range(n):
            if events[i].kind is not OpKind.WAIT_UNSHARD:
                continue
            for j in range(i + 1, n):
                if (
                    events[j].kind is OpKind.ISSUE_UNSHARD
                    and events[j].module is not events[i].module
                ):
                    plan[i].prefetch_after = (events[j].module, events[j].orientation)
                    break

        self._plan = plan
