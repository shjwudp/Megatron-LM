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

The plan can also move the *issuance time* of a gradient reduce-scatter. With
``SchedulePolicy.defer_grad_reduce`` set, an ``ISSUE_REDUCE_GRADIENTS`` op enqueues
its module (FIFO, no device work) instead of launching the reduce, and the plan
records which later op — a ``WAIT_UNSHARD``, a ``RESHARD`` — fires the queue.
Deferral is off by default and is an exact no-op then. It is a plan-level decision
only: while tracing, the scheduler records and executes the unoptimized op stream,
so the anchor is learned from the trace exactly like the prefetch is. An
``END_MICROBATCH`` op is a hard barrier — the queue is always launched there, and
no anchor is ever selected across it, so a queued reduce can never run against
another microbatch's ``is_last_microbatch`` or ``main_grad`` state.

Unshards are *orientation-aware*. An MXFP8 primary weight rests as two separate
payload buffers — row-wise (forward GEMM) and column-wise (backward GEMM) — so
the plan records which one each unshard needs, widens a materialization that has
to serve both a forward and a backward pass, and refuses to skip a reshard that
the next unshard needs in order to change orientation. Regular parameter groups
store one full parameter and ignore the request entirely.

The scheduler owns execution: it drives the real FSDP lifecycle on each module
(built with ``register_hooks=False`` when the combined scheduler is active), so
it replicates the root stream-sync that ``pre_forward`` otherwise performs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal

from .quantization import BOTH, PayloadOrientation, merge_orientations, orientation_directions

if TYPE_CHECKING:
    from .module import FsdpModule

logger = logging.getLogger(__name__)


#: Supported values of :attr:`SchedulePolicy.defer_grad_reduce`.
#:
#: ``none``            — launch the reduce-scatter where it is recorded (default).
#: ``before_prefetch`` — run it at the next ``WAIT_UNSHARD``, *before* that op's
#:                       prefetch all-gathers, deliberately letting the reduce use
#:                       the interconnect first.
#: ``after_prefetch``  — run it at the next ``WAIT_UNSHARD``, after those all-gathers
#:                       have been issued, so the critical-path all-gather wins.
#: ``next_reshard``    — run it at the next ``RESHARD``.
DEFER_GRAD_REDUCE_ANCHORS: tuple[str, ...] = (
    "none",
    "before_prefetch",
    "after_prefetch",
    "next_reshard",
)

DeferGradReduce = Literal["none", "before_prefetch", "after_prefetch", "next_reshard"]


@dataclass(frozen=True)
class SchedulePolicy:
    """Control communication scheduling for one FSDP module.

    ``None`` prefetches one successor, preserving the default behavior. ``0``
    disables prefetching. Positive values specify parameter-element budgets.

    ``defer_grad_reduce`` selects where a reduce-scatter recorded at
    ``post_backward`` is launched (see :data:`DEFER_GRAD_REDUCE_ANCHORS`). The
    prefetch budget is reused as the queue *depth* in parameter elements: ``None``
    keeps a single queued reduce per anchor, ``0`` disables deferral entirely, and
    a positive value extends the number of reduces that may be held queued until
    the accumulated ``num_parameter_elements`` reach it. ``"none"`` (the default)
    is a strict no-op.
    """

    forward_prefetch_size: int | None = None
    backward_prefetch_size: int | None = None
    defer_grad_reduce: DeferGradReduce = "none"

    def __post_init__(self) -> None:
        """Validate non-negative prefetch budgets and the deferral anchor."""
        if self.forward_prefetch_size is not None and self.forward_prefetch_size < 0:
            raise ValueError(
                "forward_prefetch_size must be non-negative, " f"got {self.forward_prefetch_size}."
            )
        if self.backward_prefetch_size is not None and self.backward_prefetch_size < 0:
            raise ValueError(
                "backward_prefetch_size must be non-negative, "
                f"got {self.backward_prefetch_size}."
            )
        if self.defer_grad_reduce not in DEFER_GRAD_REDUCE_ANCHORS:
            raise ValueError(
                f"defer_grad_reduce must be one of {DEFER_GRAD_REDUCE_ANCHORS}, "
                f"got {self.defer_grad_reduce!r}."
            )


class OpKind(Enum):
    """Kind of a fine-grained execution op the scheduler drives or observes.

    These mirror the lifecycle entry points that the combined-1F1B hooks call:
    unshard (materialize parameters), wait (let compute consume them), reshard
    (release unsharded storage), and issue-reduce-gradients (reduce-scatter).

    ``END_MICROBATCH`` is not a lifecycle entry point: it is recorded where the
    combined-1F1B scheduler finishes one microbatch's backward, and is a hard
    barrier for gradient-reduce deferral (see :meth:`TraceAndReplayScheduler.end_microbatch`).
    """

    ISSUE_UNSHARD = auto()
    WAIT_UNSHARD = auto()
    RESHARD = auto()
    ISSUE_REDUCE_GRADIENTS = auto()
    END_MICROBATCH = auto()


class _Mode(Enum):
    """Lifecycle phase of a :class:`TraceAndReplayScheduler`."""

    OFF = auto()      # Scheduler disabled: passthrough, no tracing/optimization.
    TRACING = auto()  # Recording the live op stream while executing unoptimized.
    REPLAYING = auto()  # Executing the compiled plan (prefetch/skip applied).


#: One queued (not yet launched) gradient reduce.
#:
#: The whole ``_reduce_gradient_groups()`` call is deferred, not just the collective
#: launch: no partial-grad buffer is allocated until the drain, so the queue costs no
#: GPU memory. ``is_last_microbatch`` is captured at *enqueue* time because
#: ``FsdpContext.is_last_microbatch`` is a mutable flag that the ``microbatch``
#: context manager flips, and ``reduce_partial_gradients`` reads it at *execution*
#: time.
_PendingReduce = tuple["FsdpModule", bool]


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
      same-module re-unshard that needs no payload the reshard released; the
      all-gather storage stays resident).
    - ``orientation``: for an ``ISSUE_UNSHARD``, the payload the materialization
      must carry. It is the union of what every unshard sharing that
      materialization needs, so it can be wider than the traced event's
      orientation but never narrower.
    - ``prefetch_after``: after handling this op, issue all-gathers for the given
      ``(module, orientation)`` pairs so their unshards overlap this module's
      compute. More than one entry means the plan is prefetching several modules
      ahead, bounded by the waiting module's parameter-element prefetch budget.
    - ``defer_reduce``: for an ``ISSUE_REDUCE_GRADIENTS``, do not launch the
      reduce-scatter now; enqueue it for the anchor this op selected.
    - ``reduce_gradients_after``: after handling this op, launch every gradient
      reduce queued onto it, oldest first. Mirrors ``prefetch_after``: the op that
      carries it is the *anchor*, and the module identities moved onto it are the
      reduces deferred from earlier in the op stream. ``None`` means "not an
      anchor"; an empty tuple is never stored.
    """

    trigger_event: TraceEvent
    skip: bool = False
    orientation: str | None = None
    prefetch_after: tuple[tuple[FsdpModule, str | None], ...] | None = None
    defer_reduce: bool = False
    reduce_gradients_after: tuple[FsdpModule, ...] | None = None


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
        # FIFO queue of reduces deferred to a later anchor, oldest first. Depth is
        # bounded in parameter elements by the queuing module's prefetch budget.
        self._pending_reduces: list[_PendingReduce] = []
        self._pending_reduce_elements = 0
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

    def issue_unshard(self, module: FsdpModule, orientation: PayloadOrientation) -> None:
        """Materialize ``module``'s parameters and record/validate the op.

        ``orientation`` is the payload the caller is about to compute with
        (``"rowwise"`` from a forward hook, ``"colwise"`` from a backward hook) and
        is authoritative for the materialization: this method always requests
        exactly that orientation, while tracing and while replaying alike.

        Correctness does not depend on the plan, because ``FsdpModule.unshard``
        owns the residency bookkeeping and widens in place when a requested
        direction is not already materialized (see
        ``FsdpModule._unshard_parameter_groups``): a request no wider than what is
        resident is served as-is, and a request for a direction that is missing
        gathers only the directions still missing. A module whose forward and
        backward unshards share one residency window is therefore widened by the
        module at the backward, instead of being over-materialized to ``"both"`` at
        the forward.

        ``orientation`` is also recorded on the trace event, which
        :meth:`_build_plan` uses to decide whether the reshard separating two of a
        module's materializations can be skipped, and to choose the orientation for
        a speculative prefetch. That is a scheduling input only; it never overrides
        the orientation requested here.

        For a root, syncs the all-gather stream with the current stream first,
        as an external scheduler must (``module.unshard()`` requires it for a
        root).
        """
        self._record(OpKind.ISSUE_UNSHARD, module, orientation)
        # ``prefetch`` is the FIRST parameter of ``FsdpModule.unshard``, so the
        # orientation must be passed by keyword; positionally it would bind to
        # ``prefetch`` and leave ``orientation`` at its ``BOTH`` default, silently
        # disabling per-phase narrowing on the scheduler path.
        module.unshard(orientation=orientation)

    def wait_unshard(self, module: FsdpModule) -> None:
        """Make compute wait on ``module``'s all-gather, then issue its prefetches.

        During replay, after the wait, issues the plan's appended prefetches (if
        any): all-gathers for the modules that will be unsharded next, so they
        overlap this module's compute. There can be more than one when the waiting
        module's prefetch budget allows a deeper lookahead. During tracing this is a
        plain wait (no prefetch yet).

        When the plan made this op an anchor for deferred gradient reduces, the
        queue is launched here too. ``after_prefetch`` (and ``next_reshard``) run it
        after the prefetch all-gathers have been issued — the critical-path
        all-gather gets the interconnect first; ``before_prefetch`` runs it before
        them, the opposite prediction. The position follows the policy of the
        oldest queued module, i.e. the module whose deferral selected this anchor;
        the queue is FIFO, so it can never be split.
        """
        plan_op = self._record(OpKind.WAIT_UNSHARD, module, None)
        module.wait_unshard()
        drain_before_prefetch = (
            plan_op is not None
            and plan_op.reduce_gradients_after is not None
            and self._queued_anchor_is_before_prefetch()
        )
        if drain_before_prefetch:
            self._drain_reduces(plan_op)
        if plan_op is not None and plan_op.prefetch_after:
            for target, orientation in plan_op.prefetch_after:
                self._prefetch(target, orientation)
        if not drain_before_prefetch:
            self._drain_reduces(plan_op)

    def reshard(self, module: FsdpModule) -> None:
        """Release ``module``'s unsharded storage, unless the plan skips it.

        A skipped reshard keeps the storage resident for an immediately-following
        same-module re-unshard (the "reshard + unshard pair"), turning that
        re-unshard into a no-op. A skipped reshard is never selected as a
        ``next_reshard`` anchor (see :meth:`_build_plan_defer_reduces`), so the early
        return cannot strand a queued reduce.
        """
        plan_op = self._record(OpKind.RESHARD, module, None)
        if plan_op is not None and plan_op.skip:
            return
        module.reshard()
        self._drain_reduces(plan_op)

    def issue_reduce_gradients(self, module: FsdpModule) -> None:
        """Launch ``module``'s reduce-scatters, or queue them for a later anchor.

        While tracing (or when the plan did not move this op, or while a CUDA graph
        is capturing) the reduce is launched exactly where it is recorded, which is
        today's behaviour. During replay of a ``defer_reduce`` op the module is
        appended to the pending FIFO and no device work happens; the plan's anchor
        launches it.

        Enqueueing happens *after* :meth:`_record`, so ``_op_index`` and the
        divergence accounting are identical whether or not the op was deferred.
        """
        plan_op = self._record(OpKind.ISSUE_REDUCE_GRADIENTS, module, None)
        if plan_op is not None and plan_op.defer_reduce and self._deferral_allowed():
            self._queue_reduce(module)
            return
        module._reduce_gradient_groups()

    def end_microbatch(self, module: FsdpModule) -> None:
        """Close the current microbatch's deferral scope. A hard barrier.

        Recorded where the combined-1F1B scheduler finishes one microbatch's
        backward. Every pending reduce is launched here, oldest first, and no anchor
        is ever selected across this op, so a deferred reduce can never be executed
        against another microbatch's ``is_last_microbatch`` — or after the next
        microbatch has started overwriting ``main_grad``. It also flushes from the
        queue, which is what makes the emptiness assertion in
        :meth:`FsdpContext.post_backward` hold before ``optimizer.step()``.

        While tracing this records the barrier (so replay finds it at the same op
        index) and flushes an always-empty queue, i.e. it does no work.
        """
        self._record(OpKind.END_MICROBATCH, module, None)
        self._flush_reduces()

    def pending_reduce_count(self) -> int:
        """Number of gradient reduces queued but not yet launched."""
        return len(self._pending_reduces)

    def assert_no_pending_reduces(self) -> None:
        """Raise if a queued reduce was never launched.

        Called from the context-level post-backward barrier, which is what orders
        the optimizer after the reduce-scatter stream. A queued-but-unlaunched
        reduce holds no storage, but leaving it would let ``optimizer.step()`` read
        ``main_grad`` before its DP-outer reduction ran.
        """
        if not self._pending_reduces:
            return
        names = [
            getattr(module, "_name", None) or type(module).__name__
            for module, _ in self._pending_reduces
        ]
        raise RuntimeError(
            "TraceAndReplayScheduler: %d gradient reduce(s) queued but not launched "
            "(%s); the optimizer would step on unfinalized gradients. Every queue "
            "must be flushed at its anchor or at END_MICROBATCH."
            % (len(self._pending_reduces), ", ".join(names))
        )

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

        ``orientation`` comes from the plan and is the payload that ``module``'s
        own materialization will need, so the later unshard finds its payload
        already resident instead of re-gathering the other orientation. ``None``
        (no plan available) falls back to the safe superset.
        """
        module.unshard(orientation=orientation or BOTH)

    # ------------------------------------------------------------------
    # Deferred gradient reduces (a bounded FIFO queue)
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_active() -> bool:
        """Whether a CUDA graph is being captured on the current stream.

        Deferral changes *which* fork edge first joins the reduce-scatter stream to
        an active capture: ``FsdpModule.pre_backward`` forks it once at the start of
        backward, and each module re-forks it with ``wait_stream`` before its own
        collective, so a deferred first allocation on the reduce-scatter stream
        would be a raw ``cudaMalloc`` under capture. The feature is therefore gated
        off while capturing and the reduce is issued where it was recorded.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard runtime dependency
            return False
        if not torch.cuda.is_available():
            return False
        return bool(torch.cuda.is_current_stream_capturing())

    def _deferral_allowed(self) -> bool:
        """Whether queued reduce-scatters may be used right now."""
        return not self._capture_active()

    def _queue_reduce(self, module: FsdpModule) -> None:
        """Append ``module``'s reduce to the FIFO without doing its work.

        The whole ``_reduce_gradient_groups()`` call is deferred, so no partial-grad
        buffer is allocated here and the queue costs no GPU memory. The live
        ``context.is_last_microbatch`` is captured into the queue element because
        the flag is mutable and the reduction reads it at execution time.

        The element budget is the queuing module's own prefetch budget (the same
        knob the plan's anchor selection uses). When the budget is reached this
        launches the *oldest* entries first — it never refuses to enqueue, so a
        reduce is never dropped, and it never reorders, so the launched order is the
        traced order minus the deferred elements.
        """
        budget = module._schedule_policy.forward_prefetch_size
        elements = module.num_parameter_elements
        if budget is not None and budget > 0:
            while self._pending_reduces and self._pending_reduce_elements + elements > budget:
                self._launch_oldest_reduce()
        self._pending_reduces.append((module, self._context.is_last_microbatch))
        self._pending_reduce_elements += elements

    def _launch_oldest_reduce(self) -> None:
        """Launch the oldest pending reduce (strict FIFO pop-front)."""
        module, is_last_microbatch = self._pending_reduces.pop(0)
        self._pending_reduce_elements -= module.num_parameter_elements
        module._reduce_gradient_groups(is_last_microbatch=is_last_microbatch)

    def _flush_reduces(self) -> None:
        """Launch every pending reduce, oldest first. Never reorders."""
        while self._pending_reduces:
            self._launch_oldest_reduce()

    def _drain_reduces(self, plan_op: PlanOp | None) -> None:
        """Launch the queued reduces when ``plan_op`` is one of their anchors."""
        if plan_op is None or not plan_op.reduce_gradients_after:
            return
        self._flush_reduces()

    def _queued_anchor_is_before_prefetch(self) -> bool:
        """Whether the oldest queued reduce selected a ``before_prefetch`` anchor.

        One anchor has one drain position, so the oldest queued element decides it;
        the queue is FIFO, so this is the same module whose op stream reached the
        anchor first.
        """
        for module, _ in self._pending_reduces:
            policy = getattr(module._schedule_policy, "defer_grad_reduce", "none")
            return policy == "before_prefetch"
        return False

    def _retrace(self, kind: OpKind, module: FsdpModule, orientation: str | None) -> None:
        """Reset to tracing and seed the new trace with the current op.

        Any queued reduce is launched first: it is real gradient work that may not
        be dropped just because the plan diverged. It lies earlier in the op stream
        than the diverging op, so launching it now also preserves FIFO order.
        """
        self._flush_reduces()
        self._mode = _Mode.TRACING
        self._events = []
        self._plan = []
        self._op_index = 0
        self._events.append(TraceEvent(self._op_index, kind, module, orientation))
        self._op_index += 1

    def _build_plan(self) -> None:
        """Compile ``_events`` into an optimized ``_plan``.

        Four transformations are applied to the raw trace:

        1. **Widen a materialization that serves several unshards.** Each module's
           unshards are grouped into *residency windows* delimited by its own
           reshard events: within one window the storage is materialized once, so
           the window's first unshard has to gather the union of the orientations
           its members need. A window holding both a forward (row-wise) and a
           backward (column-wise) unshard — a module whose two passes are not
           separated by a reshard — therefore materializes ``"both"``. This is what
           makes transformation 2 safe.
        2. **Skip a redundant reshard, orientation-aware.** A ``RESHARD(M)`` is
           pure waste when the storage it releases is re-materialized immediately
           with an orientation it already carries: mark it ``skip`` so the storage
           stays resident and the next unshard becomes a no-op. When that unshard
           needs the *other* orientation — the normal forward(row-wise) followed by
           backward(column-wise) transition of one FSDP unit — the reshard must
           execute, or the backward pass would run with column-wise data missing.
        3. **Append prefetch.** At each ``WAIT_UNSHARD`` op, prefetch the modules
           that will be unsharded next (skipping reshard events), each in the
           orientation its own materialization will use, so their all-gathers
           overlap the current module's compute. The lookahead is budgeted by the
           waiting module's own :class:`SchedulePolicy` — the same
           parameter-element budget the automatic path passes to
           ``_prefetch_parameter_groups`` — so ``None`` keeps a single successor, a
           positive value extends the walk until the accumulated
           ``num_parameter_elements`` reaches it, and ``0`` disables prefetching
           entirely, exactly as ``SchedulePolicy`` documents. A candidate whose
           demand unshard is separated from this point by its own reshard is dropped,
           because that reshard would release the storage the prefetch had just
           gathered.
        4. **Move an eligible gradient reduce onto its anchor.** See
           :meth:`_build_plan_defer_reduces`.
        """
        events = self._events
        plan = [PlanOp(trigger_event=e) for e in events]
        n = len(plan)

        # 1) Group each module's own ops into (opening reshard, unshards) residency
        #    windows. A module's ops are independent of every other module's, so the
        #    per-module walk below is a complete description of its residency.
        per_module_ops: dict[int, list[tuple[int, OpKind]]] = {}
        for index, event in enumerate(events):
            per_module_ops.setdefault(id(event.module), []).append((index, event.kind))

        for ops in per_module_ops.values():
            windows: list[tuple[int | None, list[int]]] = []
            opening: int | None = None
            unshards: list[int] = []
            for index, kind in ops:
                if kind is OpKind.RESHARD:
                    windows.append((opening, unshards))
                    opening, unshards = index, []
                elif kind is OpKind.ISSUE_UNSHARD:
                    unshards.append(index)
            windows.append((opening, unshards))

            resident: frozenset[str] = frozenset()
            for window_opening, window_unshards in windows:
                need = frozenset().union(
                    *(orientation_directions(events[i].orientation) for i in window_unshards)
                )
                if window_opening is not None:
                    if window_unshards and need <= resident:
                        # The storage still carries everything the window needs.
                        plan[window_opening].skip = True
                    else:
                        # The reshard runs and releases every orientation.
                        resident = frozenset()
                if window_unshards:
                    # 1) This single materialization must cover the whole window.
                    materialized = merge_orientations(need)
                    for index in window_unshards:
                        plan[index].orientation = materialized
                    resident = need

        # 3) Prefetch the modules that will be unsharded next (skipping reshard
        #    events), each in the orientation its own materialization will use, so
        #    their all-gathers overlap this module's compute. The lookahead is
        #    budgeted by the waiting module's own SchedulePolicy -- the same
        #    parameter-element budget ``prefetch="forward"`` hands to
        #    ``_prefetch_parameter_groups`` on the automatic path; the adapter sets
        #    the forward and backward sizes from one config value, so a single
        #    budget covers both passes. ``None`` stops after one successor (the
        #    historical behaviour); otherwise the walk continues until the
        #    accumulated elements reach the budget. A candidate whose own reshard
        #    falls between here and its demand unshard is skipped, since that reshard
        #    would release what the prefetch gathered.
        for i in range(n):
            if events[i].kind is not OpKind.WAIT_UNSHARD:
                continue
            budget = events[i].module._schedule_policy.forward_prefetch_size
            if budget is not None and budget <= 0:
                # ``SchedulePolicy`` defines a zero budget as "disable prefetching",
                # and the automatic path implements that by never entering its
                # accumulation loop. Match it here rather than treating 0 like
                # ``None`` (which prefetches a single successor).
                continue
            targets: list[tuple[FsdpModule, str | None]] = []
            seen = {id(events[i].module)}
            resharded: set[int] = set()
            accumulated = 0
            for j in range(i + 1, n):
                event = events[j]
                if event.kind is OpKind.RESHARD:
                    resharded.add(id(event.module))
                    continue
                if event.kind is not OpKind.ISSUE_UNSHARD:
                    continue
                module = event.module
                if id(module) in seen or id(module) in resharded:
                    continue
                seen.add(id(module))
                targets.append((module, plan[j].orientation))
                if budget is None:
                    break
                accumulated += module.num_parameter_elements
                if accumulated >= budget:
                    break
            if targets:
                plan[i].prefetch_after = tuple(targets)

        # 4) Move an eligible gradient reduce onto its anchor, if the policy asks
        #    for it. Nothing happens by default.
        self._build_plan_defer_reduces(events, plan, n)

        self._plan = plan

    def _build_plan_defer_reduces(
        self, events: list[TraceEvent], plan: list[PlanOp], n: int
    ) -> None:
        """Transformation 4: schedule the issuance time of the reduce-scatters.

        For every ``ISSUE_REDUCE_GRADIENTS`` whose module carries a non-``none``
        ``defer_grad_reduce`` policy, walk forward to the first anchor op —
        ``WAIT_UNSHARD`` for ``before_prefetch``/``after_prefetch``, ``RESHARD`` for
        ``next_reshard`` — and record on the reduce op that it must be queued, and on
        the anchor op that it must launch the queue.

        The walk **stops at an** ``END_MICROBATCH``: an anchor may never cross a
        microbatch boundary, so a reduce with no reachable anchor before the
        boundary stays exactly where it is (``defer_reduce`` stays ``False``). A
        ``RESHARD`` the plan skips is not an anchor either, since a skipped reshard
        returns before its drain.

        Depth reuses the module's prefetch budget as an element budget, with the
        same stop rule as step 3: ``None`` moves a single reduce onto an anchor,
        ``0`` (or a negative) disables deferral entirely, and a positive value keeps
        moving reduces onto one anchor until the accumulated
        ``num_parameter_elements`` reach it. The budget is tested *before* the first
        candidate is appended, so ``0`` cannot silently behave like ``None``.
        """
        if self._capture_active():
            # Constraint: under CUDA-graph capture the reduce must be issued where
            # it is recorded, so its fork edge stays part of the capture. The replay
            # path re-checks the same condition.
            return

        anchor_elements: dict[int, int] = {}
        for i in range(n):
            if events[i].kind is not OpKind.ISSUE_REDUCE_GRADIENTS:
                continue
            module = events[i].module
            policy = getattr(module._schedule_policy, "defer_grad_reduce", "none")
            if policy == "none":
                continue
            budget = module._schedule_policy.forward_prefetch_size
            if budget is not None and budget <= 0:
                # ``SchedulePolicy`` defines a zero budget as "disabled"; test it
                # before the first candidate, or ``0`` would silently defer once.
                continue

            anchor = self._find_reduce_anchor(events, plan, i, n, policy)
            if anchor is None:
                continue
            if anchor in anchor_elements:
                accumulated = anchor_elements[anchor]
                if budget is None or accumulated >= budget:
                    # This anchor already holds its budget's worth of reduces;
                    # leave this one where it is rather than queueing it behind a
                    # drain that is already beyond the depth bound.
                    continue
            else:
                accumulated = 0

            plan[i].defer_reduce = True
            queued = plan[anchor].reduce_gradients_after
            plan[anchor].reduce_gradients_after = (
                queued + (module,) if queued is not None else (module,)
            )
            anchor_elements[anchor] = accumulated + module.num_parameter_elements

    @staticmethod
    def _find_reduce_anchor(
        events: list[TraceEvent], plan: list[PlanOp], i: int, n: int, policy: str
    ) -> int | None:
        """Index of the anchor op that should launch the reduce recorded at ``i``.

        Walks forward from ``i`` and returns the first ``WAIT_UNSHARD`` (for the
        prefetch anchors) or the first non-skipped ``RESHARD`` (for
        ``next_reshard``), or ``None`` when the microbatch ends first or no such op
        follows in the trace.
        """
        for j in range(i + 1, n):
            kind = events[j].kind
            if kind is OpKind.END_MICROBATCH:
                # Hard barrier: an anchor may never cross a microbatch boundary.
                return None
            if policy == "next_reshard":
                if kind is OpKind.RESHARD and not plan[j].skip:
                    return j
                continue
            if kind is OpKind.WAIT_UNSHARD:
                return j
        return None
