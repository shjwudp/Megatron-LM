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

"""Execution-order tracer and prefetch planner for fine-grained FSDP.

The combined-1F1B + VPP schedule is occurrence-based: the same FSDP unit can
be consumed in forward and backward, model chunks interleave, and
warmup/steady/cooldown differ per pipeline rank. The static
``forward_order`` / ``backward_order`` sequences cannot express that runtime
path, so a per-context runner traces the real execution and replays it to
drive prefetch.

Two cooperating paths:

- **Trace path**: during the first global batch, every fine-grained
  execution event (consume, reshard, communication anchor) is recorded in actual order. No
  prefetch is issued and no reshard is optimized away.
- **Optimization path**: from the second batch, each real op is validated
  against the traced cycle and translated into optimization directives:
  a prefetch target after an unshard, and a skip-reshard decision when the
  traced schedule re-unshards the same module with the same orientation
  immediately.

State machine::

    TRACING --(complete_trace)--> REPLAYING --(divergence)--> TRACING
       ^                                                          |
       +----------------------(complete_trace)--------------------+

The per-op interfaces are:

- ``record_unshard(module, orientation)`` / ``record_reshard(module)``:
  trace path — record the real event, or during replay validate it against
  the traced cycle and advance the cursor.
- ``suggest_prefetch()``: optimization path — a configured-depth future
  traced unshard before the current global-batch boundary (skipping other
  event kinds) to all-gather ahead, or ``None`` while tracing or when the
  requested lookahead would cross the optimizer step.
- ``suggest_skip_reshard(module)``: optimization path — whether this reshard
  can be skipped because the next traced unshard reuses the same module and
  orientation, keeping the storage resident.
- ``record_completion(...)`` / ``record_reduce_scatter_release(...)``:
  include configured communication anchors in occurrence replay so VPP and
  recomputation cannot release a collective at the wrong occurrence.
"""

import dataclasses
import logging
from enum import Enum, auto

# Forward reference; FsdpModule is imported lazily to avoid a cycle.
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from torch import nn

    from .module import FsdpModule

logger = logging.getLogger(__name__)


class RunnerPhase(Enum):
    """Lifecycle phase of an :class:`FsdpExecutionRunner`."""

    TRACING = auto()
    REPLAYING = auto()


class EventKind(Enum):
    """Kind of an execution event on the trace path."""

    UNSHARD = auto()
    RESHARD = auto()
    COMPLETION = auto()
    RS_RELEASE = auto()


@dataclasses.dataclass(frozen=True)
class RunnerEvent:
    """One fine-grained execution event.

    Attributes:
        kind: Whether the module's parameters are consumed or resharded.
        module: The FSDP module the event applies to.
        orientation: Payload orientation (``"rowwise"`` forward,
            ``"colwise"`` backward); ``None`` for reshard events.
        anchor: Completed descendant module for completion events.
        phase: Completed execution phase for completion events.
    """

    kind: EventKind
    module: "FsdpModule"
    orientation: str | None = None
    anchor: "nn.Module | str | None" = None
    phase: "Literal['forward', 'backward'] | None" = None


class FsdpExecutionRunner:
    """Record the fine-grained execution and plan prefetches.

    The runner is owned by an :class:`FsdpContext` and driven by the
    fine-grained unshard/reshard entry points plus the global-batch boundary
    signaled by the training loop. It never decides compute order — it only
    observes events and, during replay, suggests what to prefetch and which
    reshards to skip.
    """

    def __init__(self, context, *, use_trace_replay: bool = False) -> None:
        """Create a runner in the tracing phase.

        Args:
            context: The owning :class:`FsdpContext`, used for the static
                orders in default mode.
            use_trace_replay: Enable trace-and-replay prefetch.
        """
        self._context = context
        self._use_trace_replay = use_trace_replay
        self._phase = RunnerPhase.TRACING
        self._trace: list[RunnerEvent] = []
        self._replay_index = 0
        self._cycles_observed = 0
        # Modules consumed in the current round. The fine-grained schedule
        # fires one hook per sub-module (dense, experts), so the same module
        # can be recorded several times within a round; only the first is a
        # real unshard. Cleared by record_reshard() and at the batch boundary.
        self._consumed_this_round: set[FsdpModule] = set()
        # Orientation of each module's most recent consume during replay,
        # used to decide whether a reshard can be skipped (storage only needs
        # to stay resident for an immediate same-orientation re-unshard).
        self._last_orientation: dict[FsdpModule, str] = {}
        # Diagnostics: how many events were validated during replay, how many
        # diverged (re-trace), and how many complete_trace calls ran.
        self._replayed_occurrences = 0
        self._divergences = 0
        self._complete_trace_calls = 0
        if use_trace_replay:
            logger.info("FsdpExecutionRunner: trace-and-replay prefetch enabled.")

    @property
    def phase(self) -> RunnerPhase:
        """Current runner phase."""
        return self._phase

    @property
    def is_tracing(self) -> bool:
        """Whether the runner is recording a fresh cycle."""
        return self._phase is RunnerPhase.TRACING

    @property
    def use_trace_replay(self) -> bool:
        """Whether trace-and-replay prefetch is enabled."""
        return self._use_trace_replay

    # ------------------------------------------------------------------
    # Interface 1: record execution events (consume, reshard)
    # ------------------------------------------------------------------

    def record_unshard(self, module: "FsdpModule", orientation: str) -> bool:
        """Record (tracing) or validate (replay) an unshard event.

        The fine-grained schedule fires one hook per sub-module (dense,
        experts), so the same module can arrive several times within a round;
        only the first arrival is a real unshard for the trace. Call
        ``suggest_prefetch()`` right after this returns to get the prefetch
        target (replay only).

        Args:
            module: The FSDP module being unsharded for compute.
            orientation: Payload orientation (``"rowwise"`` forward,
                ``"colwise"`` backward).
        """
        if not self._use_trace_replay:
            return True
        if module in self._consumed_this_round:
            return False
        self._consumed_this_round.add(module)
        self._last_orientation[module] = orientation
        self._validate_and_advance(EventKind.UNSHARD, module, orientation)
        return True

    def record_reshard(self, module: "FsdpModule") -> None:
        """Record (tracing) or validate (replay) a reshard event.

        The reshard ends the module's current unshard round: it clears the
        per-round dedup entry so the next unshard of the same module (e.g. the
        backward pass after the forward pass) records a fresh event. Call
        ``suggest_skip_reshard(module)`` right after this returns to learn
        whether the actual reshard can be skipped (replay only).

        Args:
            module: The FSDP module whose unsharded storage is released.
        """
        if not self._use_trace_replay:
            return
        # The reshard ends the module's unshard round; discard its dedup
        # entry so the next unshard (e.g. backward after forward) records a
        # fresh event.
        self._consumed_this_round.discard(module)
        self._validate_and_advance(EventKind.RESHARD, module, None)

    def record_completion(
        self,
        owner: "FsdpModule",
        anchor: "nn.Module | str",
        phase: "Literal['forward', 'backward']",
    ) -> None:
        """Record or validate a configured module-completion occurrence."""
        if not self._use_trace_replay:
            return
        self._validate_and_advance(EventKind.COMPLETION, owner, None, anchor=anchor, phase=phase)

    def record_reduce_scatter_release(self, owner: "FsdpModule", anchor: "nn.Module | str") -> None:
        """Record or validate a configured pre-backward RS release occurrence."""
        if not self._use_trace_replay:
            return
        self._validate_and_advance(EventKind.RS_RELEASE, owner, None, anchor=anchor)

    # ------------------------------------------------------------------
    # Interface 2: prefetch suggestion
    # ------------------------------------------------------------------

    def suggest_prefetch(
        self, module: "FsdpModule", orientation: str, *, depth: int = 1
    ) -> tuple["FsdpModule", str] | None:
        """Return a future module to all-gather ahead of this unshard.

        In default mode, resolves the static ``forward_order`` /
        ``backward_order`` successor. In trace-replay mode, returns the
        ``depth``-th future traced unshard (skipping other event kinds) with
        its recorded orientation. Replay lookahead never wraps across the
        global-batch boundary: parameters gathered before the optimizer step
        would contain stale weights afterward, and their live storage would
        also violate trace-pool planning at that boundary.

        Args:
            module: The FSDP module just unsharded for compute.
            orientation: Payload orientation (``"rowwise"`` forward,
                ``"colwise"`` backward).
            depth: One-based number of future traced unshard occurrences to
                look ahead. Values above one require trace replay.

        Returns:
            ``(module, orientation)`` to prefetch, or ``None`` while tracing
            or after a divergence.
        """
        if depth < 1:
            raise ValueError(f"Prefetch depth must be positive, got {depth}.")
        if not self._use_trace_replay:
            if depth != 1:
                raise ValueError("Prefetch depth greater than one requires trace replay.")
            return self._static_successor(module, orientation)
        # Tracing and divergence (re-trace) both disable prefetch; only a
        # validated replay cycle suggests a prefetch target.
        if self._phase is not RunnerPhase.REPLAYING or not self._trace:
            return None
        total_unshards = sum(event.kind is EventKind.UNSHARD for event in self._trace)
        if depth > total_unshards:
            raise ValueError(
                f"Prefetch depth {depth} exceeds the {total_unshards} "
                "UNSHARD occurrences in the replay trace."
            )

        remaining = depth
        for event in self._trace[self._replay_index :]:
            if event.kind is EventKind.UNSHARD:
                remaining -= 1
                if remaining == 0:
                    return event.module, event.orientation
        # Fewer than ``depth`` consumes remain in this global batch. Waiting
        # until the next batch starts keeps the gather after the optimizer
        # update and leaves the trace-pool boundary free of live allocations.
        return None

    # ------------------------------------------------------------------
    # Interface 3: reshard-skip suggestion
    # ------------------------------------------------------------------

    def suggest_skip_reshard(self, module: "FsdpModule") -> bool:
        """Return whether the reshard of ``module`` can be skipped.

        The optimization path: if the traced schedule immediately re-unshards
        the same module with the same orientation right after this reshard,
        the reshard is unnecessary — the storage can stay resident and the
        following all-gather can be skipped. Returns whether to skip the
        reshard.

        Args:
            module: The FSDP module whose unsharded storage is released.

        Returns:
            True to skip the actual reshard (keep storage resident), False to
            reshard normally.
        """
        if not self._use_trace_replay or self._phase is RunnerPhase.TRACING:
            return False
        if not self._trace:
            return False
        if self._replay_index >= len(self._trace):
            # Never retain a materialized parameter across the optimizer step.
            return False
        next_event = self._trace[self._replay_index]
        return (
            next_event.kind is EventKind.UNSHARD
            and next_event.module is module
            and next_event.orientation == self._last_orientation.get(module)
        )

    # ------------------------------------------------------------------
    # Lifecycle: batch boundary
    # ------------------------------------------------------------------

    def complete_trace(self) -> None:
        """Compile the recorded trace into the replay cycle.

        Called once by the optimizer at every global-batch boundary. The first
        batch (with a non-empty trace) transitions to ``REPLAYING``; subsequent
        calls reset the replay cursor for the next batch while keeping the
        compiled cycle.
        """
        if not self._use_trace_replay:
            return
        self._complete_trace_calls += 1
        if self._phase is RunnerPhase.TRACING and self._trace:
            self._phase = RunnerPhase.REPLAYING
            logger.info(
                "FsdpExecutionRunner: compiled %d-event trace, entering replay.",
                len(self._trace),
            )
        self._replay_index = 0
        # The batch boundary ends every module's unshard round; without this,
        # dedup entries from the trace batch (whose final unshards were never
        # followed by a reshard) would suppress the first replay unshards.
        self._consumed_this_round.clear()
        if self._phase is RunnerPhase.REPLAYING:
            self._cycles_observed += 1
        # Log every few batches so a training run shows whether replay is
        # actually validating events or stuck re-tracing.
        if self._complete_trace_calls % 10 == 0:
            self.report()

    def report(self) -> None:
        """Log the runner's replay statistics.

        A healthy runner shows ``cycles_observed`` increasing with every
        batch and ``replayed_occurrences`` much larger than ``divergences``.
        A runner that never replays (e.g. no complete_trace call, or a
        permanent divergence loop) is visible from this summary.
        """
        if self._use_trace_replay:
            logger.info(
                "FsdpExecutionRunner: phase=%s trace_len=%d cycles_observed=%d "
                "replayed_occurrences=%d divergences=%d complete_trace_calls=%d",
                self._phase.name,
                len(self._trace),
                self._cycles_observed,
                self._replayed_occurrences,
                self._divergences,
                self._complete_trace_calls,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _static_successor(
        self, module: "FsdpModule", orientation: str
    ) -> tuple["FsdpModule", str] | None:
        """Default mode: resolve the static-order successor.

        Activation recomputation runs forward hooks inside backward, whose
        forward-order prefetch must be skipped (its backward may already be
        complete and would not reshard the prefetched successor). Trace-replay
        mode owns all prefetch decisions and intentionally skips this check.
        """
        if getattr(module, "_phase", None) is not None and (
            module._phase == module.Phase.BACKWARD
            or torch._C._current_graph_task_id() != -1
        ):
            return None
        if orientation == "rowwise":
            next_module = self._context.forward_order.next_item(module)
        else:
            next_module = self._context.backward_order.next_item(module)
        if next_module is None:
            return None
        return next_module, orientation

    def _validate_and_advance(
        self,
        kind: EventKind,
        module: "FsdpModule",
        orientation: str | None,
        *,
        anchor: "nn.Module | str | None" = None,
        phase: "Literal['forward', 'backward'] | None" = None,
    ) -> None:
        """Trace (append) or validate-and-advance (replay) one event.

        The trace path records the real op stream (consume/reshard). During
        replay each real op is validated against the traced event at the
        current position; on success the cursor advances. A mismatch is a
        divergence: the trace is cleared and re-traced from this event,
        degrading to demand-only execution until a full cycle matches again.

        Args:
            kind: Expected event kind.
            module: The FSDP module the real op applies to.
            orientation: Expected orientation (``None`` for reshard).
        """
        if self._phase is RunnerPhase.TRACING:
            self._trace.append(
                RunnerEvent(
                    kind=kind, module=module, orientation=orientation, anchor=anchor, phase=phase
                )
            )
            return

        if self._replay_index >= len(self._trace):
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: replay emitted an event beyond the traced "
                "global-batch boundary. Re-tracing from this event (divergence #%d).",
                self._divergences,
            )
            self._retrace(kind, module, orientation, anchor=anchor, phase=phase)
            return

        expected = self._trace[self._replay_index]
        if (
            expected.kind is not kind
            or expected.module is not module
            or expected.orientation != orientation
            or not _same_anchor(expected.anchor, anchor)
            or expected.phase != phase
        ):
            # Schedule diverged from the trace (e.g. batch-size or topology
            # change). Re-trace from this event; prefetch stays disabled
            # until a full cycle matches again.
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: replay divergence at index %d: expected %s(%s), "
                "got %s(%s). Re-tracing from this event (divergence #%d).",
                self._replay_index,
                getattr(expected.module, "_name", None) or type(expected.module).__name__,
                expected.orientation,
                getattr(module, "_name", None) or type(module).__name__,
                orientation,
                self._divergences,
            )
            self._retrace(kind, module, orientation, anchor=anchor, phase=phase)
            return

        self._replayed_occurrences += 1
        self._replay_index += 1

    def _retrace(
        self,
        kind: EventKind,
        module: "FsdpModule",
        orientation: str | None,
        *,
        anchor: "nn.Module | str | None" = None,
        phase: "Literal['forward', 'backward'] | None" = None,
    ) -> None:
        """Reset to tracing and seed the new trace with the current event."""
        self._phase = RunnerPhase.TRACING
        self._trace = [
            RunnerEvent(
                kind=kind, module=module, orientation=orientation, anchor=anchor, phase=phase
            )
        ]
        self._replay_index = 0
        self._cycles_observed = 0
        # The divergence event ends the aborted replay round; dedup entries
        # from it must not suppress the re-traced remainder of the batch.
        # Re-mark the seed module for an unshard seed so duplicate hooks of
        # its current round stay deduped (a reshard seed ends that round).
        self._consumed_this_round.clear()
        self._last_orientation.clear()
        if kind is EventKind.UNSHARD:
            self._consumed_this_round.add(module)
        scheduler = getattr(self._context, "communication_scheduler", None)
        if scheduler is not None:
            scheduler.handle_replay_divergence()


def _same_anchor(left: "nn.Module | str | None", right: "nn.Module | str | None") -> bool:
    """Compare module anchors by identity and named anchors by value."""
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    return left is right
