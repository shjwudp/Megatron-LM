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


@dataclasses.dataclass(frozen=True)
class _PrefetchSuggestion:
    """One traced prefetch target and its last intervening reshard."""

    module: "FsdpModule"
    orientation: str
    release_after_reshard_index: int | None = None


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
        # Exact completion-event indices belonging to each traced unshard.
        # This is compiled once so VPP/microbatch occurrences cannot borrow
        # completion anchors from one another.
        self._completion_indices_by_unshard: dict[int, tuple[int, ...]] = {}
        self._unshard_index_by_completion: dict[int, int] = {}
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

    def record_reshard(self, module: "FsdpModule") -> int | None:
        """Record (tracing) or validate (replay) a reshard event.

        The reshard ends the module's current unshard round: it clears the
        per-round dedup entry so the next unshard of the same module (e.g. the
        backward pass after the forward pass) records a fresh event. Call
        ``suggest_skip_reshard(module)`` right after this returns to learn
        whether the actual reshard can be skipped (replay only).

        Args:
            module: The FSDP module whose unsharded storage is released.

        Returns:
            The exact replay trace index, or ``None`` outside validated replay.
        """
        if not self._use_trace_replay:
            return None
        # The reshard ends the module's unshard round; discard its dedup
        # entry so the next unshard (e.g. backward after forward) records a
        # fresh event.
        self._consumed_this_round.discard(module)
        return self._validate_and_advance(EventKind.RESHARD, module, None)

    def record_completion(
        self,
        owner: "FsdpModule",
        anchor: "nn.Module | str",
        phase: "Literal['forward', 'backward']",
    ) -> int | None:
        """Record or validate a configured module-completion occurrence.

        Returns:
            The exact trace index for a validated replay occurrence, or
            ``None`` while tracing, after divergence, or when replay is off.
        """
        if not self._use_trace_replay:
            return None
        return self._validate_and_advance(
            EventKind.COMPLETION, owner, None, anchor=anchor, phase=phase
        )

    def record_reduce_scatter_release(
        self, owner: "FsdpModule", anchor: "nn.Module | str"
    ) -> int | None:
        """Record or validate a configured pre-backward RS release occurrence.

        Returns:
            The exact trace index for both a newly recorded and a validated
            occurrence, or ``None`` when trace replay is disabled.
        """
        if not self._use_trace_replay:
            return None
        if self._phase is RunnerPhase.TRACING:
            self._validate_and_advance(EventKind.RS_RELEASE, owner, None, anchor=anchor)
            return len(self._trace) - 1
        trace_index = self._validate_and_advance(
            EventKind.RS_RELEASE, owner, None, anchor=anchor
        )
        # A replay mismatch seeds a replacement trace with this occurrence.
        # Return that seed index so scheduler metadata stays aligned with the
        # new trace rather than silently dropping its first release point.
        if trace_index is None and self._phase is RunnerPhase.TRACING:
            return len(self._trace) - 1
        return trace_index

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
        suggestion = self.suggest_prefetch_plan(module, orientation, depth=depth)
        if suggestion is None:
            return None
        return suggestion.module, suggestion.orientation

    def suggest_prefetch_plan(
        self, module: "FsdpModule", orientation: str, *, depth: int = 1
    ) -> _PrefetchSuggestion | None:
        """Return a future target plus any intervening target-reshard gate.

        Deep trace lookahead can name an occurrence of a module whose current
        materialization will be consumed and resharded by an earlier occurrence.
        The returned gate lets the scheduler wait for that physical reshard
        instead of issuing a gather that cannot survive until its target.
        """
        if depth < 1:
            raise ValueError(f"Prefetch depth must be positive, got {depth}.")
        if not self._use_trace_replay:
            if depth != 1:
                raise ValueError("Prefetch depth greater than one requires trace replay.")
            successor = self._static_successor(module, orientation)
            if successor is None:
                return None
            return _PrefetchSuggestion(*successor)
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
        target_index = None
        target_event = None
        for index, event in enumerate(
            self._trace[self._replay_index :], start=self._replay_index
        ):
            if event.kind is EventKind.UNSHARD:
                remaining -= 1
                if remaining == 0:
                    target_index = index
                    target_event = event
                    break
        if target_event is not None:
            assert target_index is not None and target_event.orientation is not None
            release_after_reshard_index = None
            for index in range(self._replay_index, target_index):
                event = self._trace[index]
                if (
                    event.kind is EventKind.RESHARD
                    and event.module is target_event.module
                    and not self._trace_reshard_is_skipped(index)
                ):
                    release_after_reshard_index = index
            return _PrefetchSuggestion(
                target_event.module,
                target_event.orientation,
                release_after_reshard_index,
            )
        # Fewer than ``depth`` consumes remain in this global batch. Waiting
        # until the next batch starts keeps the gather after the optimizer
        # update and leaves the trace-pool boundary free of live allocations.
        return None

    def _trace_reshard_is_skipped(self, reshard_index: int) -> bool:
        """Return whether replay retains storage across one traced reshard.

        ``suggest_skip_reshard()`` keeps a materialization live when the trace
        immediately re-unshards the same module with the same orientation. A
        deep-prefetch lifetime gate must ignore that logical RESHARD because no
        physical release occurs.
        """
        event = self._trace[reshard_index]
        if event.kind is not EventKind.RESHARD:
            raise ValueError("A reshard-skip query requires a RESHARD trace event.")
        if reshard_index + 1 >= len(self._trace):
            return False
        next_event = self._trace[reshard_index + 1]
        if next_event.kind is not EventKind.UNSHARD or next_event.module is not event.module:
            return False

        for previous in reversed(self._trace[:reshard_index]):
            if previous.kind is not EventKind.UNSHARD or previous.module is not event.module:
                continue
            return next_event.orientation == previous.orientation
        return False

    def completion_indices_for_current_unshard(
        self, module: "FsdpModule", orientation: str
    ) -> tuple[int, ...]:
        """Return completion occurrences assigned to the current unshard.

        ``record_unshard()`` must have just validated ``module`` and
        ``orientation``. The returned trace indices are exact occurrence
        tokens: completion occurrence *n* is paired with unshard occurrence
        *n* for the same owner and phase. A completion that ran before this
        unshard can therefore be reused without leaking across interleaved
        VPP occurrences.
        """
        if not self._use_trace_replay or self._phase is not RunnerPhase.REPLAYING:
            return ()
        source_index = self._replay_index - 1
        if source_index < 0:
            return ()
        source_event = self._trace[source_index]
        if (
            source_event.kind is not EventKind.UNSHARD
            or source_event.module is not module
            or source_event.orientation != orientation
        ):
            raise RuntimeError(
                "Completion lookup must immediately follow the source unshard occurrence."
            )
        return self._completion_indices_by_unshard.get(source_index, ())

    def completion_precedes_source(self, completion_index: int) -> bool:
        """Return whether this completion must be retained for a later request."""
        source_index = self._unshard_index_by_completion.get(completion_index)
        return source_index is not None and completion_index < source_index

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
            self._compile_completion_occurrences()
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
    ) -> int | None:
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
            return None

        if self._replay_index >= len(self._trace):
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: replay emitted an event beyond the traced "
                "global-batch boundary. Re-tracing from this event (divergence #%d).",
                self._divergences,
            )
            self._retrace(kind, module, orientation, anchor=anchor, phase=phase)
            return None

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
            return None

        replayed_index = self._replay_index
        self._replayed_occurrences += 1
        self._replay_index += 1
        return replayed_index

    def _compile_completion_occurrences(self) -> None:
        """Pair configured completions and source unshards by occurrence.

        Completion hooks may run before or after the first parameterized
        submodule unshards its owning FSDP unit. Pairing by event adjacency is
        therefore insufficient. For every owner and phase, the *n*-th
        occurrence of each configured anchor belongs to the *n*-th unshard.
        A cardinality mismatch is left unmapped so demand unshard remains the
        safe fallback instead of borrowing an anchor from another occurrence.
        """
        unshards: dict[tuple[int, str], list[int]] = {}
        completions: dict[tuple[int, str, tuple[str, object]], list[int]] = {}
        completion_labels: dict[tuple[int, str, tuple[str, object]], str] = {}

        for index, event in enumerate(self._trace):
            if event.kind is EventKind.UNSHARD:
                assert event.orientation is not None
                phase = _phase_for_orientation(event.orientation)
                unshards.setdefault((id(event.module), phase), []).append(index)
                continue
            if event.kind is not EventKind.COMPLETION:
                continue
            assert event.anchor is not None and event.phase is not None
            anchor_key = _anchor_key(event.anchor)
            key = (id(event.module), event.phase, anchor_key)
            completions.setdefault(key, []).append(index)
            completion_labels[key] = _anchor_label(event.anchor)

        mapped: dict[int, list[int]] = {}
        for (module_id, phase, anchor_key), completion_indices in completions.items():
            unshard_indices = unshards.get((module_id, phase), [])
            if len(completion_indices) != len(unshard_indices):
                logger.warning(
                    "FsdpExecutionRunner: cannot occurrence-map completion %s/%s: "
                    "%d completions for %d unshards; demand fallback remains enabled.",
                    completion_labels[(module_id, phase, anchor_key)],
                    phase,
                    len(completion_indices),
                    len(unshard_indices),
                )
                continue
            for unshard_index, completion_index in zip(unshard_indices, completion_indices):
                mapped.setdefault(unshard_index, []).append(completion_index)

        self._completion_indices_by_unshard = {
            unshard_index: tuple(sorted(completion_indices))
            for unshard_index, completion_indices in mapped.items()
        }
        self._unshard_index_by_completion = {
            completion_index: unshard_index
            for unshard_index, completion_indices in mapped.items()
            for completion_index in completion_indices
        }

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
        self._completion_indices_by_unshard.clear()
        self._unshard_index_by_completion.clear()
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


def _phase_for_orientation(orientation: str) -> "Literal['forward', 'backward']":
    """Translate parameter payload orientation into its execution phase."""
    if orientation == "rowwise":
        return "forward"
    if orientation == "colwise":
        return "backward"
    raise ValueError(f"Unsupported parameter orientation: {orientation!r}.")


def _anchor_key(anchor: "nn.Module | str") -> tuple[str, object]:
    """Return an identity-safe grouping key for one completion anchor."""
    if isinstance(anchor, str):
        return ("named", anchor)
    return ("module", id(anchor))


def _anchor_label(anchor: "nn.Module | str") -> str:
    """Return a compact diagnostic label for one completion anchor."""
    if isinstance(anchor, str):
        return f"@{anchor}"
    return type(anchor).__name__
