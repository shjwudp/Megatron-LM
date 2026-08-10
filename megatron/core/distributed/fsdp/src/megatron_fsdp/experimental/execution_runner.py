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
path, so a per-context runner records the actual unshard consume sequence
during the first global batch and replays it from the second batch to drive
prefetch of the true next consumer.

State machine::

    TRACING --(complete_trace)--> REPLAYING --(divergence)--> TRACING
       ^                                                          |
       +----------------------(complete_trace)--------------------+

- ``TRACING``: every consume occurrence is appended to the trace; no
  prefetch is issued. The caller (the training loop) calls
  ``complete_trace()`` once per global batch boundary; the first batch
  compiles the trace into the replay cycle.
- ``REPLAYING``: each consume occurrence is validated against the traced
  cycle; the following occurrence (wrapping around at the batch boundary)
  is returned for prefetch. A mismatch clears the cycle and returns to
  ``TRACING``, degrading to demand-only execution until a full cycle
  matches again.
"""

import dataclasses
import logging
from enum import Enum, auto

import torch

# Forward reference; FsdpModule is imported lazily to avoid a cycle.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import FsdpModule

logger = logging.getLogger(__name__)


class RunnerPhase(Enum):
    """Lifecycle phase of an :class:`FsdpExecutionRunner`."""

    TRACING = auto()
    REPLAYING = auto()


@dataclasses.dataclass(frozen=True)
class ConsumeOccurrence:
    """One fine-grained parameter consumption.

    Attributes:
        module: The FSDP module whose full parameters are consumed.
        orientation: Payload orientation (``"rowwise"`` on forward,
            ``"colwise"`` on backward).
    """

    module: "FsdpModule"
    orientation: str


class FsdpExecutionRunner:
    """Record the fine-grained consume order and plan prefetches.

    The runner is owned by an :class:`FsdpContext` and driven by the
    fine-grained unshard entry point (``FsdpModule.unshard_parameters``)
    plus the global-batch boundary signaled by the training loop. It never
    decides compute order — it only observes occurrences and, during
    replay, suggests which module to all-gather next.

    Two prefetch modes are hidden behind one API:

    - Default (``use_trace_replay=False``): normal forward/backward
      execution is assumed. ``consume_and_get_next`` returns the static
      ``forward_order`` / ``backward_order`` successor and the runner
      stays idle.
    - Trace-replay (``use_trace_replay=True``): required for complex
      schedules such as VPP + combined 1F1B whose execution does not
      follow the static orders. The first global batch is traced and
      replayed from the second batch; ``consume_and_get_next`` returns
      the traced successor with its recorded orientation.
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
        self._trace: list[ConsumeOccurrence] = []
        self._replay_index = 0
        self._cycles_observed = 0
        # Diagnostics: how many occurrences were validated during replay, how
        # many diverged (re-trace), and how many complete_trace calls ran.
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

    def consume_and_get_next(
        self, module: "FsdpModule", orientation: str
    ) -> tuple["FsdpModule", str] | None:
        """Record/validate a consume and return the module to prefetch next.

        Unified prefetch API used by every unshard entry point
        (``pre_forward``, ``pre_backward``, ``unshard_parameters``). In
        default mode it resolves the static-order successor; in
        trace-replay mode it records the occurrence (tracing) or validates
        and returns the traced successor (replay).

        Args:
            module: The FSDP module being consumed by compute.
            orientation: Payload orientation (``"rowwise"`` forward,
                ``"colwise"`` backward).

        Returns:
            ``(next_module, next_orientation)`` to prefetch, or ``None``
            when there is no successor (tracing batch, divergence, or end
            of the static order).
        """
        if not self._use_trace_replay:
            # Default mode: activation recomputation runs forward hooks
            # inside backward, whose forward-order prefetch must be skipped
            # (its backward may already be complete and would not reshard the
            # prefetched successor). Trace-replay mode owns all prefetch
            # decisions and intentionally skips this check.
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

        next_module = self.record_consume(module, orientation)
        if next_module is None:
            return None
        return next_module, self.next_prefetch_orientation()

    def complete_trace(self) -> None:
        """Compile the recorded trace into the replay cycle.

        Called by the training loop at every global-batch boundary. The
        first batch (with a non-empty trace) transitions to ``REPLAYING``;
        subsequent calls reset the replay cursor for the next batch while
        keeping the compiled cycle.
        """
        self._complete_trace_calls += 1
        if self._phase is RunnerPhase.TRACING and self._trace:
            self._phase = RunnerPhase.REPLAYING
            logger.info(
                "FsdpExecutionRunner: compiled %d-occurrence trace, entering replay.",
                len(self._trace),
            )
        self._replay_index = 0
        if self._phase is RunnerPhase.REPLAYING:
            self._cycles_observed += 1
        # Log every few batches so a training run shows whether replay is
        # actually validating occurrences or stuck re-tracing.
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

    def record_consume(self, module: "FsdpModule", orientation: str) -> "FsdpModule | None":
        """Record (tracing) or validate (replay) one consume occurrence.

        Args:
            module: The FSDP module being consumed by compute.
            orientation: Payload orientation (``"rowwise"`` forward,
                ``"colwise"`` backward).

        Returns:
            During replay, the module to prefetch next (the traced successor,
            with the orientation recorded for that occurrence), or ``None``
            while tracing or after a divergence re-trace.
        """
        if self._phase is RunnerPhase.TRACING:
            self._trace.append(ConsumeOccurrence(module=module, orientation=orientation))
            return None

        expected = self._trace[self._replay_index]
        if expected.module is not module or expected.orientation != orientation:
            # Schedule diverged from the trace (e.g. batch-size or topology
            # change). Re-trace from this occurrence; prefetch stays disabled
            # until a full cycle matches again.
            self._divergences += 1
            logger.warning(
                "FsdpExecutionRunner: replay divergence at index %d: expected %s(%s), "
                "got %s(%s). Re-tracing from this occurrence (divergence #%d).",
                self._replay_index,
                getattr(expected.module, "_name", None) or type(expected.module).__name__,
                expected.orientation,
                getattr(module, "_name", None) or type(module).__name__,
                orientation,
                self._divergences,
            )
            self._retrace(module, orientation)
            return None

        self._replayed_occurrences += 1
        self._replay_index = (self._replay_index + 1) % len(self._trace)
        return self._trace[self._replay_index].module

    def next_prefetch_orientation(self) -> str | None:
        """Return the orientation recorded for the current replay successor.

        Only meaningful immediately after ``record_consume`` returned a
        module; the returned orientation matches that successor's occurrence.
        """
        if not self._trace:
            return None
        return self._trace[self._replay_index].orientation

    def _retrace(self, module: "FsdpModule", orientation: str) -> None:
        """Reset to tracing and seed the new trace with the current occurrence."""
        self._phase = RunnerPhase.TRACING
        self._trace = [ConsumeOccurrence(module=module, orientation=orientation)]
        self._replay_index = 0
        self._cycles_observed = 0
