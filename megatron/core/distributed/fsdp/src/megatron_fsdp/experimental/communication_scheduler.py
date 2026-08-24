# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trace-guided communication scheduling for experimental Megatron-FSDP."""

import dataclasses
import logging
import math
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh

if TYPE_CHECKING:
    from .dbuffer import DBuffer
    from .module import FsdpContext, FsdpModule
    from .parameter_group import FsdpParameterGroup

logger = logging.getLogger(__name__)

CommunicationPhase = Literal["forward", "backward"]


@dataclasses.dataclass(frozen=True)
class ModuleCompletion:
    """A module occurrence after which successor prefetch may be released.

    Args:
        module: Descendant module whose completion supplies the release event.
        phase: Whether to observe the module's forward or backward completion.
    """

    module: nn.Module
    phase: CommunicationPhase

    def __post_init__(self) -> None:
        if self.phase not in ("forward", "backward"):
            raise ValueError(f"Unsupported module-completion phase: {self.phase!r}.")


@dataclasses.dataclass(frozen=True)
class NamedCompletion:
    """A completion point emitted explicitly by an external execution engine."""

    name: str
    phase: CommunicationPhase

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A named completion anchor requires a non-empty name.")
        if self.phase not in ("forward", "backward"):
            raise ValueError(f"Unsupported named-completion phase: {self.phase!r}.")


@dataclasses.dataclass(frozen=True)
class NamedPreBackward:
    """A pre-backward release point emitted by an external execution engine."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A named pre-backward anchor requires a non-empty name.")


@dataclasses.dataclass(frozen=True)
class FsdpModuleCommunicationPolicy:
    """Communication release points owned by one FSDP unit.

    Args:
        prefetch_successor_after: Completion points that may release this unit's
            traced successor all-gather.
        reduce_scatter_release_on_pre_backward: Descendant modules whose
            pre-backward entry may release a trace-bounded set of pending
            reduce-scatter requests.
        conflict_free_on_pre_backward: Descendant modules whose pre-backward
            entry is known to be outside competing communication. The point
            admits context-wide queued prefetches from configured units
            (subject to lifetime and residency limits) and every ready
            deferred reduce-scatter.
    """

    prefetch_successor_after: tuple[ModuleCompletion | NamedCompletion, ...] = ()
    reduce_scatter_release_on_pre_backward: tuple[nn.Module | NamedPreBackward, ...] = ()
    conflict_free_on_pre_backward: tuple[nn.Module | NamedPreBackward, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether this policy preserves eager communication."""
        return not (
            self.prefetch_successor_after
            or self.reduce_scatter_release_on_pre_backward
            or self.conflict_free_on_pre_backward
        )


@dataclasses.dataclass(frozen=True)
class FsdpCommunicationSchedulerConfig:
    """Context-wide limits for trace-guided communication scheduling.

    Args:
        max_pending_reduce_scatter_bytes: ``None`` infers a pending-byte limit
            from the trace, zero keeps reduce-scatter eager, and a positive
            value supplies an explicit limit.
        prefetch_depth: One-based index of the future traced ``UNSHARD``
            occurrence to prefetch. One preserves immediate-successor prefetch;
            larger values trade parameter residency for more lead time.
        max_prefetch_resident_bytes: ``None`` preserves unbounded prefetch
            residency, zero infers a one-materialization byte budget from the
            execution trace, and a positive value supplies an explicit limit.
        reduce_scatter_release_on_prefetch: Use each actual parameter-prefetch
            all-gather submission as an opportunity to release at most one ready
            deferred reduce-scatter. The two collectives remain asynchronous.
    """

    max_pending_reduce_scatter_bytes: int | None = None
    prefetch_depth: int = 1
    max_prefetch_resident_bytes: int | None = None
    reduce_scatter_release_on_prefetch: bool = False

    def __post_init__(self) -> None:
        value = self.max_pending_reduce_scatter_bytes
        if value is not None and value < 0:
            raise ValueError(
                "max_pending_reduce_scatter_bytes must be None or non-negative, " f"got {value}."
            )
        if self.prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be positive, got {self.prefetch_depth}.")
        resident_bytes = self.max_prefetch_resident_bytes
        if resident_bytes is not None and resident_bytes < 0:
            raise ValueError(
                "max_prefetch_resident_bytes must be None or non-negative, "
                f"got {resident_bytes}."
            )


@dataclasses.dataclass
class _PendingPrefetch:
    """One traced successor gather waiting for its scheduling gates."""

    sequence: int
    source: "FsdpModule"
    source_phase: CommunicationPhase
    target: "FsdpModule"
    target_orientation: str
    completion_indices: tuple[int, ...]
    completion_required: bool = False
    conflict_free_required: bool = False
    target_reshard_index: int | None = None
    target_unshard_index: int | None = None
    size_bytes: int = 0
    completed_anchor: "_CompletedPrefetchAnchor | None" = None
    retained_through_reshard: bool = False
    residency_deferred: bool = False


@dataclasses.dataclass
class _CompletedPrefetchAnchor:
    """One replay completion retained until its source request is known."""

    owner: "FsdpModule"
    anchor: nn.Module | str
    phase: CommunicationPhase
    event: torch.cuda.Event


class _ReduceScatterState(Enum):
    """Host-visible readiness of a deferred reduce-scatter request."""

    WRITING = auto()
    READY = auto()


@dataclasses.dataclass
class _DomainState:
    """Pending-byte state for one DeviceMesh collective domain."""

    mesh: DeviceMesh
    pending_bytes: int = 0
    in_flight_bytes: int = 0
    effective_budget: int = 0
    required_peak: int = 0
    trace_pending_bytes: int = 0
    trace_request_sizes: list[int] = dataclasses.field(default_factory=list)
    min_free_bytes: int | None = None
    total_device_bytes: int | None = None


@dataclasses.dataclass
class _TraceReduceScatterRequest:
    """Virtual deferred request used to infer a replay budget."""

    sequence: int
    domain: _DomainState
    size_bytes: int
    ready: bool = False
    active: bool = True


@dataclasses.dataclass
class _ReduceScatterRequest:
    """One physical reduce-scatter input from allocation through submission."""

    sequence: int
    domain: _DomainState
    group: "FsdpParameterGroup"
    module_name: str
    group_index: int
    size_bytes: int
    deferred: bool
    trace_request: _TraceReduceScatterRequest | None
    state: _ReduceScatterState = _ReduceScatterState.WRITING
    partial_grad: "DBuffer | None" = None
    ready_event: torch.cuda.Event | None = None
    is_last_microbatch: bool = True


@dataclasses.dataclass
class _InFlightReduceScatter:
    """One submitted RS input whose stream-ordered lifetime has not retired."""

    sequence: int
    domain: _DomainState
    module_name: str
    group_index: int
    size_bytes: int
    completion_event: torch.cuda.Event


class FsdpCommunicationScheduler:
    """Context-owned scheduler for delayed parameter and gradient collectives."""

    def __init__(self, context: "FsdpContext", config: FsdpCommunicationSchedulerConfig) -> None:
        """Create an empty scheduler for ``context``."""
        self._context = context
        self.config = config
        self._pending_prefetches: deque[_PendingPrefetch] = deque()
        self._retained_prefetches: deque[_PendingPrefetch] = deque()
        self._resident_prefetches: deque[_PendingPrefetch] = deque()
        self._resident_prefetch_bytes = 0
        self._effective_prefetch_resident_bytes: int | None = None
        self._completed_prefetch_anchors: dict[int, _CompletedPrefetchAnchor] = {}
        self._next_prefetch_sequence = 0
        self._delayed_prefetches = 0
        self._anchor_releases = 0
        self._latched_anchor_releases = 0
        self._demand_releases = 0
        self._retained_prefetch_reuses = 0
        self._residency_deferrals = 0
        self._residency_releases = 0
        self._domains: dict[int, _DomainState] = {}
        self._domain_order: list[_DomainState] = []
        self._reduce_scatter_requests: deque[_ReduceScatterRequest] = deque()
        self._active_request_by_group: dict[FsdpParameterGroup, _ReduceScatterRequest] = {}
        self._trace_reduce_scatter_requests: deque[_TraceReduceScatterRequest] = deque()
        self._trace_reduce_scatter_release_credits: dict[int, dict[int, int]] = {}
        self._in_flight_reduce_scatters: deque[_InFlightReduceScatter] = deque()
        self._next_reduce_scatter_sequence = 0
        self._release_anchor_count = 0
        self._budget_compiled = False
        self._reduce_scatter_releases = 0
        self._prefetch_reduce_scatter_releases = 0
        self._capacity_releases = 0
        self._final_releases = 0
        self._peak_pending_reduce_scatter_bytes = 0
        self._peak_in_flight_reduce_scatter_bytes = 0
        self._peak_active_reduce_scatter_bytes = 0

    @property
    def has_pending_prefetches(self) -> bool:
        """Return whether a successor gather is waiting for an anchor."""
        return bool(
            self._pending_prefetches or self._retained_prefetches or self._resident_prefetches
        )

    @property
    def effective_prefetch_resident_bytes(self) -> int | None:
        """Return the compiled resident-prefetch byte budget, if enabled."""
        return self._effective_prefetch_resident_bytes

    @property
    def pending_reduce_scatter_bytes(self) -> int:
        """Return deferred, unsubmitted reduce-scatter bytes across domains."""
        return sum(domain.pending_bytes for domain in self._domain_order)

    @property
    def effective_reduce_scatter_budgets(self) -> tuple[int, ...]:
        """Return inferred/overridden budgets in domain registration order."""
        return tuple(domain.effective_budget for domain in self._domain_order)

    @property
    def in_flight_reduce_scatter_bytes(self) -> int:
        """Return submitted RS-input bytes whose completion has not been observed."""
        return sum(domain.in_flight_bytes for domain in self._domain_order)

    @property
    def peak_pending_reduce_scatter_bytes(self) -> int:
        """Return the largest observed deferred, unsubmitted RS footprint."""
        return self._peak_pending_reduce_scatter_bytes

    @property
    def peak_in_flight_reduce_scatter_bytes(self) -> int:
        """Return the largest observed submitted, unretired RS footprint."""
        return self._peak_in_flight_reduce_scatter_bytes

    @property
    def peak_active_reduce_scatter_bytes(self) -> int:
        """Return the largest observed pending plus in-flight RS footprint."""
        return self._peak_active_reduce_scatter_bytes

    def register_reduce_scatter_release_anchor(
        self, owner: "FsdpModule", anchor: nn.Module | str
    ) -> None:
        """Register one unique pre-backward release point during construction."""
        del owner, anchor
        self._release_anchor_count += 1

    def schedule_prefetch(
        self,
        source: "FsdpModule",
        source_orientation: str,
        target: "FsdpModule",
        target_orientation: str,
        *,
        target_reshard_index: int | None = None,
        target_unshard_index: int | None = None,
    ) -> None:
        """Submit or defer a traced successor all-gather.

        Completion anchors belong to the source occurrence, while the payload
        orientation belongs to the target occurrence. These orientations may
        differ in an interleaved 1F1B trace.
        """
        source_phase: CommunicationPhase = (
            "forward" if source_orientation == "rowwise" else "backward"
        )
        runner = self._context.runner
        completions = tuple(
            completion
            for completion in source.communication_policy.prefetch_successor_after
            if completion.phase == source_phase
        )
        conflict_free_required = bool(
            source.communication_policy.conflict_free_on_pre_backward
        )
        completion_required = bool(completions)
        residency_limited = self._prefetch_residency_is_limited()
        if runner.is_tracing or (
            not residency_limited
            and not completion_required
            and not conflict_free_required
            and target_reshard_index is None
        ):
            reason = "trace-prefetch" if runner.is_tracing else "eager-prefetch"
            source_name = source.name if source.name else "<root>"
            target._unshard_parameter_groups(
                target_orientation, reason=reason, source=source_name, source_phase=source_phase
            )
            return

        pending = _PendingPrefetch(
            sequence=self._next_prefetch_sequence,
            source=source,
            source_phase=source_phase,
            target=target,
            target_orientation=target_orientation,
            completion_indices=(
                runner.completion_indices_for_current_unshard(source, source_orientation)
                if completion_required
                else ()
            ),
            completion_required=completion_required,
            conflict_free_required=conflict_free_required,
            target_reshard_index=target_reshard_index,
            target_unshard_index=target_unshard_index,
            size_bytes=target.unsharded_parameter_nbytes(),
        )
        self._next_prefetch_sequence += 1
        self._delayed_prefetches += 1
        source_name = source.name if source.name else "<root>"
        target_name = target.name if target.name else "<root>"
        torch.cuda.nvtx.range_push(
            f"MFSDP AG queued request={pending.sequence} source={source_name} "
            f"source_phase={source_phase} source_orientation={source_orientation} "
            f"target={target_name} "
            f"target_orientation={target_orientation} "
            f"target_reshard_index={target_reshard_index} "
            f"target_unshard_index={target_unshard_index} bytes={pending.size_bytes}"
        )
        torch.cuda.nvtx.range_pop()

        pending.completed_anchor = self._pop_latched_completion(pending)
        if residency_limited:
            self._pending_prefetches.append(pending)
            self._drain_ready_prefetches(reason="queue")
            return
        if self._prefetch_is_ready(pending):
            reason = "latched-anchor" if pending.completed_anchor is not None else "post-reshard"
            self._submit_prefetch(pending, reason=reason)
            if pending.completed_anchor is not None:
                self._latched_anchor_releases += 1
            return

        self._pending_prefetches.append(pending)

    def record_completion_anchor(
        self,
        owner: "FsdpModule",
        anchor: nn.Module | str,
        phase: CommunicationPhase,
        event: torch.cuda.Event | None = None,
    ) -> None:
        """Record an anchor occurrence and release its pending successor.

        Args:
            owner: FSDP unit whose successor policy contains ``anchor``.
            anchor: Module that just completed execution.
            phase: Completed execution phase.
            event: Optional event already recorded on the anchor's execution
                stream. A fresh event is recorded on the current stream when
                omitted.
        """
        runner = self._context.runner
        trace_index = runner.record_completion(owner, anchor, phase)
        if runner.is_tracing:
            # A replay divergence may leave speculative requests from the
            # aborted cycle. Submit them before tracing a replacement cycle.
            self.flush_prefetches(reason="trace")
            return

        if trace_index is None:
            return
        pending = self._find_matching_prefetch(trace_index)
        if pending is None:
            if not runner.completion_precedes_source(trace_index):
                return
            if event is None:
                event = self._context.current_stream().record_event()
            self._completed_prefetch_anchors[trace_index] = _CompletedPrefetchAnchor(
                owner=owner, anchor=anchor, phase=phase, event=event
            )
            return
        if event is None:
            event = self._context.current_stream().record_event()
        pending.completed_anchor = _CompletedPrefetchAnchor(
            owner=owner, anchor=anchor, phase=phase, event=event
        )
        if self._prefetch_is_ready(pending):
            if self._prefetch_residency_is_limited():
                self._drain_ready_prefetches(reason="anchor")
                return
            self._pending_prefetches.remove(pending)
            self._submit_prefetch(pending, reason="anchor")

    def record_target_reshard(self, target: "FsdpModule", trace_index: int | None) -> None:
        """Release prefetches gated by one exact physical target reshard."""
        if trace_index is None:
            return
        matched = False
        for pending in tuple(self._pending_prefetches):
            if pending.target is not target or pending.target_reshard_index != trace_index:
                continue
            pending.target_reshard_index = None
            matched = True
            if self._prefetch_residency_is_limited():
                continue
            if not self._prefetch_is_ready(pending):
                continue
            self._pending_prefetches.remove(pending)
            reason = "post-reshard"
            if pending.completed_anchor is not None:
                reason = "anchor+post-reshard"
            self._submit_prefetch(pending, reason=reason)
        if matched and self._prefetch_residency_is_limited():
            self._drain_ready_prefetches(reason="post-reshard")

    def retain_prefetches_across_reshard(
        self, target: "FsdpModule", trace_index: int | None
    ) -> bool:
        """Keep live parameters when they already serve a future depth target."""
        if trace_index is None:
            # Replay divergence retraces the rest of the current global batch.
            # Keep a prior reservation alive until demand or the optimizer boundary.
            return any(
                pending.target is target
                for pending in (*self._retained_prefetches, *self._resident_prefetches)
            )
        matches = [
            pending
            for pending in self._pending_prefetches
            if pending.target is target and pending.target_reshard_index == trace_index
        ]
        if not matches:
            return any(
                pending.target is target
                for pending in (*self._retained_prefetches, *self._resident_prefetches)
            )
        if any(
            pending.completion_required and pending.completed_anchor is None for pending in matches
        ):
            return False
        if self._prefetch_residency_is_limited():
            return self._retain_prefetch_with_residency_budget(target, matches)

        target_name = target.name if target.name else "<root>"
        for pending in matches:
            self._pending_prefetches.remove(pending)
            pending.target_reshard_index = None
            self._retained_prefetches.append(pending)
            completed = pending.completed_anchor
            anchor = (
                _completion_name(pending.source, completed.anchor)
                if completed is not None
                else None
            )
            provenance = "".join(
                f" {key}={value}"
                for key, value in (
                    ("source", pending.source.name if pending.source.name else "<root>"),
                    ("source_phase", pending.source_phase),
                    ("anchor", anchor),
                    ("request", pending.sequence),
                )
                if value is not None
            )
            torch.cuda.nvtx.range_push(
                f"MFSDP AG retained target={target_name} "
                f"orientation={pending.target_orientation} "
                f"trigger=retain-through-reshard{provenance}"
            )
            torch.cuda.nvtx.range_pop()
            if completed is not None:
                self._anchor_releases += 1
        return True

    def demand_unshard(self, target: "FsdpModule", orientation: str) -> None:
        """Consume retained storage or release the oldest queued target gather."""
        for index, resident in enumerate(self._resident_prefetches):
            if resident.target is not target or resident.target_orientation != orientation:
                continue
            del self._resident_prefetches[index]
            self._resident_prefetch_bytes -= resident.size_bytes
            target_name = target.name if target.name else "<root>"
            reuse_kind = "retained" if resident.retained_through_reshard else "resident"
            torch.cuda.nvtx.range_push(
                f"MFSDP AG {reuse_kind} reuse target={target_name} "
                f"orientation={orientation} reserved_orientation="
                f"{resident.target_orientation} request={resident.sequence} "
                f"bytes={resident.size_bytes}"
            )
            torch.cuda.nvtx.range_pop()
            if resident.retained_through_reshard:
                self._retained_prefetch_reuses += 1
            self._residency_releases += 1
            self._drain_ready_prefetches(reason="resident-reuse")
            return
        for index, retained in enumerate(self._retained_prefetches):
            if retained.target is not target:
                continue
            del self._retained_prefetches[index]
            target_name = target.name if target.name else "<root>"
            torch.cuda.nvtx.range_push(
                f"MFSDP AG retained reuse target={target_name} "
                f"orientation={orientation} reserved_orientation="
                f"{retained.target_orientation} request={retained.sequence}"
            )
            torch.cuda.nvtx.range_pop()
            self._retained_prefetch_reuses += 1
            return
        for index, pending in enumerate(self._pending_prefetches):
            if pending.target is not target or pending.target_orientation != orientation:
                continue
            # A gate marks an intervening occurrence of the same module. This
            # demand must not steal a request reserved for a later occurrence.
            if pending.target_reshard_index is not None:
                return
            del self._pending_prefetches[index]
            pending.target._unshard_parameter_groups(
                pending.target_orientation,
                reason="demand",
                source=pending.source.name if pending.source.name else "<root>",
                source_phase=pending.source_phase,
                request=pending.sequence,
            )
            self._demand_releases += 1
            if self._prefetch_residency_is_limited():
                self._drain_ready_prefetches(reason="demand")
            return

    def record_reduce_scatter_release(
        self, owner: "FsdpModule", anchor: nn.Module | str, demand_event: torch.cuda.Event | None
    ) -> None:
        """Observe a configured pre-backward reduce-scatter release point."""
        self._retire_completed_reduce_scatters()
        runner = self._context.runner
        trace_index = runner.record_reduce_scatter_release(owner, anchor)
        if runner.is_tracing:
            release_credits = self._trace_reduce_scatter_release()
            if trace_index is not None:
                self._trace_reduce_scatter_release_credits[trace_index] = release_credits
            return
        if not self._budget_compiled or trace_index is None:
            return

        release_credits = dict(self._trace_reduce_scatter_release_credits.get(trace_index, {}))
        while True:
            request = self._oldest_releasable_request(release_credits)
            if request is None:
                break
            release_credits[id(request.domain)] -= request.size_bytes
            self._submit_reduce_scatter(request, reason="anchor", demand_event=demand_event)
            self._reduce_scatter_releases += 1

    def record_conflict_free_point(
        self,
        owner: "FsdpModule",
        anchor: nn.Module | str,
        demand_event: torch.cuda.Event | None,
    ) -> None:
        """Release eligible AG prefetches and every ready deferred RS."""
        self._retire_completed_reduce_scatters()
        runner = self._context.runner
        runner.record_conflict_free_point(owner, anchor)
        if runner.is_tracing:
            # Model the same complete ready-RS drain while inferring the replay
            # budget. Prefetch remains disabled during the trace batch.
            self.flush_prefetches(reason="trace")
            self._trace_reduce_scatter_release()
            return

        while True:
            eligible = [
                pending
                for pending in self._pending_prefetches
                if pending.conflict_free_required and self._prefetch_is_ready(pending)
            ]
            if not eligible:
                break
            winner = min(eligible, key=self._prefetch_deadline_key)
            if self._prefetch_residency_is_limited() and not self._resident_capacity_allows(
                winner
            ):
                self._mark_residency_deferred(winner, reason="conflict-free-capacity")
                break
            if demand_event is None:
                demand_event = self._context.current_stream().record_event()
            self._context.allgather_stream.wait_event(demand_event)
            self._pending_prefetches.remove(winner)
            self._submit_prefetch(winner, reason="conflict-free")

        if not self._budget_compiled:
            return
        while True:
            request = self._oldest_ready_deferred_request()
            if request is None:
                return
            self._submit_reduce_scatter(
                request, reason="conflict-free", demand_event=demand_event
            )
            self._reduce_scatter_releases += 1

    def reserve_reduce_scatter(
        self, group: "FsdpParameterGroup", *, module_name: str, group_index: int
    ) -> None:
        """Reserve pending capacity before allocating a full gradient buffer."""
        self._retire_completed_reduce_scatters()
        if group in self._active_request_by_group:
            previous = self._active_request_by_group[group]
            if previous.state is not _ReduceScatterState.READY:
                raise RuntimeError(
                    "A second reduce-scatter buffer was requested while the previous "
                    "buffer for this parameter group was still being written."
                )
            while self._oldest_domain_request(previous.domain) is not previous:
                oldest = self._oldest_domain_request(previous.domain)
                if oldest is None or oldest.state is not _ReduceScatterState.READY:
                    raise RuntimeError(
                        "Cannot reuse a reduce-scatter buffer while an older request "
                        "in its collective domain is still being written."
                    )
                self._submit_reduce_scatter(oldest, reason="same-group reuse")
                self._capacity_releases += 1
            self._submit_reduce_scatter(previous, reason="same-group reuse")
            self._capacity_releases += 1

        domain = self._get_domain(group.mesh)
        size_bytes = group.partial_grad_nbytes()
        trace_request = self._trace_reduce_scatter_reserve(domain, size_bytes)

        deferred = False
        if self._budget_compiled and not self._context.runner.is_tracing:
            budget = domain.effective_budget
            while domain.pending_bytes + size_bytes > budget:
                oldest = self._oldest_domain_request(domain)
                if oldest is None or oldest.state is not _ReduceScatterState.READY:
                    break
                self._submit_reduce_scatter(oldest, reason="capacity")
                self._capacity_releases += 1
            deferred = budget > 0 and domain.pending_bytes + size_bytes <= budget

        request = _ReduceScatterRequest(
            sequence=self._next_reduce_scatter_sequence,
            domain=domain,
            group=group,
            module_name=module_name,
            group_index=group_index,
            size_bytes=size_bytes,
            deferred=deferred,
            trace_request=trace_request,
        )
        self._next_reduce_scatter_sequence += 1
        self._reduce_scatter_requests.append(request)
        self._active_request_by_group[group] = request
        if deferred:
            domain.pending_bytes += size_bytes
        self._update_reduce_scatter_peaks()
        self._emit_reduce_scatter_state("reserve", request, deferred=deferred)

    def cancel_reduce_scatter_reservation(self, group: "FsdpParameterGroup") -> None:
        """Cancel a reservation whose physical allocation failed."""
        request = self._active_request_by_group.pop(group, None)
        if request is None:
            return
        if request.deferred:
            request.domain.pending_bytes -= request.size_bytes
        if request.trace_request is not None and request.trace_request.active:
            request.trace_request.active = False
            request.trace_request.domain.trace_pending_bytes -= request.size_bytes
        self._reduce_scatter_requests.remove(request)
        self._emit_reduce_scatter_state("cancel", request, deferred=request.deferred)

    def mark_reduce_scatter_ready(
        self,
        group: "FsdpParameterGroup",
        partial_grad: "DBuffer",
        ready_event: torch.cuda.Event,
        is_last_microbatch: bool,
    ) -> None:
        """Attach completed gradient storage and submit or queue its collective."""
        request = self._active_request_by_group.get(group)
        if request is None:
            raise RuntimeError("Reduce-scatter became ready without a buffer reservation.")
        request.partial_grad = partial_grad
        request.ready_event = ready_event
        request.is_last_microbatch = is_last_microbatch
        request.state = _ReduceScatterState.READY
        if request.trace_request is not None:
            request.trace_request.ready = True

        self._emit_reduce_scatter_state("ready", request, deferred=request.deferred)

        # A submit-on-ready request cannot overtake an older collective in the
        # same domain. Force ready domain heads until every non-deferred request
        # that can legally launch has done so.
        self._drain_submit_on_ready_requests(request.domain)

    def finish_grad_sync(self) -> None:
        """Submit every ready request and fence gradient consumers against RS."""
        self._retire_completed_reduce_scatters()
        self._release_retained_prefetches(reason="finish_grad_sync")
        self._trace_reduce_scatter_flush()
        while self._reduce_scatter_requests:
            request = self._reduce_scatter_requests[0]
            if request.state is not _ReduceScatterState.READY:
                raise RuntimeError(
                    "finish_grad_sync reached a reduce-scatter buffer that is still being written."
                )
            self._submit_reduce_scatter(request, reason="finish_grad_sync")
            self._final_releases += 1
        self._context.current_stream().wait_stream(self._context.reduce_scatter_stream)

    def complete_trace(self, *, runner_was_tracing: bool) -> None:
        """Infer and freeze pending-byte limits after an execution trace."""
        # Trace indices repeat in every global batch. Never let a completed
        # CUDA event from the prior batch satisfy the next batch's occurrence.
        self._completed_prefetch_anchors.clear()
        if not runner_was_tracing:
            return
        self._compile_prefetch_resident_budget()
        if self._reduce_scatter_requests:
            raise RuntimeError("Cannot compile communication scheduling with live RS requests.")
        self._compile_reduce_scatter_budgets()

    def handle_replay_divergence(self) -> None:
        """Return to eager communication while recording a replacement trace."""
        self._completed_prefetch_anchors.clear()
        # Retained parameters remain valid within this global batch. Releasing
        # them here could invalidate the consumer that exposed the divergence.
        self.flush_prefetches(reason="replay divergence")
        for request in tuple(self._reduce_scatter_requests):
            if (
                request.state is _ReduceScatterState.READY
                and self._oldest_domain_request(request.domain) is request
            ):
                self._submit_reduce_scatter(request, reason="replay divergence")
                continue
            # The producer still owns this buffer, so it cannot be submitted.
            # A ready request may also sit behind a still-writing domain head.
            # Remove either case from the deferral budget; mark-ready (or the
            # head's later mark-ready) drains the domain in FIFO order.
            if request.deferred:
                request.domain.pending_bytes -= request.size_bytes
                request.deferred = False
        self._reset_reduce_scatter_trace()

    def flush_prefetches(self, *, reason: str) -> None:
        """Submit every queued successor gather in FIFO order."""
        if not self._pending_prefetches:
            return
        logger.warning(
            "MFSDP communication scheduler submitted %d delayed all-gathers on %s.",
            len(self._pending_prefetches),
            reason,
        )
        while self._pending_prefetches:
            pending = self._pending_prefetches.popleft()
            pending.target._unshard_parameter_groups(
                pending.target_orientation,
                reason=f"flush-{reason.replace(' ', '-')}",
                source=pending.source.name if pending.source.name else "<root>",
                source_phase=pending.source_phase,
                request=pending.sequence,
            )

    def _release_retained_prefetches(self, *, reason: str) -> None:
        """Release retained parameters that did not reach their traced demand."""
        if not self._retained_prefetches and not self._resident_prefetches:
            return
        retained = tuple(self._retained_prefetches) + tuple(self._resident_prefetches)
        self._retained_prefetches.clear()
        self._resident_prefetches.clear()
        self._resident_prefetch_bytes = 0
        logger.warning(
            "MFSDP communication scheduler released %d retained prefetches on %s.",
            len(retained),
            reason,
        )
        seen_targets: set[int] = set()
        for pending in retained:
            target_key = id(pending.target)
            if target_key in seen_targets:
                continue
            seen_targets.add(target_key)
            pending.target._reshard_parameter_groups(record_execution=False)

    def _compile_prefetch_resident_budget(self) -> None:
        """Freeze the optional replay-wide future-materialization byte budget."""
        configured = self.config.max_prefetch_resident_bytes
        if configured is None:
            self._effective_prefetch_resident_bytes = None
            return
        inferred = self._context.runner.max_prefetch_target_nbytes()
        self._effective_prefetch_resident_bytes = inferred if configured == 0 else configured
        logger.info(
            "MFSDP AG residency budget: configured=%s inferred_single_target=%d " "effective=%d",
            configured,
            inferred,
            self._effective_prefetch_resident_bytes,
        )

    def _prefetch_residency_is_limited(self) -> bool:
        """Return whether future materializations use byte-budget admission."""
        return self.config.max_prefetch_resident_bytes is not None

    @staticmethod
    def _prefetch_deadline_key(pending: _PendingPrefetch) -> tuple[float, int]:
        deadline = (
            float(pending.target_unshard_index)
            if pending.target_unshard_index is not None
            else math.inf
        )
        return deadline, pending.sequence

    def _eligible_prefetches(self) -> list[_PendingPrefetch]:
        """Return requests eligible outside a configured conflict-free point."""
        return [
            pending
            for pending in self._pending_prefetches
            if not pending.conflict_free_required
            and (not pending.completion_required or pending.completed_anchor is not None)
        ]

    def _resident_capacity_allows(self, pending: _PendingPrefetch) -> bool:
        budget = self._effective_prefetch_resident_bytes
        return budget is not None and self._resident_prefetch_bytes + pending.size_bytes <= budget

    def _mark_residency_deferred(self, pending: _PendingPrefetch, *, reason: str) -> None:
        if pending.residency_deferred:
            return
        pending.residency_deferred = True
        self._residency_deferrals += 1
        target_name = pending.target.name if pending.target.name else "<root>"
        torch.cuda.nvtx.range_push(
            f"MFSDP AG residency deferred target={target_name} "
            f"orientation={pending.target_orientation} request={pending.sequence} "
            f"bytes={pending.size_bytes} resident_bytes={self._resident_prefetch_bytes} "
            f"budget={self._effective_prefetch_resident_bytes} reason={reason}"
        )
        torch.cuda.nvtx.range_pop()

    def _retain_prefetch_with_residency_budget(
        self, target: "FsdpModule", matches: list[_PendingPrefetch]
    ) -> bool:
        """Reserve the current materialization only for the earliest deadline."""
        eligible = self._eligible_prefetches()
        if not eligible:
            return False
        winner = min(eligible, key=self._prefetch_deadline_key)
        if winner not in matches:
            for pending in matches:
                self._mark_residency_deferred(pending, reason="earlier-deadline")
            return False
        if not self._resident_capacity_allows(winner):
            self._mark_residency_deferred(winner, reason="capacity")
            return False

        self._pending_prefetches.remove(winner)
        winner.target_reshard_index = None
        winner.retained_through_reshard = True
        self._resident_prefetches.append(winner)
        self._resident_prefetch_bytes += winner.size_bytes
        target_name = target.name if target.name else "<root>"
        completed = winner.completed_anchor
        anchor = (
            _completion_name(winner.source, completed.anchor) if completed is not None else None
        )
        provenance = "".join(
            f" {key}={value}"
            for key, value in (
                ("source", winner.source.name if winner.source.name else "<root>"),
                ("source_phase", winner.source_phase),
                ("anchor", anchor),
                ("request", winner.sequence),
                ("bytes", winner.size_bytes),
                ("resident_bytes", self._resident_prefetch_bytes),
                ("budget", self._effective_prefetch_resident_bytes),
            )
            if value is not None
        )
        torch.cuda.nvtx.range_push(
            f"MFSDP AG retained target={target_name} "
            f"orientation={winner.target_orientation} "
            f"trigger=retain-through-reshard{provenance}"
        )
        torch.cuda.nvtx.range_pop()
        if completed is not None:
            self._anchor_releases += 1
        return True

    def _drain_ready_prefetches(self, *, reason: str) -> None:
        """Admit ready prefetches in earliest-demand order within the byte budget."""
        if not self._prefetch_residency_is_limited():
            return
        while True:
            eligible = self._eligible_prefetches()
            if not eligible:
                return
            winner = min(eligible, key=self._prefetch_deadline_key)
            if winner.target_reshard_index is not None:
                for pending in eligible:
                    if pending is not winner and self._prefetch_is_ready(pending):
                        self._mark_residency_deferred(
                            pending, reason="earlier-deadline-lifetime-gate"
                        )
                return
            if not self._resident_capacity_allows(winner):
                self._mark_residency_deferred(winner, reason="capacity")
                return
            self._pending_prefetches.remove(winner)
            self._submit_prefetch(winner, reason=f"residency-{reason}")

    def _get_domain(self, mesh: DeviceMesh) -> _DomainState:
        key = id(mesh)
        domain = self._domains.get(key)
        if domain is None:
            domain = _DomainState(mesh=mesh)
            self._domains[key] = domain
            self._domain_order.append(domain)
        return domain

    def _trace_reduce_scatter_reserve(
        self, domain: _DomainState, size_bytes: int
    ) -> _TraceReduceScatterRequest | None:
        if self._budget_compiled and not self._context.runner.is_tracing:
            return None
        sequence = self._next_reduce_scatter_sequence
        request = _TraceReduceScatterRequest(sequence, domain, size_bytes)
        self._trace_reduce_scatter_requests.append(request)
        domain.trace_pending_bytes += size_bytes
        domain.required_peak = max(domain.required_peak, domain.trace_pending_bytes)
        domain.trace_request_sizes.append(size_bytes)
        if self._context.device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._context.device)
            domain.min_free_bytes = (
                free_bytes
                if domain.min_free_bytes is None
                else min(domain.min_free_bytes, free_bytes)
            )
            domain.total_device_bytes = total_bytes
        return request

    def _trace_reduce_scatter_release(self) -> dict[int, int]:
        """Drain every ready virtual domain head and return per-domain byte credits."""
        if self._budget_compiled and not self._context.runner.is_tracing:
            return {}
        release_credits: dict[int, int] = {}
        while True:
            domain_heads: set[int] = set()
            released = False
            for request in self._trace_reduce_scatter_requests:
                if not request.active:
                    continue
                domain_key = id(request.domain)
                if domain_key in domain_heads:
                    continue
                domain_heads.add(domain_key)
                if not request.ready:
                    continue
                request.active = False
                request.domain.trace_pending_bytes -= request.size_bytes
                release_credits[domain_key] = (
                    release_credits.get(domain_key, 0) + request.size_bytes
                )
                released = True
                break
            if not released:
                return release_credits

    def _trace_reduce_scatter_flush(self) -> None:
        if self._budget_compiled and not self._context.runner.is_tracing:
            return
        for request in self._trace_reduce_scatter_requests:
            if not request.active:
                continue
            request.active = False
            request.domain.trace_pending_bytes -= request.size_bytes

    def _compile_reduce_scatter_budgets(self) -> None:
        configured_limit = self.config.max_pending_reduce_scatter_bytes
        total_required = sum(domain.required_peak for domain in self._domain_order)
        if self._release_anchor_count == 0 or total_required == 0 or configured_limit == 0:
            for domain in self._domain_order:
                domain.effective_budget = 0
            self._budget_compiled = True
            self._trace_reduce_scatter_requests.clear()
            return

        if configured_limit is None:
            available_values = []
            for domain in self._domain_order:
                if domain.min_free_bytes is None or domain.total_device_bytes is None:
                    continue
                safety_reserve = max(1 << 30, domain.total_device_bytes // 20)
                available_values.append(max(0, domain.min_free_bytes - safety_reserve))
            context_limit = min(available_values, default=0)
        else:
            context_limit = configured_limit

        for domain in self._domain_order:
            proportional_limit = (
                context_limit * domain.required_peak // total_required if total_required else 0
            )
            candidate = min(domain.required_peak, proportional_limit)
            alignment = math.gcd(*domain.trace_request_sizes) if domain.trace_request_sizes else 1
            candidate = candidate // alignment * alignment
            domain.effective_budget = self._synchronize_domain_budget(domain.mesh, candidate)
            logger.info(
                "MFSDP RS scheduler domain: required_peak=%d effective_budget=%d "
                "context_limit=%d proportional_limit=%d min_free=%s "
                "total_device_bytes=%s configured_limit=%s",
                domain.required_peak,
                domain.effective_budget,
                context_limit,
                proportional_limit,
                domain.min_free_bytes,
                domain.total_device_bytes,
                configured_limit,
            )

        self._budget_compiled = True
        self._trace_reduce_scatter_requests.clear()

    def _synchronize_domain_budget(self, mesh: DeviceMesh, candidate: int) -> int:
        if not dist.is_initialized() or mesh.size() == 1:
            return candidate
        budget = torch.tensor(candidate, dtype=torch.int64, device=self._context.device)
        for process_group in mesh.get_all_groups():
            dist.all_reduce(budget, op=dist.ReduceOp.MIN, group=process_group)
        return int(budget.item())

    def _oldest_domain_request(self, domain: _DomainState) -> _ReduceScatterRequest | None:
        return next(
            (request for request in self._reduce_scatter_requests if request.domain is domain), None
        )

    def _oldest_releasable_request(
        self, release_credits: dict[int, int]
    ) -> _ReduceScatterRequest | None:
        domain_heads: set[int] = set()
        for request in self._reduce_scatter_requests:
            domain_key = id(request.domain)
            if domain_key in domain_heads:
                continue
            domain_heads.add(domain_key)
            if (
                request.deferred
                and request.state is _ReduceScatterState.READY
                and request.size_bytes <= release_credits.get(domain_key, 0)
            ):
                return request
        return None

    def _oldest_ready_deferred_request(self) -> _ReduceScatterRequest | None:
        """Return the oldest ready deferred head from any collective domain."""
        domain_heads: set[int] = set()
        for request in self._reduce_scatter_requests:
            domain_key = id(request.domain)
            if domain_key in domain_heads:
                continue
            domain_heads.add(domain_key)
            if request.deferred and request.state is _ReduceScatterState.READY:
                return request
        return None

    def _release_reduce_scatter_at_prefetch(self, prefetch: _PendingPrefetch) -> None:
        """Use one actual AG submission as one asynchronous RS service opportunity."""
        if not self.config.reduce_scatter_release_on_prefetch or not self._budget_compiled:
            return
        self._retire_completed_reduce_scatters()
        domain_heads: set[int] = set()
        for request in self._reduce_scatter_requests:
            domain_key = id(request.domain)
            if domain_key in domain_heads:
                continue
            domain_heads.add(domain_key)
            if not request.deferred or request.state is not _ReduceScatterState.READY:
                continue
            self._submit_reduce_scatter(
                request,
                reason="prefetch",
                prefetch=prefetch,
            )
            self._prefetch_reduce_scatter_releases += 1
            return

    def _drain_submit_on_ready_requests(self, domain: _DomainState) -> None:
        while True:
            domain_requests = [
                request for request in self._reduce_scatter_requests if request.domain is domain
            ]
            if not domain_requests or not any(not request.deferred for request in domain_requests):
                return
            head = domain_requests[0]
            if head.state is not _ReduceScatterState.READY:
                return
            self._submit_reduce_scatter(head, reason="submit-on-ready")

    def _submit_reduce_scatter(
        self,
        request: _ReduceScatterRequest,
        *,
        reason: str,
        demand_event: torch.cuda.Event | None = None,
        prefetch: _PendingPrefetch | None = None,
    ) -> None:
        self._retire_completed_reduce_scatters()
        if request.state is not _ReduceScatterState.READY:
            raise RuntimeError("Cannot submit a reduce-scatter request before it is ready.")
        if self._oldest_domain_request(request.domain) is not request:
            raise RuntimeError("Reduce-scatter requests cannot overtake their collective domain.")
        partial_grad = request.partial_grad
        ready_event = request.ready_event
        assert partial_grad is not None and ready_event is not None

        reduce_scatter_stream = self._context.reduce_scatter_stream
        if demand_event is not None:
            reduce_scatter_stream.wait_event(demand_event)
        reduce_scatter_stream.wait_event(ready_event)
        pending_before = self.pending_reduce_scatter_bytes
        in_flight_before = self.in_flight_reduce_scatter_bytes
        pending_after = pending_before - (request.size_bytes if request.deferred else 0)
        in_flight_after = in_flight_before + request.size_bytes
        prefetch_provenance = ""
        if prefetch is not None:
            prefetch_source = prefetch.source.name if prefetch.source.name else "<root>"
            prefetch_target = prefetch.target.name if prefetch.target.name else "<root>"
            prefetch_provenance = (
                f" prefetch_source={prefetch_source} prefetch_target={prefetch_target} "
                f"prefetch_request={prefetch.sequence}"
            )
        label = (
            f"MFSDP RS target={request.module_name} group={request.group_index} "
            f"trigger={reason} request={request.sequence} bytes={request.size_bytes} "
            f"pending_before={pending_before} pending_after={pending_after} "
            f"in_flight_before={in_flight_before} in_flight_after={in_flight_after} "
            f"active_after={pending_after + in_flight_after}{prefetch_provenance}"
        )
        torch.cuda.nvtx.range_push(label)
        try:
            with torch.cuda.stream(reduce_scatter_stream):
                request.group.reduce_partial_gradients(partial_grad, request.is_last_microbatch)
                request.group.release_partial_grad_buffer()
                completion_event = reduce_scatter_stream.record_event()
        finally:
            torch.cuda.nvtx.range_pop()

        if request.deferred:
            request.domain.pending_bytes -= request.size_bytes
        request.domain.in_flight_bytes += request.size_bytes
        self._in_flight_reduce_scatters.append(
            _InFlightReduceScatter(
                sequence=request.sequence,
                domain=request.domain,
                module_name=request.module_name,
                group_index=request.group_index,
                size_bytes=request.size_bytes,
                completion_event=completion_event,
            )
        )
        self._active_request_by_group.pop(request.group, None)
        self._reduce_scatter_requests.remove(request)
        self._update_reduce_scatter_peaks()

    def _retire_completed_reduce_scatters(self) -> None:
        """Retire completed RS inputs in their single-stream submission order."""
        while self._in_flight_reduce_scatters:
            request = self._in_flight_reduce_scatters[0]
            if not request.completion_event.query():
                return
            self._in_flight_reduce_scatters.popleft()
            request.domain.in_flight_bytes -= request.size_bytes
            torch.cuda.nvtx.range_push(
                f"MFSDP RS retire target={request.module_name} group={request.group_index} "
                f"request={request.sequence} bytes={request.size_bytes} "
                f"pending_bytes={self.pending_reduce_scatter_bytes} "
                f"in_flight_bytes={self.in_flight_reduce_scatter_bytes}"
            )
            torch.cuda.nvtx.range_pop()

    def _update_reduce_scatter_peaks(self) -> None:
        """Update scheduler-lifetime logical RS footprint high-water marks."""
        pending_bytes = self.pending_reduce_scatter_bytes
        in_flight_bytes = self.in_flight_reduce_scatter_bytes
        self._peak_pending_reduce_scatter_bytes = max(
            self._peak_pending_reduce_scatter_bytes, pending_bytes
        )
        self._peak_in_flight_reduce_scatter_bytes = max(
            self._peak_in_flight_reduce_scatter_bytes, in_flight_bytes
        )
        self._peak_active_reduce_scatter_bytes = max(
            self._peak_active_reduce_scatter_bytes, pending_bytes + in_flight_bytes
        )

    def _emit_reduce_scatter_state(
        self, phase: str, request: _ReduceScatterRequest, **metadata: object
    ) -> None:
        """Emit a zero-duration NVTX marker with scheduler byte accounting."""
        suffix = "".join(f" {key}={value}" for key, value in metadata.items())
        torch.cuda.nvtx.range_push(
            f"MFSDP RS state={phase} target={request.module_name} "
            f"group={request.group_index} request={request.sequence} "
            f"bytes={request.size_bytes} pending_bytes={self.pending_reduce_scatter_bytes} "
            f"domain_pending_bytes={request.domain.pending_bytes} "
            f"budget={request.domain.effective_budget} "
            f"in_flight_bytes={self.in_flight_reduce_scatter_bytes} "
            f"domain_in_flight_bytes={request.domain.in_flight_bytes}{suffix}"
        )
        torch.cuda.nvtx.range_pop()

    def _reset_reduce_scatter_trace(self) -> None:
        self._trace_reduce_scatter_requests.clear()
        self._trace_reduce_scatter_release_credits.clear()
        self._budget_compiled = False
        for domain in self._domain_order:
            domain.pending_bytes = 0
            domain.effective_budget = 0
            domain.required_peak = 0
            domain.trace_pending_bytes = 0
            domain.trace_request_sizes.clear()
            domain.min_free_bytes = None
            domain.total_device_bytes = None

    def report(self) -> None:
        """Log delayed all-gather scheduling statistics."""
        logger.info(
            "MFSDP communication scheduler: prefetch_depth=%d delayed_prefetches=%d "
            "anchor_releases=%d latched_anchor_releases=%d demand_releases=%d "
            "retained_prefetch_reuses=%d residency_deferrals=%d "
            "residency_releases=%d resident_prefetch_bytes=%d "
            "effective_prefetch_resident_bytes=%s pending_prefetches=%d "
            "rs_anchor_releases=%d rs_prefetch_releases=%d "
            "capacity_releases=%d final_releases=%d "
            "pending_rs_bytes=%d in_flight_rs_bytes=%d "
            "peak_pending_rs_bytes=%d peak_in_flight_rs_bytes=%d "
            "peak_active_rs_bytes=%d",
            self.config.prefetch_depth,
            self._delayed_prefetches,
            self._anchor_releases,
            self._latched_anchor_releases,
            self._demand_releases,
            self._retained_prefetch_reuses,
            self._residency_deferrals,
            self._residency_releases,
            self._resident_prefetch_bytes,
            self._effective_prefetch_resident_bytes,
            len(self._pending_prefetches)
            + len(self._retained_prefetches)
            + len(self._resident_prefetches),
            self._reduce_scatter_releases,
            self._prefetch_reduce_scatter_releases,
            self._capacity_releases,
            self._final_releases,
            self.pending_reduce_scatter_bytes,
            self.in_flight_reduce_scatter_bytes,
            self.peak_pending_reduce_scatter_bytes,
            self.peak_in_flight_reduce_scatter_bytes,
            self.peak_active_reduce_scatter_bytes,
        )

    def _find_matching_prefetch(self, completion_index: int) -> _PendingPrefetch | None:
        """Find the request assigned to one exact replay completion index."""
        for pending in self._pending_prefetches:
            if completion_index not in pending.completion_indices:
                continue
            return pending
        return None

    @staticmethod
    def _prefetch_is_ready(pending: _PendingPrefetch) -> bool:
        """Return whether completion and target-lifetime gates are satisfied."""
        completion_ready = not pending.completion_required or pending.completed_anchor is not None
        return completion_ready and pending.target_reshard_index is None

    def _submit_prefetch(self, pending: _PendingPrefetch, *, reason: str) -> None:
        """Submit one prefetch after all of its scheduling gates are satisfied."""
        completed = pending.completed_anchor
        if completed is not None:
            self._context.allgather_stream.wait_event(completed.event)
        target_was_unsharded = pending.target._unshard_event is not None
        pending.target._unshard_parameter_groups(
            pending.target_orientation,
            reason=reason,
            source=pending.source.name if pending.source.name else "<root>",
            source_phase=pending.source_phase,
            anchor=(
                _completion_name(pending.source, completed.anchor)
                if completed is not None
                else None
            ),
            request=pending.sequence,
        )
        prefetch_event = pending.target._unshard_event
        if not target_was_unsharded and prefetch_event is not None:
            self._release_reduce_scatter_at_prefetch(pending)
        if self._prefetch_residency_is_limited():
            self._resident_prefetches.append(pending)
            self._resident_prefetch_bytes += pending.size_bytes
            target_name = pending.target.name if pending.target.name else "<root>"
            torch.cuda.nvtx.range_push(
                f"MFSDP AG residency admitted target={target_name} "
                f"orientation={pending.target_orientation} request={pending.sequence} "
                f"bytes={pending.size_bytes} resident_bytes={self._resident_prefetch_bytes} "
                f"budget={self._effective_prefetch_resident_bytes}"
            )
            torch.cuda.nvtx.range_pop()
        if completed is not None:
            self._anchor_releases += 1

    def _pop_latched_completion(self, pending: _PendingPrefetch) -> _CompletedPrefetchAnchor | None:
        """Return the earliest already-satisfied anchor for ``pending``."""
        for completion_index in pending.completion_indices:
            completed = self._completed_prefetch_anchors.pop(completion_index, None)
            if completed is None:
                continue
            if completed.owner is not pending.source or completed.phase != pending.source_phase:
                raise RuntimeError(
                    "A compiled completion occurrence did not match its prefetch source."
                )
            return completed
        return None


def _completion_name(owner: "FsdpModule", anchor: nn.Module | str) -> str:
    """Return a compact, stable name for one actual completion anchor."""
    if isinstance(anchor, str):
        return f"@{anchor}"
    for name, module in cast(nn.Module, owner).named_modules():
        if module is anchor:
            return name or "<self>"
    return f"<{type(anchor).__name__}>"
