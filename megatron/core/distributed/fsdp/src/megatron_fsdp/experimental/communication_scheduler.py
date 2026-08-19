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
from typing import TYPE_CHECKING, Literal

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
            pre-backward entry may release one pending reduce-scatter request.
    """

    prefetch_successor_after: tuple[ModuleCompletion | NamedCompletion, ...] = ()
    reduce_scatter_release_on_pre_backward: tuple[nn.Module | NamedPreBackward, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether this policy preserves eager communication."""
        return not (self.prefetch_successor_after or self.reduce_scatter_release_on_pre_backward)


@dataclasses.dataclass(frozen=True)
class FsdpCommunicationSchedulerConfig:
    """Context-wide limits for trace-guided communication scheduling.

    Args:
        max_pending_reduce_scatter_bytes: ``None`` infers a pending-byte limit
            from the trace, zero keeps reduce-scatter eager, and a positive
            value supplies an explicit limit.
    """

    max_pending_reduce_scatter_bytes: int | None = None

    def __post_init__(self) -> None:
        value = self.max_pending_reduce_scatter_bytes
        if value is not None and value < 0:
            raise ValueError(
                "max_pending_reduce_scatter_bytes must be None or non-negative, " f"got {value}."
            )


@dataclasses.dataclass
class _PendingPrefetch:
    """One traced successor gather waiting for a completion anchor."""

    source: "FsdpModule"
    target: "FsdpModule"
    orientation: str
    completions: tuple[ModuleCompletion | NamedCompletion, ...]


class _ReduceScatterState(Enum):
    """Host-visible readiness of a deferred reduce-scatter request."""

    WRITING = auto()
    READY = auto()


@dataclasses.dataclass
class _DomainState:
    """Pending-byte state for one DeviceMesh collective domain."""

    mesh: DeviceMesh
    pending_bytes: int = 0
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
    size_bytes: int
    deferred: bool
    trace_request: _TraceReduceScatterRequest | None
    state: _ReduceScatterState = _ReduceScatterState.WRITING
    partial_grad: "DBuffer | None" = None
    ready_event: torch.cuda.Event | None = None
    is_last_microbatch: bool = True


class FsdpCommunicationScheduler:
    """Context-owned scheduler for delayed parameter and gradient collectives."""

    def __init__(self, context: "FsdpContext", config: FsdpCommunicationSchedulerConfig) -> None:
        """Create an empty scheduler for ``context``."""
        self._context = context
        self.config = config
        self._pending_prefetches: deque[_PendingPrefetch] = deque()
        self._delayed_prefetches = 0
        self._anchor_releases = 0
        self._demand_releases = 0
        self._domains: dict[int, _DomainState] = {}
        self._domain_order: list[_DomainState] = []
        self._reduce_scatter_requests: deque[_ReduceScatterRequest] = deque()
        self._active_request_by_group: dict[FsdpParameterGroup, _ReduceScatterRequest] = {}
        self._trace_reduce_scatter_requests: deque[_TraceReduceScatterRequest] = deque()
        self._next_reduce_scatter_sequence = 0
        self._release_anchor_count = 0
        self._budget_compiled = False
        self._reduce_scatter_releases = 0
        self._capacity_releases = 0
        self._final_releases = 0

    @property
    def has_pending_prefetches(self) -> bool:
        """Return whether a successor gather is waiting for an anchor."""
        return bool(self._pending_prefetches)

    @property
    def pending_reduce_scatter_bytes(self) -> int:
        """Return deferred, unsubmitted reduce-scatter bytes across domains."""
        return sum(domain.pending_bytes for domain in self._domain_order)

    @property
    def effective_reduce_scatter_budgets(self) -> tuple[int, ...]:
        """Return inferred/overridden budgets in domain registration order."""
        return tuple(domain.effective_budget for domain in self._domain_order)

    def register_reduce_scatter_release_anchor(
        self, owner: "FsdpModule", anchor: nn.Module | str
    ) -> None:
        """Register one unique pre-backward release point during construction."""
        del owner, anchor
        self._release_anchor_count += 1

    def schedule_prefetch(
        self, source: "FsdpModule", target: "FsdpModule", orientation: str
    ) -> None:
        """Submit or defer a traced successor all-gather."""
        source_phase: CommunicationPhase = "forward" if orientation == "rowwise" else "backward"
        completions = tuple(
            completion
            for completion in source.communication_policy.prefetch_successor_after
            if completion.phase == source_phase
        )
        if not completions or self._context.runner.is_tracing:
            target._unshard_parameter_groups(orientation)
            return

        self._pending_prefetches.append(
            _PendingPrefetch(
                source=source, target=target, orientation=orientation, completions=completions
            )
        )
        self._delayed_prefetches += 1
        torch.cuda.nvtx.range_push("MFSDP delayed AG queued")
        torch.cuda.nvtx.range_pop()

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
        runner.record_completion(owner, anchor, phase)
        if runner.is_tracing:
            # A replay divergence may leave speculative requests from the
            # aborted cycle. Submit them before tracing a replacement cycle.
            self.flush_prefetches(reason="trace")
            return

        pending = self._pop_matching_prefetch(owner, anchor, phase)
        if pending is None:
            return
        if event is None:
            event = self._context.current_stream().record_event()
        self._context.allgather_stream.wait_event(event)
        pending.target._unshard_parameter_groups(pending.orientation)
        self._anchor_releases += 1
        torch.cuda.nvtx.range_push("MFSDP delayed AG anchor release")
        torch.cuda.nvtx.range_pop()

    def demand_unshard(self, target: "FsdpModule", orientation: str) -> None:
        """Release the oldest queued gather for the demanded module occurrence."""
        for index, pending in enumerate(self._pending_prefetches):
            if pending.target is not target or pending.orientation != orientation:
                continue
            del self._pending_prefetches[index]
            pending.target._unshard_parameter_groups(pending.orientation)
            self._demand_releases += 1
            torch.cuda.nvtx.range_push("MFSDP delayed AG demand release")
            torch.cuda.nvtx.range_pop()
            return

    def record_reduce_scatter_release(
        self, owner: "FsdpModule", anchor: nn.Module | str, demand_event: torch.cuda.Event | None
    ) -> None:
        """Observe a configured pre-backward reduce-scatter release point."""
        self._context.runner.record_reduce_scatter_release(owner, anchor)
        self._trace_reduce_scatter_release()
        if not self._budget_compiled or self._context.runner.is_tracing:
            return

        request = self._oldest_releasable_request()
        if request is None:
            return
        self._submit_reduce_scatter(request, reason="anchor", demand_event=demand_event)
        self._reduce_scatter_releases += 1

    def reserve_reduce_scatter(self, group: "FsdpParameterGroup") -> None:
        """Reserve pending capacity before allocating a full gradient buffer."""
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
            size_bytes=size_bytes,
            deferred=deferred,
            trace_request=trace_request,
        )
        self._next_reduce_scatter_sequence += 1
        self._reduce_scatter_requests.append(request)
        self._active_request_by_group[group] = request
        if deferred:
            domain.pending_bytes += size_bytes

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

        # A submit-on-ready request cannot overtake an older collective in the
        # same domain. Force ready domain heads until every non-deferred request
        # that can legally launch has done so.
        self._drain_submit_on_ready_requests(request.domain)

    def finish_grad_sync(self) -> None:
        """Submit every ready request and fence gradient consumers against RS."""
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
        if not runner_was_tracing:
            return
        if self._reduce_scatter_requests:
            raise RuntimeError("Cannot compile communication scheduling with live RS requests.")
        self._compile_reduce_scatter_budgets()

    def handle_replay_divergence(self) -> None:
        """Return to eager communication while recording a replacement trace."""
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
            pending.target._unshard_parameter_groups(pending.orientation)

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

    def _trace_reduce_scatter_release(self) -> None:
        if self._budget_compiled and not self._context.runner.is_tracing:
            return
        domain_heads: set[int] = set()
        for request in self._trace_reduce_scatter_requests:
            if not request.active:
                continue
            domain_key = id(request.domain)
            if domain_key in domain_heads:
                continue
            domain_heads.add(domain_key)
            if request.ready:
                request.active = False
                request.domain.trace_pending_bytes -= request.size_bytes
                return

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

    def _oldest_releasable_request(self) -> _ReduceScatterRequest | None:
        domain_heads: set[int] = set()
        for request in self._reduce_scatter_requests:
            domain_key = id(request.domain)
            if domain_key in domain_heads:
                continue
            domain_heads.add(domain_key)
            if request.deferred and request.state is _ReduceScatterState.READY:
                return request
        return None

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
    ) -> None:
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
        with torch.cuda.stream(reduce_scatter_stream):
            request.group.reduce_partial_gradients(partial_grad, request.is_last_microbatch)
            request.group.release_partial_grad_buffer()

        if request.deferred:
            request.domain.pending_bytes -= request.size_bytes
        self._active_request_by_group.pop(request.group, None)
        self._reduce_scatter_requests.remove(request)
        torch.cuda.nvtx.range_push(f"MFSDP RS release: {reason}")
        torch.cuda.nvtx.range_pop()

    def _reset_reduce_scatter_trace(self) -> None:
        self._trace_reduce_scatter_requests.clear()
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
            "MFSDP communication scheduler: delayed_prefetches=%d "
            "anchor_releases=%d demand_releases=%d pending_prefetches=%d "
            "rs_anchor_releases=%d capacity_releases=%d final_releases=%d "
            "pending_rs_bytes=%d",
            self._delayed_prefetches,
            self._anchor_releases,
            self._demand_releases,
            len(self._pending_prefetches),
            self._reduce_scatter_releases,
            self._capacity_releases,
            self._final_releases,
            self.pending_reduce_scatter_bytes,
        )

    def _pop_matching_prefetch(
        self, owner: "FsdpModule", anchor: nn.Module | str, phase: CommunicationPhase
    ) -> _PendingPrefetch | None:
        for index, pending in enumerate(self._pending_prefetches):
            if pending.source is not owner:
                continue
            if not any(
                (
                    isinstance(completion, ModuleCompletion)
                    and completion.module is anchor
                    and completion.phase == phase
                )
                or (
                    isinstance(completion, NamedCompletion)
                    and completion.name == anchor
                    and completion.phase == phase
                )
                for completion in pending.completions
            ):
                continue
            del self._pending_prefetches[index]
            return pending
        return None
