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

"""Bounded local gradient accumulation before the original inner reduction."""

import hashlib
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple
from weakref import WeakKeyDictionary, ref

import torch
import torch.distributed as dist


class GradientEvent(NamedTuple):
    """One actual group contribution, independent of the number of collectives."""

    group: ref
    is_last: bool


class GroupMetadata(NamedTuple):
    owner: ref
    group_index: int
    index: int
    reason: str
    accumulator_dtype: object
    communication_dtype: object


@dataclass
class _Lease:
    group: ref
    charge: int
    buffer: object
    allocation_stream: object
    count: int = 0
    ready: object = None
    communication_buffer: object = None
    done: object = None


class GradientWindow:
    """A shared byte envelope including held accumulators and retiring RS inputs.

    The first complete eager step uses the original reduction path and records
    group contributions. Peers agree on that trace and the policy before any
    window changes collective counts. Later FIFO decisions depend only on this
    trace, the window size and static capacities, never on GPU event queries.
    """

    def __init__(self, context, modules, eligible_modules, window_size, budget_bytes):
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("Gradient window size must be a positive integer.")
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("Gradient window budget must be a non-negative integer.")
        self.context = ref(context)
        self.window_size, self.budget_bytes = window_size, budget_bytes
        modules, eligible = tuple(modules), set(eligible_modules)
        if not eligible.issubset(modules):
            raise ValueError("Eligible gradient owners must belong to this FSDP context.")
        self.groups = WeakKeyDictionary()
        self.capacities = WeakKeyDictionary()
        for module in modules:
            for group_index, group in enumerate(module.parameter_groups):
                if not group.requires_grad:
                    continue
                reason = (
                    "ineligible_owner"
                    if module not in eligible
                    else (
                        "no_inner_network_reduction"
                        if not group.has_inner_network_reduction
                        else "eligible"
                    )
                )
                self.groups[group] = GroupMetadata(
                    ref(module),
                    group_index,
                    len(self.groups),
                    reason,
                    group.main_grad.dtype,
                    group.grad_comm_dtype,
                )
                if reason == "eligible":
                    self.capacities[group] = group.gradient_window_storage_bytes
        if window_size > 1 and max(self.capacities.values(), default=0) > budget_bytes:
            raise ValueError("Gradient window budget must fit one complete eligible group lease.")
        self.events = []
        self.recording, self.position, self.events_seen = True, 0, 0
        self.pending = OrderedDict()
        self.retiring = deque()
        self.pending_outer = WeakKeyDictionary()
        self.outer_finalized = WeakKeyDictionary()
        self.seen_groups = WeakKeyDictionary()
        self.reserved_bytes = self.peak_reserved_bytes = 0
        self.statistics = Counter()
        self.last_step_statistics = {}
        self.signature, self.frozen, self._frozen_state = None, False, None
        self._compiled_state = None

    def reason(self, group):
        if self.recording:
            return "recording"
        if self.window_size == 1:
            return "window_one"
        return self.groups[group].reason

    def effective_size(self, group):
        return self.window_size if self.reason(group) == "eligible" else 1

    def contribute(self, module, group, group_index):
        """Return whether this contribution was consumed by the bounded window."""
        self._validate_compiled_policy()
        if self.frozen or torch.cuda.is_current_stream_capturing():
            self.validate_capture()
        context = self.context()
        metadata = self.groups[group]
        if (
            group.main_grad.dtype != metadata.accumulator_dtype
            or group.grad_comm_dtype != metadata.communication_dtype
        ):
            raise RuntimeError(
                "Gradient accumulation/communication dtype changed after configuration."
            )
        if metadata.owner() is not module or metadata.group_index != group_index:
            raise RuntimeError("Gradient contribution refers to the wrong owner/group.")
        if group in self.outer_finalized:
            raise RuntimeError("Gradient contribution arrived after its last-microbatch finalize.")
        event = GradientEvent(ref(group), bool(context.is_last_microbatch))
        if self.recording:
            self.events.append(event)
        else:
            if self.position >= len(self.events) or self.events[self.position] != event:
                raise RuntimeError(f"FSDP gradient lifecycle diverged at {self.position}.")
            self.position += 1
        self.events_seen += 1
        self.statistics["contributions"] += 1
        self.seen_groups[group] = True
        self.pending_outer[group] = True
        if self.effective_size(group) == 1:
            return False
        stream = context.current_stream()
        if group not in self.pending:
            self._admit(group)
            self.pending[group] = _Lease(
                ref(group),
                self.capacities[group],
                group.allocate_partial_grad_buffer(dtype=group.main_grad.dtype),
                stream,
            )
        lease = self.pending[group]
        if lease.ready is not None:
            stream.wait_event(lease.ready)
        group.copy_gradients_to_partial_buffer(lease.buffer, accumulate=bool(lease.count))
        lease.count += 1
        lease.ready = stream.record_event()
        if event.is_last or lease.count == self.window_size:
            context._flush_inner_gradient_window(
                group, reason="last_microbatch" if event.is_last else "window_full"
            )
        if event.is_last:
            context._finalize_gradient_outer(group)
        return True

    def _admit(self, group):
        required = self.capacities[group]
        while self.reserved_bytes + required > self.budget_bytes:
            if self.retiring:
                self._retire_one()
            elif self.pending:
                victim = next(iter(self.pending))
                self.context()._flush_inner_gradient_window(victim, reason="budget_eviction")
                self.statistics["budget_evictions"] += 1
            else:
                raise RuntimeError("Gradient window cannot admit one eligible group.")
        self.reserved_bytes += required
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)

    def flush(self, group, *, reason):
        """Submit one actual inner reduction for the sum of this lease's contributions."""
        lease = self.pending.pop(group)
        if not lease.count or lease.ready is None:
            raise RuntimeError("Cannot flush an empty gradient accumulation lease.")
        context = self.context()
        inner = context.inner_reduction_stream
        inner.wait_event(lease.ready)
        with torch.cuda.stream(inner):
            # Preserve the original communication dtype. The cast's entire input
            # envelope was reserved on admission, even before it is allocated.
            lease.communication_buffer = (
                lease.buffer
                if lease.buffer.dtype == group.grad_comm_dtype
                else lease.buffer.cast(group.grad_comm_dtype)
            )
            group.reduce_inner_gradients(lease.communication_buffer)
            lease.done = inner.record_event()
        self.retiring.append(lease)
        self.statistics["inner_reductions"] += 1
        self.statistics["merged_contributions"] += lease.count
        self.statistics["flush_" + reason] += 1

    def legacy_flush(self, *, reason):
        self.statistics["inner_reductions"] += 1
        self.statistics["flush_" + reason] += 1

    def outer_finished(self, group):
        if group not in self.pending_outer or group in self.pending:
            raise RuntimeError("Outer gradient finalize has no complete inner contribution.")
        del self.pending_outer[group]
        self.outer_finalized[group] = True
        self.statistics["outer_finalizations"] += 1

    def _retire_one(self):
        lease = self.retiring.popleft()
        if lease.done is None:
            raise RuntimeError("Gradient input retired before its reduction was submitted.")
        # Match the existing allocation-stream retirement contract. No query or
        # host synchronize enters capture; reuse is ordered after inner reduction.
        lease.allocation_stream.wait_event(lease.done)
        lease.buffer = lease.communication_buffer = None
        self.reserved_bytes -= lease.charge

    def _signature(self):
        values = (
            self.window_size,
            self.budget_bytes,
            [
                (
                    meta.index,
                    meta.group_index,
                    meta.reason,
                    self.capacities.get(group, 0),
                    str(group.grad_comm_dtype),
                    str(group.main_grad.dtype),
                )
                for group, meta in self.groups.items()
            ],
            [(self.groups[event.group()].index, event.is_last) for event in self.events],
        )
        return hashlib.sha256(repr(values).encode()).hexdigest()

    def finish_step(self):
        """Flush every inner tail before remaining outer reductions and optimizer use."""
        self._validate_compiled_policy()
        if not self.events_seen:
            if self.pending or self.retiring or self.pending_outer or self.reserved_bytes:
                raise RuntimeError("Empty gradient step still owns pending reductions.")
            return
        context = self.context()
        for group in tuple(self.pending):
            context._flush_inner_gradient_window(group, reason="optimizer_tail")
        # Do not finalize one group's outer stage while another group's inner
        # optimizer tail has not even been submitted.
        for group in self.groups:
            if group in self.pending_outer:
                context._finalize_gradient_outer(group)
        while self.retiring:
            self._retire_one()
        if self.recording:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Gradient windows require one complete eager recording step.")
            context.current_stream().wait_stream(context.inner_reduction_stream)
            if context.outer_reduction_stream is not context.inner_reduction_stream:
                context.current_stream().wait_stream(context.outer_reduction_stream)
            digest = int(self._signature()[:15], 16)
            header = torch.tensor(
                [len(self.events), -len(self.events), digest, -digest],
                dtype=torch.int64,
                device=context.allgather_stream.device,
            )
            dist.all_reduce(header, op=dist.ReduceOp.MIN, group=context._prefetch_process_group)
            values = header.tolist()
            if values[0] != -values[1] or values[2] != -values[3]:
                raise RuntimeError(
                    "FSDP gradient trace, eligibility, dtype or window policy differs across ranks."
                )
            self.recording = False
            self.signature = self._signature()
            self.events = tuple(self.events)
            self.groups = MappingProxyType(self.groups)
            self.capacities = MappingProxyType(self.capacities)
            self._compiled_state = self._capture_state()
        elif self.position != len(self.events) or self.events_seen != len(self.events):
            raise RuntimeError(
                "FSDP gradient lifecycle ended before its recorded step was complete."
            )
        self.last_step_statistics = dict(self.statistics)
        self.statistics.clear()
        self.position = self.events_seen = 0
        self.outer_finalized.clear()
        if self.pending_outer or self.reserved_bytes:
            raise RuntimeError("Gradient windows remain live at the optimizer boundary.")

    def prepare_capture(self):
        if self.frozen:
            self.validate_capture()
            return
        if self.recording and not self.groups and not self.events:
            # A stage without trainable groups has no contribution collective
            # to record; still pin its explicitly configured empty policy.
            self.recording = False
            self.signature = self._signature()
            self.events = tuple(self.events)
            self.groups = MappingProxyType(self.groups)
            self.capacities = MappingProxyType(self.capacities)
            self._compiled_state = self._capture_state()
        if (
            self.recording
            or self.position
            or self.events_seen
            or self.pending
            or self.retiring
            or self.pending_outer
        ):
            raise RuntimeError("Gradient capture requires a compiled clean optimizer boundary.")
        self._validate_compiled_policy()
        if self._signature() != self.signature:
            raise RuntimeError(
                "Compiled gradient metadata or communication dtype changed before capture."
            )
        self._frozen_state = self._capture_state()
        self.frozen = True

    def _capture_state(self):
        return (
            self.window_size,
            self.budget_bytes,
            id(self.events),
            len(self.events),
            id(self.groups),
            len(self.groups),
            id(self.capacities),
            len(self.capacities),
            self.signature,
            self.recording,
        )

    def _validate_compiled_policy(self):
        if self._compiled_state is not None and self._compiled_state != self._capture_state():
            raise RuntimeError("Compiled gradient-window policy or trace changed.")

    def validate_capture(self):
        if not self.frozen or self._frozen_state != self._capture_state():
            raise RuntimeError("The captured FSDP gradient-window policy or trace changed.")
        if any(
            group.main_grad.dtype != metadata.accumulator_dtype
            or group.grad_comm_dtype != metadata.communication_dtype
            for group, metadata in self.groups.items()
        ):
            raise RuntimeError("Captured gradient accumulation/communication dtype changed.")
