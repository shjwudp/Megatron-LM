# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-weight state and synchronization for Megatron FSDP parameter groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.distributed.tensor import DeviceMesh

from .buffer_index import Placement
from .dp_buffer import DataParallelBuffer
from .mixed_precision import MixedPrecisionPolicy, WeightBufferRole
from .param_group_state import (
    ParameterGroupLayout,
    PendingWeightTransition,
    Placements,
    WeightRepresentationState,
)
from .sync_utils import last_changed_axis, resolve_axis_streams


class WeightSyncOwner(Protocol):
    """Parameter-group resources required by :class:`WeightSynchronizer`."""

    params: list[torch.nn.Parameter]
    param_idx: dict[torch.nn.Parameter, int]
    mesh: DeviceMesh
    layout: ParameterGroupLayout
    mp_policy: MixedPrecisionPolicy
    device: torch.device
    weight_buffers: dict[WeightBufferRole, DataParallelBuffer]
    main_weight_buffer: DataParallelBuffer
    transpose_weight_buffer: DataParallelBuffer | None
    _main_weight_aliases_weight: bool

    @property
    def full_placements(self) -> Placements: ...

    def reload_to_gpu(self) -> None: ...

    def _allocate_scratch(
        self, role: str, prototype: DataParallelBuffer, placements: Placements
    ) -> DataParallelBuffer: ...

    def _release_scratch(self, role: str, buffer: DataParallelBuffer | None) -> None: ...


@dataclass
class WeightUnshardPlan:
    """One weight representation participating in a batched unshard."""

    group_index: int
    synchronizer: "WeightSynchronizer"
    role: WeightBufferRole
    source_placements: Placements
    persistent_placements: Placements
    source: DataParallelBuffer
    output: DataParallelBuffer


class WeightSynchronizer:
    """Own model-weight validity, prefetch, unshard, and reshard state."""

    def __init__(self, owner: WeightSyncOwner) -> None:
        self.owner = owner
        self.representations = {
            role: WeightRepresentationState(
                persistent=buffer, valid_placements=tuple(buffer.placements)
            )
            for role, buffer in owner.weight_buffers.items()
        }
        for role, state in self.representations.items():
            if state.valid_placements == owner.full_placements:
                self._bind(state.persistent, role)

    @property
    def model(self) -> WeightRepresentationState:
        """Return the canonical model-weight representation state."""
        return self.representations[WeightBufferRole.MODEL]

    def required_roles(self, bwd_pass: bool = False) -> tuple[WeightBufferRole, ...]:
        """Return compute-weight roles required for this pass in stable order."""
        required = set()
        for param in self.owner.params:
            required.update(
                self.owner.mp_policy.weight_buffer_roles_for_unshard(param, bwd_pass=bwd_pass)
            )
        missing = required.difference(self.representations)
        if missing:
            raise RuntimeError(f"Required weight buffers are unavailable: {missing}")
        return tuple(
            role
            for role in (WeightBufferRole.MODEL, WeightBufferRole.TRANSPOSE)
            if role in required
        )

    def consume_pending(
        self, roles: tuple[WeightBufferRole, ...], stream: torch.cuda.Stream
    ) -> None:
        """Make one stream depend on pending persistent-weight transitions."""
        waited_events: set[int] = set()
        for role in roles:
            state = self.representations[role]
            pending = state.pending
            if pending is None:
                continue
            event_id = id(pending.event)
            if event_id not in waited_events:
                stream.wait_event(pending.event)
                waited_events.add(event_id)
            state.valid_placements = pending.target
            state.pending = None

    def join_pending(self) -> None:
        """Join pending persistent-weight transitions on the caller stream."""
        roles = tuple(
            role for role, state in self.representations.items() if state.pending is not None
        )
        self.consume_pending(roles, torch.cuda.current_stream())

    def _bind(self, buffer: DataParallelBuffer, role: WeightBufferRole) -> None:
        if buffer.data is None:
            raise RuntimeError("Cannot bind parameters from an unbound weight buffer")
        for param in self.owner.params:
            item_id = self.owner.param_idx[param]
            start, end = buffer.buffer_index._get_item_global_range(item_id)
            shape = buffer.buffer_index.item_index_map[item_id].shape
            self.owner.mp_policy.bind_unsharded_param(
                param, buffer.data[start:end].view(shape), role.value
            )

    def get_unsharded_buffer(
        self, role: WeightBufferRole = WeightBufferRole.MODEL
    ) -> DataParallelBuffer | None:
        """Return an available unsharded weight buffer for the requested role."""
        state = self.representations[role]
        buffer = (
            state.persistent if state.valid_placements == self.owner.full_placements else state.full
        )
        if buffer is None or buffer.data is None or buffer.data.device != self.owner.device:
            return None
        return buffer

    def weights_are_unsharded(self, bwd_pass: bool = False) -> bool:
        """Return whether all compute-weight representations for this pass are available."""
        return all(
            self.get_unsharded_buffer(role) is not None
            for role in self.required_roles(bwd_pass=bwd_pass)
        )

    @staticmethod
    @torch.no_grad()
    def prefetch_storage(
        synchronizers: Sequence["WeightSynchronizer"],
        *,
        stream: torch.cuda.Stream,
        bwd_pass: bool = False,
    ) -> torch.cuda.Event | None:
        """Asynchronously refresh pass-specific persistent weight storage."""
        plans = []
        target_placements = None
        for synchronizer in synchronizers:
            synchronizer.owner.reload_to_gpu()
            for role in synchronizer.required_roles(bwd_pass=bwd_pass):
                state = synchronizer.representations[role]
                if state.pending is not None:
                    continue
                if synchronizer.get_unsharded_buffer(role) is not None:
                    continue
                target = tuple(state.persistent.placements)
                if state.valid_placements == target:
                    continue
                if target_placements is None:
                    target_placements = target
                elif target_placements != target:
                    raise ValueError("Prefetched parameter groups must share weight placements")
                plans.append(
                    (
                        synchronizer,
                        role,
                        state.persistent.view(list(state.valid_placements)),
                        state.persistent,
                        target,
                    )
                )

        if not plans:
            return None

        DataParallelBuffer.redistribute_buffers(
            [source for _, _, source, _, _ in plans],
            list(target_placements),
            output_buffers=[output for _, _, _, output, _ in plans],
            stream=stream,
            async_op=True,
        )
        event = stream.record_event()
        for synchronizer, role, _, _, target in plans:
            synchronizer.representations[role].pending = PendingWeightTransition(
                target=target, event=event
            )
        return event

    @staticmethod
    @torch.no_grad()
    def unshard(
        synchronizers: Sequence["WeightSynchronizer"],
        stream: torch.cuda.Stream | None = None,
        *,
        streams: Sequence[torch.cuda.Stream | None] | None = None,
        bwd_pass: bool = False,
        async_op: bool = False,
    ) -> list[DataParallelBuffer]:
        """Unshard pass-specific weight representations in one coalesced axis plan."""
        if not synchronizers:
            return []
        owner = synchronizers[0].owner
        axis_streams = resolve_axis_streams(owner.mesh.ndim, stream=stream, streams=streams)
        outputs_by_group: list[dict[WeightBufferRole, DataParallelBuffer]] = [
            {} for _ in synchronizers
        ]
        required_by_group: list[tuple[WeightBufferRole, ...]] = []
        terminal_axes: list[int | None] = [None] * len(synchronizers)
        plans: list[WeightUnshardPlan] = []

        try:
            for index, synchronizer in enumerate(synchronizers):
                group = synchronizer.owner
                if group.mesh.ndim != len(axis_streams):
                    raise ValueError("All parameter groups must use the same mesh dimensionality")
                group.reload_to_gpu()
                required_roles = synchronizer.required_roles(bwd_pass=bwd_pass)
                synchronizer.consume_pending(required_roles, axis_streams[-1])
                required_by_group.append(required_roles)
                for role in required_roles:
                    state = synchronizer.representations[role]
                    compute_weight = synchronizer.get_unsharded_buffer(role)
                    if compute_weight is not None:
                        outputs_by_group[index][role] = compute_weight
                        continue

                    source = state.persistent.view(list(state.valid_placements))
                    persistent_placements = tuple(state.persistent.placements)
                    if persistent_placements == group.full_placements:
                        output = state.persistent
                    else:
                        output = group._allocate_scratch(
                            f"full_weight:{role.value}", state.persistent, group.full_placements
                        )
                        state.full = output
                    plans.append(
                        WeightUnshardPlan(
                            group_index=index,
                            synchronizer=synchronizer,
                            role=role,
                            source_placements=state.valid_placements,
                            persistent_placements=persistent_placements,
                            source=source,
                            output=output,
                        )
                    )
        except Exception:
            for plan in plans:
                state = plan.synchronizer.representations[plan.role]
                if state.full is plan.output:
                    plan.synchronizer.owner._release_scratch(
                        f"full_weight:{plan.role.value}", plan.output
                    )
                    state.full = None
            raise

        if plans:
            DataParallelBuffer.redistribute_buffers(
                [plan.source for plan in plans],
                list(owner.full_placements),
                output_buffers=[plan.output for plan in plans],
                streams=axis_streams,
                async_op=async_op,
            )

        for plan in plans:
            state = plan.synchronizer.representations[plan.role]
            state.valid_placements = plan.persistent_placements
            terminal_axis = last_changed_axis(
                plan.source_placements, plan.synchronizer.owner.full_placements
            )
            if terminal_axis is not None:
                prior_axis = terminal_axes[plan.group_index]
                terminal_axes[plan.group_index] = (
                    terminal_axis if prior_axis is None else max(prior_axis, terminal_axis)
                )
            outputs_by_group[plan.group_index][plan.role] = plan.output

        results = []
        for index, (synchronizer, required_roles) in enumerate(
            zip(synchronizers, required_by_group)
        ):
            terminal_axis = terminal_axes[index]
            terminal_stream = (
                torch.cuda.current_stream()
                if terminal_axis is None
                else axis_streams[terminal_axis]
            )
            with torch.cuda.stream(terminal_stream):
                for role in required_roles:
                    synchronizer._bind(outputs_by_group[index][role], role)
                synchronizer.owner.mp_policy.post_unshard(
                    synchronizer.owner.params, bwd_pass=bwd_pass
                )
            result_role = (
                WeightBufferRole.MODEL
                if WeightBufferRole.MODEL in required_roles
                else required_roles[0]
            )
            results.append(outputs_by_group[index][result_role])
        return results

    def reshard(self) -> None:
        """Release all full compute-weight representation leases."""
        self.owner.mp_policy.post_reshard(self.owner.params)
        for role, state in self.representations.items():
            self.owner._release_scratch(f"full_weight:{role.value}", state.full)
            state.full = None

    @torch.no_grad()
    def refresh_from_optimizer(self) -> None:
        """Install optimizer weights and record the optimizer placement as valid."""
        self.owner.reload_to_gpu()
        self.join_pending()
        self.reshard()
        if not self.owner._main_weight_aliases_weight:
            self.owner.mp_policy.copy_main_weights_to_model_weights(
                self.owner.params,
                self.owner.param_idx,
                self.owner.mesh,
                self.owner.weight_buffers[WeightBufferRole.MODEL],
                self.owner.main_weight_buffer,
                self.owner.transpose_weight_buffer,
                optimizer_placements=list(self.owner.layout.main_weight),
            )
        for state in self.representations.values():
            state.valid_placements = self.owner.layout.main_weight
