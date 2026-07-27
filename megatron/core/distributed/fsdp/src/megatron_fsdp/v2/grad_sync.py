# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gradient storage and synchronization for Megatron FSDP parameter groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.distributed.tensor import DeviceMesh, DTensor

from ..uneven_dtensor import (
    detach_uneven_dtensor_local_tensor,
    make_uneven_dtensor,
    rebind_uneven_dtensor_local_tensor,
)
from .allocator import BucketAllocator
from .buffer_index import Placement
from .dp_buffer import DataParallelBuffer
from .mixed_precision import MixedPrecisionPolicy
from .param_group_state import GradientPhase, GradientState, ParameterGroupLayout, Placements
from .sync_utils import last_changed_axis, resolve_axis_streams
from .utils import ParamGroupIdx


class GradientSyncOwner(Protocol):
    """Parameter-group resources required by :class:`GradientSynchronizer`."""

    params: list[torch.nn.Parameter]
    param_idx: dict[torch.nn.Parameter, int]
    param_group_id: ParamGroupIdx
    mesh: DeviceMesh
    layout: ParameterGroupLayout
    mp_policy: MixedPrecisionPolicy
    allocator: BucketAllocator
    gradient_scaling_factor: float | None
    requires_grad: bool
    enable_full_iteration_cuda_graph: bool
    grad_buffer: DataParallelBuffer
    _optimizer_params: list[torch.nn.Parameter]
    _optimizer_grads: list[DTensor | None]

    @property
    def full_placements(self) -> Placements: ...

    @property
    def contribution_placements(self) -> Placements: ...

    def _allocate_persistent(self, buffer: DataParallelBuffer) -> None: ...

    def _allocate_scratch(
        self, role: str, prototype: DataParallelBuffer, placements: Placements
    ) -> DataParallelBuffer: ...

    def _release_scratch(self, role: str, buffer: DataParallelBuffer | None) -> None: ...


@dataclass(frozen=True)
class PreparedGradient:
    """A logical gradient view and the storage from which further views are derived."""

    buffer: DataParallelBuffer
    storage: DataParallelBuffer


class GradientSynchronizer:
    """Own gradient storage, accumulation, reduction, and optimizer installation."""

    def __init__(self, owner: GradientSyncOwner) -> None:
        self.owner = owner
        self.state = GradientState(persistent=owner.grad_buffer)

    @property
    def accumulates_full_grad(self) -> bool:
        """Return whether microbatches accumulate in persistent full-gradient storage."""
        return (
            self.owner.layout.grad_storage == self.owner.full_placements
            and self.owner.layout.grad_accumulation == self.owner.contribution_placements
        )

    @property
    def full_grad_has_value(self) -> bool:
        """Return whether full-gradient storage contains prior accumulation."""
        return self.accumulates_full_grad and self.state.phase is GradientPhase.ACCUMULATING

    @property
    def overwrites_full_grad(self) -> bool:
        """Return whether this backward initializes rather than accumulates full gradients."""
        return self.owner.requires_grad and not self.full_grad_has_value

    @property
    def supports_fused_grad_capture(self) -> bool:
        """Return whether fused wgrad can target this group's full-gradient storage."""
        return (
            self.owner.requires_grad
            and self.overwrites_full_grad
            and self.state.persistent.dtype == self.owner.params[0].dtype
        )

    @staticmethod
    def placement_view(owner: DataParallelBuffer, placements: Placements) -> DataParallelBuffer:
        """Return a physical view reinterpreted with logical partial placements."""
        physical = tuple(
            Placement.REPLICATE if placement is Placement.PARTIAL else placement
            for placement in placements
        )
        view = owner.view(list(physical))
        return view if physical == placements else view.reinterpret(list(placements))

    def ensure_storage(self) -> None:
        """Lazily allocate persistent gradient storage for the current step."""
        if self.state.persistent.data is None:
            self.owner._allocate_persistent(self.state.persistent)
        if self.accumulates_full_grad and self.state.full is None:
            self.state.full = self.placement_view(
                self.state.persistent, self.owner.contribution_placements
            )

    def release_storage(self) -> None:
        """Release persistent gradient storage after temporary leases are gone."""
        if self.state.full is not None:
            if not self.accumulates_full_grad:
                raise RuntimeError("Temporary full-gradient storage must be released first")
            self.state.full.unbind()
            self.state.full = None
        self.state.persistent.unbind()

    def initialize_optimizer_grads(self) -> None:
        """Create gradient DTensor views over the final persistent gradient."""
        if self.state.persistent.data is None:
            raise RuntimeError("Gradient storage must be allocated before creating gradient views")
        grad_view = self.placement_view(self.state.persistent, self.owner.layout.main_weight)
        for index, (param, optimizer_param) in enumerate(
            zip(self.owner.params, self.owner._optimizer_params)
        ):
            local_grad = grad_view.tensor_view(self.owner.param_idx[param])
            if not param.requires_grad or local_grad.numel() == 0:
                self.owner._optimizer_grads[index] = None
                continue
            if self.owner._optimizer_grads[index] is None:
                self.owner._optimizer_grads[index] = make_uneven_dtensor(
                    local_grad,
                    param.shape,
                    self.owner.mesh,
                    optimizer_param.placements,
                    copy_chunk_meta_from=optimizer_param,
                )
            elif self.owner._optimizer_grads[index]._local_tensor is None:
                rebind_uneven_dtensor_local_tensor(
                    self.owner._optimizer_grads[index],
                    local_grad,
                    param.shape,
                    copy_chunk_meta_from=optimizer_param,
                )

    def prepare_storage(self) -> None:
        """Materialize persistent optimizer-gradient storage and DTensor views."""
        if not self.owner.requires_grad:
            return
        self.ensure_storage()
        self.initialize_optimizer_grads()

    def install_optimizer_grads(self) -> None:
        """Attach reduced gradients to the optimizer-facing parameters."""
        self.initialize_optimizer_grads()
        for optimizer_param, optimizer_grad in zip(
            self.owner._optimizer_params, self.owner._optimizer_grads
        ):
            if self.owner.mp_policy.use_decoupled_grad:
                optimizer_param.grad = None
                setattr(optimizer_param, "decoupled_grad", optimizer_grad)
                continue
            if optimizer_grad is not None and optimizer_param.dtype != optimizer_grad.dtype:
                raise RuntimeError(
                    "Optimizer parameter and gradient dtypes must match unless "
                    "use_decoupled_grad is enabled"
                )
            optimizer_param.grad = optimizer_grad
            if hasattr(optimizer_param, "decoupled_grad"):
                optimizer_param.decoupled_grad = None

    def release_temporaries(self) -> None:
        """Release per-backward gradient bindings and allocator-backed scratch buffers."""
        for param in self.owner.params:
            if hasattr(param, "main_grad"):
                del param.main_grad
        if not self.accumulates_full_grad:
            self.owner._release_scratch("full_grad", self.state.full)
            self.state.full = None
        self.owner._release_scratch("grad_comm", self.state.communication)
        self.state.communication = None

    def release_storage_if_unused(self) -> None:
        """Release gradient storage after optimizer-facing gradients are cleared."""
        if self.owner.enable_full_iteration_cuda_graph:
            return
        if self.state.phase is GradientPhase.ACCUMULATING:
            return
        if any(
            getattr(param, "grad", None) is not None
            or getattr(param, "decoupled_grad", None) is not None
            for param in self.owner._optimizer_params
        ):
            return
        self.zero_grad(set_to_none=True)

    def acquire_full_buffer(self) -> DataParallelBuffer:
        """Acquire the full-size local gradient buffer used by backward."""
        self.ensure_storage()
        if self.state.full is None:
            self.state.full = self.owner._allocate_scratch(
                "full_grad", self.state.persistent, self.owner.full_placements
            )
        return self.state.full

    def get_main_grad(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Return one parameter view in the current full-gradient contribution."""
        full_grad = self.acquire_full_buffer()
        item_id = self.owner.param_idx[param]
        start, end = full_grad.buffer_index._get_item_global_range(item_id)
        shape = full_grad.buffer_index.item_index_map[item_id].shape
        return full_grad.data[start:end].view(shape)

    def preprocess(self, full_grad: DataParallelBuffer) -> PreparedGradient:
        """Apply communication dtype and global scaling to one full gradient."""
        comm_dtype = self.owner.mp_policy.grad_comm_dtype or full_grad.dtype
        storage = self.state.persistent if self.accumulates_full_grad else full_grad
        if comm_dtype != full_grad.dtype:
            storage = DataParallelBuffer(
                buffer_index=full_grad.buffer_index,
                dtype=comm_dtype,
                device=full_grad.device,
                mesh=full_grad.mesh,
                placements=list(self.owner.full_placements),
            )
            storage.bind(
                self.owner.allocator.allocate(
                    key=(self.owner.param_group_id, "grad_comm"),
                    size=storage.data_size,
                    dtype=storage.dtype,
                    device=storage.device,
                ).data
            )
            storage.data.copy_(full_grad.data)
            self.state.communication = storage

        if self.owner.gradient_scaling_factor not in (None, 1.0):
            storage.data.mul_(self.owner.gradient_scaling_factor)

        if comm_dtype == full_grad.dtype and (
            tuple(full_grad.placements) == self.owner.contribution_placements
        ):
            return PreparedGradient(full_grad, storage)
        return PreparedGradient(
            self.placement_view(storage, self.owner.contribution_placements), storage
        )

    def finalize_placement(
        self,
        current: DataParallelBuffer,
        *,
        communication_storage: DataParallelBuffer,
        streams: tuple[torch.cuda.Stream, ...],
        async_op: bool,
    ) -> tuple[DataParallelBuffer, torch.cuda.Stream]:
        """Redistribute delayed mesh axes into the persistent optimizer gradient."""
        target = self.owner.layout.main_weight
        final = self.placement_view(self.state.persistent, target)
        terminal_stream = torch.cuda.current_stream()
        # Reduce in reverse mesh-axis order. On the supported 2D HSDP mesh
        # (outer DP, inner DP), this means inner reduce-scatter precedes outer
        # reduce-scatter. Weight unshard uses the inverse order.
        for axis in reversed(range(self.owner.mesh.ndim)):
            if current.placements[axis] is target[axis]:
                continue
            next_placements = current.placements.copy()
            next_placements[axis] = target[axis]
            if tuple(next_placements) == target and current.dtype == final.dtype:
                output = final
            else:
                output = self.placement_view(communication_storage, tuple(next_placements))
            with torch.cuda.stream(terminal_stream):
                DataParallelBuffer.redistribute_buffers(
                    [current],
                    next_placements,
                    output_buffers=[output],
                    streams=streams,
                    async_op=async_op,
                )
            terminal_stream = streams[axis]
            current = output

        with torch.cuda.stream(terminal_stream):
            if current.data.data_ptr() != final.data.data_ptr():
                final.data.copy_(current.data)
        return final, terminal_stream

    def _reduce(
        self, *, is_last_backward: bool, streams: tuple[torch.cuda.Stream, ...], async_op: bool
    ) -> torch.cuda.Stream:
        """Reduce one gradient contribution and return its completion stream."""
        if self.state.full is None:
            raise RuntimeError("acquire_full_grad_buffer() must run before reduce_grad()")
        if self.state.phase is GradientPhase.READY:
            raise RuntimeError("zero_grad() must run before starting another gradient")

        if self.accumulates_full_grad:
            if not is_last_backward:
                # Backward already copied or added this microbatch directly into
                # persistent full-gradient storage. Delay scaling, dtype conversion,
                # and the DP reduction until the final microbatch.
                self.state.phase = GradientPhase.ACCUMULATING
                return torch.cuda.current_stream()

            prepared = self.preprocess(self.state.full)
            _, terminal_stream = self.finalize_placement(
                prepared.buffer,
                communication_storage=prepared.storage,
                streams=streams,
                async_op=async_op,
            )
            self.state.phase = GradientPhase.READY
            self.install_optimizer_grads()
            return terminal_stream

        prepared = self.preprocess(self.state.full)
        grad_input = prepared.buffer
        accumulation = self.placement_view(
            self.state.persistent, self.owner.layout.grad_accumulation
        )
        # A pending accumulation contains reduced gradients from earlier
        # microbatches but is not yet optimizer-ready. Placement alone cannot
        # express this: ZeRO-2/FSDP use the same placement for both phases.
        has_accumulation = self.state.phase is GradientPhase.ACCUMULATING
        needs_final_redistribution = (
            self.owner.layout.grad_accumulation != self.owner.layout.main_weight
        )
        if (
            has_accumulation
            or grad_input.dtype != accumulation.dtype
            or (is_last_backward and needs_final_redistribution)
        ):
            output = self.placement_view(prepared.storage, self.owner.layout.grad_accumulation)
        else:
            output = accumulation

        terminal_stream = torch.cuda.current_stream()
        DataParallelBuffer.redistribute_buffers(
            [grad_input],
            list(self.owner.layout.grad_accumulation),
            output_buffers=[output],
            streams=streams,
            async_op=async_op,
        )
        first_axis = last_changed_axis(
            tuple(grad_input.placements), self.owner.layout.grad_accumulation
        )
        if first_axis is not None:
            terminal_stream = streams[first_axis]

        with torch.cuda.stream(terminal_stream):
            if output.data.data_ptr() != accumulation.data.data_ptr():
                if has_accumulation:
                    if is_last_backward and needs_final_redistribution:
                        # Keep the combined value in communication dtype; it is
                        # the input to the delayed DDP, ZeRO-1, or HSDP reduction.
                        output.data.add_(accumulation.data)
                    else:
                        accumulation.data.add_(output.data)
                        output = accumulation
                elif not is_last_backward or not needs_final_redistribution:
                    accumulation.data.copy_(output.data)
                    output = accumulation

            if is_last_backward:
                if needs_final_redistribution:
                    # This call remains inside the current terminal-stream
                    # context. The next axis stream therefore waits for the
                    # inner reduction and accumulation before consuming output.
                    _, terminal_stream = self.finalize_placement(
                        output,
                        communication_storage=prepared.storage,
                        streams=streams,
                        async_op=async_op,
                    )
                self.state.phase = GradientPhase.READY
                self.install_optimizer_grads()
            else:
                self.state.phase = GradientPhase.ACCUMULATING
        return terminal_stream

    @torch.no_grad()
    def reduce(
        self,
        *,
        is_last_backward: bool,
        stream: torch.cuda.Stream | None = None,
        streams: Sequence[torch.cuda.Stream | None] | None = None,
        async_op: bool = False,
    ) -> torch.cuda.Stream:
        """Reduce one microbatch and finalize delayed DP axes on the last backward."""
        caller_stream = torch.cuda.current_stream()
        axis_streams = resolve_axis_streams(self.owner.mesh.ndim, stream=stream, streams=streams)
        try:
            terminal_stream = self._reduce(
                is_last_backward=is_last_backward, streams=axis_streams, async_op=async_op
            )
        except Exception:
            self.release_temporaries()
            raise
        if terminal_stream == caller_stream:
            self.release_temporaries()
        return terminal_stream

    def optimizer_grad(self) -> DataParallelBuffer:
        """Return the optimizer gradient after final data-parallel reduction."""
        if self.state.phase is not GradientPhase.READY:
            raise RuntimeError("Gradient is not ready for the optimizer")
        return self.state.persistent.view(list(self.owner.layout.main_weight))

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset logical gradient state and optimizer-facing gradients."""
        self.release_temporaries()
        self.state.phase = GradientPhase.EMPTY
        if self.owner.enable_full_iteration_cuda_graph:
            self.prepare_storage()
            self.state.persistent.data.zero_()
            for optimizer_param in self.owner._optimizer_params:
                for grad_name in ("grad", "decoupled_grad"):
                    grad = getattr(optimizer_param, grad_name, None)
                    if grad is None:
                        continue
                    local_grad = getattr(grad, "_local_tensor", None)
                    (local_grad if local_grad is not None else grad).zero_()
                    setattr(optimizer_param, "_mfsdp_keep_grad_for_cuda_graph", True)
            return

        if set_to_none:
            for optimizer_param in self.owner._optimizer_params:
                optimizer_param.grad = None
                if hasattr(optimizer_param, "decoupled_grad"):
                    optimizer_param.decoupled_grad = None
            for optimizer_grad in self.owner._optimizer_grads:
                if optimizer_grad is not None:
                    detach_uneven_dtensor_local_tensor(optimizer_grad)
            self.release_storage()
        elif self.state.persistent.data is not None:
            self.state.persistent.data.zero_()
