# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Placement-first ParameterGroup prototype for Megatron FSDP v2.

This module is intentionally not wired into ``fully_shard`` yet. It provides a
small, reviewable implementation of the target HSDP ownership and state model
without inheriting the lifecycle structure of ``param_group.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DeviceMesh

from .allocator import BucketAllocator, TemporaryBucketAllocator
from .buffer_index import BufferIndex, Placement
from .dp_buffer import DataParallelBuffer
from .mixed_precision import MixedPrecisionPolicy
from .utils import ParamGroupIdx, _prepare_fsdp_mesh

Placements = tuple[Placement, ...]


@dataclass(frozen=True)
class ParameterGroupLayoutV2:
    """Persistent HSDP placements used by :class:`ParameterGroupV2`."""

    weight: Placements
    main_weight: Placements
    grad_storage: Placements
    grad_accumulation: Placements

    @classmethod
    def hsdp(cls, *, shard_optimizer_across_outer_dp: bool) -> "ParameterGroupLayoutV2":
        """Build the two-dimensional HSDP layout discussed in the design document."""
        outer_optimizer = (
            Placement.SHARD if shard_optimizer_across_outer_dp else Placement.REPLICATE
        )
        return cls(
            weight=(Placement.REPLICATE, Placement.SHARD),
            main_weight=(outer_optimizer, Placement.SHARD),
            grad_storage=(Placement.REPLICATE, Placement.SHARD),
            grad_accumulation=(Placement.PARTIAL, Placement.SHARD),
        )


@dataclass
class ParameterGroupStateV2:
    """Minimal value-validity and scratch-lease state."""

    weight_valid: Placements
    grad_valid: Placements | None = None
    grad_ready: bool = False
    full_weight: DataParallelBuffer | None = None
    full_grad: DataParallelBuffer | None = None


class ParameterGroupV2:
    """Own persistent HSDP values and their placement transitions.

    This prototype focuses on the three semantic distributed values:

    - persistent model weights;
    - optimizer main weights;
    - persistent accumulated/reduced gradients.

    ``DataParallelBuffer`` owns layout and communication mechanics. This class
    owns allocation, validity, parameter binding, and gradient accumulation.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        param_group_id: ParamGroupIdx,
        *,
        mesh: DeviceMesh,
        layout: ParameterGroupLayoutV2,
        mp_policy: MixedPrecisionPolicy,
        allocator: BucketAllocator | None = None,
        gradient_scaling_factor: float | None = None,
    ) -> None:
        if not params:
            raise ValueError("ParameterGroupV2 requires at least one parameter")
        mesh = _prepare_fsdp_mesh(mesh)
        if mesh.ndim != 2:
            raise ValueError(f"ParameterGroupV2 expects a 2D HSDP mesh, got {mesh.ndim}D")
        for placements in (
            layout.weight,
            layout.main_weight,
            layout.grad_storage,
            layout.grad_accumulation,
        ):
            if len(placements) != mesh.ndim:
                raise ValueError(f"Expected {mesh.ndim} placements, got {placements}")

        self.params = params
        self.param_idx = {param: index for index, param in enumerate(params)}
        self.param_group_id = param_group_id
        self.mesh = mesh
        self.layout = layout
        self.mp_policy = mp_policy
        self.mp_policy.validate_param_group(params)
        self.allocator = allocator or TemporaryBucketAllocator()
        self.gradient_scaling_factor = gradient_scaling_factor
        self.device = params[0].device
        row_sizes = [param.shape[1:].numel() for param in params if param.ndim > 1]
        self.chunk_size_factor = max(1, math.lcm(*row_sizes)) if row_sizes else 1

        self.weight_buffer: DataParallelBuffer
        self.main_weight_buffer: DataParallelBuffer
        self.grad_buffer: DataParallelBuffer
        self._main_weight_aliases_weight = False
        self._initialize_buffers()
        self.state = ParameterGroupStateV2(weight_valid=tuple(self.weight_buffer.placements))

        if self.state.weight_valid == self.full_placements:
            self._bind_weight(self.weight_buffer)

    @property
    def full_placements(self) -> Placements:
        """Fully replicated placements for this mesh."""
        return (Placement.REPLICATE,) * self.mesh.ndim

    @property
    def contribution_placements(self) -> Placements:
        """Logical placements of one local backward contribution."""
        return (Placement.PARTIAL,) * self.mesh.ndim

    def _new_index(self, *, compact_weight: bool = False) -> BufferIndex:
        index = BufferIndex(
            param_shapes=[param.shape for param in self.params],
            mesh=self.mesh,
            param_group_id=self.param_group_id,
            chunk_size_factor=self.chunk_size_factor,
        )
        if compact_weight and any(self.mp_policy.is_nvfp4_param(param) for param in self.params):
            index.compact(0.5, self.mp_policy.get_param_storage_shapes(self.params))
        return index

    def _new_buffer(
        self, dtype: torch.dtype, placements: Placements, *, compact_weight: bool = False
    ) -> DataParallelBuffer:
        return DataParallelBuffer(
            buffer_index=self._new_index(compact_weight=compact_weight),
            dtype=dtype,
            device=self.device,
            mesh=self.mesh,
            placements=list(placements),
        )

    @staticmethod
    def _allocate_persistent(buffer: DataParallelBuffer) -> None:
        buffer.bind(torch.empty(buffer.data_size, dtype=buffer.dtype, device=buffer.device))

    def _initialize_buffers(self) -> None:
        model_dtype = self.mp_policy.model_weight_buffer_dtype(self.params[0])
        self.weight_buffer = self._new_buffer(model_dtype, self.layout.weight, compact_weight=True)
        self._allocate_persistent(self.weight_buffer)
        self.weight_buffer.copy_tensors_(
            self.mp_policy.get_param_data(param) for param in self.params
        )

        main_dtype = self.mp_policy.main_params_dtype_for_param(self.params[0]) or model_dtype
        if main_dtype == model_dtype:
            self.main_weight_buffer = self.weight_buffer.view(list(self.layout.main_weight))
            self._main_weight_aliases_weight = True
        else:
            self.main_weight_buffer = self._new_buffer(main_dtype, self.layout.main_weight)
            self._allocate_persistent(self.main_weight_buffer)
            self.main_weight_buffer.copy_tensors_(
                self.mp_policy.get_high_precision_value(param).detach().to(main_dtype)
                for param in self.params
            )

        grad_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
        self.grad_buffer = self._new_buffer(grad_dtype, self.layout.grad_storage)
        self._allocate_persistent(self.grad_buffer)

    def _allocate_scratch(
        self, role: str, prototype: DataParallelBuffer, placements: Placements
    ) -> DataParallelBuffer:
        output = prototype.placeholder(list(placements))
        output.bind(
            self.allocator.allocate(
                key=(self.param_group_id, role),
                size=output.data_size,
                dtype=output.dtype,
                device=output.device,
            ).data
        )
        return output

    def _release_scratch(self, role: str, buffer: DataParallelBuffer | None) -> None:
        if buffer is None:
            return
        buffer.unbind()
        self.allocator.free((self.param_group_id, role))

    @staticmethod
    def _placement_view(owner: DataParallelBuffer, placements: Placements) -> DataParallelBuffer:
        physical = tuple(
            Placement.REPLICATE if placement is Placement.PARTIAL else placement
            for placement in placements
        )
        view = owner.view(list(physical))
        return view if physical == placements else view.reinterpret(list(placements))

    def _bind_weight(self, buffer: DataParallelBuffer) -> None:
        if buffer.data is None:
            raise RuntimeError("Cannot bind parameters from an unbound weight buffer")
        for param in self.params:
            item_id = self.param_idx[param]
            start, end = self.weight_buffer.buffer_index._get_item_global_range(item_id)
            shape = self.weight_buffer.buffer_index.item_index_map[item_id].shape
            self.mp_policy.bind_unsharded_param(
                param, buffer.data[start:end].view(shape), "model_weight"
            )

    def compute_weight(self) -> DataParallelBuffer | None:
        """Return the full compute-weight buffer when it is currently available."""
        if self.state.weight_valid == self.full_placements:
            return self.weight_buffer
        return self.state.full_weight

    @torch.no_grad()
    def unshard_weight(self) -> DataParallelBuffer:
        """Restore persistent weight validity, materialize full weights, and bind params."""
        compute_weight = self.compute_weight()
        if compute_weight is not None:
            self._bind_weight(compute_weight)
            self.mp_policy.post_unshard(self.params)
            return compute_weight

        current = self.weight_buffer.view(list(self.state.weight_valid))
        persistent_placements = tuple(self.weight_buffer.placements)
        if self.state.weight_valid != persistent_placements:
            current.redistribute(list(persistent_placements), output_buffer=self.weight_buffer)
            self.state.weight_valid = persistent_placements
            current = self.weight_buffer

        if persistent_placements != self.full_placements:
            self.state.full_weight = self._allocate_scratch(
                "full_weight", self.weight_buffer, self.full_placements
            )
            current.redistribute(list(self.full_placements), output_buffer=self.state.full_weight)
            current = self.state.full_weight

        self._bind_weight(current)
        self.mp_policy.post_unshard(self.params)
        return current

    def reshard_weight(self) -> None:
        """Release only the full compute-weight lease."""
        self.mp_policy.post_reshard(self.params)
        self._release_scratch("full_weight", self.state.full_weight)
        self.state.full_weight = None

    @torch.no_grad()
    def refresh_model_weight(self) -> None:
        """Install optimizer weights and record the optimizer placement as valid."""
        self.reshard_weight()
        if not self._main_weight_aliases_weight:
            self.mp_policy.copy_main_weights_to_model_weights(
                self.params,
                self.param_idx,
                self.mesh,
                self.weight_buffer,
                self.main_weight_buffer,
                None,
                optimizer_placements=list(self.layout.main_weight),
            )
        self.state.weight_valid = self.layout.main_weight

    def begin_backward(self) -> DataParallelBuffer:
        """Acquire and zero the full local-gradient contribution."""
        if self.state.full_grad is None:
            self.state.full_grad = self._allocate_scratch(
                "full_grad", self.grad_buffer, self.full_placements
            )
            self.state.full_grad.data.zero_()
        return self.state.full_grad

    def get_main_grad(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Return one parameter view in the current full-gradient contribution."""
        full_grad = self.begin_backward()
        item_id = self.param_idx[param]
        start, end = full_grad.buffer_index._get_item_global_range(item_id)
        shape = full_grad.buffer_index.item_index_map[item_id].shape
        return full_grad.data[start:end].view(shape)

    def _preprocess_gradient(
        self, full_grad: DataParallelBuffer
    ) -> tuple[DataParallelBuffer, tuple | None]:
        comm_dtype = self.mp_policy.grad_comm_dtype or full_grad.dtype
        workspace_key = None
        owner = full_grad
        if comm_dtype != full_grad.dtype:
            workspace_key = (self.param_group_id, "grad_comm")
            owner = DataParallelBuffer(
                buffer_index=full_grad.buffer_index,
                dtype=comm_dtype,
                device=full_grad.device,
                mesh=full_grad.mesh,
                placements=list(self.full_placements),
            )
            owner.bind(
                self.allocator.allocate(
                    key=workspace_key, size=owner.data_size, dtype=owner.dtype, device=owner.device
                ).data
            )
            owner.data.copy_(full_grad.data)

        if self.gradient_scaling_factor not in (None, 1.0):
            owner.data.mul_(self.gradient_scaling_factor)

        return self._placement_view(owner, self.contribution_placements), workspace_key

    @torch.no_grad()
    def reduce_grad(self, *, is_last_backward: bool) -> None:
        """Reduce one HSDP microbatch and finalize outer DP on the last backward."""
        if self.state.full_grad is None:
            raise RuntimeError("begin_backward() must run before reduce_grad()")

        grad_input, workspace_key = self._preprocess_gradient(self.state.full_grad)
        accumulation = self._placement_view(self.grad_buffer, self.layout.grad_accumulation)
        has_accumulation = self.state.grad_valid == self.layout.grad_accumulation
        if is_last_backward or has_accumulation or grad_input.dtype != accumulation.dtype:
            owner = grad_input._storage_owner or grad_input
            output = self._placement_view(owner, self.layout.grad_accumulation)
        else:
            output = accumulation

        try:
            grad_input.redistribute(list(self.layout.grad_accumulation), output_buffer=output)
            if output.data.data_ptr() != accumulation.data.data_ptr():
                if has_accumulation:
                    output.data.add_(accumulation.data)
                if not is_last_backward:
                    accumulation.data.copy_(output.data)
            self.state.grad_valid = self.layout.grad_accumulation
            self.state.grad_ready = False

            if is_last_backward:
                final = self._placement_view(self.grad_buffer, self.layout.main_weight)
                communication_owner = output._storage_owner or output
                communication_final = self._placement_view(
                    communication_owner, self.layout.main_weight
                )
                output.redistribute(
                    list(self.layout.main_weight), output_buffer=communication_final
                )
                final.data.copy_(communication_final.data)
                self.state.grad_valid = self.layout.main_weight
                self.state.grad_ready = True
        finally:
            if workspace_key is not None:
                self.allocator.free(workspace_key)
            self._release_scratch("full_grad", self.state.full_grad)
            self.state.full_grad = None

    def optimizer_weight(self) -> DataParallelBuffer:
        """Return the persistent optimizer-weight representation."""
        return self.main_weight_buffer

    def optimizer_grad(self) -> DataParallelBuffer:
        """Return the optimizer gradient after final HSDP reduction."""
        if not self.state.grad_ready or self.state.grad_valid != self.layout.main_weight:
            raise RuntimeError("Gradient is not ready for the optimizer")
        return self.grad_buffer.view(list(self.layout.main_weight))

    def zero_grad(self) -> None:
        """Reset logical gradient state and release an unfinished contribution."""
        self._release_scratch("full_grad", self.state.full_grad)
        self.state.full_grad = None
        self.state.grad_valid = None
        self.state.grad_ready = False
        self.grad_buffer.data.zero_()
