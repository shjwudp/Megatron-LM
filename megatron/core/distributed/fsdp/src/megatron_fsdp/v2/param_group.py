# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Placement-first data-parallel parameter group for Megatron FSDP v2."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from ..uneven_dtensor import copy_chunk_metadata, make_uneven_dtensor
from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .buffer_index import BufferIndex, Placement
from .dp_buffer import DataParallelBuffer
from .grad_sync import GradientSynchronizer
from .mixed_precision import MixedPrecisionPolicy, WeightBufferRole
from .param_group_state import (
    GradientPhase,
    GradientState,
    ParameterGroupLayout,
    ParameterGroupStateView,
    Placements,
    WeightRepresentationState,
)
from .utils import ParamGroupIdx
from .weight_sync import WeightSynchronizer

__all__ = ["GradientPhase", "ParameterGroup", "ParameterGroupLayout"]


class ParameterGroup:
    """Own persistent data-parallel values and their placement transitions.

    The implementation focuses on three semantic distributed values:

    - persistent model weights;
    - optimizer main weights;
    - persistent accumulated/reduced gradients.

    ``DataParallelBuffer`` owns layout and communication mechanics. This class
    remains the public facade and persistent-storage owner;
    :class:`WeightSynchronizer` and :class:`GradientSynchronizer` own their
    independent runtime state and synchronization algorithms.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        param_group_id: ParamGroupIdx,
        *,
        mesh: DeviceMesh,
        layout: ParameterGroupLayout,
        mp_policy: MixedPrecisionPolicy,
        allocator: BucketAllocator | None = None,
        gradient_scaling_factor: float | None = None,
    ) -> None:
        if not params:
            raise ValueError("ParameterGroup requires at least one parameter")
        if mesh.ndim not in (1, 2):
            raise ValueError(
                f"ParameterGroup expects a 1D DP or 2D hybrid-DP mesh, got {mesh.ndim}D"
            )
        layout.validate(mesh.ndim)

        self.params = params
        self.param_idx = {param: index for index, param in enumerate(params)}
        self.param_group_id = param_group_id
        self.mesh = mesh
        self.layout = layout
        self.mp_policy = mp_policy
        self.mp_policy.validate_param_group(params)
        self.allocator = allocator or TemporaryBucketAllocator()
        self.enable_full_iteration_cuda_graph = False
        self.gradient_scaling_factor = gradient_scaling_factor
        self.device = params[0].device
        self.dtype = params[0].dtype
        self.requires_grad = params[0].requires_grad
        row_sizes = [param.shape[1:].numel() for param in params if param.ndim > 1]
        self.chunk_size_factor = max(1, math.lcm(*row_sizes)) if row_sizes else 1

        self.weight_buffers: dict[WeightBufferRole, DataParallelBuffer]
        self.weight_buffer: DataParallelBuffer
        self.transpose_weight_buffer: DataParallelBuffer | None
        self.main_weight_buffer: DataParallelBuffer
        self.grad_buffer: DataParallelBuffer
        self._main_weight_aliases_weight = False
        self._initialize_buffers()
        self._optimizer_params: list[torch.nn.Parameter] = []
        self._optimizer_grads: list[DTensor | None] = []
        self._initialize_optimizer_params()
        self._weight_sync = WeightSynchronizer(self)
        self._gradient_sync = GradientSynchronizer(self)
        self.state = ParameterGroupStateView(
            self._weight_sync.representations, self._gradient_sync.state
        )

    @property
    def full_placements(self) -> Placements:
        """Fully replicated placements for this mesh."""
        return (Placement.REPLICATE,) * self.mesh.ndim

    @property
    def contribution_placements(self) -> Placements:
        """Logical placements of one local backward contribution."""
        return (Placement.PARTIAL,) * self.mesh.ndim

    @property
    def optimizer_params(self) -> list[torch.nn.Parameter]:
        """Return optimizer-facing DTensor parameters."""
        return self._optimizer_params

    @property
    def optimizer_grads(self) -> list[DTensor | None]:
        """Return gradient DTensor views matching :attr:`optimizer_params`."""
        return self._optimizer_grads

    @property
    def weight_state(self) -> dict[WeightBufferRole, WeightRepresentationState]:
        """Return runtime state for each model-weight representation."""
        return self._weight_sync.representations

    @property
    def gradient_state(self) -> GradientState:
        """Return gradient storage and accumulation state."""
        return self._gradient_sync.state

    @property
    def accumulates_full_grad(self) -> bool:
        """Return whether microbatches accumulate in persistent full-gradient storage."""
        return self._gradient_sync.accumulates_full_grad

    @property
    def full_grad_has_value(self) -> bool:
        """Return whether full-gradient storage contains prior accumulation."""
        return self._gradient_sync.full_grad_has_value

    @property
    def overwrites_full_grad(self) -> bool:
        """Return whether this backward initializes rather than accumulates full gradients."""
        return self._gradient_sync.overwrites_full_grad

    @property
    def supports_fused_grad_capture(self) -> bool:
        """Return whether fused wgrad can target this group's full-gradient storage."""
        return self._gradient_sync.supports_fused_grad_capture

    def set_allocator(self, allocator: BucketAllocator) -> None:
        """Replace the allocator used for temporary buffer leases."""
        self.allocator = allocator

    def _persistent_storage_owners(self) -> list[DataParallelBuffer]:
        """Return distinct buffers that own persistent storage."""
        owners = list(self.weight_buffers.values())
        if not self._main_weight_aliases_weight:
            owners.append(self.main_weight_buffer)
        owners.append(self.grad_buffer)
        return [buffer for buffer in owners if buffer.data is not None]

    @staticmethod
    def _rebind_dtensor_storage(
        dtensor: DTensor, local_tensor: torch.Tensor, shape: torch.Size
    ) -> None:
        """Rebind one optimizer DTensor while preserving checkpoint metadata."""
        old_local_tensor = dtensor._local_tensor
        if local_tensor.numel() == 0:
            local_shape = (0,) + tuple(shape[1:]) if len(shape) > 1 else (0,)
            new_local_tensor = local_tensor.reshape(local_shape)
        else:
            new_local_tensor = local_tensor.view(-1, *shape[1:])
        for attr_name in (
            "__create_chunk_list__",
            "__create_write_items__",
            "_chunk_meta_source",
        ):
            if hasattr(old_local_tensor, attr_name):
                setattr(new_local_tensor, attr_name, getattr(old_local_tensor, attr_name))
        dtensor._local_tensor = new_local_tensor

    def _rebuild_persistent_views(self) -> None:
        """Rebuild aliases and optimizer DTensor views after storage migration."""
        if self._main_weight_aliases_weight:
            self.main_weight_buffer = self.weight_buffer.view(list(self.layout.main_weight))

        optimizer_view = self.main_weight_buffer.view(list(self.layout.main_weight))
        for param, optimizer_param in zip(self.params, self._optimizer_params):
            self._rebind_dtensor_storage(
                optimizer_param,
                optimizer_view.tensor_view(self.param_idx[param]),
                param.shape,
            )

        if self.grad_buffer.data is None:
            return
        if self.accumulates_full_grad:
            self.gradient_state.full = self._gradient_sync.placement_view(
                self.grad_buffer, self.contribution_placements
            )
        grad_view = self._gradient_sync.placement_view(
            self.grad_buffer, self.layout.main_weight
        )
        for param, optimizer_grad in zip(self.params, self._optimizer_grads):
            if optimizer_grad is None:
                continue
            self._rebind_dtensor_storage(
                optimizer_grad,
                grad_view.tensor_view(self.param_idx[param]),
                param.shape,
            )

    @torch.no_grad()
    def offload_to_cpu(
        self, *, pin_memory: bool = False, max_cpu_bytes: int | None = None
    ) -> tuple[int, int]:
        """Move persistent storage to CPU and return offloaded/skipped byte counts."""
        self._weight_sync.join_pending()
        self.reshard_weight()
        self.release_temporary_grad_buffers()
        offloaded_bytes = 0
        skipped_bytes = 0
        owners = sorted(
            self._persistent_storage_owners(),
            key=lambda buffer: buffer.data.nbytes,
            reverse=True,
        )
        for buffer in owners:
            if buffer.data.device.type == "cpu":
                continue
            num_bytes = buffer.data.nbytes
            if max_cpu_bytes is not None and offloaded_bytes + num_bytes > max_cpu_bytes:
                skipped_bytes += num_bytes
                continue
            cpu_data = torch.empty(
                buffer.data.shape,
                dtype=buffer.data.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(buffer.data)
            _free_storage(buffer.data)
            buffer.data = cpu_data
            offloaded_bytes += num_bytes
        self._rebuild_persistent_views()
        return offloaded_bytes, skipped_bytes

    @torch.no_grad()
    def reload_to_gpu(self) -> None:
        """Move offloaded persistent storage back to its configured CUDA device."""
        moved = False
        for buffer in self._persistent_storage_owners():
            if buffer.data.device == buffer.device:
                continue
            buffer.data = buffer.data.to(buffer.device)
            moved = True
        if moved:
            self._rebuild_persistent_views()

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
        self.weight_buffers = {WeightBufferRole.MODEL: self.weight_buffer}

        self.transpose_weight_buffer = None
        if self.mp_policy.needs_transpose_weight_buffer(self.params[0]):
            self.transpose_weight_buffer = self._new_buffer(torch.uint8, self.layout.weight)
            self._allocate_persistent(self.transpose_weight_buffer)
            self.transpose_weight_buffer.copy_tensors_(
                self.mp_policy.get_param_data(param, transpose=True) for param in self.params
            )
            self.weight_buffers[WeightBufferRole.TRANSPOSE] = self.transpose_weight_buffer

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

        # Parameter data has been copied into FSDP-owned persistent storage.
        # Unshard will rebind the compute parameters to a full placement view.
        for param in self.params:
            for tensor in self.mp_policy.storage_tensors_to_free(
                param, self.weight_buffer, self.main_weight_buffer
            ):
                _free_storage(tensor)

        grad_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
        self.grad_buffer = self._new_buffer(grad_dtype, self.layout.grad_storage)

    @staticmethod
    def _dtensor_placements(placements: Placements) -> list[Replicate | Shard]:
        dtensor_placements = []
        for placement in placements:
            if placement is Placement.REPLICATE:
                dtensor_placements.append(Replicate())
            elif placement is Placement.SHARD:
                dtensor_placements.append(Shard(dim=0))
            else:
                raise ValueError(f"Optimizer values cannot use {placement} placement")
        return dtensor_placements

    def _initialize_optimizer_params(self) -> None:
        """Create optimizer-facing DTensor parameters over persistent main weights."""
        placements = self._dtensor_placements(self.layout.main_weight)
        optimizer_view = self.main_weight_buffer.view(list(self.layout.main_weight))
        if self.mesh.ndim == 2 and self.layout.main_weight[0] is Placement.SHARD:
            setattr(self.mesh, "_shard_order", [1, 0])

        for param in self.params:
            local_data = optimizer_view.tensor_view(self.param_idx[param])
            dist_data = make_uneven_dtensor(
                local_data, param.shape, self.mesh, placements, post_process_uneven=True
            )
            optimizer_param = torch.nn.Parameter(dist_data, requires_grad=param.requires_grad)
            copy_chunk_metadata(dist_data, optimizer_param)
            setattr(param, "__fsdp_param__", True)
            setattr(optimizer_param, "__fsdp_param__", True)
            self._optimizer_params.append(optimizer_param)
            self._optimizer_grads.append(None)

    def prepare_gradient_storage(self) -> None:
        """Materialize persistent optimizer-gradient storage and DTensor views."""
        self._gradient_sync.prepare_storage()

    def _ensure_grad_storage(self) -> None:
        self._gradient_sync.ensure_storage()

    def _release_grad_storage(self) -> None:
        self._gradient_sync.release_storage()

    def _initialize_optimizer_grads(self) -> None:
        self._gradient_sync.initialize_optimizer_grads()

    def _install_optimizer_grads(self) -> None:
        self._gradient_sync.install_optimizer_grads()

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

    def get_unsharded_weight_buffer(
        self, role: WeightBufferRole = WeightBufferRole.MODEL
    ) -> DataParallelBuffer | None:
        """Return an available unsharded weight buffer for the requested role."""
        return self._weight_sync.get_unsharded_buffer(role)

    def weights_are_unsharded(self, bwd_pass: bool = False) -> bool:
        """Return whether all compute-weight representations for this pass are available."""
        return self._weight_sync.weights_are_unsharded(bwd_pass=bwd_pass)

    @staticmethod
    def prefetch_weight_storage(
        param_groups: Sequence["ParameterGroup"],
        *,
        stream: torch.cuda.Stream,
        bwd_pass: bool = False,
    ) -> torch.cuda.Event | None:
        """Asynchronously refresh pass-specific persistent weight storage."""
        return WeightSynchronizer.prefetch_storage(
            [param_group._weight_sync for param_group in param_groups],
            stream=stream,
            bwd_pass=bwd_pass,
        )

    @staticmethod
    def unshard_weights(
        param_groups: Sequence["ParameterGroup"],
        stream: torch.cuda.Stream | None = None,
        *,
        streams: Sequence[torch.cuda.Stream | None] | None = None,
        bwd_pass: bool = False,
        async_op: bool = False,
    ) -> list[DataParallelBuffer]:
        """Unshard pass-specific weight representations in one coalesced axis plan."""
        return WeightSynchronizer.unshard(
            [param_group._weight_sync for param_group in param_groups],
            stream=stream,
            streams=streams,
            bwd_pass=bwd_pass,
            async_op=async_op,
        )

    @torch.no_grad()
    def unshard_weight(
        self,
        stream: torch.cuda.Stream | None = None,
        *,
        streams: Sequence[torch.cuda.Stream | None] | None = None,
        bwd_pass: bool = False,
        async_op: bool = False,
    ) -> DataParallelBuffer:
        """Unshard this parameter group and return its full compute weight."""
        return self.unshard_weights(
            [self],
            stream=stream,
            streams=streams,
            bwd_pass=bwd_pass,
            async_op=async_op,
        )[0]

    def reshard_weight(self) -> None:
        """Release all full compute-weight representation leases."""
        self._weight_sync.reshard()

    def release_temporary_grad_buffers(self) -> None:
        """Release per-backward gradient bindings and allocator-backed scratch buffers."""
        self._gradient_sync.release_temporaries()

    def release_grad_storage_if_unused(self) -> None:
        """Release gradient storage after optimizer-facing gradients are cleared."""
        self._gradient_sync.release_storage_if_unused()

    @torch.no_grad()
    def refresh_model_weight(self) -> None:
        """Install optimizer weights and record the optimizer placement as valid."""
        self._weight_sync.refresh_from_optimizer()

    def acquire_full_grad_buffer(self) -> DataParallelBuffer:
        """Acquire the full-size local gradient buffer used by backward."""
        return self._gradient_sync.acquire_full_buffer()

    def get_main_grad(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Return one parameter view in the current full-gradient contribution."""
        return self._gradient_sync.get_main_grad(param)

    @torch.no_grad()
    def reduce_grad(
        self,
        *,
        is_last_backward: bool,
        stream: torch.cuda.Stream | None = None,
        streams: Sequence[torch.cuda.Stream | None] | None = None,
        async_op: bool = False,
    ) -> torch.cuda.Stream:
        """Reduce one microbatch and finalize delayed DP axes on the last backward."""
        return self._gradient_sync.reduce(
            is_last_backward=is_last_backward,
            stream=stream,
            streams=streams,
            async_op=async_op,
        )


    def optimizer_weight(self) -> DataParallelBuffer:
        """Return the persistent optimizer-weight representation."""
        return self.main_weight_buffer

    def optimizer_grad(self) -> DataParallelBuffer:
        """Return the optimizer gradient after final data-parallel reduction."""
        return self._gradient_sync.optimizer_grad()

    def assert_model_weights_not_nan(self) -> None:
        """Assert that full compute weights contain no NaNs."""
        weight = self.get_unsharded_weight_buffer()
        if weight is None:
            raise RuntimeError("Model weights must be unsharded before checking for NaNs")
        for param in self.params:
            assert not torch.isnan(
                weight.tensor_view(self.param_idx[param])
            ).any(), "NaN detected in model weight buffer"

    def buffer_diagnostics(
        self,
    ) -> tuple[list[tuple[str, torch.dtype, int, int, bool, bool]], list[tuple[int, int] | None]]:
        """Return read-only buffer metadata and model-weight item ranges."""
        metadata = []
        buffers = [("W", self.weight_buffer)]
        if self.transpose_weight_buffer is not None:
            buffers.append(("WT", self.transpose_weight_buffer))
        buffers.extend((("MW", self.main_weight_buffer), ("G", self.grad_buffer)))
        for label, buffer in buffers:
            metadata.append(
                (
                    label,
                    buffer.dtype,
                    buffer.data_size,
                    buffer.buffer_index.bucket_meta.size,
                    self.mesh.ndim == 2 and buffer.placements[0] is Placement.SHARD,
                    buffer.placements[-1] is Placement.SHARD,
                )
            )
        ranges = []
        for param in self.params:
            item = self.weight_buffer.buffer_index.item_index_map.get(self.param_idx[param])
            ranges.append(None if item is None else (item.global_data_index, item.size))
        return metadata, ranges

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset logical gradient state and optimizer-facing gradients."""
        self._gradient_sync.zero_grad(set_to_none=set_to_none)
