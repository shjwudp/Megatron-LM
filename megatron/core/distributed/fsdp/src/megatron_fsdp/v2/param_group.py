# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Placement-first data-parallel parameter group for Megatron FSDP v2."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from ..uneven_dtensor import (
    copy_chunk_metadata,
    detach_uneven_dtensor_local_tensor,
    make_uneven_dtensor,
    rebind_uneven_dtensor_local_tensor,
)
from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .buffer_index import BufferIndex, Placement
from .dp_buffer import DataParallelBuffer
from .mixed_precision import MixedPrecisionPolicy
from .utils import ParamGroupIdx

Placements = tuple[Placement, ...]
AxisStreams = Sequence[torch.cuda.Stream | None]


@dataclass(frozen=True)
class ParameterGroupLayout:
    """Persistent data-parallel placements used by :class:`ParameterGroup`."""

    weight: Placements
    main_weight: Placements
    grad_storage: Placements
    grad_accumulation: Placements

    def validate(self, mesh_ndim: int) -> None:
        """Validate placement rank and the supported 2D HSDP axis convention."""
        for placements in (
            self.weight,
            self.main_weight,
            self.grad_storage,
            self.grad_accumulation,
        ):
            if len(placements) != mesh_ndim:
                raise ValueError(f"Expected {mesh_ndim} placements, got {placements}")

        if mesh_ndim != 2:
            return

        # A 2D layout is ordered as (outer DP, inner DP). Keeping model
        # weights and accumulated gradients sharded on the inner axis makes
        # weight unshard outer-to-inner and gradient reduction inner-to-outer.
        replicate_shard = (Placement.REPLICATE, Placement.SHARD)
        shard_shard = (Placement.SHARD, Placement.SHARD)
        if (
            self.weight != replicate_shard
            or self.main_weight not in (replicate_shard, shard_shard)
            or self.grad_storage != replicate_shard
            or self.grad_accumulation != (Placement.PARTIAL, Placement.SHARD)
        ):
            raise ValueError(
                "2D HSDP placements use (outer DP, inner DP) and require "
                "weight/grad_storage=(REPLICATE, SHARD), "
                "grad_accumulation=(PARTIAL, SHARD), and main_weight either "
                "(REPLICATE, SHARD) or (SHARD, SHARD)"
            )

    @classmethod
    def from_strategies(
        cls, sharding_strategy: str, outer_dp_sharding_strategy: str | None = None
    ) -> "ParameterGroupLayout":
        """Resolve public sharding strategies into a placement-only layout."""
        valid_inner = ("no_shard", "optim", "optim_grads", "optim_grads_params")
        if sharding_strategy not in valid_inner:
            raise ValueError(f"Unsupported sharding strategy: {sharding_strategy}")

        weight = (
            Placement.SHARD if sharding_strategy == "optim_grads_params" else Placement.REPLICATE
        )
        optimizer = Placement.REPLICATE if sharding_strategy == "no_shard" else Placement.SHARD
        reduce_each_microbatch = sharding_strategy in ("optim_grads", "optim_grads_params")
        grad_accumulation = Placement.SHARD if reduce_each_microbatch else Placement.PARTIAL
        grad_storage = Placement.SHARD if reduce_each_microbatch else Placement.REPLICATE
        inner_layout = cls(
            weight=(weight,),
            main_weight=(optimizer,),
            grad_storage=(grad_storage,),
            grad_accumulation=(grad_accumulation,),
        )
        if outer_dp_sharding_strategy is None:
            return inner_layout

        if outer_dp_sharding_strategy not in ("no_shard", "optim"):
            raise ValueError(
                f"Unsupported outer DP sharding strategy: {outer_dp_sharding_strategy}"
            )
        if outer_dp_sharding_strategy == "optim" and sharding_strategy != "optim_grads_params":
            raise NotImplementedError(
                "Outer-DP optimizer sharding requires inner optim_grads_params, "
                f"got {sharding_strategy}"
            )
        outer_optimizer = (
            Placement.SHARD if outer_dp_sharding_strategy == "optim" else Placement.REPLICATE
        )
        return cls(
            weight=(Placement.REPLICATE, inner_layout.weight[0]),
            main_weight=(outer_optimizer, inner_layout.main_weight[0]),
            grad_storage=(Placement.REPLICATE, inner_layout.grad_storage[0]),
            grad_accumulation=(Placement.PARTIAL, inner_layout.grad_accumulation[0]),
        )

    @classmethod
    def fsdp(cls) -> "ParameterGroupLayout":
        """Build a one-dimensional fully sharded layout."""
        return cls.from_strategies("optim_grads_params")

    @classmethod
    def hsdp(cls, *, shard_optimizer_across_outer_dp: bool) -> "ParameterGroupLayout":
        """Build the two-dimensional HSDP layout discussed in the design document."""
        return cls.from_strategies(
            "optim_grads_params",
            outer_dp_sharding_strategy=("optim" if shard_optimizer_across_outer_dp else "no_shard"),
        )


class GradientPhase(enum.Enum):
    """Lifecycle phase of the value stored in the persistent gradient buffer."""

    EMPTY = enum.auto()
    ACCUMULATING = enum.auto()
    READY = enum.auto()


@dataclass
class ParameterGroupState:
    """Minimal value-validity and scratch-lease state."""

    weight_valid: Placements
    grad_phase: GradientPhase = GradientPhase.EMPTY
    full_weight: DataParallelBuffer | None = None
    full_grad: DataParallelBuffer | None = None
    grad_comm: DataParallelBuffer | None = None


class ParameterGroup:
    """Own persistent data-parallel values and their placement transitions.

    The implementation focuses on three semantic distributed values:

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

        self.weight_buffer: DataParallelBuffer
        self.main_weight_buffer: DataParallelBuffer
        self.grad_buffer: DataParallelBuffer
        self._main_weight_aliases_weight = False
        self._initialize_buffers()
        self.state = ParameterGroupState(weight_valid=tuple(self.weight_buffer.placements))
        self._optimizer_params: list[torch.nn.Parameter] = []
        self._optimizer_grads: list[DTensor | None] = []
        self._initialize_optimizer_params()

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

    @property
    def optimizer_params(self) -> list[torch.nn.Parameter]:
        """Return optimizer-facing DTensor parameters."""
        return self._optimizer_params

    @property
    def optimizer_grads(self) -> list[DTensor | None]:
        """Return gradient DTensor views matching :attr:`optimizer_params`."""
        return self._optimizer_grads

    @property
    def full_grad_has_value(self) -> bool:
        """Return whether full-gradient storage contains prior accumulation."""
        return False

    @property
    def overwrites_full_grad(self) -> bool:
        """Return whether every backward overwrites its fresh full-gradient lease."""
        return self.requires_grad

    @property
    def supports_fused_grad_capture(self) -> bool:
        """Return whether fused wgrad can target this group's full-gradient storage."""
        return (
            self.requires_grad
            and self.overwrites_full_grad
            and self.grad_buffer.dtype == self.params[0].dtype
        )

    def set_allocator(self, allocator: BucketAllocator) -> None:
        """Replace the allocator used for temporary buffer leases."""
        self.allocator = allocator

    def _persistent_storage_owners(self) -> list[DataParallelBuffer]:
        """Return distinct buffers that own persistent storage."""
        owners = [self.weight_buffer]
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
        grad_view = self._placement_view(self.grad_buffer, self.layout.main_weight)
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
        self.reshard_weight()
        self.release_grad_buffer()
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

    def _ensure_grad_storage(self) -> None:
        """Lazily allocate persistent gradient storage for the current step."""
        if self.grad_buffer.data is None:
            self._allocate_persistent(self.grad_buffer)

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

    def _initialize_optimizer_grads(self) -> None:
        """Create gradient DTensor views over the final persistent gradient."""
        if self.grad_buffer.data is None:
            raise RuntimeError("Gradient storage must be allocated before creating gradient views")
        grad_view = self._placement_view(self.grad_buffer, self.layout.main_weight)
        for index, (param, optimizer_param) in enumerate(zip(self.params, self._optimizer_params)):
            local_grad = grad_view.tensor_view(self.param_idx[param])
            if not param.requires_grad or local_grad.numel() == 0:
                self._optimizer_grads[index] = None
                continue
            if self._optimizer_grads[index] is None:
                self._optimizer_grads[index] = make_uneven_dtensor(
                    local_grad,
                    param.shape,
                    self.mesh,
                    optimizer_param.placements,
                    copy_chunk_meta_from=optimizer_param,
                )
            elif self._optimizer_grads[index]._local_tensor is None:
                rebind_uneven_dtensor_local_tensor(
                    self._optimizer_grads[index],
                    local_grad,
                    param.shape,
                    copy_chunk_meta_from=optimizer_param,
                )

    def prepare_gradient_storage(self) -> None:
        """Materialize persistent optimizer-gradient storage and DTensor views."""
        if not self.requires_grad:
            return
        self._ensure_grad_storage()
        self._initialize_optimizer_grads()

    def _install_optimizer_grads(self) -> None:
        """Attach reduced gradients to the optimizer-facing parameters."""
        self._initialize_optimizer_grads()
        for optimizer_param, optimizer_grad in zip(self._optimizer_params, self._optimizer_grads):
            if self.mp_policy.use_decoupled_grad:
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

    def weights_are_unsharded(self, bwd_pass: bool = False) -> bool:
        """Return whether full compute weights are available."""
        _ = bwd_pass
        return self.compute_weight() is not None

    def _axis_streams(
        self, *, stream: torch.cuda.Stream | None = None, streams: AxisStreams | None = None
    ) -> tuple[torch.cuda.Stream, ...]:
        """Resolve a legacy shared stream or one stream per mesh axis."""
        if stream is not None and streams is not None:
            raise ValueError("Specify either stream or streams, not both")
        caller_stream = torch.cuda.current_stream()
        if streams is None:
            return (stream or caller_stream,) * self.mesh.ndim
        if len(streams) != self.mesh.ndim:
            raise ValueError(f"Expected {self.mesh.ndim} streams, got {len(streams)}")
        return tuple(axis_stream or caller_stream for axis_stream in streams)

    @staticmethod
    def _last_changed_axis(source: Placements, target: Placements) -> int | None:
        """Return the last changed axis in forward mesh order."""
        changed = [axis for axis, pair in enumerate(zip(source, target)) if pair[0] is not pair[1]]
        return changed[-1] if changed else None

    @staticmethod
    @torch.no_grad()
    def unshard_weights(
        param_groups: Sequence["ParameterGroup"],
        stream: torch.cuda.Stream | None = None,
        *,
        streams: AxisStreams | None = None,
        async_op: bool = False,
    ) -> list[DataParallelBuffer]:
        """Unshard compatible parameter groups in one coalesced axis plan."""
        if not param_groups:
            return []
        axis_streams = param_groups[0]._axis_streams(stream=stream, streams=streams)
        results: list[DataParallelBuffer | None] = [None] * len(param_groups)
        plans = []

        try:
            for index, param_group in enumerate(param_groups):
                if param_group.mesh.ndim != len(axis_streams):
                    raise ValueError("All parameter groups must use the same mesh dimensionality")
                param_group.reload_to_gpu()
                compute_weight = param_group.compute_weight()
                if compute_weight is not None:
                    param_group._bind_weight(compute_weight)
                    param_group.mp_policy.post_unshard(param_group.params)
                    results[index] = compute_weight
                    continue

                source_placements = param_group.state.weight_valid
                source = param_group.weight_buffer.view(list(source_placements))
                persistent_placements = tuple(param_group.weight_buffer.placements)
                if persistent_placements == param_group.full_placements:
                    output = param_group.weight_buffer
                else:
                    param_group.state.full_weight = param_group._allocate_scratch(
                        "full_weight", param_group.weight_buffer, param_group.full_placements
                    )
                    output = param_group.state.full_weight
                plans.append(
                    (
                        index,
                        param_group,
                        source_placements,
                        persistent_placements,
                        source,
                        output,
                    )
                )
        except Exception:
            for _, param_group, _, _, _, output in plans:
                if param_group.state.full_weight is output:
                    param_group._release_scratch("full_weight", output)
                    param_group.state.full_weight = None
            raise

        if plans:
            DataParallelBuffer.redistribute_buffers(
                [source for _, _, _, _, source, _ in plans],
                list(param_groups[0].full_placements),
                output_buffers=[output for _, _, _, _, _, output in plans],
                streams=axis_streams,
                async_op=async_op,
            )

        for (
            index,
            param_group,
            source_placements,
            persistent_placements,
            _,
            output,
        ) in plans:
            param_group.state.weight_valid = persistent_placements
            terminal_axis = param_group._last_changed_axis(
                source_placements, param_group.full_placements
            )
            terminal_stream = (
                torch.cuda.current_stream()
                if terminal_axis is None
                else axis_streams[terminal_axis]
            )
            with torch.cuda.stream(terminal_stream):
                param_group._bind_weight(output)
                param_group.mp_policy.post_unshard(param_group.params)
            results[index] = output

        if any(result is None for result in results):
            raise RuntimeError("Weight unshard did not produce every parameter-group output")
        return [result for result in results if result is not None]

    @torch.no_grad()
    def unshard_weight(
        self,
        stream: torch.cuda.Stream | None = None,
        *,
        streams: AxisStreams | None = None,
        async_op: bool = False,
    ) -> DataParallelBuffer:
        """Unshard this parameter group and return its full compute weight."""
        return self.unshard_weights(
            [self], stream=stream, streams=streams, async_op=async_op
        )[0]

    def reshard_weight(self) -> None:
        """Release only the full compute-weight lease."""
        self.mp_policy.post_reshard(self.params)
        self._release_scratch("full_weight", self.state.full_weight)
        self.state.full_weight = None

    def release_grad_buffer(self) -> None:
        """Release any full-gradient scratch lease."""
        for param in self.params:
            if hasattr(param, "main_grad"):
                del param.main_grad
        self._release_scratch("full_grad", self.state.full_grad)
        self.state.full_grad = None
        self._release_scratch("grad_comm", self.state.grad_comm)
        self.state.grad_comm = None

    def release_grad_storage_if_unused(self) -> None:
        """Release gradient storage after optimizer-facing gradients are cleared."""
        if self.enable_full_iteration_cuda_graph:
            return
        if self.state.grad_phase is GradientPhase.ACCUMULATING:
            return
        if any(
            getattr(param, "grad", None) is not None
            or getattr(param, "decoupled_grad", None) is not None
            for param in self._optimizer_params
        ):
            return
        self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def refresh_model_weight(self) -> None:
        """Install optimizer weights and record the optimizer placement as valid."""
        self.reload_to_gpu()
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
        """Acquire uninitialized storage for the full local-gradient contribution."""
        self._ensure_grad_storage()
        if self.state.full_grad is None:
            self.state.full_grad = self._allocate_scratch(
                "full_grad", self.grad_buffer, self.full_placements
            )
        return self.state.full_grad

    def get_main_grad(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Return one parameter view in the current full-gradient contribution."""
        full_grad = self.begin_backward()
        item_id = self.param_idx[param]
        start, end = full_grad.buffer_index._get_item_global_range(item_id)
        shape = full_grad.buffer_index.item_index_map[item_id].shape
        return full_grad.data[start:end].view(shape)

    def _preprocess_gradient(self, full_grad: DataParallelBuffer) -> DataParallelBuffer:
        comm_dtype = self.mp_policy.grad_comm_dtype or full_grad.dtype
        owner = full_grad
        if comm_dtype != full_grad.dtype:
            owner = DataParallelBuffer(
                buffer_index=full_grad.buffer_index,
                dtype=comm_dtype,
                device=full_grad.device,
                mesh=full_grad.mesh,
                placements=list(self.full_placements),
            )
            owner.bind(
                self.allocator.allocate(
                    key=(self.param_group_id, "grad_comm"),
                    size=owner.data_size,
                    dtype=owner.dtype,
                    device=owner.device,
                ).data
            )
            owner.data.copy_(full_grad.data)
            self.state.grad_comm = owner

        if self.gradient_scaling_factor not in (None, 1.0):
            owner.data.mul_(self.gradient_scaling_factor)

        return self._placement_view(owner, self.contribution_placements)

    def _finalize_gradient_placement(
        self, current: DataParallelBuffer, *, streams: tuple[torch.cuda.Stream, ...], async_op: bool
    ) -> tuple[DataParallelBuffer, torch.cuda.Stream]:
        """Redistribute delayed mesh axes into the persistent optimizer gradient."""
        target = self.layout.main_weight
        final = self._placement_view(self.grad_buffer, target)
        communication_owner = current._storage_owner or current
        terminal_stream = torch.cuda.current_stream()
        # Reduce in reverse mesh-axis order. On the supported 2D HSDP mesh
        # (outer DP, inner DP), this means inner reduce-scatter precedes outer
        # reduce-scatter. Weight unshard uses the inverse order.
        for axis in reversed(range(self.mesh.ndim)):
            if current.placements[axis] is target[axis]:
                continue
            next_placements = current.placements.copy()
            next_placements[axis] = target[axis]
            if tuple(next_placements) == target and current.dtype == final.dtype:
                output = final
            else:
                output = self._placement_view(communication_owner, tuple(next_placements))
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

    def _reduce_grad(
        self, *, is_last_backward: bool, streams: tuple[torch.cuda.Stream, ...], async_op: bool
    ) -> torch.cuda.Stream:
        """Reduce one gradient contribution and return its completion stream."""
        if self.state.full_grad is None:
            raise RuntimeError("begin_backward() must run before reduce_grad()")
        if self.state.grad_phase is GradientPhase.READY:
            raise RuntimeError("zero_grad() must run before starting another gradient")

        grad_input = self._preprocess_gradient(self.state.full_grad)
        accumulation = self._placement_view(self.grad_buffer, self.layout.grad_accumulation)
        # A pending accumulation contains reduced gradients from earlier
        # microbatches but is not yet optimizer-ready. Placement alone cannot
        # express this: ZeRO-2/FSDP use the same placement for both phases.
        has_accumulation = self.state.grad_phase is GradientPhase.ACCUMULATING
        needs_final_redistribution = self.layout.grad_accumulation != self.layout.main_weight
        if (
            has_accumulation
            or grad_input.dtype != accumulation.dtype
            or (is_last_backward and needs_final_redistribution)
        ):
            owner = grad_input._storage_owner or grad_input
            output = self._placement_view(owner, self.layout.grad_accumulation)
        else:
            output = accumulation

        terminal_stream = torch.cuda.current_stream()
        DataParallelBuffer.redistribute_buffers(
            [grad_input],
            list(self.layout.grad_accumulation),
            output_buffers=[output],
            streams=streams,
            async_op=async_op,
        )
        first_axis = self._last_changed_axis(
            tuple(grad_input.placements), self.layout.grad_accumulation
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
                    _, terminal_stream = self._finalize_gradient_placement(
                        output, streams=streams, async_op=async_op
                    )
                self.state.grad_phase = GradientPhase.READY
                self._install_optimizer_grads()
            else:
                self.state.grad_phase = GradientPhase.ACCUMULATING
        return terminal_stream

    @torch.no_grad()
    def reduce_grad(
        self,
        *,
        is_last_backward: bool,
        stream: torch.cuda.Stream | None = None,
        streams: AxisStreams | None = None,
        async_op: bool = False,
    ) -> torch.cuda.Stream:
        """Reduce one microbatch and finalize delayed DP axes on the last backward.

        Axis-indexed streams allow HSDP inner and outer reduce-scatter stages
        to run on distinct streams. The returned stream owns the terminal
        operation; an asynchronous caller records its completion event there.
        """
        caller_stream = torch.cuda.current_stream()
        axis_streams = self._axis_streams(stream=stream, streams=streams)
        try:
            terminal_stream = self._reduce_grad(
                is_last_backward=is_last_backward, streams=axis_streams, async_op=async_op
            )
        except Exception:
            self.release_grad_buffer()
            raise
        if terminal_stream == caller_stream:
            self.release_grad_buffer()
        return terminal_stream

    def optimizer_weight(self) -> DataParallelBuffer:
        """Return the persistent optimizer-weight representation."""
        return self.main_weight_buffer

    def optimizer_grad(self) -> DataParallelBuffer:
        """Return the optimizer gradient after final data-parallel reduction."""
        if self.state.grad_phase is not GradientPhase.READY:
            raise RuntimeError("Gradient is not ready for the optimizer")
        return self.grad_buffer.view(list(self.layout.main_weight))

    def assert_model_weights_not_nan(self) -> None:
        """Assert that full compute weights contain no NaNs."""
        weight = self.compute_weight()
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
        for label, buffer in (
            ("W", self.weight_buffer),
            ("MW", self.main_weight_buffer),
            ("G", self.grad_buffer),
        ):
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
        self.release_grad_buffer()
        self.state.grad_phase = GradientPhase.EMPTY
        if self.enable_full_iteration_cuda_graph:
            self.prepare_gradient_storage()
            self.grad_buffer.data.zero_()
            for optimizer_param in self._optimizer_params:
                for grad_name in ("grad", "decoupled_grad"):
                    grad = getattr(optimizer_param, grad_name, None)
                    if grad is None:
                        continue
                    local_grad = getattr(grad, "_local_tensor", None)
                    (local_grad if local_grad is not None else grad).zero_()
                    setattr(optimizer_param, "_mfsdp_keep_grad_for_cuda_graph", True)
            return

        if set_to_none:
            for optimizer_param in self._optimizer_params:
                optimizer_param.grad = None
                if hasattr(optimizer_param, "decoupled_grad"):
                    optimizer_param.decoupled_grad = None
            for optimizer_grad in self._optimizer_grads:
                if optimizer_grad is not None:
                    detach_uneven_dtensor_local_tensor(optimizer_grad)
            self.grad_buffer.unbind()
        elif self.grad_buffer.data is not None:
            self.grad_buffer.data.zero_()
