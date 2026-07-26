# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Parameter Group for FSDP

Groups parameters that share the same (device, dtype, requires_grad) and
manages their buffers collectively. This enables efficient memory management
and collective operations across parameters.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

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
from .mixed_precision import MixedPrecisionPolicy, WeightBufferRole
from .utils import ParamGroupIdx, _prepare_fsdp_mesh


def _zero_tensor_storage(tensor: torch.Tensor) -> None:
    """Zero a Tensor or DTensor by writing only its local storage."""
    local_tensor = getattr(tensor, "_local_tensor", None)
    target = local_tensor if local_tensor is not None else tensor
    with torch.no_grad():
        target.zero_()


@dataclass
class _WeightBufferState:
    """Track one persistent weight representation and optional full scratch."""

    buffer: DataParallelBuffer
    valid_placements: tuple[Placement, ...]
    full_buffer: Optional[DataParallelBuffer] = None

    def redistribution_source(self) -> DataParallelBuffer:
        """Return the persistent buffer or its currently valid placement view."""
        if self.valid_placements == tuple(self.buffer.placements):
            return self.buffer
        return self.buffer.view(list(self.valid_placements))

    def compute_buffer(
        self, full_placements: tuple[Placement, ...]
    ) -> Optional[DataParallelBuffer]:
        """Return the currently available full compute representation."""
        if self.valid_placements == full_placements:
            return self.buffer
        return self.full_buffer


@dataclass(frozen=True)
class ParameterGroupLayout:
    """Persistent and intermediate placements for one parameter group."""

    weight: tuple[Placement, ...]
    main_weight: tuple[Placement, ...]
    grad_storage: tuple[Placement, ...]
    grad_accumulation: tuple[Placement, ...]

    @classmethod
    def from_strategies(
        cls, sharding_strategy: str, outer_dp_sharding_strategy: str
    ) -> "ParameterGroupLayout":
        """Resolve public sharding strategies into placement-only runtime layout."""
        valid_inner = ("no_shard", "optim", "optim_grads", "optim_grads_params")
        if sharding_strategy not in valid_inner:
            raise ValueError(f"Unsupported sharding strategy: {sharding_strategy}")
        if outer_dp_sharding_strategy not in ("no_shard", "optim"):
            raise ValueError(
                f"Unsupported outer DP sharding strategy: {outer_dp_sharding_strategy}"
            )
        if outer_dp_sharding_strategy == "optim" and sharding_strategy != "optim_grads_params":
            raise NotImplementedError(
                "FSDP v2 outer-DP optimizer sharding currently requires inner "
                f"optim_grads_params, got {sharding_strategy}."
            )

        outer_optimizer = (
            Placement.SHARD if outer_dp_sharding_strategy == "optim" else Placement.REPLICATE
        )
        inner_optimizer = (
            Placement.REPLICATE if sharding_strategy == "no_shard" else Placement.SHARD
        )
        inner_weight = (
            Placement.SHARD if sharding_strategy == "optim_grads_params" else Placement.REPLICATE
        )
        inner_grad = (
            Placement.SHARD
            if sharding_strategy in ("optim_grads", "optim_grads_params")
            else Placement.REPLICATE
        )
        return cls(
            weight=(Placement.REPLICATE, inner_weight),
            main_weight=(outer_optimizer, inner_optimizer),
            grad_storage=(Placement.REPLICATE, inner_grad),
            grad_accumulation=(Placement.PARTIAL, inner_grad),
        )


class ParameterGroup:
    """
    Groups parameters sharing same properties for collective buffer management.

    All parameters in a group have the same:
    - device (cuda device)
    - dtype (data type)
    - requires_grad (whether gradients are needed)

    The group manages:
    - model_weight_buffer: stores sharded model weights
    - main_weight_buffer: optional high-precision copy for mixed precision
    - main_grad_buffer: accumulates gradients before reduction
    - dist_params: DTensor views into the buffer
    - dist_grads: DTensor gradient views
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_group_id: ParamGroupIdx,
        *,
        mp_policy: MixedPrecisionPolicy,
        mesh: Optional[DeviceMesh] = None,
        sharding_strategy: str = "optim_grads_params",
        outer_dp_sharding_strategy: str = "no_shard",
        gradient_scaling_factor: Optional[float] = None,
        allocator: Optional[BucketAllocator] = None,
    ):
        self.params = params
        self.param_idx: Dict[torch.nn.Parameter, int] = {p: i for i, p in enumerate(params)}

        # Assume all params have same device/dtype/require_grad
        # TODO: validate all params have same properties
        self.device = params[0].device
        self.dtype = params[0].dtype
        self.requires_grad = params[0].requires_grad
        self.mp_policy = mp_policy
        self.mp_policy.validate_param_group(params)

        # Setup the device mesh. Collective groups are derived from the changed
        # mesh axis only when a DataParallelBuffer redistribution executes.
        if mesh is None:
            world_ranks = torch.arange(
                torch.distributed.get_world_size(torch.distributed.group.WORLD)
            ).reshape(1, -1)
            mesh = DeviceMesh(self.device.type, world_ranks, mesh_dim_names=("dp_outer", "dp"))
        mesh = _prepare_fsdp_mesh(mesh)
        self.mesh = mesh
        self.layout = ParameterGroupLayout.from_strategies(
            sharding_strategy, outer_dp_sharding_strategy
        )
        self.param_group_id = param_group_id

        # Compute chunk size factor for alignment
        # LCM ensures params align to common boundary for efficient sharding
        if len(params) > 0 and any(p.shape[1:].numel() > 0 for p in params):
            self.chunk_size_factor = max(1, math.lcm(*[p.shape[1:].numel() for p in params]))
        else:
            self.chunk_size_factor = 1

        self.gradient_scaling_factor = gradient_scaling_factor
        self.grad_comm_dtype = self.mp_policy.grad_comm_dtype
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.enable_full_iteration_cuda_graph = False
        self._full_grad_has_value = False
        self._reduced_grad_has_value = False
        self._temporary_buffers: Dict[str, DataParallelBuffer] = {}
        self._weight_buffer_states: Dict[WeightBufferRole, _WeightBufferState] = {}

        # Buffer references (initialized in _init_buffers)
        self.model_weight_buffer: Optional[DataParallelBuffer] = None
        self.transpose_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_grad_buffer: Optional[DataParallelBuffer] = None
        # Initialize buffers and distributed parameters
        self._init_buffers()
        self._weight_buffer_states = {
            role: _WeightBufferState(buffer=buffer, valid_placements=tuple(buffer.placements))
            for role, buffer in (
                (WeightBufferRole.MODEL, self.model_weight_buffer),
                (WeightBufferRole.TRANSPOSE, self.transpose_weight_buffer),
            )
            if buffer is not None
        }
        # DTensor shells cached across set_to_none gradient-buffer releases.
        # Cached entries are detached from local storage and never exposed
        # through dist_grads until _init_dist_grads rebinds them.
        self._dist_grad_cache = list(self.dist_grads)
        self._dist_grad_cache_validated = [False for _ in self.dist_grads]

    @property
    def optimizer_params(self) -> List[torch.nn.Parameter]:
        """Return optimizer-facing distributed parameters."""
        return self.dist_params

    @property
    def optimizer_grads(self) -> List[Optional[DTensor]]:
        """Return optimizer-facing distributed gradients."""
        return self.dist_grads

    @property
    def weight_buffer(self) -> Optional[DataParallelBuffer]:
        """Return the canonical model-weight buffer."""
        return self.model_weight_buffer

    @property
    def grad_buffer(self) -> Optional[DataParallelBuffer]:
        """Return the canonical optimizer-gradient buffer."""
        return self.main_grad_buffer

    @property
    def full_grad_has_value(self) -> bool:
        """Return whether the full-gradient buffer contains prior accumulation."""
        return self._full_grad_has_value

    @property
    def overwrites_full_grad(self) -> bool:
        """Return whether every backward writes a fresh inner-DP gradient shard."""
        return self.layout.grad_storage[-1] is Placement.SHARD

    @property
    def supports_fused_grad_capture(self) -> bool:
        """Return whether fused wgrad can target this group's full-gradient storage."""
        return (
            self.requires_grad
            and self.overwrites_full_grad
            and self.grad_buffer is not None
            and self.grad_buffer.dtype == self.params[0].dtype
        )

    def set_allocator(self, allocator: BucketAllocator) -> None:
        """Replace the allocator used for this group's temporary buffer leases."""
        self.allocator = allocator

    def _buffers(self) -> List[DataParallelBuffer]:
        """Return all internal buffers owned by this parameter group."""
        return [
            buffer
            for buffer in (
                self.model_weight_buffer,
                self.transpose_weight_buffer,
                self.main_weight_buffer,
                self.main_grad_buffer,
            )
            if buffer is not None
        ]

    @staticmethod
    def _move_buffer_storage_to(
        buffer: DataParallelBuffer,
        target_device: torch.device,
        *,
        pin_memory: bool = False,
        non_blocking: bool = True,
    ) -> bool:
        """Move externally owned buffer storage as a parameter-group lifecycle operation."""
        if buffer.data is None or buffer.data.device == target_device:
            return False
        if target_device.type == "cpu" and pin_memory:
            cpu_data = torch.empty(buffer.data.shape, dtype=buffer.data.dtype, pin_memory=True)
            cpu_data.copy_(buffer.data, non_blocking=non_blocking)
            _free_storage(buffer.data)
            buffer.bind(cpu_data)
        else:
            buffer.bind(buffer.data.to(target_device, non_blocking=non_blocking))
        return True

    @staticmethod
    def offload_storage_to_cpu(
        param_groups: List["ParameterGroup"],
        *,
        pin_memory: bool = False,
        max_cpu_bytes: Optional[int] = None,
    ) -> Dict[str, int]:
        """Offload persistent group storage under one cross-group CPU-memory budget."""
        entries = [
            (buffer, buffer.data.nbytes)
            for param_group in param_groups
            for buffer in param_group._buffers()
            if buffer.data is not None and buffer.data.device.type != "cpu"
        ]
        entries.sort(key=lambda entry: entry[1], reverse=True)

        offloaded_bytes = 0
        skipped_bytes = 0
        for buffer, nbytes in entries:
            if max_cpu_bytes is not None and offloaded_bytes + nbytes > max_cpu_bytes:
                skipped_bytes += nbytes
                continue
            ParameterGroup._move_buffer_storage_to(
                buffer, torch.device("cpu"), pin_memory=pin_memory
            )
            offloaded_bytes += nbytes

        for param_group in param_groups:
            param_group._rebuild_dist_views()
        return {"offloaded_bytes": offloaded_bytes, "skipped_bytes": skipped_bytes}

    @staticmethod
    def reload_storage_to_gpu(param_groups: List["ParameterGroup"]) -> None:
        """Move persistent group storage to the current CUDA device and rebuild views."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        for param_group in param_groups:
            for buffer in param_group._buffers():
                param_group._move_buffer_storage_to(buffer, device)
            param_group._rebuild_dist_views()

    def buffer_diagnostics(
        self,
    ) -> tuple[
        List[tuple[str, torch.dtype, int, int, bool, bool]], List[Optional[tuple[int, int]]]
    ]:
        """Return read-only buffer metadata and model-weight item ranges."""
        buffer_metadata = []
        for label, buffer in (
            ("W", self.model_weight_buffer),
            ("MW", self.main_weight_buffer),
            ("G", self.main_grad_buffer),
        ):
            if buffer is not None:
                buffer_metadata.append(
                    (
                        label,
                        buffer.dtype,
                        buffer.data_size,
                        buffer.buffer_index.bucket_meta.size,
                        buffer.placements[0] is Placement.SHARD,
                        buffer.placements[1] is Placement.SHARD,
                    )
                )

        model_weight_ranges = []
        for param in self.params:
            item_index = self.model_weight_buffer.buffer_index.item_index_map.get(
                self.param_idx[param]
            )
            model_weight_ranges.append(
                None if item_index is None else (item_index.global_data_index, item_index.size)
            )
        return buffer_metadata, model_weight_ranges

    def assert_model_weights_not_nan(self) -> None:
        """Assert that every fully replicated model-weight item contains no NaNs."""
        model_weights = self._weight_buffer_states[WeightBufferRole.MODEL].compute_buffer(
            self._full_placements()
        )
        if model_weights is None:
            raise RuntimeError("Model weights must be unsharded before checking for NaNs")
        for param in self.params:
            param_data = model_weights.tensor_view(self.param_idx[param])
            assert not torch.isnan(param_data).any(), "NaN detected in model weight buffer"

    def _buffer_placements(self, role: str) -> list[Placement]:
        """Return one persistent buffer role's resolved placements."""
        if role in ("model_weight", "transpose_weight"):
            placements = self.layout.weight
        elif role == "main_weight":
            placements = self.layout.main_weight
        elif role == "main_grad":
            placements = self.layout.grad_storage
        else:
            raise ValueError(f"Unsupported data-parallel buffer role: {role}")
        return list(placements)

    def _optimizer_placements(self) -> list[Placement]:
        """Return optimizer-facing placements from the resolved layout."""
        return list(self.layout.main_weight)

    def _full_placements(self) -> tuple[Placement, ...]:
        """Return fully replicated placements for this group's mesh."""
        return (Placement.REPLICATE,) * self.mesh.ndim

    def _create_buffer(self, dtype: torch.dtype, role: str) -> DataParallelBuffer:
        """Create an unbound persistent buffer layout for one storage role."""
        buffer_index = BufferIndex(
            param_shapes=[param.shape for param in self.params],
            mesh=self.mesh,
            chunk_size_factor=self.chunk_size_factor,
            param_group_id=self.param_group_id,
        )
        if role in ("model_weight", "transpose_weight") and any(
            self.mp_policy.is_nvfp4_param(param) for param in self.params
        ):
            buffer_index.compact(0.5, self.mp_policy.get_param_storage_shapes(self.params))
        return DataParallelBuffer(
            buffer_index=buffer_index,
            dtype=dtype,
            device=self.device,
            mesh=self.mesh,
            placements=self._buffer_placements(role),
        )

    def _init_buffers(self) -> None:
        """
        Initialize all buffers based on sharding strategy.

        Buffer creation logic:
        - model_weight_buffer: always created; replicated unless "optim_grads_params"
        - main_weight_buffer: created if mp_policy.main_params_dtype is specified
          AND it differs from the model-weight dtype or requires a different
          sharding layout; otherwise the optimizer mutates model_weight_buffer
        - main_grad_buffer: created if requires_grad
        """
        # Create model weight buffers. The policy owns dtype-sensitive storage
        # choices and exposes the tensor view that should be packed.
        model_weight_dtype = self.mp_policy.model_weight_buffer_dtype(self.params[0])
        wbuf = self._create_buffer(model_weight_dtype, "model_weight")
        wbuf.bind(torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device))
        wbuf.copy_tensors_(self.mp_policy.get_param_data(param) for param in self.params)
        self.model_weight_buffer = wbuf

        if self.mp_policy.needs_transpose_weight_buffer(self.params[0]):
            tbuf = self._create_buffer(torch.uint8, "transpose_weight")
            tbuf.bind(torch.empty(tbuf.data_size, dtype=tbuf.dtype, device=self.device))
            tbuf.copy_tensors_(
                self.mp_policy.get_param_data(param, transpose=True) for param in self.params
            )
            self.transpose_weight_buffer = tbuf

        # Create main weight buffer for mixed precision. Skip the redundant
        # copy when the optimizer dtype matches the model-weight dtype AND the
        # buffer placements are identical — in that case the optimizer mutates
        # ``model_weight_buffer`` directly via the dist_param views (which the
        # code below already binds to ``model_weight_buffer`` when
        # ``main_weight_buffer`` is None). Quantized params (FP8/NVFP4) always
        # need a separate main buffer because their model-weight dtype (uint8)
        # differs from the optimizer dtype (fp32), so the dtype guard below
        # already prevents skipping them.
        main_params_dtype = self.mp_policy.main_params_dtype_for_param(self.params[0])
        if main_params_dtype is not None:
            mbuf = self._create_buffer(main_params_dtype, "main_weight")
            if main_params_dtype != model_weight_dtype or mbuf.placements != wbuf.placements:
                mbuf.bind(torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device))
                mbuf.copy_tensors_(
                    self.mp_policy.get_high_precision_value(param).detach().to(main_params_dtype)
                    for param in self.params
                )
                self.main_weight_buffer = mbuf

        # Free the original full parameter tensors now that their data has been
        # copied into the weight buffers. The module holds DTensor shard views and
        # unshard() rebinds .data to the all-gathered buffer, so the original
        # storage is never accessed again.
        for p in self.params:
            # Pass the replacement buffers so the policy can tell whether this
            # parameter's original storage has been copied into FSDP-owned storage.
            for tensor in self.mp_policy.storage_tensors_to_free(
                p, self.model_weight_buffer, self.main_weight_buffer
            ):
                _free_storage(tensor)

        for role, weight_buffer in (
            ("model_weight", self.model_weight_buffer),
            ("transpose_weight", self.transpose_weight_buffer),
        ):
            if weight_buffer is not None and weight_buffer.placements[1] is not Placement.SHARD:
                self._bind_params(role, weight_buffer, weight_buffer.data)

        # Create gradient buffer.
        if self.requires_grad:
            main_grads_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
            self.main_grad_buffer = self._create_buffer(main_grads_dtype, "main_grad")

        # Create distributed parameter views.
        self._init_dist_params()

    def _required_weight_states(
        self, bwd_pass: bool = False
    ) -> List[tuple[WeightBufferRole, _WeightBufferState]]:
        """Return required weight states in stable collective order."""
        required_roles = self.mp_policy.weight_buffer_roles_for_unshard(
            self.params[0], bwd_pass=bwd_pass
        )
        missing_roles = required_roles.difference(self._weight_buffer_states)
        if missing_roles:
            raise RuntimeError(f"Required weight buffers are unavailable: {missing_roles}")
        return [
            (role, self._weight_buffer_states[role])
            for role in (WeightBufferRole.MODEL, WeightBufferRole.TRANSPOSE)
            if role in required_roles
        ]

    def _weight_buffers_for_unshard(
        self, bwd_pass: bool = False
    ) -> List[tuple[WeightBufferRole, DataParallelBuffer, DataParallelBuffer]]:
        """Return required weight buffers that do not have a full output."""
        self._ensure_buffers_on_gpu()
        return [
            (role, state.buffer, state.redistribution_source())
            for role, state in self._required_weight_states(bwd_pass)
            if state.compute_buffer(self._full_placements()) is None
        ]

    def _acquire_full_weight_buffer(self, role: WeightBufferRole) -> DataParallelBuffer:
        """Return the persistent full buffer or acquire full scratch."""
        state = self._weight_buffer_states[role]
        compute_buffer = state.compute_buffer(self._full_placements())
        if compute_buffer is not None:
            return compute_buffer
        if tuple(state.buffer.placements) == self._full_placements():
            return state.buffer
        output = state.buffer.placeholder(list(self._full_placements()))
        output.bind(
            self.allocator.allocate(
                key=(self.param_group_id, role.value),
                size=output.data_size,
                dtype=output.dtype,
                device=output.device,
            ).data
        )
        state.full_buffer = output
        return output

    def _release_full_weight_buffer(self, role: WeightBufferRole) -> None:
        """Release one allocator-backed full-weight scratch lease."""
        state = self._weight_buffer_states.get(role)
        if state is None or state.full_buffer is None:
            return
        state.full_buffer.unbind()
        self.allocator.free((self.param_group_id, role.value))
        state.full_buffer = None

    def _acquire_temporary_buffer(
        self, role: str, persistent: DataParallelBuffer, placements: list[Placement]
    ) -> DataParallelBuffer:
        """Return a persistent view or allocator-backed full-gradient buffer."""
        if role in self._temporary_buffers:
            return self._temporary_buffers[role]
        if persistent.placements == placements:
            return persistent
        try:
            return persistent.view(placements)
        except ValueError:
            output = persistent.placeholder(placements)
            output.bind(
                self.allocator.allocate(
                    key=(self.param_group_id, role),
                    size=output.data_size,
                    dtype=output.dtype,
                    device=output.device,
                ).data
            )
            self._temporary_buffers[role] = output
            return output

    def _release_temporary_buffers(self, *roles: str) -> None:
        """Release allocator-backed temporary buffers for the requested roles."""
        for role in roles:
            output = self._temporary_buffers.pop(role, None)
            if output is not None:
                output.unbind()
                self.allocator.free((self.param_group_id, role))

    def finalize_model_weight_unshard(self, bwd_pass: bool = False) -> None:
        """Finalize model weights after the caller has waited for async communication."""
        self.mp_policy.post_unshard(self.params, bwd_pass=bwd_pass)

    @staticmethod
    def unshard_model_weights(
        param_groups: List["ParameterGroup"],
        *,
        bwd_pass: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
        async_op: bool = False,
    ) -> None:
        """Unshard and bind model weights for a communication-compatible group sequence.

        Buffer roles, placement targets, and parameter binding remain private to
        ``ParameterGroup``. The caller supplies only lifecycle context and the
        communication stream so buffers from consecutive parameter groups can
        still share coalesced collectives.
        """
        if not param_groups:
            return
        owned_weight_buffers = [
            (param_group, role, weight_buffer, source)
            for param_group in param_groups
            for role, weight_buffer, source in param_group._weight_buffers_for_unshard(
                bwd_pass=bwd_pass
            )
        ]
        full_placements = list(param_groups[0]._full_placements())
        output_buffers = [
            param_group._acquire_full_weight_buffer(role)
            for param_group, role, _, _ in owned_weight_buffers
        ]
        full_buffers = DataParallelBuffer.redistribute_buffers(
            [source for _, _, _, source in owned_weight_buffers],
            full_placements,
            output_buffers=output_buffers,
            stream=stream,
            async_op=async_op,
        )
        for (param_group, role, weight_buffer, _), full_buffer in zip(
            owned_weight_buffers, full_buffers
        ):
            param_group._bind_params(role.value, weight_buffer, full_buffer.data)
            param_group._weight_buffer_states[role].valid_placements = tuple(
                weight_buffer.placements
            )

    def unshard(self, bwd_pass: bool = False, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Unshard model weights by all-gathering from sharded buffer.

        After unshard, parameters point to full unsharded storage. FP8
        parameters rebind their TE raw payload instead of ``param.data``.
        """
        self.unshard_model_weights([self], bwd_pass=bwd_pass, stream=stream)
        self.finalize_model_weight_unshard(bwd_pass=bwd_pass)

    def _bind_params(
        self, role: str, weight_buffer: DataParallelBuffer, buffer: torch.Tensor
    ) -> None:
        """Bind this group's parameters to a fully replicated weight buffer."""
        assert buffer.numel() == weight_buffer.buffer_index.bucket_meta.size, (
            f"Buffer size {buffer.numel()} does not match expected size "
            f"{weight_buffer.buffer_index.bucket_meta.size}"
        )
        for param in self.params:
            item_id = self.param_idx[param]
            start, end = weight_buffer.buffer_index._get_item_global_range(item_id)
            item_shape = weight_buffer.buffer_index.item_index_map[item_id].shape
            param_data = buffer[start:end].view(item_shape)
            self.mp_policy.bind_unsharded_param(param, param_data, role)

    @torch.no_grad()
    def _acquire_full_grad_buffer(self) -> DataParallelBuffer:
        """Return the full gradient destination, allocating a temporary lease if needed."""
        self._init_dist_grads()
        grad_buffer = self.main_grad_buffer
        if grad_buffer is None:
            raise RuntimeError("Parameter group has no gradient buffer")
        return self._acquire_temporary_buffer(
            "main_grad", grad_buffer, [Placement.REPLICATE, Placement.REPLICATE]
        )

    def get_main_grad(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Return the full gradient item used by backward accumulation."""
        full_grad_buffer = self._acquire_full_grad_buffer()
        item_id = self.param_idx[param]
        start, end = full_grad_buffer.buffer_index._get_item_global_range(item_id)
        param_shape = full_grad_buffer.buffer_index.item_index_map[item_id].shape
        return full_grad_buffer.data[start:end].view(param_shape)

    def ensure_full_grad_buffer(self) -> None:
        """Materialize the full gradient lease before CUDA graph trace or capture."""
        self._acquire_full_grad_buffer()

    @staticmethod
    def _placement_view(
        owner: DataParallelBuffer, placements: list[Placement]
    ) -> DataParallelBuffer:
        """Return a logical placement view backed by a containing buffer owner."""
        physical_placements = [
            Placement.REPLICATE if placement is Placement.PARTIAL else placement
            for placement in placements
        ]
        output = owner.view(physical_placements)
        return output if physical_placements == placements else output.reinterpret(placements)

    def _gradient_storage_view(self, placements: list[Placement]) -> DataParallelBuffer:
        """Return a logical gradient view backed by persistent gradient storage."""
        grad_buffer = self.main_grad_buffer
        if grad_buffer is None:
            raise RuntimeError("Parameter group has no gradient buffer")
        return self._placement_view(grad_buffer, placements)

    def _preprocess_gradient(
        self, full_grad_buffer: DataParallelBuffer, input_placements: list[Placement]
    ) -> tuple[DataParallelBuffer, tuple | None]:
        """Convert and scale the full gradient once before redistribution."""
        comm_dtype = self.grad_comm_dtype or full_grad_buffer.dtype
        workspace_key = None
        communication_owner = full_grad_buffer
        if comm_dtype != full_grad_buffer.dtype:
            workspace_key = (self.param_group_id, "main_grad", "grad_comm")
            communication_owner = DataParallelBuffer(
                full_grad_buffer.buffer_index,
                comm_dtype,
                full_grad_buffer.device,
                full_grad_buffer.mesh,
                full_grad_buffer.placements,
            )
            communication_owner.bind(
                self.allocator.allocate(
                    key=workspace_key,
                    size=communication_owner.data_size,
                    dtype=communication_owner.dtype,
                    device=communication_owner.device,
                ).data
            )

        needs_scaling = self.gradient_scaling_factor not in (None, 1.0)
        if communication_owner is not full_grad_buffer or needs_scaling:
            if communication_owner is not full_grad_buffer:
                communication_owner.data.copy_(full_grad_buffer.data)
            if needs_scaling:
                communication_owner.data.mul_(self.gradient_scaling_factor)

        return self._placement_view(communication_owner, input_placements), workspace_key

    def _gradient_redistribution_output(
        self,
        input_buffer: DataParallelBuffer,
        persistent_buffer: DataParallelBuffer,
        has_accumulated_grad: bool,
    ) -> DataParallelBuffer:
        """Return a direct persistent output or a temporary communication output."""
        if not has_accumulated_grad and input_buffer.dtype == persistent_buffer.dtype:
            return persistent_buffer
        storage_owner = input_buffer._storage_owner or input_buffer
        return self._placement_view(storage_owner, persistent_buffer.placements)

    @staticmethod
    def _commit_gradient_stage(
        persistent_buffer: DataParallelBuffer,
        output_buffer: DataParallelBuffer,
        *,
        has_accumulated_grad: bool,
        continue_redistribution: bool,
    ) -> DataParallelBuffer:
        """Assign or accumulate one stage and return the next redistribution input."""
        if persistent_buffer.data.data_ptr() == output_buffer.data.data_ptr():
            if has_accumulated_grad:
                raise RuntimeError("Cannot accumulate a gradient buffer into itself")
            return persistent_buffer
        if continue_redistribution:
            if has_accumulated_grad:
                output_buffer.data.add_(persistent_buffer.data)
            return output_buffer
        if has_accumulated_grad:
            persistent_buffer.data.add_(output_buffer.data)
        else:
            persistent_buffer.data.copy_(output_buffer.data)
        return persistent_buffer

    def model_weights_are_unsharded(self, bwd_pass: bool = False) -> bool:
        """Return whether the model weights required by this pass are unsharded."""
        return all(
            state.compute_buffer(self._full_placements()) is not None
            for _, state in self._required_weight_states(bwd_pass)
        )

    def weights_are_unsharded(self, bwd_pass: bool = False) -> bool:
        """Return whether the weights required by this pass are unsharded."""
        return self.model_weights_are_unsharded(bwd_pass=bwd_pass)

    def reshard(self):
        """Detach parameter views and release temporary replicated weight leases."""
        self.mp_policy.post_reshard(self.params)
        self._release_full_weight_buffer(WeightBufferRole.MODEL)
        self._release_full_weight_buffer(WeightBufferRole.TRANSPOSE)

    def reshard_weight(self) -> None:
        """Release temporary replicated weight leases."""
        self.reshard()

    @torch.no_grad()
    def copy_main_weights_to_model_weights(self):
        """Install optimized main weights into model compute weights."""
        self._ensure_buffers_on_gpu()
        optimizer_placements = self._optimizer_placements()
        self.mp_policy.copy_main_weights_to_model_weights(
            self.params,
            self.param_idx,
            self.mesh,
            self.model_weight_buffer,
            self.main_weight_buffer,
            self.transpose_weight_buffer,
            optimizer_placements=optimizer_placements,
        )
        for role, buffer in (
            (WeightBufferRole.MODEL, self.model_weight_buffer),
            (WeightBufferRole.TRANSPOSE, self.transpose_weight_buffer),
        ):
            if buffer is None:
                continue
            self._release_full_weight_buffer(role)
            state = self._weight_buffer_states[role]
            state.valid_placements = tuple(optimizer_placements)

    def refresh_model_weight(self) -> None:
        """Install optimizer weights into model-weight storage."""
        self.copy_main_weights_to_model_weights()

    def reduce_grad(
        self, is_last_backward: bool = False, stream: Optional[torch.cuda.Stream] = None
    ):
        """
        Reduce gradients across DP ranks.

        ZeRO-2/3 reduce-scatter sharded grad buffers during backward.
        ZeRO-1 keeps grads replicated during backward and reduce-scatters
        the replicated buffer once when the optimizer syncs.
        """
        caller_stream = torch.cuda.current_stream()
        stream = stream or caller_stream
        if stream != caller_stream:
            stream.wait_stream(caller_stream)

        with torch.cuda.stream(stream):
            self._ensure_buffers_on_gpu()
            if self.main_grad_buffer is None:
                return

            # FSDPModule has staged this microbatch into the full (0, 0) gradient
            # buffer before calling here. For replicated gradient storage, that
            # buffer accumulates microbatches until the step-boundary collective.
            # For sharded gradient storage, it is fresh reduce-scatter input and is
            # consumed below on every microbatch.
            self._full_grad_has_value = True

            full_grad_buffer = self._acquire_full_grad_buffer()
            partial_placements = full_grad_buffer.placements.copy()
            partial_placements[1] = Placement.PARTIAL
            if self.mesh.size(0) > 1:
                partial_placements[0] = Placement.PARTIAL

            grad_buffer = self.main_grad_buffer
            reduce_fsdp_grad = is_last_backward or grad_buffer.placements[1] is Placement.SHARD
            if not reduce_fsdp_grad:
                return

            grad_input_buffer, workspace_key = self._preprocess_gradient(
                full_grad_buffer, partial_placements
            )
            try:
                optimizer_placements = self._optimizer_placements()
                fsdp_placements = partial_placements.copy()
                fsdp_placements[1] = optimizer_placements[1]
                fsdp_out = self._gradient_storage_view(fsdp_placements)
                fsdp_has_accumulated_grad = self._reduced_grad_has_value
                output_buffer = self._gradient_redistribution_output(
                    grad_input_buffer, fsdp_out, fsdp_has_accumulated_grad
                )
                grad_input_buffer.redistribute(fsdp_placements, output_buffer=output_buffer)

                reduce_hsdp_grad = self.mesh.size(0) > 1 and is_last_backward
                fsdp_out = self._commit_gradient_stage(
                    fsdp_out,
                    output_buffer,
                    has_accumulated_grad=fsdp_has_accumulated_grad,
                    continue_redistribution=reduce_hsdp_grad,
                )
                if fsdp_placements[1] is Placement.SHARD:
                    self._full_grad_has_value = False

                if reduce_hsdp_grad:
                    hsdp_out = self._gradient_storage_view(optimizer_placements)
                    output_buffer = self._gradient_redistribution_output(
                        fsdp_out, hsdp_out, has_accumulated_grad=False
                    )
                    fsdp_out.redistribute(optimizer_placements, output_buffer=output_buffer)
                    self._commit_gradient_stage(
                        hsdp_out,
                        output_buffer,
                        has_accumulated_grad=False,
                        continue_redistribution=False,
                    )
                self._reduced_grad_has_value = True
            finally:
                if workspace_key is not None:
                    self.allocator.free(workspace_key)

    def release_grad_buffer(self):
        """Release this group's temporary full-gradient lease."""
        if self.main_grad_buffer is not None:
            # Drop weight.main_grad views that layers.py stores during gradient-accumulation-fusion
            # backward. Those views keep the full-gradient lease alive after its
            # group reference is cleared, causing it to leak until the next backward.
            for param in self.params:
                if hasattr(param, 'main_grad'):
                    del param.main_grad
            self._release_temporary_buffers("main_grad")

    def _release_grad_storage_if_unused(self) -> None:
        """Drop ``main_grad_buffer.data`` if it has no live gradients.

        After ``zero_grad()`` (or before the first backward), all
        ``dist_param.grad`` are ``None``, so the gradient buffer holds no
        meaningful data.  Free the backing tensor — ``_init_dist_grads``
        will re-allocate on the next ``reduce_grad``.
        """
        if self.enable_full_iteration_cuda_graph:
            return
        if self.main_grad_buffer is None or self.main_grad_buffer.data is None:
            return
        # Gradient storage may contain either unreduced microbatch accumulation
        # or a collective output even when this rank owns no optimizer-facing
        # parameter shard. Keep both alive until zero_grad() clears their state.
        if self._full_grad_has_value or self._reduced_grad_has_value:
            return
        if any(
            [getattr(p, "grad", None) is not None for p in self.dist_params]
            + [getattr(p, "decoupled_grad", None) is not None for p in self.dist_params]
        ):
            return
        # Cache DTensor wrappers and their global metadata while dropping the
        # local views that retain gradient-buffer storage. dist_grads itself
        # represents only live optimizer-facing gradients, so detached shells
        # remain private until _init_dist_grads rebinds them.
        for index, dist_grad in enumerate(self.dist_grads):
            if dist_grad is not None:
                detach_uneven_dtensor_local_tensor(dist_grad)
                self._dist_grad_cache[index] = dist_grad
                self.dist_grads[index] = None
        self.main_grad_buffer.unbind()

    def release_grad_storage_if_unused(self) -> None:
        """Release stale gradient storage after optimizer gradients are cleared."""
        self._release_grad_storage_if_unused()

    def _init_dist_params(self):
        """Initialize optimizer-facing DTensor views into the weight buffers."""
        self.dist_params = []
        self.dist_grads = []  # placeholder, populated in _init_dist_grads
        optimizer_buffer = self.main_weight_buffer or self.model_weight_buffer
        buffer_placements = self._optimizer_placements()
        optimizer_view = optimizer_buffer.view(buffer_placements)
        optimizer_dtensor_placements = [
            Shard(dim=0) if placement is Placement.SHARD else Replicate()
            for placement in buffer_placements
        ]
        if buffer_placements[0] is Placement.SHARD:
            setattr(self.mesh, "_shard_order", [1, 0])

        for param in self.params:
            item_id = self.param_idx[param]
            data = optimizer_view.tensor_view(item_id)
            param_shape = (
                param.shape
                if self.main_weight_buffer is not None
                else self.mp_policy.get_param_storage_shapes([param])[0]
            )

            dist_data = make_uneven_dtensor(
                data, param_shape, self.mesh, optimizer_dtensor_placements, post_process_uneven=True
            )
            dist_param = torch.nn.Parameter(dist_data, requires_grad=param.requires_grad)
            # ``torch.nn.Parameter(DTensor)`` wraps the DTensor and creates a
            # fresh local tensor object, so Python-side uneven-DTensor metadata
            # attached by ``post_process_uneven=True`` is not preserved
            # automatically. Grad DTensor initialization later copies chunk
            # metadata from ``dist_param``; keep that invariant explicit here.
            copy_chunk_metadata(dist_data, dist_param)

            # Mark as FSDP parameter for special handling.
            setattr(param, "__fsdp_param__", True)
            setattr(dist_param, "__fsdp_param__", True)
            assert hasattr(
                dist_param._local_tensor, "__create_chunk_list__"
            ), "DTensor must have chunk metadata for FSDP"
            self.dist_params.append(dist_param)
            self.dist_grads.append(None)  # placeholder, will be set in _init_dist_grads

    def _init_dist_grads(self) -> None:
        """Lazily allocate ``main_grad_buffer.data`` and rebuild ``dist_grads``.

        The buffer layout (``BufferIndex``, offsets, shard) was created in
        ``_init_buffers``; only the backing tensor is deferred. Called from
        ``reduce_grad()`` on first use. Uses ``torch.empty`` to avoid the
        zero-init cost. ``_reduced_grad_has_value`` is ``False`` after allocation,
        so the first reduce-scatter overwrites rather than accumulates; the
        uninitialized data is never read. Subsequent calls are no-ops.
        """
        gbuf = self.main_grad_buffer
        if gbuf is None or not self.requires_grad:
            return
        if gbuf.data is not None:
            return  # already initialised

        gbuf.bind(torch.empty(gbuf.data_size, dtype=gbuf.dtype, device=self.device))

        buffer_placements = [
            Placement.SHARD if isinstance(placement, Shard) else Placement.REPLICATE
            for placement in self.dist_params[0].placements
        ]
        grad_view = gbuf.view(buffer_placements)

        for index, (p, dist_param, dist_grad) in enumerate(
            zip(self.params, self.dist_params, self._dist_grad_cache)
        ):
            item_id = self.param_idx[p]
            grad_dtensor_placements = dist_param.placements
            grad_data = grad_view.tensor_view(item_id)
            # Empty local shards are optimizer no-ops. Keeping them as None also
            # avoids fused multi-tensor optimizer failures on neighboring shards.
            if not p.requires_grad or grad_data.numel() == 0:
                self.dist_grads[index] = None
                continue
            if dist_grad is None:
                dist_grad = make_uneven_dtensor(
                    grad_data,
                    p.shape,
                    self.mesh,
                    grad_dtensor_placements,
                    copy_chunk_meta_from=dist_param,
                )
                self._dist_grad_cache[index] = dist_grad
            else:
                rebind_uneven_dtensor_local_tensor(
                    dist_grad,
                    grad_data,
                    p.shape,
                    copy_chunk_meta_from=dist_param,
                    validate=not self._dist_grad_cache_validated[index],
                )
                self._dist_grad_cache_validated[index] = True
            self.dist_grads[index] = dist_grad

    def _rebuild_dist_views(self) -> None:
        """Update ``dist_params`` and ``dist_grads`` after storage moves device.

        Moving a buffer between CPU and GPU replaces its backing tensor. Rebuild
        the local optimizer-facing DTensor views.
        """
        optimizer_buffer = self.main_weight_buffer or self.model_weight_buffer
        buffer_placements = [
            Placement.SHARD if isinstance(placement, Shard) else Placement.REPLICATE
            for placement in self.dist_params[0].placements
        ]
        optimizer_view = optimizer_buffer.view(buffer_placements)
        for param, dist_param in zip(self.params, self.dist_params):
            data = optimizer_view.tensor_view(self.param_idx[param])
            object.__setattr__(dist_param._local_tensor, 'data', data)

        if self.main_grad_buffer is not None and self.main_grad_buffer.data is not None:
            grad_view = self.main_grad_buffer.view(buffer_placements)
            for param, dist_grad in zip(self.params, self.dist_grads):
                if dist_grad is None:
                    continue
                grad_data = grad_view.tensor_view(self.param_idx[param])
                object.__setattr__(dist_grad._local_tensor, 'data', grad_data)

    def _ensure_buffers_on_gpu(self) -> None:
        """Auto-reload persistent buffers to GPU and rebuild invalidated views."""
        moved = [self._move_buffer_storage_to(buffer, self.device) for buffer in self._buffers()]
        if any(moved):
            self._rebuild_dist_views()

    def zero_grad(self, set_to_none: bool = True):
        """Zero the main gradient buffer and mark grads as zeroed."""
        self._full_grad_has_value = False
        self._reduced_grad_has_value = False
        if self.enable_full_iteration_cuda_graph:
            if self.main_grad_buffer is not None:
                if self.main_grad_buffer.data is not None:
                    self.main_grad_buffer.data.zero_()
            for dist_param in self.dist_params:
                grad = getattr(dist_param, "grad", None)
                if grad is not None:
                    _zero_tensor_storage(grad)
                    setattr(dist_param, "_mfsdp_keep_grad_for_cuda_graph", True)
                decoupled_grad = getattr(dist_param, "decoupled_grad", None)
                if decoupled_grad is not None:
                    _zero_tensor_storage(decoupled_grad)
                    setattr(dist_param, "_mfsdp_keep_grad_for_cuda_graph", True)
            return

        if set_to_none:
            for dist_param in self.dist_params:
                if dist_param.grad is not None:
                    dist_param.grad = None
                if hasattr(dist_param, "decoupled_grad"):
                    dist_param.decoupled_grad = None
            self._release_grad_storage_if_unused()
        else:
            if self.main_grad_buffer is not None and self.main_grad_buffer.data is not None:
                self.main_grad_buffer.data.zero_()
