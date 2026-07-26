# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Placement-first data-parallel ParameterGroup for Megatron FSDP v2.

The implementation provides the target DDP, ZeRO, FSDP, and HSDP ownership
model without inheriting the lifecycle structure of ``param_group.py``. It is
available through the experimental eager ``fully_shard`` integration.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ParameterGroupLayoutV2:
    """Persistent data-parallel placements used by :class:`ParameterGroupV2`."""

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
    ) -> "ParameterGroupLayoutV2":
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
    def fsdp(cls) -> "ParameterGroupLayoutV2":
        """Build a one-dimensional fully sharded layout."""
        return cls.from_strategies("optim_grads_params")

    @classmethod
    def hsdp(cls, *, shard_optimizer_across_outer_dp: bool) -> "ParameterGroupLayoutV2":
        """Build the two-dimensional HSDP layout discussed in the design document."""
        return cls.from_strategies(
            "optim_grads_params",
            outer_dp_sharding_strategy=("optim" if shard_optimizer_across_outer_dp else "no_shard"),
        )


class GradientPhaseV2(enum.Enum):
    """Lifecycle phase of the value stored in the persistent gradient buffer."""

    EMPTY = enum.auto()
    ACCUMULATING = enum.auto()
    READY = enum.auto()


@dataclass
class ParameterGroupStateV2:
    """Minimal value-validity and scratch-lease state."""

    weight_valid: Placements
    grad_phase: GradientPhaseV2 = GradientPhaseV2.EMPTY
    full_weight: DataParallelBuffer | None = None
    full_grad: DataParallelBuffer | None = None


class ParameterGroupV2:
    """Own persistent data-parallel values and their placement transitions.

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
        if mesh.ndim not in (1, 2):
            raise ValueError(
                f"ParameterGroupV2 expects a 1D DP or 2D hybrid-DP mesh, got {mesh.ndim}D"
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
        self.state = ParameterGroupStateV2(weight_valid=tuple(self.weight_buffer.placements))
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

    @torch.no_grad()
    def unshard_weight(self, stream: torch.cuda.Stream | None = None) -> DataParallelBuffer:
        """Restore persistent weight validity, materialize full weights, and bind params."""
        caller_stream = torch.cuda.current_stream()
        stream = stream or caller_stream
        if stream != caller_stream:
            stream.wait_stream(caller_stream)

        with torch.cuda.stream(stream):
            compute_weight = self.compute_weight()
            if compute_weight is not None:
                self._bind_weight(compute_weight)
                self.mp_policy.post_unshard(self.params)
                return compute_weight

            current = self.weight_buffer.view(list(self.state.weight_valid))
            persistent_placements = tuple(self.weight_buffer.placements)
            if self.state.weight_valid != persistent_placements:
                # For outer-optimizer-sharded HSDP this is [S,S] -> [R,S]:
                # restore the outer replica before unsharding the inner DP axis.
                current.redistribute(list(persistent_placements), output_buffer=self.weight_buffer)
                self.state.weight_valid = persistent_placements
                current = self.weight_buffer

            if persistent_placements != self.full_placements:
                # The remaining HSDP transition is [R,S] -> [R,R].
                self.state.full_weight = self._allocate_scratch(
                    "full_weight", self.weight_buffer, self.full_placements
                )
                current.redistribute(
                    list(self.full_placements), output_buffer=self.state.full_weight
                )
                current = self.state.full_weight

            self._bind_weight(current)
            self.mp_policy.post_unshard(self.params)
            return current

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

    def release_grad_storage_if_unused(self) -> None:
        """Release gradient storage after optimizer-facing gradients are cleared."""
        if self.enable_full_iteration_cuda_graph:
            return
        if self.state.grad_phase is GradientPhaseV2.ACCUMULATING:
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

    def _finalize_gradient_placement(self, current: DataParallelBuffer) -> DataParallelBuffer:
        """Redistribute delayed mesh axes into the persistent optimizer gradient."""
        target = self.layout.main_weight
        final = self._placement_view(self.grad_buffer, target)
        communication_owner = current._storage_owner or current
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
            current.redistribute(next_placements, output_buffer=output)
            current = output

        if current.data.data_ptr() != final.data.data_ptr():
            final.data.copy_(current.data)
        return final

    def _reduce_grad(self, *, is_last_backward: bool) -> None:
        """Run one gradient reduction on the current CUDA stream."""
        if self.state.full_grad is None:
            raise RuntimeError("begin_backward() must run before reduce_grad()")
        if self.state.grad_phase is GradientPhaseV2.READY:
            raise RuntimeError("zero_grad() must run before starting another gradient")

        grad_input, workspace_key = self._preprocess_gradient(self.state.full_grad)
        accumulation = self._placement_view(self.grad_buffer, self.layout.grad_accumulation)
        # A pending accumulation contains reduced gradients from earlier
        # microbatches but is not yet optimizer-ready. Placement alone cannot
        # express this: ZeRO-2/FSDP use the same placement for both phases.
        has_accumulation = self.state.grad_phase is GradientPhaseV2.ACCUMULATING
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

        try:
            grad_input.redistribute(list(self.layout.grad_accumulation), output_buffer=output)
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
                    self._finalize_gradient_placement(output)
                self.state.grad_phase = GradientPhaseV2.READY
                self._install_optimizer_grads()
            else:
                self.state.grad_phase = GradientPhaseV2.ACCUMULATING
        finally:
            if workspace_key is not None:
                self.allocator.free(workspace_key)

    @torch.no_grad()
    def reduce_grad(
        self, *, is_last_backward: bool, stream: torch.cuda.Stream | None = None
    ) -> None:
        """Reduce one microbatch and finalize delayed DP axes on the last backward.

        A caller-supplied stream enables overlap. In that case the caller owns
        completion tracking and releases the full-gradient lease after waiting
        for its recorded event.
        """
        caller_stream = torch.cuda.current_stream()
        stream = stream or caller_stream
        if stream != caller_stream:
            stream.wait_stream(caller_stream)
        try:
            with torch.cuda.stream(stream):
                self._reduce_grad(is_last_backward=is_last_backward)
        except Exception:
            self.release_grad_buffer()
            raise
        if stream == caller_stream:
            self.release_grad_buffer()

    def optimizer_weight(self) -> DataParallelBuffer:
        """Return the persistent optimizer-weight representation."""
        return self.main_weight_buffer

    def optimizer_grad(self) -> DataParallelBuffer:
        """Return the optimizer gradient after final data-parallel reduction."""
        if self.state.grad_phase is not GradientPhaseV2.READY:
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
        self.state.grad_phase = GradientPhaseV2.EMPTY
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
