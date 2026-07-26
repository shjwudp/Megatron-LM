# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Parameter Group for FSDP

Groups parameters that share the same (device, dtype, requires_grad) and
manages their buffers collectively. This enables efficient memory management
and collective operations across parameters.
"""

import math
from typing import Dict, List, Optional

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate, Shard

from ..uneven_dtensor import (
    copy_chunk_metadata,
    detach_uneven_dtensor_local_tensor,
    make_uneven_dtensor,
    rebind_uneven_dtensor_local_tensor,
)
from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .dp_buffer import DataParallelBuffer, Placement
from .mixed_precision import MixedPrecisionPolicy
from .utils import ParamGroupIdx, _prepare_fsdp_mesh


def _zero_tensor_storage(tensor: torch.Tensor) -> None:
    """Zero a Tensor or DTensor by writing only its local storage."""
    local_tensor = getattr(tensor, "_local_tensor", None)
    target = local_tensor if local_tensor is not None else tensor
    with torch.no_grad():
        target.zero_()


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

        # Setup device mesh and derived process group
        if mesh is None:
            world_ranks = torch.arange(
                torch.distributed.get_world_size(torch.distributed.group.WORLD)
            ).reshape(1, -1)
            mesh = DeviceMesh(self.device.type, world_ranks, mesh_dim_names=("dp_outer", "dp"))
        mesh = _prepare_fsdp_mesh(mesh)
        self.mesh = mesh
        self.outer_dp_group = self.mesh.get_group(mesh_dim=0)
        self.dp_group = self.mesh.get_group(mesh_dim=1)
        self._dp_rank = torch.distributed.get_rank(self.dp_group)
        self._dp_world_size = torch.distributed.get_world_size(self.dp_group)

        if sharding_strategy not in ("no_shard", "optim", "optim_grads", "optim_grads_params"):
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
        self.sharding_strategy = sharding_strategy
        self.outer_dp_sharding_strategy = outer_dp_sharding_strategy
        self.param_group_id = param_group_id

        # Compute chunk size factor for alignment
        # LCM ensures params align to common boundary for efficient sharding
        if len(params) > 0 and any(p.shape[1:].numel() > 0 for p in params):
            self.chunk_size_factor = max(1, math.lcm(*[p.shape[1:].numel() for p in params]))
        else:
            self.chunk_size_factor = 1

        self.gradient_scaling_factor = gradient_scaling_factor
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.enable_full_iteration_cuda_graph = False
        self._full_grad_buffer_has_accumulated_grad = False
        self._reduced_grad_buffer_has_accumulated_grad = False

        # Buffer references (initialized in _init_buffers)
        self.model_weight_buffer: Optional[DataParallelBuffer] = None
        self.transpose_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_grad_buffer: Optional[DataParallelBuffer] = None
        # Initialize buffers and distributed parameters
        self._init_buffers()
        # DTensor shells cached across set_to_none gradient-buffer releases.
        # Cached entries are detached from local storage and never exposed
        # through dist_grads until _init_dist_grads rebinds them.
        self._dist_grad_cache = list(self.dist_grads)
        self._dist_grad_cache_validated = [False for _ in self.dist_grads]

    def set_allocator(self, allocator: BucketAllocator) -> None:
        """Replace the allocator used by every buffer in this parameter group."""
        self.allocator = allocator
        for buffer in self._buffers():
            buffer.allocator = allocator

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
            if buffer.data is not None and not buffer._is_on_cpu()
        ]
        entries.sort(key=lambda entry: entry[1], reverse=True)

        offloaded_bytes = 0
        skipped_bytes = 0
        for buffer, nbytes in entries:
            if max_cpu_bytes is not None and offloaded_bytes + nbytes > max_cpu_bytes:
                skipped_bytes += nbytes
                continue
            buffer._move_data_to(torch.device("cpu"), pin_memory=pin_memory)
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
                buffer._move_data_to(device)
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
                        buffer.outer_sharded,
                        buffer.inner_sharded,
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
        for param in self.params:
            param_data = self.model_weight_buffer.get_item(
                self.param_idx[param], placements=[Placement.REPLICATE, Placement.REPLICATE]
            )
            assert not torch.isnan(param_data).any(), "NaN detected in model weight buffer"

    def _create_buffer(self, dtype: torch.dtype, role: str) -> DataParallelBuffer:
        """Create a buffer and namespace its temporary bucket by role."""
        return DataParallelBuffer(
            tensors=self.params,
            dtype=dtype,
            device=self.device,
            mesh=self.mesh,
            allocator=self.allocator,
            buffer_role=role,
            param_group_id=self.param_group_id,
            gradient_scaling_factor=self.gradient_scaling_factor,
            chunk_size_factor=self.chunk_size_factor,
            sharding_strategy=self.sharding_strategy,
            outer_dp_sharding_strategy=self.outer_dp_sharding_strategy,
            mp_policy=self.mp_policy,
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
        wbuf.init_data(torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device))
        for i, p in enumerate(self.params):
            wbuf.set_item(i, self.mp_policy.get_param_data(p))
        self.model_weight_buffer = wbuf

        if self.mp_policy.needs_transpose_weight_buffer(self.params[0]):
            tbuf = self._create_buffer(torch.uint8, "transpose_weight")
            tbuf.init_data(torch.empty(tbuf.data_size, dtype=tbuf.dtype, device=self.device))
            for i, p in enumerate(self.params):
                tbuf.set_item(i, self.mp_policy.get_param_data(p, transpose=True))
            self.transpose_weight_buffer = tbuf

        # Create main weight buffer for mixed precision. Skip the redundant
        # copy when the optimizer dtype matches the model-weight dtype AND the
        # storage placements are identical — in that case the optimizer mutates
        # ``model_weight_buffer`` directly via the dist_param views (which the
        # code below already binds to ``model_weight_buffer`` when
        # ``main_weight_buffer`` is None). Quantized params (FP8/NVFP4) always
        # need a separate main buffer because their model-weight dtype (uint8)
        # differs from the optimizer dtype (fp32), so the dtype guard below
        # already prevents skipping them.
        main_params_dtype = self.mp_policy.main_params_dtype_for_param(self.params[0])
        if main_params_dtype is not None:
            mbuf = self._create_buffer(main_params_dtype, "main_weight")
            if (
                main_params_dtype != model_weight_dtype
                or mbuf.storage_placements != wbuf.storage_placements
            ):
                mbuf.init_data(torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device))
                for i, p in enumerate(self.params):
                    item = self.mp_policy.get_high_precision_value(p)
                    mbuf.set_item(i, item.detach().to(main_params_dtype))
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

        for weight_buffer in (self.model_weight_buffer, self.transpose_weight_buffer):
            if weight_buffer is not None and not weight_buffer.inner_sharded:
                self._bind_params(weight_buffer, weight_buffer.data)

        # Create gradient buffer
        if self.requires_grad:
            main_grads_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
            gbuf = self._create_buffer(main_grads_dtype, "main_grad")
            self.main_grad_buffer = gbuf

        # Create distributed parameter views
        self._init_dist_params()

    def _weight_buffers_for_unshard(self, bwd_pass: bool = False) -> List[DataParallelBuffer]:
        """Return this group's internal weight buffers required by one compute pass."""
        self._ensure_buffers_on_gpu()
        return [
            weight_buffer
            for weight_buffer in self.mp_policy.weight_buffers_for_unshard(
                self.model_weight_buffer, self.transpose_weight_buffer, bwd_pass=bwd_pass
            )
            if weight_buffer is not None
        ]

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
        owned_weight_buffers = [
            (param_group, weight_buffer)
            for param_group in param_groups
            for weight_buffer in param_group._weight_buffers_for_unshard(bwd_pass=bwd_pass)
        ]
        full_buffers = DataParallelBuffer.redistribute_buffers(
            [weight_buffer for _, weight_buffer in owned_weight_buffers],
            [Placement.REPLICATE, Placement.REPLICATE],
            stream=stream,
            async_op=async_op,
        )
        for (param_group, weight_buffer), full_buffer in zip(owned_weight_buffers, full_buffers):
            param_group._bind_params(weight_buffer, full_buffer)

    def unshard(self, bwd_pass: bool = False, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Unshard model weights by all-gathering from sharded buffer.

        After unshard, parameters point to full unsharded storage. FP8
        parameters rebind their TE raw payload instead of ``param.data``.
        """
        self.unshard_model_weights([self], bwd_pass=bwd_pass, stream=stream)
        self.finalize_model_weight_unshard(bwd_pass=bwd_pass)

    def _bind_params(
        self, weight_buffer: DataParallelBuffer, buffer: Optional[torch.Tensor] = None
    ) -> None:
        """Bind this group's parameters to a fully replicated weight buffer."""
        if weight_buffer is self.model_weight_buffer:
            buffer_role = "model_weight"
        elif weight_buffer is self.transpose_weight_buffer:
            buffer_role = "transpose_weight"
        else:
            raise ValueError("Parameters may only be bound to this group's weight buffers")
        if buffer is None:
            assert weight_buffer.is_unsharded(), "Cannot bind params from a sharded buffer"
            buffer = weight_buffer.fetch_buffer(weight_buffer.placements)
        assert buffer.numel() == weight_buffer.buffer_index.bucket_meta.size, (
            f"Buffer size {buffer.numel()} does not match expected size "
            f"{weight_buffer.buffer_index.bucket_meta.size}"
        )
        for param in self.params:
            item_id = self.param_idx[param]
            start, end = weight_buffer.buffer_index._get_item_global_range(item_id)
            item_shape = weight_buffer.buffer_index.item_index_map[item_id].shape
            param_data = buffer[start:end].view(item_shape)
            self.mp_policy.bind_unsharded_param(param, param_data, buffer_role)

    @torch.no_grad()
    def commit_comm_output(
        self,
        grad_buffer: DataParallelBuffer,
        comm_output: torch.Tensor,
        changed_axis: int,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        accumulate: bool = False,
    ) -> None:
        """Commit a gradient redistribution result into this group's storage."""
        if grad_buffer is not self.main_grad_buffer:
            raise ValueError("Communication output may only target this group's grad buffer")
        output_buffer = grad_buffer.fetch_buffer(grad_buffer.placements)
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            if output_buffer.data_ptr() != comm_output.data_ptr():
                if accumulate:
                    output_buffer.add_(comm_output)
                else:
                    output_buffer.copy_(comm_output)
        grad_buffer.release_redistribution_workspace(changed_axis)

    def model_weights_are_unsharded(self, bwd_pass: bool = False) -> bool:
        """Return whether the model weights required by this pass are unsharded."""
        for weight_buffer in self.mp_policy.weight_buffers_for_unshard(
            self.model_weight_buffer, self.transpose_weight_buffer, bwd_pass=bwd_pass
        ):
            if weight_buffer is None:
                continue
            if not weight_buffer.is_unsharded():
                return False
        return True

    def reshard(self):
        """Reshard model weights by releasing unsharded buffer."""
        self.model_weight_buffer.redistribute(self.model_weight_buffer.storage_placements)
        if self.transpose_weight_buffer is not None:
            self.transpose_weight_buffer.redistribute(
                self.transpose_weight_buffer.storage_placements
            )
        self.mp_policy.post_reshard(self.params)

    @torch.no_grad()
    def copy_main_weights_to_model_weights(self):
        """Install optimized main weights into model compute weights."""
        self._ensure_buffers_on_gpu()
        if self.main_weight_buffer is not None and self.mp_policy.is_nvfp4_param(self.params[0]):
            full_weight_buffer = self.model_weight_buffer.fetch_buffer(
                [Placement.REPLICATE, Placement.REPLICATE]
            )
            self._bind_params(self.model_weight_buffer, full_weight_buffer)
        self.mp_policy.copy_main_weights_to_model_weights(
            self.params,
            self.param_idx,
            self.mesh,
            self.model_weight_buffer,
            self.main_weight_buffer,
            self.transpose_weight_buffer,
        )

    def reduce_grad(
        self, is_last_backward: bool = False, stream: Optional[torch.cuda.Stream] = None
    ):
        """
        Reduce gradients across DP ranks.

        ZeRO-2/3 reduce-scatter sharded grad buffers during backward.
        ZeRO-1 keeps grads replicated during backward and reduce-scatters
        the replicated buffer once when the optimizer syncs.
        """
        self._ensure_buffers_on_gpu()
        if self.main_grad_buffer is None:
            return

        # FSDPModule has staged this microbatch into the full (0, 0) gradient
        # buffer before calling here. For replicated gradient storage, that
        # buffer accumulates microbatches until the step-boundary collective.
        # For sharded gradient storage, it is fresh reduce-scatter input and is
        # consumed below on every microbatch.
        self._full_grad_buffer_has_accumulated_grad = True

        grad_buffer = self.main_grad_buffer
        partial_placements = grad_buffer.placements.copy()
        partial_placements[1] = Placement.PARTIAL
        if self.mesh.size(0) > 1:
            partial_placements[0] = Placement.PARTIAL
        DataParallelBuffer.redistribute_buffers([grad_buffer], partial_placements)

        storage = grad_buffer.storage_placements
        if is_last_backward or grad_buffer.inner_sharded:
            inner_target = Placement.SHARD if self.sharding_strategy == "optim" else storage[1]
            comm_output = grad_buffer.redistribute(
                [grad_buffer.placements[0], inner_target], stream=stream
            )
            accumulate = self._reduced_grad_buffer_has_accumulated_grad
            self.commit_comm_output(
                grad_buffer, comm_output, 1, stream=stream, accumulate=accumulate
            )
            self._reduced_grad_buffer_has_accumulated_grad = True
            if inner_target is not Placement.REPLICATE:
                self._full_grad_buffer_has_accumulated_grad = False

        if is_last_backward and self.mesh.size(0) > 1:
            outer_target = (
                Placement.SHARD if self.outer_dp_sharding_strategy == "optim" else storage[0]
            )
            comm_output = grad_buffer.redistribute(
                [outer_target, grad_buffer.placements[1]], stream=stream
            )
            self.commit_comm_output(grad_buffer, comm_output, 0, stream=stream)

    def release_grad_buffer(self):
        """Release the main gradient buffer to free memory."""
        if self.main_grad_buffer is not None:
            # Drop weight.main_grad views that layers.py stores during gradient-accumulation-fusion
            # backward. Those views keep _unsharded_buffer alive after its internal reference is
            # cleared, causing the grad buffer to leak until the next backward.
            for param in self.params:
                if hasattr(param, 'main_grad'):
                    del param.main_grad
            self.main_grad_buffer.release_unsharded_buffer()

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
        if (
            self._full_grad_buffer_has_accumulated_grad
            or self._reduced_grad_buffer_has_accumulated_grad
        ):
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
        self.main_grad_buffer.data = None

    def _init_dist_params(self):
        """Initialize optimizer-facing DTensor views into the weight buffers."""
        self.dist_params = []
        self.dist_grads = []  # placeholder, populated in _init_dist_grads
        optimizer_buffer = self.main_weight_buffer or self.model_weight_buffer
        buffer_placements = [
            Placement.SHARD if self.outer_dp_sharding_strategy == "optim" else Placement.REPLICATE,
            Placement.SHARD if self.sharding_strategy != "no_shard" else Placement.REPLICATE,
        ]
        optimizer_dtensor_placements = [
            Shard(dim=0) if placement is Placement.SHARD else Replicate()
            for placement in buffer_placements
        ]
        if buffer_placements[0] is Placement.SHARD:
            setattr(self.mesh, "_shard_order", [1, 0])

        for param in self.params:
            item_id = self.param_idx[param]
            data = optimizer_buffer.get_item(item_id, placements=buffer_placements)
            param_shape = (
                param.shape
                if self.main_weight_buffer is not None
                else self.mp_policy.get_param_storage_shapes([param])[0]
            )

            dist_data = make_uneven_dtensor(
                data, param_shape, self.mesh, optimizer_dtensor_placements, post_process_uneven=True
            )
            dist_param = torch.nn.Parameter(dist_data, requires_grad=param.requires_grad)
            dist_param = torch.nn.Parameter(dist_data, requires_grad=param.requires_grad)
            # ``torch.nn.Parameter(DTensor)`` wraps the DTensor and creates a
            # fresh local tensor object, so Python-side uneven-DTensor metadata
            # attached by ``post_process_uneven=True`` is not preserved
            # automatically. Grad DTensor initialization later copies chunk
            # metadata from ``dist_param``; keep that invariant explicit here.
            copy_chunk_metadata(dist_data, dist_param)

            # Mark as FSDP parameter for special handling
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
        ``_init_buffers``; only the backing tensor is deferred.  Called from
        ``reduce_grad()`` on first use.  Uses ``torch.empty`` to avoid the
        zero-init cost. ``_reduced_grad_buffer_has_accumulated_grad`` is
        ``False`` after allocation, so the first reduce-scatter *overwrites*
        (``local_grad_shard.copy_``)
        rather than accumulating — the uninitialized data is never read.
        Subsequent calls are no-ops.
        """
        gbuf = self.main_grad_buffer
        if gbuf is None or not self.requires_grad:
            return
        if gbuf.data is not None:
            return  # already initialised

        gbuf.placements = gbuf.storage_placements.copy()
        gbuf.init_data(torch.empty(gbuf.data_size, dtype=gbuf.dtype, device=self.device))

        buffer_placements = [
            Placement.SHARD if isinstance(placement, Shard) else Placement.REPLICATE
            for placement in self.dist_params[0].placements
        ]

        for index, (p, dist_param, dist_grad) in enumerate(
            zip(self.params, self.dist_params, self._dist_grad_cache)
        ):
            item_id = self.param_idx[p]
            grad_dtensor_placements = dist_param.placements
            grad_data = gbuf.get_item(item_id, placements=buffer_placements)
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
        """In-place update ``dist_params._local_tensor`` / ``dist_grad._local_tensor``.

        Called after any buffer's ``self.data`` changes device (offload_to_cpu /
        auto-reload). Updates local tensor views using optimizer-buffer ownership.
        """
        optimizer_buffer = self.main_weight_buffer or self.model_weight_buffer
        buffer_placements = [
            Placement.SHARD if isinstance(placement, Shard) else Placement.REPLICATE
            for placement in self.dist_params[0].placements
        ]
        for param, dist_param in zip(self.params, self.dist_params):
            data = optimizer_buffer.get_item(self.param_idx[param], placements=buffer_placements)
            object.__setattr__(dist_param._local_tensor, 'data', data)

        if self.main_grad_buffer is not None and self.main_grad_buffer.data is not None:
            for param, dist_grad in zip(self.params, self.dist_grads):
                if dist_grad is None:
                    continue
                grad_data = self.main_grad_buffer.get_item(
                    self.param_idx[param], placements=buffer_placements
                )
                object.__setattr__(dist_grad._local_tensor, 'data', grad_data)

    def _ensure_buffers_on_gpu(self) -> bool:
        """Auto-reload any buffer on CPU back to GPU.

        Returns True if any buffer was moved (views were rebuilt).
        """
        moved = False
        for buf in (
            self.model_weight_buffer,
            self.main_weight_buffer,
            self.main_grad_buffer,
            self.transpose_weight_buffer,
        ):
            if buf is not None and buf._ensure_data_on_gpu():
                moved = True
        if moved:
            self._rebuild_dist_views()
        return moved

    def zero_grad(self, set_to_none: bool = True):
        """Zero the main gradient buffer and mark grads as zeroed."""
        self._full_grad_buffer_has_accumulated_grad = False
        self._reduced_grad_buffer_has_accumulated_grad = False
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
