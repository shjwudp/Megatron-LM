# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import enum
from contextlib import nullcontext
from copy import copy
from itertools import groupby
from typing import List, Optional

import torch
from torch.distributed import _coalescing_manager
from torch.distributed.tensor import DeviceMesh

from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .utils import ParamGroupIdx


class Placement(enum.Enum):
    """Logical state of a DP buffer along one mesh dimension.

    A buffer stores two enum members ordered as ``[outer-DP, inner-DP]``.

    Supported data transitions are:

    - ``SHARD`` -> ``REPLICATE``: all-gather
    - ``PARTIAL`` -> ``REPLICATE``: all-reduce
    - ``PARTIAL`` -> ``SHARD``: reduce-scatter
    - ``REPLICATE`` -> ``SHARD``: retain the rank-owned shard

    Allocation shape is intentionally not encoded here. A ``SHARD`` buffer
    may be a compact allocation or a slice of a ``REPLICATE`` output buffer.
    """

    SHARD = "shard"
    REPLICATE = "replicate"
    PARTIAL = "partial"


class DataParallelBuffer:
    """Manage mesh-aware flat storage for tensor values.

    The buffer owns layout, placement transitions, and storage lifecycle. It
    does not retain tensor identities or bind storage to consumers; those
    operations belong to the owning ``ParameterGroup``.
    """

    def __init__(
        self,
        tensors: List[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
        mesh: DeviceMesh,
        param_group_id: ParamGroupIdx,
        mp_policy,
        *,
        allocator: Optional[BucketAllocator] = None,
        buffer_role: str = "model_weight",
        gradient_scaling_factor: Optional[float] = None,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
        outer_dp_sharding_strategy: str = "no_shard",
    ):
        # Keep BufferIndex's Placement import from forming a module-level cycle.
        from .buffer_index import BufferIndex

        assert mp_policy is not None, "DataParallelBuffer requires a mixed-precision policy"
        self.dtype = dtype
        self.device = device
        self.mesh = mesh
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.alloc_key = (param_group_id, buffer_role)
        self.grad_comm_dtype = mp_policy.grad_comm_dtype or dtype
        self._use_grad_comm_buffer = self.grad_comm_dtype != dtype

        def is_sharded_from_strategy(strategy: str) -> bool:
            if buffer_role in ("model_weight", "transpose_weight"):
                return strategy == "optim_grads_params"
            if buffer_role == "main_weight":
                return strategy != "no_shard"
            if buffer_role == "main_grad":
                return strategy in ("optim_grads", "optim_grads_params")
            raise ValueError(f"Unsupported data-parallel buffer role: {buffer_role}")

        self.outer_sharded = is_sharded_from_strategy(outer_dp_sharding_strategy)
        self.inner_sharded = is_sharded_from_strategy(sharding_strategy)
        self.storage_placements: list[Placement] = [
            Placement.SHARD if sharded else Placement.REPLICATE
            for sharded in (self.outer_sharded, self.inner_sharded)
        ]
        self.placements: list[Placement] = self.storage_placements.copy()
        self.sharding_strategy = sharding_strategy
        self.outer_dp_sharding_strategy = outer_dp_sharding_strategy
        self.gradient_scaling_factor = gradient_scaling_factor

        # Always build layout with logical shapes and shared chunk_size_factor
        # so that all buffers share the same proportional item-offset mapping.
        logical_shapes = [tensor.shape for tensor in tensors]
        self.buffer_index = BufferIndex(
            param_shapes=logical_shapes,
            mesh=mesh,
            chunk_size_factor=chunk_size_factor,
            param_group_id=param_group_id,
        )

        # Compact NVFP4 weight buffers: scale all indices proportionally so
        # the buffer holds only the packed data without fragment-binning waste.
        if buffer_role in ("model_weight", "transpose_weight") and any(
            mp_policy.is_nvfp4_param(tensor) for tensor in tensors
        ):
            compact_shapes = mp_policy.get_param_storage_shapes(tensors)
            self.buffer_index.compact(0.5, compact_shapes)

        self.data_size = self.buffer_index._get_shard_meta(self.storage_placements).size

        self.data: Optional[torch.Tensor] = None
        self._unsharded_buffer: Optional[torch.Tensor] = None
        self._storage_owner: Optional["DataParallelBuffer"] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def init_data(self, data: torch.Tensor) -> None:
        """Bind an externally allocated tensor as the persistent storage."""
        assert data.dtype == self.dtype, f"dtype mismatch: {data.dtype} vs {self.dtype}"
        assert data.numel() == self.data_size, f"size mismatch: {data.numel()} vs {self.data_size}"
        self.data = data

    # ------------------------------------------------------------------ #
    #  CPU offload
    # ------------------------------------------------------------------ #

    def _is_on_cpu(self) -> bool:
        """True if ``self.data`` is resident on CPU."""
        return self.data is not None and self.data.device.type == "cpu"

    def _ensure_data_on_gpu(self) -> bool:
        """Move ``self.data`` to GPU if currently on CPU.

        Returns True if a move happened (caller must rebuild dist views).
        """
        if not self._is_on_cpu():
            return False
        self.data = self.data.to(self.device, non_blocking=True)
        return True

    def _move_data_to(
        self, target_device: torch.device, pin_memory: bool = False, non_blocking: bool = True
    ) -> None:
        """Move ``self.data`` to *target_device*, optionally using pinned memory.

        Caller must call ``ParameterGroup._rebuild_dist_views()`` afterwards
        because ``dist_params._local_tensor`` views share ``self.data`` Storage.
        """
        if self.data is None or self.data.device == target_device:
            return
        if target_device.type == "cpu" and pin_memory:
            cpu_data = torch.empty(self.data.shape, dtype=self.data.dtype, pin_memory=True)
            cpu_data.copy_(self.data, non_blocking=non_blocking)
            _free_storage(self.data)
            self.data = cpu_data
        else:
            self.data = self.data.to(target_device, non_blocking=non_blocking)

    @torch.no_grad()
    def set_item(
        self, item_id: int, item_data: torch.Tensor, *, placements: Optional[list[Placement]] = None
    ) -> None:
        """Write a tensor item into its corresponding region of the buffer."""
        requested_placements = placements if placements is not None else self.placements
        source_slice, local_slice = self.buffer_index.local_slice_for(
            self.buffer_index._get_item_global_range(item_id),
            requested_placements,
            self.storage_placements,
        )
        if source_slice is None or local_slice is None:
            return
        self.data[local_slice].copy_(item_data.flatten()[source_slice])

    def get_item(
        self, item_id: int, *, placements: Optional[list[Placement]] = None
    ) -> torch.Tensor:
        """Read a tensor item (or its shard) from the buffer."""
        requested_placements = placements if placements is not None else self.placements
        _, local_slice = self.buffer_index.local_slice_for(
            self.buffer_index._get_item_global_range(item_id),
            requested_placements,
            self.storage_placements,
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def is_unsharded(self) -> bool:
        """Return whether this buffer currently has a full unsharded view."""
        return all(placement is Placement.REPLICATE for placement in self.placements)

    def view(self, placements: list[Placement]) -> "DataParallelBuffer":
        """Return a non-owning DP-buffer view with explicit placements.

        The returned buffer has exact placement-shaped ``data``. For example,
        taking a ``SHARD`` view of a ``REPLICATE`` placeholder returns the
        rank-owned slice while retaining the placeholder as its storage owner.

        Args:
            placements: Logical placement for each mesh dimension.

        Returns:
            A DP buffer whose data is the requested view of this buffer's storage.
        """
        if len(placements) != self.mesh.ndim:
            raise ValueError(f"Expected {self.mesh.ndim} placements, got {placements}")
        view = copy(self)
        view.data = self.fetch_buffer(placements)
        view.data_size = view.data.numel()
        view.storage_placements = placements.copy()
        view.placements = placements.copy()
        view._unsharded_buffer = None
        view._storage_owner = self
        return view

    @staticmethod
    @torch.no_grad()
    def redistribute_buffers(
        buffers: list["DataParallelBuffer"],
        target_placements: list[Placement],
        *,
        output_buffers: list["DataParallelBuffer"] | None = None,
        stream: torch.cuda.Stream | None = None,
        async_op: bool = False,
    ) -> list[torch.Tensor]:
        """Redistribute compatible buffers to one target placement vector.

        The target defines the mesh-axis plan. Each axis is completed across
        communication-compatible buffers before the next axis starts, preserving
        HSDP outer-then-inner ordering and cross-buffer collective coalescing.
        """
        if not buffers:
            return []
        if any(len(target_placements) != buffer.mesh.ndim for buffer in buffers):
            raise ValueError(
                f"Expected {buffers[0].mesh.ndim} target placements, got {target_placements}"
            )
        if output_buffers is not None and len(output_buffers) != len(buffers):
            raise ValueError(f"Expected {len(buffers)} output buffers, got {len(output_buffers)}")

        caller_stream = torch.cuda.current_stream()
        stream = stream or caller_stream

        # Allocate final outputs on the caller stream before communication can
        # run asynchronously on a separate stream.
        if output_buffers is None:
            output_buffers = [buffer.view(target_placements) for buffer in buffers]
        else:
            for buffer, output_buffer in zip(buffers, output_buffers):
                buffer._validate_output_buffer(output_buffer, target_placements)
        outputs = [output_buffer.data for output_buffer in output_buffers]
        output_by_buffer_id = {
            id(buffer): output_buffer for buffer, output_buffer in zip(buffers, output_buffers)
        }
        if stream != caller_stream:
            stream.wait_stream(caller_stream)

        def compatibility_key(buffer: "DataParallelBuffer", mesh_dim: int):
            group = buffer.mesh.get_group(mesh_dim=mesh_dim)
            return id(group), buffer.dtype, buffer.device, buffer.placements[mesh_dim]

        with torch.cuda.stream(stream):
            for mesh_dim, target in enumerate(target_placements):
                for _, compatible_buffers_iter in groupby(
                    buffers, key=lambda buffer: compatibility_key(buffer, mesh_dim)
                ):
                    compatible_buffers = [
                        buffer
                        for buffer in compatible_buffers_iter
                        if buffer.placements[mesh_dim] is not target
                    ]
                    if not compatible_buffers:
                        continue

                    group = compatible_buffers[0].mesh.get_group(mesh_dim=mesh_dim)
                    context = (
                        _coalescing_manager(group, async_ops=async_op)
                        if len(compatible_buffers) > 1
                        and torch.distributed.get_world_size(group) > 1
                        else nullcontext()
                    )
                    with context as coalescing_event:
                        for buffer in compatible_buffers:
                            axis_target = buffer.placements.copy()
                            axis_target[mesh_dim] = target
                            final_output = output_by_buffer_id[id(buffer)]
                            axis_output = final_output if axis_target == target_placements else None
                            buffer.redistribute(
                                axis_target, output_buffer=axis_output, stream=stream
                            )
                    if async_op and coalescing_event is not None:
                        coalescing_event.wait()

        return outputs

    @torch.no_grad()
    def redistribute(
        self,
        target_placements: Optional[list[Placement]] = None,
        *,
        output_buffer: "DataParallelBuffer" | None = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Redistribute one mesh axis and return the destination tensor.

        ``output_buffer`` may name an explicit destination. This permits an
        in-place all-gather whose ``REPLICATE`` output owns full storage while
        this ``SHARD`` input is a slice sharing that same storage.
        """
        if target_placements is None:
            target_placements = self.storage_placements
        assert len(target_placements) == 2

        changed_axis = None
        for axis, (source, target) in enumerate(zip(self.placements, target_placements)):
            if source == target:
                continue
            if changed_axis is not None:
                raise ValueError(
                    "redistribute supports changing only one placement axis per call: "
                    f"{self.placements} -> {target_placements}"
                )
            changed_axis = axis
        if changed_axis is None:
            if output_buffer is None:
                return self.fetch_buffer(target_placements)
            self._validate_output_buffer(output_buffer, target_placements)
            return output_buffer.data

        current_stream = torch.cuda.current_stream()
        stream = stream or current_stream
        if stream != current_stream:
            stream.wait_stream(current_stream)

        source = self.placements[changed_axis]
        target = target_placements[changed_axis]
        input_buffer = self.view(self.placements).data
        explicit_output = output_buffer is not None
        if output_buffer is None:
            output_buffer = self.view(target_placements)
        else:
            self._validate_output_buffer(output_buffer, target_placements)
        output = output_buffer.data
        group = self.mesh.get_group(mesh_dim=changed_axis)

        if source is Placement.SHARD and target is Placement.REPLICATE:
            with torch.cuda.stream(stream):
                torch.distributed.all_gather_into_tensor(output, input_buffer, group=group)
        elif source is Placement.PARTIAL:
            scaling_factor = self.gradient_scaling_factor
            scale_inner = changed_axis == 1 and scaling_factor not in (None, 1.0)
            prescale = scale_inner and (
                self.grad_comm_dtype == torch.bfloat16
                or torch.distributed.get_world_size(group) == 1
            )
            op = (
                torch.distributed.ReduceOp.SUM
                if not scale_inner or prescale
                else torch.distributed._make_nccl_premul_sum(scaling_factor)
            )

            comm_input = input_buffer
            if self._use_grad_comm_buffer:
                comm_input = self.allocator.allocate(
                    key=(self.alloc_key, "grad_reduce_input", changed_axis),
                    size=input_buffer.numel(),
                    dtype=self.grad_comm_dtype,
                    device=self.device,
                ).data
            if comm_input.is_cuda:
                comm_input.record_stream(stream)

            with torch.cuda.stream(stream):
                if comm_input is not input_buffer:
                    comm_input.copy_(input_buffer)
                if prescale:
                    comm_input.mul_(scaling_factor)
                if target is Placement.REPLICATE:
                    torch.distributed.all_reduce(comm_input, group=group, op=op)
                    output = comm_input
                else:
                    if not explicit_output or comm_input is not input_buffer:
                        input_meta = self.buffer_index._get_shard_meta(self.placements)
                        output_meta = self.buffer_index._get_shard_meta(target_placements)
                        output_offset = output_meta.global_data_index - input_meta.global_data_index
                        output = comm_input[output_offset : output_offset + output_meta.size]
                    torch.distributed.reduce_scatter_tensor(
                        output=output, input=comm_input, group=group, op=op
                    )
        elif target is Placement.PARTIAL:
            pass
        elif target is Placement.SHARD:
            if source is Placement.REPLICATE:
                self.release_unsharded_buffer()
        else:
            raise NotImplementedError(f"Unsupported placement transition: {source!r} -> {target!r}")

        self.placements[changed_axis] = target
        return output

    def _validate_output_buffer(
        self, output_buffer: "DataParallelBuffer", target_placements: list[Placement]
    ) -> None:
        """Validate an explicit redistribution destination."""
        if output_buffer.buffer_index is not self.buffer_index:
            raise ValueError("Redistribution output must share the input buffer layout")
        if output_buffer.mesh is not self.mesh:
            raise ValueError("Redistribution output must share the input buffer mesh")
        if output_buffer.dtype != self.dtype or output_buffer.device != self.device:
            raise ValueError("Redistribution output must share the input dtype and device")
        if output_buffer.placements != target_placements:
            raise ValueError(
                f"Output placements {output_buffer.placements} do not match target "
                f"{target_placements}"
            )
        expected_size = self.buffer_index._get_shard_meta(target_placements).size
        if output_buffer.data is None or output_buffer.data.numel() != expected_size:
            raise ValueError(
                f"Output buffer size does not match target: expected {expected_size}, "
                f"got {None if output_buffer.data is None else output_buffer.data.numel()}"
            )

    def release_redistribution_workspace(self, changed_axis: int) -> None:
        """Release temporary communication storage for one mesh-axis transition."""
        if self._use_grad_comm_buffer:
            self.allocator.free((self.alloc_key, "grad_reduce_input", changed_axis))

    def release_unsharded_buffer(self) -> None:
        """Release the temporary full-sized buffer without changing placements."""
        if self._unsharded_buffer is not None:
            self.allocator.free(self.alloc_key)
            self._unsharded_buffer = None

    def get_shard_view(self, placements: Optional[list[Placement]] = None) -> torch.Tensor:
        """Return a placement view inside the persistent data buffer."""
        assert self.data is not None, "DataParallelBuffer data not initialized"
        requested_placements = placements if placements is not None else self.placements
        _, local_slice = self.buffer_index.local_slice_for(
            (0, self.buffer_index.bucket_meta.size), requested_placements, self.storage_placements
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def fetch_buffer(self, placements: list[Placement]) -> torch.Tensor:
        """Return a buffer for placements, allocating temporary storage if needed.

        1. If placements match the storage placements, return self.data directly.
        2. If self.data is a known parent of the requested placements, return a
           view into self.data.
        3. Otherwise allocate/reuse the fully replicated temporary buffer and
           return either that full buffer or a view from it.

        Memory allocation always occurs on the caller stream for deterministic
        caching-allocator behaviour.
        """
        requested_meta = self.buffer_index._get_shard_meta(placements)
        if placements == self.storage_placements:
            assert self.data is not None, "DataParallelBuffer data not initialized"
            return self.data

        data_contains_requested = all(
            storage_placement is not Placement.SHARD or requested_placement is Placement.SHARD
            for storage_placement, requested_placement in zip(self.storage_placements, placements)
        )
        if data_contains_requested:
            return self.get_shard_view(placements)

        if self._unsharded_buffer is None:
            bucket = self.allocator.allocate(
                key=self.alloc_key,
                size=self.buffer_index.bucket_meta.size,
                dtype=self.dtype,
                device=self.device,
            )
            self._unsharded_buffer = bucket.data
        if all(placement is Placement.REPLICATE for placement in placements):
            return self._unsharded_buffer
        return self._unsharded_buffer[
            requested_meta.bucket_data_index : requested_meta.bucket_data_index
            + requested_meta.size
        ]
