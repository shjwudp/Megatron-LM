# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.distributed.tensor import DeviceMesh

from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .buffer_index import BufferIndex
from .mixed_precision import MixedPrecisionPolicy
from .utils import ParamGroupIdx

logger = logging.getLogger(__name__)


class DataParallelBuffer:
    """Manages a flat buffer that stores (a shard of) a group of parameters.

    On construction it builds its own BufferIndex describing the layout and
    shard ownership.  External callers interact via init_data / set_item /
    get_item only.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_idx: Dict[torch.nn.Parameter, int],
        dtype: torch.dtype,
        device: torch.device,
        mesh: DeviceMesh,
        param_group_id: ParamGroupIdx,
        mp_policy: MixedPrecisionPolicy,
        *,
        allocator: Optional[BucketAllocator] = None,
        buffer_role: str = "model_weight",
        gradient_scaling_factor: Optional[float] = None,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
        outer_dp_sharding_strategy: str = "no_shard",
    ):
        assert mp_policy is not None, "DataParallelBuffer requires a mixed-precision policy"
        self.params = params
        self.param_idx = param_idx
        self.dtype = dtype
        self.device = device
        self.mesh = mesh
        self.outer_dp_group = mesh.get_group(mesh_dim=0)
        self.dp_group = mesh.get_group(mesh_dim=1)
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.buffer_role = buffer_role
        self.alloc_key = (param_group_id, buffer_role)
        self.mp_policy = mp_policy

        def inner_strategy_shards_buffer(strategy: str) -> bool:
            if buffer_role in ("model_weight", "transpose_weight"):
                return strategy == "optim_grads_params"
            if buffer_role == "main_weight":
                return strategy != "no_shard"
            if buffer_role == "main_grad":
                return strategy in ("optim_grads", "optim_grads_params")
            raise ValueError(f"Unsupported data-parallel buffer role: {buffer_role}")

        def outer_strategy_shards_buffer(strategy: str) -> bool:
            if strategy == "no_shard":
                return False
            if strategy != "optim":
                raise ValueError(f"Unsupported outer DP sharding strategy: {strategy}")
            if buffer_role in ("model_weight", "transpose_weight"):
                return False
            if buffer_role in ("main_weight", "main_grad"):
                return sharding_strategy != "no_shard"
            raise ValueError(f"Unsupported data-parallel buffer role: {buffer_role}")

        inner_sharded = inner_strategy_shards_buffer(sharding_strategy)
        outer_sharded = outer_strategy_shards_buffer(outer_dp_sharding_strategy)
        self.inner_sharded = inner_sharded
        self.outer_sharded = outer_sharded
        if outer_sharded and inner_sharded:
            # shard_dims=(outer, inner): outer sharded, inner sharded.
            self.storage_shard_dims = (1, 1)
        elif outer_sharded:
            # shard_dims=(outer, inner): outer sharded, inner not sharded.
            self.storage_shard_dims = (1, 0)
        elif inner_sharded:
            # shard_dims=(outer, inner): outer not sharded, inner sharded.
            self.storage_shard_dims = (0, 1)
        else:
            # shard_dims=(outer, inner): outer not sharded, inner not sharded.
            self.storage_shard_dims = (0, 0)
        self.sharding_strategy = sharding_strategy
        self.outer_dp_sharding_strategy = outer_dp_sharding_strategy
        self.gradient_scaling_factor = gradient_scaling_factor

        # Always build layout with logical shapes and shared chunk_size_factor
        # so that all buffers share the same proportional item-offset mapping.
        _logical_shapes = [p.shape for p in params]
        self.buffer_index = BufferIndex(
            param_shapes=_logical_shapes,
            mesh=mesh,
            chunk_size_factor=chunk_size_factor,
            param_group_id=param_group_id,
        )

        # Compact NVFP4 weight buffers: scale all indices proportionally so
        # the buffer holds only the packed data without fragment-binning waste.
        if buffer_role in ("model_weight", "transpose_weight") and any(
            mp_policy.is_nvfp4_param(p) for p in params
        ):
            compact_shapes = mp_policy.get_param_storage_shapes(params)
            self.buffer_index.compact(0.5, compact_shapes)

        self.data_size = self.buffer_index.outer_shard_metas[self.storage_shard_dims].size

        self.data: Optional[torch.Tensor] = None
        self._unsharded_buffer: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def init_data(self, data: torch.Tensor) -> None:
        """Bind an externally allocated tensor as the persistent storage."""
        assert data.dtype == self.dtype, f"dtype mismatch: {data.dtype} vs {self.dtype}"
        assert data.numel() == self.data_size, f"size mismatch: {data.numel()} vs {self.data_size}"
        self.data = data
        self._inner_dirty = False
        self._outer_dirty = False

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
        self,
        target_device: torch.device,
        pin_memory: bool = False,
        non_blocking: bool = True,
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

    def check_no_local_overlap(self, label: str = "") -> bool:
        """
        Runtime check: verify no two items' local slices overlap within ``self.data``.

        Returns True if layout is valid (no overlaps, all slices in bounds).
        Returns False and prints diagnostic info if any overlap or bound violation is found.
        """
        if self.data is None:
            return True

        items = self.buffer_index.item_index_map
        n_items = len(items)
        if n_items == 0:
            return True

        label_prefix = f"[{label}] " if label else ""

        # Collect (local_start, local_end, item_id, global_start, size) for each item
        slices = []
        for item_id in range(n_items):
            # shard_dims=(outer, inner): use this buffer's storage shard state.
            local_start, local_end = self.buffer_index._get_item_local_range(
                item_id, shard_dims=self.storage_shard_dims
            )
            idx = self.buffer_index.item_index_map[item_id]
            slices.append((local_start, local_end, item_id, idx.global_data_index, idx.size))

        # Sort by local_start
        slices.sort(key=lambda x: x[0])

        valid = True
        data_nel = self.data.numel()

        for i in range(len(slices)):
            s_start, s_end, s_id, g_start, size = slices[i]
            shape = items[s_id].shape

            # Bounds check: end must not exceed data size
            if s_end > data_nel:
                logger.warning(
                    f"{label_prefix}OVERFLOW: item {s_id} shape={list(shape)} "
                    f"local=[{s_start}, {s_end}) but data.numel()={data_nel} "
                    f"(global=[{g_start}, {g_start + size}))"
                )
                valid = False

            # Overlap check with next item
            if i + 1 < len(slices):
                n_start, n_end, n_id, n_gstart, n_size = slices[i + 1]
                if s_end > n_start:
                    overlap = s_end - n_start
                    logger.warning(
                        f"{label_prefix}OVERLAP: item {s_id} shape={list(shape)} "
                        f"local=[{s_start}, {s_end}) overlaps item {n_id} "
                        f"local=[{n_start}, {n_end}) by {overlap} elements "
                        f"(global_{s_id}=[{g_start}, {g_start + size}), "
                        f"global_{n_id}=[{n_gstart}, {n_gstart + n_size}))"
                    )
                    valid = False

        return valid

    def check_no_global_overlap(self, label: str = "") -> bool:
        """
        Runtime check: verify no two items' **global** slices overlap.

        This checks the logical layout (should never fail if _build_layout is correct).
        """
        items = self.buffer_index.item_index_map
        n_items = len(items)
        if n_items == 0:
            return True

        label_prefix = f"[{label}] " if label else ""

        ranges = []
        for item_id in range(n_items):
            idx = items[item_id]
            ranges.append(
                (idx.global_data_index, idx.global_data_index + idx.size, item_id, idx.shape)
            )

        ranges.sort(key=lambda x: x[0])

        valid = True
        for i in range(len(ranges) - 1):
            a_start, a_end, a_id, a_shape = ranges[i]
            b_start, b_end, b_id, b_shape = ranges[i + 1]
            if a_end > b_start:
                logger.warning(
                    f"{label_prefix}GLOBAL OVERLAP: item {a_id} shape={list(a_shape)} "
                    f"[{a_start}, {a_end}) vs item {b_id} shape={list(b_shape)} "
                    f"[{b_start}, {b_end}) overlap={a_end - b_start}"
                )
                valid = False

        if valid:
            pass  # silent on success
        return valid

    def set_item(
        self,
        item_id: int,
        item_data: torch.Tensor,
        *,
        shard_dims: Optional[Iterable[int]] = None,
    ) -> None:
        """Write a parameter tensor into the corresponding region of the buffer."""
        if shard_dims is None:
            # shard_dims=(outer, inner): use this buffer's storage shard state.
            shard_dims = self.storage_shard_dims
        slice_start, slice_end = self.buffer_index._get_item_self_range(
            item_id, shard_dims=shard_dims
        )
        storage_slice_start, storage_slice_end = self.buffer_index._get_item_self_range(
            item_id, shard_dims=self.storage_shard_dims
        )
        slice_start = max(slice_start, storage_slice_start)
        slice_end = min(slice_end, storage_slice_end)
        if slice_start >= slice_end:
            return
        storage_local_start, _ = self.buffer_index._get_item_local_range(
            item_id, shard_dims=self.storage_shard_dims
        )
        local_start = storage_local_start + slice_start - storage_slice_start
        local_end = local_start + (slice_end - slice_start)
        shard = self.data[local_start:local_end]
        item_data = item_data.flatten()[slice_start:slice_end]
        shard.data.copy_(item_data.flatten())

    def get_item(
        self, item_id: int, *, shard_dims: Optional[Iterable[int]] = None
    ) -> torch.Tensor:
        """Read a parameter tensor (or its shard) from the buffer."""
        if shard_dims is None:
            # shard_dims=(outer, inner): use this buffer's storage shard state.
            shard_dims = self.storage_shard_dims
        slice_start, slice_end = self.buffer_index._get_item_self_range(
            item_id, shard_dims=shard_dims
        )
        storage_slice_start, storage_slice_end = self.buffer_index._get_item_self_range(
            item_id, shard_dims=self.storage_shard_dims
        )
        slice_start = max(slice_start, storage_slice_start)
        slice_end = min(slice_end, storage_slice_end)
        if slice_start >= slice_end:
            return self.data[:0]
        storage_local_start, _ = self.buffer_index._get_item_local_range(
            item_id, shard_dims=self.storage_shard_dims
        )
        start = storage_local_start + slice_start - storage_slice_start
        end = start + (slice_end - slice_start)
        return self.data[start:end]

    def is_unsharded(self) -> bool:
        """Return whether this buffer currently has a full unsharded view."""
        if self._outer_dirty or self._inner_dirty:
            return False
        # shard_dims=(outer, inner): (0, 0) means neither dimension is sharded.
        if self.storage_shard_dims != (0, 0):
            return self._unsharded_buffer is not None
        return self.data is not None

    @torch.no_grad()
    def unshard(
        self,
        unshard_dim: Optional[int] = 1,
        bind_params: bool = False,
    ) -> torch.Tensor:
        """All-gather selected dimensions and optionally bind params.

        ``unshard_dim`` uses mesh dim ids: ``None`` does not unshard,
        ``0`` unshards outer-DP, and ``1`` unshards inner-DP.
        """
        # If unshard_dim is set, that dimension becomes replicated in the target.
        # Otherwise, every dimension keeps the current storage state.
        target_shard_dims = (
            0 if unshard_dim == 0 else self.storage_shard_dims[0],
            0 if unshard_dim == 1 else self.storage_shard_dims[1],
        )
        dirty_flags = (self._outer_dirty, self._inner_dirty)
        storage_is_dirty = (
            unshard_dim is not None
            and self.storage_shard_dims[unshard_dim] == 0
            and dirty_flags[unshard_dim]
        )
        # If storage is replicated but dirty, dimension d acts as a sharded source.
        # Otherwise, dimension d keeps the current storage state as the source.
        source_shard_dims = (
            1 if storage_is_dirty and unshard_dim == 0 else self.storage_shard_dims[0],
            1 if storage_is_dirty and unshard_dim == 1 else self.storage_shard_dims[1],
        )
        # Only a source-sharded -> target-replicated transition needs all-gather.
        requires_unshard = (
            unshard_dim is not None
            and source_shard_dims[unshard_dim] == 1
            and target_shard_dims[unshard_dim] == 0
        )

        # Fast path: target is already available from clean local storage.
        if not requires_unshard:
            output_buffer = self.fetch_buffer(target_shard_dims)
            if bind_params and target_shard_dims == (0, 0):
                self._bind_buffer_to_params(output_buffer)
            return output_buffer

        output_shard_dims = (
            0 if unshard_dim == 0 else source_shard_dims[0],
            0 if unshard_dim == 1 else source_shard_dims[1],
        )
        group = self.outer_dp_group if unshard_dim == 0 else self.dp_group

        input_buffer = self.fetch_buffer(source_shard_dims)
        output_buffer = self.fetch_buffer(output_shard_dims)
        if torch.distributed.get_world_size(group) == 1:
            if output_buffer.data_ptr() != input_buffer.data_ptr():
                output_buffer.copy_(input_buffer)
        else:
            torch.distributed.all_gather_into_tensor(
                output_tensor=output_buffer,
                input_tensor=input_buffer,
                group=group,
            )
            if output_buffer.is_cuda:
                # Temporary all-gather buckets may be released from another stream before
                # the collective finishes; record the producer stream for allocator safety.
                output_buffer.record_stream(torch.cuda.current_stream())

        setattr(self, "_outer_dirty" if unshard_dim == 0 else "_inner_dirty", False)

        # Parameter binding needs the full compute buffer.
        if bind_params and output_shard_dims == (0, 0):
            self._bind_buffer_to_params(output_buffer)
        return output_buffer

    def _bind_buffer_to_params(self, buffer: torch.Tensor) -> None:
        """Bind the given buffer to the params according to the layout."""
        assert buffer.numel() == self.buffer_index.bucket_meta.size, (
            f"Buffer size {buffer.numel()} does not match expected size "
            f"{self.buffer_index.bucket_meta.size}"
        )
        for p in self.params:
            item_id = self.param_idx[p]
            start, end = self.buffer_index._get_item_global_range(item_id)
            idx_shape = self.buffer_index.item_index_map[item_id].shape
            param_data = buffer[start:end].view(idx_shape)
            self.mp_policy.bind_unsharded_param(p, param_data, self.buffer_role)

    @torch.no_grad()
    def reshard(self, shard_dim: Optional[int] = None) -> None:
        """Release temporary buffers allocated by ``fetch_buffer`` / ``unshard``."""
        if shard_dim is not None:
            # If storage is already replicated on this dim, unshard() returned
            # self.data or a self.data view, so no temporary buffer was allocated.
            if self.storage_shard_dims[shard_dim] == 0:
                return
        self.allocator.free(self.alloc_key)
        self._unsharded_buffer = None

    def get_shard_view(self, shard_dims: Optional[Iterable[int]] = None) -> torch.Tensor:
        """Return a shard view inside ``self.data``."""
        assert self.data is not None, "DataParallelBuffer data not initialized"
        if shard_dims is None:
            # shard_dims=(outer, inner): use this buffer's storage shard state.
            shard_dims = self.storage_shard_dims
        requested_meta = self.buffer_index._get_shard_meta(shard_dims)
        # shard_dims=(outer, inner): storage_shard_dims describes self.data's shard state.
        storage_meta = self.buffer_index.outer_shard_metas[self.storage_shard_dims]
        range_start = max(requested_meta.global_data_index, storage_meta.global_data_index)
        range_end = min(
            requested_meta.global_data_index + requested_meta.size,
            storage_meta.global_data_index + storage_meta.size,
        )
        if range_start >= range_end:
            return self.data[:0]
        local_start = storage_meta.local_data_index + range_start - storage_meta.global_data_index
        return self.data[local_start : local_start + (range_end - range_start)]

    def fetch_buffer(self, shard_dims: Tuple[int, int] = (0, 0)) -> torch.Tensor:
        """Return a buffer for ``shard_dims``, allocating temporary storage if needed.

        1. If ``shard_dims`` matches this buffer's storage layout, return
           ``self.data`` directly.
        2. If ``self.data`` is a known parent layout of the requested shard,
           return a view into ``self.data``. Example: storage ``(0, 1)`` can
           return a ``(1, 1)`` view.
        3. Otherwise allocate/reuse the full ``(0, 0)`` unsharded buffer and
           return either that full buffer or a view from it. Example: storage
           ``(1, 1)`` requesting ``(0, 1)`` must materialize the full buffer
           because one outer shard cannot cover the complete inner-DP shard.
        """
        requested_shard_dims = shard_dims

        # 1. Exact storage match: no view or temporary buffer needed.
        if requested_shard_dims == self.storage_shard_dims:
            assert self.data is not None, "DataParallelBuffer data not initialized"
            return self.data

        # 2. Parent storage layouts can directly expose a child shard view.
        data_contains_requested = all(
            storage_dim == 0 or storage_dim == requested_dim
            for storage_dim, requested_dim in zip(self.storage_shard_dims, requested_shard_dims)
        )
        if data_contains_requested:
            return self.get_shard_view(requested_shard_dims)

        # 3. Otherwise materialize the full buffer and return the requested view
        # from it. This covers HSDP storage (1, 1) -> requested (0, 1).
        if self._unsharded_buffer is None:
            bucket = self.allocator.allocate(
                key=self.alloc_key,
                size=self.buffer_index.bucket_meta.size,
                dtype=self.dtype,
                device=self.device,
            )
            self._unsharded_buffer = bucket.data
        if requested_shard_dims == (0, 0):
            return self._unsharded_buffer
        requested_meta = self.buffer_index._get_shard_meta(requested_shard_dims)
        return self._unsharded_buffer[
            requested_meta.bucket_data_index : requested_meta.bucket_data_index
            + requested_meta.size
        ]

    @torch.no_grad()
    def reduce_grad(
        self,
        grad_comm_dtype: Optional[torch.dtype] = None,
        overwrite_grad: bool = False,
        reduce_dim: Optional[int] = 1,
        reduce_scatter: bool = True,
    ):
        """Reduce gradients into the optimizer-facing local shard.

        ``reduce_dim`` uses mesh dim ids: ``None`` does not reduce,
        ``0`` reduces outer-DP, and ``1`` reduces inner-DP.
        ``reduce_scatter`` selects RS vs AR; ParameterGroup owns that strategy decision.
        """
        if reduce_dim is None:
            return

        grad_comm_dtype = grad_comm_dtype or self.dtype
        # Scale exactly once, when reducing fresh full grads over inner-DP.
        # Outer-only reduce consumes an already-scaled inner-DP result.
        if reduce_dim != 1 or self.gradient_scaling_factor in (None, 1.0):
            op = torch.distributed.ReduceOp.SUM
            prescale = False
        elif grad_comm_dtype != torch.bfloat16:
            op = torch.distributed._make_nccl_premul_sum(self.gradient_scaling_factor)
            prescale = False
        else:
            op = torch.distributed.ReduceOp.SUM
            prescale = True

        # Inner reduce consumes fresh full grads: (0, 0) -> (0, 1).
        # Outer reduce consumes the inner-reduced view: (0, 1) -> (1, 1).
        input_shard_dims = (
            0,
            0 if reduce_dim == 1 else self.storage_shard_dims[1],
        )
        # AR keeps the same shard view; RS shards the reduced dimension.
        output_shard_dims = (
            1 if reduce_scatter and reduce_dim == 0 else input_shard_dims[0],
            1 if reduce_scatter and reduce_dim == 1 else input_shard_dims[1],
        )
        input_buffer = self.fetch_buffer(input_shard_dims)
        output_buffer = self.fetch_buffer(output_shard_dims)

        # Pick the process group covering exactly the reduced dimension.
        group = self.outer_dp_group if reduce_dim == 0 else self.dp_group
        if torch.distributed.get_world_size(group) == 1:
            if output_buffer.data_ptr() != input_buffer.data_ptr():
                output_buffer.copy_(input_buffer)
            return

        comm_input = input_buffer
        input_key = None
        if grad_comm_dtype != self.dtype:
            input_key = (self.alloc_key, "grad_reduce_input", reduce_dim)
            input_bucket = self.allocator.allocate(
                key=input_key,
                size=input_buffer.numel(),
                dtype=grad_comm_dtype,
                device=self.device,
            )
            comm_input = input_bucket.data
            comm_input.copy_(input_buffer)
        if prescale:
            comm_input.mul_(self.gradient_scaling_factor)

        if not reduce_scatter:
            torch.distributed.all_reduce(comm_input, group=group, op=op)
            if input_key is not None:
                output_buffer.copy_(comm_input.to(self.dtype))
                self.allocator.free(input_key)
            return

        if input_buffer.is_cuda:
            # Keep temporary reduce-scatter buffers tied to the stream that uses them.
            input_buffer.record_stream(torch.cuda.current_stream())

        comm_output = output_buffer
        output_key = None
        if grad_comm_dtype != self.dtype or not overwrite_grad:
            output_key = (self.alloc_key, "grad_reduce_output", reduce_dim)
            output_bucket = self.allocator.allocate(
                key=output_key,
                size=output_buffer.numel(),
                dtype=grad_comm_dtype,
                device=self.device,
            )
            comm_output = output_bucket.data

        torch.distributed.reduce_scatter_tensor(
            output=comm_output,
            input=comm_input,
            group=group,
            op=op,
        )

        if output_buffer.data_ptr() != comm_output.data_ptr():
            if overwrite_grad:
                output_buffer.copy_(comm_output)
            else:
                output_buffer += comm_output
        if output_key is not None:
            self.allocator.free(output_key)
        if input_key is not None:
            self.allocator.free(input_key)


def check_all_fsdp_buffers(module) -> bool:
    """
    Scan every FSDPModule in *module* and verify no local slice overlaps
    in any buffer (model_weight, main_weight, main_grad).

    Call this at any point after FSDP initialization to catch runtime
    corruption.  Returns True if all buffers are clean.
    """
    import torch.distributed as dist

    from .fsdp_module import FSDPModule

    rank = dist.get_rank() if dist.is_initialized() else -1
    all_ok = True

    for name, child in module.named_modules():
        if not isinstance(child, FSDPModule):
            continue
        for param_names, param_group in child._named_param_groups:
            gid = f"mod={name} pg={param_group.param_group_id} rank={rank}"
            if param_group.model_weight_buffer is not None:
                ok = param_group.model_weight_buffer.check_no_local_overlap(gid + " wbuf")
                all_ok = all_ok and ok
            if param_group.main_weight_buffer is not None:
                ok = param_group.main_weight_buffer.check_no_local_overlap(gid + " mbuf")
                all_ok = all_ok and ok
            if param_group.main_grad_buffer is not None:
                ok = param_group.main_grad_buffer.check_no_local_overlap(gid + " gbuf")
                all_ok = all_ok and ok

    return all_ok
