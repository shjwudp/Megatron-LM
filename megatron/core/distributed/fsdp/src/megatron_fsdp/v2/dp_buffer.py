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
from typing import Dict, List, Optional

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

        def strategy_shards_buffer(strategy: str) -> bool:
            if buffer_role in ("model_weight", "transpose_weight"):
                return strategy == "optim_grads_params"
            if buffer_role == "main_weight":
                return strategy != "no_shard"
            if buffer_role == "main_grad":
                return strategy in ("optim_grads", "optim_grads_params")
            raise ValueError(f"Unsupported data-parallel buffer role: {buffer_role}")

        inner_sharded = strategy_shards_buffer(sharding_strategy)
        outer_sharded = strategy_shards_buffer(outer_dp_sharding_strategy)
        self.inner_sharded = inner_sharded
        self.outer_sharded = outer_sharded
        self.sharding_strategy = sharding_strategy
        self.outer_dp_sharding_strategy = outer_dp_sharding_strategy
        self.gradient_scaling_factor = gradient_scaling_factor

        # Always build layout with logical shapes and shared chunk_size_factor
        # so that all buffers share the same proportional item-offset mapping.
        _logical_shapes = [p.shape for p in params]
        self.buffer_index = BufferIndex(
            param_shapes=_logical_shapes,
            mesh=mesh,
            inner_sharded=inner_sharded,
            chunk_size_factor=chunk_size_factor,
            outer_sharded=outer_sharded,
            param_group_id=param_group_id,
        )

        # Compact NVFP4 weight buffers: scale all indices proportionally so
        # the buffer holds only the packed data without fragment-binning waste.
        if buffer_role in ("model_weight", "transpose_weight") and any(
            mp_policy.is_nvfp4_param(p) for p in params
        ):
            compact_shapes = mp_policy.get_param_storage_shapes(params)
            self.buffer_index.compact(0.5, compact_shapes)

        if self.outer_sharded:
            self.data_size = self.buffer_index.outer_shard_meta.size
        elif self.inner_sharded:
            self.data_size = self.buffer_index.shard_meta.size
        else:
            self.data_size = self.buffer_index.bucket_meta.size

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
            local_start, local_end = self.buffer_index._get_item_local_range(item_id)
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
        shard_level: Optional[str] = None,
    ) -> None:
        """Write a parameter tensor into the corresponding region of the buffer."""
        if shard_level is None:
            shard_level = (
                "outer" if self.outer_sharded else "inner" if self.inner_sharded else "full"
            )
        local_start, local_end = self.buffer_index._get_item_local_range(
            item_id, shard_level=shard_level
        )
        shard = self.data[local_start:local_end]
        if shard.numel() > 0:
            idx = self.buffer_index.item_index_map[item_id]
            if self.outer_sharded:
                storage_meta = self.buffer_index.outer_shard_meta
            elif self.inner_sharded:
                storage_meta = self.buffer_index.shard_meta
            else:
                storage_meta = None

            if storage_meta is not None:
                global_start = (
                    storage_meta.global_data_index
                    + local_start
                    - storage_meta.local_data_index
                )
            else:
                global_start = local_start
            slice_start = global_start - idx.global_data_index
            slice_end = slice_start + shard.numel()
            item_data = item_data.flatten()[slice_start:slice_end]
            shard.data.copy_(item_data.flatten())

    def get_item(self, item_id: int, *, shard_level: Optional[str] = None) -> torch.Tensor:
        """Read a parameter tensor (or its shard) from the buffer."""
        if shard_level is None:
            shard_level = (
                "outer" if self.outer_sharded else "inner" if self.inner_sharded else "full"
            )
        start, end = self.buffer_index._get_item_local_range(
            item_id, shard_level=shard_level
        )
        return self.data[start:end]

    def is_unsharded(self) -> bool:
        """Return whether this buffer currently has a full unsharded view."""
        if self._outer_dirty or self._inner_dirty:
            return False
        if self.inner_sharded:
            return self._unsharded_buffer is not None
        return self.data is not None

    @torch.no_grad()
    def unshard(
        self,
        bind_params: bool = False,
    ) -> torch.Tensor:
        """All-gather the full buffer from all shards and bind parameter storage.

        For non-distributed buffers self.data is already full, so
        self.data is returned directly. If a replicated buffer only has this
        rank's updated shard, the shard is all-gathered into self.data first.
        """
        full_buffer = self.fetch_buffer()

        if not self.inner_sharded and not self._inner_dirty:
            if bind_params:
                self._bind_buffer_to_params(full_buffer)
            return full_buffer

        shard_buffer = self.get_shard_view("inner_shard")
        torch.distributed.all_gather_into_tensor(
            output_tensor=full_buffer,
            input_tensor=shard_buffer,
            group=self.dp_group,
        )
        if full_buffer.is_cuda:
            # Temporary all-gather buckets may be released from another stream before
            # the collective finishes; record the producer stream for allocator safety.
            full_buffer.record_stream(torch.cuda.current_stream())

        if bind_params:
            self._bind_buffer_to_params(full_buffer)

        self._inner_dirty = False

        return full_buffer

    @torch.no_grad()
    def unshard_outer(self) -> torch.Tensor:
        """All-gather outer optimizer shards into this rank's local inner-DP shard."""
        full_buffer = self.get_shard_view("inner_shard")
        if torch.distributed.get_world_size(self.outer_dp_group) == 1:
            self._outer_dirty = False
            return full_buffer
        if not self._outer_dirty:
            return full_buffer

        shard_buffer = self.get_shard_view("outer_shard")
        torch.distributed.all_gather_into_tensor(
            output_tensor=full_buffer,
            input_tensor=shard_buffer,
            group=self.outer_dp_group,
        )
        if full_buffer.is_cuda:
            full_buffer.record_stream(torch.cuda.current_stream())
        self._outer_dirty = False
        return full_buffer

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
    def reshard(self) -> None:
        """Release the temporary unsharded buffer allocated by unshard()."""
        if not self.inner_sharded:
            return
        self.allocator.free(self.alloc_key)
        self._unsharded_buffer = None

    def get_shard_view(self, shard_mode: str) -> torch.Tensor:
        """Return an inner/outer persistent shard view inside ``self.data``."""
        if shard_mode == "inner_shard":
            assert self.data is not None, "DataParallelBuffer data not initialized"
            sm = self.buffer_index.shard_meta
            return self.data[sm.local_data_index : sm.local_data_index + sm.size]
        if shard_mode == "outer_shard":
            assert self.data is not None, "DataParallelBuffer data not initialized"
            assert self.buffer_index.outer_shard_meta is not None
            sm = self.buffer_index.outer_shard_meta
            return self.data[sm.local_data_index : sm.local_data_index + sm.size]
        raise ValueError(f"Unsupported shard_mode: {shard_mode}")

    def fetch_buffer(self) -> torch.Tensor:
        """Return the full unsharded buffer, allocating it if needed.

        Memory allocation always occurs on the default stream for deterministic
        caching-allocator behaviour.
        """
        if self.inner_sharded:
            if self._unsharded_buffer is None:
                bucket = self.allocator.allocate(
                    key=self.alloc_key,
                    size=self.buffer_index.bucket_meta.size,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._unsharded_buffer = bucket.data
            full = self._unsharded_buffer
        else:
            assert self.data is not None, "DataParallelBuffer data not initialized"
            full = self.data

        return full

    @torch.no_grad()
    def reduce_grad(
        self,
        grad_comm_dtype: Optional[torch.dtype] = None,
        overwrite_grad: bool = False,
    ):
        """Reduce gradients into the optimizer-facing local shard.

        For distributed buffers, this reduce-scatters a temporary full gradient
        and accumulates the result into the persistent local shard. For
        replicated buffers, this reduce-scatters the full accumulation buffer
        once into this rank's virtual shard for ZeRO-1 optimizer consumption.
        For no-shard buffers, this all-reduces the full gradient buffer.
        If grad_comm_dtype differs from self.dtype, communicate with a temporary
        casted tensor and cast the reduced result back before accumulation.
        """
        if self.sharding_strategy in ("no_shard", "optim"):
            overwrite_grad = True

        grad_comm_dtype = grad_comm_dtype or self.dtype

        if self.gradient_scaling_factor in (None, 1.0):
            op = torch.distributed.ReduceOp.SUM
            prescale = False
        elif grad_comm_dtype != torch.bfloat16:
            op = torch.distributed._make_nccl_premul_sum(self.gradient_scaling_factor)
            prescale = False
        else:
            op = torch.distributed.ReduceOp.SUM
            prescale = True

        sm = self.buffer_index.shard_meta
        local_grad_shard = self.data[sm.local_data_index : sm.local_data_index + sm.size]

        if not self.inner_sharded and self.sharding_strategy == "no_shard":
            comm_input = (
                self.data if grad_comm_dtype == self.dtype else self.data.to(grad_comm_dtype)
            )
            if prescale:
                comm_input.mul_(self.gradient_scaling_factor)
            torch.distributed.all_reduce(comm_input, group=self.dp_group, op=op)
            if grad_comm_dtype != self.dtype:
                self.data.copy_(comm_input.to(self.dtype))
            return

        if self.inner_sharded:
            # ZeRO-2/3 (optim_grads/optim_grads_params): ``self.data`` is the
            # persistent local grad shard. The full grad buffer is temporary,
            # assembled only for this reduce-scatter, and the RS result is
            # accumulated into ``local_grad_shard`` for gradient accumulation.
            input_buffer = self.fetch_buffer()
            output_offset = sm.bucket_data_index
            if input_buffer.is_cuda:
                # Keep temporary reduce-scatter buffers tied to the stream that uses them.
                input_buffer.record_stream(torch.cuda.current_stream())
        else:
            # ZeRO-1 (optim): ``self.data`` is the replicated full grad
            # accumulation buffer. The optimizer consumes only this rank's
            # virtual shard, so the one delayed RS writes directly into that
            # slice instead of accumulating into a separate shard buffer.
            input_buffer = self.data
            output_offset = sm.local_data_index

        comm_input = (
            input_buffer if grad_comm_dtype == self.dtype else input_buffer.to(grad_comm_dtype)
        )
        if prescale:
            comm_input.mul_(self.gradient_scaling_factor)

        reduced_grad_shard = comm_input[output_offset : output_offset + local_grad_shard.numel()]

        torch.distributed.reduce_scatter_tensor(
            output=reduced_grad_shard, input=comm_input, group=self.dp_group, op=op
        )

        if local_grad_shard.data_ptr() == reduced_grad_shard.data_ptr():
            return

        if overwrite_grad:
            local_grad_shard.copy_(reduced_grad_shard)
        else:
            local_grad_shard += reduced_grad_shard

    @torch.no_grad()
    def reduce_grad_outer(self, grad_comm_dtype: Optional[torch.dtype] = None) -> None:
        """Reduce optimizer-facing gradients across the outer-DP dimension."""
        grad_comm_dtype = grad_comm_dtype or self.dtype
        if self.outer_dp_group is None:
            return
        if torch.distributed.get_world_size(self.outer_dp_group) == 1:
            return

        if self.outer_dp_sharding_strategy == "no_shard":
            grad = (
                self.data
                if self.sharding_strategy == "no_shard"
                else self.get_shard_view("inner_shard")
            )
            if grad.numel() == 0:
                return

            comm_grad = grad
            comm_key = None
            if grad_comm_dtype != self.dtype:
                comm_key = (self.alloc_key, "hsdp_grad_comm")
                bucket = self.allocator.allocate(
                    key=comm_key,
                    size=grad.numel(),
                    dtype=grad_comm_dtype,
                    device=self.device,
                )
                comm_grad = bucket.data
                comm_grad.copy_(grad)

            torch.distributed.all_reduce(
                comm_grad,
                group=self.outer_dp_group,
                op=torch.distributed.ReduceOp.SUM,
            )
            if comm_key is not None:
                grad.copy_(comm_grad.to(self.dtype))
                self.allocator.free(comm_key)
            return

        if self.outer_dp_sharding_strategy != "optim":
            raise NotImplementedError(
                f"Unsupported outer-DP sharding strategy: {self.outer_dp_sharding_strategy}"
            )

        input_buffer = self.get_shard_view("inner_shard")
        output_shard = self.get_shard_view("outer_shard")
        comm_input = input_buffer
        comm_output = output_shard
        input_key = None
        output_key = None
        if grad_comm_dtype != self.dtype:
            input_key = (self.alloc_key, "hsdp_outer_grad_input")
            input_bucket = self.allocator.allocate(
                key=input_key,
                size=input_buffer.numel(),
                dtype=grad_comm_dtype,
                device=self.device,
            )
            comm_input = input_bucket.data
            comm_input.copy_(input_buffer)

            output_key = (self.alloc_key, "hsdp_outer_grad_output")
            output_bucket = self.allocator.allocate(
                key=output_key,
                size=output_shard.numel(),
                dtype=grad_comm_dtype,
                device=self.device,
            )
            comm_output = output_bucket.data

        torch.distributed.reduce_scatter_tensor(
            output=comm_output,
            input=comm_input,
            group=self.outer_dp_group,
            op=torch.distributed.ReduceOp.SUM,
        )
        if output_key is not None:
            output_shard.copy_(comm_output.to(self.dtype))
            self.allocator.free(output_key)
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
