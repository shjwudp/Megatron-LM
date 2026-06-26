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

from ..uneven_dtensor import make_uneven_dtensor, update_uneven_dtensor_chunk_metadata
from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .dp_buffer import DataParallelBuffer
from .mixed_precision import MixedPrecisionPolicy
from .utils import ParamGroupIdx, _prepare_fsdp_mesh


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
            mesh = DeviceMesh(
                self.device.type,
                world_ranks,
                mesh_dim_names=("dp_outer", "dp"),
            )
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

        # Buffer references (initialized in _init_buffers)
        self.model_weight_buffer: Optional[DataParallelBuffer] = None
        self.transpose_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_grad_buffer: Optional[DataParallelBuffer] = None
        # Initialize buffers and distributed parameters
        self._init_buffers()

    def set_allocator(self, allocator: BucketAllocator) -> None:
        self.allocator = allocator
        for buffer in (
            self.model_weight_buffer,
            self.transpose_weight_buffer,
            self.main_weight_buffer,
            self.main_grad_buffer,
        ):
            if buffer is not None:
                buffer.allocator = allocator

    def _create_buffer(self, dtype: torch.dtype, role: str) -> DataParallelBuffer:
        """Create a buffer and namespace its temporary bucket by role."""
        return DataParallelBuffer(
            params=self.params,
            param_idx=self.param_idx,
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

        # Create main weight buffer for mixed precision
        main_params_dtype = self.mp_policy.main_params_dtype_for_param(self.params[0])
        if main_params_dtype is not None:
            mbuf = self._create_buffer(main_params_dtype, "main_weight")
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
                weight_buffer._bind_buffer_to_params(weight_buffer.data)

        # Create gradient buffer
        if self.requires_grad:
            main_grads_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
            gbuf = self._create_buffer(main_grads_dtype, "main_grad")
            self.main_grad_buffer = gbuf

        # Create distributed parameter views
        self._init_dist_params()

    def unshard(self, bwd_pass: bool = False, bind_params: bool = True):
        """
        Unshard model weights by all-gathering from sharded buffer.

        After unshard, parameters point to full unsharded storage. FP8
        parameters rebind their TE raw payload instead of ``param.data``.
        """
        self._ensure_buffers_on_gpu()

        for weight_buffer in self.mp_policy.weight_buffers_for_unshard(
            self.model_weight_buffer, self.transpose_weight_buffer, bwd_pass=bwd_pass
        ):
            if weight_buffer is not None:
                if self.outer_dp_sharding_strategy == "optim":
                    # mesh dim 0 is outer-DP.
                    weight_buffer.unshard(unshard_dim=0, bind_params=False)
                # mesh dim 1 is inner-DP.
                weight_buffer.unshard(unshard_dim=1, bind_params=bind_params)

        self.mp_policy.post_unshard(self.params, bwd_pass=bwd_pass)

    def has_unsharded_weight_buffers(self, bwd_pass: bool = False) -> bool:
        """Return whether this phase can skip launching another distributed unshard."""
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
        self.model_weight_buffer.reshard()
        if self.transpose_weight_buffer is not None:
            self.transpose_weight_buffer.reshard()
        self.mp_policy.post_reshard(self.params)

    @torch.no_grad()
    def copy_main_weights_to_model_weights(self):
        """Install optimized main weights into model compute weights."""
        self._ensure_buffers_on_gpu()
        self.mp_policy.copy_main_weights_to_model_weights(
            self.params,
            self.param_idx,
            self.mesh,
            self.model_weight_buffer,
            self.main_weight_buffer,
            self.transpose_weight_buffer,
        )

    def reduce_grad(self, is_last_microbatch: bool = False):
        """
        Reduce gradients across DP ranks.

        ZeRO-2/3 reduce-scatter sharded grad buffers during backward.
        ZeRO-1 keeps grads replicated during backward and reduce-scatters
        the replicated buffer once when the optimizer syncs.
        """
        self._ensure_buffers_on_gpu()
        if self.main_grad_buffer is None:
            return

        reduce_inner = self.sharding_strategy in ("optim_grads", "optim_grads_params") or (
            is_last_microbatch and self.sharding_strategy in ("no_shard", "optim")
        )
        reduce_outer = self.outer_dp_sharding_strategy in ("optim_grads", "optim_grads_params") or (
            is_last_microbatch and self.outer_dp_sharding_strategy in ("no_shard", "optim")
        )
        if not reduce_inner and not reduce_outer:
            return

        if reduce_inner:
            # mesh dim 1 is inner-DP.
            self.main_grad_buffer.reduce_grad(
                grad_comm_dtype=self.mp_policy.grad_comm_dtype,
                overwrite_grad=self._grad_buffer_is_fresh,
                reduce_dim=1,
                reduce_scatter=self.sharding_strategy != "no_shard",
            )
            # _grad_buffer_is_fresh is False after reduce_grad() because the buffer was overwritten.
            self._grad_buffer_is_fresh = False
        if reduce_outer:
            # mesh dim 0 is outer-DP.
            self.main_grad_buffer.reduce_grad(
                grad_comm_dtype=self.mp_policy.grad_comm_dtype,
                overwrite_grad=True,
                reduce_dim=0,
                reduce_scatter=self.outer_dp_sharding_strategy != "no_shard",
            )
    def release_grad_buffer(self):
        """Release the main gradient buffer to free memory."""
        if self.main_grad_buffer is not None:
            # Drop weight.main_grad views that layers.py stores during gradient-accumulation-fusion
            # backward.  Those views keep _unsharded_buffer alive even after reshard() sets the
            # internal reference to None, causing the grad buffer to leak until the next backward.
            for param in self.params:
                if hasattr(param, 'main_grad'):
                    del param.main_grad
            self.main_grad_buffer.reshard()

    def _maybe_free_grad_data(self) -> None:
        """Drop ``main_grad_buffer.data`` if all params are zero-graded.

        After ``zero_grad()`` (or before the first backward), all
        ``dist_param.grad`` are ``None``, so the gradient buffer holds no
        meaningful data.  Free the backing tensor — ``_init_dist_grads``
        will re-allocate on the next ``reduce_grad``.
        """
        if self.main_grad_buffer is None or self.main_grad_buffer.data is None:
            return
        if any(
            [getattr(p, "grad", None) is not None for p in self.dist_params] +
            [getattr(p, "decoupled_grad", None) is not None for p in self.dist_params]
        ):
            return
        self.main_grad_buffer.data = None
        self.dist_grads = [None for _ in self.params]

    def _init_dist_params(self):
        """
        Initialize distributed parameter views (DTensors) into the buffers.

        Creates DTensor views of model weights and gradients based on sharding strategy:
        - "optim_grads_params": weights and grads sharded, full ZeRO-3
        - "optim_grads": grads sharded, weights replicated (ZeRO-2)
        - "optim": grads accumulate replicated, optimizer consumes reduced shards
        - "no_shard": replicated, no sharding (DDP-equivalent)
        """
        self.dist_params = []
        self.dist_grads = []  # placeholder, populated in _init_dist_grads
        s = self.sharding_strategy

        is_param_shard = s == "optim_grads_params"
        is_optim_shard = s != "no_shard"
        is_outer_optim_shard = self.outer_dp_sharding_strategy == "optim" and is_optim_shard
        if is_outer_optim_shard:
            setattr(self.mesh, "_shard_order", [1, 0])
        # Mesh layout is (outer, inner). Outer optim shards optimizer views on
        # both dimensions, with inner sharding applied before outer sharding.
        optim_placements = [
            Shard(dim=0) if is_outer_optim_shard else Replicate(),
            Shard(dim=0) if is_optim_shard else Replicate(),
        ]

        # Create parameter DTensor views
        for param in self.params:
            item_id = self.param_idx[param]
            if is_outer_optim_shard:
                buffer = self.main_weight_buffer or self.model_weight_buffer
                assert buffer is not None
                # shard_dims=(outer, inner): (1, 1) means both dimensions are sharded.
                data = buffer.get_item(item_id, shard_dims=(1, 1))
                if self.main_weight_buffer is not None:
                    param_shape = param.shape
                else:
                    param_shape = self.mp_policy.get_param_storage_shapes([param])[0]
            elif self.main_weight_buffer is not None:
                mbuf = self.main_weight_buffer
                # shard_dims=(outer, inner): (0, 1) is inner sharded; (0, 0) is full.
                data = mbuf.get_item(item_id, shard_dims=(0, 1) if is_optim_shard else (0, 0))
                param_shape = param.shape
            elif self.model_weight_buffer is not None:
                wbuf = self.model_weight_buffer
                # shard_dims=(outer, inner): (0, 1) is inner sharded; (0, 0) is full.
                data = wbuf.get_item(item_id, shard_dims=(0, 1) if is_param_shard else (0, 0))
                param_shape = self.mp_policy.get_param_storage_shapes([param])[0]
            else:
                data = param.data.detach()
                param_shape = param.shape

            dist_param = torch.nn.Parameter(
                make_uneven_dtensor(
                    data,
                    param_shape,
                    self.mesh,
                    optim_placements,
                    post_process_uneven=True,
                ),
                requires_grad=param.requires_grad,
            )
            # Mark as FSDP parameter for special handling
            setattr(param, "__fsdp_param__", True)
            setattr(dist_param, "__fsdp_param__", True)
            self.dist_params.append(dist_param)
            self.dist_grads.append(None)  # placeholder, will be set in _init_dist_grads

        # Update dist_param chunk metadata for checkpointing and debugging.
        for dist_param in self.dist_params:
            update_uneven_dtensor_chunk_metadata(dist_param)

    def _init_dist_grads(self) -> None:
        """Lazily allocate ``main_grad_buffer.data`` and rebuild ``dist_grads``.

        The buffer layout (``BufferIndex``, offsets, shard) was created in
        ``_init_buffers``; only the backing tensor is deferred.  Called from
        ``reduce_grad()`` on first use.  Uses ``torch.empty`` to avoid the
        zero-init cost; ``_grad_buffer_is_fresh`` is ``True`` after allocation
        so the first reduce-scatter *overwrites* (``local_grad_shard.copy_``)
        rather than accumulating — the uninitialized data is never read.
        Subsequent calls are no-ops.
        """
        gbuf = self.main_grad_buffer
        if gbuf is None or not self.requires_grad:
            return
        if gbuf.data is not None:
            return  # already initialised

        gbuf.init_data(torch.empty(gbuf.data_size, dtype=gbuf.dtype, device=self.device))

        # Rebuild dist_grads views — dist_params are unchanged
        s = self.sharding_strategy
        is_grad_shard = s != "no_shard"
        is_outer_optim_shard = self.outer_dp_sharding_strategy == "optim" and is_grad_shard
        placements = [
            Shard(dim=0) if is_outer_optim_shard else Replicate(),
            Shard(dim=0) if is_grad_shard else Replicate(),
        ]

        self.dist_grads = []
        for p in self.params:
            item_id = self.param_idx[p]
            # shard_dims=(outer, inner): (1, 1) outer+inner, (0, 1) inner, (0, 0) full.
            shard_dims = (1, 1) if is_outer_optim_shard else (0, 1) if is_grad_shard else (0, 0)
            grad_data = gbuf.get_item(item_id, shard_dims=shard_dims)
            if p.requires_grad:
                grad_dtensor = make_uneven_dtensor(
                    grad_data,
                    p.shape,
                    self.mesh,
                    placements,
                    post_process_uneven=True,
                )
                # NOTE: Do not materialize empty local grad shards in self.dist_grads.
                # Empty shards are semantically no-ops, but passing zero-numel DTensor
                # grads into fused multi-tensor optimizers such as TE FusedAdam can
                # break updates for neighboring non-empty shards.  We still construct
                # grad_dtensor above so every rank enters the metadata collective in
                # the same parameter order.
                self.dist_grads.append(grad_dtensor if grad_data.numel() > 0 else None)
            else:
                self.dist_grads.append(None)

        self._grad_buffer_is_fresh = True

    def _rebuild_dist_views(self) -> None:
        """In-place update ``dist_params._local_tensor`` / ``dist_grad._local_tensor``.

        Called after any buffer's ``self.data`` changes device (offload_to_cpu /
        auto-reload).  Updates the ``_local_tensor`` attribute inside existing
        DTensor objects so optimizer references remain valid.
        """
        s = self.sharding_strategy
        is_param_shard = s == "optim_grads_params"
        is_optim_shard = s != "no_shard"
        is_outer_optim_shard = self.outer_dp_sharding_strategy == "optim" and is_optim_shard

        for i, param in enumerate(self.params):
            dist_param = self.dist_params[i]
            if dist_param is not None:
                if is_outer_optim_shard:
                    buffer = self.main_weight_buffer or self.model_weight_buffer
                    if buffer is None:
                        continue
                    # shard_dims=(outer, inner): (1, 1) means both dimensions are sharded.
                    data = buffer.get_item(self.param_idx[param], shard_dims=(1, 1))
                elif self.main_weight_buffer is not None:
                    data = self.main_weight_buffer.get_item(
                        self.param_idx[param],
                        # shard_dims=(outer, inner): (0, 1) is inner sharded; (0, 0) is full.
                        shard_dims=(0, 1) if is_optim_shard else (0, 0),
                    )
                elif self.model_weight_buffer is not None:
                    data = self.model_weight_buffer.get_item(
                        self.param_idx[param],
                        # shard_dims=(outer, inner): (0, 1) is inner sharded; (0, 0) is full.
                        shard_dims=(0, 1) if is_param_shard else (0, 0),
                    )
                else:
                    continue
                object.__setattr__(dist_param._local_tensor, 'data', data)

        if self.main_grad_buffer is not None and self.main_grad_buffer.data is not None:
            is_grad_shard = is_optim_shard
            for i, param in enumerate(self.params):
                dist_grad = self.dist_grads[i]
                if dist_grad is not None:
                    # shard_dims=(outer, inner): (1, 1) outer+inner, (0, 1) inner, (0, 0) full.
                    shard_dims = (1, 1) if is_outer_optim_shard else (0, 1) if is_grad_shard else (0, 0)
                    grad_data = self.main_grad_buffer.get_item(
                        self.param_idx[param],
                        shard_dims=shard_dims,
                    )
                    object.__setattr__(dist_grad._local_tensor, 'data', grad_data)

    def _ensure_buffers_on_gpu(self) -> bool:
        """Auto-reload any buffer on CPU back to GPU.

        Returns True if any buffer was moved (views were rebuilt).
        """
        moved = False
        for buf in (self.model_weight_buffer, self.main_weight_buffer,
                    self.main_grad_buffer, self.transpose_weight_buffer):
            if buf is not None and buf._ensure_data_on_gpu():
                moved = True
        if moved:
            self._rebuild_dist_views()
        return moved

    def zero_grad(self, set_to_none: bool = True):
        """Zero the main gradient buffer and mark grads as zeroed."""
        if set_to_none:
            for dist_param in self.dist_params:
                if dist_param.grad is not None:
                    dist_param.grad = None
                if hasattr(dist_param, "decoupled_grad"):
                    dist_param.decoupled_grad = None
            self._maybe_free_grad_data()
        else:
            if self.main_grad_buffer is not None and self.main_grad_buffer.data is not None:
                self.main_grad_buffer.data.zero_()
        self._grad_buffer_is_fresh = True
