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
FSDP1-Compatible API backed by Megatron FSDP2.

.. warning::
    This module is **experimental** and subject to change without notice.
    It is provided for early evaluation and feedback. Do not use in
    production workloads.

This module provides a drop-in replacement for PyTorch's
``torch.distributed.fsdp.FullyShardedDataParallel`` that uses Megatron FSDP2's
``fully_shard()`` as the backend. It enables projects using FSDP1 (such as
Bagel, HuggingFace, etc.) to adopt Megatron's optimized FSDP with minimal
code changes.

Usage:
    Replace::

        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

    With::

        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp1_compat import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
"""

import logging
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from ..uneven_dtensor import uneven_dtensor_to_full_tensor
from .fsdp_module import FSDPModule
from .fully_shard import fully_shard
from .mixed_precision import FullyShardMixedPrecisionPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "FullyShardedDataParallel",
    "ShardingStrategy",
    "MixedPrecision",
    "BackwardPrefetch",
    "CPUOffload",
    "StateDictType",
    "FullStateDictConfig",
    "ShardedStateDictConfig",
    "LocalStateDictConfig",
]


class ShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


class BackwardPrefetch(Enum):
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


class StateDictType(Enum):
    FULL_STATE_DICT = auto()
    LOCAL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()


@dataclass
class MixedPrecision:
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    buffer_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True
    cast_root_forward_inputs: bool = True


@dataclass
class CPUOffload:
    offload_params: bool = False


@dataclass
class FullStateDictConfig:
    rank0_only: bool = False
    offload_to_cpu: bool = False


@dataclass
class ShardedStateDictConfig:
    offload_to_cpu: bool = False


@dataclass
class LocalStateDictConfig:
    offload_to_cpu: bool = False


_SHARDING_STRATEGY_MAP = {
    ShardingStrategy.FULL_SHARD: "optim_grads_params",
    ShardingStrategy.SHARD_GRAD_OP: "optim_grads",
    ShardingStrategy.NO_SHARD: "no_shard",
    ShardingStrategy.HYBRID_SHARD: "optim_grads_params",
    ShardingStrategy._HYBRID_SHARD_ZERO2: "optim_grads",
}


def _get_modules_to_wrap(
    root_module: nn.Module,
    auto_wrap_policy: Callable,
    ignored_modules: Optional[Set[nn.Module]] = None,
) -> List[nn.Module]:
    ignored_modules = ignored_modules or set()
    modules_to_wrap = []

    def _recurse(module: nn.Module):
        for child in module.children():
            if child in ignored_modules:
                continue
            _recurse(child)
            nonwrapped_numel = sum(
                p.numel() for p in child.parameters(recurse=True)
            )
            should_wrap = auto_wrap_policy(
                module=child,
                recurse=False,
                nonwrapped_numel=nonwrapped_numel,
            )
            if should_wrap:
                modules_to_wrap.append(child)

    _recurse(root_module)
    return modules_to_wrap


class FullyShardedDataParallel(nn.Module):
    """
    Drop-in replacement for ``torch.distributed.fsdp.FullyShardedDataParallel``
    backed by Megatron FSDP2's ``fully_shard()`` API.

    .. warning::
        This class is **experimental** and subject to breaking changes.

    This class accepts the same constructor arguments as PyTorch FSDP1 and
    internally uses Megatron FSDP2 for parameter sharding, communication
    overlap, and memory management.
    """

    _state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT
    _state_dict_config: Any = FullStateDictConfig()

    def __init__(
        self,
        module: nn.Module,
        auto_wrap_policy: Optional[Callable] = None,
        mixed_precision: Optional[MixedPrecision] = None,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        device_id: Optional[Union[int, torch.device]] = None,
        device_mesh: Optional[DeviceMesh] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        cpu_offload: Optional[CPUOffload] = None,
        ignored_modules: Optional[List[nn.Module]] = None,
        ignored_params: Optional[Set[nn.Parameter]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = True,
        param_init_fn: Optional[Callable] = None,
    ):
        super().__init__()

        warnings.warn(
            "FullyShardedDataParallel FSDP1-compat API is experimental and "
            "subject to breaking changes in future releases.",
            FutureWarning,
            stacklevel=2,
        )

        if cpu_offload is not None and cpu_offload.offload_params:
            warnings.warn(
                "CPU offload is not supported by Megatron FSDP2 backend. "
                "Ignoring cpu_offload=True.",
                stacklevel=2,
            )

        if not use_orig_params:
            warnings.warn(
                "use_orig_params=False is not supported by Megatron FSDP2. "
                "Parameters will always use DTensor views (equivalent to use_orig_params=True).",
                stacklevel=2,
            )

        if device_id is not None:
            if isinstance(device_id, int):
                torch.cuda.set_device(device_id)
            else:
                torch.cuda.set_device(device_id)
            module = module.to(torch.device("cuda", torch.cuda.current_device()))

        mesh = self._resolve_mesh(device_mesh, sharding_strategy)

        mp_policy = self._build_mp_policy(mixed_precision)

        megatron_sharding_strategy = _SHARDING_STRATEGY_MAP[sharding_strategy]

        ignored_modules_set = set(ignored_modules) if ignored_modules else set()
        ignored_params_set = set(ignored_params) if ignored_params else set()
        for mod in ignored_modules_set:
            ignored_params_set.update(mod.parameters())

        if auto_wrap_policy is not None:
            modules_to_wrap = _get_modules_to_wrap(
                module, auto_wrap_policy, ignored_modules_set
            )
            for submodule in modules_to_wrap:
                fully_shard(
                    submodule,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    ignored_params=ignored_params_set,
                    enable_unshard_prefetch=(
                        backward_prefetch is not None
                    ),
                )

        fully_shard(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            ignored_params=ignored_params_set,
            enable_unshard_prefetch=(backward_prefetch is not None),
        )

        self._fsdp_wrapped_module = module
        self._mesh = mesh
        self._sharding_strategy = sharding_strategy
        self._mixed_precision = mixed_precision

    def _resolve_mesh(
        self,
        device_mesh: Optional[DeviceMesh],
        sharding_strategy: ShardingStrategy,
    ) -> DeviceMesh:
        if device_mesh is not None:
            return device_mesh

        world_size = dist.get_world_size()

        if sharding_strategy in (
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        ):
            num_nodes = max(1, world_size // torch.cuda.device_count())
            gpus_per_node = torch.cuda.device_count()
            return init_device_mesh(
                "cuda",
                mesh_shape=(num_nodes, gpus_per_node),
                mesh_dim_names=("replicate", "shard"),
            )

        return init_device_mesh("cuda", mesh_shape=(world_size,))

    def _build_mp_policy(
        self,
        mixed_precision: Optional[MixedPrecision],
    ) -> Optional[FullyShardMixedPrecisionPolicy]:
        if mixed_precision is None:
            return None

        return FullyShardMixedPrecisionPolicy(
            grad_comm_dtype=mixed_precision.reduce_dtype,
        )

    def forward(self, *args, **kwargs):
        return self._fsdp_wrapped_module(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def named_parameters(self, prefix: str = "", recurse: bool = True, **kwargs):
        return self._fsdp_wrapped_module.named_parameters(
            prefix=prefix, recurse=recurse, **kwargs
        )

    def parameters(self, recurse: bool = True):
        return self._fsdp_wrapped_module.parameters(recurse=recurse)

    def named_modules(self, *args, **kwargs):
        return self._fsdp_wrapped_module.named_modules(*args, **kwargs)

    def modules(self):
        return self._fsdp_wrapped_module.modules()

    def named_buffers(self, prefix: str = "", recurse: bool = True, **kwargs):
        return self._fsdp_wrapped_module.named_buffers(
            prefix=prefix, recurse=recurse, **kwargs
        )

    def buffers(self, recurse: bool = True):
        return self._fsdp_wrapped_module.buffers(recurse=recurse)

    def train(self, mode: bool = True):
        self._fsdp_wrapped_module.train(mode)
        return self

    def eval(self):
        self._fsdp_wrapped_module.eval()
        return self

    @classmethod
    @contextmanager
    def state_dict_type(
        cls,
        module: "FullyShardedDataParallel",
        state_dict_type: StateDictType,
        state_dict_config: Optional[Any] = None,
        optim_state_dict_config: Optional[Any] = None,
    ):
        if state_dict_config is None:
            if state_dict_type == StateDictType.FULL_STATE_DICT:
                state_dict_config = FullStateDictConfig()
            elif state_dict_type == StateDictType.SHARDED_STATE_DICT:
                state_dict_config = ShardedStateDictConfig()
            else:
                state_dict_config = LocalStateDictConfig()

        old_type = module._state_dict_type
        old_config = module._state_dict_config
        module._state_dict_type = state_dict_type
        module._state_dict_config = state_dict_config
        try:
            yield
        finally:
            module._state_dict_type = old_type
            module._state_dict_config = old_config

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        sd_type = self._state_dict_type
        config = self._state_dict_config

        if sd_type == StateDictType.FULL_STATE_DICT:
            return self._full_state_dict(config)
        elif sd_type == StateDictType.LOCAL_STATE_DICT:
            return self._local_state_dict(config)
        elif sd_type == StateDictType.SHARDED_STATE_DICT:
            return self._sharded_state_dict(config)
        else:
            raise ValueError(f"Unknown state dict type: {sd_type}")

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True
    ):
        sd_type = self._state_dict_type
        config = self._state_dict_config

        if sd_type == StateDictType.FULL_STATE_DICT:
            return self._load_full_state_dict(state_dict, strict)
        elif sd_type == StateDictType.LOCAL_STATE_DICT:
            return self._load_local_state_dict(state_dict, strict)
        elif sd_type == StateDictType.SHARDED_STATE_DICT:
            return self._load_sharded_state_dict(state_dict, strict)
        else:
            raise ValueError(f"Unknown state dict type: {sd_type}")

    @torch.no_grad()
    def _full_state_dict(self, config: FullStateDictConfig) -> Dict[str, Any]:
        state_dict = OrderedDict()

        for name, param in self._fsdp_wrapped_module.named_parameters():
            if isinstance(param, DTensor):
                full_tensor = uneven_dtensor_to_full_tensor(param)
            else:
                full_tensor = param.data.clone()

            if config.offload_to_cpu:
                full_tensor = full_tensor.cpu()

            if config.rank0_only and dist.get_rank() != 0:
                continue
            state_dict[name] = full_tensor

        for name, buffer in self._fsdp_wrapped_module.named_buffers():
            if isinstance(buffer, DTensor):
                full_tensor = uneven_dtensor_to_full_tensor(buffer)
            else:
                full_tensor = buffer.data.clone()

            if config.offload_to_cpu:
                full_tensor = full_tensor.cpu()

            if config.rank0_only and dist.get_rank() != 0:
                continue
            state_dict[name] = full_tensor

        if config.rank0_only and dist.get_rank() != 0:
            return {}

        return state_dict

    @torch.no_grad()
    def _local_state_dict(self, config: LocalStateDictConfig) -> Dict[str, Any]:
        state_dict = OrderedDict()

        for name, param in self._fsdp_wrapped_module.named_parameters():
            if isinstance(param, DTensor):
                local_tensor = param._local_tensor.clone()
            else:
                local_tensor = param.data.clone()

            if config.offload_to_cpu:
                local_tensor = local_tensor.cpu()
            state_dict[name] = local_tensor

        for name, buffer in self._fsdp_wrapped_module.named_buffers():
            if isinstance(buffer, DTensor):
                local_tensor = buffer._local_tensor.clone()
            else:
                local_tensor = buffer.data.clone()

            if config.offload_to_cpu:
                local_tensor = local_tensor.cpu()
            state_dict[name] = local_tensor

        return state_dict

    @torch.no_grad()
    def _sharded_state_dict(self, config: ShardedStateDictConfig) -> Dict[str, Any]:
        state_dict = OrderedDict()

        for name, param in self._fsdp_wrapped_module.named_parameters():
            if config.offload_to_cpu and isinstance(param, DTensor):
                state_dict[name] = param.to("cpu")
            else:
                state_dict[name] = param

        for name, buffer in self._fsdp_wrapped_module.named_buffers():
            if config.offload_to_cpu and isinstance(buffer, DTensor):
                state_dict[name] = buffer.to("cpu")
            else:
                state_dict[name] = buffer

        return state_dict

    @torch.no_grad()
    def _load_full_state_dict(
        self, state_dict: Dict[str, Any], strict: bool
    ):
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        param_dict = dict(self._fsdp_wrapped_module.named_parameters())
        buffer_dict = dict(self._fsdp_wrapped_module.named_buffers())

        for name, param in param_dict.items():
            if name not in state_dict:
                if strict:
                    missing_keys.append(name)
                continue
            unexpected_keys.remove(name)
            full_tensor = state_dict[name]
            if isinstance(full_tensor, torch.Tensor):
                full_tensor = full_tensor.to(param.device)
            if isinstance(param, DTensor):
                mesh = param.device_mesh
                shard_group = mesh.get_group()
                rank = dist.get_rank(shard_group)
                world_size = dist.get_world_size(shard_group)
                chunk_size = (full_tensor.shape[0] + world_size - 1) // world_size
                start = rank * chunk_size
                end = min(start + chunk_size, full_tensor.shape[0])
                local_shard = full_tensor[start:end].contiguous()
                param._local_tensor.copy_(
                    local_shard[: param._local_tensor.shape[0]]
                )
            else:
                param.data.copy_(full_tensor)

        for name, buffer in buffer_dict.items():
            if name not in state_dict:
                if strict:
                    missing_keys.append(name)
                continue
            if name in unexpected_keys:
                unexpected_keys.remove(name)
            full_tensor = state_dict[name].to(buffer.device)
            if isinstance(buffer, DTensor):
                mesh = buffer.device_mesh
                shard_group = mesh.get_group()
                rank = dist.get_rank(shard_group)
                world_size = dist.get_world_size(shard_group)
                chunk_size = (full_tensor.shape[0] + world_size - 1) // world_size
                start = rank * chunk_size
                end = min(start + chunk_size, full_tensor.shape[0])
                local_shard = full_tensor[start:end].contiguous()
                buffer._local_tensor.copy_(
                    local_shard[: buffer._local_tensor.shape[0]]
                )
            else:
                buffer.data.copy_(full_tensor)

        if strict and (missing_keys or unexpected_keys):
            error_msg = ""
            if missing_keys:
                error_msg += f"Missing keys: {missing_keys}\n"
            if unexpected_keys:
                error_msg += f"Unexpected keys: {unexpected_keys}\n"
            raise RuntimeError(error_msg)

        return _IncompatibleKeys(missing_keys, unexpected_keys)

    @torch.no_grad()
    def _load_local_state_dict(
        self, state_dict: Dict[str, Any], strict: bool
    ):
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        for name, param in self._fsdp_wrapped_module.named_parameters():
            if name not in state_dict:
                if strict:
                    missing_keys.append(name)
                continue
            unexpected_keys.remove(name)
            local_tensor = state_dict[name].to(param.device)
            if isinstance(param, DTensor):
                param._local_tensor.copy_(local_tensor)
            else:
                param.data.copy_(local_tensor)

        for name, buffer in self._fsdp_wrapped_module.named_buffers():
            if name not in state_dict:
                if strict:
                    missing_keys.append(name)
                continue
            if name in unexpected_keys:
                unexpected_keys.remove(name)
            local_tensor = state_dict[name].to(buffer.device)
            if isinstance(buffer, DTensor):
                buffer._local_tensor.copy_(local_tensor)
            else:
                buffer.data.copy_(local_tensor)

        if strict and (missing_keys or unexpected_keys):
            error_msg = ""
            if missing_keys:
                error_msg += f"Missing keys: {missing_keys}\n"
            if unexpected_keys:
                error_msg += f"Unexpected keys: {unexpected_keys}\n"
            raise RuntimeError(error_msg)

        return _IncompatibleKeys(missing_keys, unexpected_keys)

    @torch.no_grad()
    def _load_sharded_state_dict(
        self, state_dict: Dict[str, Any], strict: bool
    ):
        return self._load_local_state_dict(state_dict, strict)

    @property
    def module(self) -> nn.Module:
        return self._fsdp_wrapped_module


@dataclass
class _IncompatibleKeys:
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
