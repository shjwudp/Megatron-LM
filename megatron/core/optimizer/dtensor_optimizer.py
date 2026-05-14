# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""DTensor-aware adapter for raw optimizers.

The adapter keeps DTensor parameters as the public optimizer params while
feeding only local non-empty shard Parameters to the inner raw optimizer.
"""

from typing import Dict, List, Optional

import torch

from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes


def _to_local_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if hasattr(tensor, "_local_tensor"):
        return tensor._local_tensor
    data = getattr(tensor, "data", None)
    if data is not None and data is not tensor and hasattr(data, "_local_tensor"):
        return data._local_tensor
    return tensor


def _is_dtensor(tensor: torch.Tensor) -> bool:
    data = getattr(tensor, "data", None)
    return hasattr(tensor, "_local_tensor") or (
        data is not None and data is not tensor and hasattr(data, "_local_tensor")
    )


def _param_groups_have_dtensor(param_groups: List[dict]) -> bool:
    return any(_is_dtensor(param) for group in param_groups for param in group["params"])


def _optimizer_uses_decoupled_grad(optimizer) -> bool:
    defaults = getattr(optimizer, "defaults", {})
    return defaults.get("use_decoupled_grad", False) or any(
        group.get("use_decoupled_grad", False) for group in optimizer.param_groups
    )


class DTensorParamGroups:
    """DTensor param groups plus their optimizer-facing local shard groups."""

    def __init__(self, param_groups: List[dict]):
        self.global_param_groups = param_groups
        self.global_param_to_local_param: Dict[torch.Tensor, torch.nn.Parameter] = {}
        self.local_param_to_global_param: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self.local_param_groups = self._build_local_param_groups()

    def _local_param_for(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        if not _is_dtensor(param):
            return param

        local_param = self.global_param_to_local_param.get(param)
        if local_param is not None:
            return local_param

        local_tensor = _to_local_tensor(param)
        if local_tensor is None or local_tensor.numel() == 0:
            return None

        local_param = torch.nn.Parameter(local_tensor, requires_grad=param.requires_grad)
        copy_tensor_model_parallel_attributes(local_param, param)
        setattr(local_param, "_dtensor_global_param", param)
        self.global_param_to_local_param[param] = local_param
        self.local_param_to_global_param[local_param] = param
        return local_param

    def _build_local_param_groups(self) -> List[dict]:
        local_param_groups = []
        for group in self.global_param_groups:
            local_group = group.copy()
            local_params = []
            for param in group["params"]:
                local_param = self._local_param_for(param)
                if local_param is not None:
                    local_params.append(local_param)
            local_group["params"] = local_params
            local_param_groups.append(local_group)
        return local_param_groups


class DTensorOptimizerAdapter:
    """Optimizer-like wrapper that exposes DTensor params and steps local shards."""

    def __init__(self, inner_optimizer, dtensor_param_groups: DTensorParamGroups):
        self.inner_optimizer = inner_optimizer
        self.dtensor_param_groups = dtensor_param_groups
        self.use_decoupled_grad = _optimizer_uses_decoupled_grad(inner_optimizer)

    @staticmethod
    def prepare_param_groups(param_groups: List[dict]) -> Optional[DTensorParamGroups]:
        if not _param_groups_have_dtensor(param_groups):
            return None
        return DTensorParamGroups(param_groups)

    @property
    def global_param_groups(self):
        return self.dtensor_param_groups.global_param_groups

    @property
    def global_param_to_local_param(self):
        return self.dtensor_param_groups.global_param_to_local_param

    @property
    def local_param_to_global_param(self):
        return self.dtensor_param_groups.local_param_to_global_param

    @property
    def local_param_groups(self):
        return self.dtensor_param_groups.local_param_groups

    @property
    def param_groups(self):
        return self.global_param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.dtensor_param_groups = DTensorParamGroups(value)
        self.inner_optimizer.param_groups = self.local_param_groups

    @property
    def state(self):
        return self.inner_optimizer.state

    @state.setter
    def state(self, value):
        self.inner_optimizer.state = value

    @property
    def defaults(self):
        return self.inner_optimizer.defaults

    @property
    def master_weights(self):
        return getattr(self.inner_optimizer, "master_weights", None)

    @master_weights.setter
    def master_weights(self, value):
        setattr(self.inner_optimizer, "master_weights", value)

    def __getattr__(self, name):
        if name == "inner_optimizer":
            raise AttributeError(name)
        return getattr(self.inner_optimizer, name)

    def _sync_group_options(self) -> None:
        for global_group, local_group in zip(self.global_param_groups, self.local_param_groups):
            for key, value in global_group.items():
                if key != "params":
                    local_group[key] = value

    def _grad_for(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        if self.use_decoupled_grad:
            grad = getattr(param, "decoupled_grad", None)
            if grad is not None:
                return grad
        return param.grad

    @staticmethod
    def _clear_grad(param: torch.Tensor) -> None:
        param.grad = None
        if hasattr(param, "decoupled_grad"):
            param.decoupled_grad = None

    def _set_grad(self, param: torch.Tensor, grad: torch.Tensor) -> None:
        if self.use_decoupled_grad:
            param.decoupled_grad = grad
            param.grad = None
            return
        if hasattr(param, "decoupled_grad"):
            param.decoupled_grad = None
        param.grad = grad

    def _sync_local_grads(self) -> None:
        for local_param, global_param in self.local_param_to_global_param.items():
            local_grad = _to_local_tensor(self._grad_for(global_param))
            if local_grad is None:
                self._clear_grad(local_param)
                continue

            assert local_param.numel() == local_grad.numel(), (
                f"DTensor optimizer local param/grad numel mismatch: "
                f"{local_param.numel()} vs {local_grad.numel()} for "
                f"{getattr(global_param, '_fsdp_param_name', None)}"
            )
            self._set_grad(local_param, local_grad)

    def step(self, *args, **kwargs):
        self._sync_group_options()
        self._sync_local_grads()
        return self.inner_optimizer.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = True):
        return self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.inner_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.inner_optimizer.load_state_dict(state_dict)
