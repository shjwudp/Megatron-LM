import copy
from collections import defaultdict
from typing import Any, Dict, Iterable, Union, TypeAlias, List

import torch


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class HybridDeviceOptimizer(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        offload_fraction=0.5, 
        cpu_optimizer_cls=None, 
        gpu_optimizer_cls=None, 
        pin_cpu_grads: bool=True, 
        overlap: bool=False, 
        multi_streams: bool = True,
        **kwargs
    ):
        super(HybridDeviceOptimizer, self).__init__(params, defaults={
            "cpu_optimizer_cls": cpu_optimizer_cls,
            "gpu_optimizer_cls": gpu_optimizer_cls,
            "offload_fraction": offload_fraction,
            **kwargs,
        })
        assert not overlap or multi_streams, "Overlap CPU optimizers must be used with multi CUDA streams!"

        self.offload_fraction = offload_fraction
        self.pin_cpu_grads = pin_cpu_grads

        (
            self.cpu_params,
            self.gpu_params,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
        ) = self._split_parameters_updated_on_the_cpu_and_gpu(params, offload_fraction)

        if overlap and len(self.cpu_params) > 0:
            (
                self.cpu_optimizers, 
                self.param_optimizer_mapping, 
                self.n_params
            ) = self.build_cpu_optimizer_list(cpu_optimizer_cls, self.cpu_params, **kwargs)
        else:
            self.cpu_optimizers: List[torch.optim.Optimizer] = [cpu_optimizer_cls(self.cpu_params, **kwargs)] if len(self.cpu_params) > 0 else list()
            self.param_optimizer_mapping = lambda _: 0
            self.n_params = [len(self.cpu_params)]

        if len(self.gpu_params) > 0:
            self.gpu_optimizer = gpu_optimizer_cls(self.gpu_params, **kwargs)
        else:
            self.gpu_optimizer = None

        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)
        self._d2h_stream = torch.cuda.Stream() if multi_streams else torch.cuda.current_stream()
        self._h2d_stream = torch.cuda.Stream() if overlap else torch.cuda.current_stream()
        self._step_stream = torch.cuda.Stream() if multi_streams else torch.cuda.current_stream()
        self._cpu_optimizer_map_data_event = dict()

        self.register_grad_cpu_copy_hook()
        self.register_param_copy_back_gpu_hook()

    @staticmethod
    def build_cpu_optimizer_list(cpu_optimizer_cls, cpu_params: ParamsT, **kwargs):
        """Build several cpu optimizers to enable overlap. Currently we naively 
        assign each parameter to an individual optimizer.

        Args:
            cpu_optimizer_cls (Type[torch.optim.Optimizer]): A torch optimizer class
            cpu_params (List[torch.Tensor]): The CPU parameters Tensor list
        """
        cpu_optimizers = []
        param_optimizer_mapping = dict()
        n_params = []

        if len(cpu_params) == 0:
            return cpu_optimizers, param_optimizer_mapping, n_params
        
        if not isinstance(cpu_params[0], torch.Tensor):
            for group in cpu_params:
                group_defaults = group.copy()
                params = group_defaults.pop("params")
                if isinstance(params, torch.Tensor):
                    params = [params]
                for param in params:
                    param_optimizer_mapping[param] = len(cpu_optimizers)
                    _cpu_param_group = group_defaults.copy()
                    _cpu_param_group["params"] = [param]
                    cpu_optimizers.append(
                        cpu_optimizer_cls([_cpu_param_group], **kwargs)
                    )
                    n_params.append(1)
            return cpu_optimizers, param_optimizer_mapping, n_params

        for param in cpu_params:
            param_optimizer_mapping[param] = len(cpu_optimizers)
            cpu_optimizers.append(
                cpu_optimizer_cls([param], **kwargs)
            )
            n_params.append(1)
        return cpu_optimizers, param_optimizer_mapping, n_params

    def register_grad_cpu_copy_hook(self):
        def grad_cpu_copy_hook_closure():
            def _param_generator(cpu_optimizer):
                for group in cpu_optimizer.param_groups:
                    for param in group["params"]:
                        yield param

            def grad_cpu_copy_hook(optimizer, args, kwargs):
                self._d2h_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._d2h_stream):
                    for cpu_optimizer in self.cpu_optimizers:
                        for param in _param_generator(cpu_optimizer):
                            gpu_param = self.cpu_copys_map_gpu_param[param]
                            if param not in self.cpu_copy_map_grad:
                                self.cpu_copy_map_grad[param] = torch.empty(
                                    gpu_param.grad.shape,
                                    dtype=gpu_param.grad.dtype,
                                    pin_memory=self.pin_cpu_grads
                                )
                            if hasattr(gpu_param, "grad"):
                                self.cpu_copy_map_grad[param].data.copy_(gpu_param.grad, non_blocking=True)
                                param.grad = self.cpu_copy_map_grad[param]
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                        self._cpu_optimizer_map_data_event[cpu_optimizer] = self._d2h_stream.record_event()
            return grad_cpu_copy_hook
        self.register_step_pre_hook(grad_cpu_copy_hook_closure())

    def register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                self._h2d_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._h2d_stream):      
                    for cpu_copy, gpu_param in self.cpu_copys_map_gpu_param.items():
                        gpu_param.data.copy_(cpu_copy.data, non_blocking=True)
                self._d2h_stream.record_event().wait(
                    torch.cuda.current_stream()
                )
            return param_copy_back_gpu_hook

        for cpu_optimizer in self.cpu_optimizers:
            cpu_optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())

    def state_dict(self):
        # NOTE: Merge state_dicts of cpu and gpu optimizers
        state_dicts = [optimizer.state_dict() for optimizer in self.cpu_optimizers]
        if self.gpu_optimizer:
            state_dicts.append(self.gpu_optimizer.state_dict())

        merged_state_dict = {"state": {}, "param_groups": []}
        offset = 0
        for state_dict in state_dicts:
            new_offset = offset
            new_state = {}
            for k, v in state_dict["state"].items():
                new_state[k + offset] = v
            new_param_groups = copy.deepcopy(state_dict["param_groups"])
            for group in new_param_groups:
                group["params"] = [p + offset for p in group["params"]]
                new_offset = max(group["params"]) + 1

            merged_state_dict["state"].update(new_state)
            merged_state_dict["param_groups"].extend(new_param_groups)
            offset = new_offset

        return merged_state_dict

    def load_state_dict(self, state_dict):
        # NOTE: split state_dict into cpu and gpu optimizers
        optimizers = [*self.cpu_optimizers, self.gpu_optimizer]
        param_groups_offset = 0
        for optimizer in optimizers:
            num_param_groups = len(optimizer.state_dict()["param_groups"])
            param_groups = copy.deepcopy(state_dict["param_groups"][param_groups_offset : param_groups_offset + num_param_groups])
            param_id_offset = min([min(group["params"]) for group in param_groups])
            state = {}
            for group in param_groups:
                for p in group["params"]:
                    state[p - param_id_offset] = state_dict["state"]
                group["params"] = [p - param_id_offset for p in group["params"]]
            optimizer.load_state_dict({"state": state, "param_groups": param_groups})

    def step(self, closure=None):
        self._step_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._step_stream):
            self.gpu_optimizer.step(closure)
        self._step_stream.record_event().wait(torch.cuda.current_stream())
        for cpu_optimizer in self.cpu_optimizers:
            d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
            if d2h_event is not None:
                d2h_event.synchronize()
            cpu_optimizer.step(closure)

    def _split_parameters_updated_on_the_cpu_and_gpu(self, params: ParamsT, offload_fraction: float):
        if len(params) == 0:
            return [], [], {}, {}
        
        if not isinstance(params[0], torch.Tensor):
            param_groups = params
            params = []
            for group in param_groups:
                params.extend(group["params"])
        else:
            param_groups = None

        total_params_numel = sum([param.numel() for param in params])
        offload_threshold = total_params_numel * offload_fraction

        cpu_params = []
        gpu_params = []
        gpu_params_map_cpu_copy = {}
        cpu_copys_map_gpu_param = {}
        offloaded_params_numel = 0
        for param in params:
            if offloaded_params_numel < offload_threshold:
                assert param.is_cuda
                param_cpu_copy = param.detach().cpu().pin_memory()
                param_cpu_copy.requires_grad = True
                gpu_params_map_cpu_copy[param] = param_cpu_copy
                cpu_copys_map_gpu_param[param_cpu_copy] = param
                cpu_params.append(param_cpu_copy)
            else:
                gpu_params.append(param)

            offloaded_params_numel += param.numel()

        if param_groups:
            cpu_param_groups = []
            gpu_param_groups = []
            for group in param_groups:
                group_defaults = group.copy()
                del group_defaults["params"]
                _cpu_params = []
                _gpu_params = []
                for param in group["params"]:
                    if param in gpu_params_map_cpu_copy:
                        _cpu_params.append(gpu_params_map_cpu_copy[param])
                    else:
                        _gpu_params.append(param)
                if len(_cpu_params) > 0:
                    cpu_param_groups.append({"params": _cpu_params, **group_defaults})
                if len(_gpu_params) > 0:
                    gpu_param_groups.append({"params": _gpu_params, **group_defaults})

            return cpu_param_groups, gpu_param_groups, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param
        return cpu_params, gpu_params, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param
