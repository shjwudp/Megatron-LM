from collections import defaultdict
from typing import Any, Dict, Iterable, TypeAlias, Union

import torch

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class HybridDeviceOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        offload_fraction=0.5,
        cpu_optimizer_cls=None,
        gpu_optimizer_cls=None,
        pin_cpu_grads: bool = True,
        param_update_in_fp32: bool = False,
        **kwargs
    ):
        super(HybridDeviceOptimizer, self).__init__(params, defaults={
            "offload_fraction": offload_fraction,
            "cpu_optimizer_cls": cpu_optimizer_cls,
            "gpu_optimizer_cls": gpu_optimizer_cls,
            "pin_cpu_grads": pin_cpu_grads,
            "param_update_in_fp32": param_update_in_fp32,
            **kwargs,
        })

        self.offload_fraction = offload_fraction
        self.cpu_optimizer_cls = cpu_optimizer_cls
        self.gpu_optimizer_cls = gpu_optimizer_cls
        self.pin_cpu_grads = pin_cpu_grads
        self.param_update_in_fp32 = param_update_in_fp32
        self.sub_optimizer_kwargs = kwargs

        self._init_sub_optimizers()
        self._register_state_dict_hooks()
        self._register_optimizer_step_hooks()

    def _set_sub_optimizer_grads(self):
        if self.param_update_in_fp32:
            for param in self.param_to_fp32_param:
                if param in self.gpu_params_map_cpu_copy:
                    # Skip if the param is offloaded to CPU, it should be handled
                    # in the following part.
                    continue
                fp32_param = self.param_to_fp32_param[param]
                if hasattr(param, "main_grad"):
                    fp32_param.grad = param.main_grad
                else:
                    fp32_param.grad = param.grad

        if self.cpu_optimizer is None:
            return

        # Sync the grads from GPU to CPU.
        self._data_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._data_stream):
            for group in self.cpu_optimizer.param_groups:
                for param in group["params"]:
                    gpu_param = self.cpu_copys_map_gpu_param[param]
                    if hasattr(gpu_param, "main_grad"):
                        grad = gpu_param.main_grad
                    elif hasattr(gpu_param, "grad"):
                        grad = gpu_param.grad
                    else:
                        grad = None
                        param.requires_grad = False
                        continue

                    param.requires_grad = True
                    if param not in self.cpu_copy_map_grad:
                        self.cpu_copy_map_grad[param] = torch.empty(
                            param.shape,
                            dtype=param.dtype,
                            pin_memory=self.pin_cpu_grads,
                            device="cpu",
                        )
                        param.grad = self.cpu_copy_map_grad[param]

                    self.cpu_copy_map_grad[param].data.copy_(
                        grad, non_blocking=True
                    )
            self._data_event = self._data_stream.record_event()

    def register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                self._data_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._data_stream):
                    for cpu_copy, gpu_param in self.cpu_copys_map_gpu_param.items():
                        gpu_param.data.copy_(cpu_copy.data)
                self._data_stream.record_event().wait(torch.cuda.current_stream())

            return param_copy_back_gpu_hook

        def fp32_param_copy_back_gpu_hook_closure():
            def fp32_param_copy_back_gpu_hook(optimizer, args, kwargs):
                for group in self.param_groups:
                    for param in group["params"]:
                        if param in self.gpu_params_map_cpu_copy:
                            # Skip if the param is offloaded to GPU, it has been
                            # copied back in the previous hook.
                            continue

                        if param in self.param_to_fp32_param:
                            fp32_param = self.param_to_fp32_param[param]
                            param.data.copy_(fp32_param.data)

            return fp32_param_copy_back_gpu_hook

        for optimizer in self.sub_optimizers:
            if isinstance(optimizer, self.cpu_optimizer_cls):
                optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())
            elif self.param_update_in_fp32 and isinstance(optimizer, self.gpu_optimizer_cls):
                optimizer.register_step_post_hook(fp32_param_copy_back_gpu_hook_closure())

    def step(self, closure=None):
        self._step_stream.wait_stream(torch.cuda.current_stream())
        if self.gpu_optimizer:
            with torch.cuda.stream(self._step_stream):
                self.gpu_optimizer.step(closure)
        self._step_stream.record_event().wait(torch.cuda.current_stream())
        if self._data_event is not None:
            self._data_event.synchronize()
            self._data_event = None
        if self.cpu_optimizer:
            self.cpu_optimizer.step(closure)

    def _init_sub_optimizers(self):
        (
            self.cpu_param_groups,
            self.gpu_param_groups,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
            self.param_to_fp32_param,
        ) = self._get_sub_optimizer_param_groups(self.offload_fraction)
        self.param_to_inner_param = {}
        self.inner_param_to_orig_param = {}
        for group in self.param_groups:
            for param in group["params"]:
                if param in self.param_to_fp32_param:
                    inner_param = self.param_to_fp32_param[param]
                elif param in self.gpu_params_map_cpu_copy:
                    inner_param = self.gpu_params_map_cpu_copy[param]
                else:
                    inner_param = param
                self.param_to_inner_param[param] = inner_param
                self.inner_param_to_orig_param[inner_param] = param

        self.sub_optimizers = []
        if len(self.cpu_param_groups) > 0:
            self.cpu_optimizer = self.cpu_optimizer_cls(self.cpu_param_groups)
            self.sub_optimizers.append(self.cpu_optimizer)
        else:
            self.cpu_optimizer = None
        if len(self.gpu_param_groups) > 0:
            self.gpu_optimizer = self.gpu_optimizer_cls(self.gpu_param_groups)
            self.sub_optimizers.append(self.gpu_optimizer)
        else:
            self.gpu_optimizer = None

        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)
        self._data_stream = torch.cuda.Stream()
        self._step_stream = torch.cuda.Stream()
        self._data_event: torch.cuda.Event = None

        self.register_param_copy_back_gpu_hook()

    def _get_sub_optimizer_param_groups(self, offload_fraction: float):
        params = []
        for group in self.param_groups:
            params.extend(group["params"])
        params_total_numel = sum([param.numel() for param in params])
        gpu_params_total_numel = sum([param.numel() for param in params if param.is_cuda])
        cpu_params_total_numel = params_total_numel - gpu_params_total_numel
        offload_threshold = gpu_params_total_numel * offload_fraction
        offload_params_numel = 0
        cpu_param_groups = []
        gpu_param_groups = []
        gpu_params_map_cpu_copy = {}
        cpu_copys_map_gpu_param = {}
        param_to_fp32_param = {}
        for group in self.param_groups:
            gpu_group = group.copy()
            cpu_group = group.copy()
            gpu_group["params"] = []
            cpu_group["params"] = []
            for param in group["params"]:
                orig_param = param
                cpu_copy = False
                if offload_params_numel < offload_threshold and param.is_cuda:
                    param = param.detach().clone().cpu().pin_memory()
                    offload_params_numel += param.numel()
                    cpu_copy = True
                if self.param_update_in_fp32 and param.dtype != torch.float32:
                    param = param.detach().clone().float()
                    self.state[orig_param]["fp32_param"] = param
                    param_to_fp32_param[orig_param] = param

                if cpu_copy:
                    gpu_params_map_cpu_copy[orig_param] = param
                    cpu_copys_map_gpu_param[param] = orig_param

                if param.is_cuda:
                    gpu_group["params"].append(param)
                else:
                    cpu_group["params"].append(param)
            if len(gpu_group["params"]) != 0:
                gpu_param_groups.append(gpu_group)
            if len(cpu_group["params"]) != 0:
                cpu_param_groups.append(cpu_group)

        return cpu_param_groups, gpu_param_groups, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param, param_to_fp32_param

    def _sync_sub_optimizers_state_to_hdo(self):
        """
        Update HDO state attribute to sub-optimizers.
        """

        # optimizer.state:
        # {
        #    torch.nn.Parameter: {
        #        str: Any,
        #    },
        #    ...
        # }
        new_state = defaultdict(dict)
        for optimizer in self.sub_optimizers:
            for param in optimizer.state:
                orig_param = self.inner_param_to_orig_param[param]
                new_state[orig_param] = optimizer.state[param]
                if self.param_update_in_fp32:
                    new_state[orig_param]["fp32_param"] = param
        self.state = new_state

    def _sync_hdo_state_to_sub_optimizers(self):
        for optimizer in self.sub_optimizers:
            new_state = defaultdict(dict)
            for group in optimizer.param_groups:
                for param in group["params"]:
                    orig_param = self.inner_param_to_orig_param[param]
                    new_state[param] = self.state[orig_param]
            optimizer.state = new_state
        self._update_fp32_params_by_new_state()

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        """Sync HDO new param_groups attribute (e.g. lr, wd, etc.) to sub-optimizers."""
        param_in_param_group_index = {}
        for i, group in enumerate(self.param_groups):
            for p_id, param in enumerate(group["params"]):
                inner_param = self.param_to_inner_param[param]
                param_in_param_group_index[inner_param] = (i, p_id)

        for optimizer in self.sub_optimizers:
            new_param_groups = []
            for group in optimizer.param_groups:
                new_group = group.copy()
                # After sync-up the sub-optimizer last update, we need to sync-up the
                # HDO new param_groups attributes to the sub-optimizer.
                assert len(group["params"]) > 0, "param_groups should not be empty"
                group_id, _ = param_in_param_group_index[group["params"][0]]
                update_group_attrs = self.param_groups[group_id].copy()
                del update_group_attrs["params"]
                new_group.update(update_group_attrs)

                new_param_groups.append(new_group)
            optimizer.param_groups = new_param_groups

    def _move_new_state_to_right_device(self):
        for optimizer in self.sub_optimizers:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    orig_param = self.inner_param_to_orig_param.get(k, k)
                    if isinstance(optimizer, self.defaults["cpu_optimizer_cls"]):
                        self.state[orig_param] = state[k] = v.to("cpu")
                    else:
                        self.state[orig_param] = state[k] = v.to("cuda")

    def _update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            fp32_param = self.param_to_fp32_param[param]
            fp32_param.data.copy_(v["fp32_param"])

    def _register_state_dict_hooks(self):
        def post_load_state_dict_hook(self):
            # After loading state_dict, the parameters may change, and we need to
            # reinitialize the sub-optimizers to regenerate the new parameters and
            # cpu copy pairs.
            self._init_sub_optimizers()
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()

        self.register_load_state_dict_post_hook(post_load_state_dict_hook)

    def _register_optimizer_step_hooks(self):
        def pre_step_hook(self, args, kwargs):
            # Sync param_groups to sub-optimizers before each step to make sure
            # the lr, wd, etc. are up-to-date.
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._set_sub_optimizer_grads()

        self.register_step_pre_hook(pre_step_hook)

        def post_step_hook(self, args, kwargs):
            # Sync state and param_groups to HDO after each step.
            # NOTE: It is possible for the optimizer to change the properties
            #   in param_groups.
            self._sync_sub_optimizers_state_to_hdo()

        self.register_step_post_hook(post_step_hook)

    def dummy_step(self):
        """
        The dummy step can be used to initialize the potential optimizer.state,
        which can solve the problem of checkpoint loading for an inplace operation
        such as loading a torch distributed checkpoint, for example.
        """
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = torch.randn_like(param)
        self.step()
        self.zero_grad()
