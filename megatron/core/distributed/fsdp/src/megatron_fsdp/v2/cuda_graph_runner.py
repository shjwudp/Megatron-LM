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

"""CUDA graph capture / replay for individual FSDP modules."""

import inspect
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_forward_param_names(module: torch.nn.Module) -> List[str]:
    """Return the ordered parameter names of module.forward (excluding 'self')."""
    sig = inspect.signature(module.forward)
    return [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ]


class _ForwardShim(torch.nn.Module):
    """Wraps module.forward so that non-tensor kwargs are frozen at
    capture time and tensor inputs are passed positionally in signature
    order."""

    def __init__(
        self, module: torch.nn.Module, tensor_param_names: List[str], frozen_kwargs: dict
    ):
        super().__init__()
        self.module = module
        self.tensor_param_names = tensor_param_names
        self.frozen_kwargs = frozen_kwargs

    def forward(self, *flat_tensor_args):
        kwargs = dict(zip(self.tensor_param_names, flat_tensor_args))
        kwargs.update(self.frozen_kwargs)
        return self.module.forward(**kwargs)


def _pop_hooks(module: torch.nn.Module) -> Dict[str, Any]:
    """Remove all hooks from *module* (non-recursive) and return a snapshot."""
    saved: Dict[str, Any] = {
        "_forward_pre_hooks": module._forward_pre_hooks,
        "_forward_hooks": module._forward_hooks,
        "_backward_hooks": module._backward_hooks,
        "_state_dict_hooks": module._state_dict_hooks,
        "_load_state_dict_pre_hooks": module._load_state_dict_pre_hooks,
    }
    if hasattr(module, "_backward_pre_hooks"):
        saved["_backward_pre_hooks"] = module._backward_pre_hooks

    for name, value in saved.items():
        if value is not None:
            setattr(module, name, OrderedDict())

    return saved


def _restore_hooks(module: torch.nn.Module, saved: Dict[str, Any]) -> None:
    """Put the hooks back exactly as they were."""
    for name, value in saved.items():
        if value is not None:
            setattr(module, name, value)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


class FSDPCudaGraphRunner:
    """Captures a forward+bacwkard CUDA graph for one FSDP module.

    During capture hooks are temporarily removed so the graph records
    only the user's ``forward()``, not FSDP all-gather / reduce-scatter
    collectives.  FSDP side streams are disabled for the capture region.

    Usage::

        runner = FSDPCudaGraphRunner(my_fsdp_module)
        runner.capture_forward(sample_input)
        runner.install()                       # patches module.forward
        output = my_fsdp_module(input_batch)   # replays graph, no hooks
        runner.uninstall()                     # restore original behaviour
    """

    def __init__(self, fsdp_module: torch.nn.Module):
        self._module: torch.nn.Module = fsdp_module

        # Will hold the callable returned by make_graphed_callables
        self._graphed: Optional[torch._CudaGraphCallable] = None

        self._orig_fwd: Optional[Any] = None
        self._use_cuda_graph: bool = False
        self._captured: bool = False

        # Saved during capture for install() replay flattening
        self._tensor_param_names: List[str] = []
        self._frozen_kwargs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Capture
    # ------------------------------------------------------------------

    def capture_forward(
        self,
        *sample_args,
        **sample_kwargs,
    ) -> None:
        assert self._module.cuda_graph_compatible, (
            "CUDA graph capture requires enable TracePoolAllocator"
        )

        # Introspect the module's forward signature
        param_names = _get_forward_param_names(self._module.__class__)

        # Separate tensor vs non-tensor inputs
        bound = {}
        for i, val in enumerate(sample_args):
            if i < len(param_names):
                bound[param_names[i]] = val
        bound.update(sample_kwargs)

        tensor_names = [
            n for n in param_names if n in bound and isinstance(bound[n], torch.Tensor)
        ]
        frozen_kwargs = {n: v for n, v in bound.items() if not isinstance(v, torch.Tensor)}
        flat_sample = tuple(bound[n].clone().detach() for n in tensor_names)

        # Build shim
        shim = _ForwardShim(self._module, tensor_names, frozen_kwargs)

        # Disable side-stream collectives during capture so every CUDA
        # operation lands on the default (capture) stream.
        saved_prefetch = ctx.enable_unshard_prefetch
        saved_async_reduce = ctx.enable_async_reduce_grad
        ctx = self._module._fsdp_root_context
        ctx.enable_unshard_prefetch = False
        ctx.enable_async_reduce_grad = False
        ctx.cuda_graph_active = True
        saved_hooks = _pop_hooks(self._module)
        try:
            torch.cuda.synchronize()
            self._graphed = torch.cuda.make_graphed_callables(
                shim,
                sample_args=flat_sample,
                num_warmup_iters=3,
            )
        finally:
            _restore_hooks(self._module, saved_hooks)
            ctx.enable_unshard_prefetch = saved_prefetch
            ctx.enable_async_reduce_grad = saved_async_reduce
            ctx.cuda_graph_active = False

        self._tensor_param_names = tensor_names
        self._frozen_kwargs = frozen_kwargs
        self._captured = True

    # ------------------------------------------------------------------
    # 2. Install / uninstall the patched forward
    # ------------------------------------------------------------------
    def install(self) -> None:
        if not self._captured:
            raise RuntimeError("Call capture_forward() first")
        if self._orig_fwd is not None:
            return

        self._orig_fwd = self._module.forward
        graphed = self._graphed
        param_names = _get_forward_param_names(self._module.__class__)
        tensor_names = self._tensor_param_names

        def _patched_fwd(*args, **kwargs):
            if self._use_cuda_graph:
                bound = {}
                for i, val in enumerate(args):
                    if i < len(param_names):
                        bound[param_names[i]] = val
                bound.update(kwargs)
                flat = tuple(bound[n] for n in tensor_names)
                return graphed(*flat)
            return self._orig_fwd(*args, **kwargs)

        self._module.forward = _patched_fwd
        self._use_cuda_graph = True

    def uninstall(self) -> None:
        """Restore the original ``forward``."""
        if self._orig_fwd is None:
            return
        self._module.forward = self._orig_fwd
        self._orig_fwd = None
        self._use_cuda_graph = False

    # ------------------------------------------------------------------
    # 3. Properties
    # ------------------------------------------------------------------

    @property
    def captured(self) -> bool:
        """True if ``capture_forward`` has been called successfully."""
        return self._captured

    @property
    def using_cuda_graph(self) -> bool:
        """True if the patch is currently active (install() called)."""
        return self._use_cuda_graph

    def reset(self) -> None:
        """Uninstall the patch and allow a fresh capture later."""
        self.uninstall()
        self._graphed = None
        self._captured = False
