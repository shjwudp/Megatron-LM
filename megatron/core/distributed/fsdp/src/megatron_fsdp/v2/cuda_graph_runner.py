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

"""CUDA graph capture / replay for individual FSDP v2 modules.

Built on ``te_graph_runtime.make_graphed_callables`` which supports
``capture_time_hooks`` — hooks that run outside CUDA graph capture (for
FSDP unshard / reshard) and are not replayed.  ``sample_kwargs`` is used
so modules receive keyword arguments natively.

A single ``CudaGraphRunner`` instance is stored on the root context and
orchestrates:

  1. Recording sample args for each eligible FSDP module during the
     first optimized forward pass.
  2. Calling ``make_graphed_callables`` with all modules and
     ``capture_time_hooks`` that perform unshard / reshard.
"""

import inspect
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook save / restore
# ---------------------------------------------------------------------------

_HOOK_ATTRS = [
    "_forward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]


def _pop_all_hooks(module):
    saved = []
    for sub in module.modules():
        snap = {}
        for attr in _HOOK_ATTRS:
            if hasattr(sub, attr):
                snap[attr] = getattr(sub, attr)
                setattr(sub, attr, OrderedDict())
        saved.append((sub, snap))
    return saved


def _restore_all_hooks(saved):
    for sub, snap in saved:
        for name, value in snap.items():
            if value is not None:
                setattr(sub, name, value)


class CudaGraphRunner:
    """Orchestrates per-module sample-arg recording and batch graph capture.

    Created once by the root forward pre-hook and stored on
    ``ctx.cuda_graph_runner``.
    """

    def __init__(self, graph_pool: Any, num_warmup_iters: int = 3):
        self._graph_pool = graph_pool
        self._num_warmup = num_warmup_iters
        self._captured = False

        # Per-module state recorded during the first optimized forward.
        self._sample_kwargs: Dict[int, Dict[str, Any]] = {}
        self._tensor_kwarg_names: Dict[int, List[str]] = {}
        self._frozen_kwargs: Dict[int, Dict[str, Any]] = {}
        self._modules_ordered: List[torch.nn.Module] = []

    # ---- called from hooks ------------------------------------------------

    def record_module(
        self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]
    ) -> None:
        """Record sample kwargs for *module* during the first optimized forward."""
        if self._captured:
            return
        mid = id(module)
        if mid in self._sample_kwargs:
            return

        sig = inspect.signature(module.forward)
        has_self = "self" in sig.parameters
        bound = (
            sig.bind(module, *args, **kwargs)
            if has_self
            else sig.bind(*args, **kwargs)
        )
        bound.apply_defaults()
        all_names = [n for n in sig.parameters if not (has_self and n == "self")]
        tensor_names = [
            n for n in all_names if isinstance(bound.arguments[n], torch.Tensor)
        ]
        tensor_kwargs = {n: bound.arguments[n] for n in tensor_names}
        frozen_kwargs = {
            n: bound.arguments[n]
            for n in all_names
            if n not in tensor_names
        }

        self._sample_kwargs[mid] = tensor_kwargs
        self._tensor_kwarg_names[mid] = tensor_names
        self._frozen_kwargs[mid] = frozen_kwargs
        self._modules_ordered.append(module)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: recorded module %s (id=%s), %d tensor / %d frozen kwargs",
                getattr(module, "_fsdp_module_name", module.__class__.__name__),
                id(module),
                len(tensor_names),
                len(frozen_kwargs),
            )

    def capture_and_install(self, root_module: torch.nn.Module) -> None:
        """Capture all graphs + install wrappers on recorded modules."""
        if self._captured or not self._modules_ordered:
            return
        self._captured = True

        modules = self._modules_ordered
        n = len(modules)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: capturing %d modules", n)

        from te_graph_runtime import make_graphed_callables

        sample_kwargs_list: List[Dict[str, Any]] = []
        capture_hooks: List[Dict] = []

        for m in modules:
            mid = id(m)
            sample_kwargs_list.append(self._sample_kwargs[mid])

            capture_hooks.append({
                "forward_pre_hooks": {0: _make_fwd_pre_hook(m)},
                "forward_pre_hooks_with_kwargs": {0: True},
                "forward_hooks": {0: _make_fwd_post_hook(m)},
                "forward_hooks_with_kwargs": {0: True},
                "backward_pre_hooks": {0: _make_bwd_pre_hook(m)},
                "backward_hooks": {0: _make_bwd_post_hook(m)},
            })

        # Pop real FSDP hooks so make_graphed_callables passes its assertion.
        # capture_time_hooks handle unshard/reshard during warmup + capture.
        saved_hooks = _pop_all_hooks(root_module)

        try:
            graphed = make_graphed_callables(
                tuple(modules),
                tuple(() for _ in range(n)),
                num_warmup_iters=self._num_warmup,
                sample_kwargs=tuple(sample_kwargs_list),
                pool=self._graph_pool,
                capture_time_hooks=capture_hooks,
            )
        finally:
            _restore_all_hooks(saved_hooks)

        if not isinstance(graphed, tuple):
            graphed = (graphed,)

        for module, g, names in zip(modules, graphed, self._tensor_kwarg_names.values()):
            mid = id(module)
            _install_cg(module, g, names, self._frozen_kwargs.get(mid, {}))

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: installed CUDA graphs on %d modules", n)


# ---------------------------------------------------------------------------
# capture_time_hooks (unshard / reshard outside graph, not replayed)
# ---------------------------------------------------------------------------


def _make_fwd_pre_hook(module):
    def hook(mod, args, kwargs):
        module.unshard()
    return hook


def _make_fwd_post_hook(module):
    def hook(mod, args, kwargs, output):
        module.reshard()
    return hook


def _make_bwd_pre_hook(module):
    def hook(mod, grad_output):
        module.unshard(bwd_pass=True)
    return hook


def _make_bwd_post_hook(module):
    def hook(mod, grad_input, grad_output):
        module.reshard()
        # if any(
        #     pg.sharding_strategy in ("optim_grads", "optim_grads_params")
        #     for pg in module._fsdp_param_groups
        # ):
        #     module.reduce_grad()
    return hook


# ---------------------------------------------------------------------------
# Install / uninstall keyword wrappers
# ---------------------------------------------------------------------------


def _install_cg(
    module: torch.nn.Module,
    graphed_callable: Any,
    tensor_kwarg_names: List[str],
    frozen_kwargs: Dict[str, Any],
) -> None:
    orig_forward = module.forward

    def wrapper(**kwargs):
        cg_kwargs = {n: kwargs[n] for n in tensor_kwarg_names}
        cg_kwargs.update(frozen_kwargs)
        return graphed_callable(**cg_kwargs)

    try:
        wrapper.__signature__ = inspect.signature(orig_forward)
    except Exception:
        pass

    module._fsdp_cg_orig_forward = orig_forward
    module._fsdp_cg_installed = True
    module.forward = wrapper


def uninstall_cg(module: torch.nn.Module) -> None:
    """Restore the original ``module.forward``."""
    orig = getattr(module, "_fsdp_cg_orig_forward", None)
    if orig is not None:
        module.forward = orig
        module._fsdp_cg_installed = False
        del module._fsdp_cg_orig_forward
