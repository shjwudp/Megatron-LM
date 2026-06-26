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
        self._sample_args: Dict[int, Tuple] = {}
        self._sample_kwargs: Dict[int, Dict[str, Any]] = {}
        self._modules_ordered: List[torch.nn.Module] = []

    # ---- called from hooks ------------------------------------------------

    def record_module(
        self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]
    ) -> None:
        """Record sample args for *module* during the first optimized forward."""
        if self._captured:
            return
        mid = id(module)
        if mid in self._sample_args:
            return

        sig = inspect.signature(module.forward)
        has_self = "self" in sig.parameters
        bound = (
            sig.bind(module, *args, **kwargs)
            if has_self
            else sig.bind(*args, **kwargs)
        )
        # Separate positional tensor args (→ sample_args) from the rest.
        # Non-tensor positional values and all kwargs go to sample_kwargs
        # so make_graphed_callables can reconstruct the call correctly.
        param_names = [
            n for n in sig.parameters
            if not (has_self and n == "self")
        ]
        # positional values map to the first len(bound.args) - (1 if has_self else 0) params
        pos_start = 1 if has_self else 0
        pos_values = list(bound.args[pos_start:])
        pos_names_mapped = param_names[:len(pos_values)]

        tensor_args = []
        for name, val in zip(pos_names_mapped, pos_values):
            if isinstance(val, torch.Tensor):
                tensor_args.append(val)
            else:
                bound.kwargs[name] = val

        sample_args = tuple(tensor_args)
        sample_kwargs = dict(bound.kwargs)

        self._sample_args[mid] = sample_args
        self._sample_kwargs[mid] = sample_kwargs
        self._modules_ordered.append(module)

        n_tensor = sum(1 for v in sample_args if isinstance(v, torch.Tensor))
        n_kw_tensor = sum(
            1 for v in sample_kwargs.values() if isinstance(v, torch.Tensor)
        )
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: recorded module %s (id=%s), "
                "%d args (%d tensor) + %d kwargs (%d tensor)",
                getattr(module, "_fsdp_module_name", module.__class__.__name__),
                id(module),
                len(sample_args), n_tensor,
                len(sample_kwargs), n_kw_tensor,
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

        sample_args_list: List[Tuple] = []
        sample_kwargs_list: List[Dict[str, Any]] = []
        capture_hooks: List[Dict] = []

        for m in modules:
            mid = id(m)
            # Clone tensor values so warmup gets fresh leaves without
            # residual autograd state from the first forward+backward.
            args = tuple(
                v.detach().clone().requires_grad_(v.requires_grad)
                if isinstance(v, torch.Tensor) else v
                for v in self._sample_args[mid]
            )
            kw = {
                k: v.detach().clone().requires_grad_(v.requires_grad)
                if isinstance(v, torch.Tensor) else v
                for k, v in self._sample_kwargs[mid].items()
            }
            sample_args_list.append(args)
            sample_kwargs_list.append(kw)

            capture_hooks.append({
                "forward_pre_hooks": {0: _make_fwd_pre_hook(m)},
                "forward_pre_hooks_with_kwargs": {0: True},
                "forward_hooks": {0: _make_fwd_post_hook(m)},
                "forward_hooks_with_kwargs": {0: True},
                "backward_pre_hooks": {0: _make_bwd_pre_hook(m)},
                "backward_hooks": {0: _make_bwd_post_hook(m)},
            })

        self._sample_args.clear()
        self._sample_kwargs.clear()

        # Pop real FSDP hooks so make_graphed_callables passes its assertion.
        # capture_time_hooks handle unshard/reshard during warmup + capture.
        saved_hooks = _pop_all_hooks(root_module)

        torch.cuda.reset_peak_memory_stats()
        _alloc_before = torch.cuda.memory_allocated()

        try:
            graphed = make_graphed_callables(
                tuple(modules),
                tuple(sample_args_list),
                num_warmup_iters=self._num_warmup,
                sample_kwargs=tuple(sample_kwargs_list),
                pool=self._graph_pool,
                capture_time_hooks=capture_hooks,
            )
        finally:
            _restore_all_hooks(saved_hooks)

        _alloc_after = torch.cuda.memory_allocated()
        _peak_alloc = torch.cuda.max_memory_allocated()

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: %d modules captured, "
                "alloc %+.1f MB (%d→%d)  peak_alloc %d MB",
                n,
                (_alloc_after - _alloc_before) / 1e6,
                _alloc_before // 1_000_000,
                _alloc_after // 1_000_000,
                _peak_alloc // 1_000_000,
            )

        if not isinstance(graphed, tuple):
            graphed = (graphed,)

        # make_graphed_callables already replaced module.forward with
        # the graphed version that handles kwargs natively.
        for module in modules:
            module._fsdp_cg_installed = True

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
