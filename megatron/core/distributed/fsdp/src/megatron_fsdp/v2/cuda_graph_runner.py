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

Built on top of ``torch.cuda.make_graphed_callables``.  A single
``CudaGraphRunner`` instance is stored on the root context and
orchestrates:

  1. Recording sample args for each eligible FSDP module during the
     first optimized forward pass.
  2. Calling ``make_graphed_callables`` once with all modules, in
     forward order, using a shared memory pool.

FSDP hooks are popped before capture and restored afterwards so they
fire correctly around the graphed forward during replay.
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


def _pop_all_hooks(
    module: torch.nn.Module,
) -> List[Tuple[torch.nn.Module, Dict[str, Any]]]:
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]] = []
    for sub in module.modules():
        snap: Dict[str, Any] = {}
        for attr in _HOOK_ATTRS:
            if hasattr(sub, attr):
                snap[attr] = getattr(sub, attr)
                setattr(sub, attr, OrderedDict())
        saved.append((sub, snap))
    return saved


def _restore_all_hooks(
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]],
) -> None:
    for sub, snap in saved:
        for name, value in snap.items():
            if value is not None:
                setattr(sub, name, value)


# ---------------------------------------------------------------------------
# Positional ↔ keyword-arg adapter
# ---------------------------------------------------------------------------


def _make_positional_shim(
    orig_forward: Any,
    tensor_param_names: List[str],
    frozen_kwargs: Dict[str, Any],
) -> Any:
    def shim_forward(*tensors):
        kw: Dict[str, Any] = dict(zip(tensor_param_names, tensors))
        kw.update(frozen_kwargs)
        return orig_forward(**kw)
    return shim_forward


def _install_keyword_wrapper(
    module: torch.nn.Module,
    graphed_forward: Any,
    tensor_param_names: List[str],
    orig_forward: Any,
) -> None:
    def wrapper(**kwargs):
        flat = tuple(kwargs[n] for n in tensor_param_names)
        return graphed_forward(*flat)
    try:
        wrapper.__signature__ = inspect.signature(orig_forward)
    except Exception:
        pass
    module._fsdp_cg_orig_forward = orig_forward
    module._fsdp_cg_installed = True
    module.forward = wrapper


def uninstall_cg(module: torch.nn.Module) -> None:
    orig = getattr(module, "_fsdp_cg_orig_forward", None)
    if orig is not None:
        module.forward = orig
        module._fsdp_cg_installed = False
        del module._fsdp_cg_orig_forward


# ---------------------------------------------------------------------------
# CudaGraphRunner
# ---------------------------------------------------------------------------


class CudaGraphRunner:
    """Orchestrates per-module sample-arg recording and batch graph capture.

    Created once by the root forward pre-hook and stored on
    ``ctx.cuda_graph_runner``.  No other state is stored on the context.

    Parameters
    ----------
    graph_pool:
        Shared ``torch.cuda.graph_pool_handle()``.
    num_warmup_iters:
        Warmup passes forwarded to ``make_graphed_callables``.
    """

    def __init__(
        self,
        graph_pool: Any,
        num_warmup_iters: int = 3,
    ):
        self._graph_pool = graph_pool
        self._num_warmup = num_warmup_iters
        self._captured = False

        # Per-module state recorded during the first optimized forward.
        self._sample_args: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self._tensor_names: Dict[int, List[str]] = {}
        self._frozen_kwargs: Dict[int, Dict[str, Any]] = {}
        self._modules_ordered: List[torch.nn.Module] = []

    # ---- called from hooks ------------------------------------------------

    def record_module(
        self,
        module: torch.nn.Module,
        args: Tuple,
        kwargs: Dict[str, Any],
    ) -> None:
        """Record sample args for *module* during the first optimized forward.

        Idempotent — calling twice for the same module is a no-op.
        """
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
        bound.apply_defaults()
        all_names = [
            n for n in sig.parameters
            if not (has_self and n == "self")
        ]
        tensor_names = [
            n for n in all_names
            if isinstance(bound.arguments[n], torch.Tensor)
        ]
        frozen_kwargs = {
            n: bound.arguments[n]
            for n in all_names
            if n not in tensor_names
        }
        flat_sample = tuple(bound.arguments[n] for n in tensor_names)

        self._sample_args[mid] = flat_sample
        self._tensor_names[mid] = tensor_names
        self._frozen_kwargs[mid] = frozen_kwargs
        self._modules_ordered.append(module)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: recorded module %s (id=%s), %d tensor args",
                getattr(module, "_fsdp_module_name", module.__class__.__name__),
                id(module),
                len(flat_sample),
            )

    def capture_and_install(self, root_module: torch.nn.Module) -> None:
        """Batch-capture graphs for all recorded modules and install wrappers.

        Must be called after the first optimized forward + backward
        completes (so ``plan()`` has transitioned the allocator).
        """
        if self._captured or not self._modules_ordered:
            return
        self._captured = True

        modules = self._modules_ordered
        sample_args_list = [self._sample_args[id(m)] for m in modules]
        tensor_names_list = [self._tensor_names[id(m)] for m in modules]
        frozen_kwargs_list = [self._frozen_kwargs[id(m)] for m in modules]

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: capturing %d modules", len(modules)
            )

        # 1. Pop all real hooks from the module tree.
        saved_hooks = _pop_all_hooks(root_module)

        # 2. Attach temporary unshard/reshard hooks so
        #    make_graphed_callables warmup + capture can run
        #    (params must be unsharded for forward, resharded after).
        for module in modules:
            _attach_temp_fsdp_hooks(module)

        try:
            # 3. Replace module.forwards with positional-arg shims.
            orig_forwards: List[Any] = []
            for module, names, frozen in zip(
                modules, tensor_names_list, frozen_kwargs_list
            ):
                orig = getattr(module, "_fsdp_cg_orig_forward", None) or module.forward
                orig_forwards.append(orig)
                module.forward = _make_positional_shim(orig, names, frozen)

            # 4. Capture all graphs in one call.
            graphed = torch.cuda.make_graphed_callables(
                tuple(modules),
                tuple(sample_args_list),
                num_warmup_iters=self._num_warmup,
                pool=self._graph_pool,
            )
            if not isinstance(graphed, tuple):
                graphed = (graphed,)

            # 5. Install keyword-arg wrappers over the graphed forwards.
            for module, g, names, orig in zip(
                modules, graphed, tensor_names_list, orig_forwards
            ):
                graphed_forward = module.forward
                _install_keyword_wrapper(module, graphed_forward, names, orig)

        finally:
            # 6. Drop temporary hooks, restore real FSDP hooks.
            #    During replay the real hooks handle unshard/reshard.
            _pop_all_hooks(root_module)  # discard temporary hooks
            _restore_all_hooks(saved_hooks)  # put real hooks back

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: installed CUDA graphs on %d modules",
                len(modules),
            )


# ---------------------------------------------------------------------------
# Temporary FSDP hooks (only active during make_graphed_callables capture)
# ---------------------------------------------------------------------------


def _attach_temp_fsdp_hooks(module: torch.nn.Module) -> None:
    """Attach minimal unshard / reshard hooks for *module*.

    These are temporary — they exist only during the
    ``make_graphed_callables`` capture window so warmup and capture
    have unsharded params.  After capture completes they are discarded
    and the real FSDP hooks are restored.
    """
    # pre-forward: unshard
    module.register_forward_pre_hook(_PreFwdUnshardHook(module))
    # post-forward: reshard
    module.register_forward_hook(_PostFwdReshardHook(module))
    # pre-backward: unshard (bwd_pass)
    module.register_full_backward_pre_hook(_PreBwdUnshardHook(module))
    # post-backward: reshard + reduce_grad
    module.register_full_backward_hook(_PostBwdReshardHook(module))


class _PreFwdUnshardHook:
    def __init__(self, module):
        self._module = module
    def __call__(self, mod, args, kwargs):
        self._module.unshard()
        return args, kwargs


class _PostFwdReshardHook:
    def __init__(self, module):
        self._module = module
    def __call__(self, mod, args, output):
        self._module.reshard()
        return output


class _PreBwdUnshardHook:
    def __init__(self, module):
        self._module = module
    def __call__(self, mod, grad_output):
        ctx = self._module._fsdp_root_context
        self._module.unshard(
            async_op=ctx.enable_unshard_prefetch, bwd_pass=True
        )


class _PostBwdReshardHook:
    def __init__(self, module):
        self._module = module
    def __call__(self, mod, grad_input, grad_output):
        self._module.reshard()
        if any(
            pg.sharding_strategy in ("optim_grads", "optim_grads_params")
            for pg in self._module._fsdp_param_groups
        ):
            self._module.reduce_grad()
        return grad_input
