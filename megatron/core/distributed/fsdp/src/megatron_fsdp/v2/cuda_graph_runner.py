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

Built on an inlined version of ``torch.cuda.make_graphed_callables``
with FSDP-aware unshard / reshard wrapping around warmup and capture.

A single ``CudaGraphRunner`` instance is stored on the root context and
orchestrates:

  1. Recording sample args for each eligible FSDP module during the
     first optimized forward pass.
  2. Capturing all forward + backward graphs in the correct order
     (fwds in forward-module order, bwds in reverse) using a shared
     memory pool, with manual unshard/reshard outside the graph region.

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


def _pop_all_hooks(module: torch.nn.Module) -> List[Tuple[torch.nn.Module, Dict[str, Any]]]:
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]] = []
    for sub in module.modules():
        snap: Dict[str, Any] = {}
        for attr in _HOOK_ATTRS:
            if hasattr(sub, attr):
                snap[attr] = getattr(sub, attr)
                setattr(sub, attr, OrderedDict())
        saved.append((sub, snap))
    return saved


def _restore_all_hooks(saved: List[Tuple[torch.nn.Module, Dict[str, Any]]]) -> None:
    for sub, snap in saved:
        for name, value in snap.items():
            if value is not None:
                setattr(sub, name, value)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _flatten_output(out: Any) -> Tuple[torch.Tensor, ...]:
    if isinstance(out, torch.Tensor):
        return (out,)
    return tuple(t for t in out if isinstance(t, torch.Tensor))


def _record_output_structure(out: Any) -> Tuple[bool, Optional[List[bool]]]:
    if isinstance(out, torch.Tensor):
        return False, None
    if isinstance(out, (tuple, list)):
        return True, [t is None for t in out]
    return False, None


def _unflatten_output(flat: Tuple[torch.Tensor, ...], is_tuple: bool, none_mask: Optional[List[bool]]) -> Any:
    if not is_tuple:
        return flat[0]
    if none_mask is None or not any(none_mask):
        return flat
    result = list(flat)
    for i, is_none in enumerate(none_mask):
        if is_none:
            result.insert(i, None)
    return tuple(result)


# ---------------------------------------------------------------------------
# Generator safe helper
# ---------------------------------------------------------------------------


def _ensure_generator_graph_safe(device: Optional[int] = None) -> torch.Generator:
    if device is None:
        device = torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state = gen.get_state()
    if hasattr(state, "is_inference") and state.is_inference():
        with torch.inference_mode(mode=False):
            gen.set_state(state.clone())
    return gen


# ---------------------------------------------------------------------------
# CudaGraphRunner
# ---------------------------------------------------------------------------


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
        self._sample_args: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self._tensor_names: Dict[int, List[str]] = {}
        self._frozen_kwargs: Dict[int, Dict[str, Any]] = {}
        self._modules_ordered: List[torch.nn.Module] = []

    # ---- called from hooks ------------------------------------------------

    def record_module(self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]) -> None:
        if self._captured:
            return
        mid = id(module)
        if mid in self._sample_args:
            return

        sig = inspect.signature(module.forward)
        has_self = "self" in sig.parameters
        bound = sig.bind(module, *args, **kwargs) if has_self else sig.bind(*args, **kwargs)
        bound.apply_defaults()
        all_names = [n for n in sig.parameters if not (has_self and n == "self")]
        tensor_names = [n for n in all_names if isinstance(bound.arguments[n], torch.Tensor)]
        frozen_kwargs = {n: bound.arguments[n] for n in all_names if n not in tensor_names}
        flat_sample = tuple(bound.arguments[n] for n in tensor_names)

        self._sample_args[mid] = flat_sample
        self._tensor_names[mid] = tensor_names
        self._frozen_kwargs[mid] = frozen_kwargs
        self._modules_ordered.append(module)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: recorded module %s (id=%s), %d tensor args",
                        getattr(module, "_fsdp_module_name", module.__class__.__name__),
                        id(module), len(flat_sample))

    def capture_and_install(self, root_module: torch.nn.Module) -> None:
        if self._captured or not self._modules_ordered:
            return
        self._captured = True

        modules = self._modules_ordered
        n = len(modules)
        sample_args_list = [self._sample_args[id(m)] for m in modules]
        tensor_names_list = [self._tensor_names[id(m)] for m in modules]
        frozen_kwargs_list = [self._frozen_kwargs[id(m)] for m in modules]

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: capturing %d modules", n)

        # 0. Clone sample args into fresh leaf tensors from the
        #    recorded originals, then drop the originals.
        sample_args_list = [
            tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in args)
            for args in sample_args_list
        ]
        self._sample_args.clear()
        self._tensor_names.clear()
        self._frozen_kwargs.clear()

        # 1. Pop all real hooks; restore them after capture.
        saved_hooks = _pop_all_hooks(root_module)

        try:
            # 2. Save original forwards for later restore.
            orig_forwards: List[Any] = []
            for module in modules:
                orig_forwards.append(module.forward)

            # 3. Build full input surfaces (sample_args + module params).
            per_callable_len_user_args = [len(a) for a in sample_args_list]
            per_callable_module_params = [tuple(m.parameters()) for m in modules]
            per_callable_static_input_surfaces = [
                sample_args_list[i] + per_callable_module_params[i] for i in range(n)
            ]

            # 4. Warmup on throwaway stream — must run with grad enabled
            #    because capture_and_install is called from an autograd
            #    engine callback which has grad disabled.
            torch.cuda.synchronize()
            with torch.cuda.stream(torch.cuda.Stream()), torch.enable_grad():
                for i, (module, sample_args, static_input_surface) in enumerate(
                    zip(modules, sample_args_list, per_callable_static_input_surfaces)
                ):
                    for _ in range(self._num_warmup):
                        _warmup_one(module, sample_args, static_input_surface,
                                    tensor_names_list[i], frozen_kwargs_list[i])
            torch.cuda.synchronize()

            # 5. Forward capture (all modules, forward order).
            fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(n)]
            per_module_static_outputs: List[Tuple[torch.Tensor, ...]] = []
            per_module_output_is_tuple: List[bool] = []
            per_module_output_none_mask: List[Optional[List[bool]]] = []

            for i, (module, sample_args) in enumerate(zip(modules, sample_args_list)):
                _register_generator_state(fwd_graphs[i])

                module.unshard()
                with torch.cuda.graph(fwd_graphs[i], pool=self._graph_pool), torch.enable_grad():
                    outputs = _run_module(module, sample_args, tensor_names_list[i], frozen_kwargs_list[i])
                module.reshard()

                # Record output structure from the captured outputs.
                flat = _flatten_output(outputs)
                is_tuple, none_mask = _record_output_structure(outputs)

                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    grad_count = sum(1 for o in flat if o.requires_grad)
                    logger.info(
                        "CudaGraphRunner: fwd capture module %s: %d outputs, %d require_grad",
                        getattr(module, "_fsdp_module_name", module.__class__.__name__),
                        len(flat), grad_count,
                    )

                per_module_static_outputs.append(tuple(flat))
                per_module_output_is_tuple.append(is_tuple)
                per_module_output_none_mask.append(none_mask)

            # 6. Backward capture (reverse order).
            bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(n)]
            per_module_static_grad_outputs: List[Tuple[Optional[torch.Tensor], ...]] = []
            per_module_static_grad_inputs: List[Tuple[Optional[torch.Tensor], ...]] = []

            rev_indices = list(reversed(range(n)))
            for ri, i in enumerate(rev_indices):
                module = modules[i]
                static_input_surface = per_callable_static_input_surfaces[i]
                static_outputs = per_module_static_outputs[i]
                bwd_graph = bwd_graphs[i]

                _register_generator_state(bwd_graph)

                static_grad_outs = tuple(
                    torch.empty_like(o) if o.requires_grad else None for o in static_outputs
                )
                outputs_grad = tuple(o for o in static_outputs if o.requires_grad)

                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    logger.info(
                        "CudaGraphRunner: bwd capture module %s: %d static_outputs, %d require_grad",
                        getattr(module, "_fsdp_module_name", module.__class__.__name__),
                        len(static_outputs), len(outputs_grad),
                    )

                module.unshard(bwd_pass=True)
                with torch.cuda.graph(bwd_graph, pool=self._graph_pool), torch.enable_grad():
                    if outputs_grad:
                        grad_ins = torch.autograd.grad(
                            outputs=outputs_grad,
                            inputs=tuple(a for a in static_input_surface if a.requires_grad),
                            grad_outputs=tuple(o for o in static_grad_outs if o is not None),
                            only_inputs=True,
                            allow_unused=True,
                        )
                    else:
                        grad_ins = None
                module.reshard()

                static_grad_inputs: List[Optional[torch.Tensor]] = []
                grad_idx = 0
                for arg in static_input_surface:
                    if arg.requires_grad and grad_ins is not None:
                        static_grad_inputs.append(grad_ins[grad_idx])
                        grad_idx += 1
                    else:
                        static_grad_inputs.append(None)

                per_module_static_grad_outputs.append(static_grad_outs)
                per_module_static_grad_inputs.append(tuple(static_grad_inputs))

            # Reverse the reverse-ordered lists.
            per_module_static_grad_outputs.reverse()
            per_module_static_grad_inputs.reverse()

            # 7. Install keyword wrappers with generated Graphed Functions.
            for i, module in enumerate(modules):
                graphed = _make_graphed_function(
                    fwd_graphs[i], bwd_graphs[i],
                    per_callable_module_params[i],
                    per_callable_len_user_args[i],
                    per_callable_static_input_surfaces[i],
                    per_module_static_outputs[i],
                    per_module_static_grad_outputs[i],
                    per_module_static_grad_inputs[i],
                    per_module_output_is_tuple[i],
                    per_module_output_none_mask[i],
                )
                _install_graphed_forward(module, graphed, tensor_names_list[i],
                                         orig_forwards[i])

        finally:
            _restore_all_hooks(saved_hooks)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: installed CUDA graphs on %d modules", n)


# ---------------------------------------------------------------------------
# Warmup helpers
# ---------------------------------------------------------------------------


def _run_module(module, sample_args, tensor_names, frozen_kwargs):
    kw = dict(zip(tensor_names, sample_args))
    kw.update(frozen_kwargs)
    return module.forward(**kw)


def _warmup_one(module, sample_args, static_input_surface, tensor_names, frozen_kwargs):
    module.unshard()
    outputs = _run_module(module, sample_args, tensor_names, frozen_kwargs)
    module.reshard()

    flat = _flatten_output(outputs)
    outputs_grad = tuple(o for o in flat if o.requires_grad)
    if outputs_grad:
        module.unshard(bwd_pass=True)
        torch.autograd.grad(
            outputs=outputs_grad,
            inputs=tuple(a for a in static_input_surface if a.requires_grad),
            grad_outputs=tuple(torch.empty_like(o) for o in outputs_grad),
            only_inputs=True,
            allow_unused=True,
        )
        module.reshard()


# ---------------------------------------------------------------------------
# Generator registration
# ---------------------------------------------------------------------------


def _register_generator_state(graph: torch.cuda.CUDAGraph) -> None:
    gen = _ensure_generator_graph_safe()
    graph.register_generator_state(gen)


# ---------------------------------------------------------------------------
# Graphed autograd Function + install
# ---------------------------------------------------------------------------


def _make_graphed_function(
    fwd_graph, bwd_graph,
    module_params, len_user_args,
    static_input_surface,
    static_outputs,
    static_grad_outputs,
    static_grad_inputs,
    output_is_tuple,
    output_none_mask,
):
    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            for i in range(len_user_args):
                if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                    static_input_surface[i].copy_(inputs[i])
            fwd_graph.replay()
            assert isinstance(static_outputs, tuple)
            return tuple(o.detach() for o in static_outputs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *grads):
            assert len(grads) == len(static_grad_outputs)
            for g, grad in zip(static_grad_outputs, grads):
                if g is not None and g.data_ptr() != grad.data_ptr():
                    g.copy_(grad)
            bwd_graph.replay()
            return tuple(b.detach() if b is not None else b for b in static_grad_inputs)

    def functionalized(*user_args):
        out = Graphed.apply(*(tuple(user_args) + module_params))
        return _unflatten_output(out, output_is_tuple, output_none_mask)

    return functionalized


def _install_graphed_forward(module, graphed, tensor_names, orig_forward):
    def wrapper(**kwargs):
        flat = tuple(kwargs[n] for n in tensor_names)
        return graphed(*flat)
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
