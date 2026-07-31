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

"""CUDA graph capture and replay for M-FSDP v2 modules.

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
"""  # noqa: E501

import contextlib
import gc
import inspect
import logging
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_map

from .dp_buffer import Placement

logger = logging.getLogger(__name__)

_CUDA_GRAPH_RUNTIME_ATTRS = (
    "backward_dw",
    "reset",
    "_cuda_graph_preflight",
    "_cuda_graph_set_replay_phase",
)


def _renew_fsdp_compute_parameter_leaves(
    modules: Tuple[torch.nn.Module, ...]
) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    """Create fresh compute leaves before recompute backward-graph capture.

    Each replacement shares the original CUDA storage but has a new Parameter
    identity and AccumulateGrad node. This keeps graph addresses stable while
    preventing eager-trace autograd state from entering the captured backward.

    """
    pending_gradients = []
    for module in modules:
        named_param_groups = getattr(module, "_named_param_groups", ())
        if not named_param_groups:
            continue
        for param_names, param_group in named_param_groups:
            dist_params = tuple(getattr(param_group, "dist_params", ()))
            if len(dist_params) != len(param_group.params):
                raise RuntimeError(
                    "CUDA graph capture requires one optimizer-facing distributed "
                    "parameter for each compute parameter"
                )
            for name, compute_parameter, dist_parameter in zip(
                param_names, param_group.params, dist_params
            ):
                registered_parameter = module.get_parameter(name)
                if registered_parameter is compute_parameter:
                    raise RuntimeError(
                        "CUDA graph capture must renew compute leaves after M-FSDP reshard"
                    )
                if registered_parameter is not dist_parameter:
                    raise RuntimeError(
                        "CUDA graph capture found an unexpected registered parameter identity"
                    )

            replacements = []
            for parameter in param_group.params:
                replacement = torch.nn.Parameter(
                    parameter.detach(), requires_grad=parameter.requires_grad
                )
                replacement.__dict__.update(parameter.__dict__)
                replacement.__dict__.pop("main_grad", None)
                replacements.append(replacement)
                if parameter.grad is not None:
                    pending_gradients.append((replacement, parameter.grad))

            param_group.params = replacements
            param_group.param_idx = {
                parameter: index for index, parameter in enumerate(replacements)
            }
            for buffer in (
                param_group.model_weight_buffer,
                param_group.transpose_weight_buffer,
                param_group.main_weight_buffer,
                param_group.main_grad_buffer,
            ):
                if buffer is not None:
                    buffer.params = replacements
                    buffer.param_idx = param_group.param_idx
        module._init_param_main_grad_func()
    return pending_gradients


def _restore_pending_compute_gradients(
    pending_gradients: List[Tuple[torch.nn.Parameter, torch.Tensor]]
) -> None:
    """Attach pre-capture gradients to replacement compute leaves."""
    for parameter, gradient in pending_gradients:
        parameter.grad = gradient
    pending_gradients.clear()


def _cuda_autocast_state() -> Tuple[bool, Optional[torch.dtype]]:
    """Return the CUDA autocast enabled flag and dtype.

    The autocast cache state is deliberately excluded: capture always pins
    ``cache_enabled=False``, so recording it would only reject captures for a
    difference that cannot affect the graphs.
    """
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return False, None
    try:
        dtype = torch.get_autocast_dtype("cuda")
    except AttributeError:
        dtype = torch.get_autocast_gpu_dtype()
    return True, dtype


def _normalize_forward_call(
    module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Rebuild a recorded forward call without nesting variadic arguments."""
    signature = inspect.signature(_get_cuda_graph_forward_impl(module))
    has_self = "self" in signature.parameters
    bound = signature.bind(module, *args, **kwargs) if has_self else signature.bind(*args, **kwargs)
    normalized_args = []
    normalized_kwargs = {}
    positional_remaining = len(args)
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL and bool(bound.arguments.get(name))
        for name, parameter in signature.parameters.items()
    )
    for name, parameter in signature.parameters.items():
        if has_self and name == "self":
            continue
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            normalized_args.append(value)
            positional_remaining -= 1
        elif (
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and has_varargs
            and positional_remaining
        ):
            normalized_args.append(value)
            positional_remaining -= 1
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            normalized_args.extend(value)
            positional_remaining = 0
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            normalized_kwargs.update(value)
        else:
            normalized_kwargs[name] = value
    return tuple(normalized_args), normalized_kwargs


def _requires_grad_surface(value: Any) -> Any:
    """Replace tensor leaves with their ``requires_grad`` flags."""
    return tree_map(
        lambda leaf: bool(leaf.requires_grad) if isinstance(leaf, torch.Tensor) else None, value
    )


def _tensor_storage_key(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Identify a tensor storage view."""
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.layout,
        tensor.device,
        tensor.is_conj(),
        tensor.is_neg(),
    )


def _is_direct_autograd_alias(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> bool:
    """Return whether an input is the producer output or its direct autograd view."""
    if input_tensor.numel() == 0 or output_tensor.numel() == 0:
        return False
    if input_tensor is output_tensor:
        return True
    if input_tensor.requires_grad != output_tensor.requires_grad:
        return False
    input_grad_fn = input_tensor.grad_fn
    output_grad_fn = output_tensor.grad_fn
    if input_grad_fn is None or output_grad_fn is None:
        return False
    return any(next_fn is output_grad_fn for next_fn, _ in input_grad_fn.next_functions)


def _validate_activation_recompute_lifetime(
    lifetime_events: List[Tuple[str, int]], module_count: int
) -> None:
    """Require one complete F, RF, and B sequence per captured module."""
    expected = [("forward", module_idx) for module_idx in range(module_count)]
    for module_idx in reversed(range(module_count)):
        expected.extend((("recompute", module_idx), ("backward", module_idx)))
    if lifetime_events != expected:
        raise RuntimeError(
            "Activation-recompute CUDA graphs require one captured module per "
            "checkpoint region and reverse F/RF/B execution order"
        )


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


def _get_cuda_graph_forward_impl(module: torch.nn.Module) -> Callable:
    """Return the replaceable forward behind the stable compile boundary."""
    return module.__dict__.get("_mfsdp_cuda_graph_forward_impl", module.forward)


def _set_cuda_graph_forward_impl(module: torch.nn.Module, forward: Callable) -> None:
    """Replace a forward without invalidating a compiled parent module."""
    if "_mfsdp_cuda_graph_forward_impl" in module.__dict__:
        module._mfsdp_cuda_graph_forward_impl = forward
    else:
        module.forward = forward


def _prepare_compiled_modules_for_capture(modules):
    """Convert ``Module.compile()`` modules to compiled forward bodies.

    ``nn.Module.compile()`` compiles ``Module._call_impl``, which includes
    module-hook dispatch.  FSDP removes those hooks and replaces them with
    ``capture_time_hooks`` while building its explicit CUDA graphs.  Keeping
    the compiled ``_call_impl`` can therefore trigger a guard failure and a
    lazy recompile inside CUDA stream capture.

    Compile the forward body instead, with Inductor CUDA graphs disabled so
    that the FSDP runner remains the sole CUDA-graph owner.  The returned state
    is only for rollback if explicit graph capture fails; after successful
    installation, the stale compiled ``_call_impl`` must remain disabled.
    """
    saved = []
    try:
        for module in modules:
            compiled_call_impl = getattr(module, "_compiled_call_impl", None)
            if compiled_call_impl is None:
                continue

            original_forward = module.forward
            saved.append((module, original_forward, compiled_call_impl))
            module._compiled_call_impl = None

            # Avoid wrapping a forward body that the user already compiled
            # directly.  This branch mainly handles ``module.compile()``.
            if not hasattr(original_forward, "_torchdynamo_orig_callable"):
                module.forward = torch.compile(
                    original_forward, dynamic=False, options={"triton.cudagraphs": False}
                )
    except Exception:
        _restore_compiled_modules_after_capture_failure(saved)
        raise
    return saved


def _restore_compiled_modules_after_capture_failure(saved):
    """Restore module-level compilation when explicit capture fails."""
    for module, original_forward, compiled_call_impl in saved:
        module.forward = original_forward
        module._compiled_call_impl = compiled_call_impl


def _build_input_output_aliases(
    modules: Tuple[torch.nn.Module, ...],
    sample_outputs: Dict[int, Any],
    sample_args: Dict[int, Tuple[Any, ...]],
    sample_kwargs: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Tuple[int, int]], ...]:
    """Match consumer inputs to an unambiguous earlier autograd output."""
    producer_outputs: Dict[Tuple[Any, ...], List[Tuple[int, int, torch.Tensor]]] = {}
    aliases_by_consumer = []
    for consumer_idx, module in enumerate(modules):
        mid = id(module)
        flat_args, _ = tree_flatten(sample_args[mid])
        flat_kwargs, _ = tree_flatten(list(sample_kwargs[mid].values()))
        aliases = {}
        for input_idx, input_tensor in enumerate(flat_args + flat_kwargs):
            if not isinstance(input_tensor, torch.Tensor) or input_tensor.numel() == 0:
                continue
            candidates = producer_outputs.get(_tensor_storage_key(input_tensor), ())
            producer = None
            exact_matches = [candidate for candidate in candidates if input_tensor is candidate[2]]
            if len(exact_matches) == 1:
                producer = exact_matches[0][:2]
            elif not exact_matches:
                direct_matches = [
                    candidate
                    for candidate in candidates
                    if _is_direct_autograd_alias(input_tensor, candidate[2])
                ]
                if len(direct_matches) == 1:
                    producer = direct_matches[0][:2]
            if producer is not None and producer[0] < consumer_idx:
                aliases[input_idx] = producer
        aliases_by_consumer.append(aliases)

        # Address reuse does not identify the autograd edge. Keep every
        # same-storage output from the latest producer, then link only one
        # unambiguous object or direct view.
        flat_outputs, _ = tree_flatten(sample_outputs.get(mid, ()))
        current_outputs = defaultdict(list)
        for output_idx, output in enumerate(flat_outputs):
            if isinstance(output, torch.Tensor) and output.numel() != 0:
                current_outputs[_tensor_storage_key(output)].append(
                    (consumer_idx, output_idx, output)
                )
        producer_outputs.update(current_outputs)
    return tuple(aliases_by_consumer)


class CudaGraphRunner:
    """Orchestrates per-module sample-arg recording and batch graph capture.

    Created once by the root forward pre-hook and stored on
    ``ctx.cuda_graph_runner``.
    """

    def __init__(
        self, graph_pool: Any, num_warmup_iters: int = 3, activation_recompute: bool = False
    ):
        if not isinstance(activation_recompute, bool):
            raise TypeError("activation_recompute must be a bool")
        self._graph_pool = graph_pool
        self._num_warmup = num_warmup_iters
        self._captured = False
        self._activation_recompute = activation_recompute

        # Per-module state recorded during the first optimized forward.
        self._sample_args: Dict[int, Tuple] = {}
        self._sample_kwargs: Dict[int, Dict[str, Any]] = {}
        self._sample_outputs: Dict[int, Any] = {}
        self._modules_ordered: List[torch.nn.Module] = []
        self._module_indices: Dict[int, int] = {}
        self._original_forwards: Dict[int, Callable] = {}
        self._original_graph_attrs: Dict[int, Dict[str, Any]] = {}
        self._compiled_module_state = []
        self._autocast_states: Dict[int, Tuple[bool, Optional[torch.dtype], bool]] = {}
        self._recompute_requires_grad: Dict[int, Tuple[Any, Any]] = {}
        self._lifetime_events: List[Tuple[str, int]] = []

    # ---- called from hooks ------------------------------------------------
    @property
    def captured(self) -> bool:
        """Return whether graph programs have been captured and installed."""
        return self._captured

    def preflight_record_module(self, module: torch.nn.Module) -> None:
        """Reject a second forward of a recorded module before its backward.

        Only called for normal training forwards (the hook gates on
        ``replay_phase == "forward"``).
        """
        if self._captured or not self._activation_recompute:
            return
        mid = id(module)
        if mid not in self._sample_args:
            return
        module_idx = self._module_indices[mid]
        if ("backward", module_idx) not in self._lifetime_events:
            raise RuntimeError(
                "Activation-recompute CUDA graphs require backward to finish "
                "before the next forward of the same module"
            )

    def record_module(self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]) -> None:
        """Record one module call during the first optimized forward."""
        if self._captured:
            return
        if self._activation_recompute and not torch.is_grad_enabled():
            raise RuntimeError(
                "M-FSDP CUDA Graph activation recompute currently supports only "
                "non-reentrant checkpointing"
            )
        mid = id(module)
        if mid in self._sample_args:
            return
        self._original_forwards[mid] = _get_cuda_graph_forward_impl(module)
        self._original_graph_attrs[mid] = {
            name: module.__dict__[name]
            for name in _CUDA_GRAPH_RUNTIME_ATTRS
            if name in module.__dict__
        }

        normalized_args, normalized_kwargs = _normalize_forward_call(module, args, kwargs)
        self._sample_args[mid] = normalized_args
        self._sample_kwargs[mid] = normalized_kwargs
        self._autocast_states[mid] = _cuda_autocast_state()
        module_idx = len(self._modules_ordered)
        self._module_indices[mid] = module_idx
        self._modules_ordered.append(module)
        self._lifetime_events.append(("forward", module_idx))
        flat_args, _ = tree_flatten(normalized_args)
        flat_kwargs, _ = tree_flatten(normalized_kwargs)
        n_tensor = sum(isinstance(value, torch.Tensor) for value in (*flat_args, *flat_kwargs))
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: recorded module %s (id=%s), %d kwargs (%d tensor)",
                getattr(module, "_fsdp_module_name", module.__class__.__name__),
                id(module),
                len(normalized_kwargs),
                n_tensor,
            )

    def record_module_recompute(
        self,
        module: torch.nn.Module,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one module call during checkpoint recomputation."""
        if self._captured or not self._activation_recompute:
            return
        mid = id(module)
        module_idx = self._module_indices.get(mid)
        if module_idx is None or ("recompute", module_idx) in self._lifetime_events:
            return
        recompute_requires_grad = None
        if args is not None:
            normalized_args, normalized_kwargs = _normalize_forward_call(module, args, kwargs or {})
            recompute_requires_grad = (
                _requires_grad_surface(normalized_args),
                _requires_grad_surface(normalized_kwargs),
            )
        if recompute_requires_grad is not None:
            self._recompute_requires_grad[mid] = recompute_requires_grad
        self._lifetime_events.append(("recompute", module_idx))

    def owns_module(self, module: torch.nn.Module) -> bool:
        """Return whether this runner recorded ``module`` for replay."""
        return id(module) in self._module_indices

    def prepare_module_replay(self, module: torch.nn.Module, replay_phase: str) -> None:
        """Select and validate the next F or RF replay."""
        if not self._activation_recompute or not self._captured or not self.owns_module(module):
            return
        if replay_phase not in ("forward", "recompute"):
            raise ValueError(f"Unknown CUDA graph replay phase {replay_phase!r}")
        setter = module.__dict__.get("_cuda_graph_set_replay_phase")
        if not callable(setter):
            raise RuntimeError("Captured activation-recompute module has no replay selector")
        setter(replay_phase)
        preflight = module.__dict__.get("_cuda_graph_preflight")
        if not callable(preflight):
            raise RuntimeError("Captured activation-recompute module has no replay preflight")
        preflight()

    def record_module_output(self, module: torch.nn.Module, output: Any) -> None:
        """Record an eager output for static graph linking."""
        mid = id(module)
        if self._captured:
            return
        if mid not in self._sample_args or mid in self._sample_outputs:
            return
        self._sample_outputs[mid] = output

    def reset(self) -> None:
        """Destroy captured graphs and restore the original module callables."""
        reset_function_ids = set()
        for module in self._modules_ordered:
            original_attrs = self._original_graph_attrs.get(id(module), {})
            graph_reset = module.__dict__.get("reset")
            if (
                self._captured
                and callable(graph_reset)
                and graph_reset is not original_attrs.get("reset")
                and id(graph_reset) not in reset_function_ids
            ):
                reset_function_ids.add(id(graph_reset))
                graph_reset()
            original_forward = self._original_forwards.get(id(module))
            if original_forward is not None:
                _set_cuda_graph_forward_impl(module, original_forward)
            for name in _CUDA_GRAPH_RUNTIME_ATTRS:
                if name in original_attrs:
                    setattr(module, name, original_attrs[name])
                else:
                    module.__dict__.pop(name, None)
            module.__dict__.pop("_fsdp_cg_installed", None)
            for param_group in getattr(module, "_fsdp_param_groups", ()):
                for param in param_group.params:
                    param.__dict__.pop("_mfsdp_recorded_te_wgrad", None)
                release_grad_storage = getattr(param_group, "_release_grad_storage_if_unused", None)
                if callable(release_grad_storage):
                    release_grad_storage()

        if not self._captured:
            _restore_compiled_modules_after_capture_failure(self._compiled_module_state)

        self._sample_args.clear()
        self._sample_kwargs.clear()
        self._sample_outputs.clear()
        self._modules_ordered.clear()
        self._module_indices.clear()
        self._original_forwards.clear()
        self._original_graph_attrs.clear()
        self._compiled_module_state.clear()
        self._autocast_states.clear()
        self._recompute_requires_grad.clear()
        self._lifetime_events.clear()
        self._captured = False

    def complete_module_backward(self, module: torch.nn.Module) -> bool:
        """Consume one backward event owned by this runner.

        Caller contract (mfsdp_post_backward_hook): only invoked for
        activation-recompute runners on modules this runner owns.
        """
        backward_prepared = bool(getattr(module, "_fsdp_pre_backward_done", False))
        backward_complete = bool(getattr(module, "post_backward_issued", False))

        if not self._captured:
            module_idx = self._module_indices[id(module)]
            if ("recompute", module_idx) not in self._lifetime_events:
                raise RuntimeError(
                    "Activation-recompute CUDA graphs did not observe checkpoint "
                    "recomputation before backward; use non-reentrant activation "
                    "checkpointing or disable cuda_graph_activation_recompute"
                )
            if ("backward", module_idx) in self._lifetime_events:
                return True
            self._lifetime_events.append(("backward", module_idx))
            return True
        if not backward_prepared or backward_complete:
            raise RuntimeError("M-FSDP backward completion arrived before backward preparation")
        return True

    def capture_and_install(
        self, root_module: torch.nn.Module, capture_stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """Capture all graphs + install wrappers on recorded modules."""
        if self._captured or not self._modules_ordered:
            return

        modules = tuple(self._modules_ordered)
        n = len(modules)
        autocast_states = {self._autocast_states[id(module)] for module in modules}
        if len(autocast_states) != 1:
            raise RuntimeError("CUDA graph capture requires one recorded CUDA autocast state")
        autocast_enabled, autocast_dtype = next(iter(autocast_states))
        activation_recompute = self._activation_recompute
        if activation_recompute:
            _validate_activation_recompute_lifetime(self._lifetime_events, len(modules))

        # Recording must finish before replacing Module.compile()'s callable.
        # Otherwise one checkpoint invocation can run F and RF through
        # different compiled specializations.
        self._compiled_module_state.extend(_prepare_compiled_modules_for_capture(modules))
        for module in modules:
            self._original_forwards[id(module)] = _get_cuda_graph_forward_impl(module)

        if activation_recompute:
            pending_compute_gradients = _renew_fsdp_compute_parameter_leaves(modules)
            gc.collect()
        else:
            pending_compute_gradients = []
        if activation_recompute:
            for module in modules:
                for param_group in module._fsdp_param_groups:
                    if any(param.grad is not None for param in param_group.dist_params):
                        _restore_pending_compute_gradients(pending_compute_gradients)
                        self.reset()
                        raise RuntimeError(
                            "Activation-recompute CUDA graph capture requires gradients "
                            "to be cleared before the next forward"
                        )

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: capturing %d modules", n)

        # Activation recompute requires the vendored three-program runtime.
        if activation_recompute:
            from .te_graph_runtime import make_graphed_callables
        else:
            try:
                from te_graph_runtime import make_graphed_callables
                from te_graph_runtime.graph import (
                    _MFSDP_CAPTURE_CAPABILITIES as _installed_mfsdp_capabilities,
                )
                from te_graph_runtime.graph import (
                    _get_compatible_main_grad_buffer as _installed_static_grad_support,
                )
                from te_graph_runtime.graph import (
                    _refresh_module_parameter_surface as _installed_parameter_refresh,
                )

                # Single source of truth: the vendored runtime declares what
                # M-FSDP requires; an installed package must match it.
                from .te_graph_runtime.graph import (
                    _MFSDP_CAPTURE_CAPABILITIES as required_capabilities,
                )

                if (
                    not all(
                        callable(helper)
                        for helper in (_installed_static_grad_support, _installed_parameter_refresh)
                    )
                    or not required_capabilities.issubset(_installed_mfsdp_capabilities)
                    or "use_main_grad" not in inspect.signature(make_graphed_callables).parameters
                ):
                    raise ImportError("Installed te-graph-runtime lacks M-FSDP CUDA graph support")
            except ImportError:
                from .te_graph_runtime import make_graphed_callables

        sample_args_list: List[Tuple] = []
        sample_kwargs_list: List[Dict[str, Any]] = []
        capture_hooks: List[Dict] = []

        input_output_aliases = _build_input_output_aliases(
            modules, self._sample_outputs, self._sample_args, self._sample_kwargs
        )

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: linked %d static input/output tensors",
                sum(len(aliases) for aliases in input_output_aliases),
            )

        for m in modules:
            capture_hooks.append(
                {
                    "forward_pre_hooks": {0: _make_fwd_pre_hook(m)},
                    "forward_pre_hooks_with_kwargs": {0: True},
                    "forward_hooks": {0: _make_fwd_post_hook(m)},
                    "forward_hooks_with_kwargs": {0: True},
                    "backward_pre_hooks": {
                        0: _make_bwd_pre_hook(m, activation_recompute=activation_recompute)
                    },
                    "backward_hooks": {0: _make_bwd_post_hook(m)},
                }
            )

        for m in modules:
            mid = id(m)
            recompute_requires_grad = self._recompute_requires_grad.get(mid)
            if activation_recompute and recompute_requires_grad is None:
                raise RuntimeError(
                    "Activation-recompute CUDA graph capture is missing RF input metadata"
                )
            args_requires_grad, kwargs_requires_grad = (
                recompute_requires_grad if recompute_requires_grad is not None else (None, None)
            )
            # Clone tensor values so warmup gets fresh leaves without
            # residual autograd state from the first forward+backward.
            args = _clone_capture_sample(self._sample_args[mid], args_requires_grad)
            kw = _clone_capture_sample(self._sample_kwargs[mid], kwargs_requires_grad)
            sample_args_list.append(args)
            sample_kwargs_list.append(kw)

        compiled_module_state = list(self._compiled_module_state)
        if compiled_module_state and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            logger.info(
                "CudaGraphRunner: converted %d Module.compile() wrappers to "
                "compiled forward bodies",
                len(compiled_module_state),
            )

        runtime_options = {}
        if activation_recompute:
            runtime_options["_activation_recompute"] = True
            runtime_options["_reuse_graph_input_output_buffers"] = True
        supports_input_output_aliases = (
            "_input_output_aliases" in inspect.signature(make_graphed_callables).parameters
        )
        if any(input_output_aliases):
            if not supports_input_output_aliases:
                from .te_graph_runtime import make_graphed_callables

                supports_input_output_aliases = True
            runtime_options["_input_output_aliases"] = tuple(input_output_aliases)

        # Pop real FSDP hooks so make_graphed_callables passes its assertion.
        # capture_time_hooks handle unshard/reshard during warmup + capture.
        saved_hooks = _pop_all_hooks(root_module)
        self._sample_args.clear()
        self._sample_kwargs.clear()
        self._sample_outputs.clear()
        gc.collect()

        try:
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(_restore_pending_compute_gradients, pending_compute_gradients)
                cleanup.callback(_restore_all_hooks, saved_hooks)

                autocast_kwargs = {"enabled": autocast_enabled, "cache_enabled": False}
                if autocast_enabled:
                    autocast_kwargs["dtype"] = autocast_dtype
                with torch.amp.autocast("cuda", **autocast_kwargs):
                    graphed = make_graphed_callables(
                        tuple(modules),
                        sample_args_list,
                        num_warmup_iters=self._num_warmup,
                        sample_kwargs=sample_kwargs_list,
                        pool=self._graph_pool,
                        capture_time_hooks=capture_hooks,
                        capture_stream=capture_stream,
                        use_main_grad=True,
                        **runtime_options,
                    )
        except Exception:
            for module in modules:
                try:
                    module.reshard()
                except Exception:
                    logger.exception("Failed to reshard after CUDA graph capture error")
            self.reset()
            raise
        self._captured = True

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: captured %d modules", n)

        if not isinstance(graphed, tuple):
            graphed = (graphed,)

        # make_graphed_callables already replaced module.forward with
        # the graphed version that handles kwargs natively.
        for module in modules:
            module._fsdp_cg_installed = True
        self._compiled_module_state = []

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: installed CUDA graphs on %d modules", n)


# ---------------------------------------------------------------------------
# capture_time_hooks (unshard / reshard outside graph, not replayed)
# ---------------------------------------------------------------------------


def _clone_capture_sample(value: Any, requires_grad_surface: Any = None) -> Any:
    """Clone tensor leaves using recompute-forward gradient metadata."""

    if requires_grad_surface is None:
        requires_grad_surface = _requires_grad_surface(value)
    flat_values, value_spec = tree_flatten(value)
    flat_requires_grad, requires_grad_spec = tree_flatten(requires_grad_surface)
    if requires_grad_spec != value_spec:
        raise RuntimeError("Recompute-forward input structure changed before CUDA graph capture")

    def clone_tensor(leaf, requires_grad):
        if not isinstance(leaf, torch.Tensor):
            return leaf
        return leaf.detach().clone().requires_grad_(requires_grad)

    cloned = tuple(
        clone_tensor(leaf, requires_grad)
        for leaf, requires_grad in zip(flat_values, flat_requires_grad)
    )
    return torch.utils._pytree.tree_unflatten(cloned, value_spec)


def _make_fwd_pre_hook(module):
    """Build the capture-time forward unshard hook."""

    def hook(mod, args, kwargs):
        module.unshard()

    return hook


def _make_fwd_post_hook(module):
    def hook(mod, args, kwargs, output):
        module.reshard()

    return hook


def _make_bwd_pre_hook(module, activation_recompute=False):
    """Build the capture-time backward unshard hook."""

    def hook(mod, grad_output):
        module.unshard(bwd_pass=True)
        if activation_recompute:
            module.unshard(async_op=False, bwd_pass=False)
        for param_group in module._fsdp_param_groups:
            has_fused_wgrad = any(
                getattr(param, "_mfsdp_recorded_te_wgrad", False) for param in param_group.params
            )
            overwrite_main_grad = param_group.sharding_strategy in (
                "optim_grads_params",
                "optim_grads",
            )
            for param in param_group.params:
                param.overwrite_main_grad = overwrite_main_grad
            if has_fused_wgrad and param_group.main_grad_buffer is not None:
                param_group._init_dist_grads()
                param_group.main_grad_buffer.fetch_buffer(
                    [Placement.REPLICATE, Placement.REPLICATE]
                )

    return hook


def _make_bwd_post_hook(module):
    def hook(mod, grad_input, grad_output):
        module.reshard()
        # Capture binds compatible parameter gradients directly to the full
        # main-grad buffer. The normal post-backward path releases that
        # temporary buffer after reducing it; capture does not run reduction,
        # so mirror the release here after the graph has recorded the address.
        # Otherwise each captured module leaves its TracePoolAllocator key
        # active and a later module collides with a slot whose traced lifetime
        # was non-overlapping.
        for param_group in module._fsdp_param_groups:
            for param in param_group.params:
                param.grad = None
            param_group.release_grad_buffer()

    return hook
