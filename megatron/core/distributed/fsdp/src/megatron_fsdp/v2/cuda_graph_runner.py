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
import dataclasses
import gc
import inspect
import logging
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.checkpoint import _StopRecomputationError

logger = logging.getLogger(__name__)

_CUDA_GRAPH_RUNTIME_ATTRS = (
    "backward_dw",
    "reset",
    "_cuda_graph_preflight",
    "_cuda_graph_release_pending",
    "_cuda_graph_set_replay_phase",
)


_MISSING_CAPTURE_ATTRIBUTE = object()


@dataclasses.dataclass
class _CaptureMutableState:
    """Snapshot Python and tensor state mutated by graph warmup and capture."""

    buffer_values: List[Tuple[torch.Tensor, torch.Tensor]]
    parameter_states: List[Tuple[torch.nn.Parameter, Any, Dict[str, Any]]]
    param_group_states: List[Tuple[Any, Any, Any, Optional[torch.Tensor], Any, Any, Any, Any]]

    @torch.no_grad()
    def restore(self) -> None:
        """Restore state in place and release all one-time backup tensors."""
        for buffer, value in self.buffer_values:
            buffer.copy_(value)
        for param, grad, attributes in self.parameter_states:
            param.grad = grad
            for name, value in attributes.items():
                if value is _MISSING_CAPTURE_ATTRIBUTE:
                    param.__dict__.pop(name, None)
                else:
                    setattr(param, name, value)
        for (
            param_group,
            grad_buffer,
            data,
            data_value,
            dist_grads,
            full_grad_accumulated,
            reduced_grad_accumulated,
            has_unreduced_data,
        ) in self.param_group_states:
            if grad_buffer is not None:
                grad_buffer.reshard()
                grad_buffer.data = data
                if data_value is not None:
                    data.copy_(data_value)
            if dist_grads is _MISSING_CAPTURE_ATTRIBUTE:
                param_group.__dict__.pop("dist_grads", None)
            else:
                param_group.dist_grads = dist_grads
            if full_grad_accumulated is _MISSING_CAPTURE_ATTRIBUTE:
                param_group.__dict__.pop("_full_grad_buffer_has_accumulated_grad", None)
            else:
                param_group._full_grad_buffer_has_accumulated_grad = full_grad_accumulated
            if reduced_grad_accumulated is _MISSING_CAPTURE_ATTRIBUTE:
                param_group.__dict__.pop("_reduced_grad_buffer_has_accumulated_grad", None)
            else:
                param_group._reduced_grad_buffer_has_accumulated_grad = reduced_grad_accumulated
            if has_unreduced_data is _MISSING_CAPTURE_ATTRIBUTE:
                param_group.__dict__.pop("_main_grad_buffer_has_unreduced_data", None)
            else:
                param_group._main_grad_buffer_has_unreduced_data = has_unreduced_data
        self.buffer_values.clear()
        self.parameter_states.clear()
        self.param_group_states.clear()


@dataclasses.dataclass
class _RecordedInvocation:
    """One original-forward occurrence recorded for ordered capture."""

    module: torch.nn.Module
    args: Tuple
    kwargs: Dict[str, Any]
    lane_index: int = 0
    region_index: int = -1
    recompute_requires_grad: Optional[Tuple[Any, Any]] = None
    output: Any = None
    recomputed: bool = False
    backward_done: bool = False


@dataclasses.dataclass(frozen=True)
class _BackwardInvocationToken:
    """Graph invocations reached from one root-module output."""

    epoch: int
    invocations: Tuple[Tuple[int, int], ...]


@dataclasses.dataclass(frozen=True)
class _OrderedCapturePlan:
    """Canonical custom-order inputs derived from recorded checkpoint regions."""

    modules: Tuple[torch.nn.Module, ...]
    invocations: Tuple[_RecordedInvocation, ...]
    order: Tuple[int, ...]
    order_slots: Tuple[int, ...]
    num_layers_per_chunk: Tuple[int, ...]
    module_regions: Tuple[int, ...]
    replay_events: Tuple[Tuple[str, int, int], ...]


def _snapshot_capture_mutable_state(modules: Tuple[torch.nn.Module, ...]) -> _CaptureMutableState:
    """Save state that warmup or capture may update before the real optimizer step.

    :param modules: FSDP modules about to be captured.
    :type modules: Tuple[torch.nn.Module, ...]
    :return: Restorable capture transaction state.
    :rtype: _CaptureMutableState
    :raises RuntimeError: If a distributed main-grad buffer is already unsharded.
    """
    buffer_values = []
    seen_buffers = set()
    for module in modules:
        for buffer in module.buffers():
            if id(buffer) in seen_buffers:
                continue
            seen_buffers.add(id(buffer))
            buffer_values.append((buffer, buffer.detach().clone()))

    parameter_states = []
    param_group_states = []
    seen_params = set()
    seen_param_groups = set()
    for module in modules:
        for param_group in module._fsdp_param_groups:
            if id(param_group) not in seen_param_groups:
                seen_param_groups.add(id(param_group))
                grad_buffer = getattr(param_group, "main_grad_buffer", None)
                grad_buffer_is_sharded = (
                    grad_buffer is not None and grad_buffer.storage_shard_layout != (0, 0)
                )
                if grad_buffer_is_sharded and grad_buffer._unsharded_buffer is not None:
                    raise RuntimeError(
                        "CUDA graph capture requires distributed main-grad buffers "
                        "to be resharded"
                    )
                data = grad_buffer.data if grad_buffer is not None else None
                data_value = (
                    data.detach().to(device="cpu", copy=True)
                    if data is not None and not grad_buffer_is_sharded
                    else None
                )
                param_group_states.append(
                    (
                        param_group,
                        grad_buffer,
                        data,
                        data_value,
                        getattr(param_group, "dist_grads", _MISSING_CAPTURE_ATTRIBUTE),
                        getattr(
                            param_group,
                            "_full_grad_buffer_has_accumulated_grad",
                            _MISSING_CAPTURE_ATTRIBUTE,
                        ),
                        getattr(
                            param_group,
                            "_reduced_grad_buffer_has_accumulated_grad",
                            _MISSING_CAPTURE_ATTRIBUTE,
                        ),
                        getattr(
                            param_group,
                            "_main_grad_buffer_has_unreduced_data",
                            _MISSING_CAPTURE_ATTRIBUTE,
                        ),
                    )
                )
            for param in (
                *getattr(param_group, "params", ()),
                *getattr(param_group, "dist_params", ()),
            ):
                if id(param) in seen_params:
                    continue
                seen_params.add(id(param))
                attributes = {
                    name: param.__dict__.get(name, _MISSING_CAPTURE_ATTRIBUTE)
                    for name in (
                        "main_grad",
                        "grad_added_to_main_grad",
                        "overwrite_main_grad",
                        "_mfsdp_recorded_te_wgrad",
                    )
                }
                parameter_states.append((param, param.grad, attributes))
    return _CaptureMutableState(buffer_values, parameter_states, param_group_states)


def _renew_fsdp_compute_parameter_leaves(
    modules: Tuple[torch.nn.Module, ...]
) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    """Create fresh compute leaves before recompute backward-graph capture.

    Each replacement shares the original CUDA storage but has a new Parameter
    identity and AccumulateGrad node. This keeps graph addresses stable while
    preventing eager-trace autograd state from entering the captured backward.

    :param modules: M-FSDP modules selected for three-graph capture.
    :type modules: Tuple[torch.nn.Module, ...]
    :return: Pending gradients to restore onto the replacement leaves.
    :rtype: List[Tuple[torch.nn.Parameter, torch.Tensor]]
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
    """Attach pre-capture gradients to replacement compute leaves.

    :param pending_gradients: Replacement leaf and its pending gradient.
    :type pending_gradients: List[Tuple[torch.nn.Parameter, torch.Tensor]]
    """
    for parameter, gradient in pending_gradients:
        parameter.grad = gradient
    pending_gradients.clear()


def _cuda_autocast_state() -> Tuple[bool, Optional[torch.dtype], bool]:
    """Return the current CUDA autocast state.

    :return: Autocast enabled flag, active dtype, and cache setting.
    :rtype: Tuple[bool, Optional[torch.dtype], bool]
    """
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
    cache_enabled = torch.is_autocast_cache_enabled()
    if not enabled:
        return False, None, cache_enabled
    try:
        dtype = torch.get_autocast_dtype("cuda")
    except AttributeError:
        dtype = torch.get_autocast_gpu_dtype()
    return True, dtype, cache_enabled


def _normalize_forward_call(
    module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Rebuild a recorded forward call without nesting variadic arguments.

    :param module: Module whose bound forward signature is recorded.
    :type module: torch.nn.Module
    :param args: Original positional arguments.
    :type args: Tuple[Any, ...]
    :param kwargs: Original keyword arguments.
    :type kwargs: Dict[str, Any]
    :return: Positional and keyword arguments preserving ``*args`` and ``**kwargs`` semantics.
    :rtype: Tuple[Tuple[Any, ...], Dict[str, Any]]
    """
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
    """Replace tensor leaves with their ``requires_grad`` flags.

    :param value: Input PyTree.
    :type value: Any
    :return: PyTree containing booleans for tensor leaves.
    :rtype: Any
    """
    return tree_map(
        lambda leaf: bool(leaf.requires_grad) if isinstance(leaf, torch.Tensor) else None, value
    )


def _capture_module_topology(module: torch.nn.Module) -> Tuple[Tuple[Any, ...], ...]:
    """Capture direct buffer and child-module slots for every module owner.

    :param module: Root module whose recursive owner set is captured.
    :type module: torch.nn.Module
    :return: Qualified owner names, owner references, buffer keys, and child identities.
    :rtype: Tuple[Tuple[Any, ...], ...]
    """
    return tuple(
        (
            module_name,
            owner,
            tuple(owner._buffers),
            tuple(
                (child_name, id(child) if child is not None else None)
                for child_name, child in owner._modules.items()
            ),
        )
        for module_name, owner in module.named_modules(remove_duplicate=False)
    )


def _make_module_topology_preflight(
    topology: Tuple[Tuple[Any, ...], ...], delegate: Optional[Callable] = None
) -> Callable[[], None]:
    """Build a replay check without a recursive module walk.

    :param topology: Module-owner topology captured before CUDA graph capture.
    :type topology: Tuple[Tuple[Any, ...], ...]
    :param delegate: Existing runtime preflight callback, if any.
    :type delegate: Optional[Callable]
    :return: Callback that rejects buffer-slot or child replacement changes.
    :rtype: Callable[[], None]
    """

    def preflight() -> None:
        for module_name, owner, expected_buffer_keys, expected_children in topology:
            owner_name = module_name or "<root>"
            if tuple(owner._buffers) != expected_buffer_keys:
                raise RuntimeError(
                    "CUDA graph registered buffer topology changed after capture "
                    f"at module {owner_name!r}"
                )
            current_children = tuple(
                (child_name, id(child) if child is not None else None)
                for child_name, child in owner._modules.items()
            )
            if current_children != expected_children:
                raise RuntimeError(
                    "CUDA graph child module topology changed after capture "
                    f"at module {owner_name!r}"
                )
        if callable(delegate):
            delegate()

    return preflight


def _tensor_storage_key(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Identify a tensor storage view.

    :param tensor: Tensor to identify.
    :type tensor: torch.Tensor
    :return: Storage address and view metadata.
    :rtype: Tuple[Any, ...]
    """
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
    """Return whether an input is the producer output or its direct autograd view.

    :param input_tensor: Consumer tensor to classify.
    :type input_tensor: torch.Tensor
    :param output_tensor: Earlier module output sharing the same storage view.
    :type output_tensor: torch.Tensor
    :return: Whether reconnecting their dgrad surfaces preserves the traced edge.
    :rtype: bool
    """
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


def _infer_activation_recompute_regions(
    lifetime_events: List[Tuple[str, int, int]],
    invocation_count: int,
    *,
    require_reverse_regions: bool,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[str, int], ...]]:
    """Infer serial checkpoint regions from the observed F, RF, and B order."""
    forward_events = [invocation for phase, invocation, _ in lifetime_events if phase == "forward"]
    if forward_events != list(range(invocation_count)):
        raise RuntimeError(
            "Activation-recompute CUDA graph forwards must be recorded once in execution order"
        )

    backward_regions = []
    cursor = 0
    while cursor < len(lifetime_events):
        if lifetime_events[cursor][0] == "forward":
            cursor += 1
            continue
        recompute = []
        while cursor < len(lifetime_events) and lifetime_events[cursor][0] == "recompute":
            recompute.append(lifetime_events[cursor][1])
            cursor += 1
        if not recompute:
            raise RuntimeError(
                "Activation-recompute CUDA graph schedule must start each region with RF"
            )
        backward = []
        while cursor < len(lifetime_events) and lifetime_events[cursor][0] == "backward":
            backward.append(lifetime_events[cursor][1])
            cursor += 1
        if backward != list(reversed(recompute)):
            raise RuntimeError(
                "Activation-recompute CUDA graph region must execute RF in forward order "
                "and B in reverse order"
            )
        backward_regions.append(tuple(recompute))

    flattened = [invocation for region in backward_regions for invocation in region]
    if sorted(flattened) != list(range(invocation_count)):
        raise RuntimeError(
            "Activation-recompute CUDA graph schedule must recompute and backpropagate "
            "every recorded forward exactly once"
        )
    for region in backward_regions:
        if tuple(range(region[0], region[0] + len(region))) != region:
            raise RuntimeError(
                "Activation-recompute CUDA graph modules in one checkpoint region "
                "must be contiguous in forward order"
            )
    forward_regions = sorted(backward_regions, key=lambda region: region[0])
    expected_forward_start = 0
    for region in forward_regions:
        if region[0] != expected_forward_start:
            raise RuntimeError("Activation-recompute CUDA graph checkpoint regions overlap")
        expected_forward_start += len(region)
    if require_reverse_regions and backward_regions != list(reversed(forward_regions)):
        raise RuntimeError(
            "Activation-recompute CUDA graph checkpoint regions must run backward "
            "in reverse forward order"
        )

    regions = [-1] * invocation_count
    for region_idx, region in enumerate(forward_regions):
        for invocation in region:
            regions[invocation] = region_idx

    invocation_regions = tuple(regions)
    region_events = []
    seen_forward_regions = set()
    seen_backward_regions = set()
    cursor = 0
    while cursor < len(lifetime_events):
        phase, invocation, _ = lifetime_events[cursor]
        region_idx = invocation_regions[invocation]
        if phase == "forward":
            if region_idx in seen_forward_regions:
                raise RuntimeError(
                    "Activation-recompute CUDA graph checkpoint region resumed after "
                    "another region"
                )
            region = forward_regions[region_idx]
            observed = tuple(
                event_invocation
                for event_phase, event_invocation, _ in lifetime_events[
                    cursor : cursor + len(region)
                ]
                if event_phase == "forward"
            )
            if observed != region:
                raise RuntimeError(
                    "Activation-recompute CUDA graph checkpoint region forwards must "
                    "execute contiguously"
                )
            seen_forward_regions.add(region_idx)
            region_events.append(("forward", region_idx))
            cursor += len(region)
            continue
        if phase == "recompute":
            if region_idx in seen_backward_regions:
                raise RuntimeError(
                    "Activation-recompute CUDA graph checkpoint region backward ran twice"
                )
            seen_backward_regions.add(region_idx)
            region_events.append(("backward", region_idx))
            region = forward_regions[region_idx]
            cursor += len(region) * 2
            continue
        raise RuntimeError(
            "Activation-recompute CUDA graph schedule contains an unexpected backward event"
        )

    if seen_forward_regions != set(range(len(forward_regions))) or seen_backward_regions != set(
        range(len(forward_regions))
    ):
        raise RuntimeError("Activation-recompute CUDA graph checkpoint region is incomplete")
    return tuple(regions), tuple(region_events)


def _validate_activation_recompute_lifetime(
    lifetime_events: List[Tuple[str, int, int]], module_regions: Tuple[int, ...]
) -> None:
    """Require complete serial checkpoint-region lifetimes."""
    inferred_regions, _ = _infer_activation_recompute_regions(
        lifetime_events, len(module_regions), require_reverse_regions=True
    )
    if inferred_regions != module_regions:
        raise RuntimeError("Activation-recompute CUDA graph checkpoint region grouping changed")


# ---------------------------------------------------------------------------
# NVML memory helper (real GPU memory, not just torch allocator view)
# ---------------------------------------------------------------------------


def _nvml_device_memory(device: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Return (used_MiB, total_MiB) from NVML, or None if unavailable."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        if device is None:
            device = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (info.used // (1024 * 1024), info.total // (1024 * 1024))
    except Exception:
        return None


def _mem_snapshot() -> Dict[str, int]:
    """Capture a snapshot of memory counters across torch and NVML."""
    snap = {
        "torch_alloc": torch.cuda.memory_allocated() // 1_000_000,
        "torch_reserved": torch.cuda.memory_reserved() // 1_000_000,
    }
    nvml = _nvml_device_memory()
    if nvml is not None:
        snap["nvml_used"] = nvml[0]
        snap["nvml_total"] = nvml[1]
    return snap


def _fmt_mem_snapshot(before: Dict[str, int], after: Dict[str, int], peak_alloc: int) -> str:
    """Format memory diff as a human-readable string."""
    parts = [
        f"torch_alloc {before['torch_alloc']}→{after['torch_alloc']} MB "
        f"(Δ{after['torch_alloc'] - before['torch_alloc']:+d})",
        f"torch_reserved {before['torch_reserved']}→{after['torch_reserved']} MB "
        f"(Δ{after['torch_reserved'] - before['torch_reserved']:+d})",
        f"peak_alloc {peak_alloc // 1_000_000} MB",
    ]
    if "nvml_used" in before:
        parts.append(
            f"nvml_used {before['nvml_used']}→{after['nvml_used']} MB "
            f"(Δ{after['nvml_used'] - before['nvml_used']:+d})"
        )
    return "  ".join(parts)


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
    """Return the replaceable forward behind the stable compile boundary.

    :param module: Module whose forward implementation is requested.
    :type module: torch.nn.Module
    :return: Current forward implementation.
    :rtype: Callable
    """
    return module.__dict__.get("_mfsdp_cuda_graph_forward_impl", module.forward)


def _set_cuda_graph_forward_impl(module: torch.nn.Module, forward: Callable) -> None:
    """Replace a forward without invalidating a compiled parent module.

    :param module: Module receiving the new implementation.
    :type module: torch.nn.Module
    :param forward: Replacement forward callable.
    :type forward: Callable
    :return: None.
    :rtype: None
    """
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
    """Match consumer inputs to an unambiguous earlier autograd output.

    :param modules: Captured modules in forward order.
    :type modules: Tuple[torch.nn.Module, ...]
    :param sample_outputs: Recorded outputs keyed by module identity.
    :type sample_outputs: Dict[int, Any]
    :param sample_args: Recorded positional inputs keyed by module identity.
    :type sample_args: Dict[int, Tuple[Any, ...]]
    :param sample_kwargs: Recorded keyword inputs keyed by module identity.
    :type sample_kwargs: Dict[int, Dict[str, Any]]
    :return: Consumer input indices mapped to producer output indices.
    :rtype: Tuple[Dict[int, Tuple[int, int]], ...]
    """
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
        self,
        graph_pool: Any,
        num_warmup_iters: int = 3,
        activation_recompute: bool = False,
        max_pending_forwards: int = 1,
    ):
        if not isinstance(activation_recompute, bool):
            raise TypeError("activation_recompute must be a bool")
        if not isinstance(max_pending_forwards, int) or isinstance(max_pending_forwards, bool):
            raise TypeError("max_pending_forwards must be an int")
        if max_pending_forwards < 1:
            raise ValueError("max_pending_forwards must be at least 1")
        if max_pending_forwards > 1 and not activation_recompute:
            raise ValueError("max_pending_forwards > 1 requires activation_recompute=True")
        self._graph_pool = graph_pool
        self._num_warmup = num_warmup_iters
        self._captured = False
        self._activation_recompute = activation_recompute
        self._max_pending_forwards = max_pending_forwards

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
        self._recorded_grad_enabled: Dict[int, bool] = {}
        self._recompute_requires_grad: Dict[int, Tuple[Any, Any]] = {}
        self._lifetime_events: List[Tuple[str, int, int]] = []
        self._recompute_lifetime_modules: Set[int] = set()
        self._backward_lifetime_modules: Set[int] = set()
        self._ordered_invocations: List[_RecordedInvocation] = []
        self._ordered_lifetime_events: List[Tuple[str, int, int]] = []
        self._ordered_region_events: List[Tuple[str, int]] = []
        self._ordered_replay_events: List[Tuple[str, int, int]] = []
        self._ordered_replay_cursor = 0
        self._ordered_replay_pending_backward_modules: Dict[int, int] = defaultdict(int)
        self._ordered_module_wrappers: Dict[int, List[Tuple[int, Callable]]] = {}
        self._ordered_slot_lanes: Dict[int, int] = {}
        self._ordered_inference_wrappers: Dict[int, Callable] = {}
        self._ordered_armed_backward: Optional[Tuple[int, int]] = None
        self._armed_backward_counts: Dict[int, int] = defaultdict(int)
        self._forward_scope_active = False
        self._forward_scope_invocations: List[Tuple[int, int]] = []
        self._forward_scope_lane = 0
        self._next_forward_lane = 0
        self._active_backward_invocations: Dict[int, int] = {}
        self._queued_backward_invocations: Dict[int, List[int]] = defaultdict(list)
        self._active_backward_lane: Optional[int] = None
        self._invocation_epoch = 0

    # ---- called from hooks ------------------------------------------------
    @property
    def captured(self) -> bool:
        """Return whether graph programs have been captured and installed.

        :return: Whether capture has completed.
        :rtype: bool
        """
        return self._captured

    def _ensure_forward_scope(self) -> None:
        """Start an implicit root-forward scope when none is active."""
        if self._forward_scope_active:
            return
        if self._captured and self._max_pending_forwards > 1:
            expected = self._ordered_expected_replay_event()
            if expected[0] != "forward":
                raise RuntimeError(
                    "Activation-recompute CUDA graph replay expected "
                    f"{expected[0]} before another forward"
                )
            selected_lane = self._ordered_slot_lanes[expected[2]]
        else:
            occupied_lanes = {
                invocation.lane_index
                for invocation in self._ordered_invocations
                if not invocation.backward_done
            }
            selected_lane = next(
                (
                    (self._next_forward_lane + offset) % self._max_pending_forwards
                    for offset in range(self._max_pending_forwards)
                    if (self._next_forward_lane + offset) % self._max_pending_forwards
                    not in occupied_lanes
                ),
                None,
            )
            if selected_lane is None:
                raise RuntimeError(
                    "Activation-recompute CUDA graphs reached max_pending_forwards; "
                    "finish backward or call release_pending() before another forward"
                )
        self._forward_scope_active = True
        self._forward_scope_invocations.clear()
        self._forward_scope_lane = selected_lane
        self._next_forward_lane = (selected_lane + 1) % self._max_pending_forwards

    def begin_forward_scope(self) -> None:
        """Start tracking graph invocations reached by one root forward."""
        if self._activation_recompute:
            self._ensure_forward_scope()

    def discard_forward_scope(self) -> None:
        """Discard invocation bookkeeping for a failed or completed root forward."""
        self._forward_scope_active = False
        self._forward_scope_invocations.clear()

    def _record_forward_scope_invocation(self, module_id: int, invocation: int) -> None:
        """Append one graph invocation to the current root-forward scope.

        :param module_id: Identity of the invoked module.
        :type module_id: int
        :param invocation: Recorded or captured invocation index.
        :type invocation: int
        """
        self._ensure_forward_scope()
        self._forward_scope_invocations.append((module_id, invocation))

    def _ensure_forward_lane_available(self, module: torch.nn.Module) -> None:
        """Reject reuse of a lane whose previous backward is unfinished.

        :param module: Module entering a new root-forward scope.
        :type module: torch.nn.Module
        :raises RuntimeError: If the selected lane still belongs to an older forward.
        """
        self._ensure_forward_scope()
        module_id = id(module)
        current_invocations = {
            invocation
            for candidate_module_id, invocation in self._forward_scope_invocations
            if candidate_module_id == module_id
        }
        lane_occupied = any(
            invocation_idx not in current_invocations
            and id(invocation.module) == module_id
            and invocation.lane_index == self._forward_scope_lane
            and not invocation.backward_done
            for invocation_idx, invocation in enumerate(self._ordered_invocations)
        )
        if lane_occupied:
            raise RuntimeError(
                "Activation-recompute CUDA graphs reached max_pending_forwards; "
                "finish backward or call release_pending() before another forward"
            )

    def _ordered_invocation_index(self, target: _RecordedInvocation) -> int:
        """Return an invocation index using object identity."""
        for invocation_idx, invocation in enumerate(self._ordered_invocations):
            if invocation is target:
                return invocation_idx
        raise RuntimeError("Recorded CUDA graph invocation is missing")

    def _ordered_finish_backward(self, invocation: _RecordedInvocation) -> None:
        """Mark one recorded invocation complete."""
        if invocation.backward_done:
            return
        invocation.backward_done = True
        invocation_idx = self._ordered_invocation_index(invocation)
        self._ordered_lifetime_events.append(("backward", invocation_idx, -1))

    def _selected_pending_invocation(
        self, module_id: int
    ) -> Optional[Tuple[int, _RecordedInvocation]]:
        """Return the selected unfinished invocation for one module.

        :param module_id: Identity of the recorded module.
        :type module_id: int
        :return: Invocation index and state, or ``None`` when no call is pending.
        :rtype: Optional[Tuple[int, _RecordedInvocation]]
        :raises RuntimeError: If the active microbatch lane is ambiguous.
        """
        invocation_idx = self._active_backward_invocations.get(module_id)
        if invocation_idx is not None:
            invocation = self._ordered_invocations[invocation_idx]
            if id(invocation.module) != module_id:
                raise RuntimeError("Selected backward invocation belongs to another module")
            if invocation.backward_done:
                return None
            return invocation_idx, invocation

        if self._max_pending_forwards > 1:
            raise RuntimeError(
                "Multiple pending activation-recompute forwards require an "
                "invocation-specific output backward hook"
            )
        candidates = [
            (idx, invocation)
            for idx, invocation in enumerate(self._ordered_invocations)
            if id(invocation.module) == module_id
            and not invocation.backward_done
            and (
                self._active_backward_lane is None
                or invocation.lane_index == self._active_backward_lane
            )
        ]
        if len(candidates) > 1:
            raise RuntimeError(
                "Multiple pending activation-recompute forwards require an "
                "invocation-specific output backward hook"
            )
        return candidates[0] if candidates else None

    def preflight_record_module(self, module: torch.nn.Module, replay_phase: str) -> None:
        """Reject a second recompute forward before parameter unsharding.

        :param module: FSDP module about to execute.
        :type module: torch.nn.Module
        :raises RuntimeError: If its first forward has not reached backward.
        """
        if self._captured or not self._activation_recompute:
            return
        if replay_phase not in ("forward", "recompute", "inference"):
            raise ValueError(f"Unknown CUDA graph replay phase {replay_phase!r}")
        mid = id(module)
        if self._max_pending_forwards > 1:
            if replay_phase == "recompute":
                return
            self._ensure_forward_lane_available(module)
            return
        if mid not in self._sample_args:
            return
        if replay_phase == "recompute":
            return
        if self._recorded_grad_enabled[mid]:
            raise RuntimeError(
                "Activation-recompute CUDA graphs require backward to finish "
                "before the next forward of the same module"
            )

    def record_module(self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]) -> None:
        """Record one module call during the first optimized forward.

        :param module: FSDP module invoked by the trace forward.
        :type module: torch.nn.Module
        :param args: Positional forward arguments.
        :type args: Tuple
        :param kwargs: Keyword forward arguments.
        :type kwargs: Dict[str, Any]
        """
        if self._captured:
            return
        mid = id(module)
        if self._max_pending_forwards > 1:
            self._record_ordered_module(module, args, kwargs)
            return
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
        self._recorded_grad_enabled[mid] = torch.is_grad_enabled()
        module_idx = len(self._modules_ordered)
        self._module_indices[mid] = module_idx
        self._modules_ordered.append(module)
        self._lifetime_events.append(("forward", module_idx, -1))
        self._record_forward_scope_invocation(mid, module_idx)

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

    def _record_ordered_module(
        self, module: torch.nn.Module, args: Tuple, kwargs: Dict[str, Any]
    ) -> None:
        """Record one forward occurrence for custom-order capture."""
        mid = id(module)
        self._ensure_forward_lane_available(module)

        if mid not in self._module_indices:
            self._original_forwards[mid] = _get_cuda_graph_forward_impl(module)
            self._original_graph_attrs[mid] = {
                name: module.__dict__[name]
                for name in _CUDA_GRAPH_RUNTIME_ATTRS
                if name in module.__dict__
            }
            self._module_indices[mid] = len(self._modules_ordered)
            self._modules_ordered.append(module)
            self._autocast_states[mid] = _cuda_autocast_state()
            self._recorded_grad_enabled[mid] = torch.is_grad_enabled()
        else:
            if self._autocast_states[mid] != _cuda_autocast_state():
                raise RuntimeError("CUDA graph module changed CUDA autocast state while recording")
            if self._recorded_grad_enabled[mid] != torch.is_grad_enabled():
                raise RuntimeError("CUDA graph module mixed checkpoint modes while recording")

        normalized_args, normalized_kwargs = _normalize_forward_call(module, args, kwargs)
        invocation_idx = len(self._ordered_invocations)
        self._ordered_invocations.append(
            _RecordedInvocation(
                module=module,
                args=normalized_args,
                kwargs=normalized_kwargs,
                lane_index=self._forward_scope_lane,
            )
        )
        self._ordered_lifetime_events.append(("forward", invocation_idx, -1))
        self._record_forward_scope_invocation(mid, invocation_idx)

    def record_module_recompute(
        self,
        module: torch.nn.Module,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one module call during checkpoint recomputation.

        :param module: Recorded module invoked by checkpoint recomputation.
        :type module: torch.nn.Module
        :param args: Recompute-forward positional arguments.
        :type args: Optional[Tuple[Any, ...]]
        :param kwargs: Recompute-forward keyword arguments.
        :type kwargs: Optional[Dict[str, Any]]
        :raises RuntimeError: If recomputation uses a different checkpoint region.
        """
        if self._captured or not self._activation_recompute:
            return
        mid = id(module)
        if self._max_pending_forwards > 1:
            selected = self._selected_pending_invocation(mid)
            if selected is None:
                raise RuntimeError(
                    "Activation-recompute recompute did not match a recorded forward"
                )
            invocation_idx, invocation = selected
            if invocation.recomputed:
                return
            recompute_requires_grad = None
            if args is not None:
                normalized_args, normalized_kwargs = _normalize_forward_call(
                    module, args, kwargs or {}
                )
                recompute_requires_grad = (
                    _requires_grad_surface(normalized_args),
                    _requires_grad_surface(normalized_kwargs),
                )
            invocation.recomputed = True
            invocation.recompute_requires_grad = recompute_requires_grad
            self._ordered_lifetime_events.append(("recompute", invocation_idx, -1))
            return
        module_idx = self._module_indices.get(mid)
        if module_idx is None or module_idx in self._recompute_lifetime_modules:
            return
        recompute_requires_grad = None
        if args is not None:
            normalized_args, normalized_kwargs = _normalize_forward_call(module, args, kwargs or {})
            recompute_requires_grad = (
                _requires_grad_surface(normalized_args),
                _requires_grad_surface(normalized_kwargs),
            )
        self._recompute_lifetime_modules.add(module_idx)
        if recompute_requires_grad is not None:
            self._recompute_requires_grad[mid] = recompute_requires_grad
        self._lifetime_events.append(("recompute", module_idx, -1))

    def owns_module(self, module: torch.nn.Module) -> bool:
        """Return whether this runner recorded ``module`` for CUDA Graph replay.

        :param module: Module whose capture ownership is queried.
        :type module: torch.nn.Module
        :return: Whether the runner owns the module's replay program.
        :rtype: bool
        """
        module_id = id(module)
        if self._captured and self._max_pending_forwards > 1:
            return module_id in self._ordered_module_wrappers
        return module_id in self._module_indices

    def expects_module_recompute(self, module: torch.nn.Module) -> bool:
        """Return whether a recorded invocation is awaiting RF.

        :param module: Module entering a grad-enabled forward during backward.
        :type module: torch.nn.Module
        :return: Whether the selected invocation still needs recomputation.
        :rtype: bool
        """
        if not self._activation_recompute:
            return False
        module_id = id(module)
        if self._max_pending_forwards > 1 and not self._captured:
            if not any(
                id(invocation.module) == module_id for invocation in self._ordered_invocations
            ):
                return False
            selected = self._selected_pending_invocation(module_id)
            return selected is not None and not selected[1].recomputed
        if module_id in self._active_backward_invocations:
            if not self._captured:
                module_idx = self._module_indices.get(module_id)
                return module_idx is not None and module_idx not in self._recompute_lifetime_modules
            return True
        if self._captured and self._max_pending_forwards > 1:
            expected = self._ordered_expected_replay_event()
            return expected[:2] == ("recompute", module_id)
        module_idx = self._module_indices.get(module_id)
        return (
            not self._captured
            and module_idx is not None
            and module_idx not in self._recompute_lifetime_modules
        )

    def prepare_module_replay(self, module: torch.nn.Module, replay_phase: str) -> None:
        """Select the F, RF, or inference graph before module preflight."""
        if not self._activation_recompute or not self._captured or not self.owns_module(module):
            return
        if replay_phase not in ("forward", "recompute", "inference"):
            raise ValueError(f"Unknown CUDA graph replay phase {replay_phase!r}")
        if self._max_pending_forwards > 1:
            if replay_phase == "inference":
                if (
                    self._active_backward_invocations
                    or any(self._queued_backward_invocations.values())
                    or self._ordered_armed_backward is not None
                    or any(self._ordered_replay_pending_backward_modules.values())
                ):
                    pending = {
                        module_id: count
                        for module_id, count in self._ordered_replay_pending_backward_modules.items()
                        if count
                    }
                    raise RuntimeError(
                        "Multi-invocation activation-recompute CUDA graphs require "
                        "an idle schedule before inference "
                        f"(active={self._active_backward_invocations}, "
                        f"queued={dict(self._queued_backward_invocations)}, "
                        f"armed={self._ordered_armed_backward}, pending={pending})"
                    )
                target = self._ordered_module_wrappers[id(module)][0][1]
                self._ordered_inference_wrappers[id(module)] = target
                setter = getattr(target, "_cuda_graph_set_replay_phase", None)
                if not callable(setter):
                    raise RuntimeError(
                        "Captured activation-recompute module has no replay selector"
                    )
                setter("inference")
                return
            expected = self._ordered_expected_replay_event()
            if expected[:2] != (replay_phase, id(module)):
                raise RuntimeError(
                    "Activation-recompute CUDA graph replay order changed: expected "
                    f"{expected[0]} for module id {expected[1]}, got {replay_phase} for "
                    f"module id {id(module)}"
                )
            active_invocation = self._active_backward_invocations.get(id(module))
            if replay_phase == "recompute":
                if active_invocation is None:
                    raise RuntimeError(
                        "Multiple pending activation-recompute forwards require an "
                        "invocation-specific output backward hook"
                    )
                selected_lane = self._ordered_slot_lanes[active_invocation]
                if selected_lane != self._ordered_slot_lanes[expected[2]]:
                    raise RuntimeError(
                        "Activation-recompute CUDA graph selected the wrong microbatch lane"
                    )
            target = self._ordered_wrapper(id(module), expected[2])
            if replay_phase == "forward":
                self._record_forward_scope_invocation(id(module), expected[2])
        else:
            target = module
            if replay_phase == "forward":
                self._record_forward_scope_invocation(id(module), self._module_indices[id(module)])
        setter = getattr(target, "_cuda_graph_set_replay_phase", None)
        if not callable(setter):
            raise RuntimeError("Captured activation-recompute module has no replay selector")
        setter(replay_phase)

    def backward_invocation_token(
        self, module: torch.nn.Module
    ) -> Optional[_BackwardInvocationToken]:
        """Return graph invocations reached from the current output."""
        if not self._activation_recompute:
            return None
        module_id = id(module)
        fsdp_state = getattr(module, "_fsdp_state", None)
        is_root = fsdp_state is None or getattr(fsdp_state, "_is_root", False)
        scope_invocations = tuple(self._forward_scope_invocations)
        if is_root:
            invocations = scope_invocations
            self.discard_forward_scope()
        elif self.owns_module(module):
            invocation = next(
                (
                    invocation
                    for candidate_module_id, invocation in reversed(scope_invocations)
                    if candidate_module_id == module_id
                ),
                None,
            )
            invocations = () if invocation is None else ((module_id, invocation),)
        else:
            invocations = scope_invocations
        if not invocations:
            return None
        return _BackwardInvocationToken(self._invocation_epoch, invocations)

    def select_backward_invocation(
        self, module: torch.nn.Module, invocation_token: Optional[_BackwardInvocationToken]
    ) -> None:
        """Select the invocation whose output started backward."""
        if invocation_token is None or not self._activation_recompute:
            return
        if invocation_token.epoch != self._invocation_epoch:
            raise RuntimeError(
                "Activation-recompute backward belongs to a released or superseded forward"
            )
        for module_id, invocation in reversed(invocation_token.invocations):
            self._select_backward_invocation(module_id, invocation)

    def _select_backward_invocation(self, module_id: int, invocation: int) -> None:
        """Select one module invocation from a root-output token."""
        existing = self._active_backward_invocations.get(module_id)
        queued = self._queued_backward_invocations.get(module_id, ())
        if existing == invocation or invocation in queued:
            return
        if existing is not None:
            self._queued_backward_invocations[module_id].append(invocation)
            return
        had_active_invocation = bool(self._active_backward_invocations)
        self._active_backward_invocations[module_id] = invocation
        recorded_invocation = (
            self._ordered_invocations[invocation]
            if self._max_pending_forwards > 1 and not self._captured
            else None
        )
        if recorded_invocation is not None:
            lane_index = recorded_invocation.lane_index
        elif self._max_pending_forwards > 1:
            lane_index = self._ordered_slot_lanes[invocation]
        else:
            lane_index = 0
        if had_active_invocation and self._active_backward_lane not in (None, lane_index):
            raise RuntimeError("M-FSDP checkpoint region mixed microbatch lanes")
        self._active_backward_lane = lane_index

    def _finish_active_backward_invocation(self, module_id: int) -> None:
        """Complete one invocation and promote the next shared-module occurrence.

        :param module_id: Identity of the module completing backward.
        :type module_id: int
        """
        self._active_backward_invocations.pop(module_id, None)
        queued = self._queued_backward_invocations.get(module_id)
        if queued:
            next_invocation = queued.pop(0)
            if not queued:
                self._queued_backward_invocations.pop(module_id, None)
            self._active_backward_invocations[module_id] = next_invocation
        else:
            self._queued_backward_invocations.pop(module_id, None)

    def record_module_backward(self, module: torch.nn.Module) -> bool:
        """Record the logical backward after checkpoint recomputation.

        The output-side pre-hook may run before non-reentrant recomputation.
        Its event is held until the matching recompute forward is observed.

        :param module: Recorded module whose backward work is being prepared.
        :type module: torch.nn.Module
        :return: Whether this runner owns the module's backward.
        :rtype: bool
        """
        if not self.owns_module(module):
            return False
        module_id = id(module)
        if self._captured:
            if self._max_pending_forwards > 1:
                self._ordered_arm_replay_backward(module)
            else:
                self._armed_backward_counts[module_id] += 1
            return True
        self._armed_backward_counts[module_id] += 1
        return True

    def record_module_output(self, module: torch.nn.Module, output: Any) -> None:
        """Record an eager output for static graph linking.

        :param module: Recorded FSDP module.
        :type module: torch.nn.Module
        :param output: Output from the eager sample forward.
        :type output: Any
        """
        mid = id(module)
        if self._captured:
            return
        if self._max_pending_forwards > 1:
            for invocation in reversed(self._ordered_invocations):
                if id(invocation.module) == mid and invocation.output is None:
                    invocation.output = output
                    return
            return
        if mid not in self._sample_args or mid in self._sample_outputs:
            return
        self._sample_outputs[mid] = output

    def release_pending(self) -> bool:
        """Release an abandoned recorded or captured forward.

        :return: Whether pending activation-recompute state was released.
        :rtype: bool
        """
        if not self._activation_recompute:
            return False
        if not self._captured:
            if not self._modules_ordered:
                return False
            for module in self._modules_ordered:
                module._fsdp_cg_pending_backwards = 0
            self.reset()
            return True

        releasers = []
        seen = set()
        for module in self._modules_ordered:
            releaser = module.__dict__.get("_cuda_graph_release_pending")
            if callable(releaser) and id(releaser) not in seen:
                seen.add(id(releaser))
                releasers.append(releaser)
        released = [releaser() for releaser in releasers]
        self._ordered_inference_wrappers.clear()
        if any(released):
            self._invocation_epoch += 1
            for module in self._modules_ordered:
                module._fsdp_cg_pending_backwards = 0
            if self._max_pending_forwards > 1:
                self._ordered_replay_cursor = 0
                self._ordered_replay_pending_backward_modules.clear()
                self._ordered_armed_backward = None
            self._armed_backward_counts.clear()
            self.discard_forward_scope()
            self._active_backward_invocations.clear()
            self._queued_backward_invocations.clear()
            self._active_backward_lane = None
        return any(released)

    def reset(self) -> None:
        """Destroy captured graphs and restore the original module callables."""
        self._invocation_epoch += 1
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
            module.__dict__.pop("_fsdp_cg_activation_recompute", None)
            for param_group in getattr(module, "_fsdp_param_groups", ()):
                for param in param_group.params:
                    param.__dict__.pop("_mfsdp_recorded_te_wgrad", None)
                if not getattr(param_group, "_main_grad_buffer_has_unreduced_data", False):
                    release_grad_storage = getattr(
                        param_group, "_release_grad_storage_if_unused", None
                    )
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
        self._recorded_grad_enabled.clear()
        self._recompute_requires_grad.clear()
        self._lifetime_events.clear()
        self._recompute_lifetime_modules.clear()
        self._backward_lifetime_modules.clear()
        self._ordered_invocations.clear()
        self._ordered_lifetime_events.clear()
        self._ordered_region_events.clear()
        self._ordered_replay_events.clear()
        self._ordered_replay_cursor = 0
        self._ordered_replay_pending_backward_modules.clear()
        self._ordered_module_wrappers.clear()
        self._ordered_slot_lanes.clear()
        self._ordered_inference_wrappers.clear()
        self._ordered_armed_backward = None
        self._armed_backward_counts.clear()
        self.discard_forward_scope()
        self._next_forward_lane = 0
        self._active_backward_invocations.clear()
        self._queued_backward_invocations.clear()
        self._active_backward_lane = None
        self._captured = False

    def _ordered_capture_ready(self) -> bool:
        """Return whether the configured ordered trace is complete."""
        if not self._ordered_invocations:
            return False
        module_lanes = defaultdict(set)
        for invocation in self._ordered_invocations:
            module_lanes[id(invocation.module)].add(invocation.lane_index)
            if not invocation.backward_done:
                return False
        return bool(module_lanes) and all(
            len(lanes) == self._max_pending_forwards for lanes in module_lanes.values()
        )

    def _build_ordered_capture_plan(self) -> _OrderedCapturePlan:
        """Translate recorded region lifetimes to graph.py custom-order inputs."""
        if not self._ordered_capture_ready():
            raise RuntimeError("Ordered CUDA graph trace is not complete")

        invocation_regions, region_events = _infer_activation_recompute_regions(
            self._ordered_lifetime_events,
            len(self._ordered_invocations),
            require_reverse_regions=False,
        )
        for invocation, region_idx in zip(self._ordered_invocations, invocation_regions):
            invocation.region_index = region_idx
        self._ordered_region_events = list(region_events)
        region_count = max(invocation_regions, default=-1) + 1
        region_calls = {
            region_idx: tuple(
                invocation
                for invocation in self._ordered_invocations
                if invocation.region_index == region_idx
            )
            for region_idx in range(region_count)
        }
        signatures = {
            region_idx: tuple(id(invocation.module) for invocation in calls)
            for region_idx, calls in region_calls.items()
        }
        if any(not signature for signature in signatures.values()):
            raise RuntimeError("Activation-recompute checkpoint region is empty")

        chunk_signatures: List[Tuple[int, ...]] = []
        region_chunks = {}
        for phase, region_idx in self._ordered_region_events:
            if phase != "forward":
                continue
            signature = signatures[region_idx]
            if len(set(signature)) != len(signature):
                raise RuntimeError(
                    "One activation-recompute checkpoint region cannot invoke the same "
                    "M-FSDP module twice"
                )
            if signature not in chunk_signatures:
                chunk_signatures.append(signature)
            region_chunks[region_idx] = chunk_signatures.index(signature)

        owned_modules = {}
        for chunk_idx, signature in enumerate(chunk_signatures):
            for module_id in signature:
                previous_chunk = owned_modules.setdefault(module_id, chunk_idx)
                if previous_chunk != chunk_idx:
                    raise RuntimeError(
                        "One M-FSDP module cannot belong to multiple checkpoint region shapes"
                    )
        for chunk_idx in range(len(chunk_signatures)):
            observed_lanes = {
                region_calls[region_idx][0].lane_index
                for region_idx, candidate_chunk in region_chunks.items()
                if candidate_chunk == chunk_idx
            }
            if observed_lanes != set(range(self._max_pending_forwards)):
                raise RuntimeError(
                    "Ordered CUDA graph trace must observe every pending-forward lane "
                    "for each checkpoint region shape"
                )

        modules: List[torch.nn.Module] = []
        module_regions: List[int] = []
        invocations: List[_RecordedInvocation] = []
        sample_indices = {}
        for chunk_idx, signature in enumerate(chunk_signatures):
            chunk_regions = [
                region_idx
                for phase, region_idx in self._ordered_region_events
                if phase == "forward" and region_chunks[region_idx] == chunk_idx
            ]
            first_calls = region_calls[chunk_regions[0]]
            modules.extend(invocation.module for invocation in first_calls)
            module_regions.extend([chunk_idx] * len(first_calls))
            for region_idx in chunk_regions:
                for invocation in region_calls[region_idx]:
                    sample_indices[id(invocation)] = len(invocations)
                    invocations.append(invocation)

        order = []
        order_slots = []
        replay_events = []
        chunk_region_slots = {}
        for phase, region_idx in self._ordered_region_events:
            if phase != "forward":
                continue
            lanes = {invocation.lane_index for invocation in region_calls[region_idx]}
            if len(lanes) != 1:
                raise RuntimeError(
                    "One activation-recompute checkpoint region mixed microbatch lanes"
                )
            chunk_region_slots[region_idx] = lanes.pop()
        for phase, region_idx in self._ordered_region_events:
            chunk_id = region_chunks[region_idx] + 1
            calls = region_calls[region_idx]
            order_slots.append(chunk_region_slots[region_idx])
            if phase == "forward":
                order.append(chunk_id)
                replay_events.extend(
                    ("forward", id(invocation.module), sample_indices[id(invocation)])
                    for invocation in calls
                )
            else:
                order.append(-chunk_id)
                replay_events.extend(
                    ("recompute", id(invocation.module), sample_indices[id(invocation)])
                    for invocation in calls
                )
                replay_events.extend(
                    ("backward", id(invocation.module), sample_indices[id(invocation)])
                    for invocation in reversed(calls)
                )

        return _OrderedCapturePlan(
            modules=tuple(modules),
            invocations=tuple(invocations),
            order=tuple(order),
            order_slots=tuple(order_slots),
            num_layers_per_chunk=tuple(len(signature) for signature in chunk_signatures),
            module_regions=tuple(module_regions),
            replay_events=tuple(replay_events),
        )

    def _ordered_expected_replay_event(self) -> Tuple[str, int, int]:
        """Return the next captured phase, module, and graph-slot index."""
        if not self._ordered_replay_events:
            raise RuntimeError("Ordered CUDA graph replay schedule is empty")
        return self._ordered_replay_events[self._ordered_replay_cursor]

    def _ordered_advance_replay(self, expected: Tuple[str, int, int]) -> None:
        """Advance one exact event in the cyclic replay schedule."""
        if self._ordered_expected_replay_event() != expected:
            raise RuntimeError("Activation-recompute CUDA graph replay order changed")
        self._ordered_replay_cursor = (self._ordered_replay_cursor + 1) % len(
            self._ordered_replay_events
        )

    def _ordered_wrapper(self, module_id: int, sample_idx: int) -> Callable:
        """Return a module wrapper for one recorded graph slot."""
        for wrapper_idx, wrapper in self._ordered_module_wrappers[module_id]:
            if wrapper_idx == sample_idx:
                return wrapper
        raise RuntimeError("Ordered CUDA graph wrapper slot is missing")

    def _ordered_preflight_replay(self, module: torch.nn.Module) -> None:
        """Validate the next module phase before M-FSDP unshards parameters."""
        inference_wrapper = self._ordered_inference_wrappers.get(id(module))
        if inference_wrapper is not None:
            preflight = getattr(inference_wrapper, "_cuda_graph_preflight", None)
            try:
                if callable(preflight):
                    preflight()
            except Exception:
                self._ordered_inference_wrappers.pop(id(module), None)
                release_pending = getattr(inference_wrapper, "_cuda_graph_release_pending", None)
                if callable(release_pending):
                    release_pending()
                raise
            return
        expected = self._ordered_expected_replay_event()
        if expected[1] != id(module):
            raise RuntimeError(
                "Activation-recompute CUDA graph replay order changed: expected "
                f"{expected[0]} for module id {expected[1]}, got module id {id(module)}"
            )
        wrapper = self._ordered_wrapper(id(module), expected[2])
        preflight = getattr(wrapper, "_cuda_graph_preflight", None)
        if callable(preflight):
            preflight()

    def _ordered_dispatch(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Replay one F or RF graph through its recorded invocation slot."""
        inference_wrapper = self._ordered_inference_wrappers.pop(id(module), None)
        if inference_wrapper is not None:
            return inference_wrapper(*args, **kwargs)
        expected = self._ordered_expected_replay_event()
        replay_phase = expected[0]
        if expected[1] != id(module):
            raise RuntimeError("Activation-recompute CUDA graph replay order changed")
        sample_idx = expected[2]

        wrapper = self._ordered_wrapper(id(module), sample_idx)
        try:
            output = wrapper(*args, **kwargs)
        except _StopRecomputationError:
            if replay_phase != "recompute":
                raise
            self._ordered_advance_replay(expected)
            if self._ordered_replay_pending_backward_modules[id(module)]:
                next_event = self._ordered_expected_replay_event()
                if next_event[:2] != ("backward", id(module)):
                    raise RuntimeError(
                        "M-FSDP backward preparation did not match the recorded RF/B order"
                    )
                self._ordered_replay_pending_backward_modules[id(module)] -= 1
                self._ordered_armed_backward = (id(module), next_event[2])
            raise
        self._ordered_advance_replay(expected)

        if (
            replay_phase == "recompute"
            and self._ordered_replay_pending_backward_modules[id(module)]
        ):
            next_event = self._ordered_expected_replay_event()
            if next_event[:2] != ("backward", id(module)):
                raise RuntimeError(
                    "M-FSDP backward preparation did not match the recorded RF/B order"
                )
            self._ordered_replay_pending_backward_modules[id(module)] -= 1
            self._ordered_armed_backward = (id(module), next_event[2])
        return output

    def _ordered_arm_replay_backward(self, module: torch.nn.Module) -> None:
        """Validate or defer a backward pre-hook until RF has replayed."""
        expected = self._ordered_expected_replay_event()
        module_id = id(module)
        if expected[0] == "recompute":
            self._ordered_replay_pending_backward_modules[module_id] += 1
            return
        if expected[:2] != ("backward", module_id):
            raise RuntimeError("M-FSDP backward order changed from the captured schedule")
        self._ordered_armed_backward = (module_id, expected[2])

    def complete_module_backward(
        self, module: torch.nn.Module, *, strict: bool = False, allow_unarmed: bool = False
    ) -> bool:
        """Consume one backward event owned by this runner.

        :param module: Module whose post-backward work is ready.
        :type module: torch.nn.Module
        :param strict: Raise when an owned module is not currently armed.
        :type strict: bool
        :param allow_unarmed: Claim a recomputed event at autograd finalization.
        :type allow_unarmed: bool
        :return: Whether one recorded backward event was consumed.
        :rtype: bool
        :raises RuntimeError: If strict completion changes the captured order.
        """
        if not self.owns_module(module):
            return False
        module_id = id(module)
        count = self._armed_backward_counts.get(module_id, 0)

        if not self._activation_recompute:
            if count == 0:
                if strict:
                    raise RuntimeError(
                        "M-FSDP backward completion arrived before backward preparation"
                    )
                return False
            if count == 1:
                self._armed_backward_counts.pop(module_id)
            else:
                self._armed_backward_counts[module_id] = count - 1
            return True

        if self._captured and self._max_pending_forwards > 1:
            expected = self._ordered_expected_replay_event()
            armed = self._ordered_armed_backward
            matches = expected[:2] == ("backward", module_id) and armed == (module_id, expected[2])
            if (
                not matches
                and allow_unarmed
                and armed is None
                and expected[:2] == ("backward", module_id)
            ):
                matches = True
            if not matches:
                if strict:
                    raise RuntimeError(
                        "M-FSDP backward completion changed from the captured order: "
                        f"expected={expected}, armed={armed}, module_id={module_id}"
                    )
                return False
            self._ordered_armed_backward = None
            self._ordered_advance_replay(expected)
            self._finish_active_backward_invocation(module_id)
            return True

        if not self._captured:
            if self._max_pending_forwards > 1:
                selected = self._selected_pending_invocation(module_id)
                if selected is None or not selected[1].recomputed:
                    if strict:
                        raise RuntimeError(
                            "Activation-recompute CUDA graphs did not observe checkpoint "
                            "recomputation before backward; wrap the graph-enabled module "
                            "in activation checkpointing or disable "
                            "cuda_graph_activation_recompute"
                        )
                    return False
                _, invocation = selected
                self._ordered_finish_backward(invocation)
                self._finish_active_backward_invocation(module_id)
            else:
                module_idx = self._module_indices[module_id]
                if module_idx not in self._recompute_lifetime_modules:
                    if strict:
                        raise RuntimeError(
                            "Activation-recompute CUDA graphs did not observe checkpoint "
                            "recomputation before backward; wrap the graph-enabled module "
                            "in activation checkpointing or disable "
                            "cuda_graph_activation_recompute"
                        )
                    return False
                if module_idx in self._backward_lifetime_modules:
                    return True
                self._backward_lifetime_modules.add(module_idx)
                self._lifetime_events.append(("backward", module_idx, -1))
            if count == 1:
                self._armed_backward_counts.pop(module_id)
            elif count > 1:
                self._armed_backward_counts[module_id] = count - 1
            if self._max_pending_forwards == 1:
                self._finish_active_backward_invocation(module_id)
            return True
        if count == 0:
            if strict:
                raise RuntimeError("M-FSDP backward completion arrived before backward preparation")
            return False
        if count == 1:
            self._armed_backward_counts.pop(module_id)
        else:
            self._armed_backward_counts[module_id] = count - 1
        self._finish_active_backward_invocation(module_id)
        return True

    def capture_and_install(
        self, root_module: torch.nn.Module, capture_stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """Capture all graphs + install wrappers on recorded modules."""
        if self._captured or not self._modules_ordered:
            return

        ordered_plan = None
        if self._max_pending_forwards > 1:
            if not self._ordered_capture_ready():
                return
            ordered_plan = self._build_ordered_capture_plan()

        modules = ordered_plan.modules if ordered_plan is not None else tuple(self._modules_ordered)
        n = len(modules)
        autocast_states = {self._autocast_states[id(module)] for module in modules}
        if len(autocast_states) != 1:
            raise RuntimeError("CUDA graph capture requires one recorded CUDA autocast state")
        autocast_enabled, autocast_dtype, _ = next(iter(autocast_states))
        activation_recompute = self._activation_recompute
        root_context = getattr(root_module, "_fsdp_root_context", None)
        configured_recompute = getattr(
            root_context, "cuda_graph_activation_recompute", activation_recompute
        )
        if configured_recompute != activation_recompute:
            raise RuntimeError(
                "M-FSDP CUDA Graph activation-recompute configuration changed before capture"
            )
        if activation_recompute:
            if ordered_plan is not None:
                module_regions = ordered_plan.module_regions
            else:
                module_regions, _ = _infer_activation_recompute_regions(
                    self._lifetime_events, len(modules), require_reverse_regions=True
                )
            for fsdp_module in modules:
                for module in fsdp_module.modules():
                    if _module_uses_delayed_wgrad(module):
                        raise RuntimeError(
                            "M-FSDP CUDA Graph activation recompute does not yet support "
                            "delayed backward-wgrad"
                        )

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
        try:
            capture_mutable_state = _snapshot_capture_mutable_state(modules)
        except Exception:
            _restore_pending_compute_gradients(pending_compute_gradients)
            self.reset()
            raise

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

                required_capabilities = {
                    "capture_grad_buffer_release",
                    "parameter_surface_refresh",
                    "registered_buffer_validation",
                    "static_grad_binding",
                }
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

        input_output_aliases = (
            tuple({} for _ in ordered_plan.invocations)
            if ordered_plan is not None
            else _build_input_output_aliases(
                modules, self._sample_outputs, self._sample_args, self._sample_kwargs
            )
        )

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: linked %d static input/output tensors",
                sum(len(aliases) for aliases in input_output_aliases),
            )

        module_capture_groups = {}
        for m in modules:
            mid = id(m)
            if ordered_plan is not None:
                sample_invocation = next(
                    invocation for invocation in ordered_plan.invocations if invocation.module is m
                )
                module_capture_groups[mid] = _collect_capture_sync_groups(
                    m, sample_invocation.args, sample_invocation.kwargs
                )
            else:
                module_capture_groups[mid] = _collect_capture_sync_groups(
                    m, self._sample_args[mid], self._sample_kwargs[mid]
                )
            capture_sync_groups = module_capture_groups[mid]
            capture_hooks.append(
                {
                    "forward_pre_hooks": {0: _make_fwd_pre_hook(m, capture_sync_groups)},
                    "forward_pre_hooks_with_kwargs": {0: True},
                    "forward_hooks": {0: _make_fwd_post_hook(m, capture_sync_groups)},
                    "forward_hooks_with_kwargs": {0: True},
                    "backward_pre_hooks": {
                        0: _make_bwd_pre_hook(
                            m,
                            activation_recompute=activation_recompute,
                            capture_sync_groups=capture_sync_groups,
                        )
                    },
                    "backward_hooks": {0: _make_bwd_post_hook(m, capture_sync_groups)},
                }
            )

        samples = ordered_plan.invocations if ordered_plan is not None else tuple(modules)
        for sample in samples:
            m = sample.module if ordered_plan is not None else sample
            mid = id(m)
            recompute_requires_grad = (
                sample.recompute_requires_grad
                if ordered_plan is not None
                else self._recompute_requires_grad.get(mid)
            )
            if activation_recompute and recompute_requires_grad is None:
                raise RuntimeError(
                    "Activation-recompute CUDA graph capture is missing RF input metadata"
                )
            args_requires_grad, kwargs_requires_grad = (
                recompute_requires_grad if recompute_requires_grad is not None else (None, None)
            )
            # Clone tensor values so warmup gets fresh leaves without
            # residual autograd state from the first forward+backward.
            args = _clone_capture_sample(
                sample.args if ordered_plan is not None else self._sample_args[mid],
                args_requires_grad,
            )
            kw = _clone_capture_sample(
                sample.kwargs if ordered_plan is not None else self._sample_kwargs[mid],
                kwargs_requires_grad,
            )
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
            runtime_options["_activation_recompute_regions"] = module_regions
            if ordered_plan is not None:
                runtime_options["_order"] = list(ordered_plan.order)
                runtime_options["_activation_recompute_order_slots"] = list(
                    ordered_plan.order_slots
                )
                runtime_options["_num_layers_per_chunk"] = list(ordered_plan.num_layers_per_chunk)
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
        if ordered_plan is not None:
            for invocation in self._ordered_invocations:
                invocation.args = ()
                invocation.kwargs = {}
                invocation.output = None
        gc.collect()

        try:
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(_restore_pending_compute_gradients, pending_compute_gradients)
                cleanup.callback(_restore_all_hooks, saved_hooks)
                cleanup.callback(capture_mutable_state.restore)
                torch.cuda.reset_peak_memory_stats()
                _mem_before = _mem_snapshot()

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
                        use_main_grad=ordered_plan is None,
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

        _mem_after = _mem_snapshot()
        _peak_alloc = torch.cuda.max_memory_allocated()

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(
                "CudaGraphRunner: %d modules captured %s",
                n,
                _fmt_mem_snapshot(_mem_before, _mem_after, _peak_alloc),
            )

        if not isinstance(graphed, tuple):
            graphed = (graphed,)

        if ordered_plan is not None:
            self._ordered_replay_events = list(ordered_plan.replay_events)
            self._ordered_replay_cursor = 0
            wrappers = defaultdict(list)
            for sample_idx, (invocation, wrapper) in enumerate(
                zip(ordered_plan.invocations, graphed)
            ):
                wrappers[id(invocation.module)].append((sample_idx, wrapper))
            self._ordered_module_wrappers = dict(wrappers)
            self._ordered_slot_lanes = {
                sample_idx: invocation.lane_index
                for sample_idx, invocation in enumerate(ordered_plan.invocations)
            }

            def make_ordered_forward(module):
                def ordered_forward(*args, **kwargs):
                    return self._ordered_dispatch(module, *args, **kwargs)

                return ordered_forward

            def make_ordered_release(module):
                def release():
                    released = 0
                    for _, wrapper in self._ordered_module_wrappers[id(module)]:
                        releaser = getattr(wrapper, "_cuda_graph_release_pending", None)
                        if callable(releaser) and releaser():
                            released += 1
                    if released:
                        module._fsdp_cg_pending_backwards = max(
                            0, module._fsdp_cg_pending_backwards - released
                        )
                    return bool(released)

                return release

            def make_ordered_reset(module):
                def reset():
                    for _, wrapper in self._ordered_module_wrappers[id(module)]:
                        wrapper.reset()

                return reset

            for module in modules:
                _set_cuda_graph_forward_impl(module, make_ordered_forward(module))
                module._cuda_graph_release_pending = make_ordered_release(module)
                module.reset = make_ordered_reset(module)
                module._cuda_graph_preflight = _make_module_topology_preflight(
                    _capture_module_topology(module),
                    lambda module=module: self._ordered_preflight_replay(module),
                )

        # make_graphed_callables already replaced module.forward with
        # the graphed version that handles kwargs natively.
        for module in modules:
            if ordered_plan is None:
                module._cuda_graph_preflight = _make_module_topology_preflight(
                    _capture_module_topology(module), module.__dict__.get("_cuda_graph_preflight")
                )
            module._fsdp_cg_installed = True
            module._fsdp_cg_activation_recompute = activation_recompute
        self._compiled_module_state = []

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("CudaGraphRunner: installed CUDA graphs on %d modules", n)


# ---------------------------------------------------------------------------
# capture_time_hooks (unshard / reshard outside graph, not replayed)
# ---------------------------------------------------------------------------


def _clone_capture_sample(value: Any, requires_grad_surface: Any = None) -> Any:
    """Clone tensor leaves using recompute-forward gradient metadata.

    :param value: Input PyTree.
    :type value: Any
    :param requires_grad_surface: Optional PyTree of tensor gradient flags.
    :type requires_grad_surface: Any
    :return: Cloned input PyTree.
    :rtype: Any
    """

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


def _module_uses_delayed_wgrad(module):
    """Return whether a module requests a separate delayed-wgrad graph."""
    for owner in (module, getattr(module, "config", None)):
        if owner is None:
            continue
        delayed = getattr(owner, "delay_wgrad_compute", False)
        if bool(delayed() if callable(delayed) else delayed):
            return True
    delayed = getattr(getattr(module, "wgrad_store", None), "delay_wgrad_compute", False)
    return bool(delayed() if callable(delayed) else delayed)


def _collect_capture_sync_groups(module, sample_args, sample_kwargs):
    """Collect CP groups used by a module or its captured input metadata."""
    groups = []

    def add_group(group):
        if isinstance(group, (tuple, list)):
            for child in group:
                add_group(child)
            return
        if not isinstance(group, torch.distributed.ProcessGroup):
            return
        if all(group is not existing for existing in groups):
            groups.append(group)

    def visit_input(value):
        add_group(getattr(value, "cp_group", None))
        if isinstance(value, dict):
            for child in value.values():
                visit_input(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit_input(child)

    visit_input(sample_args)
    visit_input(sample_kwargs)
    for submodule in module.modules():
        add_group(getattr(submodule, "cp_group", None))
        pg_collection = getattr(submodule, "pg_collection", None)
        add_group(getattr(pg_collection, "cp", None))
        add_group(getattr(pg_collection, "hcp", None))
    return tuple(groups)


def _capture_group_barrier(capture_sync_groups):
    """Synchronize ranks that capture a graph containing a collective."""
    for group in capture_sync_groups:
        barrier_kwargs = {"group": group}
        backend = torch.distributed.get_backend(group)
        if str(backend).lower() == "nccl":
            barrier_kwargs["device_ids"] = [torch.cuda.current_device()]
        torch.distributed.barrier(**barrier_kwargs)


def _make_fwd_pre_hook(module, capture_sync_groups=()):
    """Build the capture-time forward unshard hook.

    :param module: FSDP module to unshard.
    :type module: torch.nn.Module
    :return: Forward pre-hook callable.
    :rtype: Callable
    """

    def hook(mod, args, kwargs):
        _capture_group_barrier(capture_sync_groups)
        module.unshard()

    return hook


def _make_fwd_post_hook(module, capture_sync_groups=()):
    def hook(mod, args, kwargs, output):
        module.reshard()
        _capture_group_barrier(capture_sync_groups)

    return hook


def _make_bwd_pre_hook(module, activation_recompute=False, capture_sync_groups=()):
    """Build the capture-time backward unshard hook.

    :param module: FSDP module to unshard.
    :type module: torch.nn.Module
    :return: Backward pre-hook callable.
    :rtype: Callable
    """

    def hook(mod, grad_output):
        _capture_group_barrier(capture_sync_groups)
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
                param_group.main_grad_buffer.fetch_buffer()
                for param in param_group.params:
                    if getattr(param, "_mfsdp_recorded_te_wgrad", False):
                        param.main_grad = param.get_main_grad()

    return hook


def _make_bwd_post_hook(module, capture_sync_groups=()):
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
        _capture_group_barrier(capture_sync_groups)

    return hook
