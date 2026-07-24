# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Standalone TE-compatible CUDA graph callable runtime."""

# This file is adapted from Transformer Engine's vendored runtime. Keep its
# upstream helper signatures and documentation formatting intact.
# pylint: disable=missing-function-docstring,line-too-long

from __future__ import annotations

import contextlib
import contextvars
import functools
import gc
import inspect
import warnings
from collections import deque
from collections.abc import Iterable, Sequence
from math import ceil, prod
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

UPSTREAM_TE_VERSION = "v2.16"
UPSTREAM_TE_COMMIT = "4220403e831d29e93868f7793693ea83f6b8b05b"
UPSTREAM_TE_GRAPH_PATH = "transformer_engine/pytorch/graph.py"
_MFSDP_CAPTURE_CAPABILITIES = frozenset(
    {
        "activation_recompute",
        "activation_recompute_argument_binding",
        "activation_recompute_discard_tape",
        "activation_recompute_per_callable_checkpoint_mode",
        "activation_recompute_preflight",
        "activation_recompute_phase_resolution",
        "activation_recompute_release_pending",
        "activation_recompute_region_schedule",
        "activation_recompute_three_graph",
        "capture_grad_buffer_release",
        "checkpoint_phase_marker",
        "checkpoint_region_marker",
        "fp8_activation_recompute_metadata",
        "parameter_surface_refresh",
        "registered_buffer_validation",
        "static_dgrad_reuse",
        "static_fwd_reuse",
        "static_grad_binding",
    }
)

__all__ = [
    "UPSTREAM_TE_COMMIT",
    "UPSTREAM_TE_GRAPH_PATH",
    "UPSTREAM_TE_VERSION",
    "cuda_graph_checkpoint_context_fn",
    "cuda_graph_checkpoint_phase",
    "current_cuda_graph_checkpoint_region",
    "make_graphed_callables",
    "resolve_replay_phase",
    "wrap_cuda_graph_checkpoint",
]

_torch = None
torch = None
_tree_flatten = None
_tree_unflatten = None
_graph_pool_handle = None
_TE_AVAILABLE = None
_TE_IMPORT_ERROR = None
_FP8_ACTIVATION_RECOMPUTE_PHASE = contextvars.ContextVar(
    "te_graph_fp8_activation_recompute_phase", default=None
)
_CUDA_GRAPH_CHECKPOINT_PHASE = contextvars.ContextVar(
    "mfsdp_cuda_graph_checkpoint_phase", default=None
)
_CUDA_GRAPH_CHECKPOINT_REGION = contextvars.ContextVar(
    "mfsdp_cuda_graph_checkpoint_region", default=None
)
_CUDA_GRAPH_CHECKPOINT_MODE_TYPE = None


def current_cuda_graph_checkpoint_phase() -> Optional[str]:
    """Return the active checkpoint phase, if any."""
    return _CUDA_GRAPH_CHECKPOINT_PHASE.get()


def current_cuda_graph_checkpoint_region() -> Optional[object]:
    """Return the active checkpoint region token, if any."""
    return _CUDA_GRAPH_CHECKPOINT_REGION.get()


def resolve_replay_phase(checkpoint_phase: Optional[str], grad_enabled: bool) -> str:
    """Map a call to forward, recompute, or inference replay."""
    if checkpoint_phase not in (None, "forward", "recompute"):
        raise ValueError(f"Unknown CUDA graph checkpoint phase {checkpoint_phase!r}")
    if checkpoint_phase is not None:
        return checkpoint_phase
    return "forward" if grad_enabled else "inference"


@contextlib.contextmanager
def cuda_graph_checkpoint_phase(phase: str, region_id: object):
    """Mark one checkpoint region as original forward or recompute."""
    if phase not in ("forward", "recompute"):
        raise ValueError(f"Unknown CUDA graph checkpoint phase {phase!r}")
    if region_id is None:
        raise ValueError("CUDA graph checkpoint phase requires a region token")
    phase_token = _CUDA_GRAPH_CHECKPOINT_PHASE.set(phase)
    region_token = _CUDA_GRAPH_CHECKPOINT_REGION.set(region_id)
    try:
        yield
    finally:
        _CUDA_GRAPH_CHECKPOINT_REGION.reset(region_token)
        _CUDA_GRAPH_CHECKPOINT_PHASE.reset(phase_token)


def _cuda_graph_checkpoint_phase_mode(phase, user_context=None, region_id=None):
    """Create a phase marker that remains visible through ``torch.compile``."""
    if phase not in ("forward", "recompute"):
        raise ValueError(f"Unknown CUDA graph checkpoint phase {phase!r}")
    _require_torch()
    from torch.utils._python_dispatch import TorchDispatchMode

    global _CUDA_GRAPH_CHECKPOINT_MODE_TYPE
    if _CUDA_GRAPH_CHECKPOINT_MODE_TYPE is None:

        class _CudaGraphCheckpointPhaseMode(TorchDispatchMode):
            """Pass tensor operations through while marking checkpoint phase."""

            @classmethod
            def ignore_compile_internals(cls):
                return True

            def __init__(self, checkpoint_phase, composed_context, checkpoint_region):
                super().__init__()
                self.checkpoint_phase = checkpoint_phase
                self.checkpoint_region = checkpoint_region
                self.composed_context = composed_context
                self.phase_token = None
                self.region_token = None
                self.user_context_entered = False

            def __enter__(self):
                self.phase_token = _CUDA_GRAPH_CHECKPOINT_PHASE.set(self.checkpoint_phase)
                self.region_token = _CUDA_GRAPH_CHECKPOINT_REGION.set(self.checkpoint_region)
                try:
                    if self.composed_context is not None:
                        self.composed_context.__enter__()
                        self.user_context_entered = True
                    return super().__enter__()
                except BaseException:
                    if self.user_context_entered:
                        self.composed_context.__exit__(None, None, None)
                        self.user_context_entered = False
                    _CUDA_GRAPH_CHECKPOINT_REGION.reset(self.region_token)
                    self.region_token = None
                    _CUDA_GRAPH_CHECKPOINT_PHASE.reset(self.phase_token)
                    self.phase_token = None
                    raise

            def __exit__(self, exc_type, exc_value, traceback):
                suppress = False
                try:
                    suppress = super().__exit__(exc_type, exc_value, traceback)
                finally:
                    try:
                        if self.user_context_entered:
                            suppress = (
                                self.composed_context.__exit__(exc_type, exc_value, traceback)
                                or suppress
                            )
                    finally:
                        self.user_context_entered = False
                        if self.region_token is not None:
                            _CUDA_GRAPH_CHECKPOINT_REGION.reset(self.region_token)
                            self.region_token = None
                        if self.phase_token is not None:
                            _CUDA_GRAPH_CHECKPOINT_PHASE.reset(self.phase_token)
                            self.phase_token = None
                return suppress

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                del types
                return func(*args, **({} if kwargs is None else kwargs))

        _CUDA_GRAPH_CHECKPOINT_MODE_TYPE = _CudaGraphCheckpointPhaseMode

    return _CUDA_GRAPH_CHECKPOINT_MODE_TYPE(phase, user_context, region_id)


def cuda_graph_checkpoint_context_fn():
    """Create F and RF contexts that share one checkpoint region token."""
    region_id = object()
    return (
        _cuda_graph_checkpoint_phase_mode("forward", region_id=region_id),
        _cuda_graph_checkpoint_phase_mode("recompute", region_id=region_id),
    )


def _composed_cuda_graph_checkpoint_context_fn(user_context_fn):
    """Compose checkpoint phase markers with existing user contexts."""
    forward_context, recompute_context = user_context_fn()
    region_id = object()
    return (
        _cuda_graph_checkpoint_phase_mode("forward", forward_context, region_id),
        _cuda_graph_checkpoint_phase_mode("recompute", recompute_context, region_id),
    )


def _checkpoint_callable_option(checkpoint_fn, name, default):
    """Resolve a checkpoint option from partials or its signature."""
    candidate = checkpoint_fn
    while isinstance(candidate, functools.partial):
        if candidate.keywords and name in candidate.keywords:
            return candidate.keywords[name]
        candidate = candidate.func
    try:
        parameter = inspect.signature(checkpoint_fn).parameters.get(name)
    except (TypeError, ValueError):
        parameter = None
    if parameter is not None and parameter.default is not inspect.Parameter.empty:
        return parameter.default
    return default


def wrap_cuda_graph_checkpoint(checkpoint_fn):
    """Add F and RF phase markers to a checkpoint callable.

    The original forward and its recomputation share one region token.
    """
    if not callable(checkpoint_fn):
        raise TypeError("checkpoint_fn must be callable")

    metadata_source = checkpoint_fn
    while isinstance(metadata_source, functools.partial):
        metadata_source = metadata_source.func

    configured_use_reentrant = _checkpoint_callable_option(checkpoint_fn, "use_reentrant", None)
    configured_context_fn = _checkpoint_callable_option(checkpoint_fn, "context_fn", None)
    # Define the dispatch-mode type before Dynamo starts tracing the wrapper.
    _cuda_graph_checkpoint_phase_mode("forward")

    @functools.wraps(metadata_source)
    def checkpoint_with_phase(function, *args, **kwargs):
        torch_module = _require_torch()
        use_reentrant = kwargs.get("use_reentrant", configured_use_reentrant)
        if not torch_module.is_grad_enabled():
            return checkpoint_fn(function, *args, **kwargs)
        if use_reentrant is not False:
            region_id = object()

            def reentrant_function(*function_args, **function_kwargs):
                phase = "recompute" if torch_module.is_grad_enabled() else "forward"
                with cuda_graph_checkpoint_phase(phase, region_id):
                    return function(*function_args, **function_kwargs)

            return checkpoint_fn(reentrant_function, *args, **kwargs)

        original_context_fn = kwargs.get("context_fn", configured_context_fn)

        if original_context_fn is None:
            kwargs["context_fn"] = cuda_graph_checkpoint_context_fn
        else:
            kwargs["context_fn"] = functools.partial(
                _composed_cuda_graph_checkpoint_context_fn, original_context_fn
            )
        return checkpoint_fn(function, *args, **kwargs)

    return checkpoint_with_phase


class _UnavailableTEType:
    """Placeholder used when TransformerEngine is not installed."""


DelayedScaling = _UnavailableTEType
Recipe = Any
dist_group_type = Any
TransformerEngineBaseModule = _UnavailableTEType
BasicOperation = _UnavailableTEType
Sequential = _UnavailableTEType
OperationFuser = _UnavailableTEType


class _FP8StateStub:
    skip_fp8_weight_update_tensor = None


class _FP8GlobalStateManagerStub:
    quantization_state = _FP8StateStub()

    @staticmethod
    def is_first_fp8_module() -> bool:
        """Return false when Transformer Engine is unavailable."""
        return False

    @staticmethod
    def reduce_and_update_fp8_tensors(*args, **kwargs) -> None:
        """Ignore FP8 state updates when Transformer Engine is unavailable."""
        return None

    @staticmethod
    def is_fp8_enabled() -> bool:
        """Return false when Transformer Engine is unavailable."""
        return False

    @staticmethod
    def get_fp8_recipe() -> None:
        """Return no FP8 recipe when Transformer Engine is unavailable."""
        return None

    @staticmethod
    def get_fp8_group() -> None:
        """Return no FP8 group when Transformer Engine is unavailable."""
        return None

    @staticmethod
    def add_fp8_tensors_to_global_buffer(*args, **kwargs) -> None:
        """Ignore FP8 tensor registration without Transformer Engine."""
        return None


FP8GlobalStateManager = _FP8GlobalStateManagerStub


@contextlib.contextmanager
def _null_autocast(*args, **kwargs):
    """Provide a no-op autocast context."""
    yield


autocast = _null_autocast


@contextlib.contextmanager
def activation_recompute_forward(*args, **kwargs):
    """Fallback TE recompute context when Transformer Engine is unavailable."""
    del args, kwargs
    yield


def get_default_fp8_recipe():
    """Reject FP8 recipe discovery when Transformer Engine is unavailable."""
    raise RuntimeError(
        "FP8 graph capture requires transformer_engine. Install te-graph-runtime[te] "
        "or disable FP8/TE-specific options."
    )


def _require_torch():
    """Import torch lazily so package import does not require torch initialization."""
    global _torch, torch, _tree_flatten, _tree_unflatten, _graph_pool_handle
    if _torch is None:
        import torch as imported_torch
        from torch._C import _graph_pool_handle as imported_graph_pool_handle
        from torch.utils._pytree import tree_flatten, tree_unflatten

        _torch = imported_torch
        torch = imported_torch
        _tree_flatten = tree_flatten
        _tree_unflatten = tree_unflatten
        _graph_pool_handle = imported_graph_pool_handle
    return _torch


def _load_optional_te() -> bool:
    """Load TransformerEngine internals when available, without delegating graphing."""
    global _TE_AVAILABLE, _TE_IMPORT_ERROR
    global DelayedScaling, Recipe, dist_group_type
    global autocast, activation_recompute_forward
    global FP8GlobalStateManager, get_default_fp8_recipe
    global get_all_rng_states, graph_safe_rng_available
    global TransformerEngineBaseModule, BasicOperation, Sequential, OperationFuser

    if _TE_AVAILABLE is not None:
        return _TE_AVAILABLE
    try:
        from transformer_engine.common.recipe import DelayedScaling as te_DelayedScaling
        from transformer_engine.common.recipe import Recipe as te_Recipe
        from transformer_engine.pytorch.constants import dist_group_type as te_dist_group_type
        from transformer_engine.pytorch.distributed import (
            activation_recompute_forward as te_activation_recompute_forward,
        )
        from transformer_engine.pytorch.distributed import (
            get_all_rng_states as te_get_all_rng_states,
        )
        from transformer_engine.pytorch.distributed import (
            graph_safe_rng_available as te_graph_safe_rng_available,
        )
        from transformer_engine.pytorch.module.base import (
            TransformerEngineBaseModule as te_TransformerEngineBaseModule,
        )
        from transformer_engine.pytorch.ops import Sequential as te_Sequential
        from transformer_engine.pytorch.ops.fuser import OperationFuser as te_OperationFuser
        from transformer_engine.pytorch.ops.op import BasicOperation as te_BasicOperation
        from transformer_engine.pytorch.quantization import (
            FP8GlobalStateManager as te_FP8GlobalStateManager,
        )
        from transformer_engine.pytorch.quantization import autocast as te_autocast
        from transformer_engine.pytorch.quantization import (
            get_default_fp8_recipe as te_get_default_fp8_recipe,
        )
    except Exception as exc:  # pragma: no cover - exact import failure is environment-specific
        _TE_AVAILABLE = False
        _TE_IMPORT_ERROR = exc
        return False

    DelayedScaling = te_DelayedScaling
    Recipe = te_Recipe
    dist_group_type = te_dist_group_type
    autocast = te_autocast
    activation_recompute_forward = te_activation_recompute_forward
    FP8GlobalStateManager = te_FP8GlobalStateManager
    get_default_fp8_recipe = te_get_default_fp8_recipe
    get_all_rng_states = te_get_all_rng_states
    graph_safe_rng_available = te_graph_safe_rng_available
    TransformerEngineBaseModule = te_TransformerEngineBaseModule
    BasicOperation = te_BasicOperation
    Sequential = te_Sequential
    OperationFuser = te_OperationFuser
    _TE_AVAILABLE = True
    _TE_IMPORT_ERROR = None
    return True


def _prepare_runtime() -> bool:
    """Initialize torch and load optional Transformer Engine symbols."""
    _require_torch()
    return _load_optional_te()


def get_all_rng_states() -> Dict[Any, Any]:
    """Return no tracked Transformer Engine RNG states by default."""
    return {}


def graph_safe_rng_available() -> bool:
    """Return whether torch exposes the required graph-safe RNG APIs."""
    _torch_mod = _require_torch()
    return (
        hasattr(_torch_mod.cuda.CUDAGraph, "register_generator_state")
        and hasattr(_torch_mod.Generator, "graphsafe_set_state")
        and hasattr(_torch_mod.Generator, "graphsafe_get_state")
        and hasattr(_torch_mod.Generator, "clone_state")
    )


def _get_tracked_cuda_generators(require_generators: bool = True) -> Optional[Tuple[Any, ...]]:
    """Return tracked CUDA generators supported by graph capture.

    :param require_generators: Raise for legacy tensor states when True; return
        None so the caller can use the CUDA RNG-state fallback when False.
    :type require_generators: bool, optional
    :raises RuntimeError: If the installed RNG tracker exposes legacy tensor states.
    :return: Unique tracked CUDA generators in tracker order, or None when a
        legacy tracker requires the CUDA RNG-state fallback.
    :rtype: Optional[Tuple[torch.Generator, ...]]
    """
    torch_module = _require_torch()
    generators = []
    seen_generator_ids = set()
    for tracker_name, generator in get_all_rng_states().items():
        if not isinstance(generator, torch_module.Generator):
            if not require_generators:
                return None
            raise RuntimeError(
                "CUDA graph capture requires tracked RNG values to be torch.Generator "
                "instances, but tracker "
                f"{tracker_name!r} returned {type(generator).__name__}. Legacy tensor "
                "RNG tracker states are unsupported."
            )
        if id(generator) not in seen_generator_ids:
            seen_generator_ids.add(id(generator))
            generators.append(generator)
    return tuple(generators)


def _te_required_error(feature: str) -> RuntimeError:
    detail = f" Original import error: {_TE_IMPORT_ERROR}" if _TE_IMPORT_ERROR else ""
    return RuntimeError(
        f"{feature} requires transformer_engine internals compatible with {UPSTREAM_TE_VERSION}."
        f" Install te-graph-runtime[te] or disable TE-specific graph options.{detail}"
    )


def _module_uses_delayed_wgrad(module) -> bool:
    """Return whether a TE module requests a separate wgrad graph.

    :param module: Module or operation to inspect.
    :type module: Any
    :return: Whether delayed wgrad is active.
    :rtype: bool
    """
    for owner in (module, getattr(module, "config", None), getattr(module, "wgrad_store", None)):
        if owner is None:
            continue
        for name in ("delay_wgrad_compute", "need_backward_dw"):
            value = getattr(owner, name, False)
            if bool(value() if callable(value) else value):
                return True
    return False


def _validate_fp8_activation_recompute_support() -> None:
    """Require the TE metadata operations needed by delayed-scaling recompute.

    :raises RuntimeError: If the loaded Transformer Engine cannot preserve and
        restore forward FP8 metadata across activation recomputation.
    """
    required_methods = (
        "copy_forward_fp8_meta_tensors_for_recompute",
        "get_old_fp8_meta_tensors_for_recompute",
        "restore_fp8_meta_tensors",
    )
    missing_methods = tuple(
        method
        for method in required_methods
        if not callable(getattr(FP8GlobalStateManager, method, None))
    )
    if missing_methods:
        raise RuntimeError(
            "FP8 activation recompute requires Transformer Engine metadata support; "
            "missing FP8GlobalStateManager methods: " + ", ".join(missing_methods)
        )
    first_module_flags = getattr(activation_recompute_forward, "_is_first_fp8_module", None)
    qstate = getattr(FP8GlobalStateManager, "quantization_state", None)
    if not isinstance(first_module_flags, list) or not hasattr(
        qstate, "fp8_tensors_recompute_buffer"
    ):
        raise RuntimeError(
            "FP8 activation recompute requires Transformer Engine recompute " "queue bookkeeping"
        )


def _snapshot_fp8_recompute_bookkeeping(modules):
    """Snapshot TE's Python-side activation-recompute queues.

    :param modules: Root modules included in graph capture.
    :type modules: Tuple[torch.nn.Module, ...]
    :return: First-module flags, queued metadata, and module queue keys.
    :rtype: Tuple[Any, Tuple[Tuple[Any, ...], ...], Tuple[Tuple[Dict, bool, Any], ...]]
    """
    first_module_flags = activation_recompute_forward._is_first_fp8_module
    saved_first_module_flags = tuple(first_module_flags)
    qstate = FP8GlobalStateManager.quantization_state
    recompute_buffers = qstate.fp8_tensors_recompute_buffer
    saved_recompute_buffers = tuple(tuple(queue) for queue in recompute_buffers)
    buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
    saved_meta_positions = []
    seen_meta = set()
    for root_module in modules:
        for module in root_module.modules():
            fp8_meta = getattr(module, "fp8_meta", None)
            if not isinstance(fp8_meta, dict) or id(fp8_meta) in seen_meta:
                continue
            seen_meta.add(id(fp8_meta))
            saved_meta_positions.append(
                (fp8_meta, buffer_position_key in fp8_meta, fp8_meta.get(buffer_position_key))
            )
    return saved_first_module_flags, saved_recompute_buffers, tuple(saved_meta_positions)


def _restore_fp8_recompute_bookkeeping(snapshot) -> None:
    """Restore TE's Python-side activation-recompute queues after capture.

    :param snapshot: State returned by ``_snapshot_fp8_recompute_bookkeeping``.
    :type snapshot: Tuple[Any, Tuple[Tuple[Any, ...], ...], Tuple[Tuple[Dict, bool, Any], ...]]
    """
    saved_first_module_flags, saved_recompute_buffers, saved_meta_positions = snapshot
    first_module_flags = activation_recompute_forward._is_first_fp8_module
    first_module_flags[:] = saved_first_module_flags

    qstate = FP8GlobalStateManager.quantization_state
    qstate.fp8_tensors_recompute_buffer = [
        deque(saved_queue) for saved_queue in saved_recompute_buffers
    ]
    buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
    for fp8_meta, had_position, saved_position in saved_meta_positions:
        if had_position:
            fp8_meta[buffer_position_key] = saved_position
        else:
            fp8_meta.pop(buffer_position_key, None)


def _torch_dtype_to_np_typestr(dtype):
    _torch_mod = _require_torch()
    mapping = {
        _torch_mod.float16: "<f2",
        _torch_mod.float32: "<f4",
        _torch_mod.int64: "<i8",
        _torch_mod.int32: "<i4",
        _torch_mod.int8: "|i1",
        _torch_mod.qint8: "|u1",
        _torch_mod.bool: "|b1",
        _torch_mod.bfloat16: "<f2",
    }
    float8_dtype = getattr(_torch_mod, "float8_e4m3fn", None)
    if float8_dtype is not None:
        mapping[float8_dtype] = "|i1"
    ret = mapping.get(dtype)
    if ret is None:
        supported = ", ".join(str(d) for d in mapping)
        raise TypeError(f"Unsupported dtype: {dtype}. Supported dtypes: {supported}")
    return ret


class _WeakRefTensor:
    """Tensor-like wrapper around a CUDA data pointer for graph-pool reuse."""

    def __init__(self, data_ptr: int, dtype: Any, shape: Sequence[int]):
        self._data_ptr = data_ptr
        self.dtype = dtype
        self.shape = tuple(int(i) for i in shape)

    def data_ptr(self):
        """Return the wrapped CUDA address."""
        return self._data_ptr

    def numel(self):
        """Return the number of elements in the wrapped allocation."""
        return prod(self.shape)

    @property
    def __cuda_array_interface__(self):
        """Return CUDA array metadata for the wrapped allocation."""
        return {
            "shape": self.shape,
            "typestr": _torch_dtype_to_np_typestr(self.dtype),
            "data": (self.data_ptr() if self.numel() > 0 else 0, False),
            "version": 3,
        }


def make_weak_ref(x):
    """Return a tensor-like weak reference so CUDA graph pool memory can be reused."""
    _torch_mod = _require_torch()

    def convert_to_torch_tensor(tensor):
        if isinstance(tensor, _torch_mod.Tensor):
            return tensor
        old_ptr = tensor.data_ptr()
        new_tensor = _torch_mod.as_tensor(tensor).view(tensor.dtype)
        if old_ptr != new_tensor.data_ptr():
            raise RuntimeError("Data pointer mismatch after converting to torch.Tensor")
        return new_tensor

    if isinstance(x, _torch_mod.Tensor):
        return (
            convert_to_torch_tensor(_WeakRefTensor(x.data_ptr(), x.dtype, x.shape))
            if x.is_cuda
            else x
        )
    if isinstance(x, tuple):
        return tuple(make_weak_ref(i) for i in x)
    if isinstance(x, list):
        return [make_weak_ref(i) for i in x]
    if isinstance(x, dict):
        return {k: make_weak_ref(v) for k, v in x.items()}
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    raise TypeError(
        f"Invalid type {type(x).__name__} to make weak ref. Valid types are: "
        "torch.Tensor, tuple, list, dict, int, float, bool, and None."
    )


def _registered_buffer_signature(module) -> Tuple[Tuple[Any, ...], ...]:
    """Describe recursive registered-buffer slots and storage views."""
    return tuple(
        _registered_buffer_slot_signature(slot) for slot in _registered_buffer_slots(module)
    )


def _registered_buffer_slots(module) -> Tuple[Tuple[str, Any, str], ...]:
    """Collect direct registered-buffer slots once at capture.

    :param module: Root module whose recursive buffer slots are captured.
    :type module: torch.nn.Module
    :return: Qualified name, direct owner, and slot name tuples.
    :rtype: Tuple[Tuple[str, torch.nn.Module, str], ...]
    """
    slots = []
    for module_name, submodule in module.named_modules(remove_duplicate=False):
        for buffer_name in submodule._buffers:
            qualified_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            slots.append((qualified_name, submodule, buffer_name))
    return tuple(slots)


def _registered_buffer_slot_signature(slot) -> Tuple[Any, ...]:
    """Describe one precomputed direct registered-buffer slot.

    :param slot: Qualified name, direct owner, and buffer name.
    :type slot: Tuple[str, torch.nn.Module, str]
    :return: Immutable slot metadata and storage address.
    :rtype: Tuple[Any, ...]
    """
    torch_module = _require_torch()
    qualified_name, submodule, buffer_name = slot
    if buffer_name not in submodule._buffers:
        return (qualified_name, "missing")
    buffer = submodule._buffers[buffer_name]
    if buffer is None:
        return (qualified_name, "none")
    if isinstance(buffer, torch_module.Tensor):
        return (
            qualified_name,
            "tensor",
            buffer.untyped_storage().data_ptr(),
            buffer.storage_offset(),
            tuple(buffer.shape),
            buffer.stride(),
            buffer.dtype,
            buffer.layout,
            buffer.device,
            buffer.requires_grad,
            buffer.is_conj(),
            buffer.is_neg(),
        )
    return (qualified_name, "invalid", f"{type(buffer).__module__}.{type(buffer).__qualname__}")


_IS_GRAPH_CAPTURING = False

_T = TypeVar("_T")
SingleOrTuple = Union[_T, Tuple[_T, ...]]

_CAPTURE_TIME_HOOK_KEYS = (
    "forward_pre_hooks",
    "forward_pre_hooks_with_kwargs",
    "forward_hooks",
    "forward_hooks_with_kwargs",
    "backward_pre_hooks",
    "backward_hooks",
)


def _empty_capture_time_hooks() -> Dict[str, Dict[Any, Any]]:
    return {key: {} for key in _CAPTURE_TIME_HOOK_KEYS}


def _canonicalize_capture_time_hooks(
    num_callables: int, capture_time_hooks: Optional[List[Optional[Dict[str, Dict]]]]
) -> List[Dict[str, Dict[Any, Any]]]:
    if capture_time_hooks is None:
        return [_empty_capture_time_hooks() for _ in range(num_callables)]
    if len(capture_time_hooks) != num_callables:
        raise ValueError(
            f"capture_time_hooks has {len(capture_time_hooks)} entries but there are "
            f"{num_callables} callables"
        )

    canonicalized = []
    for callable_idx, hooks in enumerate(capture_time_hooks):
        if hooks is None:
            canonicalized.append(_empty_capture_time_hooks())
            continue
        if not isinstance(hooks, dict):
            raise TypeError(
                "capture_time_hooks entries must be dicts or None, "
                f"but entry {callable_idx} has type {type(hooks).__name__}"
            )
        unknown_keys = sorted(set(hooks) - set(_CAPTURE_TIME_HOOK_KEYS))
        if unknown_keys:
            raise ValueError(
                f"Unknown capture_time_hooks keys for callable {callable_idx}: {unknown_keys}. "
                f"Supported keys are {list(_CAPTURE_TIME_HOOK_KEYS)}"
            )

        callable_hooks = _empty_capture_time_hooks()
        for key in _CAPTURE_TIME_HOOK_KEYS:
            value = hooks.get(key, {})
            if value is None:
                value = {}
            if not isinstance(value, dict):
                raise TypeError(
                    f"capture_time_hooks[{callable_idx!r}][{key!r}] must be a dict, "
                    f"but got {type(value).__name__}"
                )
            callable_hooks[key] = dict(value)
        canonicalized.append(callable_hooks)
    return canonicalized


def _check_capture_time_hook_return(value: Any, hook_name: str, detail: str) -> None:
    if value is not None:
        raise RuntimeError(
            f"capture_time_hooks {hook_name} must not return a value ({detail} must not be "
            "modified via hook return)"
        )


def set_capture_start() -> None:
    """Record beginning of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def set_capture_end() -> None:
    """Record end of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


def is_graph_capturing() -> bool:
    """Return whether within `make_graphed_callables`."""
    return _IS_GRAPH_CAPTURING


@contextlib.contextmanager
def _fp8_activation_recompute_phase(recompute_phase: Optional[bool]):
    """Select the TE FP8 activation-recompute phase for one captured call.

    :param recompute_phase: False for the initial forward, True for the
        backward-time recompute, or None outside activation recomputation.
    :type recompute_phase: Optional[bool]
    """
    token = _FP8_ACTIVATION_RECOMPUTE_PHASE.set(recompute_phase)
    try:
        yield
    finally:
        _FP8_ACTIVATION_RECOMPUTE_PHASE.reset(token)


def graph_pool_handle():
    """
    Returns an opaque token representing the id of a graph memory pool.
    """
    _require_torch()
    return _graph_pool_handle()


@contextlib.contextmanager
def _none_grad_context_wrapper(inputs):
    """
    Wrapper to set the gradients of the inputs to None,
    in case the backward pass makes grad accumulations.
    """
    original_input_grads = []
    for input_tensor in inputs:
        original_input_grads.append(input_tensor.grad)
        input_tensor.grad = None
    try:
        yield
    finally:
        for input_tensor, original_grad in zip(inputs, original_input_grads):
            input_tensor.grad = original_grad


@contextlib.contextmanager
def _static_grad_context_wrapper(inputs, grad_buffers):
    """Bind leaf gradients to static buffers during capture.

    :param inputs: Captured backward leaves.
    :type inputs: Tuple[torch.Tensor, ...]
    :param grad_buffers: Static buffer for each leaf, or None.
    :type grad_buffers: Tuple[Optional[torch.Tensor], ...]
    """
    torch_module = _require_torch()
    if len(inputs) != len(grad_buffers):
        raise ValueError("Static gradient buffers must match backward inputs")
    original_input_grads = tuple(input_tensor.grad for input_tensor in inputs)
    static_buffers = tuple(buffer for buffer in grad_buffers if buffer is not None)
    if static_buffers:
        torch_module._foreach_zero_(static_buffers)
    for input_tensor, grad_buffer in zip(inputs, grad_buffers):
        input_tensor.grad = grad_buffer
    try:
        yield
    finally:
        for input_tensor, original_grad in zip(inputs, original_input_grads):
            input_tensor.grad = original_grad


def _get_compatible_main_grad_buffer(input_tensor):
    """Get a compatible main-grad buffer.

    :param input_tensor: Candidate parameter leaf.
    :type input_tensor: torch.Tensor
    :return: Compatible buffer, or None.
    :rtype: Optional[torch.Tensor]
    """
    torch_module = _require_torch()
    if getattr(input_tensor, "_mfsdp_recorded_te_wgrad", False):
        return None
    if getattr(input_tensor, "__fsdp_param__", False) and not getattr(
        input_tensor, "overwrite_main_grad", False
    ):
        return None

    fsdp_grad_buffer = getattr(input_tensor, "_gbuf", None)
    if fsdp_grad_buffer is not None and fsdp_grad_buffer.dtype != input_tensor.dtype:
        return None

    get_main_grad = getattr(input_tensor, "get_main_grad", None)
    if not callable(get_main_grad):
        return None

    grad_buffer = get_main_grad()
    if not isinstance(grad_buffer, torch_module.Tensor):
        return None
    if grad_buffer.requires_grad:
        return None
    if grad_buffer.shape != input_tensor.shape:
        return None
    if grad_buffer.dtype != input_tensor.dtype:
        return None
    if grad_buffer.device != input_tensor.device:
        return None
    if grad_buffer.layout != torch_module.strided or input_tensor.layout != torch_module.strided:
        return None
    if grad_buffer.stride() != input_tensor.stride():
        return None
    return grad_buffer


def _parameter_allocator_signature(input_tensor):
    """Return the allocator generation and slot bound to a parameter.

    :param input_tensor: Parameter that may reference an FSDP grad buffer.
    :type input_tensor: torch.Tensor
    :return: Allocator identity, generation, and physical slot, or None.
    :rtype: Optional[Tuple[int, int, int]]
    """
    grad_buffer = getattr(input_tensor, "_gbuf", None)
    allocator = getattr(grad_buffer, "allocator", None)
    if allocator is None:
        return None
    alloc_key = getattr(grad_buffer, "alloc_key", None)
    slot_getter = getattr(allocator, "slot_id_for_key", None)
    slot_id = slot_getter(alloc_key) if callable(slot_getter) else None
    return (id(allocator), getattr(allocator, "generation", None), slot_id)


def _get_static_grad_buffers(inputs):
    """Get static gradient buffers for captured leaves.

    :param inputs: Captured backward leaves.
    :type inputs: Tuple[torch.Tensor, ...]
    :return: Compatible buffer or None for each leaf.
    :rtype: Tuple[Optional[torch.Tensor], ...]
    """
    return tuple(_get_compatible_main_grad_buffer(input_tensor) for input_tensor in inputs)


def _returned_param_grad_clone_slots(
    static_grad_inputs, module_params, static_grad_buffers, clone_param_grads_on_return
):
    """Select parameter gradients that Graphed.backward must clone.

    ``static_grad_buffers`` records the buffers bound by
    ``_static_grad_context_wrapper``. Reusing that capture-time decision is
    important for allocators with traced lifetimes: calling ``get_main_grad``
    again here would reallocate a buffer that the capture-time FSDP post-hook
    has already released.
    """
    if not clone_param_grads_on_return:
        return (False,) * len(static_grad_inputs)
    if len(static_grad_inputs) != len(static_grad_buffers):
        raise ValueError("Static gradient inputs and buffers must have matching lengths")

    module_param_start = len(static_grad_inputs) - len(module_params)
    clone_slots = []
    for idx, (grad_input, main_grad) in enumerate(zip(static_grad_inputs, static_grad_buffers)):
        if idx < module_param_start:
            clone_slots.append(False)
            continue
        param = module_params[idx - module_param_start]
        uses_main_grad = (
            grad_input is not None
            and main_grad is not None
            and grad_input.data_ptr() == main_grad.data_ptr()
        )
        clone_slots.append(
            not uses_main_grad and not getattr(param, "skip_backward_post_hook", False)
        )
    return tuple(clone_slots)


def _static_dgrad_metadata(tensor):
    """Return metadata for a safe reusable user-input gradient buffer.

    :param tensor: User input or producer output tensor.
    :type tensor: torch.Tensor
    :return: Hashable tensor metadata, or None when reuse is unsafe.
    :rtype: Optional[Tuple[Any, ...]]
    """
    torch_module = _require_torch()
    if not isinstance(tensor, torch_module.Tensor) or tensor.layout != torch_module.strided:
        return None
    if not tensor.is_cuda or tensor.is_conj() or tensor.is_neg():
        return None
    if tensor.storage_offset() != 0 or getattr(tensor, "_base", None) is not None:
        return None
    overlap_check = getattr(torch_module, "_debug_has_internal_overlap", None)
    if not callable(overlap_check) or int(overlap_check(tensor)) != 0:
        return None
    return (tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device, tensor.layout)


def _allocate_static_dgrad_reuse_buffers(
    static_input_surfaces,
    static_outputs,
    input_output_aliases,
    user_grad_indices,
    output_requires_grad,
):
    """Allocate two alternating dgrad slots for safe adjacent aliases.

    :param static_input_surfaces: Full input surface for every callable.
    :type static_input_surfaces: Sequence[Tuple[torch.Tensor, ...]]
    :param static_outputs: Static outputs for every callable.
    :type static_outputs: Sequence[Tuple[torch.Tensor, ...]]
    :param input_output_aliases: Consumer input to producer output mappings.
    :type input_output_aliases: Sequence[Dict[int, Tuple[int, int]]]
    :param user_grad_indices: User input indices that produced gradients in warmup.
    :type user_grad_indices: Sequence[Tuple[int, ...]]
    :param output_requires_grad: Logical output gradient flags for every callable.
    :type output_requires_grad: Sequence[Tuple[bool, ...]]
    :return: Reusable buffers keyed by user-input index for every callable.
    :rtype: Tuple[Dict[int, torch.Tensor], ...]
    """
    torch_module = _require_torch()
    consumers_by_output = {}
    for consumer_idx, aliases in enumerate(input_output_aliases):
        for input_idx, producer in aliases.items():
            consumers_by_output.setdefault(producer, []).append((consumer_idx, input_idx))

    component_by_callable = list(range(len(input_output_aliases)))
    for consumer_idx, aliases in enumerate(input_output_aliases):
        for producer in aliases.values():
            producer_idx, _ = producer
            if producer_idx + 1 == consumer_idx and len(consumers_by_output.get(producer, ())) == 1:
                component_by_callable[consumer_idx] = component_by_callable[producer_idx]
                break

    reused_buffers = {}
    buffers_by_callable = []
    for consumer_idx, aliases in enumerate(input_output_aliases):
        callable_buffers = {}
        lanes_by_metadata = {}
        for input_idx in sorted(user_grad_indices[consumer_idx] or ()):
            producer = aliases.get(input_idx)
            if producer is None:
                continue
            producer_idx, output_idx = producer
            if producer_idx + 1 != consumer_idx:
                continue
            if len(consumers_by_output.get(producer, ())) != 1:
                continue
            if output_idx >= len(static_outputs[producer_idx]):
                continue
            producer_grad_flags = output_requires_grad[producer_idx]
            if producer_grad_flags is None or not producer_grad_flags[output_idx]:
                continue
            input_tensor = static_input_surfaces[consumer_idx][input_idx]
            output_tensor = static_outputs[producer_idx][output_idx]
            input_metadata = _static_dgrad_metadata(input_tensor)
            if input_metadata is None or input_metadata != _static_dgrad_metadata(output_tensor):
                continue

            lane = lanes_by_metadata.get(input_metadata, 0)
            lanes_by_metadata[input_metadata] = lane + 1
            buffer_key = (
                component_by_callable[consumer_idx],
                input_metadata,
                consumer_idx % 2,
                lane,
            )
            grad_buffer = reused_buffers.get(buffer_key)
            if grad_buffer is None:
                grad_buffer = torch_module.empty_strided(
                    input_tensor.shape,
                    input_tensor.stride(),
                    dtype=input_tensor.dtype,
                    device=input_tensor.device,
                )
                reused_buffers[buffer_key] = grad_buffer
            callable_buffers[input_idx] = grad_buffer
        buffers_by_callable.append(callable_buffers)
    return tuple(buffers_by_callable)


def _refresh_module_parameter_surface(func, user_inputs, parameter_indices=None):
    """Refresh parameters after capture-time replacement hooks.

    :param func: Captured callable.
    :type func: Callable
    :param user_inputs: Flattened user inputs.
    :type user_inputs: Tuple[torch.Tensor, ...]
    :param parameter_indices: Retained parameter positions, defaults to None.
    :type parameter_indices: Optional[Tuple[int, ...]]
    :raises RuntimeError: If a retained position no longer exists.
    :return: Live parameters and the full static input surface.
    :rtype: Tuple[Tuple[torch.nn.Parameter, ...], Tuple[torch.Tensor, ...]]
    """
    torch_module = _require_torch()
    module_params = tuple(func.parameters()) if isinstance(func, torch_module.nn.Module) else ()
    if parameter_indices is not None:
        if parameter_indices and max(parameter_indices) >= len(module_params):
            raise RuntimeError(
                "Module parameter count changed after CUDA graph warmup: "
                f"retained index {max(parameter_indices)}, current count {len(module_params)}"
            )
        module_params = tuple(module_params[idx] for idx in parameter_indices)
    return module_params, user_inputs + module_params


@contextlib.contextmanager
def _graph_context_wrapper(*args, **kwargs):
    """Wrapper around `torch.cuda.graph`.

    This wrapper is a temporary workaround for a PyTorch bug:
    automatic garbage collection can destroy a graph while another
    graph is being captured, resulting in a CUDA error. See
    https://github.com/pytorch/pytorch/pull/161037.

    """
    torch_module = _require_torch()
    gc_is_enabled = gc.isenabled()
    if gc_is_enabled:
        gc.disable()
    try:
        with torch_module.cuda.graph(*args, **kwargs):
            yield
    finally:
        if gc_is_enabled:
            gc.enable()


def _activation_recompute_region_groups(
    region_indices: Optional[Sequence[int]], callable_count: int
) -> Tuple[Tuple[int, ...], ...]:
    """Group contiguous callables by checkpoint region.

    :param region_indices: Region index for each callable, defaults to one
        region per callable.
    :type region_indices: Sequence[int], optional
    :param callable_count: Number of captured callables.
    :type callable_count: int
    :raises TypeError: If a region index is not an integer.
    :raises ValueError: If region indices are missing, reordered, or non-contiguous.
    :return: Callable indices grouped in checkpoint forward order.
    :rtype: Tuple[Tuple[int, ...], ...]
    """
    if region_indices is None:
        return tuple((idx,) for idx in range(callable_count))
    if len(region_indices) != callable_count:
        raise ValueError("Checkpoint regions must match the number of callables")

    groups = []
    for callable_idx, region_idx in enumerate(region_indices):
        if not isinstance(region_idx, int):
            raise TypeError("Checkpoint region indices must be integers")
        if not groups or groups[-1][0] != region_idx:
            if region_idx != len(groups):
                raise ValueError(
                    "Checkpoint regions must be contiguous and numbered in forward order"
                )
            groups.append((region_idx, []))
        groups[-1][1].append(callable_idx)
    return tuple(tuple(indices) for _, indices in groups)


def _activation_recompute_forward_grad_modes(
    forward_grad_enabled: Union[bool, Sequence[bool]], callable_count: int
) -> Tuple[bool, ...]:
    """Canonicalize the original-forward grad mode for each callable.

    :param forward_grad_enabled: One shared mode or one mode per callable.
    :type forward_grad_enabled: Union[bool, Sequence[bool]]
    :param callable_count: Number of captured callables.
    :type callable_count: int
    :raises TypeError: If a mode is not boolean.
    :raises ValueError: If a mode sequence has the wrong length.
    :return: Original-forward grad modes in callable order.
    :rtype: Tuple[bool, ...]
    """
    if isinstance(forward_grad_enabled, bool):
        return (forward_grad_enabled,) * callable_count
    if not isinstance(forward_grad_enabled, Sequence):
        raise TypeError(
            "_activation_recompute_forward_grad_enabled must be a bool or sequence of bool"
        )
    if len(forward_grad_enabled) != callable_count:
        raise ValueError("Forward grad modes must match the number of callables")
    if not all(isinstance(mode, bool) for mode in forward_grad_enabled):
        raise TypeError("_activation_recompute_forward_grad_enabled must contain only bool values")
    return tuple(forward_grad_enabled)


def _activation_recompute_capture_schedule(
    region_groups: Sequence[Sequence[int]],
) -> Tuple[Tuple[str, int, bool], ...]:
    """Build region-ordered recompute and backward capture events.

    :param region_groups: Callable indices grouped in checkpoint forward order.
    :type region_groups: Sequence[Sequence[int]]
    :return: Phase, callable index, and keep-unsharded flag for each event.
    :rtype: Tuple[Tuple[str, int, bool], ...]
    """
    schedule = []
    for region_group in reversed(region_groups):
        if not region_group:
            raise ValueError("Checkpoint regions must not be empty")
        schedule.extend(
            ("recompute", callable_idx, callable_idx == region_group[-1])
            for callable_idx in region_group
        )
        schedule.extend(
            ("backward", callable_idx, callable_idx == region_group[-1])
            for callable_idx in reversed(region_group)
        )
    return tuple(schedule)


def _make_graphed_callables(
    callables: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    cache_quantized_params: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    _order: Optional[List[int]] = None,
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    retain_graph_in_backward: bool = False,
    _reuse_graph_input_output_buffers: bool = False,
    _clone_param_grads_on_return: bool = True,
    _input_output_aliases: Optional[Tuple[Dict[int, Tuple[int, int]], ...]] = None,
    _activation_recompute: bool = False,
    _activation_recompute_forward_grad_enabled: Union[bool, Sequence[bool]] = False,
    _activation_recompute_regions: Optional[Sequence[int]] = None,
    _activation_recompute_order_slots: Optional[Sequence[int]] = None,
    pre_warmup_hook: Optional[Callable] = None,
    post_warmup_hook: Optional[Callable] = None,
    capture_time_hooks: Optional[List[Optional[Dict[str, Dict]]]] = None,
    capture_stream: Optional[torch.cuda.Stream] = None,
    use_main_grad: bool = False,
    _tracked_generators: Optional[Tuple[Any, ...]] = None,
) -> SingleOrTuple[Callable]:
    """
    Helper method for `make_graphed_callables`
    """

    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast "
            "caching. Please set `cache_enabled=False`."
        )

    # Default is to pass no kwargs to callables
    if sample_kwargs is None:
        if isinstance(callables, tuple):
            sample_kwargs = tuple({} for _ in range(len(sample_args)))
        else:
            sample_kwargs = {}

    # Canonicalize args as tuples
    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)
        sample_kwargs = (sample_kwargs,)
    activation_recompute_region_groups = _activation_recompute_region_groups(
        _activation_recompute_regions, len(callables)
    )
    activation_recompute_forward_grad_modes = _activation_recompute_forward_grad_modes(
        _activation_recompute_forward_grad_enabled, len(callables)
    )

    capture_time_hooks = _canonicalize_capture_time_hooks(len(callables), capture_time_hooks)
    if not isinstance(_activation_recompute, bool):
        raise TypeError(
            "_activation_recompute must be a bool, "
            f"but got {type(_activation_recompute).__name__}"
        )
    if any(activation_recompute_forward_grad_modes) and not _activation_recompute:
        raise ValueError("Grad-enabled forward capture requires activation recompute")
    if _activation_recompute and num_warmup_iters < 1:
        raise ValueError("Activation recompute requires at least one warmup iteration")
    if _activation_recompute_regions is not None and not _activation_recompute:
        raise ValueError("Checkpoint regions require activation recompute")
    if _activation_recompute_order_slots is not None and (
        not _activation_recompute or _order is None
    ):
        raise ValueError("Ordered recompute slots require activation recompute and _order")
    if _activation_recompute_order_slots is not None and len(
        _activation_recompute_order_slots
    ) != len(_order):
        raise ValueError("Ordered recompute slots must match _order length")
    if _input_output_aliases is None:
        alias_count = len(sample_args) if _order is not None else len(callables)
        _input_output_aliases = tuple({} for _ in range(alias_count))
    elif _order is not None:
        if any(_input_output_aliases):
            raise ValueError("Input/output aliases are only supported without a custom order")
        _input_output_aliases = tuple({} for _ in sample_args)
    elif len(_input_output_aliases) != len(callables):
        raise ValueError("Input/output aliases must match the number of callables")
    if any(_input_output_aliases):
        if isinstance(sample_args, tuple):
            sample_args = list(sample_args)
        if isinstance(sample_kwargs, tuple):
            sample_kwargs = list(sample_kwargs)

    # Check training/inference
    is_training = all(c.training for c in callables)
    if not is_training and any(c.training for c in callables):
        raise RuntimeError(
            "make_graphed_callables only supports when modules are all in training or all in"
            " inference mode."
        )
    if _activation_recompute and _order is None:
        for callable_obj in callables:
            modules = callable_obj.modules() if isinstance(callable_obj, torch.nn.Module) else ()
            if any(_module_uses_delayed_wgrad(module) for module in modules):
                raise RuntimeError(
                    "Activation recompute does not yet support delayed backward-wgrad graphs"
                )

    # Check sizes of args
    _order_without_wgrad = None
    delay_wgrad_compute = False
    if _order is None:
        if len(sample_args) != len(callables):
            raise ValueError(
                "Expected sample_args to have the same length as callables, "
                f"but got {len(sample_args)} sample_args for {len(callables)} callables"
            )
        if len(sample_kwargs) != len(callables):
            raise ValueError(
                "Expected sample_kwargs to have the same length as callables, "
                f"but got {len(sample_kwargs)} sample_kwargs for {len(callables)} callables"
            )
    else:
        # Custom logic for interleaved pipeline parallelism
        # Note: This is tightly coupled with the Megatron-core
        # implementation of interleaved pipeline parallelism at
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py.
        # Note: The model is assumed to consist of layers
        # (corresponding to callables) that are grouped into
        # model chunks. _num_layers_per_chunk is a list of integers
        # that indicates the number of layers in each model chunk.
        # _order is a list of chunk indices (1-indexed) that
        # indicates the order in which the layers are evaluated.
        # Positive values indicate forward passes and negative
        # values indicate backward passes. Each
        # entry in sample_args corresponds to one of the forward
        # passes.
        _order_without_wgrad = []
        for c_id in _order:
            if ceil(c_id) != c_id:
                delay_wgrad_compute = True
                continue
            _order_without_wgrad.append(c_id)
        num_model_chunks = max(_order_without_wgrad)
        num_microbatches = len(_order_without_wgrad) // num_model_chunks // 2
        if num_model_chunks * num_microbatches * 2 != len(_order_without_wgrad):
            raise ValueError(
                f"Pipeline-parallel order dimension mismatch: num_model_chunks ({num_model_chunks})"
                f" * num_microbatches ({num_microbatches}) * 2 ="
                f" {num_model_chunks * num_microbatches * 2}, but len(_order_without_wgrad) ="
                f" {len(_order_without_wgrad)}"
            )

        # When delay_wgrad_compute is enabled, each layer is treated as a model chunk, which
        # allows for fine-grained graph capture order.
        if delay_wgrad_compute:
            if _num_layers_per_chunk is None:
                raise ValueError(
                    "'_num_layers_per_chunk' must be provided when delay_wgrad_compute is True."
                )
            for num_layers in _num_layers_per_chunk:
                if num_layers != 1:
                    raise ValueError(
                        "Each model chunk must have only one layer when delay_wgrad_compute is"
                        f" True, but got {num_layers} layers."
                    )

        # Determine number of layers in each model chunk.
        if _num_layers_per_chunk is None:
            if not (
                len(sample_args) * 2 >= len(_order_without_wgrad)
                and (len(sample_args) * 2 % len(_order_without_wgrad) == 0)
            ):
                raise ValueError(
                    f"{len(sample_args)} * 2 >= {len(_order_without_wgrad)} and"
                    f" {len(sample_args)} * 2 % {len(_order_without_wgrad)} == 0"
                )
            num_layers = len(sample_args) // num_model_chunks // num_microbatches
            _num_layers_per_chunk = [num_layers] * num_model_chunks
        else:
            if not (
                isinstance(_num_layers_per_chunk, int)
                or len(_num_layers_per_chunk) == num_model_chunks
            ):
                raise ValueError(
                    "If _num_layers_per_chunk is provided, it must be an integer or a list of"
                    f" {num_model_chunks} integers, but got {_num_layers_per_chunk}."
                )
            if isinstance(_num_layers_per_chunk, int):
                _num_layers_per_chunk = [_num_layers_per_chunk] * num_model_chunks
        total_num_layers = sum(_num_layers_per_chunk)
        if len(callables) != total_num_layers:
            raise ValueError(
                f"Callables should have ({total_num_layers}) "
                + f"entries when order input is provided but got {len(callables)}."
            )
        if len(sample_args) != total_num_layers * num_microbatches:
            raise ValueError(
                f"Expected {total_num_layers * num_microbatches} "
                + f"args tuple, but got {len(sample_args)}."
            )
        if _activation_recompute_order_slots is not None:
            forward_slots = [[] for _ in range(num_model_chunks)]
            backward_slots = [[] for _ in range(num_model_chunks)]
            for chunk_id, slot in zip(_order, _activation_recompute_order_slots):
                if not isinstance(chunk_id, int):
                    raise ValueError("Ordered activation recompute does not support wgrad events")
                if not isinstance(slot, int) or isinstance(slot, bool):
                    raise TypeError("Ordered recompute slots must contain only integers")
                if slot < 0 or slot >= num_microbatches:
                    raise ValueError("Ordered recompute slot is outside the microbatch range")
                chunk_idx = abs(chunk_id) - 1
                (forward_slots if chunk_id > 0 else backward_slots)[chunk_idx].append(slot)
            expected_slots = list(range(num_microbatches))
            if any(sorted(slots) != expected_slots for slots in (*forward_slots, *backward_slots)):
                raise ValueError(
                    "Ordered recompute slots must map every chunk lane once in F and B"
                )

        # Calculate the starting index of each chunk in callables for future use.
        _prefix_num_layers = [0]
        for m_chunk in range(num_model_chunks):
            num_layers = _num_layers_per_chunk[m_chunk]
            _prefix_num_layers.append(_prefix_num_layers[-1] + num_layers)

        if len(sample_kwargs) != len(sample_args):
            raise ValueError(
                "Pipeline-parallel schedule requires sample_kwargs and sample_args to have "
                f"the same length, but got {len(sample_kwargs)} sample_kwargs "
                f"for {len(sample_args)} sample_args"
            )

    # Check reuse graph conditions and reorganize sample_args and sample_kwargs.
    # Note: When capturing a graph, we hold onto the args and kwargs so we have static buffers
    # when the graph is replayed. If two model chunk microbatches have no overlap between their
    # forward and backward, then we can reduce memory usage by reusing the same static buffers.
    if _reuse_graph_input_output_buffers and _order is None and not _activation_recompute:
        raise ValueError(
            "`_reuse_graph_input_output_buffers` requires either `_order` or "
            "activation recompute."
        )
    if _reuse_graph_input_output_buffers and not is_training:
        raise RuntimeError(
            "`_reuse_graph_input_output_buffers` is only available in training mode."
        )
    if (
        _reuse_graph_input_output_buffers
        and _order is not None
        and _activation_recompute_order_slots is None
    ):
        if isinstance(sample_args, tuple):
            sample_args = list(sample_args)
        if isinstance(sample_kwargs, tuple):
            sample_kwargs = list(sample_kwargs)

        # Reorganize args and kwargs for input tensor reuse.
        # fwd_sample_qs is keyed by model chunk index. The value is a queue of tuples.
        # Each tuple contains the sample key signature and its fwd_idx. When we finish a backward
        # chunk, we pop the corresponding fwd_idx and push to the consumed_sample_q.
        # consumed_sample_q is keyed by the sample key signature. The value is a queue of the
        # fwd_idx whose backward has been called so that we can reuse the same static buffers.
        # In this way, we can reuse the same static input buffers for the non-overlapping samples
        # with the same input signature.
        fwd_sample_qs = {}
        consumed_sample_q = {}
        fwd_idx = [0] * num_model_chunks
        for c_id in _order:
            m_chunk = abs(ceil(c_id)) - 1

            if c_id > 0:
                sample_start_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                    fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk]
                )
                fwd_sample_idx = [
                    sample_start_idx + i for i in range(_num_layers_per_chunk[m_chunk])
                ]
                if m_chunk not in fwd_sample_qs:
                    fwd_sample_qs[m_chunk] = []
                for per_callable_fwd_idx in fwd_sample_idx:
                    sample_args_keys = tuple(
                        (t.shape, t.dtype, t.layout) for t in sample_args[per_callable_fwd_idx]
                    )
                    sample_kwargs_keys = tuple(
                        (k, v.shape, v.dtype, v.layout)
                        for k, v in sorted(sample_kwargs[per_callable_fwd_idx].items())
                    )
                    sample_keys = sample_args_keys + sample_kwargs_keys

                    fwd_sample_qs[m_chunk].append((sample_keys, per_callable_fwd_idx))
                    if consumed_sample_q.get(sample_keys, []):
                        reuse_fwd_idx = consumed_sample_q[sample_keys].pop(0)
                        sample_args[per_callable_fwd_idx] = sample_args[reuse_fwd_idx]
                        sample_kwargs[per_callable_fwd_idx] = sample_kwargs[reuse_fwd_idx]
                fwd_idx[m_chunk] += 1
            elif ceil(c_id) != c_id:
                continue
            else:
                num_consumed_samples = min(
                    len(fwd_sample_qs[m_chunk]), _num_layers_per_chunk[m_chunk]
                )
                for sample_keys, per_callable_fwd_idx in fwd_sample_qs[m_chunk][
                    :num_consumed_samples
                ]:
                    if sample_keys not in consumed_sample_q:
                        consumed_sample_q[sample_keys] = []
                    consumed_sample_q[sample_keys].append(per_callable_fwd_idx)
                fwd_sample_qs[m_chunk] = fwd_sample_qs[m_chunk][num_consumed_samples:]

    if cache_quantized_params:
        # Initialize flag that controls FP8 weight updates
        qstate = FP8GlobalStateManager.quantization_state
        if qstate.skip_fp8_weight_update_tensor is None:
            qstate.skip_fp8_weight_update_tensor = torch.empty(
                1, dtype=torch.float32, device="cuda"
            )
        qstate.skip_fp8_weight_update_tensor.fill_(False)

    # Check callables
    for c in callables:
        if isinstance(c, torch.nn.Module):
            if not (
                len(c._backward_hooks) == 0
                and len(c._backward_pre_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ):
                raise RuntimeError(
                    "Modules must not have hooks registered at the time they are passed. "
                    + "However, registering hooks on modules after passing them "
                    + "through make_graphed_callables is allowed. If you need hooks during "
                    + "capture, pass them with capture_time_hooks so they run outside CUDA "
                    + "graph capture and are not replayed."
                )
            if not all(b.requires_grad is False for b in c.buffers()):
                raise RuntimeError(
                    "In any :class:`~torch.nn.Module` passed to "
                    + ":func:`~make_graphed_callables`, only parameters may be trainable. "
                    + "All buffers must have ``requires_grad=False``."
                )

    # Flatten callable arguments
    per_callable_kwargs_keys = [list(kwargs.keys()) for kwargs in sample_kwargs]
    flatten_sample_args = []
    per_callable_flat_args_len = []
    per_callable_args_spec = []
    per_callable_kwargs_spec = []
    for args, kwargs, kwargs_keys in zip(sample_args, sample_kwargs, per_callable_kwargs_keys):
        flatten_arg, args_spec = _tree_flatten(args)
        flatten_kwarg, kwargs_spec = _tree_flatten([kwargs[key] for key in kwargs_keys])
        flatten_sample_args.append(tuple(flatten_arg + flatten_kwarg))
        per_callable_flat_args_len.append(len(flatten_arg))
        per_callable_args_spec.append(args_spec)
        per_callable_kwargs_spec.append(kwargs_spec)
        if not all(isinstance(arg, torch.Tensor) for arg in flatten_arg):
            raise TypeError(
                "In the beta API, sample_args "
                + "for each callable must contain only Tensors. Other types are not allowed."
            )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    # Note: These per_callable_* variables are not actually
    # per-callable, but per-forward-pass (see description of _order).
    # The names are kept for consistency with
    # PyTorch make_graphed_callables.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_user_grad_indices = [None] * len(flatten_sample_args)
    per_callable_parameter_grad_indices = [None] * len(flatten_sample_args)
    per_callable_output_requires_grad = [None] * len(flatten_sample_args)
    if _order is None:
        per_callable_module_params = [
            tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables
        ]
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))
        ]
    else:
        per_callable_module_params = []
        for m_chunk in range(num_model_chunks):
            for _ in range(num_microbatches):
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    per_callable_module_params.append(
                        tuple(callables[_prefix_num_layers[m_chunk] + l_no].parameters())
                        if isinstance(
                            callables[_prefix_num_layers[m_chunk] + l_no], torch.nn.Module
                        )
                        else ()
                    )
        if len(per_callable_module_params) != len(flatten_sample_args):
            raise ValueError(
                "Pipeline-parallel dimension mismatch: "
                f"per_callable_module_params has {len(per_callable_module_params)} entries, "
                f"but flatten_sample_args has {len(flatten_sample_args)} entries"
            )
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i]
            for i in range(len(flatten_sample_args))
        ]

    def _link_callable_inputs(func_idx, outputs_by_producer):
        """Link consumer inputs to captured producer outputs.

        :param func_idx: Consumer index.
        :type func_idx: int
        :param outputs_by_producer: Outputs keyed by producer index.
        :type outputs_by_producer: Dict[int, Sequence[torch.Tensor]]
        :return: Linked positional and keyword arguments.
        :rtype: Tuple[Tuple[Any, ...], Dict[str, Any]]
        """
        aliases = _input_output_aliases[func_idx]
        if not aliases:
            return sample_args[func_idx], sample_kwargs[func_idx]

        linked_inputs = list(flatten_sample_args[func_idx])
        for input_idx, (producer_idx, output_idx) in aliases.items():
            if producer_idx >= func_idx:
                raise ValueError("Static input producer must precede its consumer")
            producer_outputs = outputs_by_producer[producer_idx]
            if output_idx >= len(producer_outputs):
                raise ValueError("Static input alias references a missing output")
            producer_output = producer_outputs[output_idx]
            if not isinstance(producer_output, torch.Tensor):
                raise TypeError("Static input aliases must reference tensor outputs")
            output_requires_grad = producer_output.requires_grad
            if _activation_recompute:
                logical_requires_grad = per_callable_output_requires_grad[producer_idx]
                if logical_requires_grad is None:
                    raise RuntimeError("Missing logical output gradient metadata")
                output_requires_grad = logical_requires_grad[output_idx]
            linked_inputs[input_idx] = producer_output.detach().requires_grad_(output_requires_grad)

        flat_args_len = per_callable_flat_args_len[func_idx]
        args = _tree_unflatten(linked_inputs[:flat_args_len], per_callable_args_spec[func_idx])
        kwarg_values = _tree_unflatten(
            linked_inputs[flat_args_len:], per_callable_kwargs_spec[func_idx]
        )
        kwargs = dict(zip(per_callable_kwargs_keys[func_idx], kwarg_values))
        sample_args[func_idx] = args
        sample_kwargs[func_idx] = kwargs
        flatten_sample_args[func_idx] = tuple(linked_inputs)
        per_callable_static_input_surfaces[func_idx] = (
            tuple(linked_inputs) + per_callable_module_params[func_idx]
        )
        return args, kwargs

    graph_count = len(flatten_sample_args)
    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(graph_count)]
    recompute_graphs = [
        torch.cuda.CUDAGraph() if _activation_recompute else None for _ in range(graph_count)
    ]
    per_callable_recompute_outputs = [None] * graph_count
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(graph_count)]
    bwd_dw_graphs = [torch.cuda.CUDAGraph() for _ in range(graph_count)]
    graph_replay_states = [
        {
            "generation": 0,
            "pending_generation": None,
            "phase": "idle",
            "forward_owns_backward": False,
            "recompute_rng_states": (),
            "pending_region": None,
        }
        for _ in range(graph_count)
    ]
    graph_callables = [None for _ in range(graph_count)]

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    callable_uses_default_rng = [False] * graph_count

    def discard_capture_saved_tensor(tensor):
        """Discard a tensor saved only by capture-time forward."""
        del tensor
        return None

    def reject_discarded_saved_tensor(packed):
        """Reject backward through the discarded capture-time tape."""
        del packed
        raise RuntimeError("Discarded capture-forward tensors cannot be unpacked")

    if _tracked_generators is None:
        discovered_generators = _get_tracked_cuda_generators(
            require_generators=_activation_recompute
        )
        tracked_generators = discovered_generators or ()
    else:
        tracked_generators = _tracked_generators
    per_callable_used_tracked_generators = [set() for _ in range(graph_count)]
    torch.cuda.synchronize()

    # Get warmup func and func_idx.
    warmup_func_idx = []
    warmup_func = []
    if _order is None:
        for func_idx, func in enumerate(callables):
            warmup_func_idx.append(func_idx)
            warmup_func.append(func)
    else:
        fwd_idx = [0] * num_model_chunks
        for order_idx, c_id in enumerate(_order):
            if c_id > 0:
                m_chunk = c_id - 1
                forward_slot = (
                    _activation_recompute_order_slots[order_idx]
                    if _activation_recompute_order_slots is not None
                    else fwd_idx[m_chunk]
                )
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    func = callables[_prefix_num_layers[m_chunk] + l_no]
                    func_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        forward_slot * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    warmup_func_idx.append(func_idx)
                    warmup_func.append(func)
                fwd_idx[m_chunk] += 1
    if len(warmup_func) != len(sample_args):
        raise ValueError(f"Warmup runs {len(warmup_func)} don't match args {len(sample_args)}.")
    if len(warmup_func_idx) != len(set(warmup_func_idx)):
        raise RuntimeError(
            f"Warmup runs {len(warmup_func)} but only {len(set(warmup_func_idx))} are unique."
        )

    # Filter the TE modules that cudagraph can access.
    visited_te_modules = {}
    need_bwd_dw_graph = {}

    def _call_capture_time_forward_pre_hooks(callable_idx, func, args, kwargs) -> None:
        hooks = capture_time_hooks[callable_idx]
        with_kwargs = hooks["forward_pre_hooks_with_kwargs"]
        for hook_id, hook in hooks["forward_pre_hooks"].items():
            if hook_id in with_kwargs:
                _check_capture_time_hook_return(
                    hook(func, args, kwargs), "forward_pre_hooks", "args/kwargs"
                )
            else:
                _check_capture_time_hook_return(hook(func, args), "forward_pre_hooks", "args")

    def _call_capture_time_forward_hooks(callable_idx, func, args, kwargs, outputs) -> None:
        hooks = capture_time_hooks[callable_idx]
        with_kwargs = hooks["forward_hooks_with_kwargs"]
        for hook_id, hook in hooks["forward_hooks"].items():
            if hook_id in with_kwargs:
                _check_capture_time_hook_return(
                    hook(func, args, kwargs, outputs), "forward_hooks", "output"
                )
            else:
                _check_capture_time_hook_return(
                    hook(func, args, outputs), "forward_hooks", "output"
                )

    def _call_capture_time_backward_pre_hooks(callable_idx, func, grad_outputs) -> None:
        for hook in capture_time_hooks[callable_idx]["backward_pre_hooks"].values():
            _check_capture_time_hook_return(
                hook(func, grad_outputs), "backward_pre_hooks", "grad_output"
            )

    def _call_capture_time_backward_hooks(callable_idx, func, grad_inputs, grad_outputs) -> None:
        for hook in capture_time_hooks[callable_idx]["backward_hooks"].values():
            _check_capture_time_hook_return(
                hook(func, grad_inputs, grad_outputs), "backward_hooks", "grad_input"
            )

    def _make_grad_outputs(outputs):
        return tuple(
            torch.empty_like(o) if o is not None and o.requires_grad else None for o in outputs
        )

    def _run_warmup_forward(
        func_idx,
        func,
        callable_idx,
        outputs_by_producer,
        register_discovery_hooks=True,
        record_output_requires_grad=True,
    ):
        args, kwargs = _link_callable_inputs(func_idx, outputs_by_producer)

        def hook_fn(module, inputs, outputs, func_idx=func_idx):  # pylint: disable=unused-argument
            modules = set()
            if isinstance(module, TransformerEngineBaseModule):
                modules.add(module)
            # If forward is called on a BasicOperation directly the hook will run.
            elif isinstance(module, BasicOperation):
                modules.add(module)
            elif hasattr(module, "need_backward_dw") and hasattr(module, "backward_dw"):
                modules.add(module)
            # If forward is called on a te.ops.Sequential it is not called on its constituent ops.
            elif isinstance(module, Sequential):
                if module._module_groups is None:
                    raise RuntimeError(
                        "module._module_groups should have been initialized by warmup"
                    )
                for module_group in module._module_groups:
                    if isinstance(module_group, OperationFuser):
                        for basic_op in module_group._basic_ops:
                            modules.add(basic_op)
            if modules:
                if func_idx not in visited_te_modules:
                    visited_te_modules[func_idx] = modules
                else:
                    visited_te_modules[func_idx].update(modules)

        _call_capture_time_forward_pre_hooks(callable_idx, func, args, kwargs)
        hooks = []
        if register_discovery_hooks and isinstance(func, torch.nn.Module):
            for module in func.modules():
                hooks.append(module.register_forward_hook(hook_fn))
        rng_state = torch.cuda.get_rng_state() if _activation_recompute else None
        tracked_rng_states = (
            tuple(generator.get_state() for generator in tracked_generators)
            if _activation_recompute
            else ()
        )
        outputs = func(*args, **kwargs)
        if rng_state is not None and not torch.equal(rng_state, torch.cuda.get_rng_state()):
            callable_uses_default_rng[func_idx] = True
        for generator, state in zip(tracked_generators, tracked_rng_states):
            if not torch.equal(state, generator.get_state()):
                per_callable_used_tracked_generators[func_idx].add(generator)
        for hook in hooks:
            hook.remove()
        _call_capture_time_forward_hooks(callable_idx, func, args, kwargs, outputs)
        flatten_outputs, _ = _tree_flatten(outputs)
        if record_output_requires_grad:
            output_requires_grad = tuple(
                isinstance(output, torch.Tensor) and output.requires_grad
                for output in flatten_outputs
            )
            recorded_output_requires_grad = per_callable_output_requires_grad[func_idx]
            if recorded_output_requires_grad is None:
                per_callable_output_requires_grad[func_idx] = output_requires_grad
            elif recorded_output_requires_grad != output_requires_grad:
                raise RuntimeError(
                    "Callable output gradient metadata changed across CUDA graph warmup"
                )
        return flatten_outputs

    def _run_warmup_backward(func_idx, func, outputs, warmup_iter, callable_idx) -> None:
        outputs_requiring_grad = tuple(o for o in outputs if o is not None and o.requires_grad)
        grad_outputs = _make_grad_outputs(outputs)

        if _activation_recompute and any(
            _module_uses_delayed_wgrad(module) for module in visited_te_modules.get(func_idx, set())
        ):
            raise RuntimeError(
                "Activation recompute does not yet support delayed backward-wgrad graphs"
            )

        _call_capture_time_backward_pre_hooks(callable_idx, func, grad_outputs)
        live_module_params, static_input_surface = _refresh_module_parameter_surface(
            func, flatten_sample_args[func_idx]
        )
        inputs = tuple(i for i in static_input_surface if i is not None and i.requires_grad)
        backward_autocast_context = (
            torch.amp.autocast("cuda", enabled=False)
            if _activation_recompute
            else contextlib.nullcontext()
        )
        with _none_grad_context_wrapper(inputs), backward_autocast_context:
            torch.autograd.backward(
                outputs_requiring_grad, grad_tensors=tuple(o for o in grad_outputs if o is not None)
            )
            grad_inputs = tuple(input.grad for input in inputs)
        _call_capture_time_backward_hooks(callable_idx, func, grad_inputs, grad_outputs)

        # Filter module params that get None grad from grad_inputs and remove them
        # from static_input_surface. This is to ensure that the backward hooks
        # registered to these params are not wrongly triggered.
        required_grad_input_indices = [
            idx
            for idx, arg in enumerate(static_input_surface)
            if isinstance(arg, torch.Tensor) and arg.requires_grad
        ]
        grad_by_surface_index = dict(zip(required_grad_input_indices, grad_inputs))
        user_input_count = len(flatten_sample_args[func_idx])
        user_grad_indices = tuple(
            surface_idx
            for surface_idx in required_grad_input_indices
            if surface_idx < user_input_count and grad_by_surface_index[surface_idx] is not None
        )
        recorded_user_indices = per_callable_user_grad_indices[func_idx]
        if recorded_user_indices is None:
            per_callable_user_grad_indices[func_idx] = user_grad_indices
        elif recorded_user_indices != user_grad_indices:
            raise RuntimeError(
                "User inputs producing gradients changed across CUDA graph warmup "
                f"iterations: expected {recorded_user_indices}, found {user_grad_indices} "
                f"at iteration {warmup_iter}"
            )
        for surface_idx in required_grad_input_indices:
            if surface_idx < user_input_count and grad_by_surface_index[surface_idx] is None:
                if not allow_unused_input:
                    raise RuntimeError(
                        "The input tensor requires grad, but the grad is None after backward pass."
                    )

        parameter_grad_indices = tuple(
            param_idx
            for param_idx, param in enumerate(live_module_params)
            if param.requires_grad
            and grad_by_surface_index[user_input_count + param_idx] is not None
        )
        recorded_indices = per_callable_parameter_grad_indices[func_idx]
        if recorded_indices is None:
            per_callable_parameter_grad_indices[func_idx] = parameter_grad_indices
        elif recorded_indices != parameter_grad_indices:
            raise RuntimeError(
                "Module parameters producing gradients changed across CUDA graph warmup "
                f"iterations: expected {recorded_indices}, found {parameter_grad_indices} "
                f"at iteration {warmup_iter}"
            )
        module_params_with_grad = tuple(
            live_module_params[param_idx] for param_idx in parameter_grad_indices
        )
        per_callable_module_params[func_idx] = module_params_with_grad
        per_callable_static_input_surfaces[func_idx] = (
            flatten_sample_args[func_idx] + module_params_with_grad
        )

        # Run wgrad. This is essential for some TE modules when they have
        # delay_wgrad_compute enabled.
        need_backward_dw = False
        for module in visited_te_modules.get(func_idx, set()):
            if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                need_backward_dw = True
                module.backward_dw()
        need_bwd_dw_graph[func_idx] = need_backward_dw

    def _run_warmup_iteration(warmup_iter, register_discovery_hooks):
        if _order is None:
            if _activation_recompute:
                outputs_by_producer = {}
                for func_idx, func in zip(warmup_func_idx, warmup_func):
                    outputs = _run_warmup_forward(
                        func_idx,
                        func,
                        func_idx,
                        outputs_by_producer,
                        register_discovery_hooks=register_discovery_hooks,
                    )
                    if is_training:
                        _run_warmup_backward(func_idx, func, outputs, warmup_iter, func_idx)
                    outputs_by_producer[func_idx] = tuple(
                        (
                            output.detach().requires_grad_(output.requires_grad)
                            if isinstance(output, torch.Tensor)
                            else output
                        )
                        for output in outputs
                    )
                outputs_by_producer = {}
                for func_idx, func in zip(warmup_func_idx, warmup_func):
                    forward_grad_enabled = activation_recompute_forward_grad_modes[func_idx]
                    forward_context = (
                        torch.enable_grad() if forward_grad_enabled else torch.no_grad()
                    )
                    saved_tensor_context = (
                        torch.autograd.graph.saved_tensors_hooks(
                            discard_capture_saved_tensor, reject_discarded_saved_tensor
                        )
                        if forward_grad_enabled
                        else contextlib.nullcontext()
                    )
                    with forward_context, saved_tensor_context:
                        outputs = _run_warmup_forward(
                            func_idx,
                            func,
                            func_idx,
                            outputs_by_producer,
                            register_discovery_hooks=False,
                            record_output_requires_grad=False,
                        )
                    outputs_by_producer[func_idx] = tuple(
                        output.detach() if isinstance(output, torch.Tensor) else output
                        for output in outputs
                    )
                return

            warmup_outputs = []
            outputs_by_producer = {}
            for func_idx, func in zip(warmup_func_idx, warmup_func):
                outputs = _run_warmup_forward(
                    func_idx,
                    func,
                    func_idx,
                    outputs_by_producer,
                    register_discovery_hooks=register_discovery_hooks,
                )
                outputs_by_producer[func_idx] = outputs
                warmup_outputs.append((func_idx, func, outputs))
            if is_training:
                for func_idx, func, outputs in reversed(warmup_outputs):
                    _run_warmup_backward(func_idx, func, outputs, warmup_iter, func_idx)
            return

        per_fwd_outputs = {}
        fwd_idx = [0] * num_model_chunks
        bwd_idx = [0] * num_model_chunks
        for order_idx, c_id in enumerate(_order):
            if c_id > 0:
                m_chunk = c_id - 1
                forward_slot = (
                    _activation_recompute_order_slots[order_idx]
                    if _activation_recompute_order_slots is not None
                    else fwd_idx[m_chunk]
                )
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    callable_idx = _prefix_num_layers[m_chunk] + l_no
                    per_callable_fwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        forward_slot * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    func = callables[callable_idx]
                    if _activation_recompute:
                        forward_grad_enabled = activation_recompute_forward_grad_modes[callable_idx]
                        forward_context = (
                            torch.enable_grad() if forward_grad_enabled else torch.no_grad()
                        )
                        saved_tensor_context = (
                            torch.autograd.graph.saved_tensors_hooks(
                                discard_capture_saved_tensor, reject_discarded_saved_tensor
                            )
                            if forward_grad_enabled
                            else contextlib.nullcontext()
                        )
                        with forward_context, saved_tensor_context:
                            outputs = _run_warmup_forward(
                                per_callable_fwd_idx,
                                func,
                                callable_idx,
                                per_fwd_outputs,
                                register_discovery_hooks=register_discovery_hooks,
                                record_output_requires_grad=False,
                            )
                        per_fwd_outputs[per_callable_fwd_idx] = tuple(
                            output.detach() if isinstance(output, torch.Tensor) else output
                            for output in outputs
                        )
                    else:
                        outputs = _run_warmup_forward(
                            per_callable_fwd_idx,
                            func,
                            callable_idx,
                            per_fwd_outputs,
                            register_discovery_hooks=register_discovery_hooks,
                        )
                        per_fwd_outputs[per_callable_fwd_idx] = outputs
                fwd_idx[m_chunk] += 1
            elif ceil(c_id) == c_id:
                if is_training:
                    m_chunk = -c_id - 1
                    backward_slot = (
                        _activation_recompute_order_slots[order_idx]
                        if _activation_recompute_order_slots is not None
                        else bwd_idx[m_chunk]
                    )
                    if _activation_recompute:
                        recompute_outputs = {}
                        for l_no in range(_num_layers_per_chunk[m_chunk]):
                            callable_idx = _prefix_num_layers[m_chunk] + l_no
                            per_callable_bwd_idx = (
                                _prefix_num_layers[m_chunk] * num_microbatches
                            ) + (backward_slot * _num_layers_per_chunk[m_chunk] + l_no)
                            func = callables[callable_idx]
                            outputs = _run_warmup_forward(
                                per_callable_bwd_idx,
                                func,
                                callable_idx,
                                recompute_outputs,
                                register_discovery_hooks=False,
                            )
                            recompute_outputs[per_callable_bwd_idx] = outputs
                    for l_no in reversed(range(_num_layers_per_chunk[m_chunk])):
                        callable_idx = _prefix_num_layers[m_chunk] + l_no
                        per_callable_bwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                            backward_slot * _num_layers_per_chunk[m_chunk] + l_no
                        )
                        func = callables[callable_idx]
                        outputs = (
                            recompute_outputs[per_callable_bwd_idx]
                            if _activation_recompute
                            else per_fwd_outputs[per_callable_bwd_idx]
                        )
                        _run_warmup_backward(
                            per_callable_bwd_idx, func, outputs, warmup_iter, callable_idx
                        )
                    bwd_idx[m_chunk] += 1

    # Run warmup on the same stream as capture so workspace buffers
    # stay in the same CUDA context and don't need re-allocation.
    capture_stream = capture_stream or torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        if pre_warmup_hook is not None:
            pre_warmup_hook()

        for warmup_iter in range(num_warmup_iters):
            _run_warmup_iteration(warmup_iter, register_discovery_hooks=True)

        # TE discovery temporarily registers forward hooks, and Dynamo guards
        # compiled modules on hook state. Capture runs after those hooks are
        # removed, so warm the capture-equivalent specialization as well.
        compiled_callables = any(
            getattr(func, "_compiled_call_impl", None) is not None
            or hasattr(getattr(func, "forward", None), "_torchdynamo_orig_callable")
            for func in callables
        )
        if num_warmup_iters > 0 and compiled_callables:
            _run_warmup_iteration(num_warmup_iters, register_discovery_hooks=False)

        if post_warmup_hook is not None:
            post_warmup_hook()
    torch.cuda.synchronize()

    if _activation_recompute and any(need_bwd_dw_graph.values()):
        raise RuntimeError(
            "Activation recompute does not yet support delayed backward-wgrad graphs"
        )
    if _activation_recompute and (
        any(callable_uses_default_rng) or any(per_callable_used_tracked_generators)
    ):
        raise RuntimeError(
            "Activation-recompute CUDA graphs do not support RNG-consuming callables; "
            "captured recompute cannot restore the original forward RNG offset reliably"
        )

    per_callable_recompute_rng_pairs = [()] * graph_count
    if graph_safe_rng_available():
        default_generator = torch.cuda.default_generators[torch.cuda.current_device()]
        for graph_idx in range(graph_count):
            canonical_generators = list(
                (
                    generator
                    for generator in tracked_generators
                    if generator in per_callable_used_tracked_generators[graph_idx]
                )
                if num_warmup_iters > 0
                else tracked_generators
            )
            if _activation_recompute and callable_uses_default_rng[graph_idx]:
                canonical_generators.append(default_generator)
            canonical_generators = tuple(dict.fromkeys(canonical_generators))
            for generator in tracked_generators:
                fwd_graphs[graph_idx].register_generator_state(generator)
                bwd_graphs[graph_idx].register_generator_state(generator)
                bwd_dw_graphs[graph_idx].register_generator_state(generator)
                if recompute_graphs[graph_idx] is not None:
                    recompute_graphs[graph_idx].register_generator_state(generator)
            if _activation_recompute:
                recompute_pairs = []
                for generator in canonical_generators:
                    recompute_generator = generator.graphsafe_get_state().clone_state()
                    recompute_graph = recompute_graphs[graph_idx]
                    if recompute_graph is None:
                        raise RuntimeError(
                            "Activation recompute requires a recompute-forward graph"
                        )
                    recompute_graph.register_generator_state(recompute_generator)
                    recompute_pairs.append((generator, recompute_generator))
                per_callable_recompute_rng_pairs[graph_idx] = tuple(recompute_pairs)
    elif _activation_recompute and (tracked_generators or any(callable_uses_default_rng)):
        raise RuntimeError("Activation recompute with CUDA RNG requires graph-safe generator state")

    import gc

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    if _order is not None:  # pylint: disable=too-many-nested-blocks
        per_callable_static_outputs = [None] * len(flatten_sample_args)
        per_callable_output_unflatten_spec = [None] * len(flatten_sample_args)
        per_callable_static_grad_outputs = [None] * len(flatten_sample_args)
        per_callable_static_grad_inputs = [None] * len(flatten_sample_args)
        per_callable_returned_param_grad_clone_slots = [None] * len(flatten_sample_args)
        fwd_idx = [0] * num_model_chunks
        bwd_idx = [0] * num_model_chunks
        static_grad_outputs_dict = {}
        wgrad_validation_list = [None] * len(_order)
        previous_chunk_last_callable_bwd_idx = None
        for i, c_id in enumerate(_order):
            if c_id > 0:
                if not isinstance(c_id, int):
                    raise TypeError(
                        f"Forward order value must be an integer, but got {type(c_id).__name__}."
                    )
                # Capture forward graph for model chunk c_id, microbatch fwd_idx[c_id-1]
                m_chunk = c_id - 1
                forward_slot = (
                    _activation_recompute_order_slots[i]
                    if _activation_recompute_order_slots is not None
                    else fwd_idx[m_chunk]
                )
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    callable_idx = _prefix_num_layers[m_chunk] + l_no
                    func = callables[callable_idx]
                    per_callable_fwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        forward_slot * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    args = sample_args[per_callable_fwd_idx]
                    kwargs = sample_kwargs[per_callable_fwd_idx]
                    fwd_graph = fwd_graphs[per_callable_fwd_idx]
                    _call_capture_time_forward_pre_hooks(callable_idx, func, args, kwargs)
                    forward_grad_enabled = activation_recompute_forward_grad_modes[callable_idx]
                    saved_tensor_context = (
                        torch.autograd.graph.saved_tensors_hooks(
                            discard_capture_saved_tensor, reject_discarded_saved_tensor
                        )
                        if _activation_recompute and forward_grad_enabled
                        else contextlib.nullcontext()
                    )
                    forward_context = (
                        torch.enable_grad()
                        if _activation_recompute and forward_grad_enabled
                        else torch.no_grad() if _activation_recompute else contextlib.nullcontext()
                    )
                    with (
                        saved_tensor_context,
                        forward_context,
                        _graph_context_wrapper(fwd_graph, pool=mempool, stream=capture_stream),
                        _fp8_activation_recompute_phase(False if _activation_recompute else None),
                    ):
                        outputs = func(*args, **kwargs)
                        if _activation_recompute:
                            flat_outputs_with_history, output_spec = _tree_flatten(outputs)
                            outputs = _tree_unflatten(
                                tuple(
                                    output.detach() if isinstance(output, torch.Tensor) else output
                                    for output in flat_outputs_with_history
                                ),
                                output_spec,
                            )
                    _call_capture_time_forward_hooks(callable_idx, func, args, kwargs, outputs)
                    flatten_outputs, spec = _tree_flatten(outputs)
                    per_callable_static_outputs[per_callable_fwd_idx] = tuple(flatten_outputs)
                    per_callable_output_unflatten_spec[per_callable_fwd_idx] = spec
                    graph_callables[per_callable_fwd_idx] = func
                fwd_idx[m_chunk] += 1
            else:
                # Capture backward graph for model chunk c_id, microbatch bwd_idx[-c_id-1]
                m_chunk = -ceil(c_id) - 1
                backward_slot = (
                    _activation_recompute_order_slots[i]
                    if _activation_recompute_order_slots is not None
                    else bwd_idx[m_chunk]
                )
                previous_per_callable_bwd_idx = None
                if _activation_recompute and ceil(c_id) == c_id:
                    for l_no in range(_num_layers_per_chunk[m_chunk]):
                        callable_idx = _prefix_num_layers[m_chunk] + l_no
                        per_callable_bwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                            backward_slot * _num_layers_per_chunk[m_chunk] + l_no
                        )
                        func = callables[callable_idx]
                        args = sample_args[per_callable_bwd_idx]
                        kwargs = sample_kwargs[per_callable_bwd_idx]
                        _call_capture_time_forward_pre_hooks(callable_idx, func, args, kwargs)
                        module_params, static_input_surface = _refresh_module_parameter_surface(
                            func,
                            flatten_sample_args[per_callable_bwd_idx],
                            per_callable_parameter_grad_indices[per_callable_bwd_idx],
                        )
                        per_callable_module_params[per_callable_bwd_idx] = module_params
                        per_callable_static_input_surfaces[per_callable_bwd_idx] = (
                            static_input_surface
                        )
                        recompute_graph = recompute_graphs[per_callable_bwd_idx]
                        if recompute_graph is None:
                            raise RuntimeError(
                                "Activation recompute requires a recompute-forward graph"
                            )
                        recompute_rng_pairs = per_callable_recompute_rng_pairs[per_callable_bwd_idx]
                        canonical_rng_states = tuple(
                            generator.graphsafe_get_state() for generator, _ in recompute_rng_pairs
                        )
                        for generator, recompute_generator in recompute_rng_pairs:
                            generator.graphsafe_set_state(recompute_generator)
                        try:
                            with (
                                _graph_context_wrapper(
                                    recompute_graph, pool=mempool, stream=capture_stream
                                ),
                                _fp8_activation_recompute_phase(True),
                            ):
                                recompute_outputs = func(*args, **kwargs)
                        finally:
                            for (generator, _), canonical_state in zip(
                                recompute_rng_pairs, canonical_rng_states
                            ):
                                generator.graphsafe_set_state(canonical_state)
                        recompute_flat_outputs, _ = _tree_flatten(recompute_outputs)
                        per_callable_recompute_outputs[per_callable_bwd_idx] = tuple(
                            recompute_flat_outputs
                        )
                        _call_capture_time_forward_hooks(
                            callable_idx, func, args, kwargs, recompute_outputs
                        )
                        del recompute_flat_outputs
                        del recompute_outputs
                for l_no in list(reversed(range(_num_layers_per_chunk[m_chunk]))):
                    callable_idx = _prefix_num_layers[m_chunk] + l_no
                    per_callable_bwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        backward_slot * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    if ceil(c_id) == c_id and need_bwd_dw_graph.get(per_callable_bwd_idx, False):
                        # Check if bwd graph has corresponding wgrad graph:
                        # Number of dgrad backward graphs should be equal to number of
                        # wgrad backward graphs.
                        # Note: For MCore, the validation rule is more strict (the next backward
                        # of dgrad graph must be corresponding wgrad graph).
                        if wgrad_validation_list[i] is None:
                            same_bwd_c_id_list = [i]
                            num_wgrad_c_id = 0
                            for idx in range(i + 1, len(_order)):
                                if _order[idx] > 0:
                                    continue
                                if _order[idx] == c_id:
                                    same_bwd_c_id_list.append(idx)
                                if _order[idx] + 0.5 == c_id:
                                    num_wgrad_c_id += 1
                                if len(same_bwd_c_id_list) == num_wgrad_c_id:
                                    for same_c_id_idx in same_bwd_c_id_list:
                                        wgrad_validation_list[same_c_id_idx] = True
                                    break
                                if len(same_bwd_c_id_list) < num_wgrad_c_id:
                                    # It's impossible to have more wgrad than dgrad.
                                    wgrad_validation_list[i] = False
                                    break
                            if wgrad_validation_list[i] is None:
                                wgrad_validation_list[i] = False
                            if not wgrad_validation_list[i]:
                                raise RuntimeError(
                                    f"Number of wgrad graph({num_wgrad_c_id}) doesn't match number "
                                    f"of dgrad graphs ({len(same_bwd_c_id_list)}) for chunk {c_id}."
                                )
                    elif ceil(c_id) != c_id:
                        per_callable_bwd_idx -= _num_layers_per_chunk[m_chunk]
                        if not is_training:
                            raise RuntimeError("Only training mode supports backward_dw.")
                        # If no one module needs the backward_dw, the bwd_dw_graph will be empty.
                        # Skip empty backward_dw graphs. Its order value is c_id - 0.5.
                        if ceil(c_id) - c_id != 0.5:
                            raise ValueError(
                                "The order diff of wgrad and dgrad must be 0.5, "
                                f"get {ceil(c_id) - c_id}."
                            )
                        if not need_bwd_dw_graph.get(per_callable_bwd_idx, False):
                            raise RuntimeError(
                                "No module needs wgrad computation but get float in order"
                            )
                        bwd_dw_graph = bwd_dw_graphs[per_callable_bwd_idx]
                        with _graph_context_wrapper(
                            bwd_dw_graph, pool=mempool, stream=capture_stream
                        ):
                            for module in visited_te_modules[per_callable_bwd_idx]:
                                if (
                                    hasattr(module, "need_backward_dw")
                                    and module.need_backward_dw()
                                ):
                                    module.backward_dw()
                        continue

                    static_input_surface = per_callable_static_input_surfaces[per_callable_bwd_idx]
                    static_outputs = (
                        per_callable_recompute_outputs[per_callable_bwd_idx]
                        if _activation_recompute
                        else per_callable_static_outputs[per_callable_bwd_idx]
                    )
                    bwd_graph = bwd_graphs[per_callable_bwd_idx]
                    # For now, assumes all static_outputs require grad
                    if _reuse_graph_input_output_buffers:
                        # Note for _reuse_graph_input_output_buffers: grad output is only used
                        # within backward, so we can reuse the same static buffers every time.
                        static_grad_outputs_keys = tuple(
                            (o.shape, o.dtype, o.layout)
                            for o in static_outputs
                            if o is not None and o.requires_grad
                        )
                        if static_grad_outputs_keys in static_grad_outputs_dict:
                            static_grad_outputs = static_grad_outputs_dict[static_grad_outputs_keys]
                        else:
                            static_grad_outputs = tuple(
                                torch.empty_like(o) if o is not None and o.requires_grad else None
                                for o in static_outputs
                            )
                            static_grad_outputs_dict[static_grad_outputs_keys] = static_grad_outputs
                    else:
                        static_grad_outputs = tuple(
                            torch.empty_like(o) if o is not None and o.requires_grad else None
                            for o in static_outputs
                        )
                    if is_training:
                        func = graph_callables[per_callable_bwd_idx]
                        _call_capture_time_backward_pre_hooks(
                            callable_idx, func, static_grad_outputs
                        )
                        module_params, static_input_surface = _refresh_module_parameter_surface(
                            func,
                            flatten_sample_args[per_callable_bwd_idx],
                            per_callable_parameter_grad_indices[per_callable_bwd_idx],
                        )
                        per_callable_module_params[per_callable_bwd_idx] = module_params
                        per_callable_static_input_surfaces[per_callable_bwd_idx] = (
                            static_input_surface
                        )
                        inputs = tuple(
                            i for i in static_input_surface if i is not None and i.requires_grad
                        )
                        input_grad_buffers = _get_static_grad_buffers(inputs)
                        # Enter graph capture first so buffer zeroing is recorded.
                        backward_autocast_context = (
                            torch.amp.autocast("cuda", enabled=False)
                            if _activation_recompute
                            else contextlib.nullcontext()
                        )
                        with (
                            _graph_context_wrapper(bwd_graph, pool=mempool, stream=capture_stream),
                            _static_grad_context_wrapper(inputs, input_grad_buffers),
                            _fp8_activation_recompute_phase(
                                True if _activation_recompute else None
                            ),
                            backward_autocast_context,
                        ):
                            torch.autograd.backward(
                                tuple(
                                    o for o in static_outputs if o is not None and o.requires_grad
                                ),
                                grad_tensors=tuple(o for o in static_grad_outputs if o is not None),
                                retain_graph=retain_graph_in_backward,
                            )
                            grad_inputs = tuple(input.grad for input in inputs)
                        _call_capture_time_backward_hooks(
                            callable_idx, func, grad_inputs, static_grad_outputs
                        )

                    # Constructs a tuple suitable for returning from Graphed.backward:
                    # Pads out the actually-needed grads with Nones in gradient slots for inputs
                    # that don't require grad. I couldn't think of a one-liner for this pattern.
                    static_grad_inputs = []
                    static_grad_buffers = []
                    grad_idx = 0
                    for arg in static_input_surface:
                        if is_training and isinstance(arg, torch.Tensor) and arg.requires_grad:
                            static_grad_inputs.append(grad_inputs[grad_idx])
                            static_grad_buffers.append(input_grad_buffers[grad_idx])
                            grad_idx += 1
                        else:
                            static_grad_inputs.append(None)  # type: ignore[arg-type]
                            static_grad_buffers.append(None)
                    static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]
                    static_grad_buffers = tuple(static_grad_buffers)

                    per_callable_static_grad_outputs[per_callable_bwd_idx] = static_grad_outputs
                    per_callable_static_grad_inputs[per_callable_bwd_idx] = static_grad_inputs
                    returned_param_grad_clone_slots = _returned_param_grad_clone_slots(
                        static_grad_inputs,
                        per_callable_module_params[per_callable_bwd_idx],
                        static_grad_buffers,
                        _clone_param_grads_on_return,
                    )
                    per_callable_returned_param_grad_clone_slots[per_callable_bwd_idx] = (
                        returned_param_grad_clone_slots
                    )

                    # Weak ref the static outputs and static grad inputs that are no longer needed
                    # in the following steps. These two type of tensors are both in cudagraph
                    # mempool, so we just deallocate them and let PyTorch's memory allocator
                    # reuse them elsewhere.
                    if _reuse_graph_input_output_buffers:
                        if _activation_recompute:
                            recompute_outputs = per_callable_recompute_outputs[per_callable_bwd_idx]
                            per_callable_recompute_outputs[per_callable_bwd_idx] = tuple(
                                (
                                    make_weak_ref(output).requires_grad_(output.requires_grad)
                                    if isinstance(output, torch.Tensor)
                                    and output.is_cuda
                                    and output.is_contiguous()
                                    else output
                                )
                                for output in recompute_outputs
                            )
                        if not _activation_recompute:
                            # Retained order guarantees this output is consumed at backward.
                            per_callable_static_outputs[per_callable_bwd_idx] = make_weak_ref(
                                per_callable_static_outputs[per_callable_bwd_idx]
                            )

                        # Parameter grads can be weak-refed immediately only when Graphed.backward
                        # will clone them before returning them to autograd users.
                        static_grad_inputs = per_callable_static_grad_inputs[per_callable_bwd_idx]
                        per_callable_static_grad_inputs[per_callable_bwd_idx] = tuple(
                            (
                                make_weak_ref(grad_input)
                                if returned_param_grad_clone_slots[idx] and grad_input is not None
                                else grad_input
                            )
                            for idx, grad_input in enumerate(static_grad_inputs)
                        )

                        # Weak ref the static grad inputs of the previous backward pass within the
                        # same chunk.
                        if previous_per_callable_bwd_idx is not None:
                            idx = previous_per_callable_bwd_idx
                            per_callable_static_grad_inputs[idx] = make_weak_ref(
                                per_callable_static_grad_inputs[idx]
                            )
                        previous_per_callable_bwd_idx = per_callable_bwd_idx

                        # Weak ref the static grad inputs of the previous chunk's last backward
                        # pass.
                        # Note: After a chunk's backward pass, we assume Mcore will send the grad
                        # input to another pipeline parallel rank and that the communication is
                        # finished before the end of the next chunk's backward pass.
                        if l_no == 0:
                            if previous_chunk_last_callable_bwd_idx is not None:
                                idx = previous_chunk_last_callable_bwd_idx
                                per_callable_static_grad_inputs[idx] = make_weak_ref(
                                    per_callable_static_grad_inputs[idx]
                                )
                            previous_chunk_last_callable_bwd_idx = per_callable_bwd_idx
                if ceil(c_id) == c_id:
                    bwd_idx[m_chunk] += 1
    else:
        # Capture forward graphs
        per_callable_static_outputs = []
        per_callable_output_unflatten_spec = []
        for func_idx, (func, args, kwargs, fwd_graph) in enumerate(
            zip(callables, sample_args, sample_kwargs, fwd_graphs)
        ):
            args, kwargs = _link_callable_inputs(func_idx, per_callable_static_outputs)
            _call_capture_time_forward_pre_hooks(func_idx, func, args, kwargs)

            forward_grad_enabled = activation_recompute_forward_grad_modes[func_idx]
            if _activation_recompute and forward_grad_enabled:
                saved_tensor_context = torch.autograd.graph.saved_tensors_hooks(
                    discard_capture_saved_tensor, reject_discarded_saved_tensor
                )
            else:
                saved_tensor_context = contextlib.nullcontext()
            if _activation_recompute:
                forward_autograd_context = (
                    torch.enable_grad() if forward_grad_enabled else torch.no_grad()
                )
            else:
                forward_autograd_context = contextlib.nullcontext()
            with (
                saved_tensor_context,
                forward_autograd_context,
                _graph_context_wrapper(fwd_graph, pool=mempool, stream=capture_stream),
                _fp8_activation_recompute_phase(False if _activation_recompute else None),
            ):
                outputs = func(*args, **kwargs)
                if _activation_recompute:
                    flat_outputs_with_history, output_spec = _tree_flatten(outputs)
                    outputs = _tree_unflatten(
                        tuple(
                            output.detach() if isinstance(output, torch.Tensor) else output
                            for output in flat_outputs_with_history
                        ),
                        output_spec,
                    )
            _call_capture_time_forward_hooks(func_idx, func, args, kwargs, outputs)
            graph_callables[func_idx] = func

            flatten_outputs, spec = _tree_flatten(outputs)
            per_callable_static_outputs.append(tuple(flatten_outputs))
            per_callable_output_unflatten_spec.append(spec)

        args = kwargs = outputs = flatten_outputs = None

        per_callable_static_user_grad_buffers = (
            _allocate_static_dgrad_reuse_buffers(
                per_callable_static_input_surfaces,
                per_callable_static_outputs,
                _input_output_aliases,
                per_callable_user_grad_indices,
                per_callable_output_requires_grad,
            )
            if _activation_recompute
            else tuple({} for _ in flatten_sample_args)
        )

        # Capture each checkpoint region in replay order: RF forward,
        # followed by backward in reverse.
        per_callable_static_grad_outputs = [None] * len(flatten_sample_args)
        per_callable_static_grad_inputs = [None] * len(flatten_sample_args)
        per_callable_returned_param_grad_clone_slots = [None] * len(flatten_sample_args)
        # Reuse consumer dgrad as the producer grad output when possible.
        captured_grad_inputs = {}
        consumers_by_output = {}
        for consumer_idx, aliases in enumerate(_input_output_aliases):
            for input_idx, producer in aliases.items():
                consumers_by_output.setdefault(producer, []).append((consumer_idx, input_idx))

        def _prepare_static_grad_outputs(callable_idx):
            """Allocate or reuse the captured grad-output surface."""
            prepared = per_callable_static_grad_outputs[callable_idx]
            if prepared is not None:
                return prepared
            prepared = []
            for output_idx, output in enumerate(per_callable_static_outputs[callable_idx]):
                grad_output = None
                consumers = consumers_by_output.get((callable_idx, output_idx), ())
                if len(consumers) == 1:
                    consumer_idx, input_idx = consumers[0]
                    consumer_grad_inputs = captured_grad_inputs.get(consumer_idx)
                    if consumer_grad_inputs is not None:
                        candidate = consumer_grad_inputs[input_idx]
                        if (
                            candidate is not None
                            and output is not None
                            and candidate.shape == output.shape
                            and candidate.dtype == output.dtype
                            and candidate.device == output.device
                            and candidate.stride() == output.stride()
                        ):
                            grad_output = candidate
                logical_output_grads = per_callable_output_requires_grad[callable_idx]
                output_requires_grad = (
                    output is not None and output.requires_grad
                    if logical_output_grads is None
                    else logical_output_grads[output_idx]
                )
                if grad_output is None and output is not None and output_requires_grad:
                    grad_output = torch.empty_like(output)
                prepared.append(grad_output)
            prepared = tuple(prepared)
            per_callable_static_grad_outputs[callable_idx] = prepared
            return prepared

        capture_schedule = []
        if _activation_recompute:
            capture_schedule.extend(
                _activation_recompute_capture_schedule(activation_recompute_region_groups)
            )
        else:
            capture_schedule.extend(
                ("backward", callable_idx, False)
                for callable_idx in reversed(range(len(per_callable_static_input_surfaces)))
            )

        backward_prepared = set()
        for capture_phase, bwd_idx, keep_unsharded in capture_schedule:
            static_input_surface = per_callable_static_input_surfaces[bwd_idx]
            static_outputs = per_callable_static_outputs[bwd_idx]
            recompute_graph = recompute_graphs[bwd_idx]
            bwd_graph = bwd_graphs[bwd_idx]
            bwd_dw_graph = bwd_dw_graphs[bwd_idx]
            func = graph_callables[bwd_idx]

            if capture_phase == "recompute":
                if recompute_graph is None:
                    raise RuntimeError("Activation recompute requires a recompute-forward graph")
                if keep_unsharded:
                    static_grad_outputs = _prepare_static_grad_outputs(bwd_idx)
                    _call_capture_time_backward_pre_hooks(bwd_idx, func, static_grad_outputs)
                    backward_prepared.add(bwd_idx)
                else:
                    _call_capture_time_forward_pre_hooks(
                        bwd_idx, func, sample_args[bwd_idx], sample_kwargs[bwd_idx]
                    )
                module_params, static_input_surface = _refresh_module_parameter_surface(
                    func, flatten_sample_args[bwd_idx], per_callable_parameter_grad_indices[bwd_idx]
                )
                per_callable_module_params[bwd_idx] = module_params
                per_callable_static_input_surfaces[bwd_idx] = static_input_surface
                recompute_rng_pairs = per_callable_recompute_rng_pairs[bwd_idx]
                canonical_rng_states = tuple(
                    generator.graphsafe_get_state() for generator, _ in recompute_rng_pairs
                )
                for generator, recompute_generator in recompute_rng_pairs:
                    generator.graphsafe_set_state(recompute_generator)
                try:
                    with (
                        _graph_context_wrapper(
                            recompute_graph, pool=mempool, stream=capture_stream
                        ),
                        _fp8_activation_recompute_phase(True),
                    ):
                        recompute_outputs = func(*sample_args[bwd_idx], **sample_kwargs[bwd_idx])
                finally:
                    for (generator, _), canonical_state in zip(
                        recompute_rng_pairs, canonical_rng_states
                    ):
                        generator.graphsafe_set_state(canonical_state)
                recompute_flat_outputs, _ = _tree_flatten(recompute_outputs)
                per_callable_recompute_outputs[bwd_idx] = tuple(recompute_flat_outputs)
                if not keep_unsharded:
                    _call_capture_time_forward_hooks(
                        bwd_idx,
                        func,
                        sample_args[bwd_idx],
                        sample_kwargs[bwd_idx],
                        recompute_outputs,
                    )
                del recompute_flat_outputs
                del recompute_outputs
                continue

            static_grad_outputs = _prepare_static_grad_outputs(bwd_idx)
            input_grad_buffers = ()
            if is_training:
                if bwd_idx not in backward_prepared:
                    _call_capture_time_backward_pre_hooks(bwd_idx, func, static_grad_outputs)
                    module_params, static_input_surface = _refresh_module_parameter_surface(
                        func,
                        flatten_sample_args[bwd_idx],
                        per_callable_parameter_grad_indices[bwd_idx],
                    )
                    per_callable_module_params[bwd_idx] = module_params
                    per_callable_static_input_surfaces[bwd_idx] = static_input_surface
                else:
                    backward_prepared.remove(bwd_idx)
                    module_params = per_callable_module_params[bwd_idx]
                    static_input_surface = per_callable_static_input_surfaces[bwd_idx]
                input_surface_indices = tuple(
                    surface_idx
                    for surface_idx, input_tensor in enumerate(static_input_surface)
                    if input_tensor is not None and input_tensor.requires_grad
                )
                inputs = tuple(static_input_surface[idx] for idx in input_surface_indices)
                main_grad_buffers = (
                    _get_static_grad_buffers(inputs) if use_main_grad else (None,) * len(inputs)
                )
                static_user_grad_buffers = per_callable_static_user_grad_buffers[bwd_idx]
                input_grad_buffers = tuple(
                    static_user_grad_buffers.get(surface_idx, main_grad_buffer)
                    for surface_idx, main_grad_buffer in zip(
                        input_surface_indices, main_grad_buffers
                    )
                )
                recompute_flat_outputs = None
                if _activation_recompute:
                    recompute_flat_outputs = per_callable_recompute_outputs[bwd_idx]
                    if recompute_flat_outputs is None:
                        raise RuntimeError("Missing captured recompute-forward outputs")
                # Enter graph capture first so buffer zeroing is recorded.
                backward_autocast_context = (
                    torch.amp.autocast("cuda", enabled=False)
                    if _activation_recompute
                    else contextlib.nullcontext()
                )
                with (
                    _graph_context_wrapper(bwd_graph, pool=mempool, stream=capture_stream),
                    _static_grad_context_wrapper(inputs, input_grad_buffers),
                    _fp8_activation_recompute_phase(True if _activation_recompute else None),
                    backward_autocast_context,
                ):
                    if _activation_recompute:
                        if len(recompute_flat_outputs) != len(static_outputs):
                            raise RuntimeError(
                                "Recomputed output count does not match normal forward"
                            )
                        for normal_output, recompute_output in zip(
                            static_outputs, recompute_flat_outputs
                        ):
                            if isinstance(normal_output, torch.Tensor) and (
                                not isinstance(recompute_output, torch.Tensor)
                                or normal_output.shape != recompute_output.shape
                                or normal_output.stride() != recompute_output.stride()
                                or normal_output.dtype != recompute_output.dtype
                                or normal_output.device != recompute_output.device
                            ):
                                raise RuntimeError(
                                    "Recomputed output metadata does not match normal forward"
                                )
                        output_requires_grad = per_callable_output_requires_grad[bwd_idx]
                        torch.autograd.backward(
                            tuple(
                                output
                                for output, requires_grad in zip(
                                    recompute_flat_outputs, output_requires_grad
                                )
                                if requires_grad
                            ),
                            grad_tensors=tuple(
                                grad
                                for grad, requires_grad in zip(
                                    static_grad_outputs, output_requires_grad
                                )
                                if requires_grad
                            ),
                            retain_graph=retain_graph_in_backward,
                        )
                    else:
                        torch.autograd.backward(
                            tuple(o for o in static_outputs if o is not None and o.requires_grad),
                            grad_tensors=tuple(o for o in static_grad_outputs if o is not None),
                            retain_graph=retain_graph_in_backward,
                        )
                    grad_inputs = tuple(input.grad for input in inputs)
                _call_capture_time_backward_hooks(bwd_idx, func, grad_inputs, static_grad_outputs)
                if _activation_recompute:
                    del recompute_flat_outputs
                    gc.collect()

                if need_bwd_dw_graph.get(bwd_idx, False):
                    with _graph_context_wrapper(bwd_dw_graph, pool=mempool, stream=capture_stream):
                        for module in visited_te_modules[bwd_idx]:
                            if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                                module.backward_dw()
            # Constructs a tuple suitable for returning from Graphed.backward:
            # Pads out the actually-needed grads with Nones in gradient slots for inputs that
            # don't require grad. I couldn't think of a slick one-liner for this pattern.
            static_grad_inputs = []
            static_grad_buffers = []
            grad_idx = 0
            for arg in static_input_surface:
                if is_training and isinstance(arg, torch.Tensor) and arg.requires_grad:
                    static_grad_inputs.append(grad_inputs[grad_idx])
                    static_grad_buffers.append(input_grad_buffers[grad_idx])
                    grad_idx += 1
                else:
                    static_grad_inputs.append(None)  # type: ignore[arg-type]
                    static_grad_buffers.append(None)
            static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]
            static_grad_buffers = tuple(static_grad_buffers)
            for surface_idx, expected_buffer in per_callable_static_user_grad_buffers[
                bwd_idx
            ].items():
                captured_buffer = static_grad_inputs[surface_idx]
                if (
                    captured_buffer is None
                    or captured_buffer.data_ptr() != expected_buffer.data_ptr()
                ):
                    raise RuntimeError("Autograd did not preserve the static user-dgrad buffer")
            captured_grad_inputs[bwd_idx] = static_grad_inputs

            per_callable_static_grad_outputs[bwd_idx] = static_grad_outputs
            per_callable_static_grad_inputs[bwd_idx] = static_grad_inputs
            per_callable_returned_param_grad_clone_slots[bwd_idx] = (
                _returned_param_grad_clone_slots(
                    static_grad_inputs,
                    per_callable_module_params[bwd_idx],
                    static_grad_buffers,
                    _clone_param_grads_on_return,
                )
            )

            if _reuse_graph_input_output_buffers and _activation_recompute:
                recompute_outputs = per_callable_recompute_outputs[bwd_idx]
                per_callable_recompute_outputs[bwd_idx] = tuple(
                    (
                        make_weak_ref(output).requires_grad_(output.requires_grad)
                        if isinstance(output, torch.Tensor)
                        and output.is_cuda
                        and output.is_contiguous()
                        else output
                    )
                    for output in recompute_outputs
                )
                recompute_outputs = None

            if _reuse_graph_input_output_buffers and _activation_recompute:
                linked_inputs = list(flatten_sample_args[bwd_idx])
                static_surface = list(per_callable_static_input_surfaces[bwd_idx])
                replaced_input = False
                for input_idx, producer in _input_output_aliases[bwd_idx].items():
                    producer_idx, output_idx = producer
                    if producer_idx + 1 != bwd_idx:
                        continue
                    if len(consumers_by_output.get(producer, ())) != 1:
                        continue
                    producer_outputs = list(per_callable_static_outputs[producer_idx])
                    producer_output = producer_outputs[output_idx]
                    consumer_input = static_surface[input_idx]
                    if not (
                        isinstance(producer_output, torch.Tensor)
                        and isinstance(consumer_input, torch.Tensor)
                        and producer_output.data_ptr() == consumer_input.data_ptr()
                        and producer_output.is_contiguous()
                        and consumer_input.is_contiguous()
                        and _static_dgrad_metadata(producer_output)
                        == _static_dgrad_metadata(consumer_input)
                    ):
                        continue
                    producer_storage = producer_output.untyped_storage()
                    consumer_storage = consumer_input.untyped_storage()
                    storage_nbytes = producer_storage.nbytes()
                    if (
                        producer_storage.data_ptr() != consumer_storage.data_ptr()
                        or storage_nbytes
                        != producer_output.numel() * producer_output.element_size()
                        or sum(
                            isinstance(output, torch.Tensor)
                            and output.untyped_storage().data_ptr() == producer_storage.data_ptr()
                            for output in producer_outputs
                        )
                        != 1
                        or sum(
                            isinstance(value, torch.Tensor)
                            and value.untyped_storage().data_ptr() == producer_storage.data_ptr()
                            for value in linked_inputs
                        )
                        != 1
                    ):
                        continue

                    producer_outputs[output_idx] = make_weak_ref(producer_output).requires_grad_(
                        producer_output.requires_grad
                    )
                    linked_inputs[input_idx] = make_weak_ref(consumer_input).requires_grad_(
                        consumer_input.requires_grad
                    )
                    static_surface[input_idx] = linked_inputs[input_idx]
                    per_callable_static_outputs[producer_idx] = tuple(producer_outputs)
                    replaced_input = True

                if replaced_input:
                    flatten_sample_args[bwd_idx] = tuple(linked_inputs)
                    per_callable_static_input_surfaces[bwd_idx] = tuple(static_surface)
                    flat_args_len = per_callable_flat_args_len[bwd_idx]
                    sample_args[bwd_idx] = _tree_unflatten(
                        linked_inputs[:flat_args_len], per_callable_args_spec[bwd_idx]
                    )
                    kwarg_values = _tree_unflatten(
                        linked_inputs[flat_args_len:], per_callable_kwargs_spec[bwd_idx]
                    )
                    sample_kwargs[bwd_idx] = dict(
                        zip(per_callable_kwargs_keys[bwd_idx], kwarg_values)
                    )
                    static_input_surface = per_callable_static_input_surfaces[bwd_idx]
                producer_output = consumer_input = None
                producer_storage = consumer_storage = None
                inputs = ()

        if backward_prepared:
            raise RuntimeError("Recompute capture left a module prepared for backward")

    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.
    per_callable_expected_param_ptrs = [
        tuple(param.data_ptr() for param in module_params)
        for module_params in per_callable_module_params
    ]
    per_callable_expected_allocator_signatures = [
        tuple(_parameter_allocator_signature(param) for param in module_params)
        for module_params in per_callable_module_params
    ]

    def make_graphed_autograd_function(
        fwd_graph,
        recompute_graph,
        bwd_graph,
        module_params,
        kwargs_keys,
        expected_args_spec,
        expected_kwargs_spec,
        flat_args_len,
        num_positional_args,
        call_signature,
        captured_static_arguments,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
        static_recompute_outputs,
        static_grad_outputs,
        static_grad_inputs,
        returned_param_grad_clone_slots,
        expected_param_ptrs,
        expected_allocator_signatures,
        graph_replay_state,
        output_requires_grad,
        activation_recompute,
        recompute_rng_pairs,
    ):
        class Graphed(torch.autograd.Function):
            """Autograd function for graph replay."""

            @staticmethod
            def forward(
                ctx,
                replay_phase,
                forward_builds_autograd,
                skip_fp8_weight_update,
                cuda_graph_stream,
                cuda_graph_event,
                *inputs,
            ):
                # pylint: disable=missing-function-docstring
                if activation_recompute:
                    current_phase = graph_replay_state["phase"]
                    if replay_phase == "inference":
                        if current_phase != "idle":
                            raise RuntimeError(
                                "Activation-recompute CUDA graph state is "
                                f"{current_phase}; inference requires idle"
                            )
                        ctx.owns_backward = False
                    elif replay_phase == "forward":
                        if current_phase != "idle":
                            raise RuntimeError(
                                "Activation-recompute CUDA graph expected idle "
                                f"before forward, found {current_phase}"
                            )
                        graph_replay_state["phase"] = "forward_done"
                        graph_replay_state["forward_owns_backward"] = forward_builds_autograd
                        graph_replay_state["recompute_rng_states"] = tuple(
                            generator.get_state() for generator, _ in recompute_rng_pairs
                        )
                        ctx.owns_backward = forward_builds_autograd
                    elif replay_phase == "recompute":
                        if current_phase != "forward_done":
                            raise RuntimeError(
                                "Activation-recompute CUDA graph expected "
                                "forward_done before recompute, found "
                                f"{current_phase}"
                            )
                        graph_replay_state["phase"] = "recomputed"
                        ctx.owns_backward = not graph_replay_state["forward_owns_backward"]
                    else:
                        raise ValueError(f"Unknown CUDA graph replay phase {replay_phase!r}")
                else:
                    ctx.owns_backward = True
                try:
                    return Graphed._forward_impl(
                        ctx,
                        replay_phase,
                        forward_builds_autograd,
                        skip_fp8_weight_update,
                        cuda_graph_stream,
                        cuda_graph_event,
                        *inputs,
                    )
                except Exception:
                    if activation_recompute:
                        if replay_phase != "inference":
                            graph_replay_state["pending_generation"] = None
                            graph_replay_state["phase"] = "idle"
                            graph_replay_state["forward_owns_backward"] = False
                            graph_replay_state["recompute_rng_states"] = ()
                            graph_replay_state["pending_region"] = None
                    raise

            @staticmethod
            def _forward_impl(
                ctx,
                replay_phase,
                forward_builds_autograd,
                skip_fp8_weight_update,
                cuda_graph_stream,
                cuda_graph_event,
                *inputs,
            ):
                # pylint: disable=missing-function-docstring
                graph_replay_state["generation"] += 1
                if activation_recompute and replay_phase == "forward":
                    graph_replay_state["pending_generation"] = graph_replay_state["generation"]
                ctx.forward_generation = (
                    graph_replay_state["pending_generation"]
                    if activation_recompute
                    else graph_replay_state["generation"]
                )
                ctx.replay_phase = replay_phase
                # Set flag for whether to update FP8 weight updates
                ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()
                if ctx.is_first_module and skip_fp8_weight_update is not None:
                    FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor.fill_(
                        skip_fp8_weight_update
                    )
                ctx.cuda_graph_stream = cuda_graph_stream
                ctx.cuda_graph_event = cuda_graph_event
                ctx.has_checkpoint_sentinel = False
                if activation_recompute and replay_phase != "inference" and forward_builds_autograd:
                    sentinel = next(
                        (value for value in reversed(inputs) if isinstance(value, torch.Tensor)),
                        None,
                    )
                    if sentinel is not None:
                        ctx.save_for_backward(sentinel)
                        ctx.has_checkpoint_sentinel = True
                # Copy values from new tensors into static tensors
                for i in range(len_user_args):
                    if (
                        isinstance(static_input_surface[i], torch.Tensor)
                        and inputs[i] is not None
                        and static_input_surface[i].data_ptr() != inputs[i].data_ptr()
                    ):
                        static_input_surface[i].copy_(inputs[i])

                replay_graph = fwd_graph
                replay_outputs = static_outputs
                if replay_phase == "recompute":
                    if recompute_graph is None or static_recompute_outputs is None:
                        raise RuntimeError("Recompute replay requires a captured recompute graph")
                    rng_states = graph_replay_state["recompute_rng_states"]
                    if len(rng_states) != len(recompute_rng_pairs):
                        raise RuntimeError(
                            "Activation-recompute CUDA graph is missing forward RNG state"
                        )
                    for (_, recompute_generator), rng_state in zip(recompute_rng_pairs, rng_states):
                        recompute_generator.set_state(rng_state)
                    replay_graph = recompute_graph
                    replay_outputs = static_recompute_outputs

                if cuda_graph_stream != torch.cuda.current_stream():
                    cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with cuda_graph_stream:
                        replay_graph.replay()
                    if cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(cuda_graph_stream)
                else:
                    replay_graph.replay()
                if not isinstance(replay_outputs, tuple):
                    raise TypeError(
                        "Expected replay outputs to be a tuple, but got"
                        f" {type(replay_outputs).__name__}"
                    )
                clone_inference_outputs = activation_recompute and replay_phase == "inference"
                returned_outputs = tuple(
                    (
                        o.detach().clone()
                        if clone_inference_outputs and isinstance(o, torch.Tensor)
                        else o.detach() if isinstance(o, torch.Tensor) else o
                    )
                    for o in replay_outputs
                )
                if activation_recompute:
                    non_differentiable = tuple(
                        output
                        for output, requires_grad in zip(returned_outputs, output_requires_grad)
                        if isinstance(output, torch.Tensor) and not requires_grad
                    )
                    if non_differentiable:
                        ctx.mark_non_differentiable(*non_differentiable)
                return returned_outputs

            @staticmethod
            def _backward_impl(ctx, *grads):
                # pylint: disable=missing-function-docstring

                if activation_recompute and ctx.has_checkpoint_sentinel:
                    # Trigger non-reentrant checkpoint recomputation before B.
                    _ = ctx.saved_tensors
                if activation_recompute and not ctx.owns_backward:
                    raise RuntimeError("Activation-recompute forward node does not own backward")
                if (
                    activation_recompute
                    and ctx.forward_generation != graph_replay_state["pending_generation"]
                ):
                    raise RuntimeError(
                        "Activation-recompute CUDA graph backward belongs to a "
                        "released or superseded forward"
                    )
                # Replay backward graph
                if len(grads) != len(static_grad_outputs):
                    raise ValueError(
                        "Backward graph grad dimension mismatch: "
                        f"received {len(grads)} grads, "
                        f"but expected {len(static_grad_outputs)} static_grad_outputs"
                    )
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                if activation_recompute:
                    current_phase = graph_replay_state["phase"]
                    if current_phase != "recomputed":
                        raise RuntimeError(
                            "Activation-recompute CUDA graph expected recomputed "
                            f"before backward, found {current_phase}"
                        )
                if ctx.cuda_graph_stream != torch.cuda.current_stream():
                    ctx.cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with ctx.cuda_graph_stream:
                        bwd_graph.replay()
                    if ctx.cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(ctx.cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(ctx.cuda_graph_stream)
                else:
                    bwd_graph.replay()
                graph_replay_state["generation"] += 1

                # Update FP8 scale factors if needed
                if ctx.is_first_module:
                    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

                # Input args that didn't require grad expect a None gradient.
                if not isinstance(static_grad_inputs, tuple):
                    raise TypeError(
                        "Expected static_grad_inputs to be a tuple, but got"
                        f" {type(static_grad_inputs).__name__}"
                    )
                grad_inputs = []
                for idx, grad_input in enumerate(static_grad_inputs):
                    if grad_input is None:
                        grad_inputs.append(None)
                    elif returned_param_grad_clone_slots[idx]:
                        grad_inputs.append(grad_input.detach().clone())
                    else:
                        grad_inputs.append(grad_input.detach())
                return (None, None, None, None, None) + tuple(grad_inputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                # pylint: disable=missing-function-docstring
                try:
                    return Graphed._backward_impl(ctx, *grads)
                finally:
                    if activation_recompute:
                        if (
                            ctx.owns_backward
                            and ctx.forward_generation == graph_replay_state["pending_generation"]
                        ):
                            graph_replay_state["pending_generation"] = None
                            graph_replay_state["phase"] = "idle"
                            graph_replay_state["forward_owns_backward"] = False
                            graph_replay_state["recompute_rng_states"] = ()
                            graph_replay_state["pending_region"] = None

        @torch.compiler.disable
        def functionalized(*user_args, **user_kwargs):
            replay_phase = user_kwargs.pop("_mfsdp_cuda_graph_replay_phase", "forward")
            if replay_phase not in ("forward", "recompute", "inference"):
                raise ValueError(f"Unknown CUDA graph replay phase {replay_phase!r}")
            for param, expected_ptr in zip(module_params, expected_param_ptrs):
                if param.data_ptr() != expected_ptr:
                    raise RuntimeError("CUDA graph parameter address changed after capture")
            for param, expected_signature in zip(module_params, expected_allocator_signatures):
                if (
                    expected_signature is not None
                    and _parameter_allocator_signature(param) != expected_signature
                ):
                    raise RuntimeError("CUDA graph parameter allocator plan changed")

            # Decide whether to update FP8 weights
            skip_fp8_weight_update = None
            if cache_quantized_params:
                if "is_first_microbatch" not in user_kwargs or not isinstance(
                    user_kwargs["is_first_microbatch"], bool
                ):
                    raise ValueError(
                        "`is_first_microbatch` boolean kwarg must be provided for FP8 weight"
                        " caching."
                    )

                skip_fp8_weight_update = not user_kwargs["is_first_microbatch"]

            # The cuda_graph_stream and cuda_graph_event are used in the TE CUDA graph replay.
            # When replaying the graph in the cuda graph stream, the graph replay could overlap
            # with the work on main stream.
            # When cuda_graph_event is given, it should be an external event recorded
            # in the cuda graph and is used to sync-back to the main stream.
            # If cuda_graph_event is not given, it will be None and the graph replay will block
            # the main stream until it is finished.
            if "cuda_graph_stream" in user_kwargs:
                cuda_graph_stream = user_kwargs["cuda_graph_stream"]
                user_kwargs.pop("cuda_graph_stream")
            else:
                cuda_graph_stream = torch.cuda.current_stream()
            if "cuda_graph_event" in user_kwargs:
                cuda_graph_event = user_kwargs["cuda_graph_event"]
                user_kwargs.pop("cuda_graph_event")
            else:
                cuda_graph_event = None
            # Runs the autograd function with inputs == all inputs to
            # the graph that might require grad (explicit user args +
            # module parameters)
            # Assumes module params didn't change since capture.
            # Reconstruct the same flattened arg order as capture time. A compiled
            # module may pass recorded kwargs positionally, including static defaults
            # interleaved with tensor inputs, so bind by parameter name when possible.
            arg_values = ()
            kwarg_values = []
            if call_signature is not None:
                try:
                    runtime_arguments = call_signature.bind_partial(*user_args, **user_kwargs)
                    runtime_arguments.apply_defaults()
                except TypeError as exc:
                    raise RuntimeError(
                        "CUDA graph call arguments no longer match the captured signature"
                    ) from exc
                arg_values = runtime_arguments.args[:num_positional_args]
                for key in kwargs_keys:
                    if key in runtime_arguments.arguments:
                        kwarg_values.append(runtime_arguments.arguments[key])
                    elif key in user_kwargs:
                        kwarg_values.append(user_kwargs[key])
                    else:
                        raise RuntimeError(f"CUDA graph input {key!r} is missing at replay")
                for key, captured_value in captured_static_arguments.items():
                    runtime_value = runtime_arguments.arguments.get(key, object())
                    if (
                        type(runtime_value) is not type(captured_value)
                        or runtime_value != captured_value
                    ):
                        raise RuntimeError(
                            "CUDA graph input structure or static metadata changed " "after capture"
                        )
            else:
                # Some extension callables do not expose an inspectable signature.
                # Preserve the legacy positional fallback for those callables.
                arg_values = user_args[:num_positional_args]
                user_pos_args = list(user_args[num_positional_args:])
                for key in kwargs_keys:
                    if key in user_kwargs:
                        kwarg_values.append(user_kwargs[key])
                    elif user_pos_args:
                        kwarg_values.append(user_pos_args.pop(0))
                    # else: key was a default not passed — skip (not a tensor)
            flatten_user_args, args_spec = _tree_flatten(arg_values)
            if args_spec != expected_args_spec or len(flatten_user_args) != flat_args_len:
                raise RuntimeError("CUDA graph positional input structure changed after capture")
            flatten_user_kwargs, kwargs_spec = _tree_flatten(kwarg_values)
            if kwargs_spec != expected_kwargs_spec:
                raise RuntimeError(
                    "CUDA graph input structure or static metadata changed after capture"
                )
            flatten_user_inputs = tuple(flatten_user_args) + tuple(flatten_user_kwargs)
            if len(flatten_user_inputs) != len_user_args:
                raise RuntimeError("CUDA graph flattened input count changed after capture")
            for input_idx, (static_input, runtime_input) in enumerate(
                zip(static_input_surface[:len_user_args], flatten_user_inputs)
            ):
                static_is_tensor = isinstance(static_input, torch.Tensor)
                runtime_is_tensor = isinstance(runtime_input, torch.Tensor)
                if not static_is_tensor and not runtime_is_tensor:
                    if (
                        type(static_input) is not type(runtime_input)
                        or static_input != runtime_input
                    ):
                        raise RuntimeError(
                            "CUDA graph input structure or static metadata changed " "after capture"
                        )
                    continue
                if static_is_tensor != runtime_is_tensor:
                    raise RuntimeError(
                        "CUDA graph input structure or static metadata changed after capture: "
                        f"leaf {input_idx} captured {type(static_input).__name__}, "
                        f"replayed {type(runtime_input).__name__}"
                    )
                if (
                    static_input.shape != runtime_input.shape
                    or static_input.dtype != runtime_input.dtype
                    or static_input.device != runtime_input.device
                    or static_input.layout != runtime_input.layout
                    or static_input.stride() != runtime_input.stride()
                    or (
                        torch.is_grad_enabled()
                        and static_input.requires_grad != runtime_input.requires_grad
                    )
                ):
                    raise RuntimeError("CUDA graph input tensor metadata changed after capture")
            func_args = flatten_user_inputs + module_params
            out = Graphed.apply(
                replay_phase,
                torch.is_grad_enabled(),
                skip_fp8_weight_update,
                cuda_graph_stream,
                cuda_graph_event,
                *func_args,
            )
            return _tree_unflatten(out, output_unflatten_spec)

        def preflight():
            if activation_recompute:
                checkpoint_phase = current_cuda_graph_checkpoint_phase()
                checkpoint_region = current_cuda_graph_checkpoint_region()
                current_phase = graph_replay_state["phase"]
                replay_phase = resolve_replay_phase(checkpoint_phase, torch.is_grad_enabled())
                if replay_phase == "recompute" and current_phase == "forward_done":
                    if checkpoint_region is not graph_replay_state["pending_region"]:
                        raise RuntimeError(
                            "Activation-recompute CUDA graph recompute belongs to a "
                            "released or superseded forward"
                        )
                    return
                if current_phase != "idle":
                    raise RuntimeError(
                        "Activation-recompute CUDA graph state is "
                        f"{current_phase}; next phase is {replay_phase}"
                    )

        def release_pending():
            """Release an abandoned activation-recompute forward.

            :return: Whether a pending forward was released.
            :rtype: bool
            """
            if not activation_recompute:
                return False
            if graph_replay_state["phase"] == "idle":
                return False
            graph_replay_state["generation"] += 1
            graph_replay_state["pending_generation"] = None
            graph_replay_state["phase"] = "idle"
            graph_replay_state["forward_owns_backward"] = False
            graph_replay_state["recompute_rng_states"] = ()
            graph_replay_state["pending_region"] = None
            return True

        functionalized._cuda_graph_preflight = preflight
        functionalized._cuda_graph_release_pending = release_pending
        return functionalized

    def make_graphed_attribute_functions(graph_idx):
        """Create lifecycle functions for one callable."""
        # Get te modules for current graph
        te_modules = visited_te_modules.get(graph_idx, set())
        reset_done = False

        # Attach backward_dw as an attribute to the graphed callable.
        def backward_dw():
            """Replay the delayed backward-wgrad graph when present."""
            if need_bwd_dw_graph.get(graph_idx, False):
                bwd_dw_graphs[graph_idx].replay()

                # Trigger the grad accumulation hook for wgrad graphs.
                for module in te_modules:
                    if (
                        hasattr(module, "_trigger_wgrad_accumulation_and_reduce_hooks")
                        and module.need_backward_dw()
                    ):
                        module._trigger_wgrad_accumulation_and_reduce_hooks()

        # Attach reset as an attribute to the graphed callable.
        def reset():
            """Reset all CUDA graph objects for this callable."""
            nonlocal reset_done
            if reset_done:
                return
            graph_replay_states[graph_idx]["generation"] += 1
            graph_replay_states[graph_idx]["pending_generation"] = None
            graph_replay_states[graph_idx]["phase"] = "idle"
            graph_replay_states[graph_idx]["forward_owns_backward"] = False
            graph_replay_states[graph_idx]["recompute_rng_states"] = ()
            graph_replay_states[graph_idx]["pending_region"] = None
            fwd_graphs[graph_idx].reset()
            if recompute_graphs[graph_idx] is not None:
                recompute_graphs[graph_idx].reset()
            bwd_graphs[graph_idx].reset()
            bwd_dw_graphs[graph_idx].reset()
            reset_done = True

        return backward_dw, reset

    # Put together the final graphed callables
    ret = []
    for i in range(len(sample_args)):
        func = graph_callables[i]
        signature_target = (
            func.__dict__.get("_mfsdp_cuda_graph_forward_impl", func.forward)
            if isinstance(func, torch.nn.Module)
            else func
        )
        try:
            call_signature = inspect.signature(signature_target)
            captured_arguments = call_signature.bind_partial(*sample_args[i], **sample_kwargs[i])
            captured_arguments.apply_defaults()
        except (TypeError, ValueError):
            call_signature = None
            captured_arguments = None
        captured_static_arguments = {}
        if captured_arguments is not None:
            captured_static_arguments = {
                key: value
                for key, value in captured_arguments.arguments.items()
                if key not in per_callable_kwargs_keys[i]
                and not any(isinstance(leaf, torch.Tensor) for leaf in _tree_flatten(value)[0])
            }
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            recompute_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_kwargs_keys[i],
            per_callable_args_spec[i],
            per_callable_kwargs_spec[i],
            per_callable_flat_args_len[i],
            len(sample_args[i]),
            call_signature,
            captured_static_arguments,
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_recompute_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
            per_callable_returned_param_grad_clone_slots[i],
            per_callable_expected_param_ptrs[i],
            per_callable_expected_allocator_signatures[i],
            graph_replay_states[i],
            per_callable_output_requires_grad[i],
            _activation_recompute,
            per_callable_recompute_rng_pairs[i],
        )

        te_modules = visited_te_modules.get(i, set())
        if isinstance(func, torch.nn.Module):
            registered_buffer_slots = _registered_buffer_slots(func)
            expected_buffer_surfaces = tuple(
                _registered_buffer_slot_signature(slot) for slot in registered_buffer_slots
            )

            def make_graphed_forward(
                func,
                graph_training_state,
                graphed,
                orig_fwd,
                te_modules,
                registered_buffer_slots,
                expected_buffer_surfaces,
                activation_recompute,
                graph_replay_state,
            ):
                """Wrap one module forward with runtime compatibility checks."""

                @torch.compiler.disable
                def new_fwd(*user_args, **user_kwargs):
                    replay_phase = None
                    if activation_recompute:
                        checkpoint_phase = current_cuda_graph_checkpoint_phase()
                        checkpoint_region = current_cuda_graph_checkpoint_region()
                        replay_phase = resolve_replay_phase(
                            checkpoint_phase, torch.is_grad_enabled()
                        )
                        if checkpoint_phase is None and torch.is_grad_enabled():
                            raise RuntimeError(
                                "Activation-recompute CUDA graph training requires an explicit "
                                "checkpoint phase; use wrap_cuda_graph_checkpoint"
                            )
                        if (
                            replay_phase == "recompute"
                            and checkpoint_region is not graph_replay_state["pending_region"]
                        ):
                            raise RuntimeError(
                                "Activation-recompute CUDA graph recompute belongs to a "
                                "released or superseded forward"
                            )
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        if registered_buffer_slots:
                            current_buffer_surfaces = tuple(
                                _registered_buffer_slot_signature(slot)
                                for slot in registered_buffer_slots
                            )
                            if current_buffer_surfaces != expected_buffer_surfaces:
                                raise RuntimeError(
                                    "CUDA graph registered buffer metadata or address changed "
                                    "after capture"
                                )
                        # Set the FP8 group from global amax reduction.
                        if FP8GlobalStateManager.is_fp8_enabled():
                            fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                            for m in func.modules():
                                if m not in te_modules:
                                    # Only Set the FP8 meta for the modules included by forward
                                    continue
                                if isinstance(m, TransformerEngineBaseModule):
                                    # pylint: disable-next=line-too-long
                                    from transformer_engine.pytorch.attention.dot_product_attention import (
                                        DotProductAttention,
                                    )

                                    if (
                                        isinstance(m, DotProductAttention)
                                        and not fp8_recipe.fp8_mha
                                        and not fp8_recipe.fp8_dpa
                                    ):
                                        # Don't need to update FP8 meta for non-FP8 DPA
                                        continue
                                    m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                                    m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
                                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                        m.fp8_meta
                                    )
                                elif isinstance(m, BasicOperation):
                                    for mode in ("forward", "backward"):
                                        if m.num_quantizers(mode):
                                            m._fp8_metas[mode][
                                                "fp8_group"
                                            ] = FP8GlobalStateManager.get_fp8_group()
                                            m._fp8_metas[mode][
                                                "recipe"
                                            ] = FP8GlobalStateManager.get_fp8_recipe()
                                            FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                                m._fp8_metas[mode]
                                            )
                        if replay_phase is not None:
                            user_kwargs["_mfsdp_cuda_graph_replay_phase"] = replay_phase
                        if (
                            activation_recompute
                            and replay_phase == "forward"
                            and graph_replay_state["phase"] == "idle"
                        ):
                            graph_replay_state["pending_region"] = checkpoint_region
                        output = graphed(*user_args, **user_kwargs)
                        return output
                    return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            original_forward = func.__dict__.get("_mfsdp_cuda_graph_forward_impl", func.forward)
            forward = make_graphed_forward(
                func,
                func.training,
                graphed,
                original_forward,
                te_modules,
                registered_buffer_slots,
                expected_buffer_surfaces,
                _activation_recompute,
                graph_replay_states[i],
            )
            if _order is None:
                if "_mfsdp_cuda_graph_forward_impl" in func.__dict__:
                    func._mfsdp_cuda_graph_forward_impl = forward
                else:
                    func.forward = forward
                ret.append(func)
            else:
                ret.append(forward)
        else:
            ret.append(graphed)

        backward_dw_func, reset_func = make_graphed_attribute_functions(i)
        setattr(ret[-1], "backward_dw", backward_dw_func)
        setattr(ret[-1], "reset", reset_func)
        preflight = getattr(graphed, "_cuda_graph_preflight", None)
        if callable(preflight):
            setattr(ret[-1], "_cuda_graph_preflight", preflight)
        release_pending = getattr(graphed, "_cuda_graph_release_pending", None)
        if callable(release_pending):
            setattr(ret[-1], "_cuda_graph_release_pending", release_pending)
            if not hasattr(ret[-1], "_fsdp_root_context"):
                setattr(ret[-1], "release_pending", release_pending)
    if just_one_callable:
        return ret[0]

    return tuple(ret)


def save_fp8_tensors(
    modules: Iterable[torch.nn.Module], recipe: Optional[Recipe]
) -> Optional[List[Any]]:
    """
    Returns the FP8 tensors for all modules
    with adjusted amax history sizes.
    """

    if not isinstance(recipe, DelayedScaling):
        return None

    fp8_tensors = []
    for module in modules:
        for m in module.modules():
            module_tensors = None
            if isinstance(m, TransformerEngineBaseModule):
                if m.primary_weights_in_fp8:
                    m.adjust_amax_history_length(recipe.amax_history_len)
                module_tensors = m.get_fp8_meta_tensors()
            elif isinstance(m, BasicOperation):
                m.reset_recipe_state(recipe=recipe)
                module_tensors = m._save_fp8_metas()
            fp8_tensors.append(module_tensors)
    return fp8_tensors


def restore_fp8_tensors(
    modules: Iterable[torch.nn.Module], fp8_tensors: Optional[List[Any]]
) -> None:
    """Restore FP8 tensors."""

    if fp8_tensors is None:
        return

    for module in modules:
        for m in module.modules():
            module_tensors = fp8_tensors.pop(0)
            if isinstance(m, TransformerEngineBaseModule):
                m.reset_fp8_meta_tensors(module_tensors)
            elif isinstance(m, BasicOperation):
                m._load_fp8_metas(module_tensors)
    if len(fp8_tensors) != 0:
        raise RuntimeError(
            f"Got FP8 state for {len(fp8_tensors)} more modules than expected. "
            "There is probably a discrepancy with `save_fp8_tensors`."
        )


def make_graphed_callables(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    fp8_enabled: Optional[SingleOrTuple[bool]] = None,
    fp8_calibrating: Optional[bool] = None,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    fp8_weight_caching: Optional[bool] = None,
    enabled: Optional[SingleOrTuple[bool]] = None,
    calibrating: Optional[bool] = None,
    recipe: Optional[Recipe] = None,
    amax_reduction_group: Optional[dist_group_type] = None,
    cache_quantized_params: Optional[bool] = None,
    _order: Optional[List[int]] = None,
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    retain_graph_in_backward: bool = False,
    _reuse_graph_input_output_buffers: bool = False,
    _clone_param_grads_on_return: bool = True,
    _input_output_aliases: Optional[Tuple[Dict[int, Tuple[int, int]], ...]] = None,
    _activation_recompute: bool = False,
    _activation_recompute_forward_grad_enabled: Union[bool, Sequence[bool]] = False,
    _activation_recompute_regions: Optional[Sequence[int]] = None,
    _activation_recompute_order_slots: Optional[Sequence[int]] = None,
    pre_warmup_hook: Optional[Callable] = None,
    post_warmup_hook: Optional[Callable] = None,
    capture_time_hooks: Optional[List[Optional[Dict[str, Dict]]]] = None,
    capture_stream: Optional[torch.cuda.Stream] = None,
    use_main_grad: bool = False,
) -> Union[Callable, Tuple[Callable, ...]]:
    """
    Make CUDA graph version of Transformer Engine modules

    A variation of PyTorch's `make_graphed_callables` utility function
    with support for Transformer Engine modules and FP8. Please see
    the
    original PyTorch implementation for more documentation.

    .. warning::

       Arguments 'fp8_enabled', 'fp8_calibrating', 'fp8_recipe', 'fp8_group', and
       'fp8_weight_caching' are deprecated. Use 'enabled', 'calibrating', 'recipe',
       'amax_reduction_group', and 'cache_quantized_params' instead.

    Graphing parameters
    ===================
    modules: (tuple of) callable
             Callable or callables to graph.
    sample_args: (tuple of) tuple of torch.Tensor
                 Positional arguments to callable(s).
    num_warmup_iters: int, default = 3
                      Number of warmup iterations.
    allow_unused_input: bool, default = False
                        Whether to handle case where callable inputs
                        and outputs are disconnected in compute graph.
    use_main_grad: bool, default = False
                   Whether to bind compatible leaf gradients directly to
                   caller-owned main-grad buffers during backward capture.
    sample_kwargs: (tuple of) dict, optional
                   Keyword arguments to callable(s)
    pool: (tuple of) int, default = None, optional
          An instance returned from function `torch.cuda.graph_pool_handle` that hints
          this graph may share memory with the indicated pool.
    retain_graph_in_backward: bool, default = False
                              Whether to set retain_graph=True in backward graph capture.
    _reuse_graph_input_output_buffers: bool, default = False
        Reduce memory usage by reusing input/output data buffers between
        graphs. MCore pipeline capture uses `_order`; activation recompute weakens
        contiguous, uniquely consumed internal forward boundaries after their
        backward capture. Other inputs and outputs retain their own storage.
    _clone_param_grads_on_return: bool, default = True
        Clone parameter gradients before returning them from CUDA graph replay.
        Disabling this avoids the extra clone/copy and may improve performance,
        but returned parameter gradients will alias CUDA graph static gradient
        buffers. These tensors no longer have standard PyTorch returned-gradient
        lifetime semantics: a later replay of the same graph, or reused-buffer
        replay of another callable, may overwrite retained hook or `.grad`
        tensors. Only disable this when the caller consumes returned parameter
        gradients before any such overwrite can occur.
    _activation_recompute: bool, default = False
        Capture the original forward, grad-enabled recompute forward, and
        backward as separate CUDA Graphs.
    _activation_recompute_forward_grad_enabled: bool or sequence of bool, default = False
        Capture original-forward graphs with autograd enabled while discarding
        their saved-tensor tapes. Non-reentrant checkpointing requires this mode;
        reentrant checkpointing leaves it disabled. A sequence configures each
        callable independently.
    _activation_recompute_regions: sequence of int, optional
        Checkpoint region index for each callable. Callables in one region must
        be contiguous in forward order. Recompute is captured in forward order
        within a region, followed by backward in reverse order.
    _activation_recompute_order_slots: sequence of int, optional
        Microbatch lane selected by each custom ``_order`` event. This permits
        backward order to differ from original-forward order.
    pre_warmup_hook: callable, default = None
                      A hook function that will be called once before all warmup iterations
                      (not once per callable).
    post_warmup_hook: callable, default = None
                      A hook function that will be called once after all warmup iterations
                      (not once per callable).
    capture_time_hooks: list of dict, optional
                        Per-callable hooks invoked during warmup and graph capture, but
                        intentionally executed outside CUDA graph capture so they are not
                        recorded into the graph and are not replayed. Each hook must return
                        ``None``. Each list entry corresponds to one callable and may include
                        these keys:
                        ``"forward_pre_hooks"`` maps hook IDs to hooks with signature
                        ``hook(module, args)`` or ``hook(module, args, kwargs)`` when the ID
                        is present in ``"forward_pre_hooks_with_kwargs"``;
                        ``"forward_hooks"`` maps hook IDs to hooks with signature
                        ``hook(module, args, output)`` or ``hook(module, args, kwargs, output)``
                        when the ID is present in ``"forward_hooks_with_kwargs"``;
                        ``"backward_pre_hooks"`` maps hook IDs to
                        ``hook(module, grad_output)``;
                        ``"backward_hooks"`` maps hook IDs to
                        ``hook(module, grad_input, grad_output)``.

    Quantization parameters
    =======================
    enabled: (tuple of) bool, default = False
             whether or not to enable low precision quantization (FP8/FP4).
             If tuple, the length must match the number of modules.
    calibrating: bool, default = False
                 calibration mode allows collecting statistics such as amax and scale
                 data of quantized tensors even when executing without quantization enabled.
                 This is useful for saving an inference ready checkpoint while training
                 using a higher precision.
    recipe: recipe.Recipe, default = None
            recipe used for low precision quantization.
    amax_reduction_group: torch._C._distributed_c10d.ProcessGroup, default = None
                          distributed group over which amaxes for the quantized tensors
                          are reduced at the end of each training step.
    cache_quantized_params: bool, default = False
                            Whether to cache quantized weights across microbatches. If set to
                            `True`, pass `is_first_microbatch` to TransformerEngine modules.
                            When primary weights use TE's `quantized_model_init` API with a
                            quantization-aware optimizer, set this to `False` if weight
                            transposes are calculated outside TE, for example by the optimizer.

    """

    if not isinstance(use_main_grad, bool):
        raise TypeError(f"use_main_grad must be a bool, but got {type(use_main_grad).__name__}")

    te_available = _prepare_runtime()

    # Handle deprecated args. If old kwargs are set, they are prioritized with warning.
    if fp8_enabled is not None:
        if enabled is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_enabled` kwarg "
                "in favor of `enabled`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_enabled` kwarg in favor of `enabled`. "
            "`fp8_enabled` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        enabled = fp8_enabled
    if enabled is None:
        enabled = False

    if fp8_calibrating is not None:
        if calibrating is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_calibrating` kwarg "
                "in favor of `calibrating`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_calibrating` kwarg in favor of "
            "`calibrating`. `fp8_calibrating` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        calibrating = fp8_calibrating
    if calibrating is None:
        calibrating = False

    if fp8_recipe is not None:
        if recipe is None:
            warnings.warn(
                "make_graphed_callables has deprecated `fp8_recipe` kwarg in favor of "
                "`recipe`. `fp8_recipe` will be removed in a future release.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_recipe` kwarg "
                "in favor of `recipe`, but both kwargs are set."
            )
        recipe = fp8_recipe

    if fp8_group is not None:
        if amax_reduction_group is None:
            warnings.warn(
                "make_graphed_callables has deprecated `fp8_group` kwarg in favor of "
                "`amax_reduction_group`. `fp8_group` will be removed in a future release.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_group` kwarg "
                "in favor of `amax_reduction_group`, but both kwargs are set."
            )
        amax_reduction_group = fp8_group

    if fp8_weight_caching is not None:
        if cache_quantized_params is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_weight_caching` kwarg "
                "in favor of `cache_quantized_params`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_weight_caching` kwarg in favor of "
            "`cache_quantized_params`. `fp8_weight_caching` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        cache_quantized_params = fp8_weight_caching
    if cache_quantized_params is None:
        cache_quantized_params = False

    set_capture_start()
    with contextlib.ExitStack() as cleanup:
        cleanup.callback(set_capture_end)

        # Handle single module.
        just_one_callable = False
        if not isinstance(modules, tuple):
            just_one_callable = True
            modules = (modules,)

        if not isinstance(enabled, tuple):
            if not isinstance(enabled, bool):
                raise TypeError(
                    f"enabled must be a bool or a tuple of bools, but got {type(enabled).__name__}"
                )
            enabled = (enabled,) * len(modules)
        elif len(enabled) != len(modules):
            raise ValueError(
                f"enabled length ({len(enabled)}) must match modules length ({len(modules)})"
            )
        if not te_available and (
            any(enabled)
            or calibrating
            or recipe is not None
            or amax_reduction_group is not None
            or cache_quantized_params
        ):
            raise _te_required_error("FP8/TE-specific graph capture")
        if _activation_recompute and any(enabled):
            _validate_fp8_activation_recompute_support()
        if any(enabled) and recipe is None:
            recipe = get_default_fp8_recipe()
        elif not any(enabled):
            recipe = None
        module_uses_fp8 = dict(zip((id(m) for m in modules), enabled))
        discovered_generators = _get_tracked_cuda_generators(
            require_generators=_activation_recompute
        )
        tracked_generators = discovered_generators or ()

        # Store FP8 tensors to reset later.
        saved_fp8_tensors = save_fp8_tensors(modules, recipe=recipe)
        cleanup.callback(restore_fp8_tensors, modules, saved_fp8_tensors)
        if _activation_recompute and any(enabled):
            fp8_recompute_bookkeeping = _snapshot_fp8_recompute_bookkeeping(modules)
            cleanup.callback(_restore_fp8_recompute_bookkeeping, fp8_recompute_bookkeeping)

        # FP8 wrapper.
        old_call_funcs = {}

        def wrap_autocast(block):
            """Install a graph-aware autocast wrapper for one module class.

            :param block: Module instance whose class call operator is wrapped.
            :type block: torch.nn.Module
            """
            block_cls = type(block)
            if block_cls in old_call_funcs:
                return

            old_call_funcs[block_cls] = block_cls.__call__

            # Wrap the original call function of the module class.
            def call_func(self, *args, **kwargs):
                """Call a module under graph-aware Transformer Engine autocast.

                :param self: Module instance being called.
                :type self: torch.nn.Module
                :param args: Positional module arguments.
                :type args: Any
                :param kwargs: Keyword module arguments.
                :type kwargs: Any
                :return: Module outputs.
                :rtype: Any
                """
                fp8_enabled = module_uses_fp8.get(id(self), False)
                recompute_phase = _FP8_ACTIVATION_RECOMPUTE_PHASE.get()
                recompute_context = (
                    activation_recompute_forward(
                        activation_recompute=True, recompute_phase=recompute_phase
                    )
                    if fp8_enabled and recompute_phase is not None
                    else contextlib.nullcontext()
                )
                with (
                    autocast(
                        enabled=fp8_enabled,
                        calibrating=calibrating,
                        recipe=recipe,
                        amax_reduction_group=amax_reduction_group,
                        _graph=True,
                    ),
                    recompute_context,
                ):
                    outputs = old_call_funcs[block_cls](self, *args, **kwargs)
                return outputs

            block_cls.__call__ = call_func
            cleanup.callback(setattr, block_cls, "__call__", old_call_funcs[block_cls])

        forward_funcs = []
        for module in modules:
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"Graphing for {type(module)} is not supported.")
            wrap_autocast(module)
            forward_funcs.append(module)

        if just_one_callable:
            forward_funcs = forward_funcs[0]
        else:
            forward_funcs = tuple(forward_funcs)

        # Save RNG state and restore it even if warmup or capture fails.
        if discovered_generators is not None:
            generators = (
                torch.cuda.default_generators[torch.cuda.current_device()],
                *tracked_generators,
            )
            original_rng_states = tuple(generator.get_state() for generator in generators)
            for generator, state in zip(generators, original_rng_states):
                cleanup.callback(generator.set_state, state)
        else:
            original_rng_state = torch.cuda.get_rng_state()
            cleanup.callback(torch.cuda.set_rng_state, original_rng_state)

        return _make_graphed_callables(
            forward_funcs,
            sample_args,
            num_warmup_iters=num_warmup_iters,
            allow_unused_input=allow_unused_input,
            cache_quantized_params=cache_quantized_params,
            sample_kwargs=sample_kwargs,
            _order=_order,
            _num_layers_per_chunk=_num_layers_per_chunk,
            pool=pool,
            retain_graph_in_backward=retain_graph_in_backward,
            _reuse_graph_input_output_buffers=_reuse_graph_input_output_buffers,
            _clone_param_grads_on_return=_clone_param_grads_on_return,
            _input_output_aliases=_input_output_aliases,
            _activation_recompute=_activation_recompute,
            _activation_recompute_forward_grad_enabled=(_activation_recompute_forward_grad_enabled),
            _activation_recompute_regions=_activation_recompute_regions,
            _activation_recompute_order_slots=_activation_recompute_order_slots,
            pre_warmup_hook=pre_warmup_hook,
            post_warmup_hook=post_warmup_hook,
            capture_time_hooks=capture_time_hooks,
            capture_stream=capture_stream,
            use_main_grad=use_main_grad,
            _tracked_generators=tracked_generators or (),
        )
