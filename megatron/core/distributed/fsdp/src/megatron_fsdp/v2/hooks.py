# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward and backward hook registration for Megatron-FSDP2."""

import functools
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from torch.utils.checkpoint import _StopRecomputationError

from .allocator import TracePoolAllocator
from .cuda_graph_runner import CudaGraphRunner
from .fsdp_module import FSDPModule, _FSDPState
from .utils import RegisterFSDPBackwardFunction

logger = logging.getLogger(__name__)


def _current_graph_task_id() -> int:
    """Return the active autograd graph task, or ``-1`` when unavailable.

    :return: Current autograd graph-task identifier.
    :rtype: int
    """
    getter = getattr(torch._C, "_current_graph_task_id", None)
    return getter() if callable(getter) else -1


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _is_activation_recompute(module: FSDPModule) -> bool:
    """Return whether ``module`` is running checkpoint recompute.

    :param module: M-FSDP module entering forward.
    :type module: FSDPModule
    :return: Whether this is the module's grad-enabled recompute forward.
    :rtype: bool
    """
    ctx = module._fsdp_root_context
    if not module.training or not torch.is_grad_enabled() or not ctx.backward_phase:
        return False
    try:
        from megatron.core.tensor_parallel.random import is_checkpointing

        mcore_checkpointing = is_checkpointing()
    except ImportError:
        mcore_checkpointing = False
    if not ctx.cuda_graph_activation_recompute:
        return (
            id(module) == ctx.backward_module
            or mcore_checkpointing
            or _current_graph_task_id() >= 0
        )
    checkpoint_recompute = mcore_checkpointing or _current_graph_task_id() >= 0
    if not module._fsdp_state.enable_cuda_graph or not module.cuda_graph_compatible:
        return id(module) == ctx.backward_module or checkpoint_recompute
    runner = getattr(ctx, "cuda_graph_runner", None)
    owns_module = getattr(runner, "owns_module", None)
    expects_recompute = getattr(runner, "expects_module_recompute", None)
    if callable(owns_module) and owns_module(module):
        return callable(expects_recompute) and expects_recompute(module) and checkpoint_recompute
    return id(module) == ctx.backward_module or mcore_checkpointing


def _cuda_graph_replay_phase(module: FSDPModule) -> str:
    """Return the graph program selected by the M-FSDP execution phase.

    :param module: M-FSDP module entering forward.
    :type module: FSDPModule
    :return: ``forward``, ``recompute``, or ``inference``.
    :rtype: str
    :raises RuntimeError: If a grad-enabled side forward runs during backward.
    """
    ctx = module._fsdp_root_context
    if not module.training or not torch.is_grad_enabled():
        return "inference"
    if _is_activation_recompute(module):
        return "recompute"
    if ctx.backward_phase and ctx.cuda_graph_activation_recompute:
        raise RuntimeError(
            "A grad-enabled M-FSDP forward ran during backward outside checkpoint recomputation"
        )
    return "forward"


def _find_fsdp_target(hook_module: nn.Module) -> Optional[FSDPModule]:
    """Return the nearest parent FSDPModule for *hook_module*.

    Used by fine-grained hooks registered on sub-modules to resolve the
    FSDPModule that owns the sub-module.  The reference is stored as
    ``_fsdp_parent_module`` during FSDP init (a ``weakref.ref`` to avoid
    reference cycles).

    Returns:
        The owning FSDPModule, or ``None`` if the module has no FSDP parent.
    """
    if isinstance(hook_module, FSDPModule):
        return hook_module
    parent_ref = getattr(hook_module, '_fsdp_parent_module', None)
    if parent_ref is not None:
        return parent_ref()
    return None


def _recover_stale_root_backward(target: FSDPModule) -> None:
    """Reset M-FSDP state left by an aborted backward before a new root forward.

    :param target: Root FSDP module starting a new training forward.
    :type target: FSDPModule
    """
    ctx = target._fsdp_root_context
    runner = getattr(ctx, "cuda_graph_runner", None)
    release_pending = getattr(runner, "release_pending", None)
    if callable(release_pending):
        release_pending()
    target._fsdp_state._post_backward_callback_queued = False
    ctx.forward_phase = False
    ctx.backward_phase = False
    ctx.backward_module = None
    ctx.backward_done_modules.clear()
    for fsdp_module in ctx.forward_order:
        fsdp_module._fsdp_cg_pending_backwards = 0
        fsdp_module._fsdp_pre_backward_done = False
        fsdp_module._fsdp_post_backward_hook_seen = False
        fsdp_module.post_backward_issued = False


def _should_recover_stale_root_backward(target: FSDPModule) -> bool:
    """Return whether a new root forward may recover an abandoned backward.

    :param target: Root M-FSDP module entering forward.
    :type target: FSDPModule
    :return: Whether single-invocation recovery is safe.
    :rtype: bool
    """
    ctx = target._fsdp_root_context
    return (
        target._fsdp_state._is_root
        and target.training
        and torch.is_grad_enabled()
        and ctx.backward_phase
        and _current_graph_task_id() < 0
        and getattr(ctx, "cuda_graph_max_pending_forwards", 1) == 1
    )


def _output_supports_backward(output: Any) -> bool:
    """Return whether a forward output can start autograd backward.

    :param output: Forward output PyTree.
    :type output: Any
    :return: Whether any tensor output requires gradients.
    :rtype: bool
    """
    flat_output, _ = tree_flatten(output)
    return any(isinstance(value, torch.Tensor) and value.requires_grad for value in flat_output)


@torch.compiler.disable
def mfsdp_forward_pre_hook(hook_module: nn.Module, args: Any, kwargs: Any):
    """Pre-forward hook for FSDP modules and fine-grained sub-modules.

    Resolves the target FSDPModule via :func:`_find_fsdp_target`, performs
    parameter unshard, root-phase bookkeeping, and (for direct FSDPModule
    calls only) CUDA graph capture.

    **Repeatability**: This function MUST be safe to call multiple times per
    module without observable overhead.  Fine-grained hook registration
    (``_register_forward_pre_hook(fine_grained=True)``) installs the hook on
    every sub-module of an FSDPModule.  When a sub-module's ``forward()`` is
    called, PyTorch triggers the pre-forward hook, which calls this function.
    If the enclosing FSDPModule is also directly invoked (and its own pre-forward
    hook fires), this function will be invoked again for the same target.
    The implementation must handle this gracefully — duplicating a no-op
    ``unshard()`` call or re-applying idempotent bookkeeping must not introduce
    measurable latency.
    """
    target = _find_fsdp_target(hook_module)
    if target is None:
        return

    ctx = target._fsdp_root_context
    assert not ctx.cuda_graph_active, "hooks must not fire during CUDA graph capture"
    direct_module_call = isinstance(hook_module, FSDPModule)
    if direct_module_call and _should_recover_stale_root_backward(target):
        _recover_stale_root_backward(target)
    replay_phase_stack = target.__dict__.get("_fsdp_forward_replay_phase_stack", [])
    replay_phase = (
        replay_phase_stack[-1][0]
        if not direct_module_call and replay_phase_stack
        else _cuda_graph_replay_phase(target)
    )
    replay_frame = (replay_phase, False)
    if direct_module_call:
        target.__dict__.setdefault("_fsdp_forward_replay_phase_stack", []).append(replay_frame)
    is_recompute = replay_phase == "recompute"

    # ---- root: forward-phase setup (once per micro-batch) ------------------
    if direct_module_call and target._fsdp_state._is_root and not ctx.backward_phase:
        # A plain torch optimizer clears ``dist_param.grad`` without entering
        # FSDPModule.zero_grad(). Sweep every FSDP parameter group at the root
        # boundary so stale distributed-gradient storage is released before
        # the first parameter unshard of the new forward.
        target._release_grad_storage_if_unused()
        # The last-backward contract guarantees that an optimizer decision has
        # happened before this next normal forward. Install updated main-weight
        # shards before any parameter unshard/prefetch can consume model weights.
        if ctx.model_weight_refresh_pending:
            target._copy_main_weights_to_model_weights()
        if ctx.enable_cuda_graph and ctx.cuda_graph_stream is None:
            ctx.cuda_graph_stream = torch.cuda.Stream()
            ctx.cuda_graph_pool = torch.cuda.graph_pool_handle()
        if ctx.cuda_graph_capture_pending and replay_phase == "forward":
            _maybe_capture_cuda_graphs(ctx, target)
        if replay_phase == "forward":
            ctx.forward_phase = True
            ctx.backward_phase = False

    # Deferred capture may install the wrapper on this same root module.
    cuda_graph_runner = getattr(ctx, "cuda_graph_runner", None)
    if cuda_graph_runner is not None and direct_module_call:
        cuda_graph_runner.prepare_module_replay(target, replay_phase)
    preflight = target.__dict__.get("_cuda_graph_preflight")
    if callable(preflight) and direct_module_call:
        preflight()
    if (
        direct_module_call
        and cuda_graph_runner is not None
        and target._fsdp_state.enable_cuda_graph
        and not getattr(target, "_fsdp_cg_installed", False)
        and is_recompute
    ):
        record_recompute = getattr(cuda_graph_runner, "record_module_recompute", None)
        if callable(record_recompute):
            record_recompute(target, args, kwargs)

    record_cuda_graph_module = (
        direct_module_call
        and target._fsdp_state.enable_cuda_graph
        and not getattr(target, "_fsdp_cg_installed", False)
        and replay_phase == "forward"
        and target.cuda_graph_compatible
    )
    if record_cuda_graph_module:
        if ctx.cuda_graph_runner is None:
            ctx.cuda_graph_runner = CudaGraphRunner(
                graph_pool=ctx.cuda_graph_pool,
                activation_recompute=ctx.cuda_graph_activation_recompute,
                max_pending_forwards=ctx.cuda_graph_max_pending_forwards,
            )
        ctx.cuda_graph_runner.preflight_record_module(target, replay_phase)
    if (
        direct_module_call
        and target._fsdp_state._is_root
        and replay_phase == "forward"
        and ctx.cuda_graph_runner is not None
    ):
        ctx.cuda_graph_runner.begin_forward_scope()

    if (
        direct_module_call
        and target._fsdp_state.enable_cuda_graph
        and ctx.cuda_graph_activation_recompute
        and replay_phase == "forward"
    ):
        target._fsdp_cg_pending_backwards += 1
        replay_frame = (replay_phase, True)
        target.__dict__["_fsdp_forward_replay_phase_stack"][-1] = replay_frame

    # ---- unshard parameters for this module -------------------------------
    if is_recompute:
        target.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)
        # Checkpoint recompute executes a forward kernel during backward.
        target.unshard(async_op=False, bwd_pass=False, prefetch=False)
    else:
        target.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=False)

    # ---- free stale grad data (safe to repeat, idempotent) ----------------
    for param_group in target._fsdp_param_groups:
        param_group._release_grad_storage_if_unused()

    # ---- CUDA graph: record sample args (first optimized micro-batch) -----
    # Capture runs at the next root forward after this sample is complete.
    if record_cuda_graph_module:
        ctx.cuda_graph_runner.record_module(target, args, kwargs)


@torch.compiler.disable
def mfsdp_post_forward_hook(module: nn.Module, *hook_args):
    """Post-forward hook: reshard parameters.

    Only supports direct FSDPModule calls.  Raises ``TypeError`` when
    called with a non-FSDPModule (fine-grained path is not yet handled).
    """
    if not isinstance(module, FSDPModule):
        raise TypeError(
            "mfsdp_post_forward_hook only supports FSDPModule, " f"got {type(module).__name__}"
        )
    ctx = module._fsdp_root_context
    assert not ctx.cuda_graph_active, "hooks must not fire during CUDA graph capture"
    output = hook_args[-1] if hook_args else None
    error = sys.exc_info()[1] if output is None else None
    checkpoint_early_stop = isinstance(error, _StopRecomputationError)
    failed_forward = error is not None and not checkpoint_early_stop
    keep_unsharded = checkpoint_early_stop or (
        ctx.backward_phase and id(module) == ctx.backward_module
    )
    replay_phase_stack = module.__dict__.get("_fsdp_forward_replay_phase_stack", [])
    if replay_phase_stack:
        replay_phase, pending_incremented = replay_phase_stack.pop()
        if not replay_phase_stack:
            module.__dict__.pop("_fsdp_forward_replay_phase_stack", None)
    else:
        replay_phase, pending_incremented = "inference", False
    detached_training_output = (
        output is not None and pending_incremented and not _output_supports_backward(output)
    )
    try:
        if detached_training_output:
            raise RuntimeError(
                "Activation-recompute CUDA graph forward produced no output that can "
                "start backward; detach outputs only after the checkpointed graph scope"
            )
        if (
            output is not None
            and ctx.cuda_graph_runner is not None
            and module._fsdp_state.enable_cuda_graph
            and not getattr(module, "_fsdp_cg_installed", False)
            and replay_phase == "forward"
            and module.cuda_graph_compatible
        ):
            ctx.cuda_graph_runner.record_module_output(module, output)
    finally:
        if (failed_forward or detached_training_output) and pending_incremented:
            module._fsdp_cg_pending_backwards = max(0, module._fsdp_cg_pending_backwards - 1)
        if failed_forward or detached_training_output:
            runner = getattr(ctx, "cuda_graph_runner", None)
            release_pending = getattr(runner, "release_pending", None)
            if callable(release_pending):
                release_pending()
        # Checkpoint early-stop is a successful recompute. Other exceptions
        # must release the current module before propagating.
        if not keep_unsharded:
            module.reshard()


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def _register_forward_pre_hook(module: FSDPModule, fine_grained: bool = False) -> None:
    """Register a pre-forward hook on the FSDP module or its sub-modules.

    Args:
        fsdp_module: The FSDP module to instrument.
        fine_grained: If ``True``, register on every sub-module of
            *fsdp_module* (for EP-overlap / 1F1B schedules).
            ``_fsdp_parent_module`` must already be set on sub-modules
            (done by :meth:`FSDPModule._init_fsdp_state`).
    """
    if fine_grained:
        for submodule in module.modules():
            fsdp_module = _find_fsdp_target(submodule)
            if fsdp_module is None or fsdp_module is not module:
                continue
            submodule.register_forward_pre_hook(
                mfsdp_forward_pre_hook, prepend=True, with_kwargs=True
            )
    else:
        module.register_forward_pre_hook(mfsdp_forward_pre_hook, prepend=True, with_kwargs=True)


def _register_forward_hook(module: FSDPModule):
    """Register post-forward hook to reshard parameters."""
    module._mfsdp_forward_hook = module.register_forward_hook(
        mfsdp_post_forward_hook, always_call=True
    )


# ---------------------------------------------------------------------------
# Internal: backward hook helpers
# ---------------------------------------------------------------------------


@torch.compiler.disable
def mfsdp_pre_backward_setup(
    hook_module: nn.Module, grads: Any = None, skip_final_callback: bool = False
):
    """Pre-backward hook for FSDP modules and fine-grained sub-modules.

    Resolves the target FSDPModule via :func:`_find_fsdp_target`, performs
    backward-phase root setup, parameter unshard, and TE gradient-fusion
    bookkeeping.  The ``_fsdp_pre_backward_done`` flag prevents redundant
    calls when multiple sub-modules share the same parent.

    Compatible with ``register_multi_grad_hook`` callback signature
    (module, grads).

    Args:
        hook_module: Module whose backward pass is about to start.
        grads: Gradients from ``register_multi_grad_hook`` (unused).
        skip_final_callback: If ``True``, do **not** auto-enqueue
            ``mfsdp_post_backward_final_callback``.  The caller is
            responsible for calling it manually (used by the 1F1B EP
            overlap schedule).
    """
    target = _find_fsdp_target(hook_module)
    if target is None:
        return
    if target._fsdp_pre_backward_done:
        return

    _pre_backward_setup(target, skip_final_callback=skip_final_callback)
    target._fsdp_pre_backward_done = True


def _defer_cuda_graph_grad_reduce(module: FSDPModule) -> bool:
    """Return whether parameter AccumulateGrad must finish before reduction."""
    if not getattr(module, "_fsdp_cg_installed", False):
        return False
    for param_group in module._fsdp_param_groups:
        if param_group.sharding_strategy not in ("optim_grads", "optim_grads_params"):
            continue
        for param in param_group.params:
            if (
                param.requires_grad
                and not getattr(param, "_mfsdp_recorded_te_wgrad", False)
                and param_group.main_grad_buffer.dtype != param.dtype
            ):
                return True
    return False


@torch.compiler.disable
def mfsdp_post_backward_hook(module: nn.Module, *, _runner_completion_claimed: bool = False):
    """Post-backward hook: reshard parameters and reduce gradients.

    Only supports direct FSDPModule calls.  Raises ``TypeError`` when
    called with a non-FSDPModule (fine-grained path is not yet handled).
    """
    if not isinstance(module, FSDPModule):
        raise TypeError(
            "mfsdp_post_backward_hook only supports FSDPModule, " f"got {type(module).__name__}"
        )
    ctx = module._fsdp_root_context
    assert not ctx.cuda_graph_active, "hooks must not fire during CUDA graph capture"

    for submodule in module._get_fsdp_modules(recursive=True):
        if submodule.post_backward_issued:
            continue
        pending_backwards = getattr(submodule, "_fsdp_cg_pending_backwards", 0)
        direct_graph_hook = submodule is module and submodule._fsdp_state.enable_cuda_graph
        if submodule._fsdp_post_backward_hook_seen and not (
            direct_graph_hook and _runner_completion_claimed
        ):
            continue
        runner = getattr(ctx, "cuda_graph_runner", None)
        owns_module = getattr(runner, "owns_module", None)
        runner_owns_module = direct_graph_hook and callable(owns_module) and owns_module(submodule)
        if runner_owns_module and not _runner_completion_claimed:
            runner.complete_module_backward(submodule, strict=True)
        submodule._fsdp_post_backward_hook_seen = True
        if direct_graph_hook and pending_backwards > 0:
            pending_backwards -= 1
            submodule._fsdp_cg_pending_backwards = pending_backwards
        if pending_backwards > 0:
            submodule.reshard()
            if not _defer_cuda_graph_grad_reduce(submodule):
                submodule.reduce_grad(async_op=False)
            submodule._fsdp_pre_backward_done = False
            continue
        ctx.backward_done_modules.add(id(submodule))
        submodule.reshard()
        if not _defer_cuda_graph_grad_reduce(submodule):
            submodule.reduce_grad(async_op=ctx.enable_async_reduce_grad)
        submodule.post_backward_issued = True
    ctx._advance_backward_module()


@torch.compiler.disable
def mfsdp_post_backward_final_callback(root_module: nn.Module):
    """Finalise the backward pass: drain skipped modules, reset state,
    clear fine-grained flags, and (on the first micro-batch) transition
    the bucket allocator from trace to optimized plan.

    Only supports the root FSDP module.  Raises ``TypeError`` if
    *root_module* is not an FSDPModule, or ``RuntimeError`` if it is
    not marked as root.
    """
    if not isinstance(root_module, FSDPModule):
        raise TypeError(
            "mfsdp_post_backward_final_callback only supports FSDPModule, "
            f"got {type(root_module).__name__}"
        )
    if not root_module._fsdp_state._is_root:
        raise RuntimeError("mfsdp_post_backward_final_callback requires root FSDP module")

    ctx = root_module._fsdp_root_context
    assert not ctx.cuda_graph_active, "hooks must not fire during CUDA graph capture"

    # ---- handle modules whose per-module post-backward was skipped ----
    for module in reversed(ctx.forward_order):
        pending_backwards = getattr(module, "_fsdp_cg_pending_backwards", 0)
        if (
            pending_backwards > 0
            and module._fsdp_state.enable_cuda_graph
            and not module.post_backward_issued
            and not module._fsdp_post_backward_hook_seen
        ):
            runner = getattr(ctx, "cuda_graph_runner", None)
            complete_module_backward = getattr(runner, "complete_module_backward", None)
            completion_claimed = callable(complete_module_backward) and complete_module_backward(
                module, allow_unarmed=True
            )
            backward_armed = completion_claimed or (
                runner is None and module._fsdp_pre_backward_done
            )
            if backward_armed:
                # A first-stage input may not require grad, so its input-side
                # post-backward hook is absent even though backward reached it.
                mfsdp_post_backward_hook(module, _runner_completion_claimed=True)
                continue
        if pending_backwards > 0:
            if _defer_cuda_graph_grad_reduce(module):
                module.reduce_grad(async_op=False)
            continue
        deferred_reduce = _defer_cuda_graph_grad_reduce(module)
        if not module.post_backward_issued:
            module.reshard()
        if not module.post_backward_issued or deferred_reduce:
            # Mixed-dtype graph grads are published by parameter AccumulateGrad,
            # which may run after the input-side post hook. Reduce them here,
            # after the autograd engine has completed, one module at a time.
            module.reduce_grad(async_op=ctx.enable_async_reduce_grad and not deferred_reduce)

    # Direct-bound grads can leave a late alias after reduction. Fused wgrad
    # also returns a dummy compute grad while the main-grad buffer is authoritative.
    for module in ctx.forward_order:
        if getattr(module, "_fsdp_cg_pending_backwards", 0) > 0:
            continue
        enable_cuda_graph = getattr(
            getattr(module, "_fsdp_state", None), "enable_cuda_graph", False
        )
        for param_group in module._fsdp_param_groups:
            reduce_during_backward = param_group.sharding_strategy in (
                "optim_grads",
                "optim_grads_params",
            )
            for param in param_group.params:
                recorded_fused_wgrad = getattr(param, "_mfsdp_recorded_te_wgrad", False)
                grad_added_to_main_grad = getattr(param, "grad_added_to_main_grad", False)
                if enable_cuda_graph and grad_added_to_main_grad:
                    setattr(param, "_mfsdp_recorded_te_wgrad", True)
                    recorded_fused_wgrad = True
                main_grad_is_authoritative = grad_added_to_main_grad or recorded_fused_wgrad
                if (
                    main_grad_is_authoritative
                    and not reduce_during_backward
                    and not ctx.is_last_backward
                ):
                    param_group._main_grad_buffer_has_unreduced_data = True
                if reduce_during_backward or main_grad_is_authoritative:
                    param.grad = None

    # ---- drain pending async reduce-grad events -----------------------
    stream = ctx.rs_stream
    for buckets in ctx.reduce_grad_buckets.values():
        while len(buckets) > 0:
            event, param_group = buckets.pop()
            event.wait()
            param_group.release_grad_buffer()
    torch.cuda.current_stream().wait_stream(stream)

    # ``is_last_backward`` is the optimizer-step boundary. The optimizer may
    # install model weights explicitly; otherwise the next normal pre-forward
    # hook does it lazily. Activation recompute is excluded by that hook.
    if ctx.is_last_backward:
        ctx.model_weight_refresh_pending = True

    # ---- reset root / context state for the next micro-batch ----------
    root_module._fsdp_state._post_backward_callback_queued = False
    ctx.backward_phase = False
    ctx.backward_module = None
    ctx.backward_done_modules.clear()

    # ---- clear fine-grained pre-backward flags -------------------------
    for module in ctx.forward_order:
        module._fsdp_pre_backward_done = False
        if getattr(module, "_fsdp_cg_pending_backwards", 0) > 0:
            module._fsdp_post_backward_hook_seen = False

    if any(getattr(module, "_fsdp_cg_pending_backwards", 0) for module in ctx.forward_order):
        return

    # ---- trace → optimized transition (first micro-batch only) --------
    if isinstance(ctx.bucket_allocator, TracePoolAllocator):
        bucket_alloc = ctx.bucket_allocator
        if bucket_alloc.phase == "trace":
            if torch.distributed.get_rank() == 0:
                logger.debug(bucket_alloc.dump_trace())
                for m in ctx.forward_order:
                    logger.debug(f"module_id={id(m)}, module_name={m._fsdp_module_name}")
            bucket_alloc.plan()

        elif bucket_alloc.phase != "optimized":
            raise ValueError(f"Unexpected bucket allocator phase: {bucket_alloc.phase}")

    # ---- CUDA graph: batch capture (after first optimized forward+backward) --
    runner = ctx.cuda_graph_runner
    if runner is not None and not runner.captured:
        _maybe_capture_cuda_graphs(ctx, root_module, defer=True)


def _maybe_capture_cuda_graphs(ctx, root_module, defer=False) -> None:
    """Request or run batch CUDA Graph capture.

    :param ctx: M-FSDP root context.
    :type ctx: _FSDPRootContext
    :param root_module: Root M-FSDP module.
    :type root_module: FSDPModule
    :param defer: Defer capture until the next root forward.
    :type defer: bool
    """
    runner = ctx.cuda_graph_runner
    if runner is None:
        return
    if runner.captured:
        ctx.cuda_graph_capture_pending = False
        return
    allocator = ctx.bucket_allocator
    assert isinstance(
        allocator, TracePoolAllocator
    ), "CUDA graph capture requires TracePoolAllocator"
    assert allocator.phase == "optimized", (
        f"CUDA graph capture requires allocator phase='optimized', " f"got '{allocator.phase}'"
    )
    if defer:
        ctx.cuda_graph_capture_pending = True
        return
    if not root_module.training or not torch.is_grad_enabled():
        ctx.cuda_graph_capture_pending = True
        return
    runner.capture_and_install(root_module, capture_stream=ctx.cuda_graph_stream)
    ctx.cuda_graph_capture_pending = not runner.captured


# ---------------------------------------------------------------------------
# Internal: backward hook helpers
# ---------------------------------------------------------------------------


def _create_custom_backward_hook(
    module: nn.Module, custom_backward_handler: Callable, ctx_module: Optional[nn.Module] = None
):
    """Wrap *module* so that ``custom_backward_handler`` fires as a
    pre-backward hook via ``register_multi_grad_hook``.

    Args:
        module: Module whose output tensors are instrumented.
        custom_backward_handler: Callback invoked when backward reaches
            this module.
        ctx_module: Module whose ``_fsdp_root_context`` is checked for
            CUDA-graph safety.  Defaults to *module*.
    """
    _ctx_source = ctx_module if ctx_module is not None else module

    @torch.compiler.disable
    def forward_hook(_module, inputs, output):
        if hasattr(_ctx_source, '_fsdp_root_context'):
            assert (
                not _ctx_source._fsdp_root_context.cuda_graph_active
            ), "hooks must not fire during CUDA graph capture"
        output = tree_map(lambda t: t.view_as(t) if torch.is_tensor(t) else t, output)

        output_list = []
        if isinstance(output, torch.Tensor):
            output_list = [output]
        elif isinstance(output, (tuple, list)):
            output_list = [t for t in output if isinstance(t, torch.Tensor)]

        target = _find_fsdp_target(_module)
        runner = (
            getattr(target._fsdp_root_context, "cuda_graph_runner", None)
            if target is not None
            else None
        )
        invocation_token = None
        if (
            runner is not None
            and _module is target
            and not target._fsdp_root_context.backward_phase
        ):
            invocation_token = runner.backward_invocation_token(target)

        def run_backward_handler(grads):
            if runner is not None:
                runner.select_backward_invocation(target, invocation_token)
            custom_backward_handler(_module, grads)

        torch.autograd.graph.register_multi_grad_hook(output_list, run_backward_handler, mode="any")
        return output

    return module.register_forward_hook(forward_hook)


def _pre_backward_setup(module: FSDPModule, skip_final_callback: bool = False):
    """Shared pre-backward logic: root setup, unshard, TE flags.

    Used by both the normal and fine-grained backward pre-hook paths.

    Args:
        module: The FSDPModule whose backward is starting.
        skip_final_callback: If ``True``, do not enqueue the post-backward
            final callback.  The caller must call
            ``mfsdp_post_backward_final_callback`` manually later.

    """
    ctx = module._fsdp_root_context
    assert not ctx.cuda_graph_active, "hooks must not fire during CUDA graph capture"

    # ---- root: backward-phase setup -----------------------------------
    if module._fsdp_state._is_root:
        ctx.backward_done_modules.clear()
        ctx.forward_phase = False
        ctx.backward_phase = True
        ctx._advance_backward_module()
        if not skip_final_callback and not module._fsdp_state._post_backward_callback_queued:
            _register_post_backward_final_callback(module._fsdp_state, module)

    # ---- unshard params for backward compute --------------------------
    if ctx.cuda_graph_runner is not None:
        ctx.cuda_graph_runner.record_module_backward(module)
    module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)
    if getattr(module, "_fsdp_cg_activation_recompute", False):
        # RF replays before B and needs this module's forward buffer.
        module.unshard(async_op=False, bwd_pass=False)

    # ---- reset per-module bookkeeping ---------------------------------
    module.post_backward_issued = False
    module._fsdp_post_backward_hook_seen = False

    # ---- Transformer Engine gradient-accumulation fusion ---------------
    for param_group in module._fsdp_param_groups:
        has_fused_wgrad = any(
            getattr(param, "_mfsdp_recorded_te_wgrad", False) for param in param_group.params
        )
        for param in param_group.params:
            param.grad_added_to_main_grad = False
            if getattr(module, "_fsdp_cg_installed", False):
                param.overwrite_main_grad = param_group.sharding_strategy in (
                    "optim_grads_params",
                    "optim_grads",
                )
            else:
                param.overwrite_main_grad = param_group.sharding_strategy in (
                    "optim_grads_params",
                    "optim_grads",
                ) or not getattr(param_group, "_main_grad_buffer_has_unreduced_data", False)
        if module._fsdp_state.enable_full_iteration_cuda_graph:
            param_group._init_dist_grads()
        # Keep per-module CUDA graph trace and replay on the same compatible
        # main-grad buffer allocation. Full-iteration graphs manage optimizer
        # gradient storage through their separate persistent-buffer path.
        if (
            not module._fsdp_state.enable_full_iteration_cuda_graph
            and module._fsdp_state.enable_cuda_graph
            and param_group.requires_grad
            and param_group.sharding_strategy in ("optim_grads", "optim_grads_params")
            and param_group.main_grad_buffer is not None
            and (
                has_fused_wgrad or param_group.main_grad_buffer.dtype == param_group.params[0].dtype
            )
        ):
            param_group._init_dist_grads()
            param_group.main_grad_buffer.fetch_buffer()
        if has_fused_wgrad:
            for param in param_group.params:
                if getattr(param, "_mfsdp_recorded_te_wgrad", False):
                    param.main_grad = param.get_main_grad()

    return ctx


# ---------------------------------------------------------------------------
# Backward hook registration
# ---------------------------------------------------------------------------


def _register_backward_pre_hook(
    module: FSDPModule, fine_grained: bool = False, skip_final_callback: bool = False
):
    """Register backward pre-hook using multi-grad hooks on output tensors.

    Attaches a ``register_multi_grad_hook`` to every tensor output of
    ``module.forward()``.  When autograd reaches this module during the
    backward pass, the hook fires *before* the module's own backward,
    giving FSDP a chance to unshard parameters for gradient computation.
    """
    if fine_grained:
        for submodule in module.modules():
            fsdp_module = _find_fsdp_target(submodule)
            if fsdp_module is None or fsdp_module is not module:
                continue
            submodule._mfsdp_backward_pre_hook = _create_custom_backward_hook(
                submodule,
                custom_backward_handler=lambda m, g: mfsdp_pre_backward_setup(
                    m, g, skip_final_callback=skip_final_callback
                ),
                ctx_module=module,
            )
        return

    module._mfsdp_backward_pre_hook = _create_custom_backward_hook(
        module,
        custom_backward_handler=lambda m, g: mfsdp_pre_backward_setup(
            m, g, skip_final_callback=skip_final_callback
        ),
    )


def _register_backward_hook(module: FSDPModule):
    """
    Register backward hook using autograd Function.

    This inserts a RegisterFSDPBackwardFunction in the backward pass
    that triggers ``mfsdp_post_backward_hook`` after gradients are
    computed — resharding parameters and reducing gradients.
    """

    @torch.compiler.disable
    def _register_post_backward_hook(
        post_backward_hook: Callable,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Register a post-backward hook by inserting an autograd Function.

        This approach works by registering a pre-forward hook that wraps
        input tensors in an autograd Function. The Function's backward
        calls the post_backward_hook after gradients are computed.
        """
        assert (
            not module._fsdp_root_context.cuda_graph_active
        ), "hooks must not fire during CUDA graph capture"
        if not torch.is_grad_enabled():
            return args, kwargs

        # Flatten args and kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)

        # Filter to tensors with gradients
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)

        if len(inp_tensors) == 0:
            return args, kwargs

        # Wrap inputs in autograd Function.
        # The Function's backward will call post_backward_hook.
        inp_tensors = RegisterFSDPBackwardFunction.apply(
            functools.partial(post_backward_hook, module), *inp_tensors
        )

        # Restore args and kwargs
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)

        return args, kwargs

    module._mfsdp_backward_hook = module.register_forward_pre_hook(
        functools.partial(_register_post_backward_hook, mfsdp_post_backward_hook), with_kwargs=True
    )


# ---------------------------------------------------------------------------
# Post-backward final callback
# ---------------------------------------------------------------------------


def _register_post_backward_final_callback(state: _FSDPState, module: nn.Module) -> None:
    """
    Enqueue a *single* engine callback that fires after every module's
    backward pass has completed.

    Registered once by the root FSDP module (avoids duplicates).
    Delegates to :func:`mfsdp_post_backward_final_callback`.
    """
    assert state._is_root, "Only root FSDP should register post-backward callback"
    if state._post_backward_callback_queued:
        return

    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(mfsdp_post_backward_final_callback, module)
    )
