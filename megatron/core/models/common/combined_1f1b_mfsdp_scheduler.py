# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantization import (
    COLWISE,
    ROWWISE,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.schedule import (
    TraceAndReplayScheduler,
)


def _scheduler(module: FsdpModule) -> TraceAndReplayScheduler:
    """Return the context's trace-and-replay scheduler for ``module``."""
    scheduler = module.context.scheduler
    assert isinstance(
        scheduler, TraceAndReplayScheduler
    ), "Expected a TraceAndReplayScheduler on the FSDP context."
    return scheduler


def _make_unshard_forward_hook(owner: FsdpModule):
    """Forward pre-hook: unshard the owning FSDP module before the submodule forward."""

    def hook(submodule, _args, _kwargs):
        # The scheduler drives ``unshard()`` directly, so it must perform the root
        # stream sync that ``pre_forward`` otherwise does (see ``unshard``'s
        # docstring); it has to happen before the all-gather is issued.
        if owner.is_root():
            context = owner.context
            context.allgather_stream.wait_stream(context.current_stream())
        # A forward GEMM consumes the row-wise MXFP8 payload. If the same
        # materialization also has to serve a backward pass (activation
        # recomputation with no reshard between them), the module widens it.
        scheduler = _scheduler(owner)
        scheduler.issue_unshard(owner, ROWWISE)
        scheduler.wait_unshard(owner)

    return hook


def _make_unshard_backward_hook(owner: FsdpModule):
    """Create a backward pre-hook that unshards the owning FSDP module before the submodule backward."""

    def hook(submodule, _grad_output):
        # The backward GEMM consumes the column-wise MXFP8 payload.
        scheduler = _scheduler(owner)
        scheduler.issue_unshard(owner, COLWISE)
        scheduler.wait_unshard(owner)

    return hook


def _module_post_backward_hook(module: FsdpModule) -> None:
    scheduler = _scheduler(module)
    scheduler.reshard(module)
    scheduler.issue_reduce_gradients(module)


def end_microbatch(module: FsdpModule) -> None:
    """Close the trace-and-replay reduce-deferral scope for one microbatch.

    ``module`` is any FsdpModule of the microbatch's model chunk; the scheduler
    lives on the shared :class:`FsdpContext`. This is a no-op when trace-and-replay
    is disabled, so the call site does not have to know whether a scheduler exists.

    It is invoked from the combined-1F1B schedule once one microbatch's backward —
    including its delayed ``backward_dw`` wgrad — has fully retired, i.e. exactly
    where the schedule leaves the microbatch scope. See
    ``TraceAndReplayScheduler.end_microbatch`` for why that is a hard barrier.
    """
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    scheduler = module.context.scheduler
    if isinstance(scheduler, TraceAndReplayScheduler):
        scheduler.end_microbatch(module)


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    _scheduler(module).reshard(module)


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install the sub-module hooks required by MCore combined 1F1B."""

    def register_hooks(submodule, owner):
        if isinstance(submodule, FsdpModule):
            owner = submodule  # BEFORE registering: an FSDP unit owns itself
        if len(list(submodule.parameters(recurse=False))) > 0:
            submodule.register_forward_pre_hook(
                _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))
        for child in submodule.children():
            register_hooks(child, owner)

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."
    register_hooks(module, module)

    for submodule in module.modules():
        if isinstance(submodule, FsdpModule):
            submodule.register_post_backward_hook(_module_post_backward_hook)
