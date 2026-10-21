# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration boundary shared by MFSDP v2 pipeline schedule entry points."""

from contextlib import contextmanager, nullcontext
from functools import wraps
from inspect import signature

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.iteration import capture
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


@contextmanager
def _mfsdp_iteration(model, forward_only=False):
    """Manage one shared context across a complete scheduler invocation.

    Nested calls join the active scope. CUDA graph schedules bypass capture
    because their replay does not execute Python hooks.
    """
    chunks = model if isinstance(model, list) else [model]
    # PP constructs every VPP chunk inside the same fully_shard_context.
    context = next(
        (
            module.context
            for chunk in chunks
            for module in chunk.modules()
            if isinstance(module, FsdpModule)
        ),
        None,
    )
    if context is None:
        yield
        return
    config = getattr(chunks[0], "config", None)
    if getattr(config, "cuda_graph_impl", "none") != "none":
        yield
        return
    order = context.iteration_order
    if order.active:
        yield
        return
    forward_only = forward_only or not torch.is_grad_enabled()
    if (
        forward_only
        and len(chunks) == 1
        and not getattr(config, "virtual_pipeline_model_parallel_size", None)
    ):
        # Ordinary validation retains static forward prefetch.
        yield
        return
    # VPP validation has a different schedule. Gather on demand without
    # consuming or replacing the single captured training order.
    recording = not forward_only and not order.captured
    scope = capture(context) if recording else nullcontext(order)
    with scope as order:
        try:
            if not recording:
                context.ensure_finalized()
                context.allgather_stream.wait_stream(context.current_stream())
                if forward_only:
                    order.mode = "demand"
                else:
                    order.reset()
                    order.mode = "replay"
            yield
            if not forward_only and order.position != len(order):
                raise RuntimeError("FSDP iteration ended before consuming its captured order.")
        except BaseException:
            if not forward_only:
                order.captured = False
            raise
        finally:
            order.mode = "static"


def with_mfsdp_iteration(schedule):
    """Give a pipeline scheduler an MFSDP capture/replay boundary."""
    parameters = signature(schedule)

    @wraps(schedule)
    def wrapped(*args, **kwargs):
        arguments = parameters.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        with _mfsdp_iteration(values["model"], values.get("forward_only", False)):
            return schedule(*args, **kwargs)

    return wrapped
