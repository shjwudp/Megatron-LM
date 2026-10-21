# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration boundary shared by MFSDP v2 pipeline schedule entry points."""

from contextlib import nullcontext
from functools import wraps
from inspect import signature

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.iteration import capture
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


def with_mfsdp_iteration(schedule):
    """Record/replay hooks across the complete schedule, including every VPP chunk.

    Nested combined calls join the outer scope. Non-MFSDP models and CUDA graph
    schedules retain their existing behavior; graph replay bypasses Python hooks
    and needs a separate recording lifecycle.
    """
    parameters = signature(schedule)

    @wraps(schedule)
    def wrapped(*args, **kwargs):
        arguments = parameters.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        models = values["model"]
        chunks = models if isinstance(models, list) else [models]
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
            return schedule(*args, **kwargs)
        config = getattr(chunks[0], "config", None)
        if getattr(config, "cuda_graph_impl", "none") != "none":
            return schedule(*args, **kwargs)
        order = context.iteration_order
        if order.active:
            return schedule(*args, **kwargs)
        forward_only = values.get("forward_only", False) or not torch.is_grad_enabled()
        if (
            forward_only
            and len(chunks) == 1
            and not getattr(config, "virtual_pipeline_model_parallel_size", None)
        ):
            # Ordinary validation retains static forward prefetch.
            return schedule(*args, **kwargs)
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
                result = schedule(*args, **kwargs)
                if not forward_only and order.position != len(order):
                    raise RuntimeError("FSDP iteration ended before consuming its captured order.")
                return result
            except BaseException:
                if not forward_only:
                    order.captured = False
                raise
            finally:
                order.mode = "static"

    return wrapped
