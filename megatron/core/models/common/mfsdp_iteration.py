# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration boundary shared by MFSDP v2 pipeline schedule entry points."""

from contextlib import ExitStack
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
        contexts = dict.fromkeys(
            module.context
            for chunk in chunks
            for module in chunk.modules()
            if isinstance(module, FsdpModule)
        )
        if not contexts:
            return schedule(*args, **kwargs)
        config = getattr(chunks[0], "config", None)
        if getattr(config, "cuda_graph_impl", "none") != "none":
            return schedule(*args, **kwargs)
        if any(context.iteration_order.active for context in contexts):
            if not all(context.iteration_order.active for context in contexts):
                raise RuntimeError("All FSDP contexts must join the same outer iteration scope.")
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
        with ExitStack() as stack:
            orders = []
            try:
                for context in contexts:
                    context.ensure_finalized()
                    order = context.iteration_order
                    if not forward_only and not order.captured:
                        order = stack.enter_context(capture(context))
                    else:
                        context.allgather_stream.wait_stream(context.current_stream())
                        if forward_only:
                            order.mode = "demand"
                        else:
                            order.reset()
                            order.mode = "replay"
                    orders.append(order)
                result = schedule(*args, **kwargs)
                if not forward_only and any(order.position != len(order) for order in orders):
                    raise RuntimeError("FSDP iteration ended before consuming its captured order.")
                return result
            except BaseException:
                if not forward_only:
                    for order in orders:
                        order.captured = False
                raise
            finally:
                for order in orders:
                    order.mode = "static"

    return wrapped
