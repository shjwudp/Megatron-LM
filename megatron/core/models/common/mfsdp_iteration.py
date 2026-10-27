# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pipeline schedule boundaries for captured MFSDP prefetch orders."""

from functools import wraps

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)


def mfsdp_execution_order(schedule):
    """Fork all-gathers at schedule entry and preserve training orders during evaluation."""

    @wraps(schedule)
    def wrapped(*, model, forward_only=False, **kwargs):
        chunk = model[0] if isinstance(model, list) else model
        context = getattr(getattr(chunk, "module", chunk), "context", None)
        enabled = context is not None and (
            getattr(context, "_recording_execution_order", False)
            or getattr(context, "_execution_order_captured", False)
        )
        if not enabled:
            return schedule(model=model, forward_only=forward_only, **kwargs)
        # Combined 1F1B may start with a child unit and bypass the root forward hook.
        context.allgather_stream.wait_stream(context.current_stream())
        if not forward_only:
            return schedule(model=model, forward_only=False, **kwargs)
        previous = context.forward_order, context.backward_order
        context.forward_order, context.backward_order = ExecutionOrder(), ExecutionOrder()
        try:
            return schedule(model=model, forward_only=True, **kwargs)
        finally:
            context.forward_order, context.backward_order = previous

    return wrapped
