# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Select MFSDP v2 prefetch orders at virtual-pipeline pass boundaries."""

from contextlib import contextmanager

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


class VppMfsdpScheduler:
    """Keep prefetch within the chunk selected by the pipeline scheduler.

    The shared FSDP context describes all local chunks, but consecutive passes
    need not follow their construction order. Install a chunk's orders only
    while its pass runs; normal FSDP hooks still own gathers and reductions.
    """

    def __init__(self, chunks):
        self._chunks = []
        for chunk in chunks:
            units = {module for module in chunk.modules() if isinstance(module, FsdpModule)}
            contexts = {module.context for module in units}
            orders = []
            for context in contexts:
                context.ensure_finalized()
                orders.append(
                    (
                        context,
                        tuple(module for module in context.forward_order if module in units),
                        tuple(module for module in context.backward_order if module in units),
                    )
                )
            self._chunks.append(orders)

    @contextmanager
    def pass_context(self, chunk_id):
        """Scope both orders to this pass, including backward recomputation.

        Restore the shared context even if computation raises. New cursors
        start at the pass boundary selected by the VPP scheduler.
        """
        saved = []
        try:
            for context, forward, backward in self._chunks[chunk_id]:
                saved.append((context, context.forward_order, context.backward_order))
                context.forward_order = ExecutionOrder(forward)
                context.backward_order = ExecutionOrder(backward)
            yield
        finally:
            for context, forward, backward in reversed(saved):
                context.forward_order = forward
                context.backward_order = backward
