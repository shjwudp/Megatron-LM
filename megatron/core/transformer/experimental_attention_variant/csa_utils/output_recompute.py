# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release packed CSA outputs and restore their shared storage with local kernels."""

import torch

from megatron.core.event_tracker import event_tracker
from megatron.core.tensor_parallel.random import _get_share_storage

from .fused_sparse_attention import _csa_fwd_flash_mla


class CSAOutputRecompute:
    """One forward's O/LSE, shared by the attention backward and output projection.

    Inputs are detached storage aliases: the existing MLA checkpoint releases and
    restores Q/local KV, while compressed KV and indices are already retained by
    the attention Function. No autograd graph or collective is replayed here.
    """

    def __init__(self):
        self._inputs = None
        self._outputs = None
        self._released = False

    def record(self, query, kv_parts, sink, indices, lengths, scale, out_rope, out, lse):
        """Retain aliases after forward has completed the indexer loss and output RoPE."""
        if self._inputs is not None:
            raise RuntimeError("A CSA output replay can only record one forward.")
        self._recompute_label = event_tracker.recompute_label('csa_core_output')
        self._inputs = (
            query.detach(),
            tuple(part.detach() for part in kv_parts),
            sink.detach(),
            indices.detach(),
            lengths.detach(),
            scale,
            out_rope,
        )
        self._outputs = (out.detach(), lse.detach())

    def discard_output_and_register_recompute(self, output):
        """Discard both outputs after their last forward consumer; restore before backward."""
        if self._outputs is None or self._released:
            raise RuntimeError("CSA output replay requires one packed indexer-loss forward.")
        for tensor in self._outputs:
            tensor.untyped_storage().resize_(0)
        self._released = True
        output.register_hook(lambda _grad: self.rebuild())

    @torch.no_grad()
    def rebuild(self):
        """Restore O/LSE after the caller has restored Q and local/boundary KV."""
        if self._inputs is None:
            return
        if not self._released:
            raise RuntimeError("CSA output replay ran before its outputs were released.")
        with event_tracker.recompute_scope(self._recompute_label):
            query, kv_parts, sink, indices, lengths, scale, out_rope = self._inputs
            kv_full = torch.cat(kv_parts, dim=0)
            out, lse, _ = _csa_fwd_flash_mla(
                query, kv_full, indices, scale, attn_sink=sink, topk_length=lengths
            )
            if out_rope is not None:
                out_rope.apply_(out)
            # Repoint StorageImpl rather than TensorImpl: every saved view must see
            # the restored data, without incrementing autograd version counters.
            share_storage = _get_share_storage()
            for saved, restored in zip(self._outputs, (out, lse)):
                share_storage(saved, restored)
        self._inputs = self._outputs = None
