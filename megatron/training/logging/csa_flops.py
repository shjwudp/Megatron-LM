# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model FLOPs for the GPT DSv4 compressed sparse attention path."""

import math
from collections import Counter


def _causal_pairs(total_tokens, seqlen_squared_sum, stride, cap=None):
    """Estimate sum(min(floor(p / stride), cap)), p=1..L, over documents.

    The runtime retains only sum(L) and sum(L**2). As in the DSA counter,
    use the length-weighted mean L=sum(L**2)/sum(L), interpolating the two
    adjacent integer lengths before scaling by the effective document count.
    Equal-length documents are exact; mixed lengths are an approximation.
    stride=1 counts the sliding window; stride>1 counts complete compressed keys.
    """
    if total_tokens <= 0 or seqlen_squared_sum <= 0:
        return 0.0
    mean_length = seqlen_squared_sum / total_tokens
    length = math.floor(mean_length)
    fraction = mean_length - length

    def count(size):
        groups = size // stride
        if cap is not None:
            groups = min(groups, cap)
        return stride * groups * (groups - 1) // 2 + groups * (size - stride * groups + 1)

    return ((1 - fraction) * count(length) + fraction * count(length + 1)) * (
        total_tokens / mean_length
    )


def csa_attention_flops(args, total_tokens, seqlen_squared_sum):
    """Return forward/backward model FLOPs for all GPT CSA and MTP layers.

    Use DSv4's actual projection shapes, not the legacy MLA CLI qk/kv defaults.
    Indexer inputs are detached: their projections need forward + wgrad only.
    Sparse indexer loss backpropagates through selected compressed keys, while
    retrieval scores all causal compressed keys once in forward.

    Excludes padding, activation/teacher/selected-score recomputation, optimizer,
    communication, top-k comparisons, norms, softmax and other elementwise work.
    This is a model FLOPs estimate, not executed-kernel FLOPs or hardware MFU.
    """
    depth = args.mtp_num_layers or 0
    ratios = list(args.csa_compress_ratios)
    if len(ratios) < args.num_layers + depth:
        raise ValueError("CSA FLOPs require one compression ratio per decoder and MTP layer.")
    decoder_ratios = ratios[: args.num_layers]
    mtp_ratios = ratios[args.num_layers : args.num_layers + depth]
    if depth and getattr(args, "mtp_use_repeated_layer", False):
        mtp_ratios = [mtp_ratios[0]] * depth
    layer_counts = Counter(decoder_ratios + mtp_ratios)
    if any(ratio not in (0, 4, 128) for ratio in layer_counts):
        raise ValueError("CSA compression ratios must be 0, 4 or 128.")

    hidden = args.hidden_size
    heads = args.num_attention_heads
    dim = args.v_head_dim  # Both Q/K and V are this wide, with one shared KV head.
    q_rank = args.q_lora_rank
    o_rank = args.output_projection_lora_rank
    groups = args.output_projection_groups
    projections = (
        hidden * q_rank
        + q_rank * heads * dim
        + hidden * dim
        + heads * dim * o_rank
        + groups * o_rank * hidden
    )
    window_pairs = _causal_pairs(
        total_tokens, seqlen_squared_sum, stride=1, cap=args.csa_window_size
    )
    total = 0.0
    for ratio, count in layer_counts.items():
        indexer = ratio == 4 and not args.csa_dense_mode
        all_compressed_pairs = selected_pairs = 0.0
        per_layer = 6 * total_tokens * projections
        if ratio:
            all_compressed_pairs = _causal_pairs(total_tokens, seqlen_squared_sum, ratio)
            selected_pairs = (
                _causal_pairs(total_tokens, seqlen_squared_sum, ratio, args.dsa_indexer_topk)
                if indexer
                else all_compressed_pairs
            )
            # Wkv and Wgate each produce coff * dim values per input token.
            coff = 2 if ratio == 4 else 1
            per_layer += 6 * total_tokens * 2 * hidden * coff * dim
        # QK and PV: 2 GEMMs * FMA 2 * forward/backward 3.
        per_layer += 12 * heads * dim * (window_pairs + selected_pairs)
        if indexer:
            index_heads = args.dsa_indexer_n_heads
            index_dim = args.dsa_indexer_head_dim
            loss_enabled = (args.dsa_indexer_loss_coeff or 0.0) > 0
            index_projections = (
                q_rank * index_heads * index_dim
                + hidden * index_heads
                + 4 * hidden * index_dim  # Indexer's own overlapping Wkv/Wgate.
            )
            per_layer += 2 * (2 if loss_enabled else 1) * total_tokens * index_projections
            backward_pairs = 0.0
            if loss_enabled:
                backward_pairs = (
                    selected_pairs if args.dsa_indexer_use_sparse_loss else all_compressed_pairs
                )
            per_layer += 2 * index_heads * index_dim * (all_compressed_pairs + 2 * backward_pairs)
        total += count * per_layer
    return total
