# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA FLOPs regressions against enumerated causal masks and a small model."""

from types import SimpleNamespace

import pytest

from megatron.training.logging.csa_flops import _causal_pairs, csa_attention_flops
from megatron.training.training import num_floating_point_operations


def _args(**overrides):
    args = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        v_head_dim=4,
        q_lora_rank=3,
        output_projection_groups=2,
        output_projection_lora_rank=2,
        csa_window_size=3,
        csa_compress_ratios=[4],
        csa_dense_mode=False,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=2,
        dsa_indexer_topk=2,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        num_layers=1,
        mtp_num_layers=None,
        mtp_use_repeated_layer=False,
        experimental_attention_variant="dsv4_hybrid",
        multi_latent_attention=True,
        group_query_attention=False,
        hybrid_layer_pattern=None,
        seq_length=16,
        decoder_seq_length=None,
        swiglu=True,
        ffn_hidden_size=16,
        num_experts=None,
        moe_ffn_hidden_size=None,
        moe_latent_size=None,
        moe_shared_expert_intermediate_size=None,
        # Intentionally stale MLA defaults must not determine DSv4 dimensions.
        qk_head_dim=1,
        qk_pos_emb_head_dim=1,
        kv_lora_rank=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _enumerate_pairs(length, stride, cap=None):
    """Count visible columns of each causal row without a closed-form formula."""
    total = 0
    for position in range(1, length + 1):
        visible = [end for end in range(stride, length + 1, stride) if end <= position]
        total += len(visible if cap is None else visible[:cap])
    return total


@pytest.mark.parametrize("length", [0, 1, 3, 4, 7, 8, 9, 15, 16, 127, 128, 129])
@pytest.mark.parametrize("stride,cap", [(1, 3), (4, None), (4, 2), (128, None)])
def test_equal_length_pair_counts_match_causal_mask(length, stride, cap):
    actual = _causal_pairs(3 * length, 3 * length**2, stride, cap)
    assert actual == pytest.approx(3 * _enumerate_pairs(length, stride, cap))


def test_fractional_length_interpolation():
    # sum(L)=8, sum(L^2)=34 -> effective L=4.25. No rounding away the first pool.
    expected = (0.75 * _enumerate_pairs(4, 4) + 0.25 * _enumerate_pairs(5, 4)) * 8 / 4.25
    assert _causal_pairs(8, 34, 4) == pytest.approx(expected)


@pytest.mark.parametrize("loss_coeff,sparse_loss", [(0.0, True), (0.01, True), (0.01, False)])
def test_small_csa_model_with_indexer(loss_coeff, sparse_loss):
    args = _args(dsa_indexer_loss_coeff=loss_coeff, dsa_indexer_use_sparse_loss=sparse_loss)
    length = 16
    window = _enumerate_pairs(length, 1, 3)
    selected = _enumerate_pairs(length, 4, 2)
    all_keys = _enumerate_pairs(length, 4)
    # Hand-counted projection parameters: Q/KV=80, grouped O=48, compressor=128.
    # Indexer=92: detached inputs require forward+wgrad, or forward only without KL.
    linear = (1536 + (368 if loss_coeff else 184)) * length
    indexer = 8 * all_keys
    if loss_coeff:
        indexer += 16 * (selected if sparse_loss else all_keys)
    expected = linear + 96 * (window + selected) + indexer
    assert csa_attention_flops(args, length, length**2) == expected


@pytest.mark.parametrize(
    "ratio,dense_mode,linear", [(0, False, 768), (4, True, 1536), (128, False, 1152)]
)
def test_paths_without_indexer(ratio, dense_mode, linear):
    args = _args(csa_compress_ratios=[ratio], csa_dense_mode=dense_mode)
    length = 129
    pairs = _enumerate_pairs(length, 1, 3)
    if ratio:
        pairs += _enumerate_pairs(length, ratio)
    assert csa_attention_flops(args, length, length**2) == linear * length + 96 * pairs


def test_mtp_uses_explicit_ratio_tail_and_repeated_layer():
    args = _args(num_layers=2, mtp_num_layers=2, csa_compress_ratios=[4, 0, 128, 4])
    layer_costs = {
        ratio: csa_attention_flops(_args(csa_compress_ratios=[ratio]), 129, 129**2)
        for ratio in (0, 4, 128)
    }
    assert csa_attention_flops(args, 129, 129**2) == (
        2 * layer_costs[4] + layer_costs[0] + layer_costs[128]
    )
    args.mtp_use_repeated_layer = True
    assert csa_attention_flops(args, 129, 129**2) == (
        layer_costs[4] + layer_costs[0] + 2 * layer_costs[128]
    )
    assert args.csa_compress_ratios == [4, 0, 128, 4]


def test_invalid_ratio_configuration_is_not_silently_counted():
    with pytest.raises(ValueError, match="decoder and MTP"):
        csa_attention_flops(_args(mtp_num_layers=1), 16, 256)
    with pytest.raises(ValueError, match="0, 4 or 128"):
        csa_attention_flops(_args(csa_compress_ratios=[8]), 16, 256)


def test_empty_batch_and_stale_mla_dimensions():
    args = _args()
    assert csa_attention_flops(args, 0, 0) == 0
    cfg = SimpleNamespace(tokenizer=SimpleNamespace(padded_vocab_size=32))
    before = num_floating_point_operations(args, cfg, batch_size=1)
    args.qk_head_dim, args.kv_lora_rank = 987, 654
    assert num_floating_point_operations(args, cfg, batch_size=1) == before


@pytest.mark.parametrize("packed", [False, True])
def test_training_counter_uses_csa_and_keeps_dense_mlp_and_logits(packed):
    args = _args()
    cfg = SimpleNamespace(tokenizer=SimpleNamespace(padded_vocab_size=32))
    kwargs = (
        {"total_real_tokens_in_batch": 16, "seqlen_squared_sum_in_batch": 256} if packed else {}
    )
    # One 16-token doc. CSA: 1904*16 + 96*(45+22) + 8*(28+2*22) = 37472.
    # Dense SwiGLU: 2304/token; logits: 1536/token. Old MLA path fails this value.
    expected = 37472 + (2304 + 1536) * 16
    assert num_floating_point_operations(args, cfg, batch_size=1, **kwargs) == expected
    if packed:
        # Increasing physical capacity must not add padding to the numerator.
        args.seq_length *= 8
        assert num_floating_point_operations(args, cfg, batch_size=1, **kwargs) == expected
