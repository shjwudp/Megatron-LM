# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real packed CP attention: bit-exact O/LSE restore, nested hooks and gradient parity."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from megatron.core import tensor_parallel
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.csa_utils.output_recompute import (
    CSAOutputRecompute,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _build_attention,
    _make_config,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_packed_cp import (
    _assert_match,
)


@pytest.mark.experimental
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("grouped_recompute", [False, True])
def test_csa_output_restore_and_cp_gradients(precision, grouped_recompute):
    """Compare the real CP2 path, including empty/padded documents and indexer loss."""
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(198)
        model_parallel_cuda_manual_seed(198)
        modules = ["mla_up_proj", "layernorm"]
        if grouped_recompute:
            modules.append("csa_o_proj")
        cfg = _make_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_compress_ratios=[4],
            csa_window_size=128,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=512,
            dsa_indexer_loss_coeff=0.2,
            dsa_indexer_use_sparse_loss=True,
            dsa_kernel_backend="cudnn",
            context_parallel_size=2,
            attention_cp_layout="contiguous",
            linear_cp_layout="contiguous",
            qk_layernorm=True,
            apply_rope_fusion=True,
            gradient_accumulation_fusion=False,
            recompute_granularity="selective",
            recompute_modules=modules,
            fp8="e4m3" if precision == "mxfp8" else None,
            fp8_recipe="mxfp8",
        )
        reference = _build_attention(cfg, 1, pg).cuda()
        candidate = _build_attention(
            replace(cfg, recompute_modules=modules + ["csa_core_output"]), 1, pg
        ).cuda()
        candidate.load_state_dict(reference.state_dict())
        physical = torch.tensor([0, 133, 133, 1157, 2048], dtype=torch.int32, device="cuda")
        real = torch.tensor([0, 129, 129, 1141, 2016], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=physical,
            cu_seqlens_kv_padded=physical,
            max_seqlen_q=1024,
            max_seqlen_kv=1024,
        )
        source = torch.randn(2048, 1, 256, device="cuda", dtype=torch.bfloat16)
        upstream = torch.randn_like(source)
        rows = slice(pg.cp.rank() * 1024, (pg.cp.rank() + 1) * 1024)
        active = []
        restored = []
        original_record = CSAOutputRecompute.record
        original_rebuild = CSAOutputRecompute.rebuild

        def record(state, *args):
            original_record(state, *args)
            state.expected_for_test = tuple(t.clone() for t in state._outputs)

        def rebuild(state):
            if state._inputs is None:
                return original_rebuild(state)
            aliases = state._outputs
            assert all(t.untyped_storage().nbytes() == 0 for t in aliases)
            active.append(True)
            try:
                original_rebuild(state)
            finally:
                active.pop()
            for actual, expected in zip(aliases, state.expected_for_test):
                assert not torch.isnan(actual).any()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            restored.append(True)
            del state.expected_for_test

        def run(model):
            calls = Counter()

            def wrap(name, fn):
                def invoke(*args, **kwargs):
                    assert not active, f"collective {name} inside O/LSE replay"
                    calls[name] += 1
                    return fn(*args, **kwargs)

                return invoke

            if model.config.fp8:
                model.set_for_recompute_input_layernorm()
            x = source[rows].detach().clone().requires_grad_()
            with ExitStack() as stack:
                for name in ("dist_all_gather_func", "dist_reduce_scatter_func"):
                    stack.enter_context(
                        patch.object(mappings, name, wrap(name, getattr(mappings, name)))
                    )
                for name in (
                    "all_gather_into_tensor",
                    "_all_gather_base",
                    "all_gather",
                    "reduce_scatter_tensor",
                    "_reduce_scatter_base",
                    "reduce_scatter",
                    "all_reduce",
                    "batch_isend_irecv",
                ):
                    stack.enter_context(patch.object(dist, name, wrap(name, getattr(dist, name))))
                with get_fp8_context(model.config):
                    norm = tensor_parallel.CheckpointWithoutOutput(fp8=model.config.fp8)
                    normalized = norm.checkpoint(
                        lambda t: torch.nn.functional.rms_norm(t, (256,)), x
                    )
                    output, _ = model(normalized, attention_mask=None, packed_seq_params=packed)
                    norm.discard_output_and_register_recompute(output)
                    output.backward(upstream[rows])
            result = {"output": output.detach().clone(), "input": x.grad.detach().clone()}
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                result[name] = param.grad.detach().clone()
            return result, calls

        with (
            patch.object(CSAOutputRecompute, "record", record),
            patch.object(CSAOutputRecompute, "rebuild", rebuild),
        ):
            expected, before = run(reference)
            actual, after = run(candidate)
        assert restored == [True]
        assert before == after, (before, after)
        assert sum(before.values()) > 0
        torch.testing.assert_close(actual["output"], expected["output"], rtol=0, atol=0)
        for name in actual:
            # cuDNN sparse-attention backward is not qualified as bit-exact;
            # use the existing CP direction-and-magnitude gates for all grads.
            _assert_match(actual[name], expected[name])
        with torch.no_grad(), get_fp8_context(candidate.config):
            output, _ = candidate(source[rows], attention_mask=None, packed_seq_params=packed)
            assert torch.isfinite(output).all()
    finally:
        Utils.destroy_model_parallel()
        torch.compiler.reset()
