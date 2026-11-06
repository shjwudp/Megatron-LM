# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed CSA/MTP parity across real CP and EP groups and interleaved microbatches."""

import gc
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from torch.distributed.tensor import DTensor

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    RuntimePolicy,
    fully_shard_context,
)
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import get_fsdp_root
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _make_config,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_packed_cp import (
    _assert_match,
)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """All inputs are generated locally; no dataset download is needed."""


def _model(config):
    spec = get_transformer_block_with_experimental_attention_variant_spec(config)
    mtp = get_gpt_mtp_block_spec(config, spec, True)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        mtp_block_spec=mtp,
        vocab_size=256,
        max_sequence_length=2048,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=False,
    ).cuda()
    # Match training's fused FP32 accumulation. TE's non-fused delayed-wgrad
    # API returns one microbatch in .grad; it does not accumulate microbatches.
    if config.gradient_accumulation_fusion:
        for parameter in model.parameters():
            parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
    return model


def _gradient(parameter):
    # Fused linear wgrads accumulate into main_grad. Other operators (including
    # embeddings and normalization) return ordinary autograd gradients.
    if isinstance(parameter, DTensor):
        return None if parameter.grad is None else parameter.grad.to_local()
    if parameter.grad is None:
        return parameter.main_grad
    return parameter.main_grad + parameter.grad.float()


def _wrap(model, pg):
    model = Float16Module(model.config, model)
    # Training constructs VPP chunks inside an ambient context; hook registration
    # must also work before that context is finalized.
    context = FsdpContext(
        device=torch.device("cuda", torch.cuda.current_device()),
        runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
    )
    with fully_shard_context(context):
        return FullyShardedDataParallel(
            config=model.config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                data_parallel_sharding_strategy='optim_grads',
                outer_dp_sharding_strategy='optim',
                expert_data_parallel_sharding_strategy='optim',
                expert_outer_dp_sharding_strategy='optim',
                num_distributed_optimizer_instances=2,
                megatron_fsdp_main_grads_dtype=torch.float32,
                fsdp_all_gather_in_start_param_sync=False,
            ),
            module=model,
            pg_collection=pg,
        )


def _batches(pg):
    # A document crosses the CP split; an empty document and physical padding
    # exercise boundaries that a single fixed-length example cannot cover.
    physical = torch.tensor([0, 133, 133, 1157, 2048], device="cuda", dtype=torch.int32)
    real = torch.tensor([0, 129, 129, 1141, 2016], device="cuda", dtype=torch.int32)
    rows = torch.arange(2048, device="cuda")
    document = torch.bucketize(rows, physical[1:], right=True)
    positions = rows - physical[document]
    valid = positions < (real[1:] - real[:-1])[document]
    local = slice(pg.cp.rank() * 1024, (pg.cp.rank() + 1) * 1024)
    batches = []
    for index in range(3):
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=real,
            cu_seqlens_kv=real,
            cu_seqlens_q_padded=physical,
            cu_seqlens_kv_padded=physical,
            max_seqlen_q=1024,
            max_seqlen_kv=1024,
        )
        batches.append(
            dict(
                input_ids=((rows * 7 + index * 19) % 255 + 1)[local].unsqueeze(0),
                position_ids=positions[local].unsqueeze(0),
                labels=((rows * 7 + index * 19 + 7) % 255 + 1)[local].unsqueeze(0),
                loss_mask=valid[local].float().unsqueeze(0),
                attention_mask=None,
                packed_seq_params=packed,
            )
        )
    return batches


@pytest.mark.experimental
@pytest.mark.parametrize("fsdp", [False, True], ids=["unsharded", "hfsdp"])
@pytest.mark.parametrize("delay_wgrad", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_packed_csa_mtp_outputs_and_all_parameter_gradients(
    delay_wgrad, recompute, fsdp, monkeypatch
):
    """Compare ordinary execution with a genuine F(n+1)/B(n) schedule, including embeddings."""
    if Utils.world_size < 4:
        pytest.skip("requires at least four ranks for CP2 and EP2")
    Utils.initialize_model_parallel(
        context_parallel_size=2,
        expert_model_parallel_size=2,
        num_distributed_optimizer_instances=2 if fsdp else 1,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(198)
        model_parallel_cuda_manual_seed(198)
        config = _make_config(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=64,
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            csa_compress_ratios=[4, 4],
            csa_window_size=128,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=512,
            dsa_indexer_loss_coeff=0.01,
            dsa_indexer_use_sparse_loss=True,
            dsa_kernel_backend="cudnn",
            context_parallel_size=2,
            attention_cp_layout="contiguous",
            linear_cp_layout="contiguous",
            expert_model_parallel_size=2,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
            moe_shared_expert_intermediate_size=256,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=256,
            normalization="RMSNorm",
            qk_layernorm=True,
            apply_rope_fusion=True,
            gradient_accumulation_fusion=not fsdp,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.3,
            calculate_per_token_loss=True,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=(
                ["mla_up_proj", "moe_act", "layernorm", "shared_experts"] if recompute else []
            ),
        )
        reference = _model(config)
        candidate = _model(
            replace(config, overlap_moe_expert_parallel_comm=True, delay_wgrad_compute=delay_wgrad)
        )
        candidate.load_state_dict(reference.state_dict())
        if fsdp:
            reference, candidate = _wrap(reference, pg), _wrap(candidate, pg)
        batches = _batches(pg)
        MTPLossAutoScaler.set_loss_scale(torch.ones(1, device="cuda"))
        mtp_metrics = []
        save_metrics = MTPLossLoggingHelper.save_metrics_to_tracker

        def capture_mtp_metrics(loss, correct, total, *args, **kwargs):
            mtp_metrics.append(tuple(value.detach().clone() for value in (loss, correct, total)))
            save_metrics(loss, correct, total, *args, **kwargs)

        monkeypatch.setattr(MTPLossLoggingHelper, 'save_metrics_to_tracker', capture_mtp_metrics)
        expected = []
        for index, data in enumerate(batches):
            context = reference.no_sync() if fsdp and index < len(batches) - 1 else nullcontext()
            with context:
                loss = reference(**data)
                expected.append(loss.detach().clone())
                loss.backward(data["loss_mask"])
        if fsdp:
            reference.finish_grad_sync()
        set_streams()
        candidate_model = candidate.module.module if fsdp else candidate
        plans = [candidate_model.build_schedule_plan(**data) for data in batches]
        for index, plan in enumerate(plans):
            plan._model_chunk_state.fsdp_root = get_fsdp_root(candidate)
        outputs = []
        with candidate.no_sync() if fsdp else nullcontext():
            outputs.append(TransformerModelChunkSchedulePlan.run(plans[0], None))
            for index in range(1, len(plans)):
                outputs.append(
                    TransformerModelChunkSchedulePlan.run(
                        plans[index], plans[index - 1], b_grad=batches[index - 1]["loss_mask"]
                    )
                )
        TransformerModelChunkSchedulePlan.run(None, plans[-1], b_grad=batches[-1]["loss_mask"])
        if fsdp:
            candidate.finish_grad_sync()
        for actual, wanted in zip(outputs, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert len(mtp_metrics) == 2 * len(batches)
        for wanted, actual in zip(mtp_metrics[: len(batches)], mtp_metrics[len(batches) :]):
            for expected_metric, actual_metric in zip(wanted, actual):
                torch.testing.assert_close(actual_metric, expected_metric, rtol=0, atol=0)
        checked = 0
        for (name, param), (other_name, other) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            assert name == other_name
            wanted_grad, actual_grad = _gradient(param), _gradient(other)
            assert (wanted_grad is None) == (actual_grad is None), name
            if wanted_grad is not None:
                try:
                    _assert_match(actual_grad, wanted_grad)
                except AssertionError as error:
                    raise AssertionError(f"{name}: {error}") from error
                checked += 1
        assert checked > 30
        del reference, candidate, plans, outputs
    finally:
        torch.cuda.synchronize()
        fused_a2a._buffer = None
        gc.collect()
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()
