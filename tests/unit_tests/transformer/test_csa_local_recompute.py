# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Local FC1 replay must preserve outputs and all gradients, including upstream norms."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    set_save_original_input,
)
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridAttention,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(kind, precision, enabled):
    modules = ["moe_act"] if kind == "moe" else []
    if enabled:
        modules.append("moe_fc1" if kind == "moe" else "dense_fc1_act")
    return TransformerConfig(
        num_layers=1,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        num_moe_experts=4 if kind == "moe" else None,
        moe_ffn_hidden_size=512 if kind == "moe" else None,
        moe_grouped_gemm=True,
        moe_router_topk=2,
        expert_tensor_parallel_size=1,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=quick_gelu,
        activation_func_clamp_value=7.0,
        glu_linear_offset=1.0,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8="e4m3" if precision == "mxfp8" else None,
        fp8_recipe="mxfp8",
        recompute_granularity="selective",
        recompute_modules=modules,
    )


def _model(config, kind):
    model_parallel_cuda_manual_seed(1234)
    if kind == "dense":
        return MLP(
            config,
            MLPSubmodules(TEColumnParallelLinear, TERowParallelLinear),
            ffn_hidden_size=config.ffn_hidden_size,
        ).cuda()
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(4, moe_grouped_gemm=True).mlp
    )
    return MoELayer(config, submodules).experts.cuda()


def _run(model, source, kind, grad_enabled=True):
    model.zero_grad(set_to_none=True)
    x = source.detach().clone().requires_grad_(True)
    probs = torch.linspace(0.1, 0.9, x.shape[0], device="cuda").requires_grad_(True)
    calls = []
    handle = model.linear_fc1.register_forward_hook(lambda *args: calls.append(1))
    with torch.set_grad_enabled(grad_enabled), get_fp8_context(model.config):
        if kind == "dense":
            # Reproduce TransformerLayer's outer, output-discarding norm checkpoint.
            norm = tensor_parallel.CheckpointWithoutOutput(fp8=model.config.fp8)
            normalized = norm.checkpoint(lambda t: torch.nn.functional.rms_norm(t, (256,)), x)
            output, _ = model(normalized)
            norm.discard_output_and_register_recompute(output)
        else:
            # Includes an empty expert and uneven, quantization-aligned token counts.
            counts = torch.tensor([32, 0, 64, 32], device="cpu")
            output, _ = model(x, counts, probs)
        value = output.detach().clone()
        if grad_enabled:
            output.float().square().mean().backward()
    handle.remove()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    if grad_enabled:
        grads["input"] = x.grad.detach().clone()
        if kind == "moe":
            grads["router_probs"] = probs.grad.detach().clone()
    return value, grads, len(calls)


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_local_fc1_recompute_numerics(kind, precision):
    """Exercise actual TE linears and verify replay executes, including norm ordering."""
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(91)
        source = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        baseline = _model(_config(kind, precision, False), kind)
        recomputed = _model(_config(kind, precision, True), kind)
        recomputed.load_state_dict(baseline.state_dict())
        expected, expected_grads, expected_calls = _run(baseline, source, kind)
        actual, actual_grads, actual_calls = _run(recomputed, source, kind)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual_grads.keys() == expected_grads.keys()
        for name in actual_grads:
            torch.testing.assert_close(
                actual_grads[name], expected_grads[name], rtol=0, atol=0, msg=name
            )
        assert expected_calls == 1
        assert actual_calls == 2
        with torch.no_grad(), get_fp8_context(recomputed.config):
            if kind == "dense":
                value, _ = recomputed(source)
            else:
                value, _ = recomputed(
                    source, torch.tensor([32, 0, 64, 32]), torch.ones(128, device="cuda")
                )
            assert torch.isfinite(value).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_csa_o_projection_recompute_numerics(precision):
    """Use the real grouped projection and TE consumer, checking input and weight gradients."""
    Utils.initialize_model_parallel(1, 1)
    try:
        config = _config("dense", precision, False)
        model_parallel_cuda_manual_seed(1234)
        projection = TERowParallelLinear(
            256,
            256,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        ).cuda()
        set_save_original_input(projection)
        torch.manual_seed(1234)
        weight = torch.nn.Parameter(torch.randn(4, 64, 128, device="cuda", dtype=torch.bfloat16))
        owner = SimpleNamespace(
            config=SimpleNamespace(output_projection_lora_rank=64),
            output_projection_local_groups=4,
            linear_o_group_proj=weight,
        )
        source = torch.randn(128, 1, 512, device="cuda", dtype=torch.bfloat16)
        results = []
        for enabled in (False, True):
            projection.zero_grad(set_to_none=True)
            weight.grad = None
            x = source.detach().clone().requires_grad_(True)
            calls = []

            def grouped(tensor):
                calls.append(1)
                return DSv4HybridAttention._grouped_output_projection(owner, tensor)

            with get_fp8_context(config):
                checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=None)
                intermediate = checkpoint.checkpoint(grouped, x) if enabled else grouped(x)
                output, _ = projection(intermediate)
                if enabled:
                    checkpoint.discard_output_and_register_recompute(output)
                value = output.detach().clone()
                output.float().square().mean().backward()
            results.append(
                (value, x.grad.clone(), weight.grad.clone(), projection.weight.grad.clone())
            )
            assert len(calls) == (2 if enabled else 1)
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "modules,overrides,message",
    [
        (["moe_fc1"], {}, "requires TE grouped"),
        (["moe_fc1", "moe_act", "moe"], {}, "requires TE grouped"),
        (["dense_fc1_act", "mlp"], {}, "cannot be combined"),
        (["dense_fc1_act"], {"tensor_model_parallel_size": 2}, "requires TP1"),
        (["csa_o_proj"], {}, "requires dsv4_hybrid"),
        (["dense_fc1_act"], {"fp8": "e4m3", "fp8_recipe": "delayed"}, "BF16 or MXFP8"),
    ],
)
def test_local_recompute_rejects_unsupported_combinations(modules, overrides, message):
    with pytest.raises(ValueError, match=message):
        TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            moe_grouped_gemm=True,
            add_bias_linear=False,
            recompute_granularity="selective",
            recompute_modules=modules,
            **overrides,
        )
