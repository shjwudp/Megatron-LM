# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verify hook ordering for Transformer Engine delayed weight gradients."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def _grad_snapshot(parameter: torch.nn.Parameter) -> dict[str, object]:
    """Return a printable summary without retaining the gradient tensor."""
    if parameter.grad is None:
        return {"is_none": True, "norm": None, "nonzero": None}
    grad = parameter.grad.detach().float()
    return {
        "is_none": False,
        "norm": float(torch.linalg.vector_norm(grad)),
        "nonzero": int(torch.count_nonzero(grad)),
    }


@pytest.mark.skipif(
    not is_te_min_version("2.7.0"),
    reason="Delayed wgrad without gradient-accumulation fusion requires TE 2.7 or newer.",
)
def test_post_accumulate_hook_does_not_wait_for_backward_dw(distributed_setup):
    """Distinguish PyTorch's autograd hook from TE's explicit delayed-wgrad hook."""
    if distributed_setup.device.type != "cuda":
        pytest.skip("Transformer Engine delayed wgrad requires CUDA.")

    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
        )
        # delay_wgrad_compute is normally enabled together with the full MoE-overlap
        # configuration. This focused TE Linear test does not construct an MoE layer,
        # so enable it after TransformerConfig validates the unrelated MoE settings.
        config.delay_wgrad_compute = True
        linear = TELinear(
            config.hidden_size,
            config.hidden_size,
            parallel_mode="duplicated",
            config=config,
            init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02),
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        weight = linear.weight
        phase = {"name": "setup"}
        post_accumulate_events = []
        delayed_wgrad_events = []

        def record_post_accumulate(parameter):
            post_accumulate_events.append(
                {"phase": phase["name"], "grad": _grad_snapshot(parameter)}
            )

        def record_delayed_wgrad(*_unused):
            delayed_wgrad_events.append(
                {"phase": phase["name"], "grad": _grad_snapshot(weight)}
            )

        weight.register_post_accumulate_grad_hook(record_post_accumulate)
        linear.register_wgrad_accumulation_and_reduce_hooks(record_delayed_wgrad)

        input_tensor = torch.randn(
            8,
            config.hidden_size,
            device=distributed_setup.device,
            dtype=config.params_dtype,
            requires_grad=True,
        )
        output, _ = linear(input_tensor)

        phase["name"] = "activation_backward"
        output.float().square().sum().backward()
        after_activation_backward = _grad_snapshot(weight)

        phase["name"] = "backward_dw"
        linear.backward_dw()
        torch.cuda.synchronize()
        after_backward_dw = _grad_snapshot(weight)

        result = {
            "skip_backward_post_hook": getattr(weight, "skip_backward_post_hook", None),
            "post_accumulate_events": post_accumulate_events,
            "delayed_wgrad_events": delayed_wgrad_events,
            "after_activation_backward": after_activation_backward,
            "after_backward_dw": after_backward_dw,
        }
        print(f"DELAY_WGRAD_HOOK_ORDER={result}", flush=True)

        assert [event["phase"] for event in post_accumulate_events] == [
            "activation_backward"
        ]
        assert [event["phase"] for event in delayed_wgrad_events] == ["backward_dw"]
        assert post_accumulate_events[0]["grad"]["is_none"]
        assert after_activation_backward["is_none"]
        assert not delayed_wgrad_events[0]["grad"]["is_none"]
        assert delayed_wgrad_events[0]["grad"]["nonzero"] > 0
        assert not after_backward_dw["is_none"]
        assert after_backward_dw["nonzero"] > 0
    finally:
        Utils.destroy_model_parallel()
