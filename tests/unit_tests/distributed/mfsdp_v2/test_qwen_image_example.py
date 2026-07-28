# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the experimental Megatron-FSDP QwenImage example."""

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from examples.megatron_fsdp.train_qwen_image_experimental import (
    _format_startup_log,
    _format_step_log,
    flat_dp_placements,
    fully_shard_qwen_image_transformer,
    fully_shard_qwen_image_transformer_fsdp1,
    make_flow_matching_batch,
    parse_args,
    qwen25vl_vision_tokens,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import fully_shard_optimizer


class TinyQwenImageTransformer(nn.Module):
    """Small QwenImage-shaped module for testing bottom-up FSDP application."""

    def __init__(self) -> None:
        super().__init__()
        self.img_in = nn.Linear(8, 8)
        self.transformer_blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(8, 8), nn.SiLU()) for _ in range(2)]
        )
        self.proj_out = nn.Linear(8, 8)
        self.config = type("Config", (), {"in_channels": 8, "joint_attention_dim": 12})()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the tiny transformer."""
        hidden_states = self.img_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return self.proj_out(hidden_states)


def test_qwen_image_original_benchmark_arguments_are_compatible():
    """The mfsdp_refactor QwenImage command line should remain accepted."""
    args = parse_args(
        [
            "--sharding",
            "full",
            "--cuda_profiler_capture",
            "--pretrained_model_name_or_path",
            "/tmp/qwen-image",
            "--num_gpus_per_node",
            "4",
            "--batch_size",
            "4",
            "--height",
            "512",
            "--width",
            "512",
            "--attention",
            "flash",
            "--bench_steps",
            "20",
            "--warmup_steps",
            "3",
            "--compile",
            "--real-data",
        ]
    )

    assert args.backend == "mfsdpv2"
    assert args.sharding == "full"
    assert args.model_id == "/tmp/qwen-image"
    assert args.num_gpus_per_node == 4
    assert args.batch_size == 4
    assert args.benchmark_steps == 20
    assert args.warmup_steps == 3
    assert args.cuda_profiler_capture
    assert args.compile
    assert args.check_convergence


def test_qwen_image_logs_match_original_benchmark_format():
    """Stable benchmark lines should remain compatible with the original harness."""
    args = parse_args(
        [
            "--backend",
            "mfsdpv2",
            "--batch_size",
            "4",
            "--height",
            "512",
            "--width",
            "512",
            "--compile",
            "--real-data",
        ]
    )

    assert _format_startup_log(args, world_size=4, text_sequence_length=388) == (
        "[mfsdpv2] world=4 dtype=torch.bfloat16 bs=4 img=512x512 txt=388 "
        "sharding=full compile=True gc=False"
    )
    assert (
        _format_step_log(args, step=3, elapsed=0.38553, loss=0.12345, verification=None)
        == "[mfsdpv2] bench  step   3 |   385.53 ms | loss=1.2345e-01"
    )
    assert _format_step_log(
        args, step=0, elapsed=0.5, loss=0.1234567, verification=(1.25, 60, 2.5, -3.0, 60)
    ) == (
        "[mfsdpv2] warmup step   0 | VERIFY (timing invalid) | "
        "gloss=0.123457 | gnorm=1.2500 | n_grad=60 | "
        "pnorm=2.50000000e+00 | psum=-3.00000000e+00 | n_param=60"
    )


def test_qwen_image_helper_shards_blocks_bottom_up(distributed_setup):
    """The QwenImage root and every repeated transformer block should be FSDP units."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,), mesh_dim_names=("dp",))
    model = TinyQwenImageTransformer().to(device)
    fully_shard_qwen_image_transformer(model, mesh=mesh, placements=flat_dp_placements())

    assert hasattr(model, "parameter_groups")
    assert all(hasattr(block, "parameter_groups") for block in model.transformer_blocks)

    model_input = torch.randn(2, 8, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    fully_shard_optimizer(optimizer)
    first_output = model(model_input).detach()
    optimizer.zero_grad(set_to_none=True)
    loss = model(model_input).square().mean()
    loss.backward()
    optimizer.step()

    second_output = model(model_input).detach()
    assert not torch.equal(first_output, second_output)


def test_qwen_image_flow_matching_batch_matches_model_contract(distributed_setup):
    """Synthetic data should use packed image and configured prompt dimensions."""
    device = distributed_setup.device
    model = TinyQwenImageTransformer().to(device)
    generator = torch.Generator(device=device).manual_seed(1234)
    batch = make_flow_matching_batch(
        model,
        batch_size=2,
        height=64,
        width=32,
        text_sequence_length=7,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    assert batch.model_inputs["hidden_states"].shape == (2, 8, 8)
    assert batch.model_inputs["encoder_hidden_states"].shape == (2, 7, 12)
    assert batch.model_inputs["timestep"].shape == (2,)
    assert batch.model_inputs["img_shapes"] == [(1, 4, 2), (1, 4, 2)]
    assert batch.target.shape == (2, 8, 8)
    assert qwen25vl_vision_tokens(512, 512, patch_size=14, merge_size=2) == 324


def test_qwen_image_fsdp1_helper_wraps_blocks_and_updates_weights(distributed_setup):
    """The FSDP1 reference path should run the same basic training operation."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,), mesh_dim_names=("dp",))
    model = TinyQwenImageTransformer().to(device)
    model = fully_shard_qwen_image_transformer_fsdp1(model, mesh=mesh, device=device)

    model_input = torch.randn(2, 8, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    first_output = model(model_input).detach()
    optimizer.zero_grad(set_to_none=True)
    model(model_input).square().mean().backward()
    optimizer.step()

    second_output = model(model_input).detach()
    assert not torch.equal(first_output, second_output)
