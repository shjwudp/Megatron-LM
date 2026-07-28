# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the experimental Megatron-FSDP toy-model example."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from examples.megatron_fsdp.train_toy_model_experimental import (
    ToyModel,
    build_optimizer,
    fully_shard_toy_model,
    parse_args,
    toy_placements,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import Flat, Partial, Replicate


def test_toy_example_arguments_default_to_megatron_fsdp():
    """The toy example should select current experimental FSDP by default."""
    args = parse_args(["--model-dim", "16", "--use-real-data"])

    assert args.backend == "mfsdpv2"
    assert args.model_dim == 16
    assert args.use_real_data


def test_toy_hsdp_placements_shard_outer_optimizer(distributed_setup):
    """The 2D layout should match the outer-optimizer-sharded HSDP design."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2:
        pytest.skip("This test requires an even world size of at least four.")
    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("dp_outer", "dp_inner"),
    )
    placements = toy_placements(mesh)

    assert placements.parameter == [Replicate(), Flat()]
    assert placements.gradient == [Partial(torch.distributed.ReduceOp.AVG), Flat()]
    assert placements.optimizer == [Flat(), Flat()]


@pytest.mark.parametrize("backend", ["mfsdpv2", "fsdp2"])
def test_toy_example_backend_updates_weights(distributed_setup, backend):
    """Both toy backends should complete a sharded optimizer update."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,), mesh_dim_names=("dp",))
    torch.manual_seed(1234)
    model = ToyModel(dim=8, num_layers=2).to(device=device, dtype=torch.bfloat16)
    model = fully_shard_toy_model(model, mesh=mesh, backend=backend)
    optimizer = build_optimizer(model, backend=backend, learning_rate=0.01)
    inputs = torch.randn(2, 4, 8, device=device, dtype=torch.bfloat16)

    first_output = model(inputs).detach()
    optimizer.zero_grad(set_to_none=True)
    model(inputs).float().square().mean().backward()
    optimizer.step()
    second_output = model(inputs).detach()

    assert not torch.equal(first_output, second_output)
