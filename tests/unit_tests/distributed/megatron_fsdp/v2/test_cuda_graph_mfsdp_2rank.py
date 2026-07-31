# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-rank integration test for M-FSDP activation-recompute graphs."""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard


@pytest.fixture(scope="module", autouse=True)
def distributed_environment():
    """Initialize the process group used by this integration test."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class _LinearBlock(nn.Module):
    """One linear layer used by the integration test."""

    def __init__(self):
        """Create a four-element linear layer."""
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, value):
        """Apply the linear layer."""
        return self.linear(value)


class _CheckpointedModel(nn.Module):
    """Checkpoint the same module selected for M-FSDP CUDA Graphs."""

    def __init__(self):
        """Create one non-reentrant checkpointed block."""
        super().__init__()
        self.graphed = _LinearBlock()

    def forward(self, value):
        """Run the non-reentrant checkpoint."""
        return checkpoint(self.graphed, value, use_reentrant=False)


def test_mfsdp_activation_recompute_updates_optimizer_parameters(request):
    """Capture three graphs and update parameters held by a prebuilt optimizer."""
    if torch.distributed.get_world_size() != 2:
        pytest.skip("This integration test requires two ranks")
    device = torch.device(f"cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}")
    model = _CheckpointedModel().to(device)
    with torch.no_grad():
        model.graphed.linear.weight.fill_(1)
        model.graphed.linear.bias.zero_()
    fully_shard(
        model.graphed,
        sharding_strategy="optim_grads_params",
        enable_unshard_prefetch=True,
        enable_async_reduce_grad=True,
        enable_cuda_graph=True,
        cuda_graph_activation_recompute=True,
    )
    fully_shard(model, sharding_strategy="optim_grads_params")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer_parameter_ids = tuple(id(parameter) for parameter in model.parameters())

    def release_graph_pool():
        """Release captured graphs after all outputs are gone."""
        model.release_memory_pool()
        torch.cuda.synchronize()
        torch.distributed.barrier()

    request.addfinalizer(release_graph_pool)
    for step, value in enumerate((1.0, 2.0, 3.0)):
        optimizer.zero_grad(set_to_none=True)
        sample = torch.full((2, 4), value, device=device, requires_grad=True)
        before = tuple(parameter.to_local().detach().clone() for parameter in model.parameters())
        model(sample).sum().backward()
        model.finish_grad_sync()
        optimizer.step()
        torch.cuda.synchronize()

        assert tuple(id(parameter) for parameter in model.parameters()) == optimizer_parameter_ids
        assert any(
            not torch.equal(parameter.to_local(), previous)
            for parameter, previous in zip(model.parameters(), before)
        )
        assert all(
            param_group.model_weight_buffer.placements
            == param_group.model_weight_buffer.storage_placements
            for param_group in model.graphed._fsdp_param_groups
        )
        assert all(
            not param_group.model_weight_buffer.is_unsharded()
            for param_group in model.graphed._fsdp_param_groups
        )
        if step == 2:
            assert model.graphed._fsdp_cg_installed
            assert model.graphed._fsdp_root_context.cuda_graph_activation_recompute

    context = model._fsdp_root_context
    context.backward_phase = True
    context.backward_module = None
    try:
        with pytest.raises(RuntimeError, match="outside checkpoint recomputation"):
            model.graphed(torch.full((2, 4), 11.0, device=device, requires_grad=True))
    finally:
        context.backward_phase = False
