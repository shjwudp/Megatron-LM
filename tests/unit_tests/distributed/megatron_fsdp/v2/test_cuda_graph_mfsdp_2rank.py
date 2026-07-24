# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-rank integration test for M-FSDP activation-recompute graphs."""

from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph import (
    wrap_cuda_graph_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def distributed_environment():
    """Initialize the two-rank NCCL test environment."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    yield


class _LinearBlock(nn.Module):
    """One linear layer used by the integration test."""

    def __init__(self):
        """Create a four-element linear layer."""
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, value):
        """Apply the linear layer.

        :param value: Input tensor.
        :type value: torch.Tensor
        :return: Linear output.
        :rtype: torch.Tensor
        """
        return self.linear(value)


def test_mfsdp_activation_recompute_updates_optimizer_parameters(request):
    """Capture three graphs and update parameters held by a prebuilt optimizer.

    :param request: Pytest request used for graph-pool cleanup.
    :type request: pytest.FixtureRequest
    """
    if torch.distributed.get_world_size() != 2:
        pytest.skip("This integration test requires two ranks")
    device = torch.device(f"cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}")
    model = _LinearBlock().to(device)
    with torch.no_grad():
        model.linear.weight.fill_(1)
        model.linear.bias.zero_()
    fully_shard(
        model,
        sharding_strategy="optim_grads_params",
        enable_unshard_prefetch=True,
        enable_async_reduce_grad=True,
        enable_cuda_graph=True,
        cuda_graph_activation_recompute=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer_parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    checkpoint_fn = wrap_cuda_graph_checkpoint(partial(checkpoint, use_reentrant=False))

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
        checkpoint_fn(model, sample).sum().backward()
        model.finish_grad_sync()
        optimizer.step()
        torch.cuda.synchronize()

        assert tuple(id(parameter) for parameter in model.parameters()) == optimizer_parameter_ids
        assert any(
            not torch.equal(parameter.to_local(), previous)
            for parameter, previous in zip(model.parameters(), before)
        )
        assert all(
            param_group.model_weight_buffer.storage_shard_layout != (0, 0)
            for param_group in model._fsdp_param_groups
        )
        if step >= 1:
            assert model._fsdp_cg_installed
            assert model._fsdp_cg_activation_recompute

    abandoned = checkpoint_fn(model, torch.full((2, 4), 4.0, device=device, requires_grad=True))
    del abandoned
    assert model.release_pending()
    retry = torch.full((2, 4), 5.0, device=device, requires_grad=True)
    checkpoint_fn(model, retry).sum().backward()
    model.finish_grad_sync()
    assert retry.grad is not None
