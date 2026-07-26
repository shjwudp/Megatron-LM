# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the placement-first ParameterGroupV2 prototype."""

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import TemporaryBucketAllocator
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.buffer_index import Placement
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group_v2 import (
    ParameterGroupLayoutV2,
    ParameterGroupV2,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.utils import ParamGroupIdx


@pytest.fixture(scope="session", autouse=True)
def dist_env():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _device() -> torch.device:
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def _hsdp_mesh() -> DeviceMesh:
    world_size = torch.distributed.get_world_size()
    if world_size < 4 or world_size % 2:
        pytest.skip("ParameterGroupV2 HSDP tests require an even world size >= 4")
    ranks = torch.arange(world_size, dtype=torch.int).reshape(2, world_size // 2)
    return DeviceMesh(_device().type, ranks, mesh_dim_names=("dp_outer", "dp"))


def _build_group(
    *,
    shard_optimizer_across_outer_dp: bool,
    grad_comm_dtype: torch.dtype | None = None,
    gradient_scaling_factor: float | None = None,
) -> tuple[ParameterGroupV2, torch.Tensor, TemporaryBucketAllocator]:
    values = torch.arange(128, dtype=torch.float32, device=_device())
    param = nn.Parameter(values.clone())
    allocator = TemporaryBucketAllocator()
    group = ParameterGroupV2(
        [param],
        ParamGroupIdx(0, 0),
        mesh=_hsdp_mesh(),
        layout=ParameterGroupLayoutV2.hsdp(
            shard_optimizer_across_outer_dp=shard_optimizer_across_outer_dp
        ),
        mp_policy=MixedPrecisionPolicy(
            main_grads_dtype=torch.float32, grad_comm_dtype=grad_comm_dtype
        ),
        allocator=allocator,
        gradient_scaling_factor=gradient_scaling_factor,
    )
    return group, values, allocator


def test_weight_validity_and_scratch_lifecycle():
    group, original, allocator = _build_group(shard_optimizer_across_outer_dp=True)

    assert group.state.weight_valid == (Placement.REPLICATE, Placement.SHARD)
    assert group.compute_weight() is None

    compute_weight = group.unshard_weight()
    assert compute_weight.placements == [Placement.REPLICATE, Placement.REPLICATE]
    torch.testing.assert_close(group.params[0], original)
    assert group.state.full_weight is compute_weight

    group.reshard_weight()
    assert group.state.full_weight is None
    assert allocator.buckets == {}

    group.main_weight_buffer.data.add_(torch.distributed.get_rank() + 1)
    group.refresh_model_weight()
    assert group.state.weight_valid == (Placement.SHARD, Placement.SHARD)

    refreshed = group.unshard_weight()
    replicas = [torch.empty_like(refreshed.data) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(replicas, refreshed.data)
    assert all(torch.equal(replicas[0], replica) for replica in replicas[1:])
    assert not torch.equal(refreshed.data, original)

    group.reshard_weight()
    assert allocator.buckets == {}


@pytest.mark.parametrize("shard_optimizer_across_outer_dp", [False, True])
@pytest.mark.parametrize("grad_comm_dtype", [None, torch.bfloat16])
def test_two_microbatch_hsdp_gradient(shard_optimizer_across_outer_dp, grad_comm_dtype):
    group, _, allocator = _build_group(
        shard_optimizer_across_outer_dp=shard_optimizer_across_outer_dp,
        grad_comm_dtype=grad_comm_dtype,
        gradient_scaling_factor=0.5,
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    group.begin_backward().data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert group.state.grad_valid == (Placement.PARTIAL, Placement.SHARD)
    assert not group.state.grad_ready
    assert group.state.full_grad is None
    assert allocator.buckets == {}

    group.begin_backward().data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)

    expected = 0.5 * world_size * (world_size + 2)
    optimizer_grad = group.optimizer_grad()
    torch.testing.assert_close(
        optimizer_grad.data, torch.full_like(optimizer_grad.data, expected), rtol=0, atol=0
    )
    assert group.state.grad_valid == group.layout.main_weight
    assert group.state.grad_ready
    assert group.state.full_grad is None
    assert allocator.buckets == {}

    group.zero_grad()
    assert group.state.grad_valid is None
    assert not group.state.grad_ready
    assert torch.count_nonzero(group.grad_buffer.data) == 0
