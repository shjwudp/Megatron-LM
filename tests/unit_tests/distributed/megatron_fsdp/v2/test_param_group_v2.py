# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the placement-first ParameterGroupV2 prototype."""

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import TemporaryBucketAllocator
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.buffer_index import Placement
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group_v2 import (
    GradientPhaseV2,
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


def _dp_mesh() -> DeviceMesh:
    ranks = torch.arange(torch.distributed.get_world_size(), dtype=torch.int)
    return DeviceMesh(_device().type, ranks, mesh_dim_names=("dp",))


def _build_2d_group(
    *,
    shard_optimizer_across_outer_dp: bool,
    sharding_strategy: str = "optim_grads_params",
    grad_comm_dtype: torch.dtype | None = None,
    gradient_scaling_factor: float | None = None,
    use_decoupled_grad: bool = False,
) -> tuple[ParameterGroupV2, torch.Tensor, TemporaryBucketAllocator]:
    values = torch.arange(128, dtype=torch.float32, device=_device())
    param = nn.Parameter(values.clone())
    allocator = TemporaryBucketAllocator()
    group = ParameterGroupV2(
        [param],
        ParamGroupIdx(0, 0),
        mesh=_hsdp_mesh(),
        layout=ParameterGroupLayoutV2.from_strategies(
            sharding_strategy,
            outer_dp_sharding_strategy=(
                "optim" if shard_optimizer_across_outer_dp else "no_shard"
            ),
        ),
        mp_policy=MixedPrecisionPolicy(
            main_grads_dtype=torch.float32,
            grad_comm_dtype=grad_comm_dtype,
            use_decoupled_grad=use_decoupled_grad,
        ),
        allocator=allocator,
        gradient_scaling_factor=gradient_scaling_factor,
    )
    return group, values, allocator


def _build_1d_group(
    sharding_strategy: str,
) -> tuple[ParameterGroupV2, torch.Tensor, TemporaryBucketAllocator]:
    values = torch.arange(128, dtype=torch.float32, device=_device())
    param = nn.Parameter(values.clone())
    allocator = TemporaryBucketAllocator()
    group = ParameterGroupV2(
        [param],
        ParamGroupIdx(0, 0),
        mesh=_dp_mesh(),
        layout=ParameterGroupLayoutV2.from_strategies(sharding_strategy),
        mp_policy=MixedPrecisionPolicy(main_grads_dtype=torch.float32),
        allocator=allocator,
    )
    return group, values, allocator


@pytest.mark.parametrize(
    ("sharding_strategy", "weight", "main_weight", "grad_storage", "grad_accumulation"),
    [
        (
            "no_shard",
            Placement.REPLICATE,
            Placement.REPLICATE,
            Placement.REPLICATE,
            Placement.PARTIAL,
        ),
        (
            "optim",
            Placement.REPLICATE,
            Placement.SHARD,
            Placement.REPLICATE,
            Placement.PARTIAL,
        ),
        (
            "optim_grads",
            Placement.REPLICATE,
            Placement.SHARD,
            Placement.SHARD,
            Placement.SHARD,
        ),
        (
            "optim_grads_params",
            Placement.SHARD,
            Placement.SHARD,
            Placement.SHARD,
            Placement.SHARD,
        ),
    ],
)
def test_1d_strategy_layout(
    sharding_strategy, weight, main_weight, grad_storage, grad_accumulation
):
    layout = ParameterGroupLayoutV2.from_strategies(sharding_strategy)

    assert layout.weight == (weight,)
    assert layout.main_weight == (main_weight,)
    assert layout.grad_storage == (grad_storage,)
    assert layout.grad_accumulation == (grad_accumulation,)


@pytest.mark.parametrize(
    "sharding_strategy",
    ["no_shard", "optim", "optim_grads", "optim_grads_params"],
)
def test_1d_strategy_weight_and_gradient_lifecycle(sharding_strategy):
    group, values, allocator = _build_1d_group(sharding_strategy)

    assert group.mesh.ndim == 1
    assert group.state.weight_valid == group.layout.weight
    expected_optimizer_placement = (
        Replicate() if sharding_strategy == "no_shard" else Shard(0)
    )
    assert group.optimizer_params[0].placements == (expected_optimizer_placement,)

    group.unshard_weight()
    torch.testing.assert_close(group.params[0], values)
    group.reshard_weight()

    rank = torch.distributed.get_rank()
    group.begin_backward().data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert group.state.grad_phase is GradientPhaseV2.ACCUMULATING
    if sharding_strategy in ("no_shard", "optim"):
        first_microbatch = rank + 1
    else:
        world_size = torch.distributed.get_world_size()
        first_microbatch = world_size * (world_size + 1) / 2
    torch.testing.assert_close(
        group.grad_buffer.data,
        torch.full_like(group.grad_buffer.data, first_microbatch),
        rtol=0,
        atol=0,
    )

    group.begin_backward().data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)
    world_size = torch.distributed.get_world_size()
    expected = world_size * (world_size + 2)
    torch.testing.assert_close(
        group.optimizer_grad().data,
        torch.full_like(group.optimizer_grad().data, expected),
        rtol=0,
        atol=0,
    )
    assert group.state.grad_phase is GradientPhaseV2.READY
    assert group.optimizer_params[0].grad is group.optimizer_grads[0]
    assert allocator.buckets == {}

    group.zero_grad()
    assert group.state.grad_phase is GradientPhaseV2.EMPTY


@pytest.mark.parametrize(
    "sharding_strategy",
    ["no_shard", "optim", "optim_grads", "optim_grads_params"],
)
def test_2d_strategy_gradient_lifecycle(sharding_strategy):
    group, _, allocator = _build_2d_group(
        sharding_strategy=sharding_strategy,
        shard_optimizer_across_outer_dp=False,
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    inner_size = world_size // 2

    group.begin_backward().data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert group.state.grad_phase is GradientPhaseV2.ACCUMULATING
    if sharding_strategy in ("no_shard", "optim"):
        first_microbatch = rank + 1
    else:
        outer_rank = rank // inner_size
        first_rank = outer_rank * inner_size
        first_microbatch = sum(
            inner_rank + 1 for inner_rank in range(first_rank, first_rank + inner_size)
        )
    torch.testing.assert_close(
        group.grad_buffer.data,
        torch.full_like(group.grad_buffer.data, first_microbatch),
        rtol=0,
        atol=0,
    )

    group.begin_backward().data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)
    expected = world_size * (world_size + 2)
    torch.testing.assert_close(
        group.optimizer_grad().data,
        torch.full_like(group.optimizer_grad().data, expected),
        rtol=0,
        atol=0,
    )
    assert group.state.grad_phase is GradientPhaseV2.READY
    assert allocator.buckets == {}


@pytest.mark.parametrize("shard_optimizer_across_outer_dp", [False, True])
def test_optimizer_params_own_main_weight_views(shard_optimizer_across_outer_dp):
    group, _, _ = _build_2d_group(
        shard_optimizer_across_outer_dp=shard_optimizer_across_outer_dp
    )

    assert len(group.optimizer_params) == 1
    optimizer_param = group.optimizer_params[0]
    assert isinstance(optimizer_param, DTensor)
    expected_placements = (
        (Shard(0), Shard(0))
        if shard_optimizer_across_outer_dp
        else (Replicate(), Shard(0))
    )
    assert optimizer_param.placements == expected_placements
    assert optimizer_param._local_tensor.data_ptr() == group.main_weight_buffer.view(
        list(group.layout.main_weight)
    ).tensor_view(0).data_ptr()
    assert hasattr(optimizer_param._local_tensor, "__create_chunk_list__")
    assert getattr(optimizer_param, "__fsdp_param__")
    assert getattr(group.params[0], "__fsdp_param__")
    assert group.optimizer_grads == [None]


def test_weight_validity_and_scratch_lifecycle():
    group, original, allocator = _build_2d_group(
        shard_optimizer_across_outer_dp=True
    )

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
@pytest.mark.parametrize("use_decoupled_grad", [False, True])
def test_two_microbatch_hsdp_gradient(
    shard_optimizer_across_outer_dp, grad_comm_dtype, use_decoupled_grad
):
    group, _, allocator = _build_2d_group(
        shard_optimizer_across_outer_dp=shard_optimizer_across_outer_dp,
        grad_comm_dtype=grad_comm_dtype,
        gradient_scaling_factor=0.5,
        use_decoupled_grad=use_decoupled_grad,
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    group.begin_backward().data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert group.state.grad_phase is GradientPhaseV2.ACCUMULATING
    assert group.state.full_grad is None
    assert allocator.buckets == {}

    group.begin_backward().data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)

    expected = 0.5 * world_size * (world_size + 2)
    optimizer_grad = group.optimizer_grad()
    torch.testing.assert_close(
        optimizer_grad.data, torch.full_like(optimizer_grad.data, expected), rtol=0, atol=0
    )
    assert group.state.grad_phase is GradientPhaseV2.READY
    assert group.state.full_grad is None
    assert allocator.buckets == {}

    optimizer_param = group.optimizer_params[0]
    optimizer_grad_dtensor = group.optimizer_grads[0]
    assert optimizer_grad_dtensor is not None
    assert optimizer_grad_dtensor._local_tensor.data_ptr() == optimizer_grad.tensor_view(
        0
    ).data_ptr()
    if use_decoupled_grad:
        assert optimizer_param.grad is None
        assert optimizer_param.decoupled_grad is optimizer_grad_dtensor
    else:
        assert optimizer_param.grad is optimizer_grad_dtensor
        assert getattr(optimizer_param, "decoupled_grad", None) is None

    group.zero_grad()
    assert group.state.grad_phase is GradientPhaseV2.EMPTY
    assert torch.count_nonzero(group.grad_buffer.data) == 0
    assert optimizer_param.grad is None
    assert getattr(optimizer_param, "decoupled_grad", None) is None
