# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the placement-first ParameterGroup."""

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import TemporaryBucketAllocator
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.buffer_index import Placement
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.dp_buffer import DataParallelBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group import (
    GradientPhase,
    ParameterGroup,
    ParameterGroupLayout,
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
        pytest.skip("ParameterGroup HSDP tests require an even world size >= 4")
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
) -> tuple[ParameterGroup, torch.Tensor, TemporaryBucketAllocator]:
    values = torch.arange(128, dtype=torch.float32, device=_device())
    param = nn.Parameter(values.clone())
    allocator = TemporaryBucketAllocator()
    group = ParameterGroup(
        [param],
        ParamGroupIdx(0, 0),
        mesh=_hsdp_mesh(),
        layout=ParameterGroupLayout.from_strategies(
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
    *,
    grad_comm_dtype: torch.dtype | None = None,
    gradient_scaling_factor: float | None = None,
) -> tuple[ParameterGroup, torch.Tensor, TemporaryBucketAllocator]:
    values = torch.arange(128, dtype=torch.float32, device=_device())
    param = nn.Parameter(values.clone())
    allocator = TemporaryBucketAllocator()
    group = ParameterGroup(
        [param],
        ParamGroupIdx(0, 0),
        mesh=_dp_mesh(),
        layout=ParameterGroupLayout.from_strategies(sharding_strategy),
        mp_policy=MixedPrecisionPolicy(
            main_grads_dtype=torch.float32, grad_comm_dtype=grad_comm_dtype
        ),
        allocator=allocator,
        gradient_scaling_factor=gradient_scaling_factor,
    )
    return group, values, allocator


def test_dp_buffer_view_resolves_partial_as_replicated_physical_storage():
    group, _, allocator = _build_2d_group(shard_optimizer_across_outer_dp=True)
    group.prepare_gradient_storage()

    persistent_partial = group.grad_buffer.view(
        (Placement.PARTIAL, Placement.SHARD)
    )
    persistent_physical = group.grad_buffer.view(
        (Placement.REPLICATE, Placement.SHARD)
    )
    assert persistent_partial.placements == [Placement.PARTIAL, Placement.SHARD]
    assert persistent_partial.data.data_ptr() == persistent_physical.data.data_ptr()
    with pytest.raises(ValueError, match="do not contain"):
        persistent_partial.view((Placement.REPLICATE, Placement.SHARD))

    full_grad = group.acquire_full_grad_buffer()
    full_partial = full_grad.view((Placement.PARTIAL, Placement.PARTIAL))
    inner_shard_partial = full_grad.view((Placement.PARTIAL, Placement.SHARD))
    inner_shard_physical = full_grad.view((Placement.REPLICATE, Placement.SHARD))
    assert full_partial.placements == [Placement.PARTIAL, Placement.PARTIAL]
    assert full_partial.data.data_ptr() == full_grad.data.data_ptr()
    assert inner_shard_partial.placements == [Placement.PARTIAL, Placement.SHARD]
    assert inner_shard_partial.data.data_ptr() == inner_shard_physical.data.data_ptr()

    group.release_temporary_grad_buffers()
    assert allocator.buckets == {}


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
    layout = ParameterGroupLayout.from_strategies(sharding_strategy)

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
    accumulates_full_grad = sharding_strategy in ("no_shard", "optim")

    assert group.mesh.ndim == 1
    assert group.state.weight_valid == group.layout.weight
    assert group.accumulates_full_grad is accumulates_full_grad
    assert group.state.full_grad is None
    expected_optimizer_placement = (
        Replicate() if sharding_strategy == "no_shard" else Shard(0)
    )
    assert group.optimizer_params[0].placements == (expected_optimizer_placement,)

    group.unshard_weight()
    torch.testing.assert_close(group.params[0], values)
    group.reshard_weight()

    rank = torch.distributed.get_rank()
    full_grad = group.acquire_full_grad_buffer()
    assert (full_grad.data.data_ptr() == group.grad_buffer.data.data_ptr()) is accumulates_full_grad
    assert any(key[1] == "full_grad" for key in allocator.buckets) is not accumulates_full_grad
    full_grad.data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert group.state.grad_phase is GradientPhase.ACCUMULATING
    if accumulates_full_grad:
        assert group.state.full_grad is full_grad
    else:
        assert group.state.full_grad is None
    assert group.full_grad_has_value is accumulates_full_grad
    assert group.overwrites_full_grad is not accumulates_full_grad
    if accumulates_full_grad:
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

    next_full_grad = group.acquire_full_grad_buffer()
    assert (next_full_grad is full_grad) is accumulates_full_grad
    full_grad = next_full_grad
    if group.full_grad_has_value:
        full_grad.data.add_(rank + 2)
    else:
        full_grad.data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)
    world_size = torch.distributed.get_world_size()
    expected = world_size * (world_size + 2)
    torch.testing.assert_close(
        group.optimizer_grad().data,
        torch.full_like(group.optimizer_grad().data, expected),
        rtol=0,
        atol=0,
    )
    assert group.state.grad_phase is GradientPhase.READY
    if accumulates_full_grad:
        assert group.state.full_grad is full_grad
    else:
        assert group.state.full_grad is None
    assert group.optimizer_params[0].grad is group.optimizer_grads[0]
    assert allocator.buckets == {}

    group.zero_grad()
    assert group.state.grad_phase is GradientPhase.EMPTY
    assert group.state.full_grad is None


@pytest.mark.parametrize("sharding_strategy", ["no_shard", "optim"])
def test_full_gradient_view_follows_persistent_storage_lifetime(sharding_strategy):
    group, _, _ = _build_1d_group(sharding_strategy)

    assert group.grad_buffer.data is None
    assert group.state.full_grad is None
    group.prepare_gradient_storage()
    full_grad = group.state.full_grad
    assert full_grad is not None
    assert group.acquire_full_grad_buffer() is full_grad
    grad_storage = group.grad_buffer.data
    group.release_temporary_grad_buffers()
    assert group.state.full_grad is full_grad
    assert group.state.full_grad.data.data_ptr() == grad_storage.data_ptr()

    group.zero_grad(set_to_none=False)
    assert group.state.full_grad is full_grad
    assert group.state.full_grad.data.data_ptr() == grad_storage.data_ptr()

    group.zero_grad(set_to_none=True)
    assert group.state.full_grad is None
    assert group.grad_buffer.data is None

    rebound_full_grad = group.acquire_full_grad_buffer()
    assert rebound_full_grad is not full_grad
    assert rebound_full_grad.data.data_ptr() == group.grad_buffer.data.data_ptr()


def test_temporary_full_gradient_lease_precedes_grad_storage_release():
    group, _, allocator = _build_1d_group("optim_grads_params")

    full_grad = group.acquire_full_grad_buffer()
    with pytest.raises(RuntimeError, match="Temporary full-gradient storage"):
        group._release_grad_storage()
    assert group.state.full_grad is full_grad
    assert any(key[1] == "full_grad" for key in allocator.buckets)

    group.release_temporary_grad_buffers()
    group._release_grad_storage()
    assert group.state.full_grad is None
    assert group.grad_buffer.data is None
    assert allocator.buckets == {}


def test_reduce_grad_failure_releases_temporary_buffers(monkeypatch):
    group, _, allocator = _build_1d_group(
        "optim_grads_params", grad_comm_dtype=torch.bfloat16
    )
    group.acquire_full_grad_buffer().data.fill_(1)

    def fail_redistribution(*args, **kwargs):
        raise RuntimeError("redistribution failed")

    monkeypatch.setattr(DataParallelBuffer, "redistribute_buffers", fail_redistribution)
    with pytest.raises(RuntimeError, match="redistribution failed"):
        group.reduce_grad(is_last_backward=True)

    assert group.state.full_grad is None
    assert group.state.grad_comm is None
    assert allocator.buckets == {}


def test_unshard_planning_failure_releases_earlier_weight_buffers(monkeypatch):
    first, _, first_allocator = _build_2d_group(shard_optimizer_across_outer_dp=True)
    second, _, second_allocator = _build_2d_group(shard_optimizer_across_outer_dp=True)

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(second, "_allocate_scratch", fail_allocation)
    with pytest.raises(RuntimeError, match="allocation failed"):
        ParameterGroup.unshard_weights([first, second])

    assert first.state.full_weights == {}
    assert second.state.full_weights == {}
    assert first_allocator.buckets == {}
    assert second_allocator.buckets == {}


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
    accumulates_full_grad = sharding_strategy in ("no_shard", "optim")
    assert group.accumulates_full_grad is accumulates_full_grad

    full_grad = group.acquire_full_grad_buffer()
    assert (full_grad.data.data_ptr() == group.grad_buffer.data.data_ptr()) is accumulates_full_grad
    full_grad.data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert group.state.grad_phase is GradientPhase.ACCUMULATING
    assert group.full_grad_has_value is accumulates_full_grad
    if accumulates_full_grad:
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

    full_grad = group.acquire_full_grad_buffer()
    if group.full_grad_has_value:
        full_grad.data.add_(rank + 2)
    else:
        full_grad.data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)
    expected = world_size * (world_size + 2)
    torch.testing.assert_close(
        group.optimizer_grad().data,
        torch.full_like(group.optimizer_grad().data, expected),
        rtol=0,
        atol=0,
    )
    assert group.state.grad_phase is GradientPhase.READY
    assert allocator.buckets == {}


@pytest.mark.parametrize("sharding_strategy", ["no_shard", "optim"])
def test_full_gradient_accumulation_is_preprocessed_once(sharding_strategy):
    group, _, allocator = _build_1d_group(
        sharding_strategy, grad_comm_dtype=torch.bfloat16, gradient_scaling_factor=0.5
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    group.acquire_full_grad_buffer().data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert allocator.buckets == {}

    group.acquire_full_grad_buffer().data.add_(rank + 2)
    group.reduce_grad(is_last_backward=True)

    expected = 0.5 * world_size * (world_size + 2)
    torch.testing.assert_close(
        group.optimizer_grad().data,
        torch.full_like(group.optimizer_grad().data, expected),
        rtol=0,
        atol=0,
    )
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
    assert group.dtype == group.params[0].dtype
    assert group.requires_grad
    assert not group.full_grad_has_value
    assert group.overwrites_full_grad
    assert group.supports_fused_grad_capture


def test_weight_validity_and_scratch_lifecycle():
    group, original, allocator = _build_2d_group(
        shard_optimizer_across_outer_dp=True
    )

    assert group.state.weight_valid == (Placement.REPLICATE, Placement.SHARD)
    assert group.get_unsharded_weight_buffer() is None

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


def test_outer_sharded_hsdp_collective_order(monkeypatch):
    group, _, _ = _build_2d_group(shard_optimizer_across_outer_dp=True)
    transitions = []
    redistribute = DataParallelBuffer.redistribute

    def record_redistribution(self, target_placements, **kwargs):
        transitions.append(
            (tuple(self.placements), tuple(target_placements), torch.cuda.current_stream())
        )
        return redistribute(self, target_placements, **kwargs)

    monkeypatch.setattr(DataParallelBuffer, "redistribute", record_redistribution)

    outer_ag_stream = torch.cuda.Stream()
    inner_ag_stream = torch.cuda.Stream()
    group.refresh_model_weight()
    transitions.clear()
    group.unshard_weight(streams=(outer_ag_stream, inner_ag_stream), async_op=True)
    inner_ag_stream.synchronize()
    assert transitions == [
        (
            (Placement.SHARD, Placement.SHARD),
            (Placement.REPLICATE, Placement.SHARD),
            outer_ag_stream,
        ),
        (
            (Placement.REPLICATE, Placement.SHARD),
            (Placement.REPLICATE, Placement.REPLICATE),
            inner_ag_stream,
        ),
    ]

    group.reshard_weight()
    transitions.clear()
    group.acquire_full_grad_buffer().data.fill_(torch.distributed.get_rank() + 1)
    outer_rs_stream = torch.cuda.Stream()
    inner_rs_stream = torch.cuda.Stream()
    completion_stream = group.reduce_grad(
        is_last_backward=True, streams=(outer_rs_stream, inner_rs_stream), async_op=True
    )
    assert transitions == [
        (
            (Placement.PARTIAL, Placement.PARTIAL),
            (Placement.PARTIAL, Placement.SHARD),
            inner_rs_stream,
        ),
        (
            (Placement.PARTIAL, Placement.SHARD),
            (Placement.SHARD, Placement.SHARD),
            outer_rs_stream,
        ),
    ]
    assert completion_stream == outer_rs_stream
    completion_stream.synchronize()
    group.release_temporary_grad_buffers()


def test_layout_rejects_mesh_rank_mismatch():
    layout = ParameterGroupLayout(
        weight=(Placement.SHARD,),
        main_weight=(Placement.SHARD,),
        grad_storage=(Placement.SHARD,),
        grad_accumulation=(Placement.SHARD,),
    )
    with pytest.raises(ValueError, match="Expected 2 placements"):
        layout.validate(2)


def test_gradient_storage_zeroing_is_lazy(monkeypatch):
    group, _, _ = _build_1d_group("optim_grads_params")
    assert group.grad_buffer.data is None
    allocate_scratch = group._allocate_scratch

    def allocate_with_sentinel(role, prototype, placements):
        buffer = allocate_scratch(role, prototype, placements)
        buffer.data.fill_(13)
        return buffer

    monkeypatch.setattr(group, "_allocate_scratch", allocate_with_sentinel)
    assert torch.count_nonzero(group.acquire_full_grad_buffer().data != 13) == 0

    group.grad_buffer.data.fill_(11)
    group.zero_grad()
    assert group.state.grad_phase is GradientPhase.EMPTY
    assert group.grad_buffer.data is None

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    group.acquire_full_grad_buffer().data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=True)
    expected = world_size * (world_size + 1) / 2
    torch.testing.assert_close(
        group.optimizer_grad().data,
        torch.full_like(group.optimizer_grad().data, expected),
        rtol=0,
        atol=0,
    )
    optimizer_grad = group.optimizer_grads[0]
    assert optimizer_grad is not None

    group.zero_grad()
    assert group.grad_buffer.data is None
    assert optimizer_grad._local_tensor is None

    group.acquire_full_grad_buffer().data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)
    assert group.optimizer_grads[0] is optimizer_grad
    assert optimizer_grad._local_tensor.data_ptr() == group.optimizer_grad().tensor_view(
        0
    ).data_ptr()

    group.zero_grad(set_to_none=False)
    assert group.grad_buffer.data is not None
    assert torch.count_nonzero(group.grad_buffer.data) == 0


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

    group.acquire_full_grad_buffer().data.fill_(rank + 1)
    group.reduce_grad(is_last_backward=False)
    assert group.state.grad_phase is GradientPhase.ACCUMULATING
    assert group.state.full_grad is None
    assert allocator.buckets == {}

    group.acquire_full_grad_buffer().data.fill_(rank + 2)
    group.reduce_grad(is_last_backward=True)

    expected = 0.5 * world_size * (world_size + 2)
    optimizer_grad = group.optimizer_grad()
    torch.testing.assert_close(
        optimizer_grad.data, torch.full_like(optimizer_grad.data, expected), rtol=0, atol=0
    )
    assert group.state.grad_phase is GradientPhase.READY
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
    assert group.state.grad_phase is GradientPhase.EMPTY
    assert optimizer_param.grad is None
    assert getattr(optimizer_param, "decoupled_grad", None) is None
