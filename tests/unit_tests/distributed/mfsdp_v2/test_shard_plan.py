# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Owner-compute shard planning: flat shards map to matrix row ranges and balanced owners.

Pinned here are ``ShardPlan``/``compute_shard_plan`` row-range derivation and the
``assign_owner_work`` load balancing; both run on CPU without a process group.
The pack/unpack P2P round trip is covered against the current per-parameter
communication-group API by ``test_orthogonalized_optimizer.test_owner_p2p_round_trip_multi_owner``.
"""

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.shard_plan import (
    ShardPlan,
    assign_owner_work,
    compute_shard_plan,
)


def test_compute_shard_plan_splits_rows_across_rank_shards():
    """Flat rank shards map to contiguous row ranges in rank order."""
    plan = compute_shard_plan(
        torch.Size((8, 4)), tensor_flat_offset=0, rank_flat_shard_size=16, world_size=2
    )
    assert plan.full_shape == torch.Size((8, 4))
    assert plan.row_size == 4
    assert plan.rank_rows == ((0, 4), (4, 4))
    assert plan.shard_numel(0) == 16
    assert plan.shard_numel(1) == 16

    # A matrix landing across a rank boundary is split unevenly and classified
    # as a boundary parameter.
    boundary = compute_shard_plan(
        torch.Size((6, 3)), tensor_flat_offset=0, rank_flat_shard_size=9, world_size=2
    )
    assert boundary.rank_rows == ((0, 3), (3, 3))
    assert boundary.is_boundary()


def test_compute_shard_plan_classifies_locality():
    """Fully local matrices name one owner; non-overlapping ranks own zero rows."""
    # 8 elements fit entirely in rank0's [0, 12) shard.
    local = compute_shard_plan(
        torch.Size((4, 2)), tensor_flat_offset=0, rank_flat_shard_size=12, world_size=2
    )
    assert local.rank_rows == ((0, 4), (0, 0))
    assert not local.is_boundary()
    assert local.owner_candidates() == (0,)

    # Tensor offset 12 sits entirely in rank1's shard; rank0 owns zero rows.
    empty_rank = compute_shard_plan(
        torch.Size((4, 3)), tensor_flat_offset=12, rank_flat_shard_size=12, world_size=2
    )
    assert empty_rank.rank_rows == ((0, 0), (0, 4))
    assert not empty_rank.is_boundary()
    assert empty_rank.owner_candidates() == (1,)


def test_assign_owner_work_balances_by_cost():
    """Boundary parameters go to the eligible rank with the lowest running load."""
    plan0 = ShardPlan(torch.Size((8, 8)), ((0, 4), (0, 4), (0, 4), (0, 4)), 8)  # cost 64*41
    plan1 = ShardPlan(torch.Size((4, 4)), ((0, 2), (0, 2), (0, 2), (0, 2)), 4)  # cost 16*21
    owners = assign_owner_work([plan0, plan1], num_ns_steps=5)
    # Greedy min running cost: first param -> rank0 (cost 2624), second -> rank1 (cost 336).
    assert owners == {0: 0, 1: 1}


def test_assign_owner_work_ownership_eligibility():
    """Only shard-holding ranks may own; fully local parameters have one candidate."""
    # Only ranks 0 and 2 hold shards of both parameters.
    plan0 = ShardPlan(torch.Size((8, 8)), ((0, 4), (0, 0), (0, 4), (0, 0)), 8)
    plan1 = ShardPlan(torch.Size((8, 8)), ((0, 4), (0, 0), (0, 4), (0, 0)), 8)
    owners = assign_owner_work([plan0, plan1], num_ns_steps=5)
    assert all(owner in (0, 2) for owner in owners.values())
    # Two equal-cost parameters split across the two eligible ranks.
    assert owners[0] != owners[1]

    # A fully local parameter is assigned to its single owning rank.
    assert assign_owner_work(
        [ShardPlan(torch.Size((4, 2)), ((0, 4), (0, 0)), 2)], num_ns_steps=3
    ) == {0: 0}
