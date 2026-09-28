# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure shard-planning and owner-compute packing logic for the Muon + M-FSDPv2 owner-compute P2P
algorithm.

The central data structure is `ShardPlan`, which describes how a single 2D parameter's full matrix
is split across the DP group under M-FSDPv2's all-`Flat` layout. Given shard plans,
`assign_owner_work` balances Newton-Schulz work across owner ranks, and the pack/unpack helpers
(`pack_owner_work`/`pack_update_shards`/`unpack_update_shards`) build the flat P2P send/recv
buffers. `reconstruct_full_tensor` stitches gathered shards back into the full matrix on the owner.

PORT-NOTE (LANES-MUON): ported from the prototype (dev:experimental/shard_plan.py) against
upstream's `experimental/owner_planning.py` (#6597), which covers the same owner-compute concept
under different shapes. Division of labor, so the planning library is extended rather than
duplicated:

- `ShardPlan`/`compute_shard_plan` build on `owner_planning.ParameterLayout`
  (`parameter_layout_from_shard_plan` is the bridge) because `ParameterLayout.from_group` cannot
  express what the optimizer needs: row-range slices of a 2D matrix and the HFSDP
  process-group-rank ↔ flat-shard-order permutation (`shard_order`, see
  `orthogonalized_optimizer._get_parameter_shard_order`), while `compute_shard_plan` is derived
  from raw flat offsets rather than a live `FsdpParameterGroup`.
- `assign_owner_work` delegates the balancing itself to `owner_planning.assign_owner_work`'s
  chunk-local (`chunks=`) mode — the prototype's `muon_max_params_per_owner_chunk` feature, which
  has no upstream counterpart.
- `OwnerGatherPlan`/`OwnerScatterPlan` and the pack/unpack helpers keep the prototype's per-chunk,
  per-parameter P2P shapes instead of reusing `owner_planning.OwnerGatherPlan`/`OwnerScatterPlan`,
  for two reasons: (1) one owner-communication chunk may mix parameters from several
  `FsdpParameterGroup`s that share one process group (dense + expert buckets), while upstream's
  packers bind a single mesh per `GroupOwnerLayout`; and (2) upstream's `ParameterLayout.rank_offset`
  assumes rank order == flat-shard order, which the HFSDP 2-D flattened mesh violates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch

from .layout import non_leading_numel
from .owner_planning import ParameterLayout
from .owner_planning import assign_owner_work as _assign_owner_work_across_layouts
from .owner_planning import ns_cost_fn


@dataclasses.dataclass(frozen=True)
class ShardPlan:
    """How a single 2D parameter's full matrix is split across the DP group.

    M-FSDPv2's all-`Flat` layout shards dim-0 rows contiguously. For each
    process-group rank `r`, `rank_rows[r]` gives the contiguous global row range
    `[start, start + count)` where `start`/`count` come from the flat
    DBuffer layout. A rank with `count == 0` holds no shard of this parameter.

    `rank_rows` is indexed by *process-group rank*; under HFSDP's 2-D flattened
    mesh that is a permutation of flat-shard order (see `compute_shard_plan`'s
    `shard_order`), which `owner_planning.ParameterLayout` cannot represent —
    hence this row-oriented companion rather than a plain `ParameterLayout`.
    """

    full_shape: torch.Size
    rank_rows: tuple[tuple[int, int], ...]
    row_size: int

    def __post_init__(self) -> None:
        if len(self.full_shape) != 2:
            raise ValueError(f"ShardPlan requires a 2D full_shape, got {self.full_shape}.")
        if len(self.rank_rows) == 0:
            raise ValueError("ShardPlan requires at least one rank.")
        if self.row_size != non_leading_numel(self.full_shape):
            raise ValueError(
                f"ShardPlan row_size {self.row_size} != full_shape row size "
                f"{non_leading_numel(self.full_shape)}."
            )

    @property
    def world_size(self) -> int:
        """Number of ranks in the DP group for this parameter."""
        return len(self.rank_rows)

    def rank_row_count(self, rank: int) -> int:
        """Return the number of rows owned by `rank`."""
        return self.rank_rows[rank][1]

    def shard_numel(self, rank: int) -> int:
        """Return the number of elements in `rank`'s shard."""
        return self.rank_row_count(rank) * self.row_size

    def owner_candidates(self) -> tuple[int, ...]:
        """Return the ranks that hold a non-empty shard of this parameter."""
        return tuple(r for r, (_, count) in enumerate(self.rank_rows) if count > 0)

    def is_boundary(self) -> bool:
        """True if more than one rank owns a non-empty shard of this parameter."""
        return sum(1 for _, count in self.rank_rows if count > 0) > 1

    def full_numel(self) -> int:
        """Return the total number of elements in the full (unsharded) parameter."""
        return self.full_shape.numel()


def parameter_layout_from_shard_plan(plan: ShardPlan) -> ParameterLayout:
    """Return the `owner_planning.ParameterLayout` flat view of a `ShardPlan`.

    PORT-NOTE (LANES-MUON): bridge onto upstream's planning library. `flat_counts` are the
    per-process-group-rank element counts (`rank_rows` count × `row_size`), so the eligibility,
    cost and boundary predicates line up with `ShardPlan` exactly and the shared owner balancer
    can be reused. Only use the result for those predicates (and `full_numel`/`dp_size`):
    `ParameterLayout.rank_offset` assumes rank order == flat-shard order, which does not hold
    under HFSDP's permuted `shard_order`, so the row-based pack/unpack helpers below never go
    through this view. A plan that does not tile its full shape (possible only with a truncated
    final flat shard, i.e. a misaligned layout) is rejected loudly here, where the prototype
    would have silently dropped rows.
    """
    return ParameterLayout(
        full_shape=plan.full_shape,
        flat_counts=tuple(plan.shard_numel(rank) for rank in range(plan.world_size)),
    )


def compute_shard_plan(
    full_shape: torch.Size,
    tensor_flat_offset: int,
    rank_flat_shard_size: int,
    world_size: int,
    shard_order: Sequence[int] | None = None,
) -> ShardPlan:
    """Compute the per-rank row ranges for one 2D parameter in a flat DBuffer.

    Args:
        full_shape: Global `(rows, cols)` shape of the parameter.
        tensor_flat_offset: Flat-element offset of this parameter inside the
            DBuffer's global layout.
        rank_flat_shard_size: Flat elements each DP rank owns (uniform for the
            even all-`Flat` layout: `layout.size // world_size`).
        world_size: DP group size.
        shard_order: Flat DBuffer shard index for each process-group rank.
            Defaults to process-group rank order.

    Returns:
        The `ShardPlan` describing which rows each rank owns.
    """
    if len(full_shape) != 2:
        raise ValueError(f"compute_shard_plan requires a 2D shape, got {full_shape}.")
    row_size = non_leading_numel(full_shape)
    if row_size <= 0:
        raise ValueError(f"compute_shard_plan requires non-empty rows, got shape {full_shape}.")
    tensor_end = tensor_flat_offset + full_shape.numel()

    if shard_order is None:
        shard_order = range(world_size)
    rank_rows: list[tuple[int, int]] = []
    for shard_index in shard_order:
        rank_start = shard_index * rank_flat_shard_size
        rank_end = rank_start + rank_flat_shard_size
        overlap_start = max(tensor_flat_offset, rank_start)
        overlap_end = min(tensor_end, rank_end)
        if overlap_start >= overlap_end:
            rank_rows.append((0, 0))
            continue
        if (overlap_start - tensor_flat_offset) % row_size != 0:
            raise RuntimeError(
                f"Flat shard boundary is not row-aligned for shape {full_shape}: "
                f"overlap_start={overlap_start}, tensor_flat_offset={tensor_flat_offset}."
            )
        overlap_numel = overlap_end - overlap_start
        if overlap_numel % row_size != 0:
            raise RuntimeError(
                f"Flat shard overlap is not row-aligned for shape {full_shape}: "
                f"overlap_numel={overlap_numel}, row_size={row_size}."
            )
        row_start = (overlap_start - tensor_flat_offset) // row_size
        row_count = overlap_numel // row_size
        rank_rows.append((row_start, row_count))
    return ShardPlan(
        full_shape=torch.Size(full_shape), rank_rows=tuple(rank_rows), row_size=row_size
    )


def assign_owner_work(
    plans: Sequence[ShardPlan], num_ns_steps: int, chunks: Sequence[Sequence[int]] | None = None
) -> dict[int, int]:
    """Assign one owner rank to each parameter with chunk-local load balancing.

    Only ranks that own a non-empty shard of a parameter are eligible owners.
    Parameters are processed in descending estimated Newton-Schulz cost. For
    each chunk, assignment minimizes each rank's cumulative cost, including
    fixed fully-local work and owner work from preceding chunks. The cost
    estimate for an M x N matrix is
    `M * N * (min(M, N) * num_ns_steps + 1)`.

    PORT-NOTE (LANES-MUON): the balancing itself lives in `owner_planning.assign_owner_work`
    (`chunks=` mode), so the planning library keeps a single greedy balancer; this wrapper only
    adapts `ShardPlan` to `ParameterLayout` and pins the cost function to
    `owner_planning.ns_cost_fn(num_ns_steps)`, whose formula is identical to the prototype's for
    2D shapes. With `chunks=None` the prototype balanced all parameters as one chunk, which is
    preserved by passing the single all-inclusive chunk.

    Args:
        plans: Shard plans indexed by their position in the input sequence.
        num_ns_steps: Newton-Schulz iteration count used in the cost estimate.
        chunks: Parameter indices grouped by communication chunk. When omitted,
            all parameters are balanced as one chunk.

    Returns:
        Mapping from parameter index (in `plans`) to owner rank.
    """
    if not plans:
        return {}

    if chunks is None:
        chunks = (tuple(range(len(plans))),)
    layouts = {
        param_index: parameter_layout_from_shard_plan(plan)
        for param_index, plan in enumerate(plans)
    }
    return _assign_owner_work_across_layouts(layouts, ns_cost_fn(num_ns_steps), chunks=chunks)


@dataclasses.dataclass
class OwnerGatherPlan:
    """Metadata and send buffers for the owner-gather P2P step of one chunk.

    The owner keeps its own shard locally (no self-send), so it only receives
    from the other shard-holding ranks and reconstructs each owned matrix by
    concatenating shards in rank order (own shard at the owner's rank rows).

    Attributes:
        send_buffers: Per-parameter, per-owner flattened pre-NS shard.
        recv_sizes: Per-parameter, per-source element count received by its owner.
        own_shards: This rank's local shard per owned parameter (used directly
            in reconstruction, not communicated).
        recv_offsets: Per `(param_index, src_rank)` of `(offset, numel,
            row_count)` describing where this param's shard lands inside the
            recv buffer received from `src_rank`.
        comm_groups: Per-parameter process groups in shard-plan order.
    """

    send_buffers: dict[tuple[int, int], torch.Tensor]
    recv_sizes: dict[tuple[int, int], int]
    own_shards: dict[int, torch.Tensor]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]
    comm_groups: tuple[torch.distributed.ProcessGroup, ...]


def pack_owner_work(
    plans: Sequence[ShardPlan],
    owners: dict[int, int],
    local_shards: Sequence[torch.Tensor],
    comm_groups: list[torch.distributed.ProcessGroup],
) -> OwnerGatherPlan:
    """Pack this rank's pre-NS shards into per-parameter P2P buffers.

    Args:
        plans: Shard plans in parameter order.
        owners: Mapping from parameter index to owner rank.
        local_shards: This rank's local pre-NS shard per parameter.
        comm_groups: Per-parameter process groups in shard-plan order.

    Returns:
        The `OwnerGatherPlan` for this rank.
    """
    send_buffers: dict[tuple[int, int], torch.Tensor] = {}
    recv_sizes: dict[tuple[int, int], int] = {}
    own_shards: dict[int, torch.Tensor] = {}
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]] = {}
    for param_index, (plan, shard, comm_group) in enumerate(zip(plans, local_shards, comm_groups)):
        world_size = torch.distributed.get_world_size(group=comm_group)
        this_rank = torch.distributed.get_rank(group=comm_group)
        owner = owners[param_index]
        if owner == this_rank:
            own_shards[param_index] = shard
            for src in range(world_size):
                numel = plan.shard_numel(src)
                if src != this_rank and numel > 0:
                    recv_sizes[(param_index, src)] = numel
                    recv_offsets[(param_index, src)] = (0, numel, plan.rank_row_count(src))
        else:
            numel = plan.shard_numel(this_rank)
            if numel > 0:
                send_buffers[(param_index, owner)] = shard.reshape(-1).clone()

    return OwnerGatherPlan(
        send_buffers=send_buffers,
        recv_sizes=recv_sizes,
        own_shards=own_shards,
        recv_offsets=recv_offsets,
        comm_groups=tuple(comm_groups),
    )


def reconstruct_full_tensor(
    param_index: int,
    plan: ShardPlan,
    gather_plan: OwnerGatherPlan,
    recv_buffers: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Reconstruct the full 2D tensor for one owned parameter from its per-rank shards.

    Concatenates shards in global row order: the owner's own local shard at its rank rows and each
    source's received shard at that source's rank rows. The shard content lives in `gather_plan`
    (`own_shards` + `recv_offsets`), so this works for the pre-NS owner-gather path and for a weight
    gather plan alike – pass a weight gather plan built with `pack_owner_work` to reconstruct the
    full weight parameter instead.

    Args:
        param_index: Index of the parameter within the chunk.
        plan: Shard plan for this parameter.
        gather_plan: This rank's owner-gather plan (own_shards, recv_offsets).
        recv_buffers: Per-source-rank received buffer (only sources that sent).

    Returns:
        The full `(rows, cols)` tensor.
    """
    world_size = plan.world_size
    owner_rank = torch.distributed.get_rank(group=gather_plan.comm_groups[param_index])
    shards: list[torch.Tensor] = []
    ranks_by_row = sorted(range(world_size), key=lambda rank: plan.rank_rows[rank][0])
    for src in ranks_by_row:
        row_count = plan.rank_row_count(src)
        if src == owner_rank:
            shards.append(gather_plan.own_shards[param_index])
        elif row_count == 0:
            continue
        else:
            offset, numel, _ = gather_plan.recv_offsets[(param_index, src)]
            buf = recv_buffers[(param_index, src)]
            shards.append(buf[offset : offset + numel].view(row_count, plan.row_size))
    if len(shards) == 1:
        return shards[0].contiguous()
    return torch.cat(shards, dim=0)


@dataclasses.dataclass
class OwnerScatterPlan:
    """Metadata and send buffers for the owner-scatter P2P step of one chunk.

    The owner keeps its own update shard (applied directly), so it only sends to
    the other shard-holding ranks.

    Attributes:
        send_buffers: Per-parameter, per-destination flattened update shard.
        recv_sizes: Per-parameter, per-owner element count received by its destination.
        recv_offsets: Per `(param_index, owner_rank)` of `(offset, numel,
            row_count)` describing where this param's update shard lands inside
            the recv buffer received from `owner_rank`.
        comm_groups: Per-parameter process groups in shard-plan order.
    """

    send_buffers: dict[tuple[int, int], torch.Tensor]
    recv_sizes: dict[tuple[int, int], int]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]
    comm_groups: tuple[torch.distributed.ProcessGroup, ...]


def pack_update_shards(
    full_updates: dict[int, torch.Tensor],
    plans: Sequence[ShardPlan],
    owners: dict[int, int],
    comm_groups: list[torch.distributed.ProcessGroup],
) -> OwnerScatterPlan:
    """Pack full updates into per-parameter, per-destination P2P buffers.

    Args:
        full_updates: Full update matrix per owned parameter index.
        plans: Shard plans in parameter order.
        owners: Mapping from parameter index to owner rank.
        comm_groups: Per-parameter process groups in shard-plan order.

    Returns:
        The `OwnerScatterPlan` for this rank.
    """
    send_buffers: dict[tuple[int, int], torch.Tensor] = {}
    recv_sizes: dict[tuple[int, int], int] = {}
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]] = {}
    for param_index, (plan, comm_group) in enumerate(zip(plans, comm_groups)):
        world_size = torch.distributed.get_world_size(group=comm_group)
        this_rank = torch.distributed.get_rank(group=comm_group)
        owner = owners[param_index]
        if owner == this_rank:
            full_update = full_updates[param_index]
            for dest in range(world_size):
                row_start, row_count = plan.rank_rows[dest]
                if dest != this_rank and row_count > 0:
                    send_buffers[(param_index, dest)] = (
                        full_update[row_start : row_start + row_count].reshape(-1).clone()
                    )
        else:
            numel = plan.shard_numel(this_rank)
            if numel > 0:
                recv_sizes[(param_index, owner)] = numel
                recv_offsets[(param_index, owner)] = (0, numel, plan.rank_row_count(this_rank))

    return OwnerScatterPlan(
        send_buffers=send_buffers,
        recv_sizes=recv_sizes,
        recv_offsets=recv_offsets,
        comm_groups=tuple(comm_groups),
    )


def unpack_update_shards(
    scatter_plan: OwnerScatterPlan, recv_buffers: dict[tuple[int, int], torch.Tensor]
) -> dict[int, torch.Tensor]:
    """Extract this rank's local update shards from the per-owner recv buffers.

    Args:
        scatter_plan: This rank's owner-scatter plan (recv_offsets).
        recv_buffers: Per-parameter, per-owner received buffers.

    Returns:
        Mapping from parameter index to the local update shard `(row_count, cols)`,
        for parameters this rank does NOT own.
    """
    updates: dict[int, torch.Tensor] = {}
    for (param_index, owner), (offset, numel, row_count) in scatter_plan.recv_offsets.items():
        buf = recv_buffers[(param_index, owner)]
        row_size = numel // row_count
        updates[param_index] = buf[offset : offset + numel].view(row_count, row_size)
    return updates
