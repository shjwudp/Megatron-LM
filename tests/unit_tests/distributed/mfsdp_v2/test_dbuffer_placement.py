# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP placement helpers."""

import pytest
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    changed_mesh_axes,
    changed_mesh_axis,
    flattened_mesh_group,
    placement_reduce_group,
    sharded_mesh_axes,
)


def _process_group_ranks(group) -> list[int]:
    return list(dist.get_process_group_ranks(group))


def _two_axis_mesh(distributed_setup):
    return init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("dp_outer", "dp_shard"),
    )


def test_changed_mesh_axes_reports_every_change():
    """changed_mesh_axes returns all changed axes; changed_mesh_axis stays strict."""
    assert changed_mesh_axes((Replicate(), Replicate()), (Replicate(), Replicate())) == ()
    assert changed_mesh_axes((Replicate(), Replicate()), (Shard(0), Replicate())) == (0,)
    assert changed_mesh_axes((Shard(0), Replicate()), (Replicate(), Shard(0))) == (0, 1)

    assert changed_mesh_axis((Replicate(), Replicate()), (Replicate(), Replicate())) is None
    assert changed_mesh_axis((Replicate(), Replicate()), (Replicate(), Shard(0))) == 1
    with pytest.raises(NotImplementedError, match="at most one changed placement axis"):
        changed_mesh_axis((Shard(0), Shard(0)), (Replicate(), Replicate()))


def test_sharded_mesh_axes_counts_only_shards():
    """A Partial axis is neither a replicated copy nor a disjoint shard."""
    assert sharded_mesh_axes((Replicate(), Replicate())) == ()
    assert sharded_mesh_axes((Replicate(), Shard(0))) == (1,)
    assert sharded_mesh_axes((Shard(0), Shard(0))) == (0, 1)
    with pytest.raises(NotImplementedError, match="Unsupported placement"):
        sharded_mesh_axes((Partial("avg"), Shard(0)))


def test_placement_reduce_group_spans_every_sharded_axis(distributed_setup):
    """The amax reduce group must cover every axis the buffer is sharded over."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D placement test requires an even world size of at least 4.")

    mesh = _two_axis_mesh(distributed_setup)

    # All-Replicate: every rank holds an identical copy, so a MAX is idempotent
    # and this mesh's own axis 0 is the documented fallback.
    assert _process_group_ranks(
        placement_reduce_group(mesh, (Replicate(), Replicate()))
    ) == _process_group_ranks(mesh.get_group(0))
    # Exactly one sharded axis: that axis's group already owns every block.
    assert _process_group_ranks(
        placement_reduce_group(mesh, (Replicate(), Shard(0)))
    ) == _process_group_ranks(mesh.get_group(1))
    assert _process_group_ranks(
        placement_reduce_group(mesh, (Shard(0), Replicate()))
    ) == _process_group_ranks(mesh.get_group(0))
    # Both axes sharded (the HFSDP dense optimizer placement): no single mesh axis
    # sees every block, so the flattened group over both axes is required.
    assert _process_group_ranks(
        placement_reduce_group(mesh, (Shard(0), Shard(0)))
    ) == _process_group_ranks(flattened_mesh_group(mesh))
    assert _process_group_ranks(
        placement_reduce_group(mesh, (Shard(0), Shard(0)))
    ) == _process_group_ranks(mesh._flatten().get_group())
    assert len(_process_group_ranks(placement_reduce_group(mesh, (Shard(0), Shard(0))))) > len(
        _process_group_ranks(mesh.get_group(0))
    )


def test_placement_reduce_group_prefers_recorded_flattened_group(distributed_setup):
    """A flattened group recorded on the mesh is returned as-is."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D placement test requires an even world size of at least 4.")

    mesh = _two_axis_mesh(distributed_setup)
    recorded_group = mesh._flatten().get_group()
    mesh._mfsdp_flattened_group = recorded_group

    assert placement_reduce_group(mesh, (Shard(0), Shard(0))) is recorded_group
    # A single sharded axis still uses its own axis group, not the flattened one.
    assert _process_group_ranks(
        placement_reduce_group(mesh, (Replicate(), Shard(0)))
    ) == _process_group_ranks(mesh.get_group(1))


def test_placement_reduce_group_rejects_partial(distributed_setup):
    """A Partial axis cannot be reconstructed by any reduction group."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2 != 0:
        pytest.skip("2D placement test requires an even world size of at least 4.")

    mesh = _two_axis_mesh(distributed_setup)

    with pytest.raises(NotImplementedError, match="Unsupported placement"):
        placement_reduce_group(mesh, (Partial("avg"), Shard(0)))


def test_placement_reduce_group_rejects_a_partial_shard_span(distributed_setup):
    """A shard span that is not every axis has no existing process group."""
    if distributed_setup.world_size < 8:
        pytest.skip("3D placement test requires at least 8 ranks.")

    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, 2, distributed_setup.world_size // 4),
        mesh_dim_names=("dp_a", "dp_b", "dp_c"),
    )

    with pytest.raises(NotImplementedError, match="neither empty, a single axis, nor every axis"):
        placement_reduce_group(mesh, (Replicate(), Shard(0), Shard(0)))
