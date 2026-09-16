# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DBuffer placement definitions.

DBuffer uses PyTorch DTensor's ``Placement``, ``Replicate``, and ``Partial``
types directly. ``Flat`` and ``BlockAtomic`` are DBuffer-specific dim-0
``Shard`` placements whose local storage is part of one flattened buffer.

=============  =============  ====================
Source         Destination    DBuffer operation
=============  =============  ====================
sharded        ``Replicate``  ``allgather()``
``Partial``    sharded        ``reduce_scatter()``
``Partial``    ``Replicate``  ``allreduce()``
``Replicate``  sharded        ``view()`` (local)
=============  =============  ====================

Every operation above changes exactly one mesh axis, so DBuffer composes them
one axis at a time. The helpers here answer the placement questions that
composition and its callers have to ask: which axes changed, which axes are
sharded, and which process group spans every shard of a buffer.
"""

from collections.abc import Iterable

import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.placement_types import Placement

__all__ = [
    "BlockAtomic",
    "Flat",
    "changed_mesh_axis",
    "changed_mesh_axes",
    "flattened_mesh_group",
    "placement_reduce_group",
    "sharded_mesh_axes",
]


class Flat(Shard):
    """DBuffer-specific flattened dim-0 shard placement."""

    def __init__(self) -> None:
        super().__init__(0)

    def __eq__(self, other: object) -> bool:
        # PyTorch Shard.__eq__ compares only dim, so distinguish Flat from BlockAtomic.
        return isinstance(other, Shard) and other.dim == 0 and not isinstance(other, BlockAtomic)


class BlockAtomic(Shard):
    """Flattened dim-0 shard placement that keeps ``block_size`` rows together."""

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"BlockAtomic block_size must be positive, got {block_size}.")
        super().__init__(0)
        self.block_size = block_size

    def __eq__(self, other: object) -> bool:
        # PyTorch Shard.__eq__ compares only dim, so preserve the block size as well.
        return isinstance(other, BlockAtomic) and self.block_size == other.block_size

    def __repr__(self) -> str:
        return f"BlockAtomic(block_size={self.block_size})"


def changed_mesh_axes(
    old_placements: Iterable[Placement], new_placements: Iterable[Placement]
) -> tuple[int, ...]:
    """Return every changed mesh axis, in ascending order.

    Callers that can apply one placement change at a time use this to plan the
    changes; callers that genuinely require a single changed axis keep using
    :func:`changed_mesh_axis`.
    """
    return tuple(
        axis
        for axis, (old_placement, new_placement) in enumerate(
            zip(old_placements, new_placements, strict=True)
        )
        if old_placement != new_placement
    )


def changed_mesh_axis(
    old_placements: Iterable[Placement], new_placements: Iterable[Placement]
) -> int | None:
    """Return the changed mesh axis, requiring at most one placement change."""
    changed_axis = None
    for axis, (old_placement, new_placement) in enumerate(
        zip(old_placements, new_placements, strict=True)
    ):
        if old_placement == new_placement:
            continue
        if changed_axis is not None:
            raise NotImplementedError(
                "Expected at most one changed placement axis, "
                f"got changed axes {changed_axis} and {axis}."
            )
        changed_axis = axis
    return changed_axis


def sharded_mesh_axes(placements: Iterable[Placement]) -> tuple[int, ...]:
    """Return the mesh axes that hold a disjoint shard of one buffer.

    Only a ``Shard`` placement splits one buffer into rank-disjoint blocks: a
    ``Replicate`` axis holds an identical copy of the whole buffer, and a
    ``Partial`` axis holds a full-size unreduced contribution. Neither of those
    adds blocks that a reduction over the buffer's shards would miss, and a
    ``Partial`` axis in particular cannot be reconstructed by any reduction
    group, so an unexpected placement is rejected rather than silently treated
    as sharded.
    """
    sharded_axes: list[int] = []
    for axis, placement in enumerate(placements):
        if isinstance(placement, Shard):
            sharded_axes.append(axis)
        elif not isinstance(placement, Replicate):
            raise NotImplementedError(
                f"Unsupported placement {placement!r} on mesh axis {axis}: expected Shard "
                "or Replicate, got a placement that neither shards the buffer into "
                "disjoint blocks nor replicates it."
            )
    return tuple(sharded_axes)


def flattened_mesh_group(mesh: DeviceMesh) -> dist.ProcessGroup:
    """Return the one process group spanning every axis of ``mesh``.

    ``DeviceMesh`` cannot recover the union of its axes when it was built from
    per-axis groups, so callers that own the flattened group record it on the
    mesh; otherwise derive it from the mesh itself.
    """
    flattened_group = getattr(mesh, "_mfsdp_flattened_group", None)
    if flattened_group is not None:
        return flattened_group
    return mesh._flatten().get_group()


def placement_reduce_group(mesh: DeviceMesh, placements: Iterable[Placement]) -> dist.ProcessGroup:
    """Return a group spanning every sharded axis of ``placements``.

    A reduction whose result must see the whole buffer -- such as the amax that
    Transformer Engine MAX-reduces while quantizing MXFP8 master weights --
    has to span every axis the buffer is sharded over, because each rank of
    such an axis owns a disjoint block. With more than one sharded axis that is
    the flattened group over all of them: no single mesh axis sees every block.

    With no sharded axis the buffer is fully replicated, so every rank already
    holds an identical copy and any group gives the same, idempotent result.
    This mesh's axis 0 is returned for that case. The default process group
    must NOT be used: it spans unrelated PP/TP ranks holding different
    parameters and would silently corrupt the result.
    """
    sharded_axes = sharded_mesh_axes(placements)
    if not sharded_axes:
        return mesh.get_group(0)
    if len(sharded_axes) == 1:
        return mesh.get_group(sharded_axes[0])
    if len(sharded_axes) == mesh.ndim:
        return flattened_mesh_group(mesh)
    raise NotImplementedError(
        f"Cannot reduce over sharded mesh axes {sharded_axes} of a {mesh.ndim}-dimensional "
        "mesh: they are neither empty, a single axis, nor every axis, so no existing "
        "process group spans exactly their shards."
    )
