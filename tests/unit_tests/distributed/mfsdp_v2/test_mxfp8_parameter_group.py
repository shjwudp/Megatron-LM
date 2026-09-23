# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MXFP8 payload storage and orientation-selective materialization in ``FsdpParameterGroup``.

Quantized groups rest their fp8 payloads in ``QuantizedDBuffer`` planes and
gather only the payload orientations a pass requests (row-wise forward,
column-wise backward), rebinding them onto the stable parameter tensors through
``get_tensor`` and ``clear_payloads``. TE's ``MXFP8Tensor`` is not needed for
the placement bookkeeping under test; the quantization call itself is covered
by ``test_quantized_dbuffer``.
"""

import types

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.layout import GlobalLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    BOTH,
    COLWISE,
    ROWWISE,
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    Flat,
    changed_mesh_axis,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    QuantizedDBuffer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.orthogonalized_optimizer import (
    _compute_weight_local_views,
)

TENSOR_SHAPES = (torch.Size((256, 128)),)


class _FakeQuantizer:
    """The ``Quantizer`` surface the unshard path touches, without TE."""

    def __init__(self):
        self.rowwise_usage = True
        self.columnwise_usage = True

    def set_usage(self, rowwise=None, columnwise=None):
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise


class _FakeFp8Tensor:
    """The stable ``MXFP8Tensor`` surface the unshard path rebinds, without TE.

    ``.data =`` receives the wrapper built by ``QuantizedDBuffer.get_tensor``
    and ``_quantizer`` carries the usage flags TE propagates into
    ``update_usage``. ``_rowwise_data``/``_columnwise_data`` are the raw payload
    slots ``clear_payloads`` empties.
    """

    def __init__(self, shape):
        self.shape = torch.Size(shape)
        self.data = None
        self._rowwise_data = None
        self._columnwise_data = None
        self._quantizer = _FakeQuantizer()


def _wire_quantized_group(mesh, parameter_placements, optimizer_placements, device):
    """Wire the three quantized buffers exactly as ``_initialize_buffers`` does."""
    layout = GlobalLayout.build(TENSOR_SHAPES, dp_size=mesh.size(), block_size=32)
    model_weight = QuantizedDBuffer(mesh, parameter_placements, layout, device)
    group = object.__new__(FsdpParameterGroup)
    group.mesh = mesh
    group._symm_mem_pool = None
    group.model_weight = model_weight
    group.post_optimizer_model_weight = model_weight.view(optimizer_placements)
    group._unsharded_model_weight = QuantizedDBuffer(
        mesh, (Replicate(),) * mesh.ndim, layout, device
    )
    group._rowwise_is_stale = False
    group._colwise_is_stale = False
    group._materialized_directions = frozenset()
    group.fsdp_parameters = tuple(
        types.SimpleNamespace(fqns=(f"weight{index}",), unsharded=_FakeFp8Tensor(shape))
        for index, shape in enumerate(TENSOR_SHAPES)
    )
    return group


def test_unshard_materializes_only_requested_orientations(distributed_setup, monkeypatch):
    """Only the requested payload orientations are gathered, bound, and usage-flagged.

    Widening a residency window (a recomputed forward's backward) gathers only
    the missing direction and rebinds the full materialized set -- including the
    scale grids TE drops from disabled directions -- and releasing the storage
    drops the bindings and forgets the materialization so the next cycle
    re-gathers.
    """
    if distributed_setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    group = _wire_quantized_group(mesh, (Replicate(),), (Flat(),), distributed_setup.device)

    gathered = []
    monkeypatch.setattr(
        FsdpParameterGroup,
        "_gather_payload",
        lambda self, source, target: gathered.append(
            ROWWISE if source[0] is self.model_weight.rowwise_data else COLWISE
        ),
    )
    monkeypatch.setattr(FsdpParameterGroup, "_switch_to_unsharded_parameters", lambda self: None)

    group.unshard_parameters(ROWWISE)
    assert gathered == [ROWWISE]
    assert group._materialized_directions == frozenset((ROWWISE,))
    tensor = group.fsdp_parameters[0].unsharded
    bound = tensor.data
    assert bound._rowwise_data is not None
    assert bound._rowwise_scale_inv is not None
    assert bound._columnwise_data is None
    assert bound._columnwise_scale_inv is None
    assert tensor._quantizer.rowwise_usage is True
    assert tensor._quantizer.columnwise_usage is False

    # A wider request gathers only the direction still missing and rebinds every
    # materialized direction's payloads and scale grids at once.
    group.unshard_parameters(BOTH)
    assert gathered == [ROWWISE, COLWISE]
    assert group._materialized_directions == frozenset((ROWWISE, COLWISE))
    bound = tensor.data
    assert bound._rowwise_data is not None
    assert bound._columnwise_data is not None
    assert bound._rowwise_scale_inv is not None
    assert bound._columnwise_scale_inv is not None
    assert tensor._quantizer.rowwise_usage is True
    assert tensor._quantizer.columnwise_usage is True

    # Release detaches the raw payloads and forgets what was materialized.
    tensor._rowwise_data = torch.zeros(4)
    tensor._columnwise_data = torch.zeros(4)
    group.release_unsharded_storage()
    assert tensor._rowwise_data is None
    assert tensor._columnwise_data is None
    assert group._materialized_directions == frozenset()


def test_unshard_moves_change_at_most_one_mesh_axis(distributed_setup, monkeypatch):
    """Every view/redistribute on the unshard path changes at most one mesh axis.

    ``changed_mesh_axis`` raises when two axes change, so recording its result
    on each real call proves the alignment kept the single-axis ``DBuffer``
    machinery sufficient for HFSDP dense: both stale orientations first move
    their optimizer-view planes into plane storage (outer axis, two planes per
    direction), then gather storage into the replicated unsharded planes (inner
    axis). Only placement bookkeeping is under test, so the collective is
    stubbed.
    """
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2:
        pytest.skip("2D placement test requires an even world size of at least 4.")
    mesh = init_device_mesh(
        distributed_setup.device.type,
        (2, distributed_setup.world_size // 2),
        mesh_dim_names=("dp_outer", "dp_shard"),
    )
    group = _wire_quantized_group(
        mesh, (Replicate(), Flat()), (Flat(), Flat()), distributed_setup.device
    )
    group._rowwise_is_stale = True
    group._colwise_is_stale = True
    monkeypatch.setattr(FsdpParameterGroup, "_switch_to_unsharded_parameters", lambda self: None)

    recorded_axes = []
    original_view = DBuffer.view
    original_redistribute = DBuffer.redistribute

    def recording_view(self, new_placements):
        recorded_axes.append(changed_mesh_axis(self.placements, new_placements))
        return original_view(self, new_placements)

    def recording_redistribute(self, new_placements, *, out=None):
        recorded_axes.append(changed_mesh_axis(self.placements, new_placements))
        return original_redistribute(self, new_placements, out=out)

    monkeypatch.setattr(DBuffer, "view", recording_view)
    monkeypatch.setattr(DBuffer, "redistribute", recording_redistribute)

    def fake_all_gather_into_tensor(output_tensor, input_tensor, group):
        chunk = input_tensor.numel()
        for index in range(output_tensor.numel() // chunk):
            output_tensor.narrow(0, index * chunk, chunk).copy_(input_tensor)

    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)

    group.unshard_parameters(BOTH)

    # changed_mesh_axis would have raised inside the recorders on any two-axis move.
    assert all(axis in (0, 1, None) for axis in recorded_axes), recorded_axes
    assert recorded_axes == [0, 0, 0, 0, 1, 1, 1, 1], recorded_axes
    # Both requested orientations were redistributed and gathered, so neither
    # waits for its own unshard anymore.
    assert group._rowwise_is_stale is False
    assert group._colwise_is_stale is False


def test_optimizer_layout_views_alias_storage_and_gate_staleness(distributed_setup):
    """The ``post_optimizer_*`` views alias plane storage and define staleness.

    Quantization writes land in the views (main-weight shaped under expert
    ZeRO-1 and HFSDP, unlike the coarser parameter-layout storage), and Muon's
    local-shape check must read those views -- never the storage. ZeRO-3 keeps
    both on one placement, so the view is the storage itself and can never go
    stale.
    """
    if distributed_setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    device = distributed_setup.device
    layout = GlobalLayout.build(TENSOR_SHAPES, dp_size=mesh.size(), block_size=32)
    storage = QuantizedDBuffer(mesh, (Replicate(),), layout, device)
    view = storage.view((Flat(),))
    assert view is not storage

    # The view aliases the storage allocation at exactly the optimizer-layout
    # local offset, so quantization writes land inside the plane storage.
    plane, view_plane = storage.rowwise_data, view.rowwise_data
    expected_offset, expected_numel = plane.layout.get_local_range(mesh, (Flat(),))
    assert view_plane.offset == expected_offset
    assert view_plane.local_buffer.numel() == expected_numel
    assert view_plane.local_buffer.data_ptr() == (
        plane.local_buffer.data_ptr()
        + (expected_offset - plane.offset) * view_plane.local_buffer.element_size()
    )
    # Shard-shaped compute-weight view into a coarser storage shard.
    assert view_plane.get_tensor_view(0).shape != plane.get_tensor_view(0).shape

    # Staleness is exactly "view placements != plane placements".
    group = object.__new__(FsdpParameterGroup)
    group.model_weight = storage
    group.post_optimizer_model_weight = view
    FsdpParameterGroup._update_payload_staleness(group)
    assert group._rowwise_is_stale is True
    assert group._colwise_is_stale is True

    # Muon's shard-shape check reads the compute-weight views, not the storage.
    views = dict(_compute_weight_local_views(group))
    assert list(views) == [
        "post_optimizer_model_weight",
        "post_optimizer_rowwise",
        "post_optimizer_colwise",
    ]
    assert views["post_optimizer_rowwise"] == (view.rowwise_data, view.rowwise_scale)
    assert views["post_optimizer_colwise"] == (view.columnwise_data, view.columnwise_scale)

    # ZeRO-3: storage and view share placements, so the view is the storage.
    zero3_storage = QuantizedDBuffer(mesh, (Flat(),), layout, device)
    group.model_weight = zero3_storage
    group.post_optimizer_model_weight = zero3_storage.view((Flat(),))
    assert group.post_optimizer_model_weight is zero3_storage
    FsdpParameterGroup._update_payload_staleness(group)
    assert group._rowwise_is_stale is False
    assert group._colwise_is_stale is False


def test_subgroup_layout_with_mxfp8_blocks_raises():
    """``subgroup_size`` with MXFP8 block placement is a preserved known limitation.

    Subgroup-local packing pads to row granularity only, so with block-atomic
    (32) placement some tensor offsets land off block alignment and
    ``GlobalLayout``'s block-alignment check raises. Either feature alone packs
    fine; together they are not a supported configuration (see the PORT-NOTE at
    ``layout._build_subgroup_layout``).
    """
    shapes = (torch.Size((32, 96)), torch.Size((32, 64)))
    GlobalLayout.build(shapes, dp_size=2, block_size=32)
    GlobalLayout.build(shapes, dp_size=2, block_size=1, subgroup_size=2)
    with pytest.raises(AssertionError, match="not aligned to block size"):
        GlobalLayout.build(shapes, dp_size=2, block_size=32, subgroup_size=2)
