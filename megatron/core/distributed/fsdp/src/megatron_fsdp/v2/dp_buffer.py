# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from itertools import groupby
from typing import Iterable, Optional

import torch
from torch.distributed import _coalescing_manager
from torch.distributed.tensor import DeviceMesh

from .buffer_index import BufferIndex, Placement


class DataParallelBuffer:
    """Manage mesh-aware flat storage for tensor values.

    The buffer owns layout and placement transitions. Storage is allocated by
    its caller and attached with :meth:`bind`; allocator policy and temporary
    storage lifetimes belong to the owning ``ParameterGroup``.
    """

    def __init__(
        self,
        buffer_index: BufferIndex,
        dtype: torch.dtype,
        device: torch.device,
        mesh: DeviceMesh,
        placements: list[Placement],
    ):
        if len(placements) != mesh.ndim:
            raise ValueError(f"Expected {mesh.ndim} placements, got {placements}")
        self.buffer_index = buffer_index
        self.dtype = dtype
        self.device = device
        self.mesh = mesh
        self.placements = placements.copy()
        self.data_size = self.buffer_index._get_shard_meta(self.placements).size

        self.data: Optional[torch.Tensor] = None
        self._storage_owner: Optional["DataParallelBuffer"] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def bind(self, data: torch.Tensor) -> None:
        """Bind an externally allocated tensor to this placement-shaped buffer."""
        assert data.dtype == self.dtype, f"dtype mismatch: {data.dtype} vs {self.dtype}"
        assert data.numel() == self.data_size, f"size mismatch: {data.numel()} vs {self.data_size}"
        self.data = data

    def unbind(self) -> None:
        """Detach this buffer from storage without freeing the external allocation."""
        self.data = None
        self._storage_owner = None

    def placeholder(self, placements: list[Placement]) -> "DataParallelBuffer":
        """Return an unbound buffer with this layout and explicit placements."""
        if len(placements) != self.mesh.ndim:
            raise ValueError(f"Expected {self.mesh.ndim} placements, got {placements}")
        placeholder = copy(self)
        placeholder.data_size = self.buffer_index._get_shard_meta(placements).size
        placeholder.data = None
        placeholder.placements = placements.copy()
        placeholder._storage_owner = None
        return placeholder

    @torch.no_grad()
    def copy_tensors_(self, tensors: Iterable[torch.Tensor]) -> None:
        """Copy an ordered tensor sequence into this buffer in place.

        Args:
            tensors: One tensor per layout entry, in constructor order.

        Raises:
            RuntimeError: If no storage is bound.
            ValueError: If the tensor count does not match the layout.
        """
        if self.data is None:
            raise RuntimeError("DataParallelBuffer has no bound storage")
        expected_count = len(self.buffer_index.item_index_map)
        copied_count = 0
        for tensor_id, tensor in enumerate(tensors):
            if tensor_id >= expected_count:
                raise ValueError(f"Expected {expected_count} tensors, got more")
            source_slice, local_slice = self.buffer_index.local_slice_for(
                self.buffer_index._get_item_global_range(tensor_id),
                self.placements,
                self.placements,
            )
            if source_slice is not None and local_slice is not None:
                self.data[local_slice].copy_(tensor.flatten()[source_slice])
            copied_count += 1
        if copied_count != expected_count:
            raise ValueError(f"Expected {expected_count} tensors, got {copied_count}")

    def tensor_view(self, tensor_id: int) -> torch.Tensor:
        """Return this rank's local view of one tensor.

        Args:
            tensor_id: Tensor index in constructor order.

        Returns:
            A local ``torch.Tensor`` view, which may be empty on this rank.
        """
        if self.data is None:
            raise RuntimeError("DataParallelBuffer has no bound storage")
        _, local_slice = self.buffer_index.local_slice_for(
            self.buffer_index._get_item_global_range(tensor_id), self.placements, self.placements
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def is_unsharded(self) -> bool:
        """Return whether this buffer currently has a full unsharded view."""
        return all(placement is Placement.REPLICATE for placement in self.placements)

    def view(self, placements: list[Placement]) -> "DataParallelBuffer":
        """Return a non-owning DP-buffer view with explicit placements.

        The returned buffer has exact placement-shaped ``data``. For example,
        taking a ``SHARD`` view of a ``REPLICATE`` placeholder returns the
        rank-owned slice while retaining the placeholder as its storage owner.

        Args:
            placements: Logical placement for each mesh dimension.

        Returns:
            A DP buffer whose data is the requested view of this buffer's storage.
        """
        if len(placements) != self.mesh.ndim:
            raise ValueError(f"Expected {self.mesh.ndim} placements, got {placements}")
        view = self.placeholder(placements)
        view.data = self._bound_view(placements)
        view.data_size = view.data.numel()
        view._storage_owner = self
        return view

    def reinterpret(self, placements: list[Placement]) -> "DataParallelBuffer":
        """Return an alias with same-sized storage and different validity placements.

        This is used to mark a replicated gradient contribution as ``PARTIAL``;
        it performs no communication and never allocates.
        """
        alias = self.placeholder(placements)
        if self.data is None:
            raise RuntimeError("DataParallelBuffer has no bound storage")
        if alias.data_size != self.data.numel():
            raise ValueError(
                f"Cannot reinterpret {self.data.numel()} elements as {placements} "
                f"({alias.data_size} elements)"
            )
        alias.data = self.data
        alias._storage_owner = self
        return alias

    @staticmethod
    @torch.no_grad()
    def redistribute_buffers(
        buffers: list["DataParallelBuffer"],
        target_placements: list[Placement],
        *,
        output_buffers: list["DataParallelBuffer"],
        stream: torch.cuda.Stream | None = None,
        async_op: bool = False,
    ) -> list["DataParallelBuffer"]:
        """Redistribute compatible buffers to one target placement vector.

        The target defines the mesh-axis plan. Each axis is completed across
        communication-compatible buffers before the next axis starts, preserving
        HSDP outer-then-inner ordering and cross-buffer collective coalescing.
        """
        if not buffers:
            return []
        if any(len(target_placements) != buffer.mesh.ndim for buffer in buffers):
            raise ValueError(
                f"Expected {buffers[0].mesh.ndim} target placements, got {target_placements}"
            )
        if len(output_buffers) != len(buffers):
            raise ValueError(f"Expected {len(buffers)} output buffers, got {len(output_buffers)}")

        caller_stream = torch.cuda.current_stream()
        stream = stream or caller_stream

        for buffer, output_buffer in zip(buffers, output_buffers):
            buffer._validate_output_buffer(output_buffer, target_placements)
        if stream != caller_stream:
            stream.wait_stream(caller_stream)

        def compatibility_key(
            item: tuple["DataParallelBuffer", "DataParallelBuffer"], mesh_dim: int
        ):
            buffer, _ = item
            group = buffer.mesh.get_group(mesh_dim=mesh_dim)
            return id(group), buffer.dtype, buffer.device, buffer.placements[mesh_dim]

        current_buffers = list(buffers)
        with torch.cuda.stream(stream):
            for mesh_dim, target in enumerate(target_placements):
                axis_transitions = []
                next_buffers = []
                for buffer, final_output in zip(current_buffers, output_buffers):
                    if buffer.placements[mesh_dim] is target:
                        next_buffers.append(buffer)
                        continue
                    axis_target = buffer.placements.copy()
                    axis_target[mesh_dim] = target
                    if axis_target == target_placements:
                        axis_output = final_output
                    elif (
                        buffer._storage_owner is not None
                        and buffer._storage_owner.placements == axis_target
                    ):
                        # Prefer the containing placement view when one exists.
                        # For [S, S] -> [R, S] -> [R, R], this refreshes the
                        # persistent [R, S] owner before the final all-gather.
                        axis_output = buffer._storage_owner
                    else:
                        axis_output = final_output.view(axis_target)
                    axis_transitions.append((buffer, axis_output))
                    next_buffers.append(axis_output)

                for _, compatible_items_iter in groupby(
                    axis_transitions, key=lambda item: compatibility_key(item, mesh_dim)
                ):
                    compatible_items = list(compatible_items_iter)
                    group = compatible_items[0][0].mesh.get_group(mesh_dim=mesh_dim)
                    context = (
                        _coalescing_manager(group, async_ops=async_op)
                        if len(compatible_items) > 1 and torch.distributed.get_world_size(group) > 1
                        else nullcontext()
                    )
                    with context as coalescing_event:
                        for buffer, axis_output in compatible_items:
                            buffer.redistribute(axis_output.placements, output_buffer=axis_output)
                    if async_op and coalescing_event is not None:
                        coalescing_event.wait()
                current_buffers = next_buffers

            for current, final_output in zip(current_buffers, output_buffers):
                if current is not final_output:
                    current.redistribute(target_placements, output_buffer=final_output)

        return output_buffers

    @torch.no_grad()
    def redistribute(
        self,
        target_placements: list[Placement],
        *,
        output_buffer: "DataParallelBuffer" | None = None,
    ) -> "DataParallelBuffer":
        """Redistribute one mesh axis and return the destination DP buffer.

        ``output_buffer`` may name an explicit destination. This permits an
        in-place all-gather whose ``REPLICATE`` output owns full storage while
        this ``SHARD`` input is a slice sharing that same storage.

        Dtype conversion, gradient scaling, accumulation, and temporary
        communication storage are intentionally outside this abstraction.
        """
        assert len(target_placements) == 2

        changed_axis = None
        for axis, (source, target) in enumerate(zip(self.placements, target_placements)):
            if source == target:
                continue
            if changed_axis is not None:
                raise ValueError(
                    "redistribute supports changing only one placement axis per call: "
                    f"{self.placements} -> {target_placements}"
                )
            changed_axis = axis
        if changed_axis is None:
            if output_buffer is None:
                return self
            self._validate_output_buffer(output_buffer, target_placements)
            if self.data is None:
                raise RuntimeError("Redistribution requires a bound input buffer")
            if output_buffer.data.data_ptr() != self.data.data_ptr():
                output_buffer.data.copy_(self.data)
            return output_buffer

        source = self.placements[changed_axis]
        target = target_placements[changed_axis]
        group = self.mesh.get_group(mesh_dim=changed_axis)

        if source is Placement.REPLICATE and target is Placement.SHARD:
            if output_buffer is None:
                output_buffer = self.view(target_placements)
            else:
                self._validate_output_buffer(output_buffer, target_placements)
        elif target is Placement.PARTIAL:
            if output_buffer is None:
                output_buffer = self.reinterpret(target_placements)
            else:
                self._validate_output_buffer(output_buffer, target_placements)
        else:
            if output_buffer is None:
                raise ValueError(
                    f"Redistribution {self.placements} -> {target_placements} "
                    "requires an externally bound output buffer"
                )
            self._validate_output_buffer(output_buffer, target_placements)
        input_data = self.data
        output_data = output_buffer.data
        if input_data is None or output_data is None:
            raise RuntimeError("Redistribution requires bound input and output buffers")

        if source is Placement.SHARD and target is Placement.REPLICATE:
            torch.distributed.all_gather_into_tensor(output_data, input_data, group=group)
        elif source is Placement.PARTIAL:
            if target is Placement.REPLICATE:
                if output_data.data_ptr() != input_data.data_ptr():
                    output_data.copy_(input_data)
                torch.distributed.all_reduce(output_data, group=group)
            else:
                torch.distributed.reduce_scatter_tensor(
                    output=output_data, input=input_data, group=group
                )
        elif source is Placement.REPLICATE and target is Placement.SHARD:
            local_input = self._bound_view(target_placements)
            if output_data.data_ptr() != local_input.data_ptr():
                output_data.copy_(local_input)
        elif target is Placement.PARTIAL:
            if output_data.data_ptr() != input_data.data_ptr():
                output_data.copy_(input_data)
        else:
            raise NotImplementedError(f"Unsupported placement transition: {source!r} -> {target!r}")

        return output_buffer

    def _validate_output_buffer(
        self, output_buffer: "DataParallelBuffer", target_placements: list[Placement]
    ) -> None:
        """Validate an explicit redistribution destination."""
        if output_buffer.buffer_index is not self.buffer_index:
            raise ValueError("Redistribution output must share the input buffer layout")
        if output_buffer.mesh is not self.mesh:
            raise ValueError("Redistribution output must share the input buffer mesh")
        if output_buffer.dtype != self.dtype or output_buffer.device != self.device:
            raise ValueError("Redistribution output must share the input dtype and device")
        if output_buffer.placements != target_placements:
            raise ValueError(
                f"Output placements {output_buffer.placements} do not match target "
                f"{target_placements}"
            )
        expected_size = self.buffer_index._get_shard_meta(target_placements).size
        if output_buffer.data is None or output_buffer.data.numel() != expected_size:
            raise ValueError(
                f"Output buffer size does not match target: expected {expected_size}, "
                f"got {None if output_buffer.data is None else output_buffer.data.numel()}"
            )

    def _bound_view(self, placements: list[Placement]) -> torch.Tensor:
        """Return a placement-shaped view when the bound storage contains it."""
        if self.data is None:
            raise RuntimeError("DataParallelBuffer has no bound storage")
        if placements == self.placements:
            return self.data

        data_contains_requested = all(
            storage_placement is requested_placement
            or (storage_placement is Placement.REPLICATE and requested_placement is Placement.SHARD)
            for storage_placement, requested_placement in zip(self.placements, placements)
        )
        if data_contains_requested:
            _, local_slice = self.buffer_index.local_slice_for(
                (0, self.buffer_index.bucket_meta.size), placements, self.placements
            )
            return self.data[:0] if local_slice is None else self.data[local_slice]
        raise ValueError(
            f"Buffer placements {self.placements} do not contain {placements}; "
            "bind an externally allocated output buffer"
        )
