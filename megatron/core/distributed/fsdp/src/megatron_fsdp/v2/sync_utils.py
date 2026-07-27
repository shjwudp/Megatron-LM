# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared placement and stream helpers for parameter-group synchronization."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .param_group_state import Placements


def resolve_axis_streams(
    mesh_ndim: int,
    *,
    stream: torch.cuda.Stream | None = None,
    streams: Sequence[torch.cuda.Stream | None] | None = None,
) -> tuple[torch.cuda.Stream, ...]:
    """Resolve a shared stream or one stream per mesh axis."""
    if stream is not None and streams is not None:
        raise ValueError("Specify either stream or streams, not both")
    caller_stream = torch.cuda.current_stream()
    if streams is None:
        return (stream or caller_stream,) * mesh_ndim
    if len(streams) != mesh_ndim:
        raise ValueError(f"Expected {mesh_ndim} streams, got {len(streams)}")
    return tuple(axis_stream or caller_stream for axis_stream in streams)


def last_changed_axis(source: Placements, target: Placements) -> int | None:
    """Return the last changed axis in forward mesh order."""
    changed = [axis for axis, pair in enumerate(zip(source, target)) if pair[0] is not pair[1]]
    return changed[-1] if changed else None
