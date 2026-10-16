# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit scheduler identities and injected communication actions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Step:
    """A compute window selected by the existing pipeline or node scheduler."""

    chunk: int
    microbatch: int
    phase: Literal['forward', 'backward', 'weight_grad']


@dataclass(frozen=True)
class FsdpActions:
    """Runtime-owned actions; the adapters neither implement nor synchronize collectives.

    ``demand`` materializes a unit and makes its compute stream wait for it.
    ``prefetch`` issues a nonblocking, idempotent gather for a unit.
    ``release`` safely releases materialized parameters after their consumers.
    ``finish_backward`` finalizes a unit only after all gradient producers finish;
    the runtime callback owns reduction and any final release.
    """

    demand: Callable[[object], None]
    prefetch: Callable[[object], None]
    release: Callable[[object], None]
    finish_backward: Callable[[object], None]
