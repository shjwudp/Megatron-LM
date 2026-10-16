# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FSDP prefetch scoped to each conventional VPP chunk/microbatch pass."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from .common import FsdpActions, Step


@dataclass(frozen=True)
class Chunk:
    """Explicit unit occurrences for one chunk; backward need not reverse forward."""

    forward: tuple[object, ...]
    backward: tuple[object, ...]


class VppFsdpScheduler:
    """Consume VPP decisions without implementing pipeline scheduling or P2P.

    The outer scheduler selects each pass. Module callbacks call ``demand`` in
    that pass's order; a fresh cursor is local to the pass. Existing runtime
    callbacks retain responsibility for releases and gradient completion.
    """

    def __init__(self, chunks: Mapping[int, Chunk], actions: FsdpActions) -> None:
        self._chunks = dict(chunks)
        self._actions = actions
        self._active = False
        self._forward_only = False
        self._step: Step | None = None
        self._order: tuple[object, ...] = ()
        self._position = 0
        self._forward_done: set[tuple[int, int]] = set()
        self._backward_done: set[tuple[int, int]] = set()

    @contextmanager
    def iteration(self, *, forward_only: bool = False) -> Iterator[None]:
        """Scope the complete scheduler invocation, including warmup and cooldown."""
        if self._active:
            raise RuntimeError('A VPP iteration is already active.')
        self._active = True
        self._forward_only = forward_only
        self._forward_done.clear()
        self._backward_done.clear()
        try:
            yield
            if not forward_only and self._forward_done != self._backward_done:
                raise RuntimeError('VPP iteration ended with unfinished backward passes.')
        finally:
            self._active = False
            self._step = None
            self._order = ()

    @contextmanager
    def run_pass(self, step: Step) -> Iterator[None]:
        """Select one existing VPP decision; this is the only cursor-reset boundary."""
        if not self._active or self._step is not None:
            raise RuntimeError('VPP passes require an active iteration and cannot nest.')
        if step.phase not in ('forward', 'backward'):
            raise ValueError('Conventional VPP passes must be forward or backward.')
        if self._forward_only and step.phase == 'backward':
            raise RuntimeError('Backward is not allowed in a forward-only iteration.')
        identity = (step.chunk, step.microbatch)
        completed = self._forward_done if step.phase == 'forward' else self._backward_done
        if identity in completed:
            raise RuntimeError(f'Duplicate VPP pass: {step}.')
        if step.phase == 'backward' and identity not in self._forward_done:
            raise RuntimeError('A VPP backward pass requires its completed forward pass.')
        chunk = self._chunks[step.chunk]
        self._order = chunk.forward if step.phase == 'forward' else chunk.backward
        self._position = 0
        self._step = step
        try:
            yield
            if self._position != len(self._order):
                raise RuntimeError(f'VPP pass ended before all units were consumed: {step}.')
            completed.add(identity)
        finally:
            self._step = None
            self._order = ()

    def demand(self, unit: object) -> None:
        """Demand one occurrence and prefetch its immediate successor in this pass."""
        if self._step is None:
            raise RuntimeError('VPP demand requires an active pass.')
        if self._position >= len(self._order) or self._order[self._position] is not unit:
            raise RuntimeError(f'Unexpected FSDP unit in VPP pass {self._step}.')
        self._actions.demand(unit)
        self._position += 1
        if self._position < len(self._order):
            self._actions.prefetch(self._order[self._position])
