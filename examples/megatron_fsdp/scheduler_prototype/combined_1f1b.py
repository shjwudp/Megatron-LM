# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit FSDP actions at combined 1F1B compute-node boundaries."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .common import FsdpActions, Step


@dataclass(frozen=True)
class Node:
    """Communication actions supplied by a fine-grained compute plan.

    ``finish_backward`` is opt-in, after all gradient producers (including delayed
    wgrad). It must not list the same unit in ``release``: the finalizer owns its
    final release. No parameter-readiness decision is inferred from node counts.
    """

    step: Step
    name: str
    demand: tuple[object, ...] = ()
    prefetch: tuple[object, ...] = ()
    release: tuple[object, ...] = ()
    finish_backward: tuple[object, ...] = ()


class Combined1f1bFsdpScheduler:
    """Apply explicit node actions; the existing compute plan chooses node order."""

    def __init__(self, actions: FsdpActions) -> None:
        self._actions = actions
        self._active = False
        self._in_node = False
        self._pending: dict[tuple[int, int, int], object] = {}

    @contextmanager
    def iteration(self) -> Iterator[None]:
        """Include all fine-grained nodes, releases, and delayed weight gradients."""
        if self._active:
            raise RuntimeError('A combined iteration is already active.')
        self._active = True
        self._pending.clear()
        try:
            yield
            if self._pending:
                raise RuntimeError('Combined iteration ended before backward finalization.')
        finally:
            self._active = False
            self._in_node = False
            self._pending.clear()

    @contextmanager
    def run_node(self, node: Node) -> Iterator[None]:
        """Demand before compute and release/finalize only after successful compute."""
        if not self._active or self._in_node:
            raise RuntimeError('Combined nodes require an active iteration and cannot nest.')
        if node.finish_backward and node.step.phase == 'forward':
            raise ValueError('A forward node cannot finalize backward.')
        if {id(unit) for unit in node.release} & {id(unit) for unit in node.finish_backward}:
            raise ValueError('The backward finalizer owns its final parameter release.')
        prefix = (node.step.chunk, node.step.microbatch)
        demand_ids = {id(unit) for unit in node.demand}
        for unit in node.demand:
            if node.step.phase != 'forward' and any(
                key[2] == id(unit) and key[:2] != prefix for key in self._pending
            ):
                raise RuntimeError(
                    'Finalize the previous backward window before reusing this unit.'
                )
        if len({id(unit) for unit in node.finish_backward}) != len(node.finish_backward):
            raise ValueError('A node cannot finalize the same unit twice.')
        for unit in node.finish_backward:
            if (*prefix, id(unit)) not in self._pending and id(unit) not in demand_ids:
                raise RuntimeError('Backward finalization requires a matching demand window.')
        self._in_node = True
        try:
            for unit in node.demand:
                self._actions.demand(unit)
                if node.step.phase != 'forward':
                    self._pending[(*prefix, id(unit))] = unit
            for unit in node.prefetch:
                self._actions.prefetch(unit)
            yield
            for unit in node.release:
                self._actions.release(unit)
            for unit in node.finish_backward:
                key = (*prefix, id(unit))
                if key not in self._pending:
                    raise RuntimeError('Backward finalization requires a matching demand window.')
                self._actions.finish_backward(unit)
                del self._pending[key]
        finally:
            self._in_node = False
