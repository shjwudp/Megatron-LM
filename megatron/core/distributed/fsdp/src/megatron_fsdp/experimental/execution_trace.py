# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Scheduler-independent recording of FSDP parameter demand and release events."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar
from weakref import ref

T = TypeVar('T')
Operation = Literal['unshard', 'reshard']


@dataclass(frozen=True)
class ExecutionStep:
    """Optional scheduler identity; a backward step can also describe delayed wgrad."""

    chunk: int
    microbatch: int
    phase: Literal['forward', 'backward', 'weight_grad']
    node: str | None = None


@dataclass(frozen=True)
class ExecutionEvent(Generic[T]):
    """One weakly owned parameter-lifecycle occurrence in actual launch order."""

    module_ref: ref[T]
    operation: Operation
    phase: str | None
    step: ExecutionStep | None

    @property
    def module(self) -> T:
        """Resolve the module without silently skipping deleted occurrences."""
        module = self.module_ref()
        if module is None:
            raise RuntimeError('A recorded FSDP module has been deleted.')
        return module


class ExecutionTrace(Generic[T]):
    """Record one complete schedule and replay its interleaved parameter lifetimes.

    This class does not execute work or synchronize streams. Schedulers own the
    iteration boundary and optional step labels; FSDP owns demand/release events.
    Changing the caller's key replaces the cached pattern after successful capture.
    """

    def __init__(self) -> None:
        """Start with an inactive, empty trace and no cached schedule pattern."""
        self.events: tuple[ExecutionEvent[T], ...] = ()
        self.active = False
        self.recording = False
        self._position = 0
        self._key: object = object()
        self._pending: list[ExecutionEvent[T]] = []
        self._step: ExecutionStep | None = None

    @contextmanager
    def iteration(self, *, key: object = None) -> Iterator[None]:
        """Capture a new pattern, or reset and completely consume the cached one.

        Use different keys for train/eval or changed microbatch/chunk schedules.
        Capture disables speculative prefetch. Failed capture retains the previous
        cache; failed replay is an error, not a request to silently re-record.
        """
        if self.active:
            raise RuntimeError('An execution-trace iteration is already active.')
        self.active = True
        self.recording = not self.events or key != self._key
        self._position = 0
        self._pending = []
        try:
            yield
            if self.recording:
                if not self._pending:
                    raise RuntimeError('An execution-trace recording must observe FSDP calls.')
                self.events = tuple(self._pending)
                self._key = key
            elif self._position != len(self.events):
                raise RuntimeError(
                    'Execution-trace iteration ended before all events were consumed.'
                )
        finally:
            self.active = False
            self.recording = False
            self._pending = []
            self._step = None

    @contextmanager
    def step(self, identity: ExecutionStep) -> Iterator[None]:
        """Label work explicitly at a scheduler or fine-grained node boundary."""
        if not self.active:
            raise RuntimeError('Execution steps require an active iteration.')
        previous = self._step
        self._step = identity
        try:
            yield
        finally:
            self._step = previous

    def observe(self, module: T, operation: Operation, *, phase: str | None = None) -> None:
        """Record or validate a demand/release occurrence before performing it."""
        if not self.active:
            return
        if self.recording:
            self._pending.append(ExecutionEvent(ref(module), operation, phase, self._step))
            return
        if self._position >= len(self.events):
            raise RuntimeError('FSDP execution trace exhausted before the schedule finished.')
        expected = self.events[self._position]
        if (
            expected.module is not module
            or expected.operation != operation
            or expected.phase != phase
            or expected.step != self._step
        ):
            raise RuntimeError(
                f'FSDP execution diverged at event {self._position}: '
                f'expected {expected.operation}/{expected.phase} in {expected.step}, '
                f'observed {operation}/{phase} in {self._step}.'
            )
        self._position += 1

    def prefetch_candidates(self) -> Iterator[T]:
        """Yield future demands whose parameters will not be released before use.

        An intervening release blocks that module for this lookahead. In particular,
        separate forward/backward lists cannot express this cross-phase boundary.
        Candidates are unique even when several submodule hooks demand one owner.
        """
        if not self.active or self.recording:
            return
        blocked: set[int] = set()
        seen: set[int] = set()
        for index in range(self._position, len(self.events)):
            event = self.events[index]
            module = event.module
            identity = id(module)
            if event.operation == 'reshard':
                blocked.add(identity)
            elif identity not in blocked and identity not in seen:
                seen.add(identity)
                yield module
