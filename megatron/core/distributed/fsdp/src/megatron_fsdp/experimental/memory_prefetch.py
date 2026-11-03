# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compile occurrence-specific prefetches from a recorded memory profile."""

from typing import NamedTuple

from .execution_order import ExecutionEvent, Operation
from .phase import Phase


class MemorySnapshot(NamedTuple):
    """Allocator and device memory at one execution-order occurrence, in bytes."""

    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    total_bytes: int


class PrefetchMemory(NamedTuple):
    """Conservative materialization costs and the existing per-module limits."""

    buffer_bytes: int
    parameter_elements: int
    shared_phases: bool
    backward_all_planes: bool
    forward_limit: int | None
    backward_limit: int | None


def compile_prefetch_plan(
    events: list[ExecutionEvent],
    headroom: list[int],
    metadata: dict[object, PrefetchMemory],
    retain_weights: bool,
) -> tuple[tuple[int, ...], ...]:
    """Choose a prefix at unshard and completed backward-reshard occurrences.

    All inputs must agree across the scheduling process group. Headroom has already
    reserved memory for untracked allocations. A pending prefetch reserves its full
    materialization cost until demand, even if the baseline also prefetched it.
    This deliberately overestimates extra residency instead of assuming that the
    allocator can immediately reuse cached or asynchronously released storage.
    """
    if len(events) != len(headroom):
        raise ValueError("Every execution event must have a recorded memory budget.")
    plan: list[tuple[int, ...]] = []
    active = set()
    ready = {module: set() for module in metadata}
    pending: dict[object, int] = {}
    pending_bytes = 0

    def mark_ready(event: ExecutionEvent) -> None:
        info = metadata[event.module]
        if info.shared_phases or (event.phase is Phase.BACKWARD and info.backward_all_planes):
            ready[event.module].update((Phase.FORWARD, Phase.BACKWARD))
        else:
            ready[event.module].add(event.phase)

    for position, event in enumerate(events):
        targets = []
        module = event.module
        if event.operation is Operation.RESHARD:
            following = events[position + 1] if position + 1 < len(events) else None
            retaining = (
                retain_weights
                and following is not None
                and following.module is module
                and following.operation is Operation.UNSHARD
            )
            if not retaining:
                ready[module].clear()
            active.discard(module)
            if event.phase is not Phase.BACKWARD:
                plan.append(())
                continue
        else:
            active.add(module)
            pending_bytes -= pending.pop(module, 0)
            mark_ready(event)
        info = metadata[module]
        limit = info.forward_limit if event.phase is Phase.FORWARD else info.backward_limit
        if limit == 0:
            plan.append(())
            continue

        remaining = headroom[position] - pending_bytes
        interval_headroom = headroom[position]
        prefetched_elements = 0
        blocked = set(active)
        considered = set()
        for target in range(position + 1, len(events)):
            successor = events[target]
            interval_headroom = min(interval_headroom, headroom[target])
            # Every selected allocation must survive to demand under this budget.
            remaining = min(remaining, interval_headroom - pending_bytes)
            if successor.operation is Operation.RESHARD:
                blocked.add(successor.module)
                if len(blocked) == len(metadata):
                    break
                continue
            owner = successor.module
            if owner in blocked or owner in active or owner in considered:
                continue
            considered.add(owner)
            candidate = metadata[owner]
            if successor.phase in ready[owner]:
                continue
            if limit is not None and prefetched_elements >= limit:
                break
            if candidate.buffer_bytes > remaining:
                break
            targets.append(target)
            pending[owner] = candidate.buffer_bytes
            pending_bytes += candidate.buffer_bytes
            remaining -= candidate.buffer_bytes
            if candidate.buffer_bytes:
                prefetched_elements += candidate.parameter_elements
            mark_ready(successor)
        plan.append(tuple(targets))
    return tuple(plan)
