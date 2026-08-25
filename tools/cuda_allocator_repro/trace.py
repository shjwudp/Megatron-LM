# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Normalize PyTorch CUDA allocator snapshots and plan fixed replay slots."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_LIFECYCLE_ACTIONS = {"alloc", "free_requested", "free_completed"}


def _clean_frames(frames: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(frame.get("name", "")),
            "filename": str(frame.get("filename", "")),
            "line": int(frame.get("line", 0)),
        }
        for frame in frames
    ]


def classify_mfsdp_temporary(frames: Sequence[Mapping[str, Any]]) -> str | None:
    """Classify default-allocator M-FSDP v2 AG and RS temporary allocations.

    Persistent shards and fused-wgrad buffers are deliberately excluded: the
    trace-pool allocator does not replace them, so including them would hide the
    allocator mechanism that this replay is intended to isolate.
    """
    names = {str(frame.get("name", "")) for frame in frames}
    filenames = {str(frame.get("filename", "")).replace("\\", "/") for frame in frames}
    is_experimental_mfsdp = any(
        "/megatron_fsdp/experimental/" in filename for filename in filenames
    )
    if not is_experimental_mfsdp:
        return None
    if "unshard_parameters" in names and (
        "reallocate_storage" in names or "_resize_storage" in names
    ):
        return "allgather"
    if "allocate_partial_grad_buffer" in names:
        return "reduce_scatter"
    return None


def _matches_frame_substrings(
    frames: Sequence[Mapping[str, Any]], frame_substrings: Sequence[str]
) -> bool:
    if not frame_substrings:
        return True
    haystack = "\n".join(
        f"{frame.get('filename', '')}:{frame.get('name', '')}" for frame in frames
    )
    return any(needle in haystack for needle in frame_substrings)


def _peak_bytes(
    events: Sequence[Mapping[str, Any]],
    allocations_by_id: Mapping[str, Mapping[str, Any]],
    *,
    release_action: str,
    initial_allocation_ids: Iterable[str] = (),
) -> int:
    active = set(initial_allocation_ids)
    active_bytes = sum(
        int(allocations_by_id[allocation_id]["size_bytes"]) for allocation_id in active
    )
    peak_bytes = active_bytes
    for event in events:
        allocation_id = event["allocation_id"]
        if event["action"] == "alloc":
            if allocation_id not in active:
                active.add(allocation_id)
                active_bytes += int(allocations_by_id[allocation_id]["size_bytes"])
                peak_bytes = max(peak_bytes, active_bytes)
        elif event["action"] == release_action and allocation_id in active:
            active.remove(allocation_id)
            active_bytes -= int(allocations_by_id[allocation_id]["size_bytes"])
    return peak_bytes


def normalize_snapshot(
    snapshot: Mapping[str, Any],
    *,
    device_index: int = 0,
    scope: str = "mfsdp-temporaries",
    frame_substrings: Sequence[str] = (),
    source: str | None = None,
    compact_unclassified_frames: bool = False,
) -> dict[str, Any]:
    """Convert address-based allocator events into stable allocation lifecycles."""
    if scope not in {"all", "mfsdp-temporaries"}:
        raise ValueError(f"Unsupported scope {scope!r}.")

    device_traces = snapshot.get("device_traces", [])
    if device_index < 0 or device_index >= len(device_traces):
        raise ValueError(
            f"Snapshot has {len(device_traces)} device traces; device {device_index} is absent."
        )

    trace = device_traces[device_index]
    active_by_address: dict[int, dict[str, Any]] = {}
    selected_allocations: list[dict[str, Any]] = []
    selected_events: list[dict[str, Any]] = []
    initial_allocations_from_events: list[dict[str, Any]] = []
    allocation_counter = 0
    allocated_addresses_in_trace: set[int] = set()
    overwritten_active_addresses = 0
    unmatched_free_events = 0

    for sequence, raw_event in enumerate(trace):
        action = raw_event.get("action")
        if action not in _LIFECYCLE_ACTIONS:
            continue
        address = int(raw_event.get("addr", 0))
        frames = _clean_frames(raw_event.get("frames", []))

        if action == "alloc":
            allocation_counter += 1
            allocation_id = f"allocation-{allocation_counter:06d}"
            allocated_addresses_in_trace.add(address)
            arena = classify_mfsdp_temporary(frames)
            selected = (
                scope == "all" or arena is not None
            ) and _matches_frame_substrings(frames, frame_substrings)
            retained_frames = (
                [] if compact_unclassified_frames and arena is None else frames
            )
            if address in active_by_address:
                overwritten_active_addresses += 1
            allocation = {
                "allocation_id": allocation_id,
                "captured_address": hex(address),
                "size_bytes": int(raw_event.get("size", 0)),
                "allocation_stream": int(raw_event.get("stream", 0)),
                "arena": arena or "unclassified",
                "alloc_sequence": sequence,
                "free_requested_sequence": None,
                "free_completed_sequence": None,
                "alloc_frames": retained_frames,
                "selected": selected,
            }
            active_by_address[address] = allocation
            if selected:
                selected_allocations.append(allocation)
                selected_events.append(
                    {
                        "sequence": sequence,
                        "action": action,
                        "allocation_id": allocation_id,
                        "size_bytes": allocation["size_bytes"],
                        "stream": int(raw_event.get("stream", 0)),
                        "frames": retained_frames,
                    }
                )
            continue

        allocation = active_by_address.get(address)
        if allocation is None:
            unmatched_free_events += 1
            if scope != "all":
                continue
            # Memory history can begin while blocks allocated before the
            # capture are still live. Their first observed action is a free,
            # so synthesize the missing boundary allocation and seed it in a
            # full replay. A later alloc at the same address remains a distinct
            # lifecycle.
            allocation = {
                "allocation_id": (
                    f"initial-{len(initial_allocations_from_events) + 1:06d}"
                ),
                "captured_address": hex(address),
                "size_bytes": int(raw_event.get("size", 0)),
                "requested_size_bytes": int(raw_event.get("size", 0)),
                "allocation_stream": int(raw_event.get("stream", 0)),
                "arena": "unclassified",
                "alloc_sequence": None,
                "free_requested_sequence": None,
                "free_completed_sequence": None,
                "frames": [] if compact_unclassified_frames else frames,
                "selected": _matches_frame_substrings(frames, frame_substrings),
            }
            initial_allocations_from_events.append(allocation)
            active_by_address[address] = allocation
        sequence_field = f"{action}_sequence"
        allocation[sequence_field] = sequence
        if allocation["selected"]:
            selected_events.append(
                {
                    "sequence": sequence,
                    "action": action,
                    "allocation_id": allocation["allocation_id"],
                    "size_bytes": int(raw_event.get("size", allocation["size_bytes"])),
                    "stream": int(raw_event.get("stream", 0)),
                    "frames": (
                        []
                        if compact_unclassified_frames
                        and allocation["arena"] == "unclassified"
                        else frames
                    ),
                }
            )
        if action == "free_completed":
            active_by_address.pop(address, None)

    for allocation in selected_allocations:
        allocation.pop("selected", None)

    allocations_by_id = {
        allocation["allocation_id"]: allocation for allocation in selected_allocations
    }
    arena_counts = Counter(allocation["arena"] for allocation in selected_allocations)
    stream_counts = Counter(
        allocation["allocation_stream"] for allocation in selected_allocations
    )
    device_segments = [
        segment
        for segment in snapshot.get("segments", [])
        if int(segment.get("device", -1)) == device_index
    ]
    initial_allocations = initial_allocations_from_events
    if scope == "all":
        for segment in device_segments:
            stream = int(segment.get("stream", 0))
            for block in segment.get("blocks", []):
                if not str(block.get("state", "")).startswith("active"):
                    continue
                address = int(block.get("address", 0))
                # A block allocated during the bounded window is already
                # represented by an event lifecycle. Remaining active blocks
                # are the persistent allocator state that existed before the
                # first captured event and must be seeded for a full replay.
                if address in allocated_addresses_in_trace:
                    continue
                initial_allocations.append(
                    {
                        "allocation_id": f"initial-{len(initial_allocations) + 1:06d}",
                        "captured_address": hex(address),
                        "size_bytes": int(block.get("size", 0)),
                        "requested_size_bytes": int(
                            block.get("requested_size", block.get("size", 0))
                        ),
                        "allocation_stream": stream,
                        "arena": "unclassified",
                        "alloc_sequence": None,
                        "free_requested_sequence": None,
                        "free_completed_sequence": None,
                        "frames": (
                            []
                            if compact_unclassified_frames
                            else _clean_frames(block.get("frames", []))
                        ),
                    }
                )
    for allocation in initial_allocations:
        allocation.pop("selected", None)
    initial_stream_counts = Counter(
        allocation["allocation_stream"] for allocation in initial_allocations
    )
    all_allocations_by_id = {
        **allocations_by_id,
        **{
            allocation["allocation_id"]: allocation
            for allocation in initial_allocations
        },
    }
    initial_allocation_ids = {
        allocation["allocation_id"] for allocation in initial_allocations
    }
    summary = {
        "raw_trace_event_count": len(trace),
        "selected_event_count": len(selected_events),
        "selected_allocation_count": len(selected_allocations),
        "selected_requested_bytes": sum(
            allocation["size_bytes"] for allocation in selected_allocations
        ),
        "logical_peak_bytes": _peak_bytes(
            selected_events, allocations_by_id, release_action="free_requested"
        ),
        "physical_availability_peak_bytes": _peak_bytes(
            selected_events, allocations_by_id, release_action="free_completed"
        ),
        "logical_peak_with_initial_bytes": _peak_bytes(
            selected_events,
            all_allocations_by_id,
            release_action="free_requested",
            initial_allocation_ids=initial_allocation_ids,
        ),
        "physical_availability_peak_with_initial_bytes": _peak_bytes(
            selected_events,
            all_allocations_by_id,
            release_action="free_completed",
            initial_allocation_ids=initial_allocation_ids,
        ),
        "arena_allocation_counts": dict(sorted(arena_counts.items())),
        "allocation_stream_counts": {
            str(stream): count for stream, count in sorted(stream_counts.items())
        },
        "initial_allocation_count": len(initial_allocations),
        "initial_allocated_bytes": sum(
            allocation["size_bytes"] for allocation in initial_allocations
        ),
        "initial_allocation_stream_counts": {
            str(stream): count
            for stream, count in sorted(initial_stream_counts.items())
        },
        "unclosed_free_requested": sum(
            allocation["free_requested_sequence"] is None
            for allocation in selected_allocations
        ),
        "unclosed_free_completed": sum(
            allocation["free_completed_sequence"] is None
            for allocation in selected_allocations
        ),
        "snapshot_segment_bytes": sum(
            int(segment.get("total_size", 0)) for segment in device_segments
        ),
        "snapshot_allocated_bytes": sum(
            int(segment.get("allocated_size", 0)) for segment in device_segments
        ),
        "overwritten_active_addresses": overwritten_active_addresses,
        "unmatched_free_events": unmatched_free_events,
    }
    return {
        "schema_version": 1,
        "source_snapshot": source,
        "device_index": device_index,
        "scope": scope,
        "frame_substrings": list(frame_substrings),
        "compact_unclassified_frames": compact_unclassified_frames,
        "allocator_settings": snapshot.get("allocator_settings"),
        "summary": summary,
        "initial_allocations": initial_allocations,
        "allocations": selected_allocations,
        "events": selected_events,
    }


def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def build_slot_plan(
    normalized_trace: Mapping[str, Any],
    *,
    partition: str = "arena",
    eligible_arenas: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Color logical lifetimes into persistent byte slots.

    ``partition='arena'`` mirrors trace-pool AG/RS separation. Dtype is not
    present in a PyTorch allocator snapshot, so this is a byte-level lower
    bound rather than a claim that the generated plan is the exact M-FSDP plan.
    """
    if partition not in {"none", "arena", "arena-stream"}:
        raise ValueError(f"Unsupported slot partition {partition!r}.")

    trace_end = max(
        (int(event["sequence"]) for event in normalized_trace["events"]), default=0
    )
    allocations = [
        allocation
        for allocation in normalized_trace["allocations"]
        if eligible_arenas is None or allocation["arena"] in eligible_arenas
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for allocation in allocations:
        if partition == "none":
            group = ("all",)
        elif partition == "arena":
            group = (allocation["arena"],)
        else:
            group = (allocation["arena"], allocation["allocation_stream"])
        end = allocation["free_requested_sequence"]
        allocation_with_interval = dict(allocation)
        allocation_with_interval["interval"] = (
            int(allocation["alloc_sequence"]),
            int(end) if end is not None else trace_end + 1,
        )
        groups[group].append(allocation_with_interval)

    slots: list[dict[str, Any]] = []
    allocation_to_slot: dict[str, int] = {}
    for group, group_allocations in sorted(
        groups.items(), key=lambda item: repr(item[0])
    ):
        conflicts: dict[str, set[str]] = defaultdict(set)
        for index, left in enumerate(group_allocations):
            for right in group_allocations[index + 1 :]:
                if _intervals_overlap(left["interval"], right["interval"]):
                    conflicts[left["allocation_id"]].add(right["allocation_id"])
                    conflicts[right["allocation_id"]].add(left["allocation_id"])

        group_slots: list[dict[str, Any]] = []
        for allocation in sorted(
            group_allocations,
            key=lambda item: (-int(item["size_bytes"]), item["allocation_id"]),
        ):
            unavailable = {
                allocation_to_slot[neighbor]
                for neighbor in conflicts[allocation["allocation_id"]]
                if neighbor in allocation_to_slot
            }
            candidates = [
                (
                    int(slot["capacity_bytes"]) - int(allocation["size_bytes"]),
                    slot["slot_id"],
                )
                for slot in group_slots
                if slot["slot_id"] not in unavailable
                and int(slot["capacity_bytes"]) >= int(allocation["size_bytes"])
            ]
            if candidates:
                _, slot_id = min(candidates)
                slot = slots[slot_id]
            else:
                slot_id = len(slots)
                slot = {
                    "slot_id": slot_id,
                    "group": list(group),
                    "capacity_bytes": int(allocation["size_bytes"]),
                    "allocation_ids": [],
                }
                slots.append(slot)
                group_slots.append(slot)
            slot["allocation_ids"].append(allocation["allocation_id"])
            allocation_to_slot[allocation["allocation_id"]] = slot_id

    return {
        "partition": partition,
        "eligible_arenas": (
            sorted(eligible_arenas) if eligible_arenas is not None else None
        ),
        "slot_count": len(slots),
        "slot_bytes": sum(int(slot["capacity_bytes"]) for slot in slots),
        "slots": slots,
        "allocation_to_slot": allocation_to_slot,
    }

