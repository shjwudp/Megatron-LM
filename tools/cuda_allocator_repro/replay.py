#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Replay a normalized allocation lifecycle on one CUDA GPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from tools.cuda_allocator_repro.trace import build_slot_plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Normalized allocator trace JSON")
    parser.add_argument(
        "--mode",
        choices=("caching", "pool", "hybrid-pool", "cold-to-hybrid-pool"),
        required=True,
    )
    parser.add_argument(
        "--steady-trace",
        type=Path,
        help="Steady-state full trace used after a cold trace-pool planning boundary",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size-scale", type=float, default=1.0)
    parser.add_argument(
        "--free-at",
        choices=("requested", "completed"),
        default="completed",
        help="When caching replay releases storage; completed preserves captured reuse eligibility",
    )
    parser.add_argument(
        "--slot-partition",
        choices=("none", "arena", "arena-stream"),
        default="arena",
    )
    parser.add_argument(
        "--touch", action="store_true", help="Zero every physical allocation"
    )
    parser.add_argument(
        "--seed-initial-state",
        action="store_true",
        help="Materialize persistent blocks that predate the captured window",
    )
    parser.add_argument("--output", type=Path, help="Write replay metrics as JSON")
    return parser.parse_args()


def _scaled_size(size_bytes: int, scale: float) -> int:
    return max(1, math.ceil(size_bytes * scale))


def _memory_stats() -> dict[str, int]:
    stats = torch.cuda.memory_stats()
    return {
        key: int(stats.get(key, 0))
        for key in (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "requested_bytes.all.current",
            "requested_bytes.all.peak",
            "segment.all.current",
            "segment.all.peak",
            "inactive_split_bytes.all.current",
            "inactive_split_bytes.all.peak",
        )
    }


def _stream_map(trace: dict[str, Any], device: int) -> dict[int, torch.cuda.Stream]:
    captured_streams = sorted(
        {int(event["stream"]) for event in trace["events"]}
        | {
            int(allocation["allocation_stream"])
            for allocation in trace.get("initial_allocations", [])
        }
    )
    default_stream = torch.cuda.default_stream(device)
    return {
        stream: default_stream if stream == 0 else torch.cuda.Stream(device=device)
        for stream in captured_streams
    }


def _release_tensor(
    tensors: dict[str, torch.Tensor],
    allocation_id: str,
    stream: torch.cuda.Stream,
) -> None:
    with torch.cuda.stream(stream):
        tensor = tensors.pop(allocation_id, None)
        del tensor


def _run_caching_replay(
    trace: dict[str, Any],
    streams: dict[int, torch.cuda.Stream],
    *,
    repeats: int,
    scale: float,
    free_at: str,
    touch: bool,
    initial_live: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    final_live = {}
    for repeat in range(repeats):
        torch.cuda.nvtx.range_push(f"allocator_replay_caching_{repeat}")
        live = initial_live if repeat == 0 and initial_live is not None else {}
        pending: dict[str, torch.Tensor] = {}
        for event in trace["events"]:
            allocation_id = event["allocation_id"]
            stream = streams[int(event["stream"])]
            if event["action"] == "alloc":
                with torch.cuda.stream(stream):
                    tensor = torch.empty(
                        _scaled_size(int(event["size_bytes"]), scale),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    if touch:
                        tensor.zero_()
                live[allocation_id] = tensor
            elif event["action"] == "free_requested":
                if free_at == "requested":
                    _release_tensor(live, allocation_id, stream)
                else:
                    tensor = live.pop(allocation_id, None)
                    if tensor is not None:
                        pending[allocation_id] = tensor
            elif event["action"] == "free_completed" and free_at == "completed":
                _release_tensor(pending, allocation_id, stream)

        # Preserve allocations that cross the final capture boundary so the
        # measured end state matches the recorded window. Repeated playback is
        # a cache-stress diagnostic; intermediate boundary state is cleared
        # because allocation IDs are local to one captured window.
        if repeat == repeats - 1:
            final_live = live
        else:
            live.clear()
        pending.clear()
        torch.cuda.nvtx.range_pop()
    return final_live


def _seed_initial_allocations(
    trace: dict[str, Any],
    streams: dict[int, torch.cuda.Stream],
    *,
    scale: float,
    touch: bool,
) -> dict[str, torch.Tensor]:
    tensors = {}
    for allocation in trace.get("initial_allocations", []):
        stream = streams[int(allocation["allocation_stream"])]
        with torch.cuda.stream(stream):
            tensor = torch.empty(
                _scaled_size(int(allocation["size_bytes"]), scale),
                dtype=torch.uint8,
                device="cuda",
            )
            if touch:
                tensor.zero_()
        tensors[allocation["allocation_id"]] = tensor
    return tensors


def _final_boundary_allocations(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    allocations_by_id = {
        allocation["allocation_id"]: allocation
        for allocation in [
            *trace.get("initial_allocations", []),
            *trace.get("allocations", []),
        ]
    }
    active = {
        allocation["allocation_id"]
        for allocation in trace.get("initial_allocations", [])
    }
    for event in trace["events"]:
        allocation_id = event["allocation_id"]
        if event["action"] == "alloc":
            active.add(allocation_id)
        elif event["action"] == "free_requested":
            active.discard(allocation_id)
    return {
        allocation_id: allocations_by_id[allocation_id]
        for allocation_id in sorted(active)
    }


def build_boundary_mapping(
    source_trace: dict[str, Any], target_trace: dict[str, Any]
) -> tuple[dict[str, str], dict[str, int]]:
    """Map a cold trace's final live blocks to a steady trace's initial blocks.

    Captures made in separate processes do not share allocation IDs or stream
    handles. Exact address-and-size pairs are matched first, then exact sizes,
    with stable size-order pairing for the small remainder. The latter keeps
    the live byte footprint while acknowledging minor allocator-address drift.
    """
    source = _final_boundary_allocations(source_trace)
    target = {
        allocation["allocation_id"]: allocation
        for allocation in target_trace.get("initial_allocations", [])
    }
    if len(source) != len(target):
        raise ValueError(
            "Trace boundaries have different live allocation counts: "
            f"{len(source)} != {len(target)}."
        )

    remaining_source = dict(source)
    remaining_target = dict(target)
    target_by_address_size = {
        (allocation["captured_address"], int(allocation["size_bytes"])): allocation_id
        for allocation_id, allocation in remaining_target.items()
    }
    mapping: dict[str, str] = {}
    matched_by_address_size = 0
    for source_id, allocation in sorted(remaining_source.items()):
        key = (allocation["captured_address"], int(allocation["size_bytes"]))
        target_id = target_by_address_size.get(key)
        if target_id is None or target_id not in remaining_target:
            continue
        mapping[target_id] = source_id
        remaining_source.pop(source_id)
        remaining_target.pop(target_id)
        matched_by_address_size += 1

    target_ids_by_size: dict[int, list[str]] = {}
    for target_id, allocation in sorted(remaining_target.items()):
        target_ids_by_size.setdefault(int(allocation["size_bytes"]), []).append(
            target_id
        )
    matched_by_size = 0
    for source_id, allocation in sorted(list(remaining_source.items())):
        target_ids = target_ids_by_size.get(int(allocation["size_bytes"]))
        if not target_ids:
            continue
        target_id = target_ids.pop(0)
        mapping[target_id] = source_id
        remaining_source.pop(source_id)
        remaining_target.pop(target_id)
        matched_by_size += 1

    source_remainder = sorted(
        remaining_source.items(),
        key=lambda item: (int(item[1]["size_bytes"]), item[0]),
    )
    target_remainder = sorted(
        remaining_target.items(),
        key=lambda item: (int(item[1]["size_bytes"]), item[0]),
    )
    for (source_id, _), (target_id, _) in zip(
        source_remainder, target_remainder, strict=True
    ):
        mapping[target_id] = source_id

    size_mismatch_count = sum(
        int(source[source_id]["size_bytes"]) != int(target[target_id]["size_bytes"])
        for target_id, source_id in mapping.items()
    )
    summary = {
        "allocation_count": len(mapping),
        "matched_by_address_size": matched_by_address_size,
        "matched_by_size": matched_by_size,
        "matched_by_size_order": len(source_remainder),
        "size_mismatch_count": size_mismatch_count,
        "source_bytes": sum(int(item["size_bytes"]) for item in source.values()),
        "target_bytes": sum(int(item["size_bytes"]) for item in target.values()),
        "absolute_paired_size_delta_bytes": sum(
            abs(
                int(source[source_id]["size_bytes"])
                - int(target[target_id]["size_bytes"])
            )
            for target_id, source_id in mapping.items()
        ),
    }
    return mapping, summary


def _remap_boundary_tensors(
    tensors: dict[str, torch.Tensor], mapping: dict[str, str]
) -> dict[str, torch.Tensor]:
    missing_source_ids = set(mapping.values()) - tensors.keys()
    if missing_source_ids:
        raise RuntimeError(
            "Cold replay did not retain all mapped boundary tensors: "
            f"{tuple(sorted(missing_source_ids))!r}."
        )
    remapped = {
        target_id: tensors[source_id] for target_id, source_id in mapping.items()
    }
    tensors.clear()
    return remapped


def _materialize_pool(
    trace: dict[str, Any],
    *,
    partition: str,
    scale: float,
    touch: bool,
    eligible_arenas: set[str] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, int], dict[str, Any]]:
    plan = build_slot_plan(trace, partition=partition, eligible_arenas=eligible_arenas)
    slots: dict[int, torch.Tensor] = {}
    for slot in plan["slots"]:
        tensor = torch.empty(
            _scaled_size(int(slot["capacity_bytes"]), scale),
            dtype=torch.uint8,
            device="cuda",
        )
        if touch:
            tensor.zero_()
        slots[int(slot["slot_id"])] = tensor
    return slots, plan["allocation_to_slot"], plan


def _run_pool_replay(
    trace: dict[str, Any],
    streams: dict[int, torch.cuda.Stream],
    slots: dict[int, torch.Tensor],
    allocation_to_slot: dict[str, int],
    *,
    repeats: int,
    scale: float,
) -> None:
    for repeat in range(repeats):
        torch.cuda.nvtx.range_push(f"allocator_replay_pool_{repeat}")
        live: dict[str, torch.Tensor] = {}
        for event in trace["events"]:
            allocation_id = event["allocation_id"]
            if event["action"] == "alloc":
                slot = slots[int(allocation_to_slot[allocation_id])]
                size = _scaled_size(int(event["size_bytes"]), scale)
                with torch.cuda.stream(streams[int(event["stream"])]):
                    live[allocation_id] = slot.narrow(0, 0, size)
            elif event["action"] == "free_requested":
                live.pop(allocation_id, None)
        live.clear()
        torch.cuda.nvtx.range_pop()


def _run_hybrid_pool_replay(
    trace: dict[str, Any],
    streams: dict[int, torch.cuda.Stream],
    slots: dict[int, torch.Tensor],
    allocation_to_slot: dict[str, int],
    *,
    repeats: int,
    scale: float,
    free_at: str,
    touch: bool,
    initial_live: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    pooled_ids = set(allocation_to_slot)
    final_live = {}
    for repeat in range(repeats):
        torch.cuda.nvtx.range_push(f"allocator_replay_hybrid_pool_{repeat}")
        live_common = initial_live if repeat == 0 and initial_live is not None else {}
        pending_common: dict[str, torch.Tensor] = {}
        live_pool: dict[str, torch.Tensor] = {}
        for event in trace["events"]:
            allocation_id = event["allocation_id"]
            stream = streams[int(event["stream"])]
            if allocation_id in pooled_ids:
                if event["action"] == "alloc":
                    slot = slots[int(allocation_to_slot[allocation_id])]
                    size = _scaled_size(int(event["size_bytes"]), scale)
                    with torch.cuda.stream(stream):
                        live_pool[allocation_id] = slot.narrow(0, 0, size)
                elif event["action"] == "free_requested":
                    live_pool.pop(allocation_id, None)
                continue

            if event["action"] == "alloc":
                with torch.cuda.stream(stream):
                    tensor = torch.empty(
                        _scaled_size(int(event["size_bytes"]), scale),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    if touch:
                        tensor.zero_()
                live_common[allocation_id] = tensor
            elif event["action"] == "free_requested":
                if free_at == "requested":
                    _release_tensor(live_common, allocation_id, stream)
                else:
                    tensor = live_common.pop(allocation_id, None)
                    if tensor is not None:
                        pending_common[allocation_id] = tensor
            elif event["action"] == "free_completed" and free_at == "completed":
                _release_tensor(pending_common, allocation_id, stream)

        if repeat == repeats - 1:
            final_live = live_common
        else:
            live_common.clear()
        pending_common.clear()
        live_pool.clear()
        torch.cuda.nvtx.range_pop()
    return final_live


def _run_cold_to_hybrid_pool(
    args: argparse.Namespace,
    cold_trace: dict[str, Any],
    *,
    start_free_bytes: int,
    total_bytes: int,
    baseline_stats: dict[str, int],
) -> dict[str, Any]:
    if args.steady_trace is None:
        raise ValueError("--steady-trace is required for cold-to-hybrid-pool mode.")
    if not args.seed_initial_state:
        raise ValueError(
            "cold-to-hybrid-pool mode requires --seed-initial-state to preserve "
            "the captured cold boundary."
        )
    with args.steady_trace.open(encoding="utf-8") as stream:
        steady_trace = json.load(stream)

    cold_streams = _stream_map(cold_trace, args.device)
    cold_initial_tensors = _seed_initial_allocations(
        cold_trace,
        cold_streams,
        scale=args.size_scale,
        touch=args.touch,
    )
    seeded_initial_allocation_count = len(cold_initial_tensors)
    seeded_initial_bytes = sum(
        tensor.numel() for tensor in cold_initial_tensors.values()
    )
    seeded_memory_stats = _memory_stats()

    torch.cuda.reset_peak_memory_stats(args.device)
    cold_final_tensors = _run_caching_replay(
        cold_trace,
        cold_streams,
        repeats=1,
        scale=args.size_scale,
        free_at=args.free_at,
        touch=args.touch,
        initial_live=cold_initial_tensors,
    )
    torch.cuda.synchronize(args.device)
    cold_end_free_bytes, _ = torch.cuda.mem_get_info(args.device)
    cold_replay_memory_stats = _memory_stats()

    steady_streams = _stream_map(steady_trace, args.device)
    pool_slots, allocation_to_slot, pool_plan = _materialize_pool(
        steady_trace,
        partition=args.slot_partition,
        scale=args.size_scale,
        touch=args.touch,
        eligible_arenas={"allgather", "reduce_scatter"},
    )
    boundary_mapping, boundary_mapping_summary = build_boundary_mapping(
        cold_trace, steady_trace
    )
    steady_initial_tensors = _remap_boundary_tensors(
        cold_final_tensors, boundary_mapping
    )

    torch.cuda.synchronize(args.device)
    pre_trim_free_bytes, _ = torch.cuda.mem_get_info(args.device)
    pre_trim_memory_stats = _memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)
    post_trim_free_bytes, _ = torch.cuda.mem_get_info(args.device)
    post_trim_memory_stats = _memory_stats()

    torch.cuda.reset_peak_memory_stats(args.device)
    final_boundary_tensors = _run_hybrid_pool_replay(
        steady_trace,
        steady_streams,
        pool_slots,
        allocation_to_slot,
        repeats=args.repeats,
        scale=args.size_scale,
        free_at=args.free_at,
        touch=args.touch,
        initial_live=steady_initial_tensors,
    )
    torch.cuda.synchronize(args.device)
    end_free_bytes, observed_total_bytes = torch.cuda.mem_get_info(args.device)
    if int(total_bytes) != int(observed_total_bytes):
        raise RuntimeError("CUDA device total memory changed during replay.")

    return {
        "schema_version": 3,
        "trace": str(args.trace),
        "steady_trace": str(args.steady_trace),
        "mode": args.mode,
        "repeats": args.repeats,
        "size_scale": args.size_scale,
        "free_at": args.free_at,
        "slot_partition": args.slot_partition,
        "touch": args.touch,
        "seed_initial_state": args.seed_initial_state,
        "seeded_initial_allocation_count": seeded_initial_allocation_count,
        "seeded_initial_bytes": seeded_initial_bytes,
        "final_boundary_allocation_count": len(final_boundary_tensors),
        "final_boundary_bytes": sum(
            tensor.numel() for tensor in final_boundary_tensors.values()
        ),
        "boundary_mapping": boundary_mapping_summary,
        "device": args.device,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_alloc_conf": os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
        "device_total_bytes": int(observed_total_bytes),
        "cold_device_used_delta_bytes": int(start_free_bytes - cold_end_free_bytes),
        "pre_trim_device_used_delta_bytes": int(start_free_bytes - pre_trim_free_bytes),
        "post_trim_device_used_delta_bytes": int(
            start_free_bytes - post_trim_free_bytes
        ),
        "device_used_delta_bytes": int(start_free_bytes - end_free_bytes),
        "baseline_memory_stats": baseline_stats,
        "seeded_memory_stats": seeded_memory_stats,
        "cold_replay_memory_stats": cold_replay_memory_stats,
        "pre_trim_memory_stats": pre_trim_memory_stats,
        "post_trim_memory_stats": post_trim_memory_stats,
        "replay_memory_stats": _memory_stats(),
        "pool_plan": {
            "slot_count": pool_plan["slot_count"],
            "unscaled_slot_bytes": pool_plan["slot_bytes"],
            "scaled_slot_bytes": sum(tensor.numel() for tensor in pool_slots.values()),
        },
        "captured_summary": cold_trace.get("summary"),
        "steady_captured_summary": steady_trace.get("summary"),
    }


def _write_and_print_metrics(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2, sort_keys=True)
            stream.write("\n")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    stats = metrics["replay_memory_stats"]
    print(
        f"{args.mode}: peak allocated={stats['allocated_bytes.all.peak'] / 1024**3:.3f} GiB, "
        f"peak reserved={stats['reserved_bytes.all.peak'] / 1024**3:.3f} GiB, "
        f"end device delta={metrics['device_used_delta_bytes'] / 1024**3:.3f} GiB"
    )


def main() -> None:
    args = _parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if not (0 < args.size_scale <= 1):
        raise ValueError("--size-scale must be in (0, 1].")
    if args.mode != "cold-to-hybrid-pool" and args.steady_trace is not None:
        raise ValueError("--steady-trace is only valid in cold-to-hybrid-pool mode.")
    if not torch.cuda.is_available():
        raise RuntimeError("This replay requires a CUDA GPU.")

    with args.trace.open(encoding="utf-8") as stream:
        trace = json.load(stream)
    torch.cuda.set_device(args.device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)
    start_free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    baseline_stats = _memory_stats()
    if args.mode == "cold-to-hybrid-pool":
        metrics = _run_cold_to_hybrid_pool(
            args,
            trace,
            start_free_bytes=int(start_free_bytes),
            total_bytes=int(total_bytes),
            baseline_stats=baseline_stats,
        )
        _write_and_print_metrics(args, metrics)
        return

    streams = _stream_map(trace, args.device)
    initial_tensors = (
        _seed_initial_allocations(
            trace,
            streams,
            scale=args.size_scale,
            touch=args.touch,
        )
        if args.seed_initial_state
        else {}
    )
    seeded_initial_allocation_count = len(initial_tensors)
    seeded_initial_bytes = sum(tensor.numel() for tensor in initial_tensors.values())
    seeded_memory_stats = _memory_stats()

    pool_plan = None
    pool_slots: dict[int, torch.Tensor] = {}
    if args.mode in {"pool", "hybrid-pool"}:
        pool_slots, allocation_to_slot, pool_plan = _materialize_pool(
            trace,
            partition=args.slot_partition,
            scale=args.size_scale,
            touch=args.touch,
            eligible_arenas=(
                {"allgather", "reduce_scatter"} if args.mode == "hybrid-pool" else None
            ),
        )
    prepared_memory_stats = _memory_stats()

    torch.cuda.reset_peak_memory_stats(args.device)
    final_boundary_tensors = {}
    if args.mode == "caching":
        final_boundary_tensors = _run_caching_replay(
            trace,
            streams,
            repeats=args.repeats,
            scale=args.size_scale,
            free_at=args.free_at,
            touch=args.touch,
            initial_live=initial_tensors,
        )
    elif args.mode == "pool":
        _run_pool_replay(
            trace,
            streams,
            pool_slots,
            allocation_to_slot,
            repeats=args.repeats,
            scale=args.size_scale,
        )
    else:
        final_boundary_tensors = _run_hybrid_pool_replay(
            trace,
            streams,
            pool_slots,
            allocation_to_slot,
            repeats=args.repeats,
            scale=args.size_scale,
            free_at=args.free_at,
            touch=args.touch,
            initial_live=initial_tensors,
        )
    torch.cuda.synchronize(args.device)

    end_free_bytes, observed_total_bytes = torch.cuda.mem_get_info(args.device)
    metrics = {
        "schema_version": 2,
        "trace": str(args.trace),
        "mode": args.mode,
        "repeats": args.repeats,
        "size_scale": args.size_scale,
        "free_at": (args.free_at if args.mode in {"caching", "hybrid-pool"} else None),
        "slot_partition": (
            args.slot_partition if args.mode in {"pool", "hybrid-pool"} else None
        ),
        "touch": args.touch,
        "seed_initial_state": args.seed_initial_state,
        "seeded_initial_allocation_count": seeded_initial_allocation_count,
        "seeded_initial_bytes": seeded_initial_bytes,
        "final_boundary_allocation_count": len(final_boundary_tensors),
        "final_boundary_bytes": sum(
            tensor.numel() for tensor in final_boundary_tensors.values()
        ),
        "device": args.device,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_alloc_conf": os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
        "device_total_bytes": int(observed_total_bytes),
        "device_used_delta_bytes": int(start_free_bytes - end_free_bytes),
        "baseline_memory_stats": baseline_stats,
        "seeded_memory_stats": seeded_memory_stats,
        "prepared_memory_stats": prepared_memory_stats,
        "replay_memory_stats": _memory_stats(),
        "pool_plan": (
            {
                "slot_count": pool_plan["slot_count"],
                "unscaled_slot_bytes": pool_plan["slot_bytes"],
                "scaled_slot_bytes": sum(
                    tensor.numel() for tensor in pool_slots.values()
                ),
            }
            if pool_plan is not None
            else None
        ),
        "captured_summary": trace.get("summary"),
    }
    if int(total_bytes) != int(observed_total_bytes):
        raise RuntimeError("CUDA device total memory changed during replay.")

    _write_and_print_metrics(args, metrics)


if __name__ == "__main__":
    main()

