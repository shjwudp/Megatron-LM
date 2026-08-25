#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Normalize a PyTorch CUDA allocator snapshot into replayable lifecycles."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from tools.cuda_allocator_repro.trace import normalize_snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def _drop_frames(trace: dict) -> None:
    """Remove stack payload after allocations have been classified."""
    for allocation in trace["initial_allocations"]:
        allocation.pop("frames", None)
        allocation.pop("alloc_frames", None)
    for allocation in trace["allocations"]:
        allocation.pop("frames", None)
        allocation.pop("alloc_frames", None)
    for event in trace["events"]:
        event.pop("frames", None)


def main() -> None:
    args = _parse_args()
    with args.snapshot.open("rb") as stream:
        snapshot = pickle.load(stream)
    trace = normalize_snapshot(
        snapshot,
        device_index=args.device,
        scope="all",
        source=args.snapshot.name,
        compact_unclassified_frames=not args.keep_frames,
    )
    if not args.keep_frames:
        _drop_frames(trace)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(
            trace,
            stream,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
            sort_keys=args.pretty,
        )
        stream.write("\n")
    print(json.dumps(trace["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
