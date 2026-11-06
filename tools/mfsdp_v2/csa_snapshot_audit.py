#!/usr/bin/env python3
"""Audit a truncated PyTorch allocator history using only the standard library.

Example (outputs must be new files):
    python tools/mfsdp_v2/csa_snapshot_audit.py \
        --input /tmp/proxy-snapshots/hfsdp_ep2_*gather_rank[04].pickle \
        --output-dir memory_snapshot_analysis/proxy-local/snapshot/audit

FC is requested memory not yet free_completed; FR is requested memory not yet
free_requested. Their difference is pending memory. Both include the baseline
reconstructed backwards from final active blocks. Event indices are zero based;
-1 means immediately before the first retained event. No torch import is needed.

Only load trusted pickle files. No input or existing output is overwritten.
"""

import argparse
import bisect
import collections
import csv
import hashlib
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

GIB = 1 << 30
MEMORY_ACTIONS = {"alloc", "free_requested", "free_completed"}
RESERVE_DELTA = {"segment_alloc": 1, "segment_map": 1, "segment_free": -1, "segment_unmap": -1}
OTHER_ACTIONS = {"snapshot", "oom"}
METRICS = ("fc_bytes", "fr_bytes", "pending_bytes", "fc_count", "fr_count", "pending_count")


class AuditError(ValueError):
    """The recorded history cannot satisfy the allocator lifecycle contract."""


def require(condition, message):
    """Reject invalid data even when Python runs with -O."""
    if not condition:
        raise AuditError(message)


def encoded(value):
    """Preserve every frame field and frame order in the grouping key."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stack_key(frames, stream):
    """Use full recorded frames and allocation stream as the identity."""
    require(isinstance(frames, list), "frames must be a list")
    require(all(isinstance(f, dict) for f in frames), "frames must contain dictionaries")
    return encoded([stream, frames])


def category(frames):
    """Source-backed labels; these do not determine whether a block is saved."""
    names = {f.get("name", "") for f in frames}
    for frame in frames:
        if frame.get("name") == "_initialize_buffers":
            return f"init_buffer@L{frame.get('line')}"
    if names & {"get_weight_workspace", "quantize_weight"}:
        return "te_weight_workspace"
    if "get_cublas_workspace" in names:
        return "te_cublas_workspace"
    if any(
        "optimizer" in f.get("filename", "") or "muon.py" in f.get("filename", "") for f in frames
    ):
        return "optimizer"
    for name, label in (
        ("allocate_partial_grad_buffer", "partial_grad"),
        ("reduce_inner_gradients", "reduced_grad"),
        ("reduce_outer_gradients", "reduced_grad"),
        ("unshard_parameters", "unshard_weight"),
    ):
        if name in names:
            return label
    if any("cross_entropy" in f.get("filename", "") for f in frames):
        return "cross_entropy"
    if "flash_mla_sparse_fwd" in names:
        return "flash_mla_forward"
    if any(
        "csa_utils/" in f.get("filename", "") or f.get("filename", "").endswith("/csa.py")
        for f in frames
    ):
        return "csa"
    return "other" if frames else "unattributed"


@dataclass
class Block:
    """One address generation, including its observed allocation/free boundaries."""

    address: int
    size: int
    stream: int
    key: str
    provenance: str
    alloc: int | None = None
    fr: int | None = None
    fc: int | None = None
    allocator_size: int | None = None
    initial_pending: bool = False

    @property
    def persistent(self):
        """Resident and untouched throughout this retained window, not all training."""
        return self.alloc is None and self.fr is None and self.fc is None

    def alive(self, index):
        """Whether the generation is FC-live immediately after an event."""
        return (self.alloc is None or self.alloc <= index) and (self.fc is None or self.fc > index)

    def pending(self, index):
        """Whether an FC-live generation has already been requested for release."""
        return self.initial_pending or (self.fr is not None and self.fr <= index)


class LiveBlocks:
    """Validate address generations and overlapping requested address ranges."""

    def __init__(self):
        self.blocks = {}
        self.addresses = []

    def add(self, block, pending, context):
        """Insert a generation only into an unoccupied requested address range."""
        addr = block.address
        require(addr not in self.blocks, f"{context}: address reuse before FC: {addr}")
        pos = bisect.bisect_left(self.addresses, addr)
        if pos:
            prev = self.blocks[self.addresses[pos - 1]][0]
            require(prev.address + prev.size <= addr, f"{context}: overlapping allocation")
        if pos < len(self.addresses):
            require(addr + block.size <= self.addresses[pos], f"{context}: overlapping allocation")
        self.addresses.insert(pos, addr)
        self.blocks[addr] = (block, pending)

    def get(self, event, context):
        """Resolve an event to a live generation and validate size/owner stream."""
        addr = event["addr"]
        require(addr in self.blocks, f"{context}: unknown address {addr}")
        block, pending = self.blocks[addr]
        require(block.size == event["size"], f"{context}: requested size mismatch at {addr}")
        require(block.stream == event["stream"], f"{context}: allocation stream mismatch at {addr}")
        return block, pending

    def remove(self, addr):
        """Remove a previously validated generation."""
        del self.blocks[addr]
        self.addresses.pop(bisect.bisect_left(self.addresses, addr))


def final_blocks(snapshot, device):
    """Validate segment layout and distinguish requested from allocator bytes."""
    live = LiveBlocks()
    reserved = 0
    segments = sorted(snapshot["segments"], key=lambda s: s["address"])
    previous_end = -1
    for segment in segments:
        require(segment["device"] == device, "multiple devices in final segments")
        addr = segment["address"]
        require(addr >= previous_end, "overlapping final segments")
        end = addr + segment["total_size"]
        previous_end = end
        reserved += segment["total_size"]
        for raw in segment["blocks"]:
            require(raw["address"] == addr, "noncontiguous/duplicate final block address")
            require(raw["size"] > 0, "nonpositive final allocator block size")
            addr += raw["size"]
            state = raw["state"]
            require(
                state
                in {"inactive", "active_allocated", "active_awaiting_free", "active_pending_free"},
                f"unknown block state {state}",
            )
            if state == "inactive":
                continue
            require(0 < raw["requested_size"] <= raw["size"], "invalid final requested size")
            block = Block(
                raw["address"],
                raw["requested_size"],
                segment["stream"],
                stack_key(raw["frames"], segment["stream"]),
                "final_segment",
                allocator_size=raw["size"],
            )
            live.add(block, state != "active_allocated", "final segments")
        require(addr == end, "final segment block sizes do not sum to total_size")
    return live, reserved


def reconstruct(snapshot, device):
    """Undo FC by resurrecting pending blocks, FR by making them live, alloc by removal."""
    trace = snapshot["device_traces"][device]
    live, reserved = final_blocks(snapshot, device)
    final = {a: (b.size, b.stream, b.key, pending) for a, (b, pending) in live.blocks.items()}
    blocks = [b for b, _ in live.blocks.values()]
    event_blocks = {}
    for index in range(len(trace) - 1, -1, -1):
        event = trace[index]
        action = event["action"]
        context = f"reverse event {index} ({action})"
        require(
            action in MEMORY_ACTIONS | RESERVE_DELTA.keys() | OTHER_ACTIONS,
            f"{context}: unsupported action",
        )
        if action not in MEMORY_ACTIONS:
            continue
        require(event["size"] > 0 and event["addr"] >= 0, f"{context}: invalid address/size")
        if action == "free_completed":
            block = Block(
                event["addr"],
                event["size"],
                event["stream"],
                stack_key(event["frames"], event["stream"]),
                "free_completed_only_unverified",
                fc=index,
            )
            live.add(block, True, context)
            blocks.append(block)
        else:
            block, pending = live.get(event, context)
            if action == "free_requested":
                require(
                    pending, f"{context}: duplicate FR or FR without subsequent FC/pending final"
                )
                block.fr = index
                # Delayed FC may carry the stack of the allocation triggering GC.
                # FR matches the allocation context in these snapshots; alloc wins
                # whenever retained. FC-only prefix attribution stays unverified.
                block.key = stack_key(event["frames"], block.stream)
                block.provenance = "free_requested_context"
                live.blocks[block.address] = (block, False)
            else:
                require(not pending, f"{context}: FC without preceding FR")
                block.alloc = index
                block.key = stack_key(event["frames"], block.stream)
                block.provenance = "alloc_event"
                live.remove(block.address)
        event_blocks[index] = block
    baseline = list(live.blocks.values())
    for block, pending in baseline:
        block.initial_pending = pending
    return trace, blocks, event_blocks, baseline, final, reserved


def balances(blocks, index):
    """Compute an independent balance directly from allocation lifetimes."""
    result = dict.fromkeys(METRICS, 0)
    for block in blocks:
        if block.alive(index):
            result["fc_bytes"] += block.size
            result["fc_count"] += 1
            prefix = "pending" if block.pending(index) else "fr"
            result[f"{prefix}_bytes"] += block.size
            result[f"{prefix}_count"] += 1
    return result


def adjust(values, block, action):
    """Apply one lifecycle event to count and requested-byte balances."""
    if action == "alloc":
        deltas = {"fc": 1, "fr": 1}
    elif action == "free_requested":
        deltas = {"fr": -1, "pending": 1}
    else:
        deltas = {"fc": -1, "pending": -1}
    for prefix, delta in deltas.items():
        values[f"{prefix}_bytes"] += delta * block.size
        values[f"{prefix}_count"] += delta
    require(all(v >= 0 for v in values.values()), "negative replay balance")
    for suffix in ("bytes", "count"):
        require(
            values[f"fc_{suffix}"] == values[f"fr_{suffix}"] + values[f"pending_{suffix}"],
            "FC != FR + pending",
        )


def update_max(maxima, values, index):
    """Retain the first independent maximum of each count/byte metric."""
    for metric, value in values.items():
        if metric not in maxima or value > maxima[metric]["value"]:
            maxima[metric] = {"value": value, "event": index}


def group_catalog(blocks):
    """Collect complete stack identities and allocation/free traffic statistics."""
    groups = {}
    for block in blocks:
        if block.key not in groups:
            stream, frames = json.loads(block.key)
            groups[block.key] = {
                "id": hashlib.sha256(block.key.encode()).hexdigest(),
                "stream": stream,
                "frames": frames,
                "category": category(frames),
                "dynamic": False,
                "provenance_counts": collections.Counter(),
                "events": collections.Counter(),
                "maxima": {},
            }
        group = groups[block.key]
        group["dynamic"] |= not block.persistent
        group["provenance_counts"][block.provenance] += 1
        for action, index in (
            ("alloc", block.alloc),
            ("free_requested", block.fr),
            ("free_completed", block.fc),
        ):
            if index is not None:
                group["events"][action + "_count"] += 1
                group["events"][action + "_bytes"] += block.size
    return groups


def replay(trace, blocks, event_blocks, baseline, final, reserved, groups):
    """Forward validation plus exact, unsampled global and full-stack maxima."""
    live = LiveBlocks()
    values = balances(blocks, -1)
    group_values = {key: dict.fromkeys(METRICS, 0) for key in groups}
    for block, pending in baseline:
        live.add(block, pending, "baseline")
        adjust(group_values[block.key], block, "alloc")
        if pending:
            adjust(group_values[block.key], block, "free_requested")
    for key, group in groups.items():
        group["initial"] = dict(group_values[key])
        update_max(group["maxima"], group_values[key], -1)
    maxima = {}
    update_max(maxima, values, -1)
    reserved_initial = reserved - sum(
        RESERVE_DELTA.get(e["action"], 0) * e.get("size", 0) for e in trace
    )
    require(reserved_initial >= values["fc_bytes"], "invalid initial reserved balance")
    current_reserved = reserved_initial
    timeline = [dict(event=-1, **values, reserved_bytes=current_reserved)]
    for index, event in enumerate(trace):
        action = event["action"]
        context = f"forward event {index} ({action})"
        if action in MEMORY_ACTIONS:
            block = event_blocks[index]
            if action == "alloc":
                live.add(block, False, context)
            else:
                found, pending = live.get(event, context)
                require(found is block, f"{context}: wrong address generation")
                if action == "free_requested":
                    require(not pending, f"{context}: duplicate FR")
                    live.blocks[block.address] = (block, True)
                else:
                    require(pending, f"{context}: FC before FR")
                    live.remove(block.address)
            adjust(values, block, action)
            adjust(group_values[block.key], block, action)
            update_max(groups[block.key]["maxima"], group_values[block.key], index)
        current_reserved += RESERVE_DELTA.get(action, 0) * event.get("size", 0)
        require(current_reserved >= values["fc_bytes"], f"{context}: requested exceeds reserved")
        update_max(maxima, values, index)
        timeline.append(dict(event=index, **values, reserved_bytes=current_reserved))
    actual = {a: (b.size, b.stream, b.key, p) for a, (b, p) in live.blocks.items()}
    require(actual == final, "final address/size/stream/full-stack/state closure mismatch")
    require(current_reserved == reserved, "final reserved closure mismatch")
    require(values == balances(blocks, len(trace) - 1), "final lifecycle balance mismatch")
    for key, group in groups.items():
        group["final"] = group_values[key]
    return timeline, maxima


def stable_window(timeline, width):
    """Maximum rolling minimum of total FC; an explicit event-window floor."""
    width = min(width, len(timeline))
    queue = collections.deque()
    best = None
    for i, row in enumerate(timeline):
        while queue and queue[0] <= i - width:
            queue.popleft()
        while queue and timeline[queue[-1]]["fc_bytes"] > row["fc_bytes"]:
            queue.pop()
        queue.append(i)
        if i + 1 >= width:
            low = timeline[queue[0]]
            if best is None or low["fc_bytes"] > best["fc_bytes"]:
                best = {
                    "event": low["event"],
                    "fc_bytes": low["fc_bytes"],
                    "start_event": timeline[i - width + 1]["event"],
                    "end_event": row["event"],
                    "window_events": width,
                }
    return best


def peak_groups(blocks, groups, index):
    """Describe every live generation at one exact event, grouped by full stack."""
    result = {}
    for block in blocks:
        if not block.alive(index):
            continue
        key = block.key
        if key not in result:
            result[key] = {
                "group_id": groups[key]["id"],
                "category": groups[key]["category"],
                "stream": block.stream,
                **dict.fromkeys(METRICS, 0),
                "persistent_bytes": 0,
                "requested_size_histogram": collections.Counter(),
                "allocator_size_histogram_known_only": collections.Counter(),
                "blocks": [],
            }
        row = result[key]
        prefix = "pending" if block.pending(index) else "fr"
        row["fc_bytes"] += block.size
        row["fc_count"] += 1
        row[f"{prefix}_bytes"] += block.size
        row[f"{prefix}_count"] += 1
        row["persistent_bytes"] += block.size if block.persistent else 0
        row["requested_size_histogram"][block.size] += 1
        if block.allocator_size is not None:
            row["allocator_size_histogram_known_only"][block.allocator_size] += 1
        row["blocks"].append(
            {
                "address": block.address,
                "requested_bytes": block.size,
                "allocator_bytes": block.allocator_size,
                "alloc_event": block.alloc,
                "fr_event": block.fr,
                "fc_event": block.fc,
                "pending": block.pending(index),
                "persistent": block.persistent,
                "stack_provenance": block.provenance,
            }
        )
    return sorted(result.values(), key=lambda r: (-r["fc_bytes"], r["group_id"]))


def csa_episodes(trace, event_blocks, timeline):
    """Observe each packed CSA forward allocation burst and its exit balance.

    The first non-CSA allocation closes a burst. Stack depth may limit detection;
    this is an observable allocation boundary, not an instrumented Python return.
    """
    episodes = []
    start = None
    for index, event in enumerate(trace):
        if event["action"] != "alloc":
            continue
        _, frames = json.loads(event_blocks[index].key)
        inside = any(
            f.get("filename", "").endswith("/packed_sparse_attention.py")
            and f.get("name") == "forward"
            for f in frames
        )
        if inside and start is None:
            start = index
        if not inside and start is not None:
            end = index - 1
            peak = max(timeline[start + 1 : end + 2], key=lambda r: r["fc_bytes"])
            episodes.append(
                {
                    "start_event": start,
                    "end_event": end,
                    "raw_peak_event": peak["event"],
                    "raw_peak_fc_bytes": peak["fc_bytes"],
                    "exit_fc_bytes": timeline[end + 1]["fc_bytes"],
                    "exit_fr_bytes": timeline[end + 1]["fr_bytes"],
                }
            )
            start = None
    return episodes


def audit(snapshot, stable_events=256, device=None):
    """Return a validated, JSON-compatible report and exact timeline."""
    require(isinstance(snapshot, dict), "snapshot must be a dictionary")
    require(isinstance(snapshot.get("segments"), list), "missing/invalid segments")
    require(isinstance(snapshot.get("device_traces"), list), "missing/invalid device_traces")
    require(stable_events > 0, "stable_events must be positive")
    devices = {s["device"] for s in snapshot["segments"]}
    if device is None:
        candidates = devices or {
            i
            for i, t in enumerate(snapshot["device_traces"])
            if any(e["action"] != "snapshot" for e in t)
        }
        require(len(candidates) <= 1, "ambiguous device; pass --device")
        device = next(iter(candidates), 0)
    require(not devices or devices == {device}, "selected device does not match final segments")
    # An explicitly empty trace list is a valid baseline-only snapshot.
    if not snapshot["device_traces"]:
        snapshot = dict(snapshot, device_traces=[[] for _ in range(device + 1)])
    require(0 <= device < len(snapshot["device_traces"]), "missing selected device trace")
    for i, trace in enumerate(snapshot["device_traces"]):
        require(isinstance(trace, list), "invalid device trace")
        if i != device:
            require(
                all(e["action"] == "snapshot" for e in trace),
                "other device contains allocation events",
            )
    trace, blocks, event_blocks, baseline, final, reserved = reconstruct(snapshot, device)
    groups = group_catalog(blocks)
    timeline, maxima = replay(trace, blocks, event_blocks, baseline, final, reserved, groups)
    floor = stable_window(timeline, stable_events)
    episodes = csa_episodes(trace, event_blocks, timeline)
    indices = {
        "fc_peak": maxima["fc_bytes"]["event"],
        "fr_peak": maxima["fr_bytes"]["event"],
        "pending_peak": maxima["pending_bytes"]["event"],
        "stable_window_floor": floor["event"],
    }
    if episodes:
        indices["csa_raw_peak"] = max(episodes, key=lambda e: e["raw_peak_fc_bytes"])[
            "raw_peak_event"
        ]
        indices["csa_exit_peak"] = max(episodes, key=lambda e: e["exit_fc_bytes"])["end_event"]
    peaks = {}
    for label, index in indices.items():
        details = peak_groups(blocks, groups, index)
        row = timeline[index + 1]
        for metric in METRICS:
            require(
                sum(g[metric] for g in details) == row[metric], f"{label}: peak group sum mismatch"
            )
        peaks[label] = {
            **row,
            "time_us": trace[index].get("time_us") if index >= 0 else None,
            "groups": details,
        }
    fc_groups = {r["group_id"]: r for r in peaks["fc_peak"]["groups"]}
    for group in groups.values():
        row = fc_groups.get(group["id"], {})
        group["at_fc_peak"] = {m: row.get(m, 0) for m in METRICS}
    persistent = [b for b in blocks if b.persistent]
    report = {
        "schema_version": 1,
        "validation": "PASS",
        "device": device,
        "event_count": len(trace),
        "action_counts": dict(collections.Counter(e["action"] for e in trace)),
        "definitions": {
            "bytes": "Requested bytes, not rounded allocator block size; GiB = bytes / 2**30.",
            "fc": "Live until free_completed, including reconstructed baseline.",
            "fr": "Live until free_requested, including reconstructed baseline.",
            "persistent": (
                "Same allocation generation untouched throughout the retained window only."
            ),
            "dynamic": (
                "Full-stack + allocation-stream group with at least one alloc/FR/FC in window."
            ),
            "frames": (
                "All recorded fields and frame order; only snapshot-recorded stack depth "
                "is available."
            ),
            "stable_window_floor": (
                "Maximum rolling minimum of total FC over the specified event window; "
                "not an iteration label or proof of saved tensors."
            ),
            "csa_exit_peak": (
                "Total immediately before the next allocation outside packed CSA forward; "
                "includes baseline. Boundary inferred from allocation stacks."
            ),
            "allocator_bytes": (
                "Known only for generations active in final segments; null elsewhere, "
                "never guessed from requested size."
            ),
            "attribution": (
                "alloc > FR context > FC-only context (unverified). Delayed FC stacks may "
                "describe the triggering allocation."
            ),
            "saved": (
                "Not inferred from other/category or duration alone. "
                "Use source save_for_backward and lifetimes."
            ),
        },
        "baseline": timeline[0],
        "final": timeline[-1],
        "maxima": maxima,
        "persistent": {
            "count": len(persistent),
            "bytes": sum(b.size for b in persistent),
            "groups": peak_groups(persistent, groups, -1),
        },
        "boundary_fc_only": {
            "count": sum(b.provenance == "free_completed_only_unverified" for b in blocks),
            "bytes": sum(
                b.size for b in blocks if b.provenance == "free_completed_only_unverified"
            ),
        },
        "stable_window": floor,
        "peaks": peaks,
        "csa_episodes": episodes,
        "snapshot_markers": [i for i, e in enumerate(trace) if e["action"] == "snapshot"],
        "groups": sorted(groups.values(), key=lambda g: g["id"]),
    }
    report["free_context_comparison"] = {
        action: {
            "events_with_retained_alloc": sum(
                e["action"] == action and event_blocks[i].alloc is not None
                for i, e in enumerate(trace)
            ),
            "different_from_alloc_stack": sum(
                e["action"] == action
                and event_blocks[i].alloc is not None
                and stack_key(e["frames"], e["stream"]) != event_blocks[i].key
                for i, e in enumerate(trace)
            ),
        }
        for action in ("free_requested", "free_completed")
    }
    marker_indices = [-1] + report["snapshot_markers"] + [len(trace) - 1]
    report["marker_windows"] = []
    for start, end in zip(marker_indices, marker_indices[1:]):
        if end > start:
            peak = max(timeline[start + 1 : end + 2], key=lambda r: r["fc_bytes"])
            report["marker_windows"].append(
                {"start_event": start, "end_event": end, "fc_peak": peak}
            )
    return report, timeline, blocks


def write_json(path, value):
    """Write a new UTF-8 JSON artifact, refusing existing files."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_tsv(path, rows, columns):
    """Write a new TSV with JSON-escaped complete stacks in single cells."""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export(report, timeline, blocks, source, output):
    """Publish new artifacts only, after all lifecycle and group checks passed."""
    paths = {
        suffix: output / (source.stem + suffix)
        for suffix in (
            ".audit.json",
            ".dynamic.tsv",
            ".timeline.tsv",
            ".lifetimes.tsv",
            ".peak_groups.tsv",
        )
    }
    require(not any(p.exists() for p in paths.values()), f"output already exists for {source.stem}")
    report["source"] = str(source.resolve())
    report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dynamic = []
    for group in report["groups"]:
        if not group["dynamic"]:
            continue
        row = {"group_id": group["id"], "stream": group["stream"], "category": group["category"]}
        for metric in METRICS:
            row[f"max_{metric}"] = group["maxima"][metric]["value"]
            row[f"max_{metric}_event"] = group["maxima"][metric]["event"]
            row[f"at_peak_{metric}"] = group["at_fc_peak"][metric]
            row[f"initial_{metric}"] = group["initial"][metric]
            row[f"final_{metric}"] = group["final"][metric]
        row["event_stats_json"] = encoded(group["events"])
        row["provenance_json"] = encoded(group["provenance_counts"])
        row["frames_json"] = encoded(group["frames"])
        row["full_call_path"] = " -> ".join(
            f"{f.get('filename')}:{f.get('line')}:{f.get('name')}" for f in group["frames"]
        )
        dynamic.append(row)
    dynamic.sort(
        key=lambda r: (-r["at_peak_pending_bytes"], -r["max_pending_bytes"], r["group_id"])
    )
    columns = (
        ["group_id", "stream", "category"]
        + [
            f"{prefix}{metric}{suffix}"
            for metric in METRICS
            for prefix, suffix in (
                ("max_", ""),
                ("max_", "_event"),
                ("at_peak_", ""),
                ("initial_", ""),
                ("final_", ""),
            )
        ]
        + ["event_stats_json", "provenance_json", "frames_json", "full_call_path"]
    )
    write_tsv(paths[".dynamic.tsv"], dynamic, columns)
    write_tsv(paths[".timeline.tsv"], timeline, ["event", *METRICS, "reserved_bytes"])
    life_rows = (
        {
            "address": b.address,
            "requested_bytes": b.size,
            "allocator_bytes": b.allocator_size,
            "stream": b.stream,
            "group_id": hashlib.sha256(b.key.encode()).hexdigest(),
            "alloc_event": b.alloc,
            "fr_event": b.fr,
            "fc_event": b.fc,
            "initial_pending": b.initial_pending,
            "persistent": b.persistent,
            "provenance": b.provenance,
        }
        for b in blocks
    )
    write_tsv(
        paths[".lifetimes.tsv"],
        life_rows,
        [
            "address",
            "requested_bytes",
            "allocator_bytes",
            "stream",
            "group_id",
            "alloc_event",
            "fr_event",
            "fc_event",
            "initial_pending",
            "persistent",
            "provenance",
        ],
    )
    peak_rows = []
    for label, peak in report["peaks"].items():
        for group in peak["groups"]:
            peak_rows.append(
                {
                    "peak": label,
                    "event": peak["event"],
                    **group,
                    "requested_sizes_json": encoded(group["requested_size_histogram"]),
                }
            )
    write_tsv(
        paths[".peak_groups.tsv"],
        peak_rows,
        [
            "peak",
            "event",
            "group_id",
            "category",
            "stream",
            *METRICS,
            "persistent_bytes",
            "requested_sizes_json",
        ],
    )
    # JSON is written last and serves as the successful audit manifest.
    write_json(paths[".audit.json"], report)
    return paths


def self_test():
    """Compare every suffix of independent generated lifecycles against an oracle."""
    import copy

    frames = [{"filename": "model.py", "name": "forward", "line": 3, "extra": "kept"}]

    def event(action, addr=1024, size=100, stream=0):
        return {
            "action": action,
            "addr": addr,
            "size": size,
            "stream": stream,
            "frames": copy.deepcopy(frames),
        }

    def snapshot(trace, active=None):
        active = active or []
        segments = []
        for addr, size, pending in active:
            segments.append(
                {
                    "device": 0,
                    "address": addr,
                    "total_size": size,
                    "stream": 0,
                    "blocks": [
                        {
                            "address": addr,
                            "size": size,
                            "requested_size": size,
                            "state": "active_awaiting_free" if pending else "active_allocated",
                            "frames": copy.deepcopy(frames),
                        }
                    ],
                }
            )
        # Reserved memory for historical allocations, disjoint from active blocks.
        segments.append(
            {
                "device": 0,
                "address": 1000000,
                "total_size": 1000000,
                "stream": 0,
                "blocks": [{"address": 1000000, "size": 1000000, "state": "inactive"}],
            }
        )
        return {"segments": segments, "device_traces": [trace]}

    checked = 0
    rng = random.Random(718)
    trace = []
    active = {}
    expected = [dict.fromkeys(METRICS, 0)]
    for _ in range(120):
        available = [a for a in (1024, 2048, 3072, 4096) if a not in active]
        choices = (["alloc"] if available else []) + (["free"] if active else [])
        action = rng.choice(choices)
        if action == "alloc":
            addr = rng.choice(available)
            size = rng.choice((8, 20, 100, 300))
            active[addr] = (size, False)
            trace.append(event("alloc", addr, size))
        else:
            addr = rng.choice(list(active))
            size, pending = active[addr]
            trace.append(event("free_completed" if pending else "free_requested", addr, size))
            if pending:
                del active[addr]
            else:
                active[addr] = (size, True)
        fc = sum(s for s, _ in active.values())
        fr = sum(s for s, p in active.values() if not p)
        count = len(active)
        pending_count = sum(p for _, p in active.values())
        expected.append(
            dict(
                fc_bytes=fc,
                fr_bytes=fr,
                pending_bytes=fc - fr,
                fc_count=count,
                fr_count=count - pending_count,
                pending_count=pending_count,
            )
        )
    final = [(a, size, pending) for a, (size, pending) in active.items()]
    for cut in range(len(trace) + 1):
        report, timeline, _ = audit(snapshot(trace[cut:], final), stable_events=7)
        for row, wanted in zip(timeline, expected[cut:]):
            require({m: row[m] for m in METRICS} == wanted, f"oracle suffix {cut}")
        for metric in METRICS:
            require(
                report["maxima"][metric]["value"] == max(r[metric] for r in expected[cut:]),
                f"oracle peak {metric}",
            )
        window = min(7, len(expected) - cut)
        floor = max(
            min(row["fc_bytes"] for row in expected[start : start + window])
            for start in range(cut, len(expected) - window + 1)
        )
        require(report["stable_window"]["fc_bytes"] == floor, "rolling floor oracle")
        checked += 1
    # A persistent block uses requested_size, never rounded block size.
    s = snapshot([], [(1024, 512, True)])
    s["segments"][0]["blocks"][0]["requested_size"] = 7
    r, _, _ = audit(s)
    require(
        r["baseline"]["fc_bytes"] == 7 and r["baseline"]["fr_bytes"] == 0,
        "persistent pending requested bytes",
    )
    checked += 1
    # Count and byte maxima need not occur together; reuse creates a new generation.
    sequence = [
        event("alloc", 1024, 100),
        event("alloc", 2048, 1),
        event("free_requested", 1024, 100),
        event("free_completed", 1024, 100),
        event("alloc", 1024, 1),
        event("free_requested", 2048, 1),
        event("free_requested", 1024, 1),
        event("free_completed", 2048, 1),
        event("free_completed", 1024, 1),
    ]
    r, _, _ = audit(snapshot(sequence))
    require(r["maxima"]["pending_bytes"] == {"value": 100, "event": 2}, "pending bytes")
    require(r["maxima"]["pending_count"] == {"value": 2, "event": 6}, "pending count")
    checked += 1
    # Full frame fields (including C++ and extra metadata) and stream distinguish groups.
    s = snapshot([], [(1024 + i * 1024, 100, False) for i in range(4)])
    s["device_traces"][0] = [event("alloc", 1024 + i * 1024) for i in range(4)]
    for i in range(4):
        e = s["device_traces"][0][i]
        seg = s["segments"][i]
        if i == 1:
            e["frames"][0]["extra"] = "distinct"
        if i == 2:
            e["frames"].append({"filename": "kernel.cpp", "line": 9, "name": "cpp_alloc"})
        if i == 3:
            e["stream"] = 7
            seg["stream"] = 7
        seg["blocks"][0]["frames"] = copy.deepcopy(e["frames"])
    r, _, _ = audit(s)
    require(len(r["groups"]) == 4, "full stack and stream grouping")
    require(all(g["dynamic"] for g in r["groups"]), "single-block dynamic groups retained")
    checked += 1
    sequence = [event("alloc"), event("free_requested"), event("free_completed")]
    sequence[-1]["frames"] = [{"filename": "other.py", "line": 1, "name": "gc_trigger"}]
    r, _, _ = audit(snapshot(sequence))
    require(r["groups"][0]["frames"] == frames, "delayed FC must not overwrite alloc stack")
    require(
        r["free_context_comparison"]["free_completed"]["different_from_alloc_stack"] == 1,
        "delayed FC attribution diagnostic",
    )
    checked += 1
    invalid = [
        snapshot([event("alloc"), event("alloc")]),
        snapshot([event("alloc"), event("free_completed")]),
        snapshot([event("free_requested")]),
        snapshot(
            [
                event("alloc"),
                event("free_requested"),
                event("free_requested"),
                event("free_completed"),
            ]
        ),
        snapshot([event("alloc"), event("free_requested", size=99), event("free_completed")]),
        snapshot([event("alloc"), event("free_requested", stream=1), event("free_completed")]),
        snapshot([event("mystery")]),
        snapshot([event("alloc")]),
        snapshot([], [(1024, 100, False), (1024, 100, False)]),
        snapshot([], [(1024, 100, False), (1050, 100, False)]),
        {"segments": []},
    ]
    for index, bad in enumerate(invalid):
        try:
            audit(bad)
        except (AuditError, KeyError):
            checked += 1
        else:
            raise AuditError(f"corruption test {index} accepted")
    return {
        "validation": "PASS",
        "cases": checked,
        "coverage": (
            "121 random-history suffix oracles; requested vs rounded pending baseline; "
            "independent count/byte maxima; full stack/stream separation; delayed FC stacks; "
            "11 corrupt allocation/free/address/stream/schema cases"
        ),
    }


def main():
    """Audit explicit inputs and publish new artifacts to the explicit output directory."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, nargs="+", help="Trusted PyTorch snapshot pickle(s)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--stable-window-events", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.input and not args.self_test:
        parser.error("--input or --self-test is required")
    try:
        output = args.output_dir.resolve()
        for source in args.input or []:
            require(
                source.resolve().parent != output and source.resolve().parent not in output.parents,
                "output must not be inside original snapshot directory",
            )
        output.mkdir(parents=True, exist_ok=True)
        if args.self_test:
            result = self_test()
            write_json(output / "self_test.json", result)
            sys.stdout.write(encoded(result) + "\n")
        summaries = []
        for source in args.input or []:
            sys.stdout.write(f"[audit] {source}\n")
            sys.stdout.flush()
            with source.open("rb") as handle:
                snapshot = pickle.load(handle)
            report, timeline, blocks = audit(snapshot, args.stable_window_events, args.device)
            paths = export(report, timeline, blocks, source, output)
            summaries.append(
                {
                    "source": report["source"],
                    "source_sha256": report["source_sha256"],
                    "validation": report["validation"],
                    "event_count": report["event_count"],
                    "baseline": report["baseline"],
                    "persistent_bytes": report["persistent"]["bytes"],
                    "persistent_count": report["persistent"]["count"],
                    "peaks": {
                        k: {f: v for f, v in p.items() if f != "groups"}
                        for k, p in report["peaks"].items()
                    },
                    "artifacts": {k: str(v) for k, v in paths.items()},
                }
            )
            peak = report["peaks"]["fc_peak"]
            sys.stdout.write(
                f"[audit] PASS device={report['device']} FC={peak['fc_bytes']/GIB:.6f} GiB "
                f"FR={peak['fr_bytes']/GIB:.6f} GiB pending={peak['pending_bytes']/GIB:.6f} GiB "
                f"event={peak['event']}\n"
            )
            sys.stdout.flush()
        if summaries:
            write_json(output / "summary.json", summaries)
    except (AuditError, OSError, KeyError, TypeError, pickle.UnpicklingError) as error:
        sys.stderr.write(f"[audit] FAIL: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
