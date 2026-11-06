"""Prepare copies for the internal analyzer's single-device, per-iteration contract.

PyTorch 2.12 appends the current snapshot marker after materializing the dump.
Thus the tail after its last marker belongs to the requested iteration. Allocation
and free events are copied exactly; raw inputs are never modified.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = []
    for file in sorted(args.source.glob("snapshot_rank*_iter*.pickle")):
        raw = file.read_bytes()
        snapshot = pickle.loads(raw)
        devices = {segment["device"] for segment in snapshot["segments"]}
        if len(devices) != 1:
            raise ValueError(f"Expected one allocated device: {file}")
        device = next(iter(devices))
        traces = snapshot["device_traces"][device]
        markers = [i for i, event in enumerate(traces) if event["action"] == "snapshot"]
        start = markers[-1] + 1 if markers else 0
        if markers and start == len(traces):
            raise ValueError("Unexpected trailing snapshot marker; check PyTorch dump semantics")
        snapshot["device_traces"][device] = traces[start:]
        removed = {}
        for other, trace in enumerate(snapshot["device_traces"]):
            if other == device:
                continue
            if any(event["action"] != "snapshot" for event in trace):
                raise ValueError(f"Unexpected events on device {other}: {file}")
            removed[other] = len(trace)
            snapshot["device_traces"][other] = []
        with (args.output / file.name).open("wb") as output:
            pickle.dump(snapshot, output, pickle.HIGHEST_PROTOCOL)
        audit.append(
            {
                "source": str(file.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "active_device": device,
                "original_events": len(traces),
                "slice_start": start,
                "kept_events": len(traces) - start,
                "removed_empty_device_markers": removed,
            }
        )
    if not audit:
        raise ValueError("No snapshot files found")
    (args.output / "normalization.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {len(audit)} snapshot copies; run analyzer with --no-cumulative-slice")


if __name__ == "__main__":
    main()
