"""Summarize one completed experiment without mixing instantaneous and peak memory."""

import argparse
import json
import re
import statistics
from pathlib import Path

STEP = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
)
GIB = 2**30


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--first", type=int, default=41)
    parser.add_argument("--last", type=int, default=60)
    parser.add_argument("--timer-interval", type=int, default=10)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.world <= 0 or args.pp <= 0 or args.world % args.pp:
        parser.error("--world must be positive and divisible by positive --pp")
    if args.first < 1 or args.last < args.first or args.timer_interval <= 0:
        parser.error("Use a valid one-based window and a positive --timer-interval")
    exits = list(args.run.rglob("exit_code"))
    if not exits or any(int(file.read_text()) != 0 for file in exits):
        raise ValueError("Run has no completion record or contains a nonzero exit code")
    by_step = {}
    for log in args.run.rglob("train.log"):
        for match in STEP.finditer(log.read_text()):
            iteration = int(match[1])
            value = {
                "iteration": iteration,
                "iteration_ms": float(match[2]),
                "tflops": float(match[3]),
            }
            if iteration in by_step and by_step[iteration] != value:
                raise ValueError(f"Conflicting throughput logs at step {iteration}")
            by_step[iteration] = value
    wanted = set(range(args.first, args.last + 1))
    if wanted - by_step.keys():
        raise ValueError(f"Incomplete stable window: missing {sorted(wanted - by_step.keys())}")
    selected = [by_step[i] for i in sorted(wanted)]
    no_timer = [r for r in selected if (r["iteration"] - 1) % args.timer_interval]
    result = {
        "window": [args.first, args.last],
        "memory_scope": "whole run, lifetime allocator maxima and sampled device peak",
        "steps": selected,
        "tflops": stats([r["tflops"] for r in selected]),
        "iteration_ms": stats([r["iteration_ms"] for r in selected]),
        "without_timer": {
            "iterations": [r["iteration"] for r in no_timer],
            "tflops": stats([r["tflops"] for r in no_timer]) if no_timer else None,
            "iteration_ms": stats([r["iteration_ms"] for r in no_timer]) if no_timer else None,
        },
    }
    ranks = {}
    for file in args.run.rglob("rank*.jsonl"):
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if not rows:
            continue
        rank = rows[0]["rank"]
        if rank in ranks or {r["rank"] for r in rows} != {rank}:
            raise ValueError(f"Duplicate or mixed rank file: {file}")
        iterations = [r["iteration"] for r in rows]
        if iterations != sorted(set(iterations)):
            raise ValueError(f"Duplicate or nonmonotonic iteration metrics: {file}")
        if wanted - {r["iteration"] for r in rows}:
            raise ValueError(f"Incomplete metrics for rank {rank}")
        if any(r["device_sample_error"] for r in rows):
            raise ValueError(f"Device sampling failed for rank {rank}")
        ranks[rank] = {
            "rank": rank,
            "pp": rank // (args.world // args.pp),
            **{
                k + "_GiB": max(r[k] for r in rows) / GIB
                for k in ["max_allocated", "max_reserved", "device_sampled_peak"]
            },
            "max_sampled_non_torch_GiB": max(r["device_used"] - r["reserved"] for r in rows) / GIB,
        }
    if set(ranks) != set(range(args.world)):
        raise ValueError("Missing rank metrics or incorrect --world")
    prediction = json.loads(args.prediction.read_text()) if args.prediction else None
    stages = []
    for pp in range(args.pp):
        members = [r for r in ranks.values() if r["pp"] == pp]
        peak = max(members, key=lambda r: r["max_allocated_GiB"])
        stage = {
            "pp": pp,
            "max_allocated_rank": peak["rank"],
            **{key: max(r[key] for r in members) for key in peak if key.endswith("_GiB")},
        }
        if prediction:
            predicted = prediction["stages"][pp]
            stage.update(
                predicted_allocated_GiB=predicted["totalBytes"] / GIB,
                predicted_budget_GiB=predicted["budgetBytes"] / GIB,
                allocated_residual_GiB=stage["max_allocated_GiB"] - predicted["totalBytes"] / GIB,
                device_budget_residual_GiB=stage["device_sampled_peak_GiB"]
                - predicted["budgetBytes"] / GIB,
            )
        stages.append(stage)
    result.update(ranks=list(ranks.values()), stages=stages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: result[k] for k in ["window", "tflops", "iteration_ms", "stages"]}, indent=2)
    )


if __name__ == "__main__":
    main()
