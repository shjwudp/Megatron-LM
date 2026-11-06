"""Print the reduce-scatter input (partial_grad) and unsharded weight timelines of the last
backward in a CUDA allocator snapshot, with the owning layer and stream.

Usage: python fsdp_rs_timeline.py <snapshot.pickle> [--events N]
"""
import argparse
import pickle
import re


def owner(frames):
    """Best-effort owner: the nearest Megatron frame that names a layer/module."""
    for f in frames:
        n = f.get("name", "")
        if n in ("post_backward", "_reduce_gradient_groups", "unshard", "pre_backward", "pre_forward"):
            return n
    return "?"


def kind(frames):
    names = [f.get("name", "") for f in frames]
    if any("allocate_partial_grad_buffer" in n for n in names):
        return "RS_in"
    if any(n in ("reduce_inner_gradients", "reduce_outer_gradients") for n in names):
        return "RS_out"
    if any(n in ("_allgather_model_weight", "unshard_parameters") for n in names):
        return "AG_w"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--events", type=int, default=60)
    args = ap.parse_args()
    snap = pickle.load(open(args.snapshot, "rb"))
    dev = max(range(len(snap["device_traces"])), key=lambda i: len(snap["device_traces"][i]))
    trace = snap["device_traces"][dev]
    live = {}
    rows = []
    for idx, ev in enumerate(trace):
        a = ev["action"]
        if a == "alloc":
            k = kind(ev.get("frames", []))
            if k:
                live[ev["addr"]] = (k, ev["size"], ev.get("stream", 0), idx)
                rows.append((idx, "+", k, ev["size"], ev.get("stream", 0), owner(ev.get("frames", []))))
        elif a == "free_requested" and ev["addr"] in live:
            k, size, stream, _ = live[ev["addr"]]
            rows.append((idx, "req-free", k, size, ev.get("stream", 0), ""))
        elif a == "free_completed" and ev["addr"] in live:
            k, size, stream, _ = live.pop(ev["addr"])
            rows.append((idx, "-", k, size, ev.get("stream", 0), ""))
    # Last backward = after the last RS_in alloc of the last step, go back far enough.
    rs_idx = [i for i, r in enumerate(rows) if r[2] == "RS_in" and r[1] == "+"]
    start = max(0, rs_idx[-1] - args.events) if rs_idx else 0
    counts = {"RS_in": 0, "RS_out": 0, "AG_w": 0}
    for r in rows[:start]:
        if r[1] == "+":
            counts[r[2]] += 1
        elif r[1] == "-":
            counts[r[2]] -= 1
    print(f"{'event':>7s} {'op':8s} {'kind':6s} {'MiB':>8s} {'stream':>14s}  live(RS_in,RS_out,AG_w)  site")
    for r in rows[start:]:
        if r[1] == "+":
            counts[r[2]] += 1
        elif r[1] == "-":
            counts[r[2]] -= 1
        print(f"{r[0]:7d} {r[1]:8s} {r[2]:6s} {r[3]/2**20:8.1f} {r[4]:14d}  "
              f"({counts['RS_in']},{counts['RS_out']},{counts['AG_w']})  {r[5]}")


if __name__ == "__main__":
    main()
