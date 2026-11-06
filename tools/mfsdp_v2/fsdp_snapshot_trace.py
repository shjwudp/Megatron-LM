"""Replay a CUDA allocator snapshot (``--record-memory-history`` / ``--memory-snapshot-path``) and summarize MFSDP v2 buffers by call site.

Usage: python fsdp_snapshot_trace.py <snapshot.pickle> [--last-iters N]

Each traced alloc is classified by the first matching frame of its Python stack:
  partial_grad  FsdpParameterGroup.allocate_partial_grad_buffer (RS input, per backward)
  reduced_grad  DBuffer.redistribute / reduce_scatter outputs inside reduce_*_gradients
  unshard_w     unshard (all-gather) of compute weights
  main_grad     persistent main_grad DBuffer (built at init)
  main_weight   persistent FP32 main weight
  model_weight  persistent compute weight (BF16 or MXFP8 planes)
  te_weight_ws  TE weight quantization workspace (get_weight_workspace / quantize)
  optimizer     optimizer state / step temporaries
  other
Reports, per category, the number of live blocks and bytes at the global peak and the
maximum number of concurrently live blocks, plus the timeline of partial_grad blocks.
"""
import argparse
import collections
import pickle

RULES = [
    ("partial_grad", ("allocate_partial_grad_buffer",)),
    ("reduced_grad", ("reduce_inner_gradients", "reduce_outer_gradients")),
    ("unshard_w", ("_allgather_model_weight", "unshard_parameters", "reallocate_storage")),
    ("te_weight_ws", ("get_weight_workspace",)),
    ("optimizer", ("step_with_ready_grads", "_init_group", "FsdpMuon", "_step_group")),
    ("main_grad", ("_initialize_buffers",)),
]


def classify(frames):
    names = [f.get("name", "") for f in frames]
    files = [f.get("filename", "") for f in frames]
    for cat, keys in RULES:
        if any(k in n for n in names for k in keys):
            if cat == "main_grad":
                # Split init-time buffers by the line inside _initialize_buffers.
                for f in frames:
                    if f.get("name") == "_initialize_buffers":
                        src = f.get("line", 0)
                        return f"init_buffer@L{src}"
            return cat
    if any("megatron_fsdp" in f for f in files):
        return "fsdp_other"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    snap = pickle.load(open(args.snapshot, "rb"))
    dev = max(range(len(snap["device_traces"])), key=lambda i: len(snap["device_traces"][i]))
    trace = snap["device_traces"][dev]
    live = {}
    cat_bytes = collections.Counter()
    cat_count = collections.Counter()
    cat_max_count = collections.Counter()
    cat_max_bytes = collections.Counter()
    total = 0
    peak = (0, None, None)
    partial_events = []
    for idx, ev in enumerate(trace):
        action = ev["action"]
        addr, size = ev["addr"], ev["size"]
        if action == "alloc":
            cat = classify(ev.get("frames", []))
            live[addr] = (size, cat)
            cat_bytes[cat] += size
            cat_count[cat] += 1
            total += size
            cat_max_count[cat] = max(cat_max_count[cat], cat_count[cat])
            cat_max_bytes[cat] = max(cat_max_bytes[cat], cat_bytes[cat])
            if cat == "partial_grad":
                partial_events.append((idx, "+", size, cat_count[cat]))
            if total > peak[0]:
                peak = (total, idx, dict(cat_bytes), dict(cat_count))
        elif action in ("free_completed",):
            if addr in live:
                size, cat = live.pop(addr)
                cat_bytes[cat] -= size
                cat_count[cat] -= 1
                total -= size
                if cat == "partial_grad":
                    partial_events.append((idx, "-", size, cat_count[cat]))
    print(f"device {dev}: {len(trace)} trace events (window), traced-live peak {peak[0]/2**30:.2f} GiB at event {peak[1]}")
    print(f"{'category':28s} {'@peak GiB':>10s} {'@peak n':>8s} {'max GiB':>9s} {'max n':>6s}")
    cats = sorted(set(cat_max_bytes), key=lambda c: -cat_max_bytes[c])
    for c in cats[: args.top]:
        print(f"{c:28s} {peak[2].get(c,0)/2**30:10.3f} {peak[3].get(c,0):8d} {cat_max_bytes[c]/2**30:9.3f} {cat_max_count[c]:6d}")
    print("\npartial_grad (RS input) lifetime: max concurrently live =", cat_max_count["partial_grad"])
    hist = collections.Counter(n for (_, op, _, n) in partial_events if op == "+")
    print("  live count right after each alloc:", dict(sorted(hist.items())))
    sizes = collections.Counter(round(s / 2**20, 1) for (_, op, s, _) in partial_events if op == "+")
    print("  alloc sizes MiB (count):", dict(sorted(sizes.items())))
    by, n = live_blocks(snap)
    print("\nlive at dump (after optimizer step):")
    for c in sorted(by, key=lambda c: -by[c])[: args.top]:
        print(f"  {c:40s} {by[c]/2**30:8.3f} GiB  n={n[c]}")


def live_blocks(snap):
    """Classify blocks live at dump time (persistent buffers predate the trace window)."""
    by = collections.Counter()
    n = collections.Counter()
    for seg in snap["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] != "active_allocated":
                continue
            frames = blk.get("frames", [])
            names = [f.get("name", "") for f in frames]
            cat = None
            for f in frames:
                if f.get("name") == "_initialize_buffers":
                    cat = f"init_buffer@L{f.get('line')}"
                    break
                if f.get("name") == "__init__" and "quantized_dbuffer" in f.get("filename", ""):
                    cat = "QuantizedDBuffer.__init__"
            if cat is None:
                cat = classify(frames)
            by[cat] += blk["size"]
            n[cat] += 1
    return by, n


if __name__ == "__main__":
    main()
