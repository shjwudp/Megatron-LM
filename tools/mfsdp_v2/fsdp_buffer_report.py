"""Debug entry for pretrain_gpt.py: after the first optimizer step, report every MFSDP v2 parameter group's
persistent buffers (dtype, placements, local bytes) on profiled ranks, then keep training.

Run as the training entry (e.g. ``torchrun ... tools/mfsdp_v2/fsdp_buffer_report.py <pretrain args>``);
set MFSDP_DEBUG_RANKS to choose reporting ranks.

Answers: is the compute weight BF16 or MXFP8, how many gradient buffers exist per group,
and which placements (ZeRO level) each class uses.
"""
import collections
import os
import runpy
import sys

import torch

import megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group as pgm

_groups = []
_orig_init = pgm.FsdpParameterGroup.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    _groups.append(self)


pgm.FsdpParameterGroup.__init__ = _init


def _report_ranks():
    """Ranks that print, from MFSDP_DEBUG_RANKS (comma-separated, default "0")."""
    return {int(r) for r in os.environ.get("MFSDP_DEBUG_RANKS", "0").split(",") if r}


def _nbytes(buf):
    if buf is None:
        return 0
    if hasattr(buf, "local_buffer"):
        t = buf.local_buffer
        return t.numel() * t.element_size() if t.untyped_storage().nbytes() > 0 else 0
    total = 0
    for name in ("rowwise_data", "columnwise_data", "rowwise_scale", "columnwise_scale"):
        sub = getattr(buf, name, None)
        if sub is not None:
            total += _nbytes(sub)
    return total


def _desc(buf):
    if buf is None:
        return "None"
    kind = type(buf).__name__
    dtype = getattr(buf, "dtype", None) if kind == "DBuffer" else "mxfp8(row+col)"
    plc = ",".join(type(p).__name__.replace("Replicate", "R").replace("Partial", "P").replace("RowAtomic", "S").replace("BlockAtomic", "S32").replace("Shard", "S") for p in buf.placements)
    return f"{kind}[{dtype}]({plc}) {_nbytes(buf)/2**20:.1f}MiB"


def report():
    rank = int(os.environ.get("RANK", "0"))
    if rank not in _report_ranks():
        return
    agg = collections.defaultdict(lambda: collections.Counter())
    shown = set()
    for g in _groups:
        if not g.requires_grad:
            continue
        # Expert groups live on the expert-DP mesh; FQNs are relative to TEGroupedMLP.
        is_expert = any(k in type(g._owning_module()).__name__ for k in ("GroupedMLP", "SequentialMLP"))
        cls = "expert" if is_expert else "dense"
        a = agg[cls]
        a["main_weight"] += _nbytes(g.main_weight)
        a["model_weight"] += _nbytes(g.model_weight) if g.model_weight is not g.main_weight else 0
        a["unsharded_weight(live)"] += (
            _nbytes(g._unsharded_model_weight) if g._unsharded_model_weight is not g.model_weight else 0
        )
        a["main_grad"] += _nbytes(g.main_grad)
        a["groups"] += 1
        key = (cls, type(g.model_weight).__name__, repr(g.main_grad.placements))
        if key not in shown:
            shown.add(key)
            print(
                f"[buffers] r{rank} {cls:6s} mesh={tuple(g.mesh.shape)} "
                f"main_weight={_desc(g.main_weight)} | model_weight={_desc(g.model_weight)} | "
                f"unsharded={_desc(g._unsharded_model_weight)} | main_grad={_desc(g.main_grad)} | "
                f"opt_view_is_main_grad={g.pre_optimizer_main_grad is g.main_grad}",
                flush=True,
            )
    for cls, a in agg.items():
        print(f"[buffers] r{rank} {cls:6s} totals " + " ".join(f"{k}={v/2**30:.3f}GiB" if k != "groups" else f"groups={v}" for k, v in a.items()), flush=True)


import megatron.training.training as tr  # noqa: E402

_orig_train_step = tr.train_step
_done = [False]


def _train_step(*args, **kwargs):
    out = _orig_train_step(*args, **kwargs)
    if not _done[0]:
        _done[0] = True
        torch.cuda.synchronize()
        report()
    return out


tr.train_step = _train_step
sys.argv[0] = "pretrain_gpt.py"
runpy.run_path("pretrain_gpt.py", run_name="__main__")
