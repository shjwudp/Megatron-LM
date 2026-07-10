#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark LLaMA 3.1: Megatron-FSDP v2 vs PyTorch FSDP2.

Uses the torchtitan model definition directly (``pip install torchtitan`` required).
Measures throughput (tokens/s), peak GPU memory, and ms/step.

Prerequisites:
    pip install torchtitan

Usage:
    torchrun --nproc_per_node=8 examples/megatron_fsdp_v2/torchtitan/bench_llama.py \\
        --backend mfsdp --flavor 8B --batch-size 1 --seq-len 8192 --bench-steps 20 --warmup-steps 5

Debug OOM with memory snapshots (loadable at https://pytorch.org/memory_viz):
    torchrun --nproc_per_node=8 examples/megatron_fsdp_v2/torchtitan/bench_llama.py \\
        --backend mfsdp --flavor 8B --batch-size 1 --seq-len 8192 \\
        --record-memory-history /tmp/mem_dump [--record-memory-history-oom-only]
"""

import argparse
import atexit
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh


# ---------------------------------------------------------------------------
# Memory history recording (for OOM debugging via pytorch.org/memory_viz)
# ---------------------------------------------------------------------------

class MemoryHistoryManager:
    """Records CUDA memory history and dumps snapshots on OOM and/or exit."""

    def __init__(self, out_dir: str, oom_only: bool = False):
        self._out_dir = out_dir
        self._oom_only = oom_only
        self._dumped = False

    def start(self):
        Path(self._out_dir).mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(
            max_entries=200000, stacks="all",
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[rank0] Memory history recording enabled, "
                  f"dump dir={self._out_dir} oom_only={self._oom_only}")

    def dump(self, tag: str = ""):
        if self._dumped:
            return
        self._dumped = True
        rank = dist.get_rank() if dist.is_initialized() else 0
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self._out_dir, f"memory_snapshot_rank{rank}{suffix}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(path)
            if rank == 0:
                print(f"[rank0] Memory snapshot dumped: {path}")
        except Exception as e:
            if rank == 0:
                print(f"[rank0] Memory snapshot dump failed: {e}")

    def stop(self):
        torch.cuda.memory._record_memory_history(enabled=None)

    def dump_on_normal_exit(self):
        if self._oom_only or self._dumped:
            return
        self.dump()

    def dump_on_oom(self):
        self.dump(tag="OOM")


def _fmt_bytes(n: int) -> str:
    for power, suffix in [(4, "TB"), (3, "GB"), (2, "MB"), (1, "KB"), (0, "B")]:
        unit = 1024 ** power
        if n >= unit:
            return f"{n / unit:.2f} {suffix}"
    return f"{n} B"


def _mem_log(tag="", rank=None):
    if rank is None:
        rank = dist.get_rank()
    alloc = torch.cuda.memory_allocated()
    max_alloc = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
    prefix = f"[rank{rank}] {tag}" if tag else f"[rank{rank}]"
    print(f"{prefix} alloc={_fmt_bytes(alloc)} max_alloc={_fmt_bytes(max_alloc)} "
          f"reserved={_fmt_bytes(reserved)} max_reserved={_fmt_bytes(max_reserved)}")


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_fsdp_torch(model: nn.Module, mesh, mp_policy):
    from torch.distributed.fsdp import fully_shard

    for layer in model.layers.values():
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    return model


def wrap_fsdp_megatron(model: nn.Module, mesh, mp_policy, sharding_strategy):
    from megatron_fsdp.v2 import fully_shard

    for layer in model.layers.values():
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    return model


# ---------------------------------------------------------------------------
# Model builder (real torchtitan — no vendor)
# ---------------------------------------------------------------------------

def build_llama(args):
    """Build a Llama 3.1 model via torchtitan's model_registry.

    Returns (model, config) tuple.
    """
    from torchtitan.models.llama3 import llama3_configs, Llama3Model

    flavor = args.flavor
    if flavor not in llama3_configs:
        raise ValueError(f"Unknown flavor '{flavor}'. Choices: {list(llama3_configs)}")
    cfg = llama3_configs[flavor]
    config = cfg() if callable(cfg) else cfg
    return Llama3Model(config), config


# ---------------------------------------------------------------------------
# Single-backend bench
# ---------------------------------------------------------------------------

def _chunked_loss(logits, chunk_size):
    """Memory-efficient mean loss over logits, chunked along dim 0."""
    num_chunks = (logits.size(0) + chunk_size - 1) // chunk_size
    total = 0.0
    count = 0
    for chunk in logits.chunk(num_chunks, dim=0):
        total += chunk.float().sum()
        count += chunk.numel()
    return total / count


def bench_one(args, device, mem_mgr=None):
    rank = dist.get_rank()

    with torch.device("meta"):
        model, config = build_llama(args)

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))

    if args.backend == "mfsdp":
        from megatron_fsdp.v2 import MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, main_grads_dtype=torch.bfloat16)
        model = wrap_fsdp_megatron(model, mesh, mp, args.sharding_strategy)
    else:
        from torch.distributed.fsdp import MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        model = wrap_fsdp_torch(model, mesh, mp)

    model.to_empty(device=device)
    model.train()

    if args.debug_fsdp and rank == 0:
        print(f"[mfsdp] debug logging enabled, "
              f"forward_order={len(model._fsdp_root_context.forward_order)} modules")
        for module in model._fsdp_root_context.forward_order:
            module._log_parameter_groups()

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=1e-4, fused=True)

    _mem_log("after_model_init", rank=rank)

    torch.cuda.reset_peak_memory_stats(device)

    tokens_per_step = args.batch_size * args.seq_len
    step_times = []
    step = 0

    try:
        for step in range(args.warmup_steps + args.bench_steps):
            tokens = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(tokens)
            loss = _chunked_loss(logits, args.loss_chunk_size)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if step >= args.warmup_steps:
                step_times.append(dt)
            if rank == 0 and step % 5 == 0:
                tag = "warmup" if step < args.warmup_steps else "bench "
                print(f"[{args.backend}] {tag} step {step:3d} | {dt * 1000:8.2f} ms")
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        if rank == 0:
            print(f"\n[rank0] OOM at step {step}")
            traceback.print_exc()
        if mem_mgr is not None:
            mem_mgr.dump_on_oom()
        raise

    avg_ms = sum(step_times) / max(len(step_times), 1) * 1000
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    tps = tokens_per_step / (avg_ms / 1000)
    if rank == 0:
        print(f"[{args.backend}] avg={avg_ms:.1f} ms/step | tps={tps:.0f} tok/s | peak_mem={peak_gb:.2f} GB")

    _mem_log("final", rank=rank)

    return {"tps": tps, "peak_gb": peak_gb, "avg_ms": avg_ms}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLaMA 3.1 FSDP benchmark (Megatron-FSDP v2 vs PyTorch FSDP2)")
    p.add_argument("--backend", choices=["torchfsdp", "mfsdp"], default="mfsdp")
    p.add_argument("--flavor", default="debugmodel")
    p.add_argument("--sharding-strategy", default="optim_grads_params",
                   choices=["no_shard", "optim", "optim_grads", "optim_grads_params"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--bench-steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--loss-chunk-size", type=int, default=2048,
                   help="Sequence chunk size for memory-efficient loss computation")
    p.add_argument("--record-memory-history", type=str, default=None, metavar="DIR",
                   help="Enable CUDA memory recording (max_entries=200000). "
                        "Dumps a snapshot to DIR on normal exit AND on OOM. "
                        "Files: memory_snapshot_rank{N}.pickle. "
                        "Loadable at https://pytorch.org/memory_viz.")
    p.add_argument("--record-memory-history-oom-only", action="store_true",
                   help="When set with --record-memory-history, only dump the "
                        "snapshot on OOM (skip normal exit dump).")
    p.add_argument("--debug-fsdp", action="store_true",
                   help="Enable verbose M-FSDP v2 debug logging (unshard/reshard/prefetch events).")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if rank == 0:
        print(f"flavor={args.flavor} bs={args.batch_size} seq={args.seq_len} "
              f"sharding={args.sharding_strategy} world={world_size}")

    mem_mgr = None
    if args.record_memory_history:
        mem_mgr = MemoryHistoryManager(
            out_dir=args.record_memory_history,
            oom_only=args.record_memory_history_oom_only,
        )
        mem_mgr.start()
        if not args.record_memory_history_oom_only:
            atexit.register(mem_mgr.dump_on_normal_exit)

    if args.debug_fsdp:
        import logging as _logging
        _logging.getLogger("megatron_fsdp").setLevel(_logging.INFO)
        import megatron_fsdp.v2.hooks as mfsdp_hooks
        mfsdp_hooks._DEBUG_FSDP = True
        if rank == 0:
            print("[mfsdp] debug logging enabled")

    try:
        bench_one(args, device, mem_mgr=mem_mgr)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if rank == 0:
            print(f"\n[rank0] OOM / CUDA error: {exc}")
            traceback.print_exc()
        if mem_mgr is not None:
            mem_mgr.dump_on_oom()
        raise

    if mem_mgr is not None:
        mem_mgr.dump_on_normal_exit()
        mem_mgr.stop()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
