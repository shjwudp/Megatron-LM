#!/usr/bin/env bash
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

"""Benchmark LLaMA 3: Megatron-FSDP v2 vs PyTorch FSDP2.

Uses the torchtitan model definition (``pip install torchtitan`` required).
Measures throughput (tokens/s), peak GPU memory, and ms/step.

Usage:
    torchrun --nproc_per_node=8 examples/megatron_fsdp_v2/torchtitan/bench_llama.py \\
        --backend mfsdp --model 8b --batch-size 1 --seq-len 8192 --bench-steps 20 --warmup-steps 5

Benchmark both backends in one run:
    torchrun --nproc_per_node=8 examples/megatron_fsdp_v2/torchtitan/bench_llama.py \\
        --bench-both --model 8b --batch-size 1 --seq-len 8192 --bench-steps 20 --warmup-steps 5
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh


# ---------------------------------------------------------------------------
# Model configs (Llama 3.1 — matches torchtitan's config_registry)
# ---------------------------------------------------------------------------

LLAMA3_CONFIGS: dict[str, dict] = {
    "debugmodel": {
        "dim": 256, "n_layers": 2, "n_heads": 16, "n_kv_heads": 8,
        "ffn_dim_multiplier": None, "multiple_of": 256,
        "vocab_size": 128256, "norm_eps": 1e-5,
        "rope_theta": 500000.0, "max_seq_len": 2048,
    },
    "8b": {
        "dim": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
        "ffn_dim_multiplier": 1.3, "multiple_of": 1024,
        "vocab_size": 128256, "norm_eps": 1e-5,
        "rope_theta": 500000.0, "max_seq_len": 8192,
    },
}


def build_torchtitan_llama3(model_size: str) -> nn.Module:
    """Build the real torchtitan Llama 3.1 model via its config system."""
    from torchtitan.models.llama3 import model as llama3_model
    from torchtitan.models.llama3.model_config import ModelConfig

    if model_size not in LLAMA3_CONFIGS:
        raise ValueError(f"Unknown model size '{model_size}'. Choices: {list(LLAMA3_CONFIGS)}")
    cfg = LLAMA3_CONFIGS[model_size]
    spec = ModelConfig(**cfg, n_kv_heads=cfg["n_kv_heads"],
                       ffn_dim_multiplier=cfg.get("ffn_dim_multiplier"))
    return llama3_model.Transformer(spec)


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_fsdp_torch(model: nn.Module, mesh, mp_policy, sharding_strategy: str):
    from torch.distributed.fsdp import FSDPModule, fully_shard

    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    assert isinstance(model, FSDPModule)
    return model


def wrap_fsdp_megatron(model: nn.Module, mesh, mp_policy, sharding_strategy: str):
    from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import (
        FSDPModule,
        fully_shard,
    )

    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, sharding_strategy=sharding_strategy,
                    enable_unshard_prefetch=True, enable_async_reduce_grad=True)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, sharding_strategy=sharding_strategy,
                enable_unshard_prefetch=True, enable_async_reduce_grad=True)
    assert isinstance(model, FSDPModule)
    return model


# ---------------------------------------------------------------------------
# Single-backend bench
# ---------------------------------------------------------------------------

def bench_one(args, device):
    rank = dist.get_rank()

    # --- build model ---
    with torch.device("meta"):
        model = build_torchtitan_llama3(args.model)

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))

    if args.backend == "mfsdp":
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(main_params_dtype=torch.bfloat16, main_grads_dtype=torch.bfloat16,
                                 grad_comm_dtype=torch.bfloat16)
        model = wrap_fsdp_megatron(model, mesh, mp, args.sharding_strategy)
    else:
        from torch.distributed.fsdp import MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        model = wrap_fsdp_torch(model, mesh, mp, args.sharding_strategy)

    model.to_empty(device=device)
    model.train()

    # --- optimizer ---
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=1e-4, fused=True)
    if args.backend == "mfsdp":
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fully_shard_optimizer
        fully_shard_optimizer(optim, model_parameters=params)

    torch.cuda.reset_peak_memory_stats(device)

    # --- bench loop ---
    tokens_per_step = args.batch_size * args.seq_len
    step_times = []

    for step in range(args.warmup_steps + args.bench_steps):
        tokens = torch.randint(0, 128000, (args.batch_size, args.seq_len), device=device)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        logits = model(tokens)
        loss = logits.float().mean()
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

    avg_ms = sum(step_times) / max(len(step_times), 1) * 1000
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    tps = tokens_per_step / (avg_ms / 1000)
    if rank == 0:
        print(f"[{args.backend}] avg={avg_ms:.1f} ms/step | tps={tps:.0f} tok/s | peak_mem={peak_gb:.2f} GB")
    return {"tps": tps, "peak_gb": peak_gb, "avg_ms": avg_ms}


# ---------------------------------------------------------------------------
# Dual-bench
# ---------------------------------------------------------------------------

def bench_both(args, device):
    results = {}
    for backend in ["torchfsdp", "mfsdp"]:
        args.backend = backend
        results[backend] = bench_one(args, device)
        dist.barrier()

    rank = dist.get_rank()
    if rank == 0:
        print("\n" + "=" * 60)
        print("  Comparison")
        print("=" * 60)
        base = results["torchfsdp"]
        for name, r in results.items():
            tps_diff = (r["tps"] / max(base["tps"], 1) - 1) * 100
            mem_diff = (r["peak_gb"] / max(base["peak_gb"], 1) - 1) * 100
            print(f"  {name:12s} tps={r['tps']:>10.0f} ({tps_diff:+6.1f}%)  "
                  f"peak_mem={r['peak_gb']:>6.1f} GB ({mem_diff:+6.1f}%)  "
                  f"avg={r['avg_ms']:.0f} ms/step")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLaMA 3 FSDP benchmark (Megatron-FSDP v2 vs PyTorch FSDP2)")
    p.add_argument("--backend", choices=["torchfsdp", "mfsdp"], default="mfsdp")
    p.add_argument("--bench-both", action="store_true")
    p.add_argument("--model", choices=list(LLAMA3_CONFIGS), default="debugmodel")
    p.add_argument("--sharding-strategy", default="optim_grads_params",
                   choices=["no_shard", "optim", "optim_grads", "optim_grads_params"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--bench-steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    if rank == 0:
        print(f"model={args.model} bs={args.batch_size} seq={args.seq_len} "
              f"sharding={args.sharding_strategy} world={world}")

    if args.bench_both:
        bench_both(args, device)
    else:
        bench_one(args, device)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
