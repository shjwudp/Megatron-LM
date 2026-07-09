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
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_fsdp_torch(model: nn.Module, mesh, mp_policy):
    from torch.distributed.fsdp import fully_shard

    for layer in model.layers.values():
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    return model


def wrap_fsdp_megatron(model: nn.Module, mesh, mp_policy, sharding_strategy):
    from megatron_fsdp.v2 import fully_shard

    for layer in model.layers.values():
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, sharding_strategy=sharding_strategy,
                    enable_unshard_prefetch=True, enable_async_reduce_grad=True)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, sharding_strategy=sharding_strategy,
                enable_unshard_prefetch=True, enable_async_reduce_grad=True)
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


def bench_one(args, device):
    rank = dist.get_rank()

    with torch.device("meta"):
        model, config = build_llama(args)

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))

    if args.backend == "mfsdp":
        from megatron_fsdp.v2 import MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(main_params_dtype=torch.bfloat16, main_grads_dtype=torch.bfloat16,
                                  grad_comm_dtype=torch.bfloat16)
        model = wrap_fsdp_megatron(model, mesh, mp, args.sharding_strategy)
    else:
        from torch.distributed.fsdp import MixedPrecisionPolicy as TorchMixedPrecisionPolicy
        mp = TorchMixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        model = wrap_fsdp_torch(model, mesh, mp)

    model.to_empty(device=device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=1e-4, fused=True)

    torch.cuda.reset_peak_memory_stats(device)

    tokens_per_step = args.batch_size * args.seq_len
    step_times = []

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

    avg_ms = sum(step_times) / max(len(step_times), 1) * 1000
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    tps = tokens_per_step / (avg_ms / 1000)
    if rank == 0:
        print(f"[{args.backend}] avg={avg_ms:.1f} ms/step | tps={tps:.0f} tok/s | peak_mem={peak_gb:.2f} GB")
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

    bench_one(args, device)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
