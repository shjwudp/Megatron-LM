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

"""
Example: FSDP1-compatible API backed by Megatron FSDP2.

This example demonstrates how to use the FullyShardedDataParallel drop-in
replacement with the same API as PyTorch FSDP1, but powered by Megatron FSDP2.

This mirrors the usage pattern from projects like Bagel:
https://github.com/ByteDance-Seed/Bagel/blob/main/train/fsdp_utils.py

Run:
    torchrun --nproc_per_node=4 fsdp1_compat_example.py
    torchrun --nproc_per_node=4 fsdp1_compat_example.py --use-torch-fsdp1
"""

import argparse
import functools
import os

import torch
import torch.distributed as dist
import torch.nn as nn

# ─── Toggle between Megatron FSDP2 backend and PyTorch FSDP1 ───────────────


def get_fsdp_imports(use_torch_fsdp1: bool):
    if use_torch_fsdp1:
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            CPUOffload,
            FullStateDictConfig,
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
            StateDictType,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    else:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp1_compat import (
            BackwardPrefetch,
            CPUOffload,
            FullStateDictConfig,
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
            StateDictType,
        )

    return {
        "FSDP": FullyShardedDataParallel,
        "MixedPrecision": MixedPrecision,
        "ShardingStrategy": ShardingStrategy,
        "BackwardPrefetch": BackwardPrefetch,
        "CPUOffload": CPUOffload,
        "StateDictType": StateDictType,
        "FullStateDictConfig": FullStateDictConfig,
        "transformer_auto_wrap_policy": transformer_auto_wrap_policy,
    }


# ─── Model Definition ──────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, dim: int = 512, n_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(n_layers)]
        )
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ─── FSDP Wrapping (Bagel-style) ──────────────────────────────────────────


def wrap_model_with_fsdp(model, imports, sharding_strategy="FULL_SHARD"):
    FSDP = imports["FSDP"]
    MixedPrecision = imports["MixedPrecision"]
    ShardingStrategy = imports["ShardingStrategy"]
    BackwardPrefetch = imports["BackwardPrefetch"]
    CPUOffload = imports["CPUOffload"]
    transformer_auto_wrap_policy = imports["transformer_auto_wrap_policy"]

    return FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[sharding_strategy],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=False),
    )


# ─── Checkpoint Save/Load (Bagel-style) ───────────────────────────────────


def save_full_state_dict(model, path, imports):
    FSDP = imports["FSDP"]
    StateDictType = imports["StateDictType"]
    FullStateDictConfig = imports["FullStateDictConfig"]

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            torch.save(state_dict, path)
            print(f"[rank0] Saved full state dict to {path}")


def load_full_state_dict(model, path, imports):
    FSDP = imports["FSDP"]
    StateDictType = imports["StateDictType"]
    FullStateDictConfig = imports["FullStateDictConfig"]

    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False, offload_to_cpu=False),
    ):
        model.load_state_dict(state_dict)

    if dist.get_rank() == 0:
        print(f"[rank0] Loaded full state dict from {path}")


# ─── Training Loop ────────────────────────────────────────────────────────


def train(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    imports = get_fsdp_imports(args.use_torch_fsdp1)
    backend_name = "PyTorch FSDP1" if args.use_torch_fsdp1 else "Megatron FSDP2"

    model = SimpleTransformer(dim=args.dim, n_layers=args.n_layers)
    model = wrap_model_with_fsdp(model, imports, args.sharding_strategy)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if rank == 0:
        print(f"Using backend: {backend_name}")
        print(f"Sharding strategy: {args.sharding_strategy}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {param_count:,}")

    model.train()
    for step in range(args.num_steps):
        x = torch.randn(args.batch_size, args.seq_len, args.dim, device="cuda")
        y = model(x)
        loss = y.sum() / (world_size * args.batch_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 5 == 0 and rank == 0:
            print(f"  step={step} loss={loss.item():.4f}")

    if args.save_ckpt:
        save_full_state_dict(model, "/tmp/fsdp1_compat_ckpt.pt", imports)

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-torch-fsdp1", action="store_true")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD")
    parser.add_argument("--save-ckpt", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
