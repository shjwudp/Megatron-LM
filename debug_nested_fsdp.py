"""Minimal standalone reproducer for nested FSDP hang at optimizer.step().

Usage:
    torchrun --nproc_per_node=2 debug_nested_fsdp.py
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist

# -- import fully_shard directly (assumes repo root is in PYTHONPATH) -------------------
_src = os.path.join(os.path.dirname(__file__), "megatron", "core", "distributed", "fsdp", "src")
sys.path.insert(0, _src)

from megatron_fsdp.v2.fully_shard import fully_shard  # noqa: E402
from megatron_fsdp.uneven_dtensor import get_state_dict  # noqa: E402

# -- distributed init -------------------------------------------------------------------
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

# -- mock model (MoE transformer layer with expert inside) -------------------------------


class ExpertBlock(nn.Module):
    def __init__(self, hidden=64, ffn_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MOETransformerLayer(nn.Module):
    def __init__(self, hidden=64, ffn_hidden=128):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden)
        self.experts = ExpertBlock(hidden, ffn_hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.attn(x)
        h = self.experts(h)
        return self.norm(h + x)


# -- main --------------------------------------------------------------------------------
torch.manual_seed(42)
model = MOETransformerLayer(64, 128).to(device)

print(f"[rank {rank}] wrapping experts...", flush=True)
model.experts = fully_shard(model.experts)

print(f"[rank {rank}] wrapping outer layer...", flush=True)
model = fully_shard(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"[rank {rank}] forward...", flush=True)
x = torch.randn(2, 64, device=device)
out = model(x)
loss = out.sum()

print(f"[rank {rank}] backward...", flush=True)
loss.backward()

print(f"[rank {rank}] sync...", flush=True)
torch.cuda.synchronize()

print(f"[rank {rank}] optimizer.step()...", flush=True)
optimizer.step()

print(f"[rank {rank}] get_state_dict...", flush=True)
model_sd, opt_sd = get_state_dict(model, optimizer)
assert len(model_sd) > 0

print(f"[rank {rank}] PASSED", flush=True)
dist.destroy_process_group()
