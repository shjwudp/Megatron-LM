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
CUDA Graph via forward_pre_hook + TracePoolAllocator + fake unshard/reshard.

Three concepts combined in one example:

1.  ``torch.cuda.make_graphed_callables`` — capture forward() as a CUDA graph
    and replay from ``register_forward_pre_hook``.

2.  ``TracePoolAllocator`` — two-phase bucket allocator.  Micro-batch 0 traces
    alloc/free events; ``plan()`` builds a static pool; subsequent micro-batches
    reuse the pool without calling ``torch.empty``.

3.  **Fake unshard / reshard** — allocate a flat buffer, copy params in
    (simulating all-gather), then free it (simulating reshard).  No real NCCL
    collectives — the point is exercising the allocator's memory pattern.

Lifecycle
---------

Micro-batch 0 (trace)
    root_pre_forward      → forward_phase = True
    unshard               → allocator.allocate("unshard_buf") → copy params
    CUDA-graph capture    → graphs recorded while params point at unshard buf
    reshard               → allocator.free("unshard_buf") → restore params
    backward              → alloc/free for grad accumulation
    post_backward         → plan() → optimized phase → enable flexible mode

Micro-batches 1+
    root_pre_forward      → disable flexible, reset_cursor
    unshard               → allocator.allocate("unshard_buf") from pool
    CUDA-graph replay     → (same buffer addresses as capture → correct)
    reshard               → allocator.free("unshard_buf")

How to run
----------
    torchrun --nproc-per-node=1 examples/megatron_fsdp/cuda_graph_hook_example.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import (
        TracePoolAllocator,
    )
except ImportError:
    from megatron_fsdp.v2.allocator import TracePoolAllocator


# ---------------------------------------------------------------------------
# ToyBlock — the compute unit
# ---------------------------------------------------------------------------


class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


# ---------------------------------------------------------------------------
# FakeFSDPWrapper — fake unshard / reshard + CUDA graph via pre-hook
# ---------------------------------------------------------------------------

def detach_from_default_stream(module: nn.Module):
    """Re-allocate all params/buffers on a side stream so they have no default-stream history."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for name, param in module.named_parameters():
            new_data = torch.empty_like(param)
            new_data.copy_(param)
            param.data = new_data  # .data swap preserves the Parameter object (optimizer still valid!)
        for name, buf in module.named_buffers():
            new_data = torch.empty_like(buf)
            new_data.copy_(buf)
            buf.data = new_data
    s.synchronize()

class FakeFSDPWrapper(nn.Module):
    """Wraps a sub-module, fakes FSDP unshard/reshard, manages CUDA graph.

    - On every forward: unshard → (CUDA graph or eager) → reshard.
    - CUDA graph is captured on the *wrapper* so it sees the unsharded param
      addresses.  After ``plan()`` the pool guarantees the same addresses.
    """

    _GRAPH_INPUT_KEY = "_graph_input"

    def __init__(
        self,
        module: nn.Module,
        allocator: TracePoolAllocator,
        layer_idx: int,
    ):
        super().__init__()
        self.module = module
        self._allocator = allocator
        self._layer_idx = layer_idx

        # unshard / reshard bookkeeping
        self._original_params: Dict[int, torch.Tensor] = {}
        self._unshard_buf: Optional[torch.Tensor] = None
        self._unshard_buf_numel: int = 0

        # CUDA graph
        self._graphed_forward: Optional[callable] = None
        self._graph_captured: bool = False
        self._graph_input_buf: Optional[torch.Tensor] = None
        self._graph_output: Optional[torch.Tensor] = None
        self._step = 0
        self._training_stream = torch.cuda.Stream()

    # ---- unshard / reshard ----------------------------------------------

    def _buf_key(self) -> str:
        return f"layer_{self._layer_idx}_unshard"

    def unshard(self):
        numel = sum(p.numel() for p in self.module.parameters())
        if numel == 0:
            return
        self._unshard_buf_numel = numel
        bucket = self._allocator.allocate(
            key=self._buf_key(),
            size=numel,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        self._unshard_buf = bucket.data

        offset = 0
        for p in self.module.parameters():
            self._original_params[id(p)] = p.data
            flat = self._unshard_buf[offset : offset + p.numel()].view_as(p)
            flat.copy_(p)
            p.data = flat
            offset += p.numel()

    def reshard(self):
        if self._unshard_buf is None:
            return
        for p in self.module.parameters():
            orig = self._original_params.pop(id(p), None)
            if orig is not None:
                p.data = orig
        self._allocator.free(key=self._buf_key())
        self._unshard_buf = None

    # ---- forward (orchestrates unshard → compute → reshard) -------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # with torch.cuda.stream(self._training_stream):
        self.unshard()
        try:
            # Capture on first optimized forward (before computing the real output)
            if not self._graph_captured and (self._allocator._phase == "optimized" or self._step > 1):
                self._capture_graph(x)

            if self._graphed_forward is not None:
                return self._graphed_forward(x)
            else:
                return self.module(x)
        finally:
            self.reshard()
        self._step += 1

    # ---- CUDA graph capture ---------------------------------------------

    def _capture_graph(self, sample_x: torch.Tensor):
        """Capture the compute-only portion (module.forward) while unsharded."""
        sample_x = sample_x.detach().clone().requires_grad_(True)
        detach_from_default_stream(self.module)
        torch.cuda.make_graphed_callables(
            self.module, (sample_x,)
        )
        self._graph_captured = True
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"  [CUDA graph] captured layer {self._layer_idx}")


# ---------------------------------------------------------------------------
# TrainContext — drives TracePoolAllocator lifecycle
# ---------------------------------------------------------------------------


class TrainContext:
    def __init__(self, allocator: TracePoolAllocator):
        self.allocator = allocator
        self.forward_phase = False
        self.backward_phase = False

    @property
    def phase(self) -> str:
        return self.allocator.phase

    def root_pre_forward(self):
        self.forward_phase = True
        self.backward_phase = False
        if self.phase == "optimized":
            self.allocator.disable_flexible_mode()
            self.allocator.reset_cursor()

    def root_pre_backward(self, root_module):
        self.forward_phase = False
        self.backward_phase = True
        for module in root_module.modules():
            if isinstance(module, FakeFSDPWrapper):
                module.unshard()

    def root_post_backward(self, root_module):
        self.backward_phase = False
        for module in root_module.modules():
            if isinstance(module, FakeFSDPWrapper):
                module.reshard()

        if self.phase == "trace":
            total_elems = self.allocator.plan()
            self.allocator.reset_cursor()
            self.allocator.enable_flexible_mode()
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(
                    f"  [plan] total_elems={total_elems} "
                    f"({total_elems * 4} B)  phase={self.phase}"
                )


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------


def init_distributed() -> None:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------


def build_model(
    dim: int, n_layers: int, allocator: TracePoolAllocator
) -> nn.Sequential:
    wrappers = []
    for i in range(n_layers):
        block = ToyBlock(dim)
        wrapper = FakeFSDPWrapper(block, allocator, layer_idx=i)
        wrappers.append(wrapper)
    return nn.Sequential(*wrappers).to("cuda")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    ctx: TrainContext,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model.train()

    for epoch in range(args.epochs):
        for s in range(args.steps_per_epoch):
            ctx.root_pre_forward()

            x = torch.randn(args.batch_size, args.model_dim, device="cuda")
            y = model(x)
            loss = y.sum() / (world_size * args.batch_size)

            ctx.root_pre_backward(model)
            loss.backward()

            ctx.root_post_backward(model)

            optimizer.step()
            optimizer.zero_grad()

            if s % args.log_interval == 0 and rank == 0:
                print(
                    f"[rank0] epoch={epoch} step={s} loss={loss.item():.4f} "
                    f"phase={ctx.phase}"
                )

    if rank == 0:
        print(f"\n{ctx.allocator.dump_trace()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA Graph + TracePoolAllocator + fake unshard/reshard"
    )
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_distributed()

    allocator = TracePoolAllocator()
    ctx = TrainContext(allocator)
    model = build_model(args.model_dim, args.n_layers, allocator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, optimizer, args, ctx)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
