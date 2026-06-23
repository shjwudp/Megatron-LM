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
CUDA Graph via forward_pre_hook — minimal working example.

Key ideas
---------
1. torch.cuda.make_graphed_callables creates a statically-shaped CUDA graph for
   each ToyBlock.forward call.  Replaying the graph is much cheaper than running
   the eager Python forward.

2. The graphed callable operates on **fixed memory addresses**.  You must copy
   new data into the capture-time buffers before each replay.

3. A forward_pre_hook on each ToyBlock intercepts the call, copies the incoming
   tensor into a pre-allocated buffer, replays the CUDA graph, and returns the
   result — effectively short-circuiting the normal forward().

How to run
----------
    torchrun --nproc-per-node=1 examples/megatron_fsdp/cuda_graph_hook_example2.py
"""

import argparse
import copy
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def copy_module_detached(m: nn.Module) -> nn.Module:
    # 1) Deepcopy module to duplicate structure, buffers, etc.
    m_copy = copy.deepcopy(m)  # uses Tensor.clone(), preserving requires_grad [web:8][web:9]

    # 2) Re-wrap parameters from their raw .data so they have no grad_fn
    for name, p in m_copy.named_parameters(recurse=True):
        with torch.no_grad():
            new_p = p.detach().clone()  # new memory, no history [web:4][web:9]
        new_p.requires_grad = p.requires_grad
        setattr(m_copy, name, nn.Parameter(new_p, requires_grad=new_p.requires_grad))

    # 3) Buffers: detach & clone to drop history, keep values
    for name, buf in m_copy.named_buffers(recurse=True):
        new_b = buf.detach().clone()
        setattr(m_copy, name, new_b)

    return m_copy

# -----------------------
# Model definition
# -----------------------

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

class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        # Populated by capture_cuda_graph() — None when graphs are disabled
        self._graphed_callable: Optional[callable] = None
        self._graph_input_buf: Optional[torch.Tensor] = None
        self._step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._step += 1
        return self.linear2(torch.relu(self.linear1(x)))

    def capture_cuda_graph(self, sample_input: torch.Tensor) -> None:
        """
        Capture self.forward into a CUDA graph.

        sample_input defines the shape / dtype / device / strides that the
        graph will be specialised on.  The returned callable must be called
        with a tensor at the same memory address as sample_input.
        """
        sample_input = sample_input.detach().requires_grad_(True)
        _forward_pre_hooks = self._forward_pre_hooks
        self._forward_pre_hooks = []
        # module2 = self.cuda()
        # # module2 = copy_module_detached(self)
        # for param in module2.parameters():
        #     del param.grad

        torch.cuda.synchronize()
        import gc
        gc.collect()

        # # Warm-up on a SIDE stream (not default)
        # s = torch.cuda.Stream()
        # with torch.cuda.stream(s):
        #     _ = module2(sample_input)
        # s.synchronize()

        graph_pool = torch.cuda.graph_pool_handle()
        detach_from_default_stream(self)  # ensure no default stream history on params/buffers

        _graphed_callable = torch.cuda.make_graphed_callables(
            self, (sample_input,), pool=graph_pool
        )
        # self.forward = module2.forward
        print(f"Captured CUDA graph for {self} with input shape {sample_input.shape}"
              f" _graphed_callable={_graphed_callable}")
        self._forward_pre_hooks = _forward_pre_hooks
        return _graphed_callable

    def _forward_pre_hook(
        self, module: nn.Module, args: Tuple[torch.Tensor]
    ) -> Optional[Tuple[torch.Tensor]]:
        """Forward pre-hook: replaces eager forward with CUDA-graph replay."""
        print("self._graphed_callable:", self._graphed_callable, self._step)
        if self._graphed_callable is None and self._step > 1:
            self._graphed_callable = True
            self.capture_cuda_graph(args[0])


# -----------------------
# CUDA graph pre-hook installer
# -----------------------

def install_cuda_graph_hooks(model: nn.Module, sample_input: torch.Tensor) -> None:
    """Walk model, capture CUDA graphs for every ToyBlock, register pre-hooks."""
    for name, module in list(model.named_modules()):
        if isinstance(module, ToyBlock):
            # captured_cuda_graph = module.capture_cuda_graph(sample_input)
            # setattr(model, name, captured_cuda_graph)  # replace module with graphed callable
            module.register_forward_pre_hook(module._forward_pre_hook, with_kwargs=False)


# -----------------------
# Distributed init
# -----------------------

def init_distributed() -> None:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())


# -----------------------
# Training loop
# -----------------------

def train(
    model: nn.Module, optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> None:
    sample_input = torch.randn(args.batch_size, args.model_dim, device="cuda")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model.train()

    for i, block in enumerate(model.modules()):
        if i == 0:
            x = torch.randn(args.batch_size, args.model_dim, device="cuda")
            block(x)
            break

    train_stream = torch.cuda.Stream()
    with torch.cuda.stream(train_stream):
        for epoch in range(args.epochs):
            for step in range(args.steps_per_epoch):
                x = torch.randn(args.batch_size, args.model_dim, device="cuda")
                if step == 1 and epoch == 0:
                    torch.cuda.synchronize()
                    torch.cuda.current_stream().synchronize()
                    for i, block in enumerate(model.modules()):
                        if i == 0:
                            continue
                        if isinstance(block, ToyBlock) and block._graphed_callable is None:
                            block.capture_cuda_graph(x)

                y = model(x)
                loss = y.sum() / (world_size * args.batch_size)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if step % args.log_interval == 0 and rank == 0:
                    print(f"[rank0] epoch={epoch} step={step} loss={loss.item():.4f}")


# -----------------------
# Entry point
# -----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA Graph via forward_pre_hook toy example"
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

    model = nn.Sequential(
        ToyBlock(args.model_dim),
        ToyBlock(args.model_dim),
        ToyBlock(args.model_dim),
    ).to("cuda")

    print("model and optimizer ready. installing CUDA graph hooks...")

    # sample_input = torch.randn(args.batch_size, args.model_dim, device="cuda")
    # install_cuda_graph_hooks(model, sample_input)

    print("good to go! running training loop with CUDA graph hooks installed...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, optimizer, args)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
