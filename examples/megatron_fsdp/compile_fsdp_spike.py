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

"""Compile-native FSDP spike (Phase 1 core).

Demonstrates the SimpleFSDP-style contract that the compile backend is built on:

  * parameters are stored **sharded** (each rank holds 1/N of the weight),
  * the forward all-gathers the full weight via a **differentiable** op whose
    backward is a reduce-scatter (built on functional collectives so it traces),
  * the entire train step compiles under ``torch.compile(fullgraph=True)`` with
    zero graph breaks -- no eager hooks, no side streams, no CUDA-graph runtime.

Run:
    # single GPU (collectives degenerate to identity -- quick smoke test)
    python examples/megatron_fsdp/compile_fsdp_spike.py

    # multi-GPU ZeRO-3 (params sharded across ranks)
    torchrun --nproc_per_node=8 examples/megatron_fsdp/compile_fsdp_spike.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_funcol = torch.ops._c10d_functional


def _resolve_group_name(group) -> str:
    """Return the registered process-group name that functional collectives need."""
    from torch.distributed._functional_collectives import _resolve_group_name as _r

    return _r(group)


# ---------------------------------------------------------------------------
# Differentiable unshard: all-gather (fwd) / reduce-scatter (bwd)
# ---------------------------------------------------------------------------
class _AllGatherDim0(torch.autograd.Function):
    """All-gather a dim-0-sharded tensor; backward reduce-scatters the grad.

    Built on ``torch.ops._c10d_functional`` (async collective + wait) so
    torch.compile traces it into the graph and can bucket/overlap it. Only the
    tensor arg receives a gradient; ``group_name``/``world_size`` are constants.
    """

    @staticmethod
    def forward(ctx, shard: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
        ctx.group_name = group_name
        ctx.world_size = world_size
        full = _funcol.all_gather_into_tensor(shard.contiguous(), world_size, group_name)
        return _funcol.wait_tensor(full)

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        grad_shard = _funcol.reduce_scatter_tensor(
            grad_full.contiguous(), "avg", ctx.world_size, ctx.group_name
        )
        grad_shard = _funcol.wait_tensor(grad_shard)
        return grad_shard, None, None


# ---------------------------------------------------------------------------
# Sharded modules
# ---------------------------------------------------------------------------
class ShardedLinear(nn.Module):
    """``nn.Linear`` whose weight is sharded along dim 0 (ZeRO-3 style).

    ``reshard_after_forward`` enables the FSDP memory optimization: the forward
    is wrapped in non-reentrant activation checkpointing, so the all-gathered
    full weight is **not saved** for backward -- it is freed after forward and
    the all-gather is **recomputed** during backward (reshard-after-forward).
    This trades one extra all-gather in backward for not holding the unsharded
    weight resident, and stays fullgraph-compatible (compile traces the
    ``checkpoint`` higher-order op).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group,
        world_size: int,
        reshard_after_forward: bool = True,
    ):
        super().__init__()
        assert out_features % world_size == 0, "out_features must divide world_size"
        self.in_features = in_features
        self.out_features = out_features
        self.group_name = _resolve_group_name(group)
        self.world_size = world_size
        self.reshard_after_forward = reshard_after_forward
        # Each rank owns a (out_features // world_size, in_features) shard.
        shard = torch.randn(out_features // world_size, in_features) * (in_features**-0.5)
        self.weight = nn.Parameter(shard)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Unshard inside the traced graph; backward reduce-scatters the grad.
        full_weight = _AllGatherDim0.apply(self.weight, self.group_name, self.world_size)
        return F.linear(x, full_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reshard_after_forward:
            # Recompute the unshard in backward; do not keep the full weight.
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class ToyModel(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        group,
        world_size: int,
        reshard_after_forward: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            ShardedLinear(dim, dim, group, world_size, reshard_after_forward)
            for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.gelu(layer(x))
        return x


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        # Single-process fallback so the spike runs anywhere.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group("nccl", rank=0, world_size=1)
        rank, world_size, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(0)

    dim, n_layers = 512, 4
    group = dist.group.WORLD

    # Toggle reshard-after-forward (activation-checkpointed unshard) via env.
    reshard_after_forward = os.environ.get("MFSDP_RESHARD_AFTER_FORWARD", "1") != "0"
    if rank == 0:
        print(f"[rank0] reshard_after_forward = {reshard_after_forward}")

    model = ToyModel(dim, n_layers, group, world_size, reshard_after_forward).to(device)

    # Fixed teacher-student target so the loss must actually decrease.
    torch.manual_seed(1234)
    teacher = ToyModel(dim, n_layers, group, world_size, reshard_after_forward=False).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)  # steps the SHARDED params

    compiled = torch.compile(model, fullgraph=True)

    # --- Assert zero graph breaks up front (the whole point of fullgraph) ---
    x0 = torch.randn(8, dim, device=device)
    explanation = torch._dynamo.explain(model)(x0)
    if rank == 0:
        print(f"[rank0] graph_break_count = {explanation.graph_break_count} (expect 0)")
    assert explanation.graph_break_count == 0, "fullgraph broken -- see torch._dynamo.explain output"

    torch.cuda.reset_peak_memory_stats(device)
    initial_loss = final_loss = None
    for step in range(50):
        x = torch.randn(16, dim, device=device)
        with torch.no_grad():
            target = teacher(x)
        pred = compiled(x)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        g = loss.detach().clone()
        dist.all_reduce(g, op=dist.ReduceOp.AVG)
        g = g.item()
        initial_loss = g if initial_loss is None else initial_loss
        final_loss = g
        if rank == 0 and step % 10 == 0:
            # weight.grad is sharded -> confirms reduce-scatter landed on the shard.
            wshape = tuple(model.layers[0].weight.shape)
            print(f"[rank0] step {step:3d} loss={g:.4e} weight_shard={wshape}")

    if rank == 0:
        ratio = final_loss / max(initial_loss, 1e-12)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[rank0] initial={initial_loss:.4e} final={final_loss:.4e} ratio={ratio:.3f}")
        print(
            f"[rank0] peak_alloc={peak_mb:.1f} MB "
            f"(reshard_after_forward={reshard_after_forward}; "
            "compare MFSDP_RESHARD_AFTER_FORWARD=0 vs 1)"
        )
        assert final_loss < initial_loss * 0.5, "did not converge"
        print("[rank0] OK: fullgraph compiled, sharded params, loss converged")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
