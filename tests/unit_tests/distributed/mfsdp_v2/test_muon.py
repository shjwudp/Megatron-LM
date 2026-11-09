# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare FSDP Muon training losses with unsharded Emerging Optimizers Muon."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic

pytest.importorskip("emerging_optimizers")
from emerging_optimizers.orthogonalized_optimizers.muon import Muon

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import muon


class TinyModel(nn.Module):
    """Hidden matrices optimized by Muon, with an output layer optimized by SGD."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(8, 16, bias=False), nn.ReLU(), nn.Linear(16, 4, bias=False)
        )
        self.block2 = nn.Sequential(
            nn.Linear(4, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 1, bias=False),
            nn.Linear(1, 1),  # Output projection and bias belong to another optimizer.
        )

    def forward(self, x):
        """Run both blocks."""
        return self.block2(self.block1(x))


def _train(
    model: nn.Module, optimizers: tuple[torch.optim.Optimizer, ...], inputs: torch.Tensor
) -> torch.Tensor:
    """Train on the given batches and return the loss curve."""
    losses = []
    for x in inputs:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=inputs.dtype, enabled=inputs.dtype == torch.bfloat16):
            loss = model(x).float().sum()
        losses.append(loss.detach())
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    return torch.stack(losses)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("weight_decay_method", ["decoupled", "independent", "l2", "palm"])
@pytest.mark.parametrize(
    "nesterov,kernel_options",
    [
        pytest.param(False, {}, id="momentum"),
        pytest.param(
            True,
            dict(
                coefficient_type="simple",
                num_ns_steps=3,
                scale_mode="shape_scaling",
                extra_scale_factor=0.2,
            ),
            id="nesterov",
        ),
    ],
)
def test_muon_matches_reference(
    distributed_setup, dtype, weight_decay_method, nesterov, kernel_options
):
    """FSDP and unsharded Muon should produce identical loss curves.

    Each linear layer is its own FSDP group, so several groups' communication is in flight at once.
    """
    world_size = distributed_setup.world_size
    if world_size < 2 or world_size % 2:
        pytest.skip("Requires an even number of ranks for two-rank DP meshes.")
    device = distributed_setup.device
    # With more than two GPUs, also exercise meshes whose global ranks do not start at zero.
    mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("replica", "dp"))["dp"]
    torch.manual_seed(1234)
    model = TinyModel().to(device=device, dtype=dtype)
    # Match FSDP's FP32 optimizer weights; autocast handles BF16 computation.
    reference = copy.deepcopy(model).float()
    reference_params = dict(reference.named_parameters())
    with fully_shard_context(FsdpContext(device=device)):
        for layer in (m for m in model.modules() if isinstance(m, nn.Linear)):
            fully_shard(
                layer,
                mesh=mesh,
                placements=Placements(
                    dp_axes=[0],
                    parameter=[RowAtomic()],
                    gradient=[RowAtomic()],
                    optimizer=[RowAtomic()],
                ),
            )
    params = dict(model.named_parameters())
    kwargs = dict(
        lr=0.05,
        momentum=0.9,
        nesterov=nesterov,
        weight_decay_method=weight_decay_method,
        **kernel_options,
    )
    excluded_names = ["block2.3.weight", "block2.3.bias"]
    muon_names = [name for name in params if name not in excluded_names]
    optimizer = muon.FsdpMuon([params[name] for name in muon_names], **kwargs)
    reference_optimizer = Muon([reference_params[name] for name in muon_names], **kwargs)
    other_optimizer = torch.optim.SGD([params[n] for n in excluded_names], lr=0.01, foreach=False)
    fully_shard_optimizer(other_optimizer)
    reference_other = torch.optim.SGD([reference_params[n] for n in excluded_names], lr=0.01)

    # Every rank uses identical batches, so the reference needs no gradient reduction.
    inputs = torch.randn(5, 4, 8, device=device, dtype=dtype)
    reference_losses = _train(reference, (reference_optimizer, reference_other), inputs)
    losses = _train(model, (optimizer, other_optimizer), inputs)
    torch.testing.assert_close(losses, reference_losses, atol=0, rtol=0)


@pytest.mark.parametrize("strategy", ["sharded", "hybrid"])
def test_configured_muon_matches_split_reference(distributed_setup, strategy):
    """Configured owner-compute Muon must match complete-matrix reference updates."""
    from types import SimpleNamespace

    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
    from megatron.core.optimizer.fsdp_muon import MCoreFsdpMuon

    world = distributed_setup.world_size
    if world < 4 or world % 4:
        pytest.skip("Needs a multiple of four ranks")
    mesh = init_device_mesh("cuda", (world // 4, 2, 2), mesh_dim_names=("copy", "outer", "inner"))
    mesh = mesh["outer", "inner"] if strategy == "hybrid" else mesh["inner"]
    torch.manual_seed(171)
    model = TinyModel().to(distributed_setup.device)
    reference = copy.deepcopy(model)
    with fully_shard_context(FsdpContext(device=distributed_setup.device)):
        for layer in (m for m in model.modules() if isinstance(m, nn.Linear)):
            fully_shard(
                layer,
                mesh=mesh,
                placements=Placements(
                    dp_axes=list(range(mesh.ndim)),
                    parameter=[RowAtomic()] * mesh.ndim,
                    gradient=[RowAtomic()] * mesh.ndim,
                    optimizer=[RowAtomic()] * mesh.ndim,
                ),
            )
    params, refs = dict(model.named_parameters()), dict(reference.named_parameters())
    for name, kind in (
        ("block1.0.weight", "fc1"),
        ("block1.2.weight", "eh"),
        ("block2.0.weight", "index_q"),
    ):
        for mapping in (params, refs):
            mapping[name].muon_split_kind = kind
            mapping[name].muon_head_dim = 4
    scalar = ("block2.3.weight", "block2.3.bias")
    matrices = [name for name in params if name not in scalar]
    config = OptimizerConfig(
        lr=0.01,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_fp32_matmul_prec="high",
        muon_coefficient_type="r2s10",
        muon_num_ns_steps=10,
        muon_extra_scale_factor=0.2,
        muon_ns_eps=1e-15,
        muon_split_qkv=False,
        muon_split_head=True,
        muon_split_fc1=True,
        muon_split_eh=True,
    )
    optimizer = MCoreFsdpMuon(
        [params[n] for n in matrices],
        config,
        SimpleNamespace(num_attention_heads=2, num_query_groups=2, kv_channels=4),
    )
    ref_optimizer = TensorParallelMuon(
        [refs[n] for n in matrices],
        lr=0.01,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        fp32_matmul_prec="high",
        coefficient_type="r2s10",
        num_ns_steps=10,
        extra_scale_factor=0.2,
        ns_eps=1e-15,
        split_qkv=False,
        split_head=True,
        split_fc1=True,
        split_eh=True,
    )
    scalar_optimizer = torch.optim.AdamW([params[n] for n in scalar], lr=0.01, foreach=False)
    fully_shard_optimizer(scalar_optimizer)
    ref_scalar = torch.optim.AdamW([refs[n] for n in scalar], lr=0.01, foreach=False)
    inputs = torch.randn(5, 4, 8, device=distributed_setup.device)
    ref_losses = _train(reference, (ref_optimizer, ref_scalar), inputs)
    losses = _train(model, (optimizer, scalar_optimizer), inputs)
    torch.testing.assert_close(losses, ref_losses, atol=0, rtol=0)
    from torch.distributed.tensor import Replicate

    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
        get_containing_parameter_group,
    )

    # RowAtomic packs several tensors into one buffer. A tensor's row owner can
    # differ from ordinary DTensor Shard(0); reconstruct through the buffer layout.
    names_by_id = {id(p): name for name, p in params.items()}
    groups = dict.fromkeys(get_containing_parameter_group(p) for p in params.values())
    for group in groups:
        full = group.main_weight.redistribute([Replicate()] * group.mesh.ndim)
        for i, parameter in enumerate(group.fsdp_parameters):
            name = names_by_id[id(parameter.sharded)]
            torch.testing.assert_close(full.get_tensor_view(i), refs[name], atol=0, rtol=0)
