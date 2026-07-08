# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression tests for gradient statistics with outer optimizer sharding."""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import _get_fsdp_grad_stats_domain
from megatron.core.optimizer.clip_grads import (
    _all_reduce_grad_stats,
    get_grad_norm_fp32,
)
from megatron.core.optimizer.optimizer import MegatronOptimizer


@pytest.mark.parametrize(
    "op", [torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.MAX]
)
def test_grad_stats_reduce_each_composite_group(monkeypatch, op):
    """Every scalar reduction visits both orthogonal process groups in order."""
    intra_group = object()
    inter_group = object()
    calls = []

    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda _tensor, op=None, group=None: calls.append((op, group)),
    )

    _all_reduce_grad_stats(torch.tensor(1.0), op, (intra_group, inter_group))

    assert calls == [(op, intra_group), (op, inter_group)]

    calls.clear()
    _all_reduce_grad_stats(torch.tensor(1.0), op, intra_group)
    assert calls == [(op, intra_group)]

    calls.clear()
    _all_reduce_grad_stats(torch.tensor(1.0), op, None)
    assert calls == [(op, None)]


def test_has_grad_norm_group_reduces_each_composite_group(monkeypatch):
    """The cached group-presence flag must cover the full HSDP Cartesian domain."""
    intra_group = object()
    inter_group = object()
    calls = []
    real_tensor = torch.tensor

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda data, dtype=None, device=None: real_tensor(data, dtype=dtype),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda _tensor, op=None, group=None: calls.append((op, group)),
    )

    class MockOptimizer:
        has_grad_norm_group = MegatronOptimizer.has_grad_norm_group

        def get_parameters(self):
            return [SimpleNamespace(grad_norm_group="mtp")]

        def get_grad_stats_parallel_group(self):
            return (intra_group, inter_group)

    optimizer = MockOptimizer()
    assert optimizer.has_grad_norm_group("mtp") is True
    assert calls == [
        (torch.distributed.ReduceOp.MAX, intra_group),
        (torch.distributed.ReduceOp.MAX, inter_group),
    ]

    calls.clear()
    assert optimizer.has_grad_norm_group("mtp") is True
    assert calls == []


def test_composite_groups_match_full_cartesian_grad_norm():
    """Two orthogonal reductions must reproduce the known full four-rank norm."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("run with torchrun --nproc_per_node=4")

    owns_default_group = not torch.distributed.is_initialized()
    if owns_default_group:
        torch.distributed.init_process_group(backend="gloo")

    try:
        rank = torch.distributed.get_rank()
        group_specs = ([0, 1], [2, 3], [0, 2], [1, 3])
        rank_groups = {}
        for ranks in group_specs:
            group = torch.distributed.new_group(ranks=ranks, backend="gloo")
            if rank in ranks:
                rank_groups[tuple(ranks)] = group

        intra_group = rank_groups[(0, 1) if rank < 2 else (2, 3)]
        inter_group = rank_groups[(0, 2) if rank % 2 == 0 else (1, 3)]
        local_grad = torch.tensor([float(rank + 1)])

        grad_norm = get_grad_norm_fp32(
            [local_grad],
            norm_type=3.0,
            grad_stats_parallel_group=(intra_group, inter_group),
        )

        # sum(rank_value**3) = 1 + 8 + 27 + 64 = 100.
        assert float(grad_norm) == pytest.approx(100.0 ** (1.0 / 3.0), rel=1.0e-6)
    finally:
        torch.distributed.barrier()
        if owns_default_group:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("use_v2", "inner_strategy", "outer_strategy", "has_inter_group", "expected"),
    [
        pytest.param(
            True, "optim_grads_params", "optim", True, "composite", id="v2_outer_optim"
        ),
        pytest.param(
            False, "optim_grads_params", "optim", True, "intra", id="v1_outer_optim"
        ),
        pytest.param(
            True, "optim_grads_params", "no_shard", True, "intra", id="outer_no_shard"
        ),
        pytest.param(True, "no_shard", "no_shard", True, "mp", id="inner_no_shard"),
        pytest.param(
            True, "optim_grads_params", "optim", False, "intra", id="no_outer_dim"
        ),
    ],
)
def test_fsdp_grad_stats_domain(
    use_v2,
    inner_strategy,
    outer_strategy,
    has_inter_group,
    expected,
):
    """Only v2 outer optimizer sharding adds the orthogonal outer dimension."""
    mp_group = object()
    intra_group = object()
    inter_group = object() if has_inter_group else None
    ddp_config = SimpleNamespace(
        use_megatron_fsdp_v2=use_v2,
        data_parallel_sharding_strategy=inner_strategy,
        outer_dp_sharding_strategy=outer_strategy,
    )

    domain = _get_fsdp_grad_stats_domain(
        ddp_config,
        mp_group,
        intra_group,
        inter_group,
    )

    if expected == "composite":
        assert domain == (intra_group, inter_group)
    elif expected == "mp":
        assert domain is mp_group
    else:
        assert domain is intra_group
