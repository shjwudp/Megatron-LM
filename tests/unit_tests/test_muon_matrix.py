# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare logical Muon splits with an explicit polynomial update oracle."""

import os

import pytest
import torch

from megatron.core.optimizer.muon_matrix import R2S10, MuonMatrixOrthogonalizer
from megatron.core.optimizer.optimizer import copy_optimizer_param_metadata


def _oracle(matrix, steps):
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.T if transposed else matrix
    x = x / x.norm().clamp_min(1e-15)
    for i in range(steps):
        a, b, c = R2S10[min(i, 9)]
        gram = x @ x.T
        polynomial = torch.addmm(gram, gram, gram, alpha=c, beta=b)
        x = torch.addmm(x, polynomial, x, beta=a)
    x = x.T if transposed else x
    return x * (max(matrix.shape) ** 0.5 * 0.2)


@pytest.mark.parametrize(
    "kind,shape,dim,split_size",
    [
        (None, (8, 12), None, None),
        ("fc1", (16, 12), 0, 8),
        ("eh", (8, 24), 1, 12),
        ("index_q", (16, 12), 0, 4),
        ("qkv", (24, 12), 0, 4),
    ],
)
@pytest.mark.parametrize("steps", [10, 12])
def test_reference_splits(kind, shape, dim, split_size, steps):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    prior = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        torch.manual_seed(123)
        source = torch.nn.Parameter(torch.randn(shape, device="cuda"))
        source.muon_split_kind, source.muon_head_dim = kind, 4
        master = source.detach().clone()
        copy_optimizer_param_metadata(master, source)
        grad = torch.randn_like(master)
        kernel = MuonMatrixOrthogonalizer(
            coefficient_type="r2s10",
            num_ns_steps=steps,
            extra_scale_factor=0.2,
            ns_eps=1e-15,
            split_head=True,
            split_fc1=True,
            split_eh=True,
        )
        parts = [grad] if dim is None else grad.split(split_size, dim=dim)
        expected = [_oracle(part.contiguous(), steps) for part in parts]
        expected = expected[0] if dim is None else torch.cat(expected, dim=dim)
        torch.testing.assert_close(kernel(master, grad), expected, atol=1e-6, rtol=1e-5)
        assert torch.count_nonzero(kernel(master, torch.zeros_like(grad))) == 0
    finally:
        torch.set_float32_matmul_precision(prior)
