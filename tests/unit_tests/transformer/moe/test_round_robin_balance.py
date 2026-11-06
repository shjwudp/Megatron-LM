# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import apply_round_robin_logits


@pytest.mark.parametrize('tokens,experts,topk', [(128, 64, 4), (7, 5, 3)])
def test_round_robin_counts_and_ste(tokens, experts, topk):
    logits = torch.randn(tokens, experts, device='cuda', requires_grad=True)
    result = apply_round_robin_logits(logits, topk, int(os.getenv('RANK', '0')))
    counts = result.topk(topk, dim=-1).indices.flatten().bincount(minlength=experts)
    if tokens % experts == 0:
        assert counts.min() == counts.max()
    else:
        assert counts.max() - counts.min() <= topk
    gradient = torch.randn_like(logits)
    result.backward(gradient)
    torch.testing.assert_close(logits.grad, gradient, rtol=0, atol=0)
