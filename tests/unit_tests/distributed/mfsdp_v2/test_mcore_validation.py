# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate per-chunk models against a global expert-parallel topology."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_adapter_v2 import FullyShardedDataParallelV2
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize("expert_parameter", [False, True])
def test_dense_chunk_in_expert_parallel_model(expert_parameter):
    config = TransformerConfig(
        num_layers=4,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        gradient_accumulation_fusion=False,
    )
    module = torch.nn.Linear(16, 16, bias=False)
    module.weight.allreduce = not expert_parameter
    groups = SimpleNamespace(
        dp_cp=Mock(),
        tp=Mock(size=Mock(return_value=1)),
        ep=Mock(size=Mock(return_value=2)),
        expt_dp=Mock(),
    )
    ddp = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        data_parallel_sharding_strategy="optim_grads",
    )
    if expert_parameter:
        with pytest.raises(ValueError, match="expert parameters require MoE transformer layers"):
            FullyShardedDataParallelV2._validate_config(config, ddp, module, groups, False)
    else:
        FullyShardedDataParallelV2._validate_config(config, ddp, module, groups, False)
