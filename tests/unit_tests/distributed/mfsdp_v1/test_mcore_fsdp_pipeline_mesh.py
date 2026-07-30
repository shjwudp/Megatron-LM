# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _get_dp_tp_mesh, _get_hsdp_tp_mesh


class _FakeGroup:
    def __init__(self, ranks):
        self.ranks = ranks

    def size(self):
        return len(self.ranks)


@pytest.mark.parametrize("rank", [0, 5])
def test_expert_dp_tp_mesh_uses_pipeline_stage_ranks(rank):
    stage_start = 0 if rank < 4 else 4
    stage_ranks = list(range(stage_start, stage_start + 4))
    singleton_group = _FakeGroup([rank])

    with (
        patch(
            "megatron.core.distributed.fsdp.mcore_fsdp_adapter.dist.get_rank",
            return_value=rank,
        ),
        patch(
            "megatron.core.distributed.fsdp.mcore_fsdp_adapter.dist.get_world_size",
            return_value=1,
        ),
        patch(
            "megatron.core.distributed.fsdp.mcore_fsdp_adapter.dist.get_process_group_ranks",
            side_effect=lambda group: group.ranks,
        ),
    ):
        mesh = _get_dp_tp_mesh(
            singleton_group,
            singleton_group,
            ep_size=4,
            stage_ranks=stage_ranks,
        )

    torch.testing.assert_close(mesh, torch.tensor([[rank]]))


@pytest.mark.parametrize("rank", [0, 5])
def test_expert_hsdp_tp_mesh_uses_pipeline_stage_ranks(rank):
    stage_start = 0 if rank < 4 else 4
    stage_ranks = list(range(stage_start, stage_start + 4))
    singleton_group = _FakeGroup([rank])

    with (
        patch(
            "megatron.core.distributed.fsdp.mcore_fsdp_adapter.dist.get_rank",
            return_value=rank,
        ),
        patch(
            "megatron.core.distributed.fsdp.mcore_fsdp_adapter.dist.get_process_group_ranks",
            side_effect=lambda group: group.ranks,
        ),
    ):
        mesh = _get_hsdp_tp_mesh(
            singleton_group,
            singleton_group,
            singleton_group,
            ep_size=4,
            stage_ranks=stage_ranks,
        )

    torch.testing.assert_close(mesh, torch.tensor([[[rank]]]))
