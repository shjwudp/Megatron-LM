# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from megatron.core.full_cuda_graph import FullCudaGraphWrapper, _has_megatron_fsdp_v2


class _FSDPWrapper(torch.nn.Module):
    def __init__(self, use_megatron_fsdp_v2):
        super().__init__()
        self.ddp_config = SimpleNamespace(use_megatron_fsdp_v2=use_megatron_fsdp_v2)


def test_has_megatron_fsdp_v2_walks_nested_model_list():
    root = torch.nn.Module()
    v1_child = _FSDPWrapper(use_megatron_fsdp_v2=False)
    v2_child = _FSDPWrapper(use_megatron_fsdp_v2=True)

    assert not _has_megatron_fsdp_v2([root, v1_child])
    assert _has_megatron_fsdp_v2([root, torch.nn.Sequential(v2_child)])


def test_v2_fsdp_forward_only_stays_eager():
    eager_result = object()
    forward_backward_func = Mock(return_value=eager_result)
    data_iterator = [iter(())]

    with patch("megatron.core.full_cuda_graph.torch.cuda.Stream"):
        wrapper = FullCudaGraphWrapper(forward_backward_func)
    wrapper.data_read = Mock()

    result = wrapper(
        model=[torch.nn.Sequential(_FSDPWrapper(use_megatron_fsdp_v2=True))],
        data_iterator=data_iterator,
        num_microbatches=1,
        seq_length=None,
        forward_only=True,
    )

    assert result is eager_result
    wrapper.data_read.assert_not_called()
    assert forward_backward_func.call_args.kwargs["data_iterator"] is data_iterator
