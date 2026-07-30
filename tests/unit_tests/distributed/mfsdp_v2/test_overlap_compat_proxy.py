# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import utils


def test_find_megatron_fsdp_returns_cached_experimental_proxy(monkeypatch):
    root = torch.nn.Module()
    root._prepare_forward_parameters = mock.Mock()
    root.pre_backward = mock.Mock()
    reduce_scatter_stream = object()
    current_stream = mock.Mock()
    root.context = SimpleNamespace(
        current_stream=lambda: current_stream,
        reduce_scatter_stream=reduce_scatter_stream,
    )
    monkeypatch.setattr(utils, "_find_experimental_fsdp_root", lambda _model: root)

    proxy = utils.find_megatron_fsdp(root)

    assert proxy is utils.find_megatron_fsdp(root)
    assert proxy.ddp_config.data_parallel_sharding_strategy == "optim_grads_params"

    layer = SimpleNamespace(
        pre_forward=mock.Mock(),
        pre_backward=mock.Mock(),
        post_forward=mock.Mock(),
    )
    proxy._replace_param_with_raw_if_needed()
    proxy.pre_forward()
    proxy.pre_backward()
    proxy.pre_forward_module(layer)
    proxy.pre_backward_module(layer)
    proxy.post_forward_release_module(layer)
    proxy.post_backward_release_module(layer)
    proxy.post_backward()

    root._prepare_forward_parameters.assert_called_once_with()
    root.pre_backward.assert_called_once_with()
    layer.pre_forward.assert_called_once_with()
    layer.pre_backward.assert_called_once_with()
    layer.post_forward.assert_called_once_with()
    current_stream.wait_stream.assert_called_once_with(reduce_scatter_stream)
