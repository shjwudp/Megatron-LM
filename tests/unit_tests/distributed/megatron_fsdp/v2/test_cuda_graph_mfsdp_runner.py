# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused CPU tests for M-FSDP CUDA Graph recording."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.cuda_graph_runner import (
    CudaGraphRunner,
    _capture_module_topology,
    _make_module_topology_preflight,
    _normalize_forward_call,
    _renew_fsdp_compute_parameter_leaves,
    _validate_activation_recompute_lifetime,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph import (
    cuda_graph_checkpoint_phase,
)


def test_normalize_forward_call_preserves_variadic_arguments():
    """Keep variadic arguments at their original call level."""

    class VariadicModule(torch.nn.Module):
        """Module with positional, variadic, and keyword inputs."""

        def forward(self, hidden_states, /, metadata=None, *extra_states, **kwargs):
            """Return the input tensor.

            :param hidden_states: Input tensor.
            :type hidden_states: torch.Tensor
            :param metadata: Optional metadata.
            :type metadata: Any
            :param extra_states: Additional positional tensors.
            :type extra_states: tuple
            :param kwargs: Additional keyword values.
            :type kwargs: dict
            :return: Input tensor.
            :rtype: torch.Tensor
            """
            del metadata, extra_states, kwargs
            return hidden_states

    module = VariadicModule()
    hidden = torch.ones(2)
    extra = torch.full((2,), 2.0)
    args, kwargs = _normalize_forward_call(
        module, (hidden, None, extra), {"rotary": (torch.ones(2),)}
    )
    assert args == (hidden, None, extra)
    assert tuple(kwargs) == ("rotary",)


def test_module_topology_preflight_uses_cached_owners():
    """Reject slot replacement without a recursive replay-time walk."""
    module = torch.nn.Sequential(torch.nn.Linear(2, 2))
    module[0].register_buffer("scale", torch.ones(1))
    preflight = _make_module_topology_preflight(_capture_module_topology(module))

    module[0].register_buffer("offset", torch.zeros(1))
    with pytest.raises(RuntimeError, match="registered buffer topology changed"):
        preflight()
    del module[0]._buffers["offset"]

    def fail_recursive_walk(*args, **kwargs):
        """Reject an unexpected recursive walk."""
        del args, kwargs
        raise AssertionError("replay walked named_modules")

    module.named_modules = fail_recursive_walk
    preflight()


def test_activation_recompute_validates_serial_regions():
    """Accept serial regions and reject a recompute-order mismatch."""
    _validate_activation_recompute_lifetime(
        [
            ("forward", 0, 0),
            ("forward", 1, 0),
            ("forward", 2, 1),
            ("recompute", 2, 1),
            ("backward", 2, 1),
            ("recompute", 0, 0),
            ("recompute", 1, 0),
            ("backward", 1, 0),
            ("backward", 0, 0),
        ],
        module_regions=(0, 0, 1),
    )
    with pytest.raises(RuntimeError, match="complete serial checkpoint-region"):
        _validate_activation_recompute_lifetime(
            [
                ("forward", 0, 0),
                ("forward", 1, 0),
                ("recompute", 1, 0),
                ("recompute", 0, 0),
                ("backward", 1, 0),
                ("backward", 0, 0),
            ],
            module_regions=(0, 0),
        )


def test_runner_rejects_changed_checkpoint_region():
    """Require one checkpoint token for original forward and recompute."""
    module = torch.nn.Linear(4, 4)
    module._fsdp_param_groups = ()
    runner = CudaGraphRunner(graph_pool=None, activation_recompute=True)
    with cuda_graph_checkpoint_phase("forward", object()):
        runner.record_module(module, (torch.ones(2, 4),), {})
    with (
        cuda_graph_checkpoint_phase("recompute", object()),
        pytest.raises(RuntimeError, match="changed checkpoint region"),
    ):
        runner.record_module_recompute(module)


def test_capture_renews_only_internal_compute_leaves():
    """Keep optimizer-facing registered parameters stable during leaf renewal."""
    module = torch.nn.Linear(2, 2)
    compute_params = list(module.parameters())
    names = ["weight", "bias"]
    param_idx = {parameter: index for index, parameter in enumerate(compute_params)}
    buffers = [SimpleNamespace(params=compute_params, param_idx=param_idx) for _ in range(2)]
    dist_params = [torch.nn.Parameter(parameter.detach().clone()) for parameter in compute_params]
    module.weight, module.bias = dist_params
    param_group = SimpleNamespace(
        params=compute_params,
        dist_params=dist_params,
        param_idx=param_idx,
        model_weight_buffer=buffers[0],
        transpose_weight_buffer=None,
        main_weight_buffer=None,
        main_grad_buffer=buffers[1],
    )
    module._named_param_groups = [(names, param_group)]
    module._init_param_main_grad_func = lambda: None

    _renew_fsdp_compute_parameter_leaves((module,))

    assert list(module.parameters()) == dist_params
    assert all(new is not old for new, old in zip(param_group.params, compute_params))
    assert [new.data_ptr() for new in param_group.params] == [
        old.data_ptr() for old in compute_params
    ]
    assert buffers[0].params is param_group.params
    assert buffers[1].params is param_group.params
