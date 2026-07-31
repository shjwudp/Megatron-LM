# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA tests for split activation-recompute graph replay."""

import copy

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph import (
    make_graphed_callables,
)


def _checkpoint(function, *args):
    """Run one checkpoint while selecting the low-level graph phase.

    :param function: Graphed callable or ordered tuple of graphed callables.
    :type function: Callable or Tuple[Callable, ...]
    :param args: Positional checkpoint inputs.
    :type args: Any
    :return: Checkpoint output.
    :rtype: Any
    """
    functions = function if isinstance(function, tuple) else (function,)
    call_count = 0

    def run(value):
        """Dispatch the original or recompute forward.

        :param value: Current checkpoint tensor.
        :type value: torch.Tensor
        :return: Output of the selected graph sequence.
        :rtype: torch.Tensor
        """
        nonlocal call_count
        phase = "forward" if call_count == 0 else "recompute"
        call_count += 1
        for graph in functions:
            graph._cuda_graph_set_replay_phase(phase)
        for graph in functions:
            value = graph(value)
        return value

    return torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_matches_eager():
    """Match eager F, RF, and B for one checkpointed module."""
    torch.manual_seed(2026)
    module = torch.nn.Linear(8, 8, device="cuda")
    eager_module = copy.deepcopy(module)
    sample = torch.randn(4, 8, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (sample,),
        num_warmup_iters=1,
        _activation_recompute=True,
        _reuse_graph_input_output_buffers=True,
    )

    graph_input = torch.randn_like(sample, requires_grad=True)
    eager_input = graph_input.detach().clone().requires_grad_()
    graph_output = _checkpoint(graphed, graph_input)
    eager_output = torch.utils.checkpoint.checkpoint(eager_module, eager_input, use_reentrant=False)
    graph_output.square().mean().backward()
    eager_output.square().mean().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(graph_input.grad, eager_input.grad, rtol=0, atol=0)
    for graph_param, eager_param in zip(module.parameters(), eager_module.parameters()):
        torch.testing.assert_close(graph_param.grad, eager_param.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_nonreentrant_forward_preserves_grad_mode():
    """Capture non-reentrant F with grad enabled while discarding its tape."""

    class GradModeModule(torch.nn.Module):
        """Expose the grad mode used by forward capture."""

        def forward(self, value):
            """Return a grad-mode-dependent value.

            :param value: Input tensor.
            :type value: torch.Tensor
            :return: Grad-mode-dependent output tensor.
            :rtype: torch.Tensor
            """
            offset = 1.0 if torch.is_grad_enabled() else -1.0
            return value.sin() + offset

    module = GradModeModule().cuda()
    eager_module = copy.deepcopy(module)
    sample = torch.randn(4, 8, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module,
        (sample,),
        num_warmup_iters=1,
        _activation_recompute=True,
        _reuse_graph_input_output_buffers=True,
    )

    graph_input = torch.randn_like(sample, requires_grad=True)
    eager_input = graph_input.detach().clone().requires_grad_()
    graph_output = _checkpoint(graphed, graph_input)
    eager_output = torch.utils.checkpoint.checkpoint(eager_module, eager_input, use_reentrant=False)
    graph_output.sum().backward()
    eager_output.sum().backward()

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(graph_input.grad, eager_input.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_rejects_default_rng():
    """Reject RNG-consuming capture before forward and recompute can diverge."""
    sample = torch.ones(4, 8, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="do not support RNG-consuming callables"):
        make_graphed_callables(
            torch.nn.Dropout(0.5).cuda(),
            (),
            sample_kwargs={"input": sample},
            num_warmup_iters=1,
            _activation_recompute=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_rejects_custom_order():
    """Reject pipeline scheduling from the initial single-forward implementation."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)

    with pytest.raises(ValueError, match="does not support a custom order"):
        make_graphed_callables(
            module, (sample,), _activation_recompute=True, _order=[1, -1], _num_layers_per_chunk=[1]
        )
