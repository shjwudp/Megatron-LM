# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA tests for split activation-recompute graph replay."""

import copy

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.te_graph_runtime.graph import (
    make_graphed_callables,
)
from megatron.core.tensor_parallel.random import is_checkpointing


def _checkpoint(function, *args, use_reentrant):
    """Run one checkpoint while selecting the low-level graph phase."""
    functions = function if isinstance(function, tuple) else (function,)
    call_count = 0

    def run(value):
        nonlocal call_count
        phase = "forward" if call_count == 0 else "recompute"
        call_count += 1
        for graph in functions:
            graph._cuda_graph_set_replay_phase(phase)
        for graph in functions:
            value = graph(value)
        return value

    return torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=use_reentrant)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "nonreentrant"])
def test_activation_recompute_matches_eager_region(use_reentrant):
    """Match eager F, RF, and B for two modules in one checkpoint region.

    :param use_reentrant: Whether checkpointing uses reentrant autograd.
    :type use_reentrant: bool
    """
    torch.manual_seed(2026)
    first = torch.nn.Linear(8, 8, device="cuda")
    second = torch.nn.Linear(8, 8, device="cuda")
    eager_first = copy.deepcopy(first)
    eager_second = copy.deepcopy(second)
    sample = torch.randn(4, 8, device="cuda", requires_grad=True)
    first_graph, second_graph = make_graphed_callables(
        (first, second),
        ((sample,), (sample.detach().clone().requires_grad_(),)),
        num_warmup_iters=1,
        _input_output_aliases=({}, {0: (0, 0)}),
        _activation_recompute=True,
        _activation_recompute_regions=(0, 0),
        _reuse_graph_input_output_buffers=True,
    )

    graph_input = torch.randn_like(sample, requires_grad=True)
    eager_input = graph_input.detach().clone().requires_grad_()
    graph_output = _checkpoint(
        (first_graph, second_graph), graph_input, use_reentrant=use_reentrant
    )
    eager_output = torch.utils.checkpoint.checkpoint(
        lambda value: eager_second(eager_first(value)), eager_input, use_reentrant=use_reentrant
    )
    graph_output.square().mean().backward()
    eager_output.square().mean().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(graph_input.grad, eager_input.grad, rtol=0, atol=0)
    graph_params = tuple(first.parameters()) + tuple(second.parameters())
    eager_params = tuple(eager_first.parameters()) + tuple(eager_second.parameters())
    for graph_param, eager_param in zip(graph_params, eager_params):
        torch.testing.assert_close(graph_param.grad, eager_param.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_rejects_overlapping_forward():
    """Reject a second forward before it overwrites the pending graph state."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, _activation_recompute=True
    )

    first_input = torch.randn_like(sample, requires_grad=True)
    first_output = _checkpoint(graphed, first_input, use_reentrant=True)
    with pytest.raises(RuntimeError, match="forward_done"):
        _checkpoint(graphed, torch.randn_like(sample, requires_grad=True), use_reentrant=True)
    first_output.sum().backward()

    second_input = torch.randn_like(sample, requires_grad=True)
    _checkpoint(graphed, second_input, use_reentrant=True).sum().backward()
    assert first_input.grad is not None
    assert second_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_release_pending_invalidates_nonreentrant_output():
    """Release an abandoned non-reentrant forward and reject its backward."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, _activation_recompute=True
    )

    abandoned_input = torch.randn_like(sample, requires_grad=True)
    abandoned = _checkpoint(graphed, abandoned_input, use_reentrant=False)
    assert graphed.release_pending()
    fresh_input = torch.randn_like(sample, requires_grad=True)
    fresh = _checkpoint(graphed, fresh_input, use_reentrant=False)
    with pytest.raises(RuntimeError, match="released or superseded"):
        abandoned.sum().backward()
    fresh.sum().backward()
    assert fresh_input.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_release_pending_restarts_reentrant_forward():
    """Release a reentrant forward before starting a fresh F, RF, and B."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, _activation_recompute=True
    )

    abandoned = _checkpoint(
        graphed, torch.randn_like(sample, requires_grad=True), use_reentrant=True
    )
    del abandoned
    assert graphed.release_pending()
    fresh_input = torch.randn_like(sample, requires_grad=True)
    _checkpoint(graphed, fresh_input, use_reentrant=True).sum().backward()
    assert fresh_input.grad is not None


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
def test_activation_recompute_inference_outputs_are_independent():
    """Clone no-grad outputs so a later replay cannot overwrite them."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, _activation_recompute=True
    )

    with torch.no_grad():
        graphed._cuda_graph_set_replay_phase("inference")
        first = graphed(input=torch.full_like(sample, 2.0))
        expected = first.clone()
        graphed._cuda_graph_set_replay_phase("inference")
        second = graphed(input=torch.full_like(sample, 3.0))

    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU")
def test_activation_recompute_inference_shape_mismatch_falls_back_to_eager():
    """Run eager inference when input metadata does not match the graph."""
    module = torch.nn.Linear(4, 4, device="cuda")
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(
        module, (), sample_kwargs={"input": sample}, num_warmup_iters=1, _activation_recompute=True
    )
    inference_input = torch.randn(3, 4, device="cuda")
    expected = torch.nn.functional.linear(
        inference_input, module.weight.detach(), module.bias.detach()
    )

    graphed._cuda_graph_set_replay_phase("inference")
    with torch.no_grad():
        actual = graphed(inference_input)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE checkpoint requires a GPU")
def test_te_checkpoint_recompute_has_active_graph_task():
    """Identify actual TE non-reentrant recompute through the autograd graph task."""
    te = pytest.importorskip("transformer_engine.pytorch")
    te_checkpoint = pytest.importorskip("transformer_engine.pytorch.distributed").checkpoint
    module = te.Linear(4, 4, device="cuda")
    observations = []

    def record_phase(_module, _inputs):
        """Record the two checkpoint classifiers before each TE forward."""
        observations.append((torch._C._current_graph_task_id(), is_checkpointing()))

    handle = module.register_forward_pre_hook(record_phase)
    try:
        value = torch.randn(2, 4, device="cuda", requires_grad=True)
        te_checkpoint(module, value, use_reentrant=False).sum().backward()
    finally:
        handle.remove()

    assert observations[0] == (-1, False)
    assert any(
        graph_task_id >= 0 and not mcore_marker for graph_task_id, mcore_marker in observations[1:]
    )
