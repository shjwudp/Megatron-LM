# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-rank integration test for M-FSDP activation-recompute graphs."""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard


@pytest.fixture(scope="module", autouse=True)
def distributed_environment():
    """Initialize the two-rank NCCL test environment."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    yield


class _LinearBlock(nn.Module):
    """One linear layer used by the integration test."""

    def __init__(self):
        """Create a four-element linear layer."""
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, value):
        """Apply the linear layer.

        :param value: Input tensor.
        :type value: torch.Tensor
        :return: Linear output.
        :rtype: torch.Tensor
        """
        return self.linear(value)


class _TwoLayerRegion(nn.Module):
    """Two separately sharded modules in one checkpoint region."""

    def __init__(self):
        """Create two bias-free linear layers."""
        super().__init__()
        self.first = _LinearBlock()
        self.second = _LinearBlock()
        self.first.linear.bias = None
        self.second.linear.bias = None

    def _checkpointed_forward(self, value):
        """Apply both layers.

        :param value: Input tensor.
        :type value: torch.Tensor
        :return: Region output.
        :rtype: torch.Tensor
        """
        return self.second(self.first(value))

    def forward(self, value):
        """Checkpoint both sharded layers.

        :param value: Input tensor.
        :type value: torch.Tensor
        :return: Checkpointed region output.
        :rtype: torch.Tensor
        """
        return checkpoint(self._checkpointed_forward, value, use_reentrant=False)


class _SelectiveCheckpointRegion(nn.Module):
    """Checkpoint pre, one graphed module, and post together."""

    def __init__(self):
        """Create a non-reentrant checkpoint around one graphed module."""
        super().__init__()
        self.pre = nn.ReLU()
        self.graphed = _LinearBlock()
        self.post = nn.SiLU()

    def _checkpoint_body(self, value):
        """Apply the larger checkpoint region.

        :param value: Region input.
        :type value: torch.Tensor
        :return: Region output.
        :rtype: torch.Tensor
        """
        return self.post(self.graphed(self.pre(value)))

    def forward(self, value):
        """Run the non-reentrant checkpoint.

        :param value: Region input.
        :type value: torch.Tensor
        :return: Checkpointed output.
        :rtype: torch.Tensor
        """
        return checkpoint(self._checkpoint_body, value, use_reentrant=False)


def test_mfsdp_activation_recompute_updates_optimizer_parameters(request):
    """Capture three graphs and update parameters held by a prebuilt optimizer.

    :param request: Pytest request used for graph-pool cleanup.
    :type request: pytest.FixtureRequest
    """
    if torch.distributed.get_world_size() != 2:
        pytest.skip("This integration test requires two ranks")
    device = torch.device(f"cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}")
    model = _SelectiveCheckpointRegion().to(device)
    with torch.no_grad():
        model.graphed.linear.weight.fill_(1)
        model.graphed.linear.bias.zero_()
    fully_shard(
        model.graphed,
        sharding_strategy="optim_grads_params",
        enable_unshard_prefetch=True,
        enable_async_reduce_grad=True,
        enable_cuda_graph=True,
        cuda_graph_activation_recompute=True,
    )
    fully_shard(model, sharding_strategy="optim_grads_params")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer_parameter_ids = tuple(id(parameter) for parameter in model.parameters())

    def release_graph_pool():
        """Release captured graphs after all outputs are gone."""
        model.release_memory_pool()
        torch.cuda.synchronize()
        torch.distributed.barrier()

    request.addfinalizer(release_graph_pool)
    for step, value in enumerate((1.0, 2.0, 3.0)):
        if step == 1:
            with torch.no_grad():
                first_probe = model(torch.full((2, 4), 6.0, device=device))
                expected_probe = first_probe.clone()
                model(torch.full((2, 4), 6.5, device=device))
            torch.testing.assert_close(first_probe, expected_probe)
            assert model.graphed._fsdp_cg_pending_backwards == 0
            del first_probe, expected_probe
        if step == 2:
            runner = model._fsdp_root_context.cuda_graph_runner
            assert runner is not None
            assert model._fsdp_root_context.cuda_graph_capture_pending
            assert not runner.captured
            model.eval()
            with torch.no_grad():
                first_eval = model(torch.full((2, 4), 7.0, device=device))
                expected_eval = first_eval.clone()
                model(torch.full((2, 4), 8.0, device=device))
            torch.testing.assert_close(first_eval, expected_eval)
            assert model.graphed._fsdp_cg_pending_backwards == 0
            assert model._fsdp_root_context.cuda_graph_capture_pending
            assert not runner.captured
            del first_eval, expected_eval
            model.train()

        optimizer.zero_grad(set_to_none=True)
        sample = torch.full((2, 4), value, device=device, requires_grad=True)
        before = tuple(parameter.to_local().detach().clone() for parameter in model.parameters())
        model(sample).sum().backward()
        model.finish_grad_sync()
        optimizer.step()
        torch.cuda.synchronize()

        assert tuple(id(parameter) for parameter in model.parameters()) == optimizer_parameter_ids
        assert any(
            not torch.equal(parameter.to_local(), previous)
            for parameter, previous in zip(model.parameters(), before)
        )
        assert all(
            param_group.model_weight_buffer.storage_shard_layout != (0, 0)
            for param_group in model.graphed._fsdp_param_groups
        )
        if step == 2:
            assert model.graphed._fsdp_cg_installed
            assert model.graphed._fsdp_cg_activation_recompute

    with torch.no_grad():
        first_inference = model(torch.full((2, 4), 9.0, device=device))
        expected_inference = first_inference.clone()
        model(torch.full((2, 4), 10.0, device=device))
    torch.testing.assert_close(first_inference, expected_inference)
    assert model.graphed._fsdp_cg_pending_backwards == 0

    context = model._fsdp_root_context
    context.backward_phase = True
    context.backward_module = None
    try:
        with pytest.raises(RuntimeError, match="outside checkpoint recomputation"):
            model.graphed(torch.full((2, 4), 11.0, device=device, requires_grad=True))
    finally:
        context.backward_phase = False
    assert model.graphed._fsdp_cg_pending_backwards == 0

    abandoned = model(torch.full((2, 4), 4.0, device=device, requires_grad=True))
    assert model.release_pending()
    with pytest.raises(RuntimeError, match="released or superseded"):
        abandoned.sum().backward()
    retry = torch.full((2, 4), 5.0, device=device, requires_grad=True)
    model(retry).sum().backward()
    model.finish_grad_sync()
    assert retry.grad is not None
