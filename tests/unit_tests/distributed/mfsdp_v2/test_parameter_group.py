"""Gradient-packing storage lifetime tests."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)


def test_gradient_packing_preserves_side_stream_storage(distributed_setup):
    """Keep source storage alive until its consuming stream finishes packing."""
    device = distributed_setup.device
    if device.type != "cuda":
        pytest.skip("This test requires CUDA streams.")
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    shape = (1024, 1024)
    with torch.cuda.stream(producer):
        parameter = SimpleNamespace(grad=torch.ones(shape, device=device))
    with torch.cuda.stream(consumer):
        destination = torch.empty(shape, device=device)
        consumer.wait_stream(producer)
        torch.cuda._sleep(200_000_000)
        group = SimpleNamespace(fsdp_parameters=[SimpleNamespace(unsharded=parameter)])
        partial_grad = SimpleNamespace(get_tensor_view=lambda index: destination)
        FsdpParameterGroup.copy_gradients_to_partial_buffer(group, partial_grad)
    assert parameter.grad is None
    with torch.cuda.stream(producer):
        replacement = torch.empty(shape, device=device)
        replacement.zero_()
    producer.synchronize()
    consumer.synchronize()
    torch.testing.assert_close(destination, torch.ones_like(destination), rtol=0, atol=0)
