"""Gradient communication precision and packing storage lifetime tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import (
    MixedPrecisionPolicy,
)


@pytest.mark.parametrize("main_grads_dtype", [None, torch.float32])
@pytest.mark.parametrize("grad_comm_dtype", [None, torch.float32, torch.bfloat16])
def test_gradient_communication_honors_precision_policy(
    distributed_setup, monkeypatch, main_grads_dtype, grad_comm_dtype
):
    """Use the requested communication dtype without changing accumulation precision."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(8, 8, bias=False, device=device, dtype=torch.bfloat16)
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    policy = MixedPrecisionPolicy(
        main_grads_dtype=main_grads_dtype, grad_comm_dtype=grad_comm_dtype
    )
    with fully_shard_context(device=device):
        fully_shard(
            model, mesh=mesh, placements=placements, mixed_precision_policy=policy
        )
    expected_main_dtype = main_grads_dtype or torch.bfloat16
    expected_comm_dtype = grad_comm_dtype or expected_main_dtype
    observed = []
    original_allocate = FsdpParameterGroup.allocate_partial_grad_buffer

    def allocate(group):
        buffer = original_allocate(group)
        observed.append(buffer.dtype)
        return buffer

    monkeypatch.setattr(FsdpParameterGroup, "allocate_partial_grad_buffer", allocate)
    model(torch.ones((2, 8), device=device, dtype=torch.bfloat16)).sum().backward()
    assert observed == [expected_comm_dtype]
    for parameter in model.parameters():
        assert parameter.grad.dtype == expected_main_dtype
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter.grad, 2), rtol=0, atol=0
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
    torch.testing.assert_close(
        destination, torch.ones_like(destination), rtol=0, atol=0
    )
