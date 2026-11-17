# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental Megatron-FSDP runtime contexts.

Test prefetch-order construction here; test communication/computation overlap
in ``test_overlap.py``.
"""

from copy import deepcopy
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionAction,
    ExecutionEvent,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


class NestedModel(nn.Module):
    """Model with direct and child-owned parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(4))
        self.inner = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested model."""
        return self.inner(x) + self.bias


class MultiChildModel(nn.Module):
    """Model with direct parameters and multiple child FsdpModules."""

    def __init__(self, dim: int, num_children: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_children)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through every child layer with a root-owned bias."""
        x = x + self.bias
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class BranchModel(nn.Module):
    """Nested branch with its own child FsdpModule."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.inner = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested branch."""
        return torch.relu(self.inner(x) + self.bias)


class NestedSiblingModel(nn.Module):
    """Model with a nested left subtree and a right sibling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.left = BranchModel(dim)
        self.right = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested subtree before the right sibling."""
        return self.right(self.left(x) + self.bias)


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def test_prefetch_order_view_filters_actions_and_rebuilds():
    """Prefetch uses module entries and rebuilds its view when the source is replaced."""
    context = object.__new__(FsdpContext)
    context.execution_order = ExecutionOrder()
    context._prefetch_order_view = None
    current, following = nn.Identity(), nn.Identity()
    current.phase = Phase.FORWARD
    following._unshard_parameter_groups = Mock()
    order = context.execution_order
    view = context.prefetch_order_view
    other_view = order.view(actions=ExecutionAction)
    order.append(ExecutionEvent(current, Phase.FORWARD, ExecutionAction.UNSHARD))
    current_id = order.append(ExecutionEvent(current, Phase.FORWARD))
    order.append(ExecutionEvent(current, Phase.FORWARD, ExecutionAction.RESHARD))
    following_id = order.append(ExecutionEvent(following, Phase.BACKWARD))
    other_view.advance()
    other_view.advance()

    assert context.prefetch_order_view is view
    assert view.event_ids == (current_id, following_id)
    FsdpModule._prefetch_parameter_groups(current, view, None)
    following._unshard_parameter_groups.assert_called_once_with(Phase.BACKWARD)
    assert view.position == 1 and other_view.position == 2

    replacement = ExecutionOrder([ExecutionEvent(following, Phase.FORWARD)])
    context.execution_order = replacement
    assert context.prefetch_order_view is not view
    assert context.prefetch_order_view.source is replacement
    assert context.prefetch_order_view.position == 0
    assert view.source is order and view.position == 1


@pytest.mark.parametrize("caller_managed", [False, True])
def test_recorded_execution_order_seals_at_sync_boundary(caller_managed):
    """Seal only at the chosen complete-step boundary, using module-entry coordinates."""
    context = object.__new__(FsdpContext)
    context.execution_order = ExecutionOrder()
    context._prefetch_order_view = None
    context.prefetch_order = PrefetchOrder.RECORDED
    context._recording_execution_order = True
    context.caller_managed_grad_sync = caller_managed
    context._delayed_post_backward_callbacks = []
    context._post_backward_hook_registered = False
    context.is_last_microbatch = True
    context.inner_reduction_stream = Mock()
    context.outer_reduction_stream = Mock()
    compute_stream = Mock()
    context.current_stream = Mock(return_value=compute_stream)
    boundary = context.finish_grad_sync if caller_managed else context.post_backward
    module = nn.Identity()
    order = context.execution_order

    boundary()
    assert context.is_recording_execution_order and not order.is_sealed
    order.append(ExecutionEvent(module, Phase.BACKWARD, ExecutionAction.GRAD_ALLOC))
    boundary()
    assert context.is_recording_execution_order and not order.is_sealed

    order.append(ExecutionEvent(module, Phase.FORWARD))
    order.append(ExecutionEvent(module, Phase.BACKWARD))
    context.post_backward()
    if caller_managed:
        assert context.is_recording_execution_order and not order.is_sealed
        order.append(ExecutionEvent(module, Phase.BACKWARD, ExecutionAction.GRAD_REDUCE))
        order.append(ExecutionEvent(module, Phase.FORWARD))
        order.append(ExecutionEvent(module, Phase.BACKWARD))
        context.post_backward()
        assert context.is_recording_execution_order and not order.is_sealed
        context.finish_grad_sync()
    assert not context.is_recording_execution_order and order.is_sealed
    view = context.prefetch_order_view
    assert view.event_ids == ((1, 2, 4, 5) if caller_managed else (1, 2))
    compute_stream.wait_stream.assert_any_call(context.inner_reduction_stream)
    compute_stream.wait_stream.assert_any_call(context.outer_reduction_stream)
    with pytest.raises(RuntimeError, match="sealed"):
        order.append(ExecutionEvent(module, Phase.FORWARD))

    other_view = order.view(actions=ExecutionAction)
    other_view.advance()
    view.advance()
    with pytest.raises(RuntimeError, match="recorded order was complete"):
        boundary()
    assert other_view.position == 1
    for _ in range(len(view) - 1):
        view.advance()
    boundary()
    assert view.position == 0 and other_view.position == 1


def test_recorded_execution_order(distributed_setup):
    """Recorded orders follow execution when submodules are registered in reverse."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.second = nn.Linear(4, 4, bias=False)
            self.first = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.second(self.first(x))

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = Model().to(device)
    unsharded_weights = {layer: layer.weight for layer in model.children()}
    with fully_shard_context(
        FsdpContext(device=device, prefetch_order=PrefetchOrder.RECORDED)
    ) as context:
        for layer in model.children():
            fully_shard(layer, mesh, _default_placements())
        fully_shard(model, mesh, _default_placements())

    inputs = torch.ones(2, 4, device=device)
    assert context.is_recording_execution_order
    model(inputs).sum().backward()
    assert not context.is_recording_execution_order
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD) for module in [model, model.first, model.second]
    ] + [ExecutionEvent(module, Phase.BACKWARD) for module in [model, model.second, model.first]]

    def check_prefetched(layer, _args):
        storage_bytes = unsharded_weights[layer].untyped_storage().nbytes()
        assert storage_bytes > 0

    for layer in model.children():
        # Check before FSDP's own hooks, so on-demand unsharding cannot satisfy the assertion.
        layer.register_forward_pre_hook(check_prefetched, prepend=True)
        layer.register_full_backward_pre_hook(check_prefetched, prepend=True)
    model.zero_grad(set_to_none=True)
    model(inputs).sum().backward()


def test_registration_order_seeks_reordered_execution(distributed_setup):
    """The branch supports reordered pipeline calls but still rejects unknown events."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4, bias=False)
            self.second = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.first(self.second(x))

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(1234)
    model = Model().to(device)
    reference = deepcopy(model)
    with fully_shard_context(FsdpContext(device=device)) as context:
        for layer in model.children():
            fully_shard(layer, mesh, _default_placements())
        fully_shard(model, mesh, _default_placements())

    for _ in range(3):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        inputs = torch.ones(2, 4, device=device, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        actual = model(inputs)
        expected = reference(reference_inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, atol=0, rtol=0)

    # Seeking is allowed only for events already present in the execution trace.
    context.execution_order = ExecutionOrder([ExecutionEvent(model, Phase.FORWARD)])
    with pytest.raises(RuntimeError, match="diverged from the expected execution order"):
        model(torch.ones(2, 4, device=device))


@pytest.mark.parametrize("overlap_allgather_reduction", [False, True])
def test_context_defaults_to_current_cuda_device(distributed_setup, overlap_allgather_reduction):
    """The policy controls stream sharing without changing the default device."""
    context = FsdpContext(
        runtime_policy=RuntimePolicy(overlap_allgather_reduction=overlap_allgather_reduction)
    )
    assert context.allgather_stream.device == torch.device("cuda", torch.cuda.current_device())
    assert (
        context.allgather_stream is not context.inner_reduction_stream
    ) == overlap_allgather_reduction


def test_context_rejects_non_cuda_device(distributed_setup):
    """The context constructor should reject non-CUDA devices."""
    with pytest.raises(ValueError, match="CUDA device"):
        FsdpContext(device=torch.device("cpu"))


def test_child_then_parent_share_one_context(distributed_setup):
    """Modules constructed together should eagerly share one context."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    context = FsdpContext(device=device)
    with fully_shard_context(context):
        fully_shard(model.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())
        assert model.context is context
        assert model.inner.context is context

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.inner.context is model.context
    assert model.is_root()
    assert not model.inner.is_root()


def test_context_does_not_keep_modules_alive(distributed_setup):
    """Keeping a finalized context alive must not retain its roots or descendants."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel().to(device)

    with fully_shard_context(FsdpContext(device=device)) as context:
        fully_shard(model.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())

    model_ref = ref(model)
    child_ref = ref(model.inner)
    del model

    assert model_ref() is None
    assert child_ref() is None
    assert list(context.execution_order) == []


def test_two_child_subtrees_then_parent_share_one_context(distributed_setup):
    """One construction scope should assign one context across child subtrees."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.layers[0], mesh=mesh, placements=_default_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.layers[0].context is model.context
    assert model.layers[1].context is model.context


def test_sibling_roots_share_context_and_cross_root_orders(distributed_setup):
    """Independent roots should share streams and follow construction order."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.layers[0], mesh=mesh, placements=_default_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_default_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.layers[0].context
    assert model.layers[1].context is context
    assert model.layers[0].is_root()
    assert model.layers[1].is_root()
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD) for module in [model.layers[0], model.layers[1]]
    ] + [ExecutionEvent(module, Phase.BACKWARD) for module in [model.layers[1], model.layers[0]]]


@pytest.mark.parametrize("caller_managed_grad_sync", [False, True])
def test_sibling_roots_finish_grad_sync_once_after_last_microbatch(
    distributed_setup, monkeypatch, caller_managed_grad_sync
):
    """Sibling roots should share one final wait across accumulated microbatches."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(
        FsdpContext(
            device=device,
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=caller_managed_grad_sync),
        )
    ) as context:
        fully_shard(model.layers[0], mesh=mesh, placements=_default_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_default_placements())

    finish_grad_sync = Mock(wraps=context.finish_grad_sync)
    monkeypatch.setattr(context, "finish_grad_sync", finish_grad_sync)

    with microbatch(context, is_last=False):
        model(torch.ones(2, 4, device=device)).sum().backward()
    finish_grad_sync.assert_not_called()

    model(torch.ones(2, 4, device=device)).sum().backward()

    if caller_managed_grad_sync:
        finish_grad_sync.assert_not_called()
        context.finish_grad_sync()
    finish_grad_sync.assert_called_once_with()


def test_nested_prefetch_orders_use_dfs(distributed_setup):
    """Nested FsdpModules should use DFS orders for one-step prefetch."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.left.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model.left, mesh=mesh, placements=_default_placements())
        fully_shard(model.right, mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.context
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD)
        for module in [model, model.left, model.left.inner, model.right]
    ] + [
        ExecutionEvent(module, Phase.BACKWARD)
        for module in [model, model.right, model.left, model.left.inner]
    ]


def test_register_post_backward_hook_handles_parameterless_module(distributed_setup):
    """A parameterless FSDP unit should invoke the external scheduler callback."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Identity().to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(
            model,
            mesh=mesh,
            placements=_default_placements(),
            schedule_policy=SchedulePolicy(register_hooks=False),
        )

    callback_modules = []
    model.register_post_backward_hook(callback_modules.append)
    model(torch.ones(2, 4, device=device, requires_grad=True)).sum().backward()

    assert callback_modules == [model]


def test_nested_and_sibling_roots_use_cross_root_orders(distributed_setup):
    """Context orders should concatenate nested roots at construction boundaries."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.left.inner, mesh=mesh, placements=_default_placements())
        fully_shard(model.left, mesh=mesh, placements=_default_placements())
        fully_shard(model.right, mesh=mesh, placements=_default_placements())

    context = model.left.context
    assert model.left.is_root()
    assert model.right.is_root()
    assert not model.left.inner.is_root()
    assert list(context.execution_order) == [
        ExecutionEvent(module, Phase.FORWARD)
        for module in [model.left, model.left.inner, model.right]
    ] + [
        ExecutionEvent(module, Phase.BACKWARD)
        for module in [model.right, model.left, model.left.inner]
    ]


def test_fully_shard_requires_context(distributed_setup):
    """fully_shard should reject construction without an active context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)

    with pytest.raises(RuntimeError, match="inside fully_shard_context"):
        fully_shard(model, mesh=mesh, placements=_default_placements())


def test_forward_requires_finalized_context(distributed_setup):
    """Forward should be unavailable until construction scope exit."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)
    x = torch.ones(2, 4, device=device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model, mesh=mesh, placements=_default_placements())
        with pytest.raises(RuntimeError, match="Exit fully_shard_context"):
            model(x)

    model(x)


def test_fully_shard_context_rejects_nesting(distributed_setup):
    """A construction scope should reject an ambiguous nested context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model[0], mesh=mesh, placements=_default_placements())
        outer_context = model[0].context
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(FsdpContext(device=device)):
                pass
        fully_shard(model[1], mesh=mesh, placements=_default_placements())

    assert model[0].context is outer_context
    assert model[1].context is outer_context


def test_fully_shard_rejects_child_from_another_context(distributed_setup):
    """A parent cannot join a context different from an FSDP child context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    with fully_shard_context(FsdpContext(device=device)) as first_context:
        fully_shard(model.inner, mesh=mesh, placements=_default_placements())

    with fully_shard_context(FsdpContext(device=device)):
        with pytest.raises(ValueError, match="another fully_shard_context"):
            fully_shard(model, mesh=mesh, placements=_default_placements())

    assert model.inner.context is first_context


def test_microbatch_scopes_context(distributed_setup):
    """microbatch() should scope state on the supplied FSDP context."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)).to(device)
    with fully_shard_context(FsdpContext(device=device)) as context:
        for layer in model:
            fully_shard(layer, mesh=mesh, placements=_default_placements())

    with microbatch(context, is_last=False):
        assert not context.is_last_microbatch

    assert context.is_last_microbatch


@pytest.mark.parametrize("overlap_allgather_reduction", [False, True])
@pytest.mark.parametrize("overlap_outer_communication", [False, True])
def test_communication_stream_assignment(
    distributed_setup, overlap_allgather_reduction, overlap_outer_communication
):
    """All-gather/reduction and inner/outer stream separation are independent."""
    context = FsdpContext(
        runtime_policy=RuntimePolicy(
            overlap_allgather_reduction=overlap_allgather_reduction,
            overlap_outer_communication=overlap_outer_communication,
        )
    )
    assert (
        context.allgather_stream != context.inner_reduction_stream
    ) == overlap_allgather_reduction
    assert (
        context.outer_reduction_stream != context.inner_reduction_stream
    ) == overlap_outer_communication
    if overlap_outer_communication:
        assert context.outer_reduction_stream != context.allgather_stream
