# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-microbatch collective counts, optional weight retention, and SGD parity.

These small distributed GPU tests check scheduling and values, not elapsed-time
overlap. The collective observers delegate to the real distributed operations.
"""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionEvent,
    Operation,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase

_NUM_MICROBATCHES = 3
_NUM_UPDATES = 3
_RTOL = 1e-5
_ATOL = 1e-6


class NestedRoot(nn.Module):
    """A root-owned input projection whose gradient follows both FSDP children."""

    def __init__(self, width: int, *, shared: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width) + 0.05 * torch.randn(width, width))
        self.first = nn.Linear(width, width, bias=False)
        self.second = nn.Linear(width, width, bias=False)
        self.shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optionally reuse the first child with both depths contributing to loss."""
        projected = torch.tanh(nn.functional.linear(x, self.weight))
        hidden = torch.tanh(self.first(projected))
        if self.shared:
            repeated = torch.tanh(self.first(hidden + projected))
            hidden = hidden + repeated
        return self.second(hidden)


class RootCollectives:
    """Observe real root collectives and demand without changing their dependencies."""

    def __init__(self, model, context, monkeypatch, *, retain_weights: bool) -> None:
        self.model = model
        self.context = context
        self.step = 0
        self.microbatch = 0
        self.retain_weights = retain_weights
        self.events = []
        self.in_root_reduction = False
        (self.group,) = model.parameter_groups
        (self.parameter,) = self.group.fsdp_parameters
        all_gather = dist.all_gather_into_tensor
        reduce_scatter = dist.reduce_scatter_tensor
        reduce_gradients = self.group.reduce_partial_gradients

        def observed_all_gather(output_tensor, input_tensor, group=None, **kwargs):
            if input_tensor is self.group.model_weight.local_buffer:
                assert group is self.group.mesh.get_group(0)
                assert torch.cuda.current_stream() == context.allgather_stream
                self.record("ag")
            return all_gather(output_tensor, input_tensor, group=group, **kwargs)

        def observed_reduce_scatter(output, input, op=dist.ReduceOp.SUM, group=None, **kwargs):
            if self.in_root_reduction:
                assert group is self.group.mesh.get_group(0)
                assert torch.cuda.current_stream() == context.reduce_scatter_stream
                self.record("rs")
            return reduce_scatter(output, input, op=op, group=group, **kwargs)

        def observed_reduce_gradients(partial_grad, *, is_last_microbatch):
            assert not self.in_root_reduction
            self.in_root_reduction = True
            try:
                return reduce_gradients(partial_grad, is_last_microbatch=is_last_microbatch)
            finally:
                self.in_root_reduction = False

        monkeypatch.setattr(dist, "all_gather_into_tensor", observed_all_gather)
        monkeypatch.setattr(dist, "reduce_scatter_tensor", observed_reduce_scatter)
        monkeypatch.setattr(self.group, "reduce_partial_gradients", observed_reduce_gradients)
        # Prepend so this observes demand before FSDP's own forward-pre hook.
        self.demand_hook = model.register_forward_pre_hook(self.before_demand, prepend=True)
        # Append so this marks completion of FSDP's native forward-pre hook.
        self.ready_hook = model.register_forward_pre_hook(self.after_demand)

    def record(self, kind) -> None:
        """Record host submission order and the owner's current lifecycle phase."""
        self.events.append((kind, self.step, self.microbatch, self.model.phase))

    def before_demand(self, module, inputs) -> None:
        """Check that retained weights survived the preceding backward cleanup."""
        self.record("demand")
        if self.retain_weights and self.step > 0 and self.microbatch > 0:
            assert Phase.FORWARD in self.group._unshard_events
            assert self.group._unsharded_model_weight.is_allocated
            assert module.weight is self.parameter.unsharded

    def after_demand(self, module, inputs) -> None:
        """Mark readiness after the native hook has served this root's demand."""
        self.record("ready")

    def assert_window(self) -> None:
        """Retention reuses root weights without gathering at later forward demands."""
        events = [event for event in self.events if event[1] == self.step]
        assert sum(event[0] == "demand" for event in events) == _NUM_MICROBATCHES
        assert sum(event[0] == "ready" for event in events) == _NUM_MICROBATCHES
        assert sum(event[0] == "rs" for event in events) == _NUM_MICROBATCHES
        retaining = self.retain_weights and self.step > 0
        assert sum(event[0] == "ag" for event in events) == (
            1 if retaining else 2 * _NUM_MICROBATCHES
        )
        if not retaining:
            return
        for mb in range(_NUM_MICROBATCHES - 1):
            next_demand = next(
                i for i, event in enumerate(events) if event[0] == "demand" and event[2] == mb + 1
            )
            next_ready = next(
                i for i, event in enumerate(events) if event[0] == "ready" and event[2] == mb + 1
            )
            assert next_demand < next_ready, events
            assert not any(
                event[0] == "ag" for event in events[next_demand + 1 : next_ready]
            ), events


def _assert_owned_values(model, reference, *, gradients: bool) -> None:
    """Compare every owned value using FSDP's packed layout, including empty shards."""
    expected_parameters = dict(reference.named_parameters())
    seen = set()
    for owner_name, owner in (("", model), ("first", model.first), ("second", model.second)):
        for group in owner.parameter_groups:
            for index, parameter in enumerate(group.fsdp_parameters):
                (local_name,) = parameter.fqns
                name = f"{owner_name}.{local_name}" if owner_name else local_name
                assert name not in seen
                seen.add(name)
                expected = expected_parameters[name]
                if gradients:
                    assert expected.grad is not None
                    assert parameter.sharded.grad is not None
                    expected = expected.grad
                    actual = parameter.sharded.grad.to_local()
                else:
                    expected = expected.detach()
                    actual = group.main_weight.get_tensor_view(index)
                owned = group.main_weight._get_owned_range(index)
                if owned is None:
                    assert actual.numel() == 0
                    continue
                expected = expected.flatten().narrow(0, owned.tensor_relative_offset, owned.numel)
                assert actual.numel() == owned.numel
                assert torch.isfinite(actual).all(), name
                torch.testing.assert_close(
                    actual.flatten(), expected, rtol=_RTOL, atol=_ATOL, msg=name
                )
    assert seen == set(expected_parameters)


def _train_and_compare(
    distributed_setup, monkeypatch, *, shared: bool, retain_weights: bool
) -> None:
    device = distributed_setup.device
    world_size = distributed_setup.world_size
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    width = max(16, 4 * world_size)
    torch.manual_seed(1234)
    reference = NestedRoot(width, shared=shared).to(device)
    model = deepcopy(reference)
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(
        FsdpContext(
            device=device,
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
            prefetch_order=PrefetchOrder.RECORDED,
            retain_weights=retain_weights,
        )
    ) as context:
        fully_shard(model.first, mesh=mesh, placements=placements)
        fully_shard(model.second, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    observer = RootCollectives(model, context, monkeypatch, retain_weights=retain_weights)
    torch.manual_seed(5678 + distributed_setup.rank)
    inputs = torch.randn(_NUM_UPDATES, _NUM_MICROBATCHES, 2, width, device=device)
    targets = torch.randn_like(inputs)
    recorded_order = None
    try:
        for step in range(_NUM_UPDATES):
            observer.step = step
            observer.microbatch = 0
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            # An optimizer boundary must not retain next-window compute weights.
            assert Phase.FORWARD not in observer.group._unshard_events
            assert model.weight is observer.parameter.sharded
            losses = []
            reference_losses = []
            for mb in range(_NUM_MICROBATCHES):
                observer.microbatch = mb
                expected = reference(inputs[step, mb])
                reference_loss = nn.functional.mse_loss(expected, targets[step, mb])
                (reference_loss / _NUM_MICROBATCHES).backward()
                reference_losses.append(reference_loss.detach())
                with microbatch(context, is_last=mb == _NUM_MICROBATCHES - 1):
                    actual = model(inputs[step, mb])
                    loss = nn.functional.mse_loss(actual, targets[step, mb])
                    (loss / _NUM_MICROBATCHES).backward()
                losses.append(loss.detach())
                # A first backward must not truncate a caller-owned recorded window.
                assert context.is_recording_execution_order == (step == 0)

            window_order = list(context.execution_order)
            context.finish_grad_sync()
            assert not context.is_recording_execution_order
            assert context.execution_order.position == 0
            order = list(context.execution_order)
            assert order == window_order
            for owner in (model, model.first, model.second):
                for group in owner.parameter_groups:
                    assert not group._unshard_events
                    assert not group._unsharded_model_weight.is_allocated
            if recorded_order is None:
                recorded_order = order
                assert order.count(ExecutionEvent(model, Phase.FORWARD)) == _NUM_MICROBATCHES
                assert order.count(ExecutionEvent(model, Phase.BACKWARD)) == _NUM_MICROBATCHES
                for phase in (Phase.FORWARD, Phase.BACKWARD):
                    assert order.count(ExecutionEvent(model, phase, Operation.RESHARD)) == (
                        _NUM_MICROBATCHES
                    )
                assert order.count(ExecutionEvent(model.first, Phase.FORWARD)) == (
                    _NUM_MICROBATCHES * (2 if shared else 1)
                )
                # A reused layer accumulates both uses before its one completion hook.
                assert order.count(ExecutionEvent(model.first, Phase.BACKWARD)) == _NUM_MICROBATCHES
                assert order.count(
                    ExecutionEvent(model.first, Phase.FORWARD, Operation.RESHARD)
                ) == _NUM_MICROBATCHES * (2 if shared else 1)
                assert (
                    order.count(ExecutionEvent(model.first, Phase.BACKWARD, Operation.RESHARD))
                    == _NUM_MICROBATCHES
                )
            else:
                assert order == recorded_order
            observer.assert_window()
            for parameter in reference.parameters():
                assert parameter.grad is not None
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=mesh.get_group(0))
                parameter.grad.div_(world_size)
                assert torch.isfinite(parameter.grad).all()
                assert torch.count_nonzero(parameter.grad) > 0
            torch.testing.assert_close(
                torch.stack(losses), torch.stack(reference_losses), rtol=_RTOL, atol=_ATOL
            )
            _assert_owned_values(model, reference, gradients=True)
            before = {name: p.detach().clone() for name, p in reference.named_parameters()}
            reference_optimizer.step()
            optimizer.step()
            _assert_owned_values(model, reference, gradients=False)
            for name, parameter in reference.named_parameters():
                assert torch.count_nonzero(parameter.detach() - before[name]) > 0, name
    finally:
        observer.demand_hook.remove()
        observer.ready_hook.remove()
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("retain_weights", [False, True], ids=["reshard", "retain"])
def test_nested_root_microbatch_prefetch_preserves_gradients_and_updates(
    distributed_setup, monkeypatch, retain_weights
):
    """Root collective counts follow retention without changing gradients or SGD."""
    _train_and_compare(distributed_setup, monkeypatch, shared=False, retain_weights=retain_weights)


@pytest.mark.parametrize("retain_weights", [False, True], ids=["reshard", "retain"])
def test_shared_layer_microbatch_prefetch_preserves_gradients_and_updates(
    distributed_setup, monkeypatch, retain_weights
):
    """Two forward uses of one child preserve accumulated gradients and fresh weights."""
    _train_and_compare(distributed_setup, monkeypatch, shared=True, retain_weights=retain_weights)
