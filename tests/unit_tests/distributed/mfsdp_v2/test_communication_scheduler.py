# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for trace-guided M-FSDP v2 communication scheduling."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    FsdpCommunicationSchedulerConfig,
    FsdpModuleCommunicationPolicy,
    ModuleCompletion,
    NamedCompletion,
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.models.common.utils import TransformerLayerNode
from megatron.core.pipeline_parallel.utils import ScheduleNode


class TinyModel(nn.Module):
    """Two-layer model with independently shardable FSDP units."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two-layer network."""
        return self.fc2(self.relu(self.fc1(x)))


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def test_scheduler_config_rejects_negative_pending_bytes() -> None:
    """Only auto, eager, and positive pending-byte policies are valid."""
    with pytest.raises(ValueError, match="non-negative"):
        FsdpCommunicationSchedulerConfig(max_pending_reduce_scatter_bytes=-1)


def test_scheduler_config_rejects_nonpositive_prefetch_depth() -> None:
    """Prefetch depth is a one-based traced-occurrence distance."""
    with pytest.raises(ValueError, match="prefetch_depth must be positive"):
        FsdpCommunicationSchedulerConfig(prefetch_depth=0)


def test_ddp_pending_byte_override_requires_release_rule() -> None:
    """A standalone byte limit must not be silently ignored by MCore."""
    with pytest.raises(ValueError, match="requires at least one"):
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            fsdp_max_pending_reduce_scatter_bytes=1024,
        )


def test_ddp_prefetch_depth_requires_prefetch_rule() -> None:
    """A non-default depth must not be silently ignored without an AG rule."""
    with pytest.raises(ValueError, match="requires at least one"):
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            fsdp_prefetch_depth=2,
        )


def test_static_runner_rejects_deep_prefetch(distributed_setup) -> None:
    """Depth greater than one relies on occurrence trace replay."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    with fully_shard_context(device=device):
        fully_shard(module, mesh=mesh, placements=_flat_placements())

    with pytest.raises(ValueError, match="requires trace replay"):
        module.context.runner.suggest_prefetch(module, "rowwise", depth=2)


def test_nonempty_policy_requires_context_scheduler(distributed_setup) -> None:
    """A module policy must not silently run without its context scheduler."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4).to(device)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(module, "forward"),)
    )

    with fully_shard_context(device=device):
        with pytest.raises(ValueError, match="requires fully_shard_context"):
            fully_shard(
                module, mesh=mesh, placements=_flat_placements(), communication_policy=policy
            )


def test_scheduler_enables_trace_and_requires_matching_reuse(distributed_setup) -> None:
    """VPP chunks must share one equal communication scheduler configuration."""
    device = distributed_setup.device
    config = FsdpCommunicationSchedulerConfig(max_pending_reduce_scatter_bytes=0)

    with fully_shard_context(device=device, communication_scheduler=config) as context:
        assert context.runner.use_trace_replay
        with fully_shard_context(
            device=device, reuse_existing=True, communication_scheduler=config
        ) as reused:
            assert reused is context
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(
                device=device,
                reuse_existing=True,
                communication_scheduler=FsdpCommunicationSchedulerConfig(1024),
            ):
                pass


def test_completion_anchor_releases_traced_successor(distributed_setup, monkeypatch) -> None:
    """A replayed successor AG should wait for its configured source anchor."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    config = FsdpCommunicationSchedulerConfig(max_pending_reduce_scatter_bytes=0)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(modules[0], "forward"),)
    )
    with fully_shard_context(device=device, communication_scheduler=config) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        fully_shard(modules[1], mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None
    runner.record_unshard(modules[0], "rowwise")
    runner.record_completion(modules[0], modules[0], "forward")
    runner.record_unshard(modules[1], "rowwise")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        modules[1],
        "_unshard_parameter_groups",
        lambda orientation, *, reason: calls.append((orientation, reason)),
    )
    assert runner.record_unshard(modules[0], "rowwise")
    successor = runner.suggest_prefetch(modules[0], "rowwise")
    assert successor == (modules[1], "rowwise")
    scheduler.schedule_prefetch(modules[0], "rowwise", *successor)
    assert scheduler.has_pending_prefetches
    assert not calls

    scheduler.record_completion_anchor(modules[0], modules[0], "forward")
    assert calls == [("rowwise", "anchor")]
    assert not scheduler.has_pending_prefetches


def test_mixed_orientation_prefetch_uses_future_backward_anchor(
    distributed_setup, monkeypatch
) -> None:
    """A backward source may prefetch a forward target at a later backward anchor."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(
            NamedCompletion("early", "backward"),
            NamedCompletion("forward-only", "forward"),
            NamedCompletion("late", "backward"),
        )
    )
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig(0)
    ) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        fully_shard(modules[1], mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None

    # The first backward anchor has already passed when the backward source
    # discovers a forward-oriented target. The next backward anchor is the
    # legal release point; the intervening forward anchor must not release it.
    runner.record_completion(modules[0], "early", "backward")
    runner.record_unshard(modules[0], "colwise")
    runner.record_completion(modules[0], "forward-only", "forward")
    runner.record_completion(modules[0], "late", "backward")
    runner.record_unshard(modules[1], "rowwise")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        modules[1],
        "_unshard_parameter_groups",
        lambda orientation, *, reason: calls.append((orientation, reason)),
    )

    scheduler.record_completion_anchor(modules[0], "early", "backward")
    assert runner.record_unshard(modules[0], "colwise")
    successor = runner.suggest_prefetch(modules[0], "colwise")
    assert successor == (modules[1], "rowwise")
    scheduler.schedule_prefetch(modules[0], "colwise", *successor)
    assert scheduler.has_pending_prefetches

    scheduler.record_completion_anchor(modules[0], "forward-only", "forward")
    assert not calls
    assert scheduler.has_pending_prefetches

    scheduler.record_completion_anchor(modules[0], "late", "backward")
    assert calls == [("rowwise", "anchor")]
    assert not scheduler.has_pending_prefetches


@pytest.mark.parametrize("bwd_dw_callables, expected_completions", [([], 1), ([object()], 0)])
def test_delayed_wgrad_only_defers_nodes_with_wgrad(
    monkeypatch, bwd_dw_callables, expected_completions
) -> None:
    """Communication-only nodes must still emit backward completion with delayed wgrad."""
    node = TransformerLayerNode.__new__(TransformerLayerNode)
    node.name = "moe_dispatch"
    node.event = object()
    node.delay_wgrad_compute = True
    node.bwd_dw_callables = bwd_dw_callables
    node.is_layer_first_node = False
    node._fsdp_pre_backward_communication_hook = None
    completions = []
    node._fsdp_post_backward_communication_hook = (
        lambda name, event: completions.append((name, event))
    )
    monkeypatch.setattr(ScheduleNode, "backward", lambda _self, *_grad: "grad")

    assert node.backward(object()) == "grad"
    assert len(completions) == expected_completions


def test_module_prefetches_configured_future_occurrence(distributed_setup, monkeypatch) -> None:
    """The module path should queue the configured-depth target at its anchor."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    config = FsdpCommunicationSchedulerConfig(
        max_pending_reduce_scatter_bytes=0, prefetch_depth=2
    )
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(modules[0], "forward"),)
    )
    with fully_shard_context(device=device, communication_scheduler=config) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        for module in modules[1:]:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None
    runner.record_unshard(modules[0], "rowwise")
    runner.record_completion(modules[0], modules[0], "forward")
    runner.record_unshard(modules[1], "rowwise")
    runner.record_unshard(modules[2], "rowwise")
    runner.complete_trace()

    calls = []
    for index, module in enumerate(modules):
        monkeypatch.setattr(
            module,
            "_unshard_parameter_groups",
            lambda orientation, *, reason, index=index: calls.append(
                (index, orientation, reason)
            ),
        )

    modules[0]._unshard_and_prefetch("rowwise")
    assert calls == [(0, "rowwise", "consumer")]
    assert scheduler.has_pending_prefetches

    scheduler.record_completion_anchor(modules[0], modules[0], "forward")
    assert calls == [(0, "rowwise", "consumer"), (2, "rowwise", "anchor")]
    assert not scheduler.has_pending_prefetches


def test_all_gather_nvtx_range_wraps_parameter_group_submission(
    distributed_setup, monkeypatch
) -> None:
    """Every AG launch should carry its module, group, orientation, and release path."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    with fully_shard_context(device=device):
        fully_shard(module, mesh=mesh, placements=_flat_placements())

    group = module.parameter_groups[0]
    module_name = module.name if module.name else "<root>"
    expected_label = (
        f"MFSDP AG module={module_name} group=0 orientation=colwise release=anchor"
    )
    active_ranges = []
    events = []

    def range_push(label: str) -> None:
        active_ranges.append(label)
        events.append(("push", label))

    def range_pop() -> None:
        events.append(("pop", active_ranges.pop()))

    def unshard_parameters(orientation: str) -> None:
        assert orientation == "colwise"
        assert active_ranges[-1] == expected_label

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", range_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", range_pop)
    monkeypatch.setattr(group, "unshard_parameters", unshard_parameters)

    module._unshard_parameter_groups("colwise", reason="anchor")

    assert events == [("push", expected_label), ("pop", expected_label)]


def test_demand_unshard_is_delayed_prefetch_backstop(distributed_setup, monkeypatch) -> None:
    """A target consumer must submit a queued gather when its anchor was missed."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(modules[0], "forward"),)
    )
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig(0)
    ) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        fully_shard(modules[1], mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None
    runner.record_unshard(modules[0], "rowwise")
    runner.record_completion(modules[0], modules[0], "forward")
    runner.record_unshard(modules[1], "rowwise")
    runner.complete_trace()
    runner.record_unshard(modules[0], "rowwise")
    successor = runner.suggest_prefetch(modules[0], "rowwise")
    assert successor is not None

    calls = []
    monkeypatch.setattr(
        modules[1],
        "_unshard_parameter_groups",
        lambda orientation, *, reason: calls.append((orientation, reason)),
    )
    scheduler.schedule_prefetch(modules[0], "rowwise", *successor)
    scheduler.demand_unshard(modules[1], "rowwise")
    assert calls == [("rowwise", "demand")]
    assert not scheduler.has_pending_prefetches


def test_demand_unshard_releases_only_matching_occurrence(
    distributed_setup, monkeypatch
) -> None:
    """A demand must retain future prefetches for other orientations."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    forward_policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(modules[0], "forward"),)
    )
    backward_policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(modules[1], "backward"),)
    )
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig(0)
    ) as context:
        fully_shard(
            modules[0],
            mesh=mesh,
            placements=_flat_placements(),
            communication_policy=forward_policy,
        )
        fully_shard(
            modules[1],
            mesh=mesh,
            placements=_flat_placements(),
            communication_policy=backward_policy,
        )
        fully_shard(modules[2], mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None
    runner.record_unshard(modules[0], "rowwise")
    runner.record_completion(modules[1], modules[1], "backward")
    runner.complete_trace()
    runner.record_unshard(modules[0], "rowwise")

    calls = []
    monkeypatch.setattr(
        modules[2],
        "_unshard_parameter_groups",
        lambda orientation, *, reason: calls.append((orientation, reason)),
    )
    scheduler.schedule_prefetch(modules[0], "rowwise", modules[2], "rowwise")
    scheduler.schedule_prefetch(modules[1], "colwise", modules[2], "colwise")

    scheduler.demand_unshard(modules[2], "rowwise")
    assert calls == [("rowwise", "demand")]
    assert scheduler.has_pending_prefetches

    scheduler.record_completion_anchor(modules[1], modules[1], "backward")
    assert calls == [("rowwise", "demand"), ("colwise", "anchor")]
    assert not scheduler.has_pending_prefetches


def test_reduce_scatter_waits_for_pre_backward_release(distributed_setup, monkeypatch) -> None:
    """Replay should defer a ready RS until a configured pre-backward point."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    policy = FsdpModuleCommunicationPolicy(reduce_scatter_release_on_pre_backward=(module,))
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig(1 << 30)
    ) as context:
        fully_shard(module, mesh=mesh, placements=_flat_placements(), communication_policy=policy)

    scheduler = context.communication_scheduler
    assert scheduler is not None
    group = module.parameter_groups[0]
    calls = []
    active_ranges = []
    submission_labels = []

    def range_push(label: str) -> None:
        active_ranges.append(label)

    def range_pop() -> None:
        active_ranges.pop()

    def reduce_partial_gradients(partial_grad, is_last) -> None:
        calls.append((partial_grad, is_last))
        submission_labels.append(active_ranges[-1])

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", range_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", range_pop)
    monkeypatch.setattr(
        group,
        "reduce_partial_gradients",
        reduce_partial_gradients,
    )
    monkeypatch.setattr(group, "release_partial_grad_buffer", lambda: None)

    # Trace one request and one legal release occurrence. Physical communication
    # remains eager during this first cycle.
    context.runner.record_unshard(module, "rowwise")
    scheduler.reserve_reduce_scatter(group, module_name="<root>", group_index=0)
    trace_partial_grad = object()
    scheduler.mark_reduce_scatter_ready(
        group, trace_partial_grad, context.current_stream().record_event(), True
    )
    scheduler.record_reduce_scatter_release(module, module, None)
    scheduler.finish_grad_sync()
    context.complete_trace()
    assert scheduler.effective_reduce_scatter_budgets[0] > 0
    assert calls == [(trace_partial_grad, True)]

    replay_partial_grad = object()
    context.runner.record_unshard(module, "rowwise")
    scheduler.reserve_reduce_scatter(group, module_name="<root>", group_index=0)
    scheduler.mark_reduce_scatter_ready(
        group, replay_partial_grad, context.current_stream().record_event(), False
    )
    assert scheduler.pending_reduce_scatter_bytes == group.partial_grad_nbytes()
    assert calls == [(trace_partial_grad, True)]

    scheduler.record_reduce_scatter_release(module, module, None)
    assert scheduler.pending_reduce_scatter_bytes == 0
    assert calls == [(trace_partial_grad, True), (replay_partial_grad, False)]
    assert submission_labels == [
        "MFSDP RS module=<root> group=0 release=submit-on-ready",
        "MFSDP RS module=<root> group=0 release=anchor",
    ]


def test_same_group_reuse_drains_older_domain_requests(distributed_setup, monkeypatch) -> None:
    """A recurring VPP unit must preserve domain FIFO before reusing its buffer."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    policy = FsdpModuleCommunicationPolicy(reduce_scatter_release_on_pre_backward=(modules[0],))
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig(1 << 30)
    ) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        fully_shard(modules[1], mesh=mesh, placements=_flat_placements())

    scheduler = context.communication_scheduler
    assert scheduler is not None
    groups = [module.parameter_groups[0] for module in modules]
    calls = []
    for index, group in enumerate(groups):
        monkeypatch.setattr(
            group,
            "reduce_partial_gradients",
            lambda _partial_grad, _is_last, index=index: calls.append(index),
        )
        monkeypatch.setattr(group, "release_partial_grad_buffer", lambda: None)

    context.runner.record_unshard(modules[0], "rowwise")
    for group_index, group in enumerate(groups):
        scheduler.reserve_reduce_scatter(
            group, module_name=f"module.{group_index}", group_index=0
        )
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), True
        )
    scheduler.finish_grad_sync()
    context.complete_trace()
    calls.clear()

    context.runner.record_unshard(modules[0], "rowwise")
    for group_index, group in enumerate(groups):
        scheduler.reserve_reduce_scatter(
            group, module_name=f"module.{group_index}", group_index=0
        )
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), True
        )
    assert scheduler.pending_reduce_scatter_bytes == sum(
        group.partial_grad_nbytes() for group in groups
    )

    scheduler.reserve_reduce_scatter(groups[1], module_name="module.1", group_index=0)
    assert calls == [0, 1]
    assert scheduler.pending_reduce_scatter_bytes == groups[1].partial_grad_nbytes()
    scheduler.cancel_reduce_scatter_reservation(groups[1])


def test_scheduler_training_matches_eager_baseline(distributed_setup) -> None:
    """Delayed AG and RS must preserve multi-step training numerics."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    model = TinyModel().to(device)
    model.load_state_dict(baseline.state_dict())

    scheduler_config = FsdpCommunicationSchedulerConfig(1 << 30)
    fc1_policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(model.fc1, "forward"),),
        reduce_scatter_release_on_pre_backward=(model.fc1,),
    )
    fc2_policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(ModuleCompletion(model.fc2, "forward"),)
    )
    with fully_shard_context(device=device, communication_scheduler=scheduler_config) as context:
        fully_shard(
            model.fc1, mesh=mesh, placements=_flat_placements(), communication_policy=fc1_policy
        )
        fully_shard(
            model.fc2, mesh=mesh, placements=_flat_placements(), communication_policy=fc2_policy
        )

    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(3, 8, device=device)
    target = torch.randn(3, 4, device=device)
    baseline_losses = []
    scheduled_losses = []

    for _ in range(4):
        baseline_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        baseline_loss = torch.nn.functional.mse_loss(baseline(x), target)
        scheduled_loss = torch.nn.functional.mse_loss(model(x), target)
        baseline_loss.backward()
        scheduled_loss.backward()
        baseline_optimizer.step()
        optimizer.step()
        context.complete_trace()
        baseline_losses.append(baseline_loss.detach())
        scheduled_losses.append(scheduled_loss.detach())

    torch.testing.assert_close(torch.stack(scheduled_losses), torch.stack(baseline_losses))
