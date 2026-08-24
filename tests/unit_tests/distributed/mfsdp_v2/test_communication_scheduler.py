# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for trace-guided M-FSDP v2 communication scheduling."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp import mcore_fsdp_adapter
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


def test_scheduler_config_rejects_negative_prefetch_resident_bytes() -> None:
    """The automatic one-target budget is zero; negative limits are invalid."""
    with pytest.raises(ValueError, match="max_prefetch_resident_bytes"):
        FsdpCommunicationSchedulerConfig(max_prefetch_resident_bytes=-1)


def test_ddp_pending_byte_override_requires_release_rule() -> None:
    """A standalone byte limit must not be silently ignored by MCore."""
    with pytest.raises(ValueError, match="requires at least one"):
        DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            fsdp_max_pending_reduce_scatter_bytes=1024,
        )


def test_ddp_prefetch_depth_enables_immediate_trace_prefetch() -> None:
    """A non-default depth should not require a delayed-release rule."""
    ddp_config = DistributedDataParallelConfig(
        use_megatron_fsdp=True, megatron_fsdp_version=2, fsdp_prefetch_depth=2
    )

    scheduler_config = (
        mcore_fsdp_adapter.FullyShardedDataParallelV2._communication_scheduler_config(ddp_config)
    )
    assert scheduler_config is not None
    assert scheduler_config.prefetch_depth == 2


def test_ddp_prefetch_residency_limit_enables_scheduler() -> None:
    """The automatic residency budget should not require a delayed-release rule."""
    ddp_config = DistributedDataParallelConfig(
        use_megatron_fsdp=True, megatron_fsdp_version=2, fsdp_max_prefetch_resident_bytes=0
    )

    scheduler_config = (
        mcore_fsdp_adapter.FullyShardedDataParallelV2._communication_scheduler_config(ddp_config)
    )
    assert scheduler_config is not None
    assert scheduler_config.max_prefetch_resident_bytes == 0


def test_static_runner_rejects_deep_prefetch(distributed_setup) -> None:
    """Depth greater than one relies on occurrence trace replay."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    with fully_shard_context(device=device):
        fully_shard(module, mesh=mesh, placements=_flat_placements())

    with pytest.raises(ValueError, match="requires trace replay"):
        module.context.runner.suggest_prefetch(module, "rowwise", depth=2)


def test_depth_only_prefetch_submits_at_source_unshard(distributed_setup, monkeypatch) -> None:
    """Depth lookahead without an after rule should use the earliest replay point."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    source, middle, target = modules
    with fully_shard_context(
        device=device,
        communication_scheduler=FsdpCommunicationSchedulerConfig(
            max_pending_reduce_scatter_bytes=0, prefetch_depth=2
        ),
    ) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None

    runner.record_unshard(source, "rowwise")
    runner.record_unshard(middle, "rowwise")
    runner.record_unshard(target, "colwise")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        target,
        "_unshard_parameter_groups",
        lambda orientation, *, reason, **metadata: calls.append(
            (orientation, reason, metadata.get("source_phase"))
        ),
    )

    assert runner.record_unshard(source, "rowwise")
    suggestion = runner.suggest_prefetch_plan(source, "rowwise", depth=2)
    assert suggestion is not None
    scheduler.schedule_prefetch(
        source,
        "rowwise",
        suggestion.module,
        suggestion.orientation,
        target_reshard_index=suggestion.release_after_reshard_index,
    )

    assert calls == [("colwise", "eager-prefetch", "forward")]
    assert not scheduler.has_pending_prefetches


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
        lambda orientation, **metadata: calls.append((orientation, metadata)),
    )
    assert runner.record_unshard(modules[0], "rowwise")
    successor = runner.suggest_prefetch(modules[0], "rowwise")
    assert successor == (modules[1], "rowwise")
    scheduler.schedule_prefetch(modules[0], "rowwise", *successor)
    assert scheduler.has_pending_prefetches
    assert not calls

    scheduler.record_completion_anchor(modules[0], modules[0], "forward")
    assert calls == [
        (
            "rowwise",
            {
                "reason": "anchor",
                "source": "<root>",
                "source_phase": "forward",
                "anchor": "<self>",
                "request": 0,
            },
        )
    ]
    assert not scheduler.has_pending_prefetches


def test_mixed_orientation_prefetch_reuses_prior_backward_anchor(
    distributed_setup, monkeypatch
) -> None:
    """An already-completed source anchor should release its exact occurrence."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(
            NamedCompletion("early", "backward"),
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
    # discovers a forward-oriented target. Its exact replay event remains a
    # satisfied "after" condition for this source occurrence.
    runner.record_completion(modules[0], "early", "backward")
    runner.record_unshard(modules[0], "colwise")
    runner.record_completion(modules[0], "late", "backward")
    runner.record_unshard(modules[1], "rowwise")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        modules[1],
        "_unshard_parameter_groups",
        lambda orientation, *, reason, **metadata: calls.append(
            (orientation, reason, metadata.get("anchor"))
        ),
    )

    scheduler.record_completion_anchor(modules[0], "early", "backward")
    assert runner.record_unshard(modules[0], "colwise")
    successor = runner.suggest_prefetch(modules[0], "colwise")
    assert successor == (modules[1], "rowwise")
    scheduler.schedule_prefetch(modules[0], "colwise", *successor)
    assert calls == [("rowwise", "latched-anchor", "@early")]
    assert not scheduler.has_pending_prefetches

    scheduler.record_completion_anchor(modules[0], "late", "backward")
    assert calls == [("rowwise", "latched-anchor", "@early")]
    assert not scheduler.has_pending_prefetches


def test_latched_anchor_does_not_cross_vpp_occurrences(distributed_setup, monkeypatch) -> None:
    """An unused anchor from one source occurrence must not release the next."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(
            NamedCompletion("early", "backward"),
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

    # Both anchors precede occurrence 0. Occurrence 1 starts before either of
    # its own anchors, leaving occurrence 0's unused "late" event deliberately
    # cached when occurrence 1 schedules its request.
    runner.record_completion(modules[0], "early", "backward")
    runner.record_completion(modules[0], "late", "backward")
    runner.record_unshard(modules[0], "colwise")
    runner.record_reshard(modules[0])
    runner.record_unshard(modules[0], "colwise")
    runner.record_completion(modules[0], "early", "backward")
    runner.record_completion(modules[0], "late", "backward")
    runner.record_unshard(modules[1], "rowwise")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        modules[1],
        "_unshard_parameter_groups",
        lambda orientation, *, reason, **metadata: calls.append(
            (orientation, reason, metadata.get("anchor"))
        ),
    )

    scheduler.record_completion_anchor(modules[0], "early", "backward")
    scheduler.record_completion_anchor(modules[0], "late", "backward")
    assert runner.record_unshard(modules[0], "colwise")
    scheduler.schedule_prefetch(modules[0], "colwise", modules[1], "rowwise")
    assert calls == [("rowwise", "latched-anchor", "@early")]

    runner.record_reshard(modules[0])
    assert runner.record_unshard(modules[0], "colwise")
    scheduler.schedule_prefetch(modules[0], "colwise", modules[1], "rowwise")
    assert scheduler.has_pending_prefetches
    assert calls == [("rowwise", "latched-anchor", "@early")]

    scheduler.record_completion_anchor(modules[0], "early", "backward")
    assert calls == [("rowwise", "latched-anchor", "@early"), ("rowwise", "anchor", "@early")]
    assert not scheduler.has_pending_prefetches


def test_latched_anchor_expires_at_global_batch_boundary(distributed_setup) -> None:
    """A cached CUDA event must not survive reuse of trace indices next batch."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(NamedCompletion("early", "backward"),)
    )
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig(0)
    ) as context:
        fully_shard(module, mesh=mesh, placements=_flat_placements(), communication_policy=policy)

    scheduler = context.communication_scheduler
    assert scheduler is not None
    context.runner.record_completion(module, "early", "backward")
    context.runner.record_unshard(module, "colwise")
    context.complete_trace()

    scheduler.record_completion_anchor(module, "early", "backward")
    assert scheduler._completed_prefetch_anchors
    context.runner.record_unshard(module, "colwise")
    context.complete_trace()
    assert not scheduler._completed_prefetch_anchors


def test_latched_anchor_expires_on_replay_divergence(distributed_setup) -> None:
    """A re-trace must discard CUDA events identified by the abandoned trace."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    policy = FsdpModuleCommunicationPolicy(
        prefetch_successor_after=(NamedCompletion("early", "backward"),)
    )
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig(0)
    ) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        fully_shard(modules[1], mesh=mesh, placements=_flat_placements())

    scheduler = context.communication_scheduler
    assert scheduler is not None
    context.runner.record_completion(modules[0], "early", "backward")
    context.runner.record_unshard(modules[0], "colwise")
    context.complete_trace()

    scheduler.record_completion_anchor(modules[0], "early", "backward")
    assert scheduler._completed_prefetch_anchors
    context.runner.record_unshard(modules[1], "rowwise")
    assert context.runner.is_tracing
    assert not scheduler._completed_prefetch_anchors


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
    node._fsdp_post_backward_communication_hook = lambda name, event: completions.append(
        (name, event)
    )
    monkeypatch.setattr(ScheduleNode, "backward", lambda _self, *_grad: "grad")

    assert node.backward(object()) == "grad"
    assert len(completions) == expected_completions


def test_module_prefetches_configured_future_occurrence(distributed_setup, monkeypatch) -> None:
    """The module path should queue the configured-depth target at its anchor."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    config = FsdpCommunicationSchedulerConfig(max_pending_reduce_scatter_bytes=0, prefetch_depth=2)
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
            lambda orientation, *, reason, index=index, **_metadata: calls.append(
                (index, orientation, reason)
            ),
        )

    modules[0]._unshard_and_prefetch("rowwise")
    assert calls == [(0, "rowwise", "consumer")]
    assert scheduler.has_pending_prefetches

    scheduler.record_completion_anchor(modules[0], modules[0], "forward")
    assert calls == [(0, "rowwise", "consumer"), (2, "rowwise", "anchor")]
    assert not scheduler.has_pending_prefetches


def test_deep_prefetch_retains_intervening_target_occurrence(
    distributed_setup, monkeypatch
) -> None:
    """A depth target should survive an earlier occurrence's physical reshard."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    source, target, middle = modules
    with fully_shard_context(
        device=device,
        communication_scheduler=FsdpCommunicationSchedulerConfig(
            max_pending_reduce_scatter_bytes=0, prefetch_depth=3
        ),
    ) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None

    runner.record_unshard(source, "rowwise")
    runner.record_unshard(target, "colwise")
    runner.record_reshard(target)
    runner.record_unshard(middle, "rowwise")
    runner.record_unshard(target, "rowwise")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        target,
        "_unshard_parameter_groups",
        lambda orientation, *, reason, **_metadata: calls.append((orientation, reason)),
    )

    assert runner.record_unshard(source, "rowwise")
    suggestion = runner.suggest_prefetch_plan(source, "rowwise", depth=3)
    assert suggestion is not None
    assert suggestion.module is target
    assert suggestion.orientation == "rowwise"
    assert suggestion.release_after_reshard_index is not None
    scheduler.schedule_prefetch(
        source,
        "rowwise",
        target,
        "rowwise",
        target_reshard_index=suggestion.release_after_reshard_index,
    )
    assert scheduler.has_pending_prefetches
    assert not calls

    assert runner.record_unshard(target, "colwise")
    reshard_index = runner.record_reshard(target)
    assert scheduler.retain_prefetches_across_reshard(target, reshard_index)
    assert scheduler.has_pending_prefetches
    assert not calls

    scheduler.demand_unshard(target, "rowwise")
    assert not scheduler.has_pending_prefetches
    assert not calls


def test_prefetch_residency_budget_reserves_earliest_demand(distributed_setup, monkeypatch) -> None:
    """One auto-sized slot should retain the earliest demand, then re-gather the later one."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)]).to(device)
    first_source, second_source, later_target, earlier_target = modules
    with fully_shard_context(
        device=device,
        communication_scheduler=FsdpCommunicationSchedulerConfig(
            max_pending_reduce_scatter_bytes=0, prefetch_depth=3, max_prefetch_resident_bytes=0
        ),
    ) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    runner = context.runner
    scheduler = context.communication_scheduler
    assert scheduler is not None

    # Compile the automatic budget to exactly one largest traced materialization.
    runner.record_unshard(later_target, "rowwise")
    runner.record_unshard(earlier_target, "rowwise")
    context.complete_trace()
    target_bytes = later_target.unsharded_parameter_nbytes()
    assert target_bytes > 0
    assert scheduler.effective_prefetch_resident_bytes == target_bytes

    calls = []
    monkeypatch.setattr(
        later_target,
        "_unshard_parameter_groups",
        lambda orientation, *, reason, **_metadata: calls.append((orientation, reason)),
    )

    scheduler.schedule_prefetch(
        first_source,
        "rowwise",
        later_target,
        "rowwise",
        target_reshard_index=10,
        target_unshard_index=30,
    )
    scheduler.schedule_prefetch(
        second_source,
        "rowwise",
        earlier_target,
        "rowwise",
        target_reshard_index=11,
        target_unshard_index=20,
    )

    # Although the later target reaches its reshard first, full-trace EDF keeps
    # the only slot available for the earlier demand.
    assert not scheduler.retain_prefetches_across_reshard(later_target, 10)
    scheduler.record_target_reshard(later_target, 10)
    assert not calls
    assert scheduler.retain_prefetches_across_reshard(earlier_target, 11)

    # Consuming the earlier target releases the slot and immediately submits
    # the already-ready later target while it still has trace lead time.
    scheduler.demand_unshard(earlier_target, "rowwise")
    assert calls == [("rowwise", "residency-resident-reuse")]
    scheduler.demand_unshard(later_target, "rowwise")
    assert not scheduler.has_pending_prefetches


def test_module_reshard_honors_retained_prefetch(distributed_setup, monkeypatch) -> None:
    """A retained target reservation must bypass physical reshard and release."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    with fully_shard_context(
        device=device,
        communication_scheduler=FsdpCommunicationSchedulerConfig(
            max_pending_reduce_scatter_bytes=0, prefetch_depth=2
        ),
    ) as context:
        fully_shard(module, mesh=mesh, placements=_flat_placements())

    scheduler = context.communication_scheduler
    assert scheduler is not None
    calls = []
    monkeypatch.setattr(context.runner, "record_reshard", lambda target: 17)
    monkeypatch.setattr(context.runner, "suggest_skip_reshard", lambda target: False)
    monkeypatch.setattr(
        scheduler,
        "retain_prefetches_across_reshard",
        lambda target, trace_index: calls.append((target, trace_index)) or True,
    )
    for group in module._parameter_groups:
        monkeypatch.setattr(
            group,
            "reshard_parameters",
            lambda: pytest.fail("retained materialization was physically resharded"),
        )

    module._reshard_parameter_groups()
    assert calls == [(module, 17)]


def test_deep_prefetch_ignores_logically_skipped_target_reshard(distributed_setup) -> None:
    """A skipped reshard must not create an unsatisfiable lifetime gate."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)
    with fully_shard_context(
        device=device,
        communication_scheduler=FsdpCommunicationSchedulerConfig(
            max_pending_reduce_scatter_bytes=0, prefetch_depth=2
        ),
    ) as context:
        for module in modules:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    source, target = modules
    runner = context.runner

    runner.record_unshard(source, "rowwise")
    runner.record_unshard(target, "rowwise")
    runner.record_reshard(target)
    runner.record_unshard(target, "rowwise")
    runner.complete_trace()

    assert runner.record_unshard(source, "rowwise")
    suggestion = runner.suggest_prefetch_plan(source, "rowwise", depth=2)
    assert suggestion is not None
    assert suggestion.module is target
    assert suggestion.orientation == "rowwise"
    assert suggestion.release_after_reshard_index is None

    assert runner.record_unshard(target, "rowwise")
    runner.record_reshard(target)
    assert runner.suggest_skip_reshard(target)


def test_all_gather_nvtx_range_wraps_parameter_group_submission(
    distributed_setup, monkeypatch
) -> None:
    """Every AG launch should identify its target, trigger, and scheduler provenance."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    with fully_shard_context(device=device):
        fully_shard(module, mesh=mesh, placements=_flat_placements())

    group = module.parameter_groups[0]
    module_name = module.name if module.name else "<root>"
    expected_label = (
        f"MFSDP AG target={module_name} group=0 orientation=colwise trigger=anchor "
        "source=layers.2 source_phase=backward anchor=@moe_combine request=7"
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

    module._unshard_parameter_groups(
        "colwise",
        reason="anchor",
        source="layers.2",
        source_phase="backward",
        anchor="@moe_combine",
        request=7,
    )

    assert events == [("push", expected_label), ("pop", expected_label)]


def test_all_gather_nvtx_marks_scheduler_noop(distributed_setup, monkeypatch) -> None:
    """A consumed request that launches no AG must remain visible in the trace."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    module = nn.Linear(4, 4, bias=False).to(device)
    with fully_shard_context(device=device):
        fully_shard(module, mesh=mesh, placements=_flat_placements())

    module_name = module.name if module.name else "<root>"
    expected_label = (
        f"MFSDP AG skipped target={module_name} orientation=rowwise trigger=anchor "
        "source=layers.2 source_phase=forward anchor=@moe_combine request=11 "
        "state=already-unsharded"
    )
    labels = []
    module._unshard_event = object()
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", labels.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    module._unshard_parameter_groups(
        "rowwise",
        reason="anchor",
        source="layers.2",
        source_phase="forward",
        anchor="@moe_combine",
        request=11,
    )

    assert labels == [expected_label]


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
        lambda orientation, *, reason, **_metadata: calls.append((orientation, reason)),
    )
    scheduler.schedule_prefetch(modules[0], "rowwise", *successor)
    scheduler.demand_unshard(modules[1], "rowwise")
    assert calls == [("rowwise", "demand")]
    assert not scheduler.has_pending_prefetches


def test_demand_unshard_releases_only_matching_occurrence(distributed_setup, monkeypatch) -> None:
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
    runner.record_unshard(modules[1], "colwise")
    runner.record_completion(modules[1], modules[1], "backward")
    runner.complete_trace()

    calls = []
    monkeypatch.setattr(
        modules[2],
        "_unshard_parameter_groups",
        lambda orientation, *, reason, **_metadata: calls.append((orientation, reason)),
    )
    runner.record_unshard(modules[0], "rowwise")
    scheduler.schedule_prefetch(modules[0], "rowwise", modules[2], "rowwise")
    runner.record_unshard(modules[1], "colwise")
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
    monkeypatch.setattr(group, "reduce_partial_gradients", reduce_partial_gradients)
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
    assert [label.split(" pending_before=", 1)[0] for label in submission_labels] == [
        f"MFSDP RS target=<root> group=0 trigger=submit-on-ready request=0 "
        f"bytes={group.partial_grad_nbytes()}",
        f"MFSDP RS target=<root> group=0 trigger=anchor request=1 "
        f"bytes={group.partial_grad_nbytes()}",
    ]
    assert all(" in_flight_after=" in label for label in submission_labels)


def test_reduce_scatter_anchor_replays_trace_inferred_byte_credit(
    distributed_setup, monkeypatch
) -> None:
    """One replay anchor should drain all requests seen in its trace interval."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    policy = FsdpModuleCommunicationPolicy(reduce_scatter_release_on_pre_backward=(modules[0],))
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig()
    ) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        for module in modules[1:]:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

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
    for index, group in enumerate(groups):
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{index}", group_index=0)
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), True
        )
    scheduler.record_reduce_scatter_release(modules[0], modules[0], None)
    scheduler.finish_grad_sync()
    context.complete_trace()

    interval_bytes = sum(group.partial_grad_nbytes() for group in groups)
    assert scheduler.effective_reduce_scatter_budgets == (interval_bytes,)
    assert calls == [0, 1, 2]
    calls.clear()

    context.runner.record_unshard(modules[0], "rowwise")
    for index, group in enumerate(groups):
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{index}", group_index=0)
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), False
        )
    assert scheduler.pending_reduce_scatter_bytes == interval_bytes
    assert calls == []

    scheduler.record_reduce_scatter_release(modules[0], modules[0], None)
    assert scheduler.pending_reduce_scatter_bytes == 0
    assert scheduler.peak_pending_reduce_scatter_bytes == interval_bytes
    assert scheduler.peak_active_reduce_scatter_bytes >= interval_bytes
    assert calls == [0, 1, 2]
    scheduler.finish_grad_sync()


def test_reduce_scatter_anchor_does_not_exceed_its_trace_credit(
    distributed_setup, monkeypatch
) -> None:
    """An anchor must not consume ready work traced for a later occurrence."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)]).to(device)
    policy = FsdpModuleCommunicationPolicy(reduce_scatter_release_on_pre_backward=(modules[0],))
    with fully_shard_context(
        device=device, communication_scheduler=FsdpCommunicationSchedulerConfig()
    ) as context:
        fully_shard(
            modules[0], mesh=mesh, placements=_flat_placements(), communication_policy=policy
        )
        for module in modules[1:]:
            fully_shard(module, mesh=mesh, placements=_flat_placements())

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

    # Trace one request at the first occurrence and two at the second. The
    # second interval sets a two-request budget, while the first occurrence's
    # replay credit remains exactly one request.
    context.runner.record_unshard(modules[0], "rowwise")
    scheduler.reserve_reduce_scatter(groups[0], module_name="module.0", group_index=0)
    scheduler.mark_reduce_scatter_ready(
        groups[0], object(), context.current_stream().record_event(), True
    )
    scheduler.record_reduce_scatter_release(modules[0], modules[0], None)
    for index, group in enumerate(groups[1:], start=1):
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{index}", group_index=0)
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), True
        )
    scheduler.record_reduce_scatter_release(modules[0], modules[0], None)
    scheduler.finish_grad_sync()
    context.complete_trace()
    calls.clear()

    first_size = groups[0].partial_grad_nbytes()
    second_size = groups[1].partial_grad_nbytes()
    assert scheduler.effective_reduce_scatter_budgets == (
        groups[1].partial_grad_nbytes() + groups[2].partial_grad_nbytes(),
    )

    context.runner.record_unshard(modules[0], "rowwise")
    for index, group in enumerate(groups[:2]):
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{index}", group_index=0)
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), False
        )
    scheduler.record_reduce_scatter_release(modules[0], modules[0], None)
    assert calls == [0]
    assert scheduler.pending_reduce_scatter_bytes == second_size

    scheduler.record_reduce_scatter_release(modules[0], modules[0], None)
    assert calls == [0, 1]
    assert scheduler.pending_reduce_scatter_bytes == 0
    assert first_size == second_size
    scheduler.finish_grad_sync()


def test_actual_prefetch_releases_one_ready_reduce_scatter(
    distributed_setup, monkeypatch
) -> None:
    """One submitted AG should add one ordered RS opportunity after its event."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    modules = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)]).to(device)
    source, target, grad_module_0, grad_module_1 = modules
    policy = FsdpModuleCommunicationPolicy(reduce_scatter_release_on_pre_backward=(source,))
    with fully_shard_context(
        device=device,
        communication_scheduler=FsdpCommunicationSchedulerConfig(
            max_pending_reduce_scatter_bytes=1 << 30,
            max_prefetch_resident_bytes=1 << 30,
            reduce_scatter_release_on_prefetch=True,
        ),
    ) as context:
        fully_shard(source, mesh=mesh, placements=_flat_placements(), communication_policy=policy)
        for module in (target, grad_module_0, grad_module_1):
            fully_shard(module, mesh=mesh, placements=_flat_placements())

    scheduler = context.communication_scheduler
    assert scheduler is not None
    groups = [grad_module_0.parameter_groups[0], grad_module_1.parameter_groups[0]]
    calls = []
    for index, group in enumerate(groups):
        monkeypatch.setattr(
            group,
            "reduce_partial_gradients",
            lambda _partial_grad, _is_last, index=index: calls.append(index),
        )
        monkeypatch.setattr(group, "release_partial_grad_buffer", lambda: None)

    context.runner.record_unshard(source, "rowwise")
    for index, group in enumerate(groups):
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{index}", group_index=0)
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), True
        )
    scheduler.record_reduce_scatter_release(source, source, None)
    scheduler.finish_grad_sync()
    context.complete_trace()
    calls.clear()

    submissions = []
    submit_reduce_scatter = scheduler._submit_reduce_scatter

    def record_submission(request, *, reason, demand_event=None, prefetch=None):
        submissions.append((reason, demand_event, prefetch))
        return submit_reduce_scatter(
            request,
            reason=reason,
            demand_event=demand_event,
            prefetch=prefetch,
        )

    monkeypatch.setattr(scheduler, "_submit_reduce_scatter", record_submission)

    context.runner.record_unshard(source, "rowwise")
    for index, group in enumerate(groups):
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{index}", group_index=0)
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), False
        )

    scheduler.schedule_prefetch(source, "rowwise", target, "rowwise")
    assert calls == [0]
    assert scheduler.pending_reduce_scatter_bytes == groups[1].partial_grad_nbytes()
    assert submissions[0][0] == "prefetch"
    assert submissions[0][1] is target._unshard_event
    assert submissions[0][2] is not None

    scheduler.record_reduce_scatter_release(source, source, None)
    assert calls == [0, 1]
    assert scheduler.pending_reduce_scatter_bytes == 0
    scheduler.finish_grad_sync()


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
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{group_index}", group_index=0)
        scheduler.mark_reduce_scatter_ready(
            group, object(), context.current_stream().record_event(), True
        )
    scheduler.finish_grad_sync()
    context.complete_trace()
    calls.clear()

    context.runner.record_unshard(modules[0], "rowwise")
    for group_index, group in enumerate(groups):
        scheduler.reserve_reduce_scatter(group, module_name=f"module.{group_index}", group_index=0)
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
