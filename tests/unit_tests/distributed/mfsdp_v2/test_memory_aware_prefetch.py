# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU correctness for calibrated occurrence plans, lifecycle safety, and full graphs.

Memory samples are controlled; all model math and distributed operations are real.
These tests establish values and scheduling invariants, not elapsed-time overlap.
"""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import Operation
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.memory_prefetch import (
    MemorySnapshot,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

_GIB = 1 << 30
_HEADROOM = 8 << 20
_MICROBATCHES = 3


class _RepeatedRoot(nn.Module):
    """Root-owned weights and two children, with two forward uses of the first."""

    def __init__(self, width, device):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width, device=device) / 2)
        self.first = nn.Linear(width, width, bias=False, device=device)
        self.second = nn.Linear(width, width, bias=False, device=device)

    def forward(self, x):
        projected = torch.tanh(nn.functional.linear(x, self.weight))
        hidden = torch.tanh(self.first(projected))
        return self.second(hidden + torch.tanh(self.first(hidden + projected)))


def _owners(model):
    return (("", model), ("first", model.first), ("second", model.second))


def _shard(model, mesh, *, enabled, retain=False, limit=None, replicated_second=False):
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    context = FsdpContext(
        device=torch.device(mesh.device_type, torch.cuda.current_device()),
        prefetch_order=PrefetchOrder.RECORDED,
        prefetch_process_group=mesh.get_group(0),
        retain_weights=retain,
        runtime_policy=RuntimePolicy(caller_managed_grad_sync=True, memory_aware_prefetch=enabled),
    )
    policy = SchedulePolicy(forward_prefetch_size=limit, backward_prefetch_size=limit)
    precision = MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.float32,
    )
    with fully_shard_context(context):
        for _, owner in reversed(_owners(model)):
            owner_placements = placements
            if replicated_second and owner is model.second:
                owner_placements = Placements(
                    dp_axes=[0], parameter=[Replicate()], gradient=[Shard(0)], optimizer=[Shard(0)]
                )
            fully_shard(
                owner,
                mesh=mesh,
                placements=owner_placements,
                mixed_precision_policy=precision,
                schedule_policy=policy,
            )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)
    fully_shard_optimizer(optimizer)
    return context, optimizer


def _snapshot(model, *, gradients):
    result = {}
    for prefix, owner in _owners(model):
        for group in owner.parameter_groups:
            for index, parameter in enumerate(group.fsdp_parameters):
                (local_name,) = parameter.fqns
                name = f"{prefix}.{local_name}" if prefix else local_name
                assert name not in result
                value = (
                    parameter.sharded.grad.to_local()
                    if gradients
                    else group.main_weight.get_tensor_view(index)
                )
                result[name] = value.detach().clone()
    return result


def _train(model, context, optimizer, inputs, *, autocast=None):
    optimizer.zero_grad(set_to_none=False)
    losses = []
    for mb, x in enumerate(inputs):
        with microbatch(context, is_last=mb == len(inputs) - 1):
            if autocast is None:
                output = model(x)
            else:
                with autocast():
                    output = model(x)
            loss = output.float().square().mean() / len(inputs)
            loss.backward()
        losses.append(loss.detach())
    context.finish_grad_sync()
    gradients = _snapshot(model, gradients=True)
    optimizer.step()
    return torch.stack(losses), gradients, _snapshot(model, gradients=False)


def _assert_result(actual, expected, *, rtol=1e-5, atol=1e-6):
    torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
    for actual_values, expected_values in zip(actual[1:], expected[1:]):
        assert actual_values.keys() == expected_values.keys()
        for name in actual_values:
            assert torch.isfinite(actual_values[name]).all(), name
            torch.testing.assert_close(
                actual_values[name], expected_values[name], rtol=rtol, atol=atol, msg=name
            )


def _assert_reference(model, reference, values, *, gradients):
    expected_parameters = dict(reference.named_parameters())
    assert values.keys() == expected_parameters.keys()
    for prefix, owner in _owners(model):
        for group in owner.parameter_groups:
            for index, parameter in enumerate(group.fsdp_parameters):
                (local_name,) = parameter.fqns
                name = f"{prefix}.{local_name}" if prefix else local_name
                expected = expected_parameters[name]
                expected = expected.grad if gradients else expected.detach()
                owned = group.main_weight._get_owned_range(index)
                if owned is None:
                    assert values[name].numel() == 0
                    continue
                expected = expected.flatten().narrow(0, owned.tensor_relative_offset, owned.numel)
                torch.testing.assert_close(
                    values[name].flatten(), expected, rtol=1e-5, atol=1e-6, msg=name
                )


def _sampler(context, model, setup, calls, *, asymmetric=False):
    def sample():
        events = list(context.execution_order)
        position = (
            len(events) - 1
            if context.is_recording_execution_order
            else context.execution_order.position
        )
        event = events[position]
        root_forward = sum(
            e.module is model and e.phase is Phase.FORWARD and e.operation is Operation.UNSHARD
            for e in events[: position + 1]
        )
        # Pass 1 intentionally differs: the compiled budget must use calibration.
        headroom = 2 * _HEADROOM if context._memory_profile_pass == 0 else _HEADROOM
        if asymmetric:
            if setup.rank != setup.world_size - 1:
                headroom = 4 * _HEADROOM
            elif (
                context._memory_profile_pass == 1
                and event.module is model
                and event.phase is Phase.FORWARD
                and event.operation is Operation.UNSHARD
                and root_forward == 2
            ):
                headroom = 0
        calls.append((context._memory_profile_pass, position, headroom))
        return MemorySnapshot(_GIB // 4, _GIB // 2, _GIB + headroom, 8 * _GIB)

    return sample


def _forbid_sampling():
    raise AssertionError("A frozen or disabled plan must not sample memory.")


def _mesh(setup):
    if setup.world_size < 2:
        pytest.skip("Memory-consensus coverage requires at least two ranks.")
    return init_device_mesh(setup.device.type, (setup.world_size,))


@pytest.mark.parametrize("retain_weights", [False, True], ids=["reshard", "retain"])
def test_occurrence_budgets_use_rank_consensus_and_preserve_sgd(
    distributed_setup, monkeypatch, retain_weights
):
    """The second pass and the lowest-memory nonzero rank control repeated occurrences."""
    setup = distributed_setup
    mesh = _mesh(setup)
    torch.manual_seed(231)
    reference = _RepeatedRoot(max(16, 4 * setup.world_size), setup.device)
    model, disabled = deepcopy(reference), deepcopy(reference)
    context, optimizer = _shard(model, mesh, enabled=True, retain=retain_weights)
    disabled_context, disabled_optimizer = _shard(
        disabled, mesh, enabled=False, retain=retain_weights
    )
    monkeypatch.setattr(disabled_context, "_sample_memory", _forbid_sampling)
    calls = []
    monkeypatch.setattr(
        context, "_sample_memory", _sampler(context, model, setup, calls, asymmetric=True)
    )
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, foreach=False)
    torch.manual_seed(834 + setup.rank)
    inputs = torch.randn(3, _MICROBATCHES, 4, reference.weight.shape[0], device=setup.device)
    frozen = None
    for step in range(3):
        reference_optimizer.zero_grad(set_to_none=False)
        losses = []
        for x in inputs[step]:
            loss = reference(x).square().mean() / _MICROBATCHES
            loss.backward()
            losses.append(loss.detach())
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad, group=mesh.get_group(0))
            parameter.grad.div_(setup.world_size)
        actual = _train(model, context, optimizer, inputs[step])
        # Retirement consensus is required even with memory planning disabled.
        # Only its first recording may add control collectives; replay must not.
        control_collectives = []
        all_reduce = dist.all_reduce
        with monkeypatch.context() as guard:

            def observe_all_reduce(*args, **kwargs):
                control_collectives.append((kwargs.get("op"), kwargs.get("group")))
                return all_reduce(*args, **kwargs)

            guard.setattr(dist, "all_reduce", observe_all_reduce)
            baseline = _train(disabled, disabled_context, disabled_optimizer, inputs[step])
        expected_collectives = (
            [(dist.ReduceOp.MIN, mesh.get_group(0)), (dist.ReduceOp.MAX, mesh.get_group(0))]
            if step == 0
            else []
        )
        assert control_collectives == expected_collectives
        assert disabled_context._memory_profile_pass == 0
        assert disabled_context._memory_prefetch_plan is None
        torch.testing.assert_close(actual[0], torch.stack(losses), rtol=1e-5, atol=1e-6)
        _assert_reference(model, reference, actual[1], gradients=True)
        reference_optimizer.step()
        _assert_reference(model, reference, actual[2], gradients=False)
        _assert_result(actual, baseline)
        assert context._memory_profile_pass == min(step + 1, 2)
        # Shared VPP callers may finish the same context again without another pass.
        context.finish_grad_sync()
        assert context._memory_profile_pass == min(step + 1, 2)
        if step == 0:
            assert context._memory_prefetch_plan is None
        if step == 1:
            events = list(context.execution_order)
            positions = [
                i
                for i, e in enumerate(events)
                if e.module is model
                and e.phase is Phase.FORWARD
                and e.operation is Operation.UNSHARD
            ]
            assert len(positions) == _MICROBATCHES
            budgets = context._memory_prefetch_budgets
            assert budgets == [0 if i == positions[1] else _HEADROOM for i in range(len(events))]
            plan = context._memory_prefetch_plan
            assert isinstance(plan, tuple) and len(plan) == len(events)
            assert plan[positions[0]] and not plan[positions[1]]
            assert len(context._event_memory) == len(events)
            assert len(calls) == 2 * len(events)
            # Compare the complete occurrence table, not just its size, across ranks.
            encoded = torch.tensor(
                budgets + [int(j in targets) for targets in plan for j in range(len(events))],
                dtype=torch.int64,
                device=setup.device,
            )
            peers = [torch.empty_like(encoded) for _ in range(setup.world_size)]
            dist.all_gather(peers, encoded, group=mesh.get_group(0))
            assert all(torch.equal(encoded, peer) for peer in peers)
            frozen = (plan, tuple(context._event_memory), len(calls))
            monkeypatch.setattr(context, "_sample_memory", _forbid_sampling)
    assert frozen == (context._memory_prefetch_plan, tuple(context._event_memory), len(calls))
    assert all(torch.count_nonzero(value) for value in actual[1].values() if value.numel())


@pytest.mark.parametrize("limit", [None, 0, 512], ids=["expand", "disabled", "positive-cap"])
def test_prefetch_targets_survive_until_demand_without_crossing_reshard(
    distributed_setup, monkeypatch, limit
):
    """A reused owner is never fetched across its future reshard; cost-zero owners remain eligible."""
    setup = distributed_setup
    mesh = _mesh(setup)
    torch.manual_seed(342)
    model = _RepeatedRoot(16, setup.device)
    context, optimizer = _shard(
        model, mesh, enabled=True, retain=True, limit=limit, replicated_second=True
    )
    calls = []
    monkeypatch.setattr(context, "_sample_memory", _sampler(context, model, setup, calls))
    torch.manual_seed(435 + setup.rank)
    inputs = torch.randn(_MICROBATCHES, 4, 16, device=setup.device)
    for _ in range(2):
        _train(model, context, optimizer, inputs)
    events, plan = list(context.execution_order), context._memory_prefetch_plan
    assert model.second.prefetch_memory.buffer_bytes == 0
    active = set()
    unsafe_pairs = 0
    for position, event in enumerate(events):
        if event.operation is Operation.RESHARD:
            active.discard(event.module)
            if event.phase is Phase.FORWARD:
                assert not plan[position]
        else:
            active.add(event.module)
        for target in range(position + 1, len(events)):
            candidate = events[target]
            if candidate.operation is Operation.UNSHARD and any(
                e.module is candidate.module and e.operation is Operation.RESHARD
                for e in events[position + 1 : target]
            ):
                unsafe_pairs += 1
                assert target not in plan[position]
        materialized_elements = 0
        for target in plan[position]:
            assert position < target < len(events)
            candidate = events[target]
            assert candidate.operation is Operation.UNSHARD
            assert candidate.module not in active
            if candidate.module.prefetch_memory.buffer_bytes:
                materialized_elements += candidate.module.num_parameter_elements
        if limit is not None:
            assert materialized_elements <= limit
    assert unsafe_pairs, "The repeated trace must contain invalid future-RESHARD candidates."
    assert events[-1].operation is Operation.RESHARD and not plan[-1]
    if limit == 0:
        assert not any(plan)
    else:
        assert any(plan)
        assert any(events[target].module is model.second for targets in plan for target in targets)

    # Delegate every operation. Observe host submission order, without claiming
    # a GPU execution dependency between these independent communication streams.
    original_unshard = FsdpModule._unshard_parameter_groups
    original_prefetch = FsdpModule.prefetch
    original_gather = FsdpParameterGroup._allgather_model_weight
    original_reduce = FsdpParameterGroup.reduce_partial_gradients
    pending = {}
    observed = []
    source_stack = []
    submissions = []

    def observe_prefetch(owner, phase, *, position=None):
        if owner.context is not context:
            return original_prefetch(owner, phase, position=position)
        source = context.execution_order.position if position is None else position
        event = events[source]
        assert event.module is owner and event.phase is phase
        if event.operation is Operation.RESHARD:
            assert phase is Phase.BACKWARD and owner.phase is Phase.RESTING
            assert context.execution_order.position == (source + 1) % len(events)
        else:
            assert context.execution_order.position == source
        source_stack.append(source)
        try:
            result = original_prefetch(owner, phase, position=position)
            expected_cursor = (source + 1) % len(events)
            assert context.execution_order.position == expected_cursor
            return result
        finally:
            source_stack.pop()

    def observe_gather(group, **kwargs):
        if any(group is candidate for candidate in model.first.parameter_groups):
            submissions.append(("child_ag", source_stack[-1] if source_stack else None))
        return original_gather(group, **kwargs)

    def observe_reduce(group, partial_grad, *, is_last_microbatch):
        if any(group is candidate for candidate in model.parameter_groups):
            submissions.append(("root_rs", context.execution_order.position))
        return original_reduce(group, partial_grad, is_last_microbatch=is_last_microbatch)

    def observe(owner, phase):
        if owner.context is not context:
            return original_unshard(owner, phase)
        position = context.execution_order.position
        current = events[position]
        if not source_stack:
            if owner is model and phase is Phase.FORWARD:
                submissions.append(("root_demand", position))
            assert current.operation is Operation.UNSHARD and current.phase is phase
            assert current.module is owner
            saved = pending.pop(position, None)
            if saved is not None:
                assert all(
                    group._unshard_events.get(phase) is event for group, event in saved
                ), "An intervening cleanup invalidated a compiled prefetch."
            gathers_before = sum(kind == "child_ag" for kind, _ in submissions)
            original_unshard(owner, phase)
            if saved is not None:
                assert gathers_before == sum(kind == "child_ag" for kind, _ in submissions)
            return
        source = source_stack[-1]
        targets = [
            target
            for target in plan[source]
            if events[target].module is owner and events[target].phase is phase
        ]
        assert len(targets) == 1 and owner.phase is Phase.RESTING
        (target,) = targets
        assert target not in pending
        original_unshard(owner, phase)
        pending[target] = [
            (group, group._unshard_events[phase]) for group in owner.parameter_groups
        ]
        observed.append((source, target))

    monkeypatch.setattr(FsdpModule, "prefetch", observe_prefetch)
    monkeypatch.setattr(FsdpModule, "_unshard_parameter_groups", observe)
    monkeypatch.setattr(FsdpParameterGroup, "_allgather_model_weight", observe_gather)
    monkeypatch.setattr(FsdpParameterGroup, "reduce_partial_gradients", observe_reduce)
    monkeypatch.setattr(context, "_sample_memory", _forbid_sampling)
    before = _snapshot(model, gradients=False)
    result = _train(model, context, optimizer, inputs)
    assert not pending
    assert observed == [(i, target) for i, targets in enumerate(plan) for target in targets]
    assert context.execution_order.position == 0
    if limit != 0:
        root_forwards = [
            i
            for i, event in enumerate(events)
            if event.module is model
            and event.phase is Phase.FORWARD
            and event.operation is Operation.UNSHARD
        ]
        early = [
            source
            for source, target in observed
            if events[source].operation is Operation.RESHARD
            and events[source].phase is Phase.BACKWARD
            and events[target].module is model.first
            and events[target].phase is Phase.FORWARD
            and source < root_forwards[1] < target < root_forwards[2]
        ]
        assert early, "MB2's first child must be prefetched from MB1 backward completion."
        early_gather = next(
            i for i, entry in enumerate(submissions) if entry == ("child_ag", early[0])
        )
        root_reductions = [i for i, (kind, _) in enumerate(submissions) if kind == "root_rs"]
        next_demand = submissions.index(("root_demand", root_forwards[1]))
        assert len(root_reductions) == _MICROBATCHES
        assert early_gather < root_reductions[0] < next_demand
    assert any(not torch.equal(before[name], result[2][name]) for name in before)
    assert all(torch.isfinite(value).all() for value in result[1].values())


@pytest.mark.launch_on_gb200
def test_frozen_plan_captures_mixed_bf16_mxfp8_without_resampling(distributed_setup, monkeypatch):
    """A full train-step graph preserves a mixed BF16/MXFP8 model's losses, grads, and SGD."""
    setup = distributed_setup
    if torch.cuda.get_device_capability(setup.device)[0] < 10:
        pytest.skip("Primary MXFP8 weights require Blackwell or newer.")
    te = pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
        QuantizedDBuffer,
    )

    class MixedRoot(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(64, device=setup.device, dtype=torch.bfloat16) / 2)
            self.first = te.Linear(64, 128, params_dtype=torch.bfloat16, device=setup.device)
            self.second = te.Linear(128, 64, params_dtype=torch.bfloat16, device=setup.device)

        def forward(self, x):
            x = nn.functional.linear(x, self.weight)
            return self.second(nn.functional.gelu(self.first(x)))

    mesh = _mesh(setup)
    recipe = MXFP8BlockScaling()
    with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        torch.manual_seed(765)
        model = MixedRoot()
        torch.manual_seed(765)
        baseline = MixedRoot()
    assert isinstance(model.first.weight, MXFP8Tensor)
    context, optimizer = _shard(model, mesh, enabled=True, retain=True)
    baseline_context, baseline_optimizer = _shard(baseline, mesh, enabled=False, retain=True)
    monkeypatch.setattr(baseline_context, "_sample_memory", _forbid_sampling)
    calls = []
    real_sampler = context._sample_memory
    monkeypatch.setattr(context, "_sample_memory", _sampler(context, model, setup, calls))
    quantized = [
        group
        for _, owner in _owners(model)
        for group in owner.parameter_groups
        if isinstance(group._unsharded_model_weight, QuantizedDBuffer)
    ]
    assert len(quantized) == 2
    for group in quantized:
        buffer = group._unsharded_model_weight
        physical = sum(
            plane.local_buffer.numel() * plane.local_buffer.element_size()
            for plane in buffer.planes_for(rowwise=True, columnwise=True)
        )
        # Padded per-parameter TE scale copies are additional to both packed planes.
        assert group.unshard_storage_size > physical
        assert all(isinstance(p.unsharded, MXFP8Tensor) for p in group.fsdp_parameters)

    torch.manual_seed(876 + setup.rank)
    inputs = torch.randn(2, 32, 64, dtype=torch.bfloat16, device=setup.device)
    stream = torch.cuda.Stream(setup.device)
    stream.wait_stream(torch.cuda.current_stream(setup.device))

    def train_enabled():
        return _train(
            model, context, optimizer, inputs, autocast=lambda: te.autocast(recipe=recipe)
        )

    def train_baseline():
        return _train(
            baseline,
            baseline_context,
            baseline_optimizer,
            inputs,
            autocast=lambda: te.autocast(recipe=recipe),
        )

    # Warm up on the eventual capture stream, including a frozen eager iteration.
    for step in range(3):
        stream.wait_stream(torch.cuda.current_stream(setup.device))
        with torch.cuda.stream(stream):
            actual = train_enabled()
        torch.cuda.current_stream(setup.device).wait_stream(stream)
        _assert_result(actual, train_baseline(), rtol=3e-3, atol=3e-4)
        if step == 0:
            # Check the real sampling guard in an actual GPU capture, without
            # leaving a half-captured model's parameter lifecycle to be reused.
            monkeypatch.setattr(context, "_sample_memory", real_sampler)
            premature = torch.cuda.CUDAGraph()
            with pytest.raises(RuntimeError, match="two eager iterations"):
                with torch.cuda.graph(premature, stream=stream):
                    context._record_memory(0)
            monkeypatch.setattr(context, "_sample_memory", _sampler(context, model, setup, calls))
        elif step == 1:
            assert context._memory_profile_pass == 2 and any(context._memory_prefetch_plan)
            monkeypatch.setattr(context, "_sample_memory", _forbid_sampling)

            def forbid_compile():
                raise AssertionError("A frozen plan must not run consensus or compile again.")

            monkeypatch.setattr(context, "_compile_memory_prefetch_plan", forbid_compile)

    frozen = (
        context._memory_prefetch_plan,
        tuple(context._memory_prefetch_budgets),
        tuple(context._event_memory),
        len(calls),
    )
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream(setup.device))
    with torch.cuda.graph(graph, stream=stream):
        captured = train_enabled()
    # Capture records an update; it does not advance the independent eager model.
    for _ in range(3):
        stream.wait_stream(torch.cuda.current_stream(setup.device))
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream(setup.device).wait_stream(stream)
        _assert_result(captured, train_baseline(), rtol=3e-3, atol=3e-4)
        assert context._memory_profile_pass == 2
        assert frozen == (
            context._memory_prefetch_plan,
            tuple(context._memory_prefetch_budgets),
            tuple(context._event_memory),
            len(calls),
        )
    torch.cuda.synchronize(setup.device)
    del graph, captured, premature
