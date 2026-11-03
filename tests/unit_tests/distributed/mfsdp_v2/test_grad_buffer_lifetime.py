# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check bounded gradient staging against a reference, with and without graph replay."""

import importlib
from copy import deepcopy
from weakref import ref

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.ops.basic import BasicLinear, GroupedLinear
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import ShardLayout
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


class MixedLayer(nn.Module):
    """Two dtype groups in one owner, both computing in FP32."""

    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(8, device=device) / 2)
        self.bias = nn.Parameter(torch.zeros(8, device=device, dtype=torch.bfloat16))

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias.to(torch.bfloat16).float())


@pytest.mark.parametrize(
    "kind,quantized,strategy,layout,capture,delayed",
    [
        ("linear", False, "optim_grads_params", ShardLayout.PADDED, False, False),
        ("linear", False, "optim", ShardLayout.UNPADDED, False, False),
        ("linear", True, "optim_grads_params", ShardLayout.UNPADDED, False, False),
        ("linear", True, "optim_grads_params", ShardLayout.PADDED, True, False),
        ("linear", True, "optim", ShardLayout.UNPADDED, True, False),
        ("basic", False, "optim_grads_params", ShardLayout.PADDED, True, False),
        ("grouped", False, "optim_grads_params", ShardLayout.PADDED, True, False),
        ("grouped", True, "optim", ShardLayout.UNPADDED, True, False),
        ("linear", False, "optim_grads_params", ShardLayout.PADDED, False, True),
    ],
)
def test_te_fused_gradients_match_reference(
    distributed_setup, monkeypatch, kind, quantized, strategy, layout, capture, delayed
):
    """Real TE wgrad output, repeated uses, resets, mixed contributions and graph replay.

    Reference wgrad GEMMs also accumulate into FP32, avoiding a BF16 intermediate
    gradient or FP32 -> BF16 -> MXFP8 weight-refresh difference between the two paths.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    recipe = MXFP8BlockScaling()

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            # Keep a second compute dtype in this owner so fused TE outputs and
            # ordinary gradients share one module-level reduction/lifetime batch.
            self.bias32 = nn.Parameter(torch.zeros(64, dtype=torch.float32, device=device))
            if kind == "linear":
                self.proj = te.Linear(
                    64,
                    64,
                    params_dtype=torch.bfloat16,
                    device=device,
                    fuse_wgrad_accumulation=True,
                    delay_wgrad_compute=delayed,
                )
            elif kind == "basic":
                self.proj = BasicLinear(
                    64, 64, dtype=torch.bfloat16, device=device, accumulate_into_main_grad=True
                )
            else:
                self.proj = GroupedLinear(
                    2,
                    64,
                    64,
                    dtype=torch.bfloat16,
                    device="meta" if quantized else device,
                    accumulate_into_main_grad=True,
                    single_grouped_weight=False,
                    single_grouped_bias=False,
                )
                if quantized:
                    # Follow TEGroupedMLP._make_fused_ops: classic TE owns primary
                    # parameters and their preserved FP32 initialization; the ops
                    # shell uses those same objects rather than reinitializing them.
                    initialized = te.GroupedLinear(
                        2,
                        64,
                        64,
                        params_dtype=torch.bfloat16,
                        device=device,
                        fuse_wgrad_accumulation=True,
                        single_grouped_weight=False,
                        single_grouped_bias=False,
                    )
                    self.proj.register_parameter("weight", None)
                    self.proj.register_parameter("bias", None)
                    for name, parameter in initialized.named_parameters(recurse=False):
                        self.proj.register_parameter(name, parameter)
            # TE's group_quantize_mxfp8 requires a first logical dimension
            # divisible by128, beyond the basic MXFP8 block-size requirement.
            self.register_buffer("splits", torch.tensor([128, 128], device=device))
            if kind != "grouped" and not delayed:
                self.proj.weight.zero_out_wgrad = True

        def forward(self, value):
            if kind == "grouped":
                result = self.proj(value, self.splits)
            elif delayed:
                result = self.proj(value)
            else:
                # Two uses of one TE weight, plus a real autograd contribution.
                # Zero dummies must not hide it or double the fused GEMM output.
                result = self.proj(value) + self.proj(value * 0.5)
                result = result + nn.functional.linear(value, self.proj.weight) * 0.125
            return result.float() + self.bias32

    def build():
        torch.manual_seed(1729)
        with te.quantized_model_init(
            enabled=quantized, recipe=recipe, preserve_high_precision_init_val=True
        ):
            return Projection()

    model, reference = build(), build()
    for candidate in (model, reference):
        weights = [param for name, param in candidate.named_parameters() if "weight" in name]
        assert weights
        assert all(isinstance(param, MXFP8Tensor) == quantized for param in weights)
        if quantized:
            assert all(
                callable(getattr(param, "get_high_precision_init_val", None)) for param in weights
            )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected = reference_parameters[name]
        if isinstance(parameter, MXFP8Tensor):
            torch.testing.assert_close(
                parameter.get_high_precision_init_val(),
                expected.get_high_precision_init_val(),
                rtol=0,
                atol=0,
            )
            actual_compute, expected_compute = parameter.dequantize(), expected.dequantize()
        else:
            actual_compute, expected_compute = parameter.detach(), expected.detach()
        torch.testing.assert_close(actual_compute, expected_compute, rtol=0, atol=0)
    reference_main = {
        name: (
            param.get_high_precision_init_val()
            .detach()
            .to(device=device, dtype=torch.float32)
            .clone()
            if isinstance(param, MXFP8Tensor)
            else param.detach().float().clone()
        )
        for name, param in reference_parameters.items()
    }
    for param in reference_parameters.values():
        param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device=device)
        param.grad_added_to_main_grad = False
        param.zero_out_wgrad = True
    sharded = strategy == "optim_grads_params"
    with fully_shard_context(
        FsdpContext(
            device=device,
            prefetch_order=PrefetchOrder.RECORDED,
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
        )
    ) as context:
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0],
                parameter=[Shard(0) if sharded else Replicate()],
                gradient=[Shard(0) if sharded else Partial("avg")],
                optimizer=[Shard(0)],
            ),
            mixed_precision_policy=MixedPrecisionPolicy(
                main_params_dtype=torch.float32,
                main_grads_dtype=torch.float32,
                grad_comm_dtype=torch.float32,
            ),
            grad_scale=0.25,
            gradient_accumulation_fusion=True,
            shard_layout=layout,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    # Instrument the actual TE backward GEMM, not just the public fusion flag.
    te_impl = importlib.import_module(
        "transformer_engine.pytorch.module.linear"
        if kind == "linear"
        else (
            "transformer_engine.pytorch.ops.basic.basic_linear"
            if kind == "basic"
            else "transformer_engine.pytorch.ops.basic.grouped_linear"
        )
    )
    observed = []
    record_model = False
    for name in ("general_gemm", "general_grouped_gemm", "general_grouped_gemm_for_grouped_tensor"):
        if not hasattr(te_impl, name):
            continue
        original = getattr(te_impl, name)

        def observe(*args, _original=original, **kwargs):
            if record_model and kwargs.get("layout") == "NT":
                output = kwargs.get("out", args[2] if len(args) > 2 else None)
                outputs = output if isinstance(output, list) else [output]
                pointers = {
                    parameter.unsharded.main_grad.data_ptr()
                    for group in model.parameter_groups
                    for parameter in group.fsdp_parameters
                }
                for tensor in outputs:
                    if tensor is not None:
                        tensor = getattr(tensor, "rowwise_data", tensor)
                        assert tensor.dtype == torch.float32
                        assert tensor.data_ptr() in pointers
                observed.append(
                    {
                        "has_out": output is not None,
                        "out_dtype": kwargs.get("out_dtype"),
                        "accumulate": kwargs.get("accumulate"),
                    }
                )
            return _original(*args, **kwargs)

        monkeypatch.setattr(te_impl, name, observe)

    torch.manual_seed(19 + distributed_setup.rank)
    inputs = torch.randn(
        3, 256 if kind == "grouped" else 32, 64, dtype=torch.bfloat16, device=device
    )

    def train():
        nonlocal record_model
        record_model = True
        optimizer.zero_grad(set_to_none=False)
        losses = []
        for mb, value in enumerate(inputs):
            with microbatch(context, is_last=mb == 2):
                with te.autocast(enabled=quantized, recipe=recipe):
                    loss = model(value).float().square().mean() / 3
                loss.backward()
                if delayed:
                    model.proj.backward_dw()
            losses.append(loss.detach())
        context.finish_grad_sync()
        assert not context._pending_grad_buffers
        # Save pre-update shards: the optimizer must consume these exact gradients.
        grads = {id(p): p.grad.to_local().detach().clone() for p in model.parameters()}
        optimizer.step()
        record_model = False
        return torch.stack(losses), grads

    @torch.no_grad()
    def check(actual):
        actual_losses, actual_grads = actual
        expected_losses = []
        for param in reference_parameters.values():
            param.grad = None
            param.main_grad.zero_()
            param.grad_added_to_main_grad = False
        with torch.enable_grad():
            for value in inputs:
                with te.autocast(enabled=quantized, recipe=recipe):
                    loss = reference(value).float().square().mean() / 3
                loss.backward()
                if delayed:
                    reference.proj.backward_dw()
                expected_losses.append(loss.detach())
                # Match DDP/FSDP's per-microbatch promotion, not PyTorch's BF16
                # accumulation across microbatches for ordinary bias/embedding grads.
                for param in reference_parameters.values():
                    if param.grad is not None:
                        param.main_grad.add_(param.grad)
                        param.grad = None
        failures = []

        def compare(actual_tensor, expected, label):
            if (
                not torch.isfinite(actual_tensor).all().item()
                or not torch.isfinite(expected).all().item()
            ):
                failures.append(f"{label}: nonfinite values")
            try:
                torch.testing.assert_close(actual_tensor, expected, rtol=1e-5, atol=1e-6)
            except AssertionError as error:
                failures.append(f"{label}: {error}")

        compare(actual_losses, torch.stack(expected_losses), "loss")
        for param in reference_parameters.values():
            dist.all_reduce(param.main_grad, op=dist.ReduceOp.AVG, group=mesh.get_group())
            param.main_grad.mul_(0.25)
            if not torch.count_nonzero(param.main_grad).item():
                failures.append("The reference gradient has no signal")
        for group in model.parameter_groups:
            for index, parameter in enumerate(group.fsdp_parameters):
                (name,) = parameter.fqns
                baseline = reference_parameters[name]
                owned = group.main_weight._get_owned_range(index)
                actual_grad = actual_grads[id(parameter.sharded)]
                previous_main = reference_main[name].clone()
                reference_main[name].add_(baseline.main_grad, alpha=-0.01)
                if torch.equal(previous_main, reference_main[name]):
                    failures.append(f"{name}: no reference update")
                if owned is not None:
                    for actual_tensor, expected in (
                        (actual_grad, baseline.main_grad),
                        (group.main_weight.get_tensor_view(index), reference_main[name]),
                    ):
                        shard = expected.flatten().narrow(
                            0, owned.tensor_relative_offset, owned.numel
                        )
                        compare(actual_tensor.flatten(), shard, name)
                else:
                    if actual_grad.numel() != 0:
                        failures.append(f"{name}: an unowned shard is not empty")
                if isinstance(baseline, MXFP8Tensor):
                    baseline.quantize_(reference_main[name])
                else:
                    baseline.copy_(reference_main[name])
        if not any(record["accumulate"] and record["has_out"] for record in observed):
            failures.append("No actual TE wgrad GEMM accumulated into full main_grad")
        # A rank-local owned-shard assertion must not strand peers in the next
        # reference collective or fixture teardown barrier.
        all_failures = [None] * distributed_setup.world_size
        dist.all_gather_object(all_failures, failures, group=mesh.get_group())
        assert not any(all_failures), all_failures

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    for _ in range(2):
        with torch.cuda.stream(stream):
            actual = train()
        torch.cuda.current_stream(device).wait_stream(stream)
        check(actual)
        stream.wait_stream(torch.cuda.current_stream(device))
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = train()
    for _ in range(3):
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            if capture:
                graph.replay()
            else:
                actual = train()
        torch.cuda.current_stream(device).wait_stream(stream)
        check(actual)
    print(
        "TE_FUSED_WGRAD_CALL_PROOF",
        {
            "kind": kind,
            "mxfp8": quantized,
            "graph": capture,
            "delayed": delayed,
            "fused_output_calls": len(observed),
            "output_dtype": "float32",
            "output_pointer_matches_full_main_grad": True,
        },
    )


@pytest.mark.parametrize("strategy", ["optim", "optim_grads_params"])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_grad_buffer_lifetime_matches_reference(distributed_setup, monkeypatch, strategy, capture):
    """Exercise dtype groups, delayed local accumulation, allocator reuse, and final draining."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Sequential(*(MixedLayer(device) for _ in range(3)))
    reference = deepcopy(model).float()
    sharded = strategy == "optim_grads_params"
    placements = Placements(
        dp_axes=[0],
        parameter=[Shard(0) if sharded else Replicate()],
        gradient=[Shard(0) if sharded else Partial("avg")],
        optimizer=[Shard(0)],
    )
    with fully_shard_context(
        FsdpContext(
            device=device,
            prefetch_order=PrefetchOrder.RECORDED,
            prefetch_process_group=mesh.get_group(0),
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
        )
    ) as context:
        for layer in model:
            fully_shard(
                layer,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=MixedPrecisionPolicy(
                    main_params_dtype=torch.float32,
                    main_grads_dtype=torch.float32,
                    grad_comm_dtype=torch.float32,
                ),
            )
        fully_shard(model, mesh=mesh, placements=placements)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    torch.manual_seed(73 + distributed_setup.rank)
    inputs = torch.randn(3, 4, 8, device=device)
    reduce_gradients = FsdpParameterGroup.reduce_partial_gradients

    def delayed_reduce(group, partial_grad, *, is_last_microbatch):
        # ZeRO-1's intermediate microbatches consume this input without any NCCL call.
        torch.cuda._sleep(100_000)
        return reduce_gradients(group, partial_grad, is_last_microbatch=is_last_microbatch)

    monkeypatch.setattr(FsdpParameterGroup, "reduce_partial_gradients", delayed_reduce)

    def train():
        optimizer.zero_grad(set_to_none=False)
        losses = []
        for mb, x in enumerate(inputs):
            with microbatch(context, is_last=mb == len(inputs) - 1):
                loss = model(x).square().mean() / len(inputs)
                loss.backward()
            losses.append(loss.detach())
            assert len(context._pending_grad_buffers) == 2
            assert all(len(batch.buffers) == 2 for batch in context._pending_grad_buffers)
            # Reuse small compute-stream blocks while reduction inputs are still in flight.
            for _ in range(8):
                torch.empty(64, device=device).fill_(1.0e6)
        context.finish_grad_sync()
        assert not context._pending_grad_buffers
        optimizer.step()
        return torch.stack(losses)

    def check(actual_losses):
        reference_optimizer.zero_grad(set_to_none=False)
        expected_losses = []
        for x in inputs:
            loss = reference(x).square().mean() / len(inputs)
            loss.backward()
            expected_losses.append(loss.detach())
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG, group=mesh.get_group())
        torch.testing.assert_close(
            actual_losses, torch.stack(expected_losses), rtol=1e-5, atol=1e-6
        )
        reference_optimizer.step()
        for layer, expected_layer in zip(model, reference):
            for group in layer.parameter_groups:
                for index, parameter in enumerate(group.fsdp_parameters):
                    (name,) = parameter.fqns
                    expected = getattr(expected_layer, name)
                    owned = group.main_weight._get_owned_range(index)
                    if owned is None:
                        continue
                    for actual, baseline in (
                        (parameter.sharded.grad.to_local(), expected.grad),
                        (group.main_weight.get_tensor_view(index), expected.detach()),
                    ):
                        expected_shard = baseline.flatten().narrow(
                            0, owned.tensor_relative_offset, owned.numel
                        )
                        torch.testing.assert_close(
                            actual.flatten(), expected_shard, rtol=1e-5, atol=1e-6
                        )

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    for _ in range(2):
        with torch.cuda.stream(stream):
            losses = train()
        torch.cuda.current_stream(device).wait_stream(stream)
        check(losses)
        stream.wait_stream(torch.cuda.current_stream(device))

    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            losses = train()
    for _ in range(3):
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            if capture:
                graph.replay()
            else:
                losses = train()
        torch.cuda.current_stream(device).wait_stream(stream)
        check(losses)


def test_grad_buffer_release_orders_reuse_on_allocation_stream(distributed_setup):
    """Finalizing on another stream must still protect the input's original allocator."""
    device = distributed_setup.device
    context = FsdpContext(device=device)
    allocation_stream = torch.cuda.Stream(device)
    allocation_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(allocation_stream):
        source = torch.full((4096,), 7.0, device=device)
        result = torch.empty_like(source)
    context.reduce_scatter_stream.wait_stream(allocation_stream)
    with torch.cuda.stream(context.reduce_scatter_stream):
        torch.cuda._sleep(100_000)
        result.copy_(source)
    context._retain_grad_buffers([source], allocation_stream)
    source_ref = ref(source)
    del source
    assert source_ref() is not None

    context.finish_grad_sync()
    assert not context._pending_grad_buffers
    assert source_ref() is None
    with torch.cuda.stream(allocation_stream):
        replacement = torch.empty_like(result)
        replacement.fill_(-9.0)
    torch.cuda.current_stream(device).wait_stream(allocation_stream)
    torch.testing.assert_close(result, torch.full_like(result, 7.0), rtol=0, atol=0)
    torch.testing.assert_close(replacement, torch.full_like(replacement, -9.0), rtol=0, atol=0)


def test_forward_release_replays_the_recorded_occurrence(distributed_setup, monkeypatch):
    """Repeated calls retire inputs at the recorded occurrence, including graph replay."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Identity()
    with fully_shard_context(
        FsdpContext(
            device=device,
            prefetch_order=PrefetchOrder.RECORDED,
            prefetch_process_group=mesh.get_group(0),
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
        )
    ) as context:
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
            ),
        )
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        result = torch.empty(4096, device=device)

    def submit():
        with torch.cuda.stream(stream):
            source = torch.full_like(result, 7.0)
        context.reduce_scatter_stream.wait_stream(stream)
        with torch.cuda.stream(context.reduce_scatter_stream):
            torch.cuda._sleep(100_000)
            result.copy_(source)
        context._retain_grad_buffers([source], stream)
        return ref(source)

    with torch.cuda.stream(stream):
        context.allgather_stream.wait_stream(stream)
        model.unshard(Phase.FORWARD, prefetch=True)
    source_ref = submit()
    context._pending_grad_buffers[0].done.synchronize()
    with torch.cuda.stream(stream):
        context.allgather_stream.wait_stream(stream)
        model.unshard(Phase.FORWARD, prefetch=True)
        context.finish_grad_sync()
    assert source_ref() is None
    assert context._grad_buffer_release_plan == {1: (0,)}

    def unexpected_query(event):
        raise AssertionError("Replay must use the recorded plan, not query events")

    monkeypatch.setattr(torch.cuda.Event, "query", unexpected_query)

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("Replay must not compile another retirement plan")

    monkeypatch.setattr(dist, "all_reduce", unexpected_collective)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        context.allgather_stream.wait_stream(stream)
        model.unshard(Phase.FORWARD, prefetch=True)
        source_ref = submit()
        assert context._pending_grad_buffers[0].position == 0
        assert source_ref() is not None
        context.allgather_stream.wait_stream(stream)
        model.unshard(Phase.FORWARD, prefetch=True)
        assert source_ref() is None
        assert not context._pending_grad_buffers
        context.finish_grad_sync()
        replacement = torch.empty_like(result)
        replacement.fill_(-9.0)
    for _ in range(3):
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.testing.assert_close(result, torch.full_like(result, 7.0), rtol=0, atol=0)
        torch.testing.assert_close(replacement, torch.full_like(replacement, -9.0), rtol=0, atol=0)
        stream.wait_stream(torch.cuda.current_stream(device))
