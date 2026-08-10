# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental Megatron-FSDP runtime contexts."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
)


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


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def test_child_then_parent_share_one_context(distributed_setup):
    """Modules constructed together should eagerly share one context."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    with fully_shard_context(device=device) as context:
        fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())
        assert model.context is context
        assert model.inner.context is context

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.inner.context is model.context
    assert model.is_root()
    assert not model.inner.is_root()


def test_two_child_subtrees_then_parent_share_one_context(distributed_setup):
    """One construction scope should assign one context across child subtrees."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    assert model.layers[0].context is model.context
    assert model.layers[1].context is model.context


def test_sibling_roots_share_context_and_cross_root_orders(distributed_setup):
    """Independent roots should share streams and follow construction order."""
def test_fine_grained_hooks_preserve_registered_module_hierarchy(distributed_setup):
    """Fine-grained parent references must not become registered child modules."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)
    module_names = tuple(name for name, _ in model.named_modules())
    layer_keys = tuple(model.layers._modules)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    assert tuple(name for name, _ in model.named_modules()) == module_names
    assert tuple(model.layers._modules) == layer_keys


def test_sibling_roots_without_parent_keep_separate_contexts(distributed_setup):
    """Independent FSDP roots should not share runtime scheduling state."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.layers[0].context
    assert model.layers[1].context is context
    assert model.layers[0].is_root()
    assert model.layers[1].is_root()
    assert list(context.forward_order) == [model.layers[0], model.layers[1]]
    assert list(context.backward_order) == [model.layers[1], model.layers[0]]


def test_nested_prefetch_orders_use_dfs(distributed_setup):
    """Nested FsdpModules should use DFS orders for one-step prefetch."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.left.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model.left, mesh=mesh, placements=_flat_placements())
        fully_shard(model.right, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    with torch.no_grad():
        model(torch.ones(2, 4, device=device))

    context = model.context
    assert list(context.forward_order) == [model, model.left, model.left.inner, model.right]
    assert list(context.backward_order) == [model, model.right, model.left, model.left.inner]


def test_nested_and_sibling_roots_use_cross_root_orders(distributed_setup):
    """Context orders should concatenate nested roots at construction boundaries."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedSiblingModel(dim=4).to(device)

    with fully_shard_context(device=device):
        fully_shard(model.left.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model.left, mesh=mesh, placements=_flat_placements())
        fully_shard(model.right, mesh=mesh, placements=_flat_placements())

    context = model.left.context
    assert model.left.is_root()
    assert model.right.is_root()
    assert not model.left.inner.is_root()
    assert list(context.forward_order) == [model.left, model.left.inner, model.right]
    assert list(context.backward_order) == [model.right, model.left, model.left.inner]


def test_fully_shard_requires_context(distributed_setup):
    """fully_shard should reject construction without an active context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)

    with pytest.raises(RuntimeError, match="inside fully_shard_context"):
        fully_shard(model, mesh=mesh, placements=_flat_placements())


def test_forward_requires_finalized_context(distributed_setup):
    """Forward should be unavailable until construction scope exit."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)
    x = torch.ones(2, 4, device=device)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())
        with pytest.raises(RuntimeError, match="Exit fully_shard_context"):
            model(x)

    model(x)


def test_fully_shard_context_rejects_nesting(distributed_setup):
    """A construction scope should reject an ambiguous nested context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).to(device)

    with fully_shard_context(device=device):
        fully_shard(model[0], mesh=mesh, placements=_flat_placements())
        outer_context = model[0].context
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(device=device):
                pass
        fully_shard(model[1], mesh=mesh, placements=_flat_placements())

    assert model[0].context is outer_context
    assert model[1].context is outer_context


def test_fully_shard_rejects_child_from_another_context(distributed_setup):
    """A parent cannot join a context different from an FSDP child context."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel()

    with fully_shard_context(device=device) as first_context:
        fully_shard(model.inner, mesh=mesh, placements=_flat_placements())

    with fully_shard_context(device=device):
        with pytest.raises(ValueError, match="another fully_shard_context"):
            fully_shard(model, mesh=mesh, placements=_flat_placements())

    assert model.inner.context is first_context
def test_post_backward_release_processes_nested_fsdp_modules_once(distributed_setup, monkeypatch):
    """Manual 1F1B release should include nested units without reducing twice."""
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = NestedModel().to(device)

    with fully_shard_context(device=device):
        fully_shard(
            model.inner, mesh=mesh, placements=_flat_placements(), skip_backward_callback=True
        )
        fully_shard(model, mesh=mesh, placements=_flat_placements(), skip_backward_callback=True)

    calls = []
    for name, module in (("root", model), ("inner", model.inner)):
        monkeypatch.setattr(
            module,
            "_reshard_parameter_groups",
            lambda name=name: calls.append((name, "reshard")),
        )
        monkeypatch.setattr(
            module, "_reduce_gradient_groups", lambda name=name: calls.append((name, "reduce"))
        )

    model.post_backward_release_module()
    model.post_backward_release_module()

    # Each nested unit is resharded and reduced exactly once per backward;
    # the relative order of the two operations is not a contract.
    assert sorted(calls) == [
        ("inner", "reduce"),
        ("inner", "reshard"),
        ("root", "reduce"),
        ("root", "reshard"),
    ]


def test_vpp_chunks_share_one_context_via_reuse(distributed_setup):
    """VPP chunks wrapped inside one scope should share a single FsdpContext.

    Simulates the training-loop wrapping of multiple virtual-pipeline chunks:
    the outer fully_shard_context() is opened once, and each chunk's adapter
    (modeled here by nested reuse_existing scopes) joins it instead of
    creating a new context. All chunks must share streams and cross-root
    prefetch orders.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    chunks = [MultiChildModel(dim=4, num_children=2).to(device) for _ in range(2)]

    with fully_shard_context(device=device) as outer:
        for chunk in chunks:
            # Mirrors FullyShardedDataParallelV2.__init__ wrapping a chunk:
            # reuse_existing joins the training-loop context.
            with fully_shard_context(device=device, reuse_existing=True):
                fully_shard(chunk, mesh=mesh, placements=_flat_placements())

        assert chunk.context is outer

    # After finalize, every chunk root is registered in the shared context's
    # cross-root orders.
    for chunk in chunks:
        assert chunk.context is outer
        assert chunk.is_root()

    assert len(list(outer.forward_order)) == 2
    assert len(list(outer.backward_order)) == 2
    assert outer.allgather_stream is chunks[0].context.allgather_stream
    assert outer.reduce_scatter_stream is chunks[0].context.reduce_scatter_stream


def test_vpp_chunks_reuse_context_on_same_device_only(distributed_setup):
    """reuse_existing must join only a context on the same device."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        # A different device must not be joined silently; the ambient context
        # is CUDA so requesting a CPU context keeps the nesting rejection.
        with pytest.raises(RuntimeError, match="does not support nesting"):
            with fully_shard_context(device=torch.device("cpu"), reuse_existing=True):
                pass
        fully_shard(model, mesh=mesh, placements=_flat_placements())
def test_prefetch_traces_and_replays_actual_consume_order(distributed_setup):
    """The runner should trace batch 1 and replay the actual consume order.

    The fine-grained schedule can consume modules in an order that differs
    from forward_order/backward_order (e.g. F L0 -> B L2 -> F L1). The first
    batch traces that order and returns no prefetch; later batches replay it
    and prefetch the actual next consumer.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    runner = model.context.runner
    assert runner.is_tracing

    # Batch 1 (trace): consume in schedule order F L0, B L2, F L1. No
    # prefetch during tracing.
    assert runner.record_consume(layers[0], "rowwise") is None
    assert runner.record_consume(layers[2], "colwise") is None
    assert runner.record_consume(layers[1], "rowwise") is None
    assert runner.is_tracing

    # Batch boundary compiles the trace into the replay cycle.
    runner.complete_trace()
    assert not runner.is_tracing

    # Batch 2 (replay): consume in the same order; each call returns the
    # traced next consumer (with wrap-around at the batch boundary).
    assert runner.record_consume(layers[0], "rowwise") is layers[2]
    assert runner.next_prefetch_orientation() == "colwise"
    assert runner.record_consume(layers[2], "colwise") is layers[1]
    assert runner.record_consume(layers[1], "rowwise") is layers[0]

    # Divergence re-traces from the mismatching occurrence.
    assert runner.record_consume(layers[0], "colwise") is None
    assert runner.is_tracing


def test_prefetch_releases_stale_unconsumed_modules(distributed_setup):
    """Unshard should release prefetched modules the schedule never consumes.

    After replay, the prefetched successor is the traced next consumer, so a
    module that is prefetched but skipped must be resharded instead of
    accumulating unsharded storage, while the consuming module and the newly
    prefetched successor stay materialized.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=3).to(device)

    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements(), fine_grained=True)
        fully_shard(model, mesh=mesh, placements=_flat_placements(), fine_grained=True)

    layers = model.layers
    ctx = model.context

    # Trace batch: L0 -> L2 -> L1, then compile.
    ctx.runner.record_consume(layers[0], "rowwise")
    ctx.runner.record_consume(layers[2], "colwise")
    ctx.runner.record_consume(layers[1], "rowwise")
    ctx.runner.complete_trace()

    # Replay batch: consume L0, prefetch L2 (traced successor).
    layers[0].unshard_parameters()
    assert ctx._prefetched_modules == {layers[2]}
    assert layers[0]._unshard_event is not None

    # Consume L1 out of traced order (schedule divergence): L2 was prefetched
    # but is skipped, so it must be resharded and removed from the trace.
    layers[1].unshard_parameters()
    assert layers[2]._unshard_event is None
    assert layers[2] not in ctx._prefetched_modules
    # The consumers stay unsharded and out of the prefetch trace.
    assert layers[0]._unshard_event is not None
    assert layers[1]._unshard_event is not None
    assert layers[0] not in ctx._prefetched_modules
    assert layers[1] not in ctx._prefetched_modules


def test_eager_pre_forward_feeds_context_runner(distributed_setup):
    """The eager pre_forward path must also feed the context-wide runner.

    The runner is shared across the full FsdpContext, so a consume driven by
    the eager forward hooks is traced identically to a fine-grained consume:
    batch 1 records and prefetches via the static order fallback, batch 2
    replays the traced order.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)

    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    layers = model.layers
    ctx = model.context
    assert ctx.runner.is_tracing

    # Eager forward consumes are recorded on the shared runner.
    with torch.no_grad():
        model(torch.ones(2, 4, device=device))
    assert not ctx.runner.is_tracing
    assert len(ctx.runner._trace) >= 2  # root + child layers
