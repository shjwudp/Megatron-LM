# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient completion for a shared parameter under combined 1F1B.

The combined/fine-grained 1F1B backward is one autograd GraphTask per schedule
node over detached node inputs, so a parameter's grad hooks fire once per
``(parameter, consuming node)`` pair rather than once per iteration. A shared
parameter owes one contribution per consuming node -- the embedding under MTP is
the visible case -- and a fire count cannot decide when the window is complete.

``MultiplicityReadiness`` replaces the fire count with a per-parameter
declaration: a shortfall holds the window open, a surplus raises.
``FsdpModule.register_post_backward_hook`` turns that declaration into one
finalize per window, across as many GraphTasks as the window spans. Both are
exercised here.
"""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.countdown import (
    MultiplicityReadiness,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


class TwoNodeUnit(nn.Module):
    """Two parameters behind two graph nodes, one FSDP unit."""

    def __init__(self) -> None:
        """Build the two projections."""
        super().__init__()
        self.first = nn.Linear(16, 16, bias=False)
        self.second = nn.Linear(16, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project twice so each parameter owns a distinct graph node."""
        return self.second(self.first(inputs))


def _sharded_unit(setup, multiplicity: int):
    """Fully shard a ``TwoNodeUnit`` and install a completion hook behind a spy."""
    torch.manual_seed(setup.rank)
    model = TwoNodeUnit().to(setup.device)
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=setup.device):
        # ``register_hooks=False`` mirrors production, where the combined
        # scheduler installs its own completion hooks instead of the default.
        fully_shard(model, mesh=mesh, placements=placements, register_hooks=False)
    model.set_grad_multiplicity(
        {parameter.fqns: multiplicity for parameter in model._trainable_fsdp_parameters()}
    )
    finalized = []
    model.register_post_backward_hook(lambda hooked_module: finalized.append(hooked_module))
    return model, mesh, finalized


def _graph_task(model: FsdpModule, mesh, setup) -> None:
    """Run one forward-backward: one independent autograd GraphTask.

    The schedule hands each node a detached DTensor, so the input is a DTensor
    too -- the unit's weights are sharded DTensors and cannot mix with a plain
    tensor inside ``aten.mm``.
    """
    inputs = DTensor.from_local(torch.randn(4, 16, device=setup.device), mesh, [Replicate()])
    model(inputs).sum().backward()


class TestSharedParameterAcrossGraphTasks:
    """One parameter, two consuming schedule nodes, one completion window."""

    def test_a_declared_second_contribution_keeps_the_window_open(self):
        """The positive case: the window does not close after the first node."""
        embedding, norm = ("embedding.weight",), ("norm.weight",)
        readiness = MultiplicityReadiness({embedding: 2, norm: 1})
        readiness.mark(norm)
        readiness.mark(embedding)  # PreProcessNode embedding lookup
        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({embedding})
        readiness.mark(embedding)  # MTP pre-dispatch node
        assert readiness.is_complete()

    def test_an_undeclared_second_contribution_fails_loudly(self):
        """The negative case: a default expectation of one closes the window early.

        This is the defect the multiplicity exists to prevent: the window --
        and with it the reshard and reduce-scatter -- closes after the first
        node alone, and the second contribution then arrives as a surplus and
        is raised rather than silently absorbed into the next window.
        """
        key = ("embedding.weight",)
        readiness = MultiplicityReadiness({key: 1})
        readiness.mark(key)
        assert readiness.is_complete()  # closed early: the reduce would fire here
        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(key)

    def test_reset_rearms_the_window_and_keeps_the_declaration(self):
        """The same parameter owes the same count on every iteration."""
        key = ("embedding.weight",)
        readiness = MultiplicityReadiness({key: 2})
        for _ in range(3):
            readiness.mark(key)
            readiness.mark(key)
            assert readiness.is_complete()
            readiness.reset()
            assert not readiness.is_complete()
        assert readiness.expected == {key: 2}

    def test_the_declaration_is_a_live_mapping(self):
        """``set_grad_multiplicity`` raises the bar in place, before the window opens."""
        embedding, norm = ("embedding.weight",), ("norm.weight",)
        readiness = MultiplicityReadiness({embedding: 1, norm: 1})
        assert readiness.expected_total == 2
        readiness.expected.update({embedding: 2})
        assert readiness.expected_total == 3

    def test_an_unknown_parameter_is_a_caller_error(self):
        """A key outside the declaration must not be swallowed."""
        with pytest.raises(KeyError):
            MultiplicityReadiness({("embedding.weight",): 1}).mark(("norm.weight",))


class TestPostBackwardHookAcrossGraphTasks:
    """``register_post_backward_hook`` with one GraphTask per schedule node."""

    def test_the_window_finalizes_once_across_graph_tasks(self, distributed_setup):
        """Two GraphTasks owe two contributions; the hook fires after the second.

        This is the coexistence under test: a finalize per GraphTask would
        reshard and reduce after the first schedule node's backward and drop the
        second node's contribution. The declared multiplicity holds the window
        open until every GraphTask has landed.
        """
        model, mesh, finalized = _sharded_unit(distributed_setup, multiplicity=2)

        _graph_task(model, mesh, distributed_setup)  # first schedule node
        assert finalized == [], "the window closed after the first GraphTask"
        _graph_task(model, mesh, distributed_setup)  # second schedule node
        assert len(finalized) == 1, f"expected one finalize, got {len(finalized)}"

    def test_an_undeclared_second_graph_task_fails_loudly(self, distributed_setup):
        """A declaration of one under two GraphTasks raises instead of double-reducing.

        The first GraphTask completes the window and finalizes; the second's
        contribution then arrives as a surplus and is raised rather than
        silently accumulated into the next window.
        """
        model, mesh, finalized = _sharded_unit(distributed_setup, multiplicity=1)

        _graph_task(model, mesh, distributed_setup)
        assert len(finalized) == 1
        with pytest.raises(Exception, match="over-fired"):
            _graph_task(model, mesh, distributed_setup)
