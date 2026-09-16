# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-CPU tests for the MFSDP v2 pre-step gradient-placement diagnostic.

The check only compares ``tuple(param.grad.placements)`` against the owning group's
``main_weight.placements`` and formats the surrounding state on a mismatch, so it is
exercised here with lightweight stubs: no process group, no GPU, and no
``FullyShardedOptimizer`` instance.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from megatron.core.optimizer.fully_sharded_optimizer import (
    MFSDP_CHECK_GRAD_PLACEMENTS_ENV,
    check_gradient_placements,
    grad_placement_check_enabled,
)

OPTIMIZER_LABEL = "FullyShardedOptimizer(inner=FsdpMuon)"


def _parameter_with_grad(placements):
    """Build a stub parameter whose ``grad`` is a stub DTensor with ``placements``."""
    parameter = SimpleNamespace()
    parameter.grad = None if placements is None else SimpleNamespace(placements=placements)
    return parameter


def _attach_group(
    parameter,
    *,
    fqns=("weight",),
    main_weight_placements=(Shard(0), Shard(0)),
    main_grad_placements=(Replicate(), Shard(0)),
    pre_optimizer_main_grad_placements=(Shard(0), Shard(0)),
    mesh_shape=(2, 2),
    mesh_size=4,
    stale_flags=None,
):
    """Attach a stub ``FsdpParameterGroup`` backedge the way the real code does."""
    group = SimpleNamespace(
        main_weight=SimpleNamespace(placements=main_weight_placements),
        main_grad=SimpleNamespace(placements=main_grad_placements),
        pre_optimizer_main_grad=SimpleNamespace(placements=pre_optimizer_main_grad_placements),
        mesh=SimpleNamespace(mesh=torch.zeros(mesh_shape), size=lambda: mesh_size),
        fsdp_parameters=[SimpleNamespace(sharded=parameter, fqns=fqns)],
        **({} if stale_flags is None else stale_flags),
    )
    # Production stores a weakref here; get_containing_parameter_group() just calls it.
    parameter._mfsdp_parameter_group = lambda: group
    return group


def test_matching_placements_pass_and_report_counts():
    """A gradient already in the optimizer layout passes and is summarized once."""
    parameter_a = _parameter_with_grad((Shard(0), Shard(0)))
    _attach_group(parameter_a, fqns=("layers.0.weight",))
    parameter_b = _parameter_with_grad((Shard(0), Shard(0)))
    _attach_group(parameter_b, fqns=("layers.1.weight",))

    result = check_gradient_placements([parameter_a, parameter_b], optimizer_label=OPTIMIZER_LABEL)

    assert result.checked == 2
    assert result.skipped_no_grad == 0
    assert result.observed == (("[Shard(dim=0), Shard(dim=0)]", "[Shard(dim=0), Shard(dim=0)]"),)


def test_expected_layout_is_read_per_group_not_hardcoded():
    """Dense HFSDP ``[Shard, Shard]`` and expert ZeRO-1 ``[Shard]`` both validate in place."""
    dense = _parameter_with_grad((Shard(0), Shard(0)))
    _attach_group(
        dense,
        fqns=("decoder.weight",),
        main_weight_placements=(Shard(0), Shard(0)),
        pre_optimizer_main_grad_placements=(Shard(0), Shard(0)),
        mesh_shape=(2, 2),
        mesh_size=4,
    )
    expert = _parameter_with_grad((Shard(0),))
    _attach_group(
        expert,
        fqns=("experts.weight",),
        main_weight_placements=(Shard(0),),
        main_grad_placements=(Shard(0),),
        pre_optimizer_main_grad_placements=(Shard(0),),
        mesh_shape=(4,),
        mesh_size=4,
    )

    result = check_gradient_placements([dense, expert], optimizer_label=OPTIMIZER_LABEL)

    assert result.checked == 2
    assert result.observed == (
        ("[Shard(dim=0), Shard(dim=0)]", "[Shard(dim=0), Shard(dim=0)]"),
        ("[Shard(dim=0)]", "[Shard(dim=0)]"),
    )


def test_gradient_none_is_skipped_but_counted():
    """A parameter with no gradient is not stepped, so it is counted and skipped."""
    stepped = _parameter_with_grad((Shard(0),))
    _attach_group(stepped, main_weight_placements=(Shard(0),))
    dormant = _parameter_with_grad(None)

    result = check_gradient_placements([stepped, dormant], optimizer_label=OPTIMIZER_LABEL)

    assert result.checked == 1
    assert result.skipped_no_grad == 1


def test_gradient_in_parameter_layout_raises_full_diagnostic():
    """A grad left in the parameter layout is caught with the whole explanatory payload."""
    parameter = _parameter_with_grad((Replicate(), Shard(0)))
    _attach_group(
        parameter,
        fqns=("decoder.layers.3.mlp.linear_fc1.weight",),
        main_weight_placements=(Shard(0), Shard(0)),
        main_grad_placements=(Replicate(), Shard(0)),
        pre_optimizer_main_grad_placements=(Shard(0), Shard(0)),
        mesh_shape=(2, 2),
        mesh_size=4,
        stale_flags={
            "_model_weight_is_stale": False,
            "_main_grad_is_stale": True,
            "_rowwise_is_stale": True,
            "_colwise_is_stale": False,
        },
    )

    with pytest.raises(RuntimeError) as excinfo:
        check_gradient_placements([parameter], optimizer_label=OPTIMIZER_LABEL)

    message = str(excinfo.value)
    assert "MFSDP v2 gradient-placement check failed" in message
    assert OPTIMIZER_LABEL in message
    assert "decoder.layers.3.mlp.linear_fc1.weight" in message
    assert "grad.placements=[Replicate(), Shard(dim=0)]" in message
    assert "expected(main_weight).placements=[Shard(dim=0), Shard(dim=0)]" in message
    assert "main_grad.placements=[Replicate(), Shard(dim=0)]" in message
    assert "pre_optimizer_main_grad.placements=[Shard(dim=0), Shard(dim=0)]" in message
    assert "mesh.shape=(2, 2)" in message
    assert "mesh.size=4" in message
    assert "_main_grad_is_stale=True" in message
    assert "_model_weight_is_stale=False" in message
    assert "_rowwise_is_stale=True" in message
    assert "violations=1" in message


def test_non_dtensor_gradient_is_reported():
    """A plain-tensor grad has no placements and cannot be in any DTensor layout."""
    parameter = SimpleNamespace(grad=torch.zeros(4))
    _attach_group(parameter)

    with pytest.raises(RuntimeError) as excinfo:
        check_gradient_placements([parameter], optimizer_label=OPTIMIZER_LABEL)

    assert "grad.placements=<none>" in str(excinfo.value)


def test_missing_parameter_group_is_reported():
    """A stepped parameter that no group owns is anomalous for MFSDP v2."""
    parameter = _parameter_with_grad((Shard(0),))

    with pytest.raises(RuntimeError) as excinfo:
        check_gradient_placements([parameter], optimizer_label=OPTIMIZER_LABEL)

    message = str(excinfo.value)
    assert "parameter_group=<none>" in message
    assert "grad.placements=[Shard(dim=0)]" in message
    assert "violations=1" in message


def test_violation_report_is_capped():
    """The report is bounded so one failing step stays readable."""
    parameters = []
    for index in range(20):
        parameter = _parameter_with_grad((Replicate(), Shard(0)))
        _attach_group(parameter, fqns=(f"layers.{index}.weight",))
        parameters.append(parameter)

    with pytest.raises(RuntimeError) as excinfo:
        check_gradient_placements(parameters, optimizer_label=OPTIMIZER_LABEL)

    message = str(excinfo.value)
    assert "violations=20" in message
    assert "+4 more violating parameter(s) not shown" in message


def test_env_gate(monkeypatch):
    """The diagnostic is off unless MFSDP_CHECK_GRAD_PLACEMENTS is set to a truthy value."""
    monkeypatch.delenv(MFSDP_CHECK_GRAD_PLACEMENTS_ENV, raising=False)
    assert grad_placement_check_enabled() is False

    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(MFSDP_CHECK_GRAD_PLACEMENTS_ENV, value)
        assert grad_placement_check_enabled() is True

    for value in ("0", "false", "off", ""):
        monkeypatch.setenv(MFSDP_CHECK_GRAD_PLACEMENTS_ENV, value)
        assert grad_placement_check_enabled() is False
