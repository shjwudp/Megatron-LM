# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient completion for a shared parameter under combined 1F1B.

The combined 1F1B backward is one autograd GraphTask per schedule node over
detached node inputs, so a parameter's hooks fire once per ``(parameter,
consuming node)`` pair rather than once per iteration. The shared embedding is
the visible case: the PreProcessNode and the MTP pre-dispatch node each consume
it, so one iteration yields two contributions for one parameter and a fire count
cannot decide when the window is complete.

``MultiplicityReadiness`` replaces that count with a per-parameter declaration --
a shortfall holds the window open, a surplus raises -- and
``_unit_grad_multiplicity`` derives it. Its historical mistakes are pinned here:
``mtp_depth`` applied to every parameter, and the multiplicity taken from
``len(fqns)`` instead of object identity.

``countdown`` is stdlib-only, so the accounting tests load it from source and run
anywhere. The declaration tests need the real module and are gated on a host
precondition evaluated *before* that import, never on ``except ImportError``.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest

_EMBEDDING = ("module.embedding.word_embeddings.weight",)
_NORM = ("module.decoder.final_layernorm.weight",)
_SCHEDULER = "megatron.core.models.common.combined_1f1b_mfsdp_scheduler"
_COUNTDOWN = (
    Path(__file__).resolve().parents[4]
    / "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/countdown.py"
)


def _load_countdown():
    """Import ``countdown.py`` from source; it depends on nothing but stdlib."""
    spec = importlib.util.spec_from_file_location("_countdown_under_test", _COUNTDOWN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _host_can_import_stack() -> bool:
    """Return whether this host can import the Megatron stack at all."""
    try:
        importlib.import_module(_SCHEDULER)
    except Exception:  # any failure here is the precondition, not the test
        return False
    return True


MultiplicityReadiness = _load_countdown().MultiplicityReadiness

_needs_stack = pytest.mark.skipif(
    not _host_can_import_stack(),
    reason="host precondition unmet: this torch cannot import the Megatron stack, "
    "which is unrelated to the gradient multiplicity under test.",
)


class TestSharedParameterAcrossGraphTasks:
    """One parameter, two consuming schedule nodes, one completion window."""

    def test_a_declared_second_contribution_keeps_the_window_open(self):
        """The positive case: the window does not close after the first node."""
        readiness = MultiplicityReadiness({_EMBEDDING: 2, _NORM: 1})
        readiness.mark(_NORM)
        readiness.mark(_EMBEDDING)  # PreProcessNode embedding lookup
        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({_EMBEDDING})
        readiness.mark(_EMBEDDING)  # MTP pre-dispatch node
        assert readiness.is_complete()

    def test_an_undeclared_second_contribution_fails_loudly(self):
        """The negative case: a default expectation of one closes the window early.

        This is the defect the multiplicity exists to prevent. The window -- and
        with it the reshard and reduce-scatter -- closes after the PreProcessNode
        alone, and the MTP node's contribution then arrives as a surplus and is
        raised instead of being silently absorbed.
        """
        readiness = MultiplicityReadiness({_EMBEDDING: 1})
        readiness.mark(_EMBEDDING)
        assert readiness.is_complete()  # closed early: the reduce would fire here
        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(_EMBEDDING)

    def test_a_surplus_is_not_rolled_back(self):
        """An over-fire leaves the count above the declaration, so it stays visible."""
        readiness = MultiplicityReadiness({_EMBEDDING: 1})
        readiness.mark(_EMBEDDING)
        with pytest.raises(ValueError):
            readiness.mark(_EMBEDDING)
        with pytest.raises(ValueError):
            readiness.mark(_EMBEDDING)

    def test_reset_rearms_the_window_and_keeps_the_declaration(self):
        """The same parameter owes the same count on every iteration."""
        readiness = MultiplicityReadiness({_EMBEDDING: 2})
        for _ in range(3):
            readiness.mark(_EMBEDDING)
            readiness.mark(_EMBEDDING)
            assert readiness.is_complete()
            readiness.reset()
            assert not readiness.is_complete()
        assert readiness.expected == {_EMBEDDING: 2}

    def test_the_declaration_is_a_live_mapping(self):
        """``set_grad_multiplicity`` raises the bar in place, before the window opens."""
        readiness = MultiplicityReadiness({_EMBEDDING: 1, _NORM: 1})
        assert readiness.expected_total == 2
        readiness.expected.update({_EMBEDDING: 2})
        assert readiness.expected_total == 3

    def test_an_unknown_parameter_is_a_caller_error(self):
        """A key outside the declaration must not be swallowed."""
        with pytest.raises(KeyError):
            MultiplicityReadiness({_EMBEDDING: 1}).mark(_NORM)


class _FsdpParameter:
    """The three fields ``_unit_grad_multiplicity`` reads off an ``FsdpParameter``."""

    def __init__(self, fqns, unsharded, sharded=None):
        self.fqns, self.unsharded, self.sharded = fqns, unsharded, sharded


class _Unit:
    """The one method ``_unit_grad_multiplicity`` calls on an ``FsdpModule``."""

    def __init__(self, *parameters):
        self._parameters = parameters

    def _trainable_fsdp_parameters(self):
        return iter(self._parameters)


@pytest.mark.internal
@_needs_stack
class TestDeclaredMultiplicity:
    """The per-parameter declaration ``register_combined_1f1b_hooks`` installs."""

    @pytest.fixture(autouse=True)
    def _declaration(self):
        self.declare = importlib.import_module(_SCHEDULER)._unit_grad_multiplicity

    @pytest.mark.parametrize(
        ("mtp_depth", "tied", "expected"),
        [(0, False, 1), (1, False, 2), (0, True, 2), (1, True, 3)],
    )
    def test_only_the_embedding_accrues_extra_consumers(self, mtp_depth, tied, expected):
        """The MTP node adds one and a tied output projection another; nothing else moves."""
        weight = object()
        unit = _Unit(_FsdpParameter(_EMBEDDING, weight), _FsdpParameter(_NORM, object()))
        declaration = self.declare(unit, mtp_depth, weight, tied)
        assert (declaration[_EMBEDDING], declaration[_NORM]) == (expected, 1)

    def test_a_tied_parameter_contributes_once_whatever_its_fqns(self):
        """One physical parameter under two FQNs fires once per node: ``len(fqns)`` is not it."""
        tied = _EMBEDDING + ("module.output_layer.weight",)
        declaration = self.declare(_Unit(_FsdpParameter(tied, object())), 1, object(), True)
        assert declaration == {tied: 1}

    @pytest.mark.parametrize("half, sharded_first", [("unsharded", False), ("sharded", True)])
    def test_either_fsdp_half_is_recognised(self, half, sharded_first):
        """``_set_module_parameter`` installs ``.sharded`` or ``.unsharded``; the tree holds one."""
        weight = object()
        halves = (None, weight) if sharded_first else (weight, None)
        unit = _Unit(_FsdpParameter(_EMBEDDING, *halves), _FsdpParameter(_NORM, object()))
        assert self.declare(unit, 1, weight, False)[_EMBEDDING] == 2

    def test_a_missing_embedding_weight_matches_nothing(self):
        """``None`` must not satisfy the identity test for a parameter with an unset half."""
        unit = _Unit(_FsdpParameter(_EMBEDDING, object()), _FsdpParameter(_NORM, object()))
        assert set(self.declare(unit, 1, None, True).values()) == {1}
