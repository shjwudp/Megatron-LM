# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient multiplicity for the MFSDP v2 combined/fine-grained 1F1B path.

The combined 1F1B schedule runs backward as one autograd GraphTask per schedule
node, on detached node inputs. A parameter's gradient callbacks therefore arrive
once per ``(parameter, consuming node)`` pair rather than once per parameter:

* a parameter shared by two nodes fires **twice** in one iteration;
* a parameter no node consumed fires **zero** times.

Completion cannot be inferred from a fire count under those conditions, so the
schedule declares how many contributions each parameter owes and
``MultiplicityReadiness`` counts against that declaration. ``missing`` and the
over-fire ``ValueError`` are the two directions: a shortfall means a reduce was
held back, a surplus means the declaration was too small and an earlier window
closed early.

``_unit_grad_multiplicity`` is where the declaration is derived, and it has
historically been wrong in exactly two ways. Both are pinned here as regression
guards, because both are silent rather than loud in production:

1. **Enumeration base.** ``mtp_depth`` was once applied to *every* parameter, so
   every ordinary parameter waited for a contribution that never arrived. The
   declaration is per-parameter, and only the shared embedding accrues extra
   consumers.
2. **Identity, not FQN count.** A tied parameter is one physical ``nn.Parameter``
   registered under several FQNs, and it fires once per consuming node. Its
   multiplicity must therefore not be derived from ``len(fqns)``; matching the
   embedding must be done by object identity.

``countdown`` is deliberately dependency-free (standard library only), so the
accounting contract is loaded straight from source and tested without a GPU. The
two scheduler helpers are reached through the real module instead, which imports
``FsdpModule`` at module scope and so needs the full Megatron/torch stack; those
tests are gated on a host precondition evaluated *before* that import, never on an
``except ImportError`` -- an import failure on a host that satisfies the
precondition must fail loudly rather than dissolve into a skip.
"""

import ast
import importlib
import importlib.util
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]

_COUNTDOWN_PATH = (
    _REPO_ROOT
    / "megatron"
    / "core"
    / "distributed"
    / "fsdp"
    / "src"
    / "megatron_fsdp"
    / "experimental"
    / "countdown.py"
)

_SCHEDULER_MODULE = "megatron.core.models.common.combined_1f1b_mfsdp_scheduler"

# The environmental precondition: the Megatron stack in this tree needs a torch
# dtype that older torch releases do not define, so on such a host the scheduler
# module cannot be imported for reasons unrelated to the code under test.
_REQUIRED_TORCH_ATTR = "float8_e8m0fnu"

_MEGATRON_STACK_SKIP_REASON = (
    f"host precondition unmet: torch has no '{_REQUIRED_TORCH_ATTR}', so this host "
    "cannot import the Megatron stack for reasons unrelated to the gradient "
    "multiplicity under test. The dependency-free accounting contract and the gate "
    "discipline test still run here."
)


def _host_can_import_megatron_stack() -> bool:
    """Return whether this host can import the Megatron stack at all.

    Evaluated BEFORE any Megatron import and answerable without touching the import
    graph, which is the point: the scheduler helpers live in a module whose scope
    imports ``FsdpModule``, so the import fails on a host whose torch predates the
    stack. Only that environmental cause is tested here, so the skip decision rests
    on a fact that has nothing to do with the import under test -- a broken import
    graph can never reach this function and cannot turn into a skip.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of this suite
        return False
    return hasattr(torch, _REQUIRED_TORCH_ATTR)


HOST_CAN_IMPORT_MEGATRON_STACK = _host_can_import_megatron_stack()


def _load_countdown():
    """Load the dependency-free ``countdown`` module without the MFSDP package.

    Importing it through ``megatron_fsdp`` executes the package ``__init__``, which
    needs the CUDA/TE stack; the module itself imports nothing outside the standard
    library, so it is loaded from source and runs on any host.
    """
    spec = importlib.util.spec_from_file_location("mfsdp_countdown_under_test", _COUNTDOWN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


countdown = _load_countdown()
MultiplicityReadiness = countdown.MultiplicityReadiness


def _import_scheduler_module():
    """Import the scheduler module, failing loudly instead of skipping.

    ``pytest.fail`` and not a skip: every caller is gated on
    ``HOST_CAN_IMPORT_MEGATRON_STACK``, so a host that reaches this function has
    already excluded the environmental cause, and an import failure here is a real
    defect that must not be reported as an unavailable environment.
    """
    try:
        return importlib.import_module(_SCHEDULER_MODULE)
    except (ImportError, AttributeError) as error:
        pytest.fail(
            f"importing {_SCHEDULER_MODULE!r} raised {type(error).__name__}: {error}. "
            f"This host satisfies the {_REQUIRED_TORCH_ATTR} precondition, so the "
            "environmental cause is excluded and the failure is real."
        )


@pytest.fixture(scope="module")
def multiplicity_readiness():
    """The production completion signal: counting against a declaration."""
    return MultiplicityReadiness


@pytest.fixture(scope="module")
def active_mtp_layers():
    """The scheduler's ``_active_mtp_layers``, from the real module."""
    return _import_scheduler_module()._active_mtp_layers


@pytest.fixture(scope="module")
def unit_grad_multiplicity():
    """The scheduler's ``_unit_grad_multiplicity``, from the real module."""
    return _import_scheduler_module()._unit_grad_multiplicity


class TestMultiplicityReadinessContract:
    """The declaration-counting contract the schedule's completion rests on."""

    def test_expected_total_sums_the_declared_multiplicities(self, multiplicity_readiness):
        """``expected_total`` is the number of marks one correct window produces."""
        readiness = multiplicity_readiness({("E",): 2, ("N",): 1})

        assert readiness.expected_total == 3

    def test_an_empty_declaration_is_trivially_complete(self, multiplicity_readiness):
        """A unit owning no trainable parameter has nothing to wait for.

        ``FsdpModule`` branches on ``expected_total == 0`` to fall back to the
        full-backward hook, so "no parameters" has to read as complete and not as a
        window that can never close.
        """
        readiness = multiplicity_readiness({})

        assert readiness.expected_total == 0
        assert readiness.missing() == frozenset()
        assert readiness.is_complete()

    def test_marking_an_unknown_key_is_an_error_not_a_silent_drop(self, multiplicity_readiness):
        """An unknown key is a caller bug: dropping it would hide a contribution."""
        readiness = multiplicity_readiness({("owned",): 1})

        with pytest.raises(KeyError, match="not a known parameter"):
            readiness.mark(("not-owned",))

    def test_marking_beyond_the_declaration_raises(self, multiplicity_readiness):
        """More contributions than declared means the declaration is too small."""
        readiness = multiplicity_readiness({("E",): 1})
        readiness.mark(("E",))

        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(("E",))

    def test_an_over_fire_leaves_the_observation_recorded(self, multiplicity_readiness):
        """``mark`` counts before it complains, so the surplus stays on the record.

        The count is deliberately not rolled back: the extra contribution really
        happened, and a window that hid it would look satisfied on the next check.
        """
        readiness = multiplicity_readiness({("E",): 1})
        readiness.mark(("E",))
        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(("E",))

        assert readiness.missing() == frozenset()
        assert readiness.is_complete()

        # Still above the declaration, so the next contribution is refused too.
        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(("E",))

    def test_a_key_that_was_never_marked_is_missing(self, multiplicity_readiness):
        """A parameter no schedule node consumed is a shortfall, not an absence."""
        readiness = multiplicity_readiness({("a",): 1, ("b",): 2})
        readiness.mark(("b",))

        assert readiness.missing() == frozenset({("a",), ("b",)})
        assert not readiness.is_complete()

    def test_missing_names_only_the_key_below_its_declaration(self, multiplicity_readiness):
        """With the satisfied key marked, ``missing`` names exactly the short one."""
        readiness = multiplicity_readiness({("a",): 1, ("b",): 2})
        readiness.mark(("a",))
        readiness.mark(("b",))

        assert readiness.missing() == frozenset({("b",)})
        assert not readiness.is_complete()

        readiness.mark(("b",))
        assert readiness.missing() == frozenset()
        assert readiness.is_complete()

    def test_reset_zeroes_the_counts_and_keeps_the_declaration(self, multiplicity_readiness):
        """The next window reuses the same declaration; only the counts restart."""
        readiness = multiplicity_readiness({("E",): 2, ("N",): 1})
        readiness.mark(("E",))
        readiness.mark(("E",))
        readiness.mark(("N",))
        assert readiness.is_complete()

        readiness.reset()

        assert readiness.expected_total == 3
        assert readiness.missing() == frozenset({("E",), ("N",)})
        assert not readiness.is_complete()

    def test_reset_re_arms_the_same_declaration_for_the_next_window(self, multiplicity_readiness):
        """The property the fix exists for: a correct declaration closes once per
        window, whatever the interleaving of the consuming nodes, with no drift."""
        readiness = multiplicity_readiness({("E",): 2, ("N",): 1})
        verdicts = []

        for _ in range(5):
            verdicts.append(readiness.is_complete())
            readiness.mark(("E",))
            readiness.mark(("N",))
            readiness.mark(("E",))
            verdicts.append(readiness.is_complete())
            readiness.reset()

        assert verdicts[0::2] == [False] * 5
        assert verdicts[1::2] == [True] * 5

    def test_a_shared_consumer_is_absorbed_by_its_declared_multiplicity(
        self, multiplicity_readiness
    ):
        """Declaring the shared embedding twice is what keeps the window open.

        The combined-path case the declaration exists for: the embedding is consumed
        by the pre-process node and by an MTP pre-dispatch node, so a single mark is
        not yet the end of its backward.
        """
        readiness = multiplicity_readiness({("E",): 2, ("N",): 1})

        readiness.mark(("E",))
        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({("E",), ("N",)})

        readiness.mark(("E",))
        readiness.mark(("N",))
        assert readiness.is_complete()

    def test_expected_is_a_live_reference_not_a_copy(self, multiplicity_readiness):
        """``set_grad_multiplicity`` mutates the declaration through this property.

        Returning a copy would silently discard the schedule's declaration, and every
        shared parameter would then wait for a contribution that never arrives.
        """
        readiness = multiplicity_readiness({("E",): 1})

        # The same object every access, not a fresh copy of the declaration.
        assert readiness.expected is readiness.expected

        readiness.expected.update({("E",): 2})

        assert readiness.expected_total == 2
        readiness.mark(("E",))
        assert not readiness.is_complete()
        readiness.mark(("E",))
        assert readiness.is_complete()

        # A declaration raised on an open window takes effect immediately.
        readiness.reset()
        readiness.mark(("E",))
        assert not readiness.is_complete()
        readiness.mark(("E",))
        assert readiness.is_complete()

        readiness.expected.update({("E",): 3})
        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({("E",)})


class TestActiveMtpLayers:
    """``_active_mtp_layers`` decides whether THIS stage runs MTP at all.

    It is the input that makes the embedding's extra MTP consumer exist, so a wrong
    answer here is a wrong multiplicity: too high holds the reduce back, too low
    closes the window early. An unanswerable configuration must therefore raise
    rather than guess.
    """

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_a_model_without_mtp_layers_reports_zero(self, active_mtp_layers):
        """An absent ``mtp_num_layers`` is the ordinary no-MTP case."""
        module = types.SimpleNamespace(config=types.SimpleNamespace())

        assert active_mtp_layers(module) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_an_explicit_none_mtp_num_layers_reports_zero(self, active_mtp_layers):
        """``None`` is the config's "unset" spelling and must not be truthy-read."""
        module = types.SimpleNamespace(config=types.SimpleNamespace(mtp_num_layers=None))

        assert active_mtp_layers(module) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_mtp_num_layers_zero_reports_zero(self, active_mtp_layers):
        """An explicit zero is the same answer as unset."""
        module = types.SimpleNamespace(config=types.SimpleNamespace(mtp_num_layers=0))

        assert active_mtp_layers(module) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_an_mtp_layer_on_this_stage_reports_one(self, active_mtp_layers):
        """One MTP layer, and this stage owns it: the embedding has one more consumer."""
        module = types.SimpleNamespace(
            config=types.SimpleNamespace(mtp_num_layers=1), mtp_process=True
        )

        assert active_mtp_layers(module) == 1

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_an_mtp_layer_off_this_stage_reports_zero(self, active_mtp_layers):
        """MTP configured but owned by another pipeline stage adds no consumer here."""
        module = types.SimpleNamespace(
            config=types.SimpleNamespace(mtp_num_layers=1), mtp_process=False
        )

        assert active_mtp_layers(module) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_no_mtp_layers_never_consults_mtp_process(self, active_mtp_layers):
        """A model without MTP need not expose ``mtp_process`` at all.

        The zero-depth early return must precede the attribute probe, so a non-MTP
        model is unaffected by the guard added for the MTP case.
        """
        module = types.SimpleNamespace(config=types.SimpleNamespace(mtp_num_layers=0))

        assert not hasattr(module, "mtp_process")
        assert active_mtp_layers(module) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_a_model_that_forgets_mtp_process_is_an_error_not_a_guess(self, active_mtp_layers):
        """MTP is configured but this stage cannot say whether it owns it: refuse."""
        module = types.SimpleNamespace(config=types.SimpleNamespace(mtp_num_layers=1))

        assert not hasattr(module, "mtp_process")
        with pytest.raises(AssertionError, match="mtp_process"):
            active_mtp_layers(module)

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_mtp_deeper_than_one_layer_is_rejected(self, active_mtp_layers):
        """The per-parameter accounting only models one MTP layer, so deeper is loud.

        ``overlap_moe_expert_parallel_comm`` already constrains ``mtp_num_layers`` to
        at most 1, and the multiplicity of a deeper stack is not derived here;
        returning 1 would silently under-declare the embedding.
        """
        module = types.SimpleNamespace(
            config=types.SimpleNamespace(mtp_num_layers=2), mtp_process=True
        )

        with pytest.raises(AssertionError, match="got 2"):
            active_mtp_layers(module)


_EMBEDDING_FQNS = ("module.embedding.word_embeddings.weight",)
_NORM_FQNS = ("module.decoder.final_layernorm.weight",)


class _FakeFsdpParameter:
    """Stand-in for ``FsdpParameter``: the fields the multiplicity reads.

    ``sharded`` and ``unsharded`` are the two objects FSDP swaps into the module
    tree (``_set_module_parameter`` installs one or the other). ``sharded``
    defaults to ``None`` -- a fake-only "not installed" state; the real field is
    always a concrete ``nn.Parameter`` -- because most cases only exercise the
    unsharded half.
    """

    def __init__(self, fqns, unsharded, sharded=None):
        """Record this parameter's FQNs and the two objects it can be installed as."""
        self.fqns = fqns
        self.unsharded = unsharded
        self.sharded = sharded


class _FakeUnit:
    """Stand-in for ``FsdpModule``: only the trainable-parameter iteration."""

    def __init__(self, parameters):
        """Wrap ``parameters`` as the unit's trainable parameters."""
        self._parameters = tuple(parameters)

    def _trainable_fsdp_parameters(self):
        """Yield this unit's trainable parameters, as the real ``FsdpModule`` does."""
        return iter(self._parameters)


def _unit_with(embedding_weight, other_weight):
    """A unit owning one shared embedding and one ordinary parameter."""
    return _FakeUnit(
        [
            _FakeFsdpParameter(_EMBEDDING_FQNS, embedding_weight),
            _FakeFsdpParameter(_NORM_FQNS, other_weight),
        ]
    )


class TestUnitGradMultiplicity:
    """The per-parameter declaration the combined 1F1B path installs."""

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    @pytest.mark.parametrize("mtp_depth", [0, 1])
    @pytest.mark.parametrize("tied", [False, True])
    def test_non_embedding_parameters_always_contribute_once(
        self, unit_grad_multiplicity, mtp_depth, tied
    ):
        """REGRESSION GUARD: the MTP depth applies to the embedding only.

        An earlier revision started every parameter from ``1 + mtp_depth``, so an
        ordinary parameter declared two contributions and waited for one that no
        schedule node could produce. The extra consumers belong to the shared
        embedding alone, whatever the MTP depth and tie setting are.
        """
        embedding_weight = object()
        unit = _unit_with(embedding_weight, object())

        declaration = unit_grad_multiplicity(unit, mtp_depth, embedding_weight, tied)

        assert declaration[_NORM_FQNS] == 1

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    @pytest.mark.parametrize(
        ("mtp_depth", "tied", "expected_consumers"),
        [(0, False, 1), (1, False, 2), (0, True, 2), (1, True, 3)],
    )
    def test_the_embedding_contribution_matrix(
        self, unit_grad_multiplicity, mtp_depth, tied, expected_consumers
    ):
        """Every combination of the embedding's two extra consumers.

        The base contribution is the pre-process embedding lookup; the MTP
        pre-dispatch node adds one, and a tied output projection adds another by
        running its matmul against this very weight object.
        """
        embedding_weight = object()
        unit = _unit_with(embedding_weight, object())

        declaration = unit_grad_multiplicity(unit, mtp_depth, embedding_weight, tied)

        assert declaration[_EMBEDDING_FQNS] == expected_consumers
        assert declaration[_NORM_FQNS] == 1

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_the_declaration_is_keyed_by_fqns(self, unit_grad_multiplicity):
        """Keys are ``FsdpParameter.fqns`` tuples, the key space ``mark`` uses."""
        embedding_weight = object()
        unit = _unit_with(embedding_weight, object())

        declaration = unit_grad_multiplicity(unit, 1, embedding_weight, False)

        assert set(declaration) == {_EMBEDDING_FQNS, _NORM_FQNS}

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_a_tied_parameter_with_two_fqns_still_contributes_once(self, unit_grad_multiplicity):
        """REGRESSION GUARD: multiplicity is not ``len(fqns)``.

        A tied weight is one physical ``nn.Parameter`` registered under several
        FQNs, and its hooks are installed once, so it fires once per consuming
        schedule node. Deriving the count from the FQN list would double it, and the
        reduce would never be triggered.
        """
        tied_fqns = ("module.embedding.word_embeddings.weight", "module.output_layer.weight")
        unit = _FakeUnit([_FakeFsdpParameter(tied_fqns, object())])

        declaration = unit_grad_multiplicity(unit, 1, object(), True)

        assert declaration[tied_fqns] == 1
        # One key for the one physical parameter, not one key per FQN.
        assert len(declaration) == 1

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_the_embedding_is_matched_by_identity(self, unit_grad_multiplicity):
        """REGRESSION GUARD: identity decides, not FQNs and not equality.

        A tied output layer leaves ``output_layer.weight`` as ``None`` and receives
        the weight at forward time, so a name- or attribute-based test cannot
        recognise the shared embedding. Only the object that IS the embedding weight
        accrues the extra consumers.
        """
        embedding_weight = object()
        # Same FQN as the embedding, but a different object.
        same_fqns_other_object = _FakeFsdpParameter(_EMBEDDING_FQNS, object())
        # Different FQN, but the very same object.
        other_fqns_same_object = _FakeFsdpParameter(("module.other.weight",), embedding_weight)
        unit = _FakeUnit([same_fqns_other_object, other_fqns_same_object])

        declaration = unit_grad_multiplicity(unit, 1, embedding_weight, False)

        assert declaration[_EMBEDDING_FQNS] == 1
        assert declaration[("module.other.weight",)] == 2

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_the_sharded_half_of_the_swap_is_matched_too(self, unit_grad_multiplicity):
        """REGRESSION GUARD: the module tree normally holds ``sharded``, not ``unsharded``.

        ``parameter_group._set_module_parameter`` installs either
        ``FsdpParameter.sharded`` or ``FsdpParameter.unsharded`` and is the only
        writer of ``module._parameters``, so the weight read through the module tree
        is one of those two -- which one depends on whether the last switch was a
        reshard or an unshard. A revision that compared against ``unsharded`` alone
        never fired on hardware: the embedding was declared once while its hook fired
        twice (PreProcessNode plus the MTP pre-dispatch node), and the over-fire guard
        raised ``ValueError: ('module.embedding.word_embeddings.weight',)
        over-fired: 2 > 1``. This test pins the case the module tree actually
        presents.
        """
        embedding_weight = object()
        unit = _FakeUnit(
            [
                # Installed as *sharded*; ``unsharded`` is a different object.
                _FakeFsdpParameter(_EMBEDDING_FQNS, object(), sharded=embedding_weight),
                _FakeFsdpParameter(_NORM_FQNS, object(), sharded=object()),
            ]
        )

        declaration = unit_grad_multiplicity(unit, 1, embedding_weight, False)

        assert declaration[_EMBEDDING_FQNS] == 2
        assert declaration[_NORM_FQNS] == 1

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_matching_either_half_does_not_leak_to_a_neighbour(self, unit_grad_multiplicity):
        """Only the parameter that owns the object accrues the extra consumers.

        Widening the match to both halves must not turn it into a name-based test:
        the ordinary parameter is installed as the very same *kind* of object and
        must still be declared once.
        """
        embedding_weight = object()
        unit = _FakeUnit(
            [
                _FakeFsdpParameter(_EMBEDDING_FQNS, embedding_weight, sharded=object()),
                _FakeFsdpParameter(_NORM_FQNS, object(), sharded=object()),
            ]
        )

        declaration = unit_grad_multiplicity(unit, 1, embedding_weight, False)

        assert declaration[_EMBEDDING_FQNS] == 2
        assert declaration[_NORM_FQNS] == 1

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_an_embedding_owned_by_another_stage_leaves_every_parameter_at_one(
        self, unit_grad_multiplicity
    ):
        """A stage that does not own the embedding declares no extra consumer.

        On a pipeline stage without this chunk's embedding the root's embedding
        weight is not among the unit's parameters, so nothing may be matched and no
        parameter may be left waiting.
        """
        unit = _unit_with(object(), object())

        declaration = unit_grad_multiplicity(unit, 1, object(), True)

        assert set(declaration.values()) == {1}

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_a_missing_embedding_weight_leaves_every_parameter_at_one(self, unit_grad_multiplicity):
        """REGRESSION GUARD: ``None`` must match nothing.

        A model without an embedding on this stage resolves the weight to ``None``.
        Comparing it only through the identity tests let ``None is None`` succeed
        against a parameter whose ``sharded``/``unsharded`` is unset, which handed
        *every* parameter the extra consumers (``1 + mtp_depth + tied``) instead of
        leaving them at one. ``_unit_grad_multiplicity`` therefore rules ``None`` out
        explicitly.
        """
        unit = _unit_with(object(), object())

        declaration = unit_grad_multiplicity(unit, 1, None, True)

        assert set(declaration.values()) == {1}

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_a_unit_with_no_trainable_parameters_declares_nothing(self, unit_grad_multiplicity):
        """Nothing to wait for, which is the ``expected_total == 0`` branch."""
        declaration = unit_grad_multiplicity(_FakeUnit([]), 1, object(), True)

        assert declaration == {}
        assert MultiplicityReadiness(declaration).expected_total == 0
        assert MultiplicityReadiness(declaration).is_complete()

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_the_declaration_covers_exactly_the_trainable_parameters(self, unit_grad_multiplicity):
        """A dropped parameter would wait forever, so the key set must not shrink."""
        embedding_weight = object()
        unit = _unit_with(embedding_weight, object())

        declaration = unit_grad_multiplicity(unit, 1, embedding_weight, True)

        declared = {parameter.fqns for parameter in unit._trainable_fsdp_parameters()}
        assert set(declaration) == declared
        assert all(consumers >= 1 for consumers in declaration.values())


class TestCombinedWindowIntegration:
    """The declaration and the counting signal driven together.

    These tie the two halves of the fix: a declaration derived by
    ``_unit_grad_multiplicity`` is fed to the ``MultiplicityReadiness`` the
    ``FsdpModule`` counts against, and the outcome of one combined-1F1B window is
    observed in both the correct and the two wrong directions.
    """

    def _declared_readiness(self, unit_grad_multiplicity, mtp_depth, tied):
        """Build the declaration for one embedding plus one ordinary parameter."""
        embedding_weight = object()
        unit = _unit_with(embedding_weight, object())
        declaration = unit_grad_multiplicity(unit, mtp_depth, embedding_weight, tied)
        return declaration, MultiplicityReadiness(declaration)

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_the_declared_shared_embedding_closes_the_window(self, unit_grad_multiplicity):
        """The reported MTP scenario: the embedding fires twice, the norm once.

        With one MTP layer the shared embedding is consumed by the pre-process
        embedding lookup and by the MTP pre-dispatch node, while the decoder's final
        norm is consumed by a single node. Declaring that shape closes the window
        exactly once, at the end of the real fire sequence.
        """
        declaration, readiness = self._declared_readiness(unit_grad_multiplicity, 1, False)

        assert declaration[_EMBEDDING_FQNS] == 2
        assert declaration[_NORM_FQNS] == 1

        readiness.mark(_EMBEDDING_FQNS)
        assert not readiness.is_complete()

        # The MTP pre-dispatch node's contribution: the second and last one.
        readiness.mark(_EMBEDDING_FQNS)
        assert not readiness.is_complete()

        readiness.mark(_NORM_FQNS)
        assert readiness.is_complete()
        assert readiness.expected_total == 3

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_an_under_declared_embedding_over_fires_loudly(self, unit_grad_multiplicity):
        """The direction that used to drift silently: a spare contribution.

        If the schedule really consumed the embedding twice while only one
        contribution was declared, the surplus is refused at the offending mark
        instead of being charged to the next window.
        """
        _, readiness = self._declared_readiness(unit_grad_multiplicity, 0, False)

        readiness.mark(_EMBEDDING_FQNS)

        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(_EMBEDDING_FQNS)

    @pytest.mark.internal
    @pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, reason=_MEGATRON_STACK_SKIP_REASON)
    def test_an_over_declared_embedding_leaves_the_window_open(self, unit_grad_multiplicity):
        """The other wrong direction: a contribution that can never arrive.

        Declaring one consumer too many holds the reduce back, so the window must
        stay observably incomplete rather than close on the real fire sequence.
        """
        _, readiness = self._declared_readiness(unit_grad_multiplicity, 1, False)
        readiness.expected.update({_EMBEDDING_FQNS: 3})

        readiness.mark(_EMBEDDING_FQNS)
        readiness.mark(_EMBEDDING_FQNS)
        readiness.mark(_NORM_FQNS)

        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({_EMBEDDING_FQNS})


_GATED_CLASS_NAMES = (
    "TestActiveMtpLayers",
    "TestUnitGradMultiplicity",
    "TestCombinedWindowIntegration",
)

_IMPORT_HELPER_NAMES = ("_import_scheduler_module",)


def _methods_named(tree, class_name):
    """Return the ``test_*`` method nodes of the named class, or ``[]``."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
            ]
    return []


def _module_function(tree, function_name):
    """Return the named module-level function node, or ``None``."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    return None


def _import_error_swallowing_handlers(node):
    """Return the ``except ImportError`` handlers in ``node`` that skip."""
    return [
        ast.unparse(handler)
        for handler in ast.walk(node)
        if isinstance(handler, ast.ExceptHandler)
        and handler.type is not None
        and "ImportError" in ast.unparse(handler.type)
        and "skip" in ast.unparse(handler)
    ]


class TestSkipGateDiscipline:
    """Keep the host gate a precondition rather than a hidden import failure."""

    def test_the_gate_is_a_precondition_not_an_exception_handler(self):
        """The skip must be decided on the host, before any Megatron import.

        Checked structurally so it holds regardless of test order or host:

        * the predicate is answerable without touching Megatron, so a broken import
          graph cannot influence the skip decision;
        * the predicate agrees with the environmental fact it claims to test;
        * every test that needs the scheduler module is gated by a ``skipif`` marker
          evaluated before its body runs, and neither those bodies nor the shared
          import helper turns an ``ImportError`` into a skip.
        """
        import torch

        assert HOST_CAN_IMPORT_MEGATRON_STACK == hasattr(torch, _REQUIRED_TORCH_ATTR)

        tree = ast.parse(Path(__file__).read_text())

        predicate = _module_function(tree, "_host_can_import_megatron_stack")
        assert predicate is not None, "_host_can_import_megatron_stack is missing"
        imported = [
            ast.unparse(node)
            for node in ast.walk(predicate)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert imported, "the host precondition must inspect the host, not nothing."
        assert all("megatron" not in statement.lower() for statement in imported), (
            "the host precondition must be answerable without importing Megatron, "
            "otherwise a broken import graph could turn itself into a skip. "
            f"Found: {imported}"
        )

        ungated = []
        swallowing = []
        for class_name in _GATED_CLASS_NAMES:
            methods = _methods_named(tree, class_name)
            assert methods, f"{class_name} has no tests: the gate guards nothing."
            for method in methods:
                decorators = [ast.unparse(decorator) for decorator in method.decorator_list]
                if not any(
                    "skipif" in decorator and "HOST_CAN_IMPORT_MEGATRON_STACK" in decorator
                    for decorator in decorators
                ):
                    ungated.append(f"{class_name}.{method.name}")
                swallowing.extend(
                    f"{class_name}.{method.name}: {handler}"
                    for handler in _import_error_swallowing_handlers(method)
                )

        for helper_name in _IMPORT_HELPER_NAMES:
            helper = _module_function(tree, helper_name)
            assert helper is not None, f"{helper_name} is missing"
            swallowing.extend(
                f"{helper_name}: {handler}" for handler in _import_error_swallowing_handlers(helper)
            )

        assert not ungated, (
            "every test that imports the scheduler module must be gated by "
            "@pytest.mark.skipif(not HOST_CAN_IMPORT_MEGATRON_STACK, ...), so the skip "
            f"is decided before the import. Ungated: {ungated}"
        )
        assert not swallowing, (
            "an ImportError must never become a skip: that is exactly what would let a "
            f"real import failure hide behind the environment. Found: {swallowing}"
        )
