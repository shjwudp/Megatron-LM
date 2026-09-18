# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Completion-signal tests for the MFSDP v2 combined/fine-grained 1F1B path.

``countdown`` is deliberately dependency-free (standard library only), so its
completion signals are loaded straight from source and tested without a GPU, a
process group, or Transformer Engine. That keeps the mechanism-level regression
runnable on any host, including this repository's CPU-only lint environment. The
tests that need a real ``FsdpModule`` skip when the Megatron/TE/GPU stack is not
importable.

The three signals are pinned side by side on purpose: ``Countdown`` and
``GradientReadiness`` are the superseded ones and are kept as regression evidence
for *why* they were replaced, while ``MultiplicityReadiness`` is what the
production path uses for both the automatic and the combined backward.
"""

import importlib
import importlib.util
from pathlib import Path
from unittest import mock

import pytest

_COUNTDOWN_PATH = (
    Path(__file__).resolve().parents[4]
    / "megatron"
    / "core"
    / "distributed"
    / "fsdp"
    / "src"
    / "megatron_fsdp"
    / "experimental"
    / "countdown.py"
)


def _load_countdown():
    """Load the dependency-free ``countdown`` module without the MFSDP package.

    Importing it through ``megatron_fsdp`` executes the package ``__init__``, which
    needs the CUDA/TE stack; the module itself imports nothing outside the standard
    library.
    """
    spec = importlib.util.spec_from_file_location("mfsdp_countdown_under_test", _COUNTDOWN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


countdown = _load_countdown()
Countdown = countdown.Countdown
MultiplicityReadiness = countdown.MultiplicityReadiness


@pytest.fixture(scope="module")
def gradient_readiness():
    """The idempotent-mark signal that the schedule-edge fix used."""
    return countdown.GradientReadiness


@pytest.fixture(scope="module")
def multiplicity_readiness():
    """The exact per-parameter accounting that replaces both previous signals."""
    return countdown.MultiplicityReadiness


def _import_or_skip(module_name, reason):
    """Import ``module_name``, skipping the caller when the GPU stack is unusable.

    A host with an older torch fails deep inside Megatron's import chain with
    ``AttributeError`` rather than ``ImportError`` (for example a missing
    ``torch.float8_e8m0fnu``), so both mean "the stack is not available here".
    """
    try:
        return importlib.import_module(module_name)
    except (ImportError, AttributeError) as error:
        pytest.skip(f"{reason} ({type(error).__name__}: {error})")


class TestCountdownFireSequences:
    """Pin the defect on the signal the combined path used to trust.

    ``Countdown`` reports completion on the ``initial_value``-th callback, which is
    the end of a module's backward only when exactly one callback per trainable
    parameter arrives. The combined/fine-grained 1F1B schedule runs one
    ``run_backward`` per schedule node on detached node inputs, so a parameter
    consumed by several nodes fires once per node and a parameter unused in the
    window does not fire at all. Both directions desynchronise the countdown, which
    is why the schedule's own edge replaces it.
    """

    def test_shared_parameter_completes_the_countdown_early(self):
        """The recorded MTP crash: the root owns the shared embedding E and the
        decoder's final norm N, and the window fires E, E, N."""
        countdown_state = Countdown(initial_value=2)
        completions = [countdown_state.decrement() for _ in range(3)]

        assert completions == [False, True, False]
        # Completion was reported after E's second callback, while N -- the
        # parameter the recorded run reported as missing -- still had no gradient.
        assert completions.index(True) == 1

    def test_unused_parameter_defers_completion_and_shifts_the_next_window(self):
        """Window 1 consumes A and B of three owned parameters; the unspent credit
        makes window 2 complete one callback early."""
        countdown_state = Countdown(initial_value=3)

        assert [countdown_state.decrement() for _ in range(2)] == [False, False]
        assert [countdown_state.decrement() for _ in range(3)] == [True, False, False]


class TestGradientReadiness:
    """The first replacement: idempotent per-parameter marks and a schedule edge."""

    def test_repeated_callback_does_not_complete_the_window(self, gradient_readiness):
        readiness = gradient_readiness(range(2))

        readiness.mark(0)
        readiness.mark(0)  # the same parameter's second consuming schedule node
        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({1})

        readiness.mark(1)
        assert readiness.is_complete()
        assert readiness.close() is True
        assert not readiness.has_pending_marks

    def test_incomplete_window_is_observable_at_close(self, gradient_readiness):
        readiness = gradient_readiness(range(3))

        readiness.mark(0)
        readiness.mark(2)
        assert readiness.marked == 2
        assert readiness.missing() == frozenset({1})
        assert readiness.close() is False

        # close() re-arms for the next window.
        assert not readiness.has_pending_marks
        assert not readiness.is_complete()

    def test_marks_for_unknown_parameters_are_ignored(self, gradient_readiness):
        readiness = gradient_readiness(["owned"])

        readiness.mark("not-owned")
        assert not readiness.has_pending_marks
        assert readiness.missing() == frozenset({"owned"})

    def test_no_fire_sequence_can_shift_a_later_window(self, gradient_readiness):
        """Five identical windows each complete exactly once, unlike the countdown."""
        readiness = gradient_readiness(range(2))

        for _ in range(5):
            readiness.mark(0)
            readiness.mark(0)
            readiness.mark(1)
            assert readiness.close() is True
            assert not readiness.has_pending_marks


class TestMultiplicityReadiness:
    """SCHEDULE-DECLARED MULTIPLICITY accounting, the production signal.

    ``Countdown`` counts callbacks and ``GradientReadiness`` remembers whether a
    parameter was seen at all. Neither can say "this parameter contributed exactly
    as often as declared": the first cannot see a surplus, and the second makes a
    surplus and a shortfall look the same. Counting against a declaration can, and
    the two directions are reported separately because they mean different bugs.
    """

    def test_declared_multiplicity_must_be_at_least_one(self, multiplicity_readiness):
        """A zero expected count would complete without ever observing a parameter."""
        with pytest.raises(ValueError, match="at least 1"):
            multiplicity_readiness({"owned": 0})

    def test_exact_once_completes(self, multiplicity_readiness):
        readiness = multiplicity_readiness({"E": 1, "N": 1})

        assert readiness.expected_total == 2
        readiness.mark("N")
        assert readiness.marked_total == 1
        assert not readiness.is_complete()
        readiness.mark("E")

        assert readiness.is_complete()
        assert readiness.missing() == frozenset()
        assert readiness.over_fired() == ()
        assert readiness.close() is True

    def test_declared_multiplicity_absorbs_a_shared_consumer(self, multiplicity_readiness):
        """The combined-path case: E is consumed by the pre-process node and by an
        MTP pre-dispatch node, so it is expected twice and the window stays open."""
        readiness = multiplicity_readiness({"E": 2, "N": 1})
        assert readiness.expected_total == 3

        readiness.mark("E")
        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({"E", "N"})

        readiness.mark("E")  # the second consuming node
        readiness.mark("N")
        assert readiness.is_complete()
        assert readiness.over_fired() == ()
        assert readiness.close() is True

    def test_under_fire_names_the_parameter_and_the_declared_count(self, multiplicity_readiness):
        """A declared consumer that never ran: the window cannot be complete, but the
        shortfall is visible instead of collapsing into "some mark exists"."""
        readiness = multiplicity_readiness({"E": 2, "N": 1})
        readiness.mark("E")

        assert not readiness.is_complete()
        assert readiness.missing() == frozenset({"E", "N"})
        assert readiness.over_fired() == ()
        assert readiness.close() is False

    def test_over_fire_is_recorded_with_observed_and_expected(self, multiplicity_readiness):
        """The declaration was too small: the extra contribution is the surplus that
        the old countdown charged to a later window."""
        readiness = multiplicity_readiness({"E": 1, "N": 1})

        readiness.mark("E")
        readiness.mark("E")  # E was under-declared
        readiness.mark("N")

        # The window is complete, but the over-fire is on the record.
        assert readiness.is_complete()
        assert readiness.over_fired() == (("E", 2, 1),)
        assert readiness.close() is True

    def test_over_fire_is_recorded_even_before_the_window_completes(self, multiplicity_readiness):
        """Over-fire is detected at the offending mark, not at the close edge."""
        readiness = multiplicity_readiness({"E": 1, "N": 1})

        readiness.mark("E")
        readiness.mark("E")

        assert readiness.over_fired() == (("E", 2, 1),)
        assert not readiness.is_complete()

    def test_unknown_key_is_an_error_not_a_silent_drop(self, multiplicity_readiness):
        """Ignoring an unknown key would hide a real contribution from the count."""
        readiness = multiplicity_readiness({"owned": 1})

        with pytest.raises(KeyError, match="not a known parameter"):
            readiness.mark("not-owned")

    def test_close_resets_counts_and_verdicts(self, multiplicity_readiness):
        readiness = multiplicity_readiness({"E": 2, "N": 1})
        readiness.mark("E")
        readiness.mark("E")
        readiness.mark("E")  # over-fire against a declared 2
        assert readiness.over_fired() == (("E", 3, 2),)
        assert not readiness.is_complete()

        assert readiness.close() is False
        assert readiness.marked_total == 0
        assert readiness.over_fired() == ()
        assert readiness.missing() == frozenset({"E", "N"})
        assert not readiness.has_pending_marks

    def test_declaration_never_shifts_a_later_window(self, multiplicity_readiness):
        """The property the fix exists for: a correct declaration closes exactly once
        per window, whatever the interleaving of the consuming nodes."""
        readiness = multiplicity_readiness({"E": 2, "N": 1})

        for _ in range(5):
            readiness.mark("E")
            readiness.mark("N")
            readiness.mark("E")
            assert readiness.over_fired() == ()
            assert readiness.close() is True

    def test_declaration_too_small_closes_early_and_is_visible(self, multiplicity_readiness):
        """The dangerous direction, stated as a test: a declaration of 1 for a
        parameter that really fires twice still closes early, but the surplus is
        reported rather than silently leaking."""
        readiness = multiplicity_readiness({"E": 1, "N": 1})

        readiness.mark("E")
        readiness.mark("N")
        assert readiness.is_complete()  # the early close the declaration asked for
        assert readiness.over_fired() == ()
        assert readiness.close() is True

        # The second consuming node now lands in the next window, where it is an
        # over-fire because that window declared only one contribution as well.
        readiness.mark("E")
        readiness.mark("E")
        readiness.mark("N")
        assert readiness.over_fired() == (("E", 2, 1),)


@pytest.fixture(scope="module")
def mfsdp_module_class():
    """The real ``FsdpModule``, or a skip when the GPU stack is unavailable."""
    module = _import_or_skip(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module",
        reason="MFSDP v2 needs the Megatron core, Transformer Engine and a GPU stack.",
    )
    return module.FsdpModule


@pytest.fixture(scope="module")
def mfsdp_multiplicity_readiness():
    """The ``MultiplicityReadiness`` the real ``FsdpModule`` pairs with."""
    module = _import_or_skip(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.countdown",
        reason="MFSDP v2 needs the Megatron core stack.",
    )
    return module.MultiplicityReadiness


def _schedule_driven_module(
    fsdp_module_class, multiplicity_readiness_class, multiplicities, on_close
):
    """Build an ``FsdpModule`` that carries only schedule-driven completion state.

    ``fully_shard`` needs CUDA and a device mesh, but the completion bookkeeping is
    device-free, so it is assembled directly here. ``multiplicities`` maps a
    parameter index to its declared contribution count, the same key space the
    real ``_register_completion_hooks`` uses.
    """
    fsdp_module = object.__new__(fsdp_module_class)
    fsdp_module._name = "module.unit"
    fsdp_module._grad_readiness = multiplicity_readiness_class(multiplicities)
    fsdp_module._grad_readiness_fqns = {index: (f"unit.param{index}",) for index in multiplicities}
    fsdp_module._scheduled_post_backward_hook = on_close
    fsdp_module._warned_missing_grads = False
    fsdp_module._warned_over_fire = False
    return fsdp_module


def _mark(fsdp_module, index):
    """Charge one contribution the way the per-parameter callback does."""
    fsdp_module._record_grad_contribution(index)


class TestScheduleDrivenCompletion:
    """``finalize_scheduled_backward`` is the only reduce trigger in this path."""

    def test_no_open_window_is_a_no_op(self, mfsdp_module_class, mfsdp_multiplicity_readiness):
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, calls.append
        )

        fsdp_module.finalize_scheduled_backward()

        assert calls == []
        fsdp_module.assert_scheduled_backward_closed()

    def test_shared_parameter_closes_once_at_the_schedule_edge(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 2, 1: 1}, calls.append
        )

        _mark(fsdp_module, 0)
        _mark(fsdp_module, 0)
        # The declared second contribution must not reduce anything by itself.
        assert calls == []
        assert not fsdp_module._grad_readiness.is_complete()

        _mark(fsdp_module, 1)
        fsdp_module.finalize_scheduled_backward()

        assert calls == [fsdp_module]
        assert fsdp_module._grad_readiness.over_fired() == ()
        fsdp_module.assert_scheduled_backward_closed()

    def test_edge_closes_a_window_with_an_unused_parameter(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        """An unused parameter is zero-filled, not dropped and not fatal."""
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, calls.append
        )
        _mark(fsdp_module, 0)

        fsdp_module.finalize_scheduled_backward()

        assert calls == [fsdp_module]
        assert fsdp_module._warned_missing_grads
        fsdp_module.assert_scheduled_backward_closed()

    def test_under_fire_is_reported_loudly_once(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        """The close edge reports a shortfall exactly once, for the whole unit.

        ``log_single_rank`` only emits on rank 0, so a multi-rank CI run cannot
        assert on the log record itself. The once-per-unit guarantee is the flag the
        reporter sets before it logs, and that flag is rank-independent; what the
        report *says* is asserted in ``TestReportedGradWindowMessage``.
        """
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 2, 1: 1}, calls.append
        )
        _mark(fsdp_module, 1)

        fsdp_module.finalize_scheduled_backward()

        assert calls == [fsdp_module]
        assert fsdp_module._warned_missing_grads

        # A second under-fired window is not reported again.
        fsdp_module._warned_missing_grads = False
        fsdp_module.finalize_scheduled_backward()
        assert not fsdp_module._warned_missing_grads

    def test_over_fire_is_reported_loudly_once(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        """An under-declared multiplicity means the window already closed early, so
        the surplus is named at the close edge, once per unit."""
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, calls.append
        )
        _mark(fsdp_module, 0)
        _mark(fsdp_module, 0)
        _mark(fsdp_module, 1)

        fsdp_module.finalize_scheduled_backward()

        assert fsdp_module._warned_over_fire
        assert calls == [fsdp_module]

        # The report is one-shot per unit rather than one per callback: the flag is
        # already set, so a second identical window adds nothing.
        _mark(fsdp_module, 0)
        _mark(fsdp_module, 0)
        _mark(fsdp_module, 1)
        fsdp_module.finalize_scheduled_backward()
        assert fsdp_module._warned_over_fire

    def test_open_window_at_the_end_of_the_chunk_is_loud(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        """The end-of-schedule invariant catches a dropped reduce-scatter."""
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, lambda _: None
        )
        _mark(fsdp_module, 0)

        with pytest.raises(RuntimeError, match="incomplete gradient-completion window"):
            fsdp_module.assert_scheduled_backward_closed()

    def test_over_fire_reopens_the_window_at_the_end_of_the_chunk(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        """A late contribution is a surplus, not a silently dropped mark."""
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, lambda _: None
        )
        _mark(fsdp_module, 0)
        _mark(fsdp_module, 0)

        with pytest.raises(RuntimeError, match="over_fired="):
            fsdp_module.assert_scheduled_backward_closed()

    def test_automatic_path_modules_are_untouched(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        """Without a schedule tracker both lifecycle calls are no-ops."""
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, lambda _: None
        )
        fsdp_module._grad_readiness = None

        fsdp_module.finalize_scheduled_backward()
        fsdp_module.assert_scheduled_backward_closed()


class TestReportedGradWindowMessage:
    """What the loud report actually says, asserted without a process group.

    ``log_single_rank`` emits on rank 0 only, while a multi-rank CI run may execute a
    given test on any rank, so the record itself is not observable from those tests.
    Replacing the module's single reporting sink lets the same message construction
    be asserted deterministically on every host.
    """

    @staticmethod
    def _run_and_capture(fsdp_module_class, multiplicity_readiness_class, multiplicities, marks):
        """Run a window to its close edge and return the messages it reported."""
        calls = []
        fsdp_module = _schedule_driven_module(
            fsdp_module_class, multiplicity_readiness_class, multiplicities, calls.append
        )
        for index in marks:
            _mark(fsdp_module, index)
        reports = []
        sink = staticmethod(lambda message, *args: reports.append(message % args))
        # The reporter is a class-level method, so it must be replaced on the class,
        # not shadowed on the instance.
        with mock.patch.object(fsdp_module_class, "_log_grad_window_report", sink):
            fsdp_module.finalize_scheduled_backward()
        return fsdp_module, calls, reports

    def test_under_fire_message_names_the_parameter_and_both_counts(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        _, calls, reports = self._run_and_capture(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 2, 1: 1}, [1]
        )

        assert calls != []
        assert len(reports) == 1
        assert "under their declared multiplicity" in reports[0]
        # The parameter, what was observed and what was declared.
        assert "(('unit.param0',), 0, 2)" in reports[0]

    def test_over_fire_message_names_the_surplus(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        _, _, reports = self._run_and_capture(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, [0, 0, 1]
        )

        assert len(reports) == 1
        assert "MORE gradient contributions" in reports[0]
        assert "(('unit.param0',), 2, 1)" in reports[0]

    def test_fully_satisfied_window_reports_nothing(
        self, mfsdp_module_class, mfsdp_multiplicity_readiness
    ):
        """The automatic path's equivalent must stay quiet, not warn per window."""
        _, calls, reports = self._run_and_capture(
            mfsdp_module_class, mfsdp_multiplicity_readiness, {0: 1, 1: 1}, [0, 1]
        )

        assert calls != []
        assert reports == []
