# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Completion-signal tests for the MFSDP v2 combined/fine-grained 1F1B path.

``countdown`` is deliberately dependency-free (standard library only), so its two
completion signals are loaded straight from source and tested without a GPU, a
process group, or Transformer Engine. That keeps the mechanism-level regression
runnable on any host, including this repository's CPU-only lint environment. The
tests that need a real ``FsdpModule`` skip when the Megatron/TE/GPU stack is not
importable.
"""

import importlib
import importlib.util
from pathlib import Path

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


@pytest.fixture(scope="module")
def gradient_readiness():
    """The completion signal that replaces the countdown in this path."""
    return countdown.GradientReadiness


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
    """The replacement: idempotent per-parameter marks and a schedule-declared close."""

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


@pytest.fixture(scope="module")
def mfsdp_module_class():
    """The real ``FsdpModule``, or a skip when the GPU stack is unavailable."""
    module = _import_or_skip(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module",
        reason="MFSDP v2 needs the Megatron core, Transformer Engine and a GPU stack.",
    )
    return module.FsdpModule


@pytest.fixture(scope="module")
def mfsdp_gradient_readiness():
    """The ``GradientReadiness`` the real ``FsdpModule`` pairs with."""
    module = _import_or_skip(
        "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.countdown",
        reason="MFSDP v2 needs the Megatron core stack.",
    )
    return module.GradientReadiness


def _schedule_driven_module(fsdp_module_class, gradient_readiness_class, parameter_count, on_close):
    """Build an ``FsdpModule`` that carries only schedule-driven completion state.

    ``fully_shard`` needs CUDA and a device mesh, but the completion bookkeeping is
    device-free, so it is assembled directly here.
    """
    fsdp_module = object.__new__(fsdp_module_class)
    fsdp_module._name = "module.unit"
    fsdp_module._grad_readiness = gradient_readiness_class(range(parameter_count))
    fsdp_module._grad_readiness_fqns = {
        index: (f"unit.param{index}",) for index in range(parameter_count)
    }
    fsdp_module._scheduled_post_backward_hook = on_close
    fsdp_module._warned_missing_grads = False
    return fsdp_module


class TestScheduleDrivenCompletion:
    """``finalize_scheduled_backward`` is the only reduce trigger in this path."""

    def test_no_open_window_is_a_no_op(self, mfsdp_module_class, mfsdp_gradient_readiness):
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_gradient_readiness, 2, calls.append
        )

        fsdp_module.finalize_scheduled_backward()

        assert calls == []
        fsdp_module.assert_scheduled_backward_closed()

    def test_shared_parameter_closes_once_at_the_schedule_edge(
        self, mfsdp_module_class, mfsdp_gradient_readiness
    ):
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_gradient_readiness, 2, calls.append
        )

        fsdp_module._grad_readiness.mark(0)
        fsdp_module._grad_readiness.mark(0)
        # The over-fired parameter must not reduce anything by itself.
        assert calls == []
        assert not fsdp_module._grad_readiness.is_complete()

        fsdp_module._grad_readiness.mark(1)
        fsdp_module.finalize_scheduled_backward()

        assert calls == [fsdp_module]
        fsdp_module.assert_scheduled_backward_closed()

    def test_edge_closes_a_window_with_an_unused_parameter(
        self, mfsdp_module_class, mfsdp_gradient_readiness
    ):
        """An unused parameter is zero-filled, not dropped and not fatal."""
        calls = []
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_gradient_readiness, 2, calls.append
        )
        fsdp_module._grad_readiness.mark(0)

        fsdp_module.finalize_scheduled_backward()

        assert calls == [fsdp_module]
        assert fsdp_module._warned_missing_grads
        fsdp_module.assert_scheduled_backward_closed()

    def test_open_window_at_the_end_of_the_chunk_is_loud(
        self, mfsdp_module_class, mfsdp_gradient_readiness
    ):
        """The end-of-schedule invariant catches a dropped reduce-scatter."""
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_gradient_readiness, 2, lambda _: None
        )
        fsdp_module._grad_readiness.mark(0)

        with pytest.raises(RuntimeError, match="incomplete gradient-completion window"):
            fsdp_module.assert_scheduled_backward_closed()

    def test_automatic_path_modules_are_untouched(
        self, mfsdp_module_class, mfsdp_gradient_readiness
    ):
        """Without a schedule tracker both lifecycle calls are no-ops."""
        fsdp_module = _schedule_driven_module(
            mfsdp_module_class, mfsdp_gradient_readiness, 2, lambda _: None
        )
        fsdp_module._grad_readiness = None

        fsdp_module.finalize_scheduled_backward()
        fsdp_module.assert_scheduled_backward_closed()
