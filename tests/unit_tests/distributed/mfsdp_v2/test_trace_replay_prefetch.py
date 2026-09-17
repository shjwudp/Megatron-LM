# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the trace-and-replay scheduler's plan-level prefetch sets.

These drive :meth:`TraceAndReplayScheduler._build_plan` directly with stub modules,
so they need neither a GPU nor an FSDP context. They pin the properties the
multi-module prefetch relies on: the default stays a single successor, a budget
extends the lookahead by parameter elements, targets are distinct and never the
module being waited on, and a target whose own reshard falls before its demand
unshard is dropped because that reshard would release what the prefetch gathered.
"""

from types import SimpleNamespace

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.schedule import (
    OpKind,
    TraceAndReplayScheduler,
    TraceEvent,
)

ROWWISE = "rowwise"
COLWISE = "colwise"


class _Module:
    """Stand-in for FsdpModule: ``_build_plan`` needs only identity and size."""

    def __init__(self, name: str, elements: int = 100) -> None:
        self.name = name
        self.num_parameter_elements = elements

    def __repr__(self) -> str:
        return self.name


def _build(spec, budget):
    """Compile ``spec`` into a plan. ``spec`` is a list of (kind, module, orientation)."""
    events = [
        TraceEvent(op_index=index, kind=kind, module=module, orientation=orientation)
        for index, (kind, module, orientation) in enumerate(spec)
    ]
    scheduler = TraceAndReplayScheduler(SimpleNamespace(), prefetch_budget=budget)
    scheduler._events = events
    scheduler._build_plan()
    return scheduler._plan


def _targets(plan, wait_index, events_spec_len=None):
    """The prefetch targets recorded on the WAIT_UNSHARD op at ``wait_index``."""
    prefetch = plan[wait_index].prefetch_after
    return () if not prefetch else prefetch


def test_default_budget_keeps_a_single_successor():
    """A ``None`` budget must behave exactly as before: one module of lookahead."""
    a, b, c = _Module("A"), _Module("B"), _Module("C")
    spec = [
        (OpKind.ISSUE_UNSHARD, a, ROWWISE),
        (OpKind.WAIT_UNSHARD, a, None),
        (OpKind.ISSUE_UNSHARD, b, ROWWISE),
        (OpKind.WAIT_UNSHARD, b, None),
        (OpKind.ISSUE_UNSHARD, c, ROWWISE),
        (OpKind.WAIT_UNSHARD, c, None),
    ]
    plan = _build(spec, budget=None)

    assert _targets(plan, 1) == ((b, ROWWISE),)
    assert _targets(plan, 3) == ((c, ROWWISE),)
    # Nothing is unsharded after C, so there is nothing to prefetch.
    assert _targets(plan, 5) == ()


def test_default_budget_skips_a_resharded_successor_to_the_next_valid_one():
    """``None`` keeps depth one, and the reshard cutoff still applies.

    The historical walk prefetched the next distinct module unconditionally, so
    here it would have prefetched ``B`` and let ``B``'s own reshard release that
    gather. With the cutoff, ``B`` is skipped and the next module still valid at
    its demand unshard (``C``) becomes the single successor instead. Depth stays
    one: the walk does not continue on to ``D``.
    """
    a, b, c, d = _Module("A"), _Module("B"), _Module("C"), _Module("D")
    spec = [
        (OpKind.ISSUE_UNSHARD, a, ROWWISE),
        (OpKind.WAIT_UNSHARD, a, None),
        (OpKind.RESHARD, b, None),  # would release anything gathered for B
        (OpKind.ISSUE_UNSHARD, b, ROWWISE),
        (OpKind.WAIT_UNSHARD, b, None),
        (OpKind.ISSUE_UNSHARD, c, ROWWISE),
        (OpKind.WAIT_UNSHARD, c, None),
        (OpKind.ISSUE_UNSHARD, d, ROWWISE),
    ]
    plan = _build(spec, budget=None)

    assert _targets(plan, 1) == ((c, ROWWISE),)
    assert len(_targets(plan, 1)) == 1, "a None budget must stay at depth one"
    assert _targets(plan, 6) == ((d, ROWWISE),)


def test_budget_extends_the_lookahead_by_parameter_elements():
    """The walk continues until the accumulated elements reach the budget."""
    mods = [_Module(name) for name in ("A", "B", "C", "D")]  # 100 elements each
    spec = []
    for module in mods:
        spec.append((OpKind.ISSUE_UNSHARD, module, ROWWISE))
        spec.append((OpKind.WAIT_UNSHARD, module, None))

    # Budget of 150 elements: B fits (100), C crosses it (200) and is included.
    plan = _build(spec, budget=150)
    assert _targets(plan, 1) == ((mods[1], ROWWISE), (mods[2], ROWWISE))

    # A budget large enough to reach the end prefetches every remaining module.
    plan = _build(spec, budget=10_000)
    assert _targets(plan, 1) == (
        (mods[1], ROWWISE),
        (mods[2], ROWWISE),
        (mods[3], ROWWISE),
    )


def test_targets_are_distinct_and_never_the_waited_module():
    """One entry per module, and never the module whose wait is being served."""
    a, b = _Module("A"), _Module("B")
    spec = [
        (OpKind.ISSUE_UNSHARD, a, ROWWISE),
        (OpKind.WAIT_UNSHARD, a, None),
        # B is unsharded twice (forward then backward) before A is touched again.
        (OpKind.ISSUE_UNSHARD, b, ROWWISE),
        (OpKind.WAIT_UNSHARD, b, None),
        (OpKind.ISSUE_UNSHARD, b, COLWISE),
        (OpKind.WAIT_UNSHARD, b, None),
        (OpKind.ISSUE_UNSHARD, a, COLWISE),
    ]
    plan = _build(spec, budget=10_000)

    targets = [module for module, _ in _targets(plan, 1)]
    assert a not in targets, "the waited-on module must not be prefetched"
    assert targets.count(b) == 1, "each lookahead module appears at most once"


def test_target_resharded_before_its_unshard_is_dropped():
    """A reshard between here and the demand unshard makes the prefetch useless."""
    a, b = _Module("A"), _Module("B")
    spec = [
        (OpKind.ISSUE_UNSHARD, a, ROWWISE),
        (OpKind.WAIT_UNSHARD, a, None),
        (OpKind.RESHARD, b, None),  # gathers issued now would be released here
        (OpKind.ISSUE_UNSHARD, b, ROWWISE),
    ]
    plan = _build(spec, budget=10_000)
    assert _targets(plan, 1) == ()


def test_target_unsharded_before_its_reshard_is_kept():
    """The reshard only matters when it falls *between* the wait and the unshard."""
    a, b = _Module("A"), _Module("B")
    spec = [
        (OpKind.ISSUE_UNSHARD, a, ROWWISE),
        (OpKind.WAIT_UNSHARD, a, None),
        (OpKind.ISSUE_UNSHARD, b, ROWWISE),  # served by the prefetch
        (OpKind.RESHARD, b, None),
    ]
    plan = _build(spec, budget=10_000)
    assert _targets(plan, 1) == ((b, ROWWISE),)


def test_prefetch_carries_the_target_windows_orientation():
    """A target is prefetched in the orientation its own materialization will use."""
    a, b = _Module("A"), _Module("B")
    spec = [
        (OpKind.ISSUE_UNSHARD, a, ROWWISE),
        (OpKind.WAIT_UNSHARD, a, None),
        (OpKind.ISSUE_UNSHARD, b, COLWISE),
    ]
    plan = _build(spec, budget=None)
    assert _targets(plan, 1) == ((b, COLWISE),)


def test_no_prefetch_is_issued_while_tracing():
    """With no compiled plan there is nothing to prefetch, so the wait stays inert."""

    class _Stub:
        def __init__(self) -> None:
            self.waited = False

        def wait_unshard(self) -> None:
            self.waited = True

    module = _Stub()
    scheduler = TraceAndReplayScheduler(SimpleNamespace(), prefetch_budget=10_000)
    issued = []
    scheduler._prefetch = lambda target, orientation: issued.append((target, orientation))

    scheduler.wait_unshard(module)

    assert module.waited, "the wait itself must still happen"
    assert issued == [], "tracing must not issue prefetches"
