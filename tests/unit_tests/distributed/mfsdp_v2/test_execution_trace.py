# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure trace semantics; no CUDA or process group is needed by these cases."""

from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_trace import (
    ExecutionStep,
    ExecutionTrace,
)


@pytest.mark.parametrize('combined', [False, True], ids=['vpp', 'combined_1f1b'])
def test_interleaved_steps_replay_as_one_order(combined):
    """Revisited chunks and fine-grained forward/backward nodes share one cursor."""
    modules = [nn.Module(), nn.Module()]
    steps = [
        ExecutionStep(0, 0, 'forward', 'attention'),
        ExecutionStep(0, 1, 'forward', 'attention'),
        ExecutionStep(1, 0, 'forward', 'attention'),
        ExecutionStep(1, 0, 'backward', 'mlp'),
        ExecutionStep(0, 2, 'forward', 'attention'),
        ExecutionStep(0, 0, 'backward', 'mlp'),
        ExecutionStep(0, 0, 'weight_grad', 'mlp'),
    ]
    if combined:
        steps.insert(4, ExecutionStep(0, 1, 'forward', 'mlp'))
        steps.insert(5, ExecutionStep(1, 0, 'backward', 'attention'))
    trace = ExecutionTrace()
    for iteration in range(3):
        with trace.iteration(key=('train', combined)):
            assert trace.recording is (iteration == 0)
            for step in steps:
                with trace.step(step):
                    trace.observe(modules[step.chunk], 'unshard', phase=step.phase)
                    trace.observe(modules[step.chunk], 'reshard')
    assert [event.step for event in trace.events[::2]] == steps
    # Mode changes start a fresh trace; no assumption that backward will occur.
    with trace.iteration(key='validation'):
        trace.observe(modules[0], 'unshard', phase='forward')
        trace.observe(modules[0], 'reshard')
    assert len(trace.events) == 2


def test_lookahead_respects_releases_across_phases():
    """A release before a future demand forbids prefetching that module now."""
    first, released, ready = nn.Module(), nn.Module(), nn.Module()
    trace = ExecutionTrace()
    events = [
        (first, 'unshard', 'forward'),
        (released, 'reshard', None),
        (ready, 'unshard', 'backward'),
        (released, 'unshard', 'backward'),
        (ready, 'unshard', 'backward'),
        (first, 'reshard', None),
        (first, 'unshard', 'backward'),
    ]
    with trace.iteration():
        for module, operation, phase in events:
            trace.observe(module, operation, phase=phase)
            assert list(trace.prefetch_candidates()) == []
    with trace.iteration():
        trace.observe(*events[0][:2], phase='forward')
        assert list(trace.prefetch_candidates()) == [ready]
        trace.observe(released, 'reshard')
        assert list(trace.prefetch_candidates()) == [ready, released]
        for module, operation, phase in events[2:]:
            trace.observe(module, operation, phase=phase)


def test_failures_keep_cached_trace_and_modules_are_weakly_owned():
    """Failed capture, incomplete replay, wrong metadata, and dead modules fail visibly."""
    module = nn.Module()
    trace = ExecutionTrace()
    with trace.iteration():
        trace.observe(module, 'unshard')
        trace.observe(module, 'reshard')
    cached = trace.events
    with pytest.raises(RuntimeError, match='aborted'):
        with trace.iteration(key='changed'):
            trace.observe(module, 'unshard')
            raise RuntimeError('aborted')
    assert trace.events is cached
    with pytest.raises(RuntimeError, match='before all events'):
        with trace.iteration():
            trace.observe(module, 'unshard')
    with pytest.raises(RuntimeError, match='diverged'):
        with trace.iteration():
            with trace.step(ExecutionStep(0, 0, 'forward')):
                trace.observe(module, 'unshard')
    with pytest.raises(RuntimeError, match='already active'):
        with trace.iteration():
            with trace.iteration():
                pass
    module_ref = ref(module)
    del module
    assert module_ref() is None
    with pytest.raises(RuntimeError, match='deleted'):
        _ = trace.events[0].module
