"""Behavioral checks for sampled timelines and local iteration dumps."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.event_tracker import (
    event_tracker,
    trace_layer_schedule,
    trace_module,
    trace_schedule_plan,
    trace_step,
    tracked_coalescing_manager,
)
from megatron.core.pipeline_parallel.schedules import custom_backward, deallocate_output_tensor
from megatron.core.utils import configure_nvtx_profiling


@pytest.fixture(scope='session', autouse=True)
def ensure_test_data():
    """These tests generate their own inputs and require no downloaded dataset."""


@pytest.fixture(autouse=True)
def tracker_env(monkeypatch, tmp_path):
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    monkeypatch.setenv('ENABLE_PP_TIMER', '1')
    monkeypatch.delenv('ENABLE_PP_TIMER_DUMP_ONLY', raising=False)
    monkeypatch.setenv('ENABLE_PP_TIMER_CPU', '1')
    monkeypatch.setenv('ENABLE_PP_TIMER_ITER', '2-3')
    monkeypatch.setenv('ENABLE_PP_TIMER_DUMP_DIR', str(tmp_path))
    monkeypatch.setenv('NVTX_EVENT_TRACKER_VERBOSE', '0')
    event_tracker.reset()
    configure_nvtx_profiling(False)
    yield
    torch.cuda.synchronize()
    event_tracker.reset()
    configure_nvtx_profiling(False)


@pytest.mark.parametrize('timer_enabled', ['0', '1'])
def test_dump_only_writes_each_completed_iteration(monkeypatch, tmp_path, timer_enabled):
    monkeypatch.setenv('ENABLE_PP_TIMER', timer_enabled)
    monkeypatch.setenv('ENABLE_PP_TIMER_DUMP_ONLY', '1')
    monkeypatch.setenv('ENABLE_PP_TIMER_ITER', '20-21')
    monkeypatch.setenv('ENABLE_PP_TIMER_INTERVAL', '100')

    @trace_step
    def step(iteration, config):
        return torch.ones(16, device='cuda').square().sum().item()

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    for iteration in (0, 1, 2):
        assert step(iteration, SimpleNamespace(cuda_graph_impl='none')) == 16
        path = tmp_path / f'event_times_iter{iteration}_rank_{rank}.json'
        assert path.exists(), 'Dump must be present before the step returns'
        payload = json.loads(path.read_text())
        assert payload['dump_only'] is True
        assert payload['iteration'] == iteration
        assert payload['events']['iteration/end'] >= payload['events']['iteration/start']
        cpu = tmp_path / f'event_times_cpu_iter{iteration}_rank_{rank}.json'
        assert json.loads(cpu.read_text())['iteration/end'] >= 0
        assert not event_tracker.enabled


@pytest.fixture
def nvtx_capture(monkeypatch):
    """Capture real call nesting and async labels without depending on an nsys install."""
    stack, ranges, labels = [], [], []

    def push(name):
        stack.append(name)
        ranges.append(tuple(stack))

    def start(name):
        labels.append(name)
        return len(labels)

    monkeypatch.setattr(torch.cuda.nvtx, 'range_push', push)
    monkeypatch.setattr(torch.cuda.nvtx, 'range_pop', stack.pop)
    monkeypatch.setattr(torch.cuda.nvtx, 'range_start', start)
    monkeypatch.setattr(torch.cuda.nvtx, 'range_end', lambda handle: None)
    configure_nvtx_profiling(True)
    yield stack, ranges, labels
    assert not stack


@pytest.mark.parametrize('has_forward,has_backward', [(True, True), (True, False), (False, True)])
def test_schedule_hierarchy_and_phase(nvtx_capture, has_forward, has_backward):
    stack, ranges, _ = nvtx_capture
    forward = (
        SimpleNamespace(
            chunk_state=SimpleNamespace(microbatch_id=3, vp_stage=1),
            layer=SimpleNamespace(layer_number=17),
        )
        if has_forward
        else None
    )
    backward = (
        SimpleNamespace(
            chunk_state=SimpleNamespace(microbatch_id=1, vp_stage=0),
            layer=SimpleNamespace(layer_number=9),
        )
        if has_backward
        else None
    )

    @trace_layer_schedule
    def layer_run(f_layer, b_layer):
        with event_tracker.nvtx_scope('attn'):
            assert len(stack) == 2

    @trace_schedule_plan
    def chunk_run(f_plan, b_plan):
        layer_run(forward, backward)

    assert event_tracker.begin(2)
    chunk_run(
        SimpleNamespace(_model_chunk_state=forward.chunk_state) if forward else None,
        SimpleNamespace(_model_chunk_state=backward.chunk_state) if backward else None,
    )
    if has_forward and has_backward:
        expected = 'iter-3-microbatch-3f-1b_layer_17f-layer_9b_vpp-1f-0b'
    elif has_forward:
        expected = 'iter-3-microbatch-3_layer_17f_vpp-1'
    else:
        expected = 'iter-3-microbatch-1_layer_9b_vpp-0'
    assert ranges[-1] == (expected, 'attn')
    assert any(k.startswith('forward-backward-') for k in event_tracker.events) == (
        has_forward and has_backward
    )
    event_tracker.finish()


def test_nvtx_covers_unsampled_iterations_without_cuda_events(nvtx_capture, monkeypatch, tmp_path):
    _, ranges, labels = nvtx_capture
    monkeypatch.delenv('ENABLE_PP_TIMER_ITER')
    monkeypatch.setenv('ENABLE_PP_TIMER_INTERVAL', '10')
    layer = _Layer()
    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    expected = torch.autograd.grad(layer(x, residual=x).sum(), x)[0]

    @trace_step
    def step(iteration, config):
        actual = torch.autograd.grad(layer(x, residual=x).sum(), x)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if iteration != 30:
            work = SimpleNamespace(wait=lambda: pytest.fail('NVTX-only must not wait on work'))
            event_tracker.record('collective', 'start')
            event_tracker.record_completion('collective', work, None)

    step(30, SimpleNamespace(cuda_graph_impl='none'))
    pool = dict(event_tracker.pool)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'Event', lambda **kw: pytest.fail('Unexpected CUDA event'))
        patch.setattr(torch.cuda, 'Stream', lambda **kw: pytest.fail('Unexpected CUDA stream'))
        patch.setattr(torch.cuda, 'synchronize', lambda: pytest.fail('Unexpected synchronization'))
        step(31, SimpleNamespace(cuda_graph_impl='none'))
        monkeypatch.setenv('ENABLE_PP_TIMER', '0')
        step(32, SimpleNamespace(cuda_graph_impl='none'))
    assert event_tracker.pool == pool
    assert len(list(tmp_path.glob('event_times_iter*.json'))) == 1
    for iteration in (30, 31, 32):
        assert (f'iter-{iteration + 1}',) in ranges
    assert labels.count('mlp backward') == 3


@pytest.mark.parametrize('kind', ['checkpoint', 'without_output', 'transformer_engine'])
def test_recompute_nvtx_captures_origin_and_preserves_gradients(kind, nvtx_capture):
    from megatron.core.extensions.transformer_engine import te_checkpoint
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
    from megatron.core.tensor_parallel.random import checkpoint as tp_checkpoint
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    stack, ranges, _ = nvtx_capture
    x = torch.randn(16, device='cuda', requires_grad=True)
    expected = torch.autograd.grad(x.sin().square().sum(), x)[0]
    assert event_tracker.begin(2)
    event_tracker.batch, event_tracker.chunk, event_tracker.layer = 3, 1, 7
    if kind == 'without_output':
        checkpoint_object = CheckpointWithoutOutput(fp8=None)
        y = checkpoint_object.checkpoint(torch.sin, x)
    elif kind == 'transformer_engine':
        y = te_checkpoint(torch.sin, False, get_cuda_rng_tracker, torch.distributed.group.WORLD, x)
    else:
        y = tp_checkpoint(torch.sin, False, x)
    loss = y.square().sum()
    if kind == 'without_output':
        checkpoint_object.discard_output_and_register_recompute(loss)
    assert not any(path[-1].startswith('recompute ') for path in ranges)
    event_tracker.layer = 99
    loss.backward()
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    replays = [path[-1] for path in ranges if path[-1].startswith('recompute ')]
    assert len(replays) == 1
    assert replays[0] == 'recompute sin [L7,mb3,vpp1]'
    assert not stack and not event_tracker.in_recompute
    event_tracker.finish()


def test_recompute_scope_closes_on_exception(nvtx_capture):
    stack, _, _ = nvtx_capture
    with pytest.raises(ValueError, match='replay failed'):
        with event_tracker.recompute_scope('recompute/test'):
            assert event_tracker.in_recompute and stack
            raise ValueError('replay failed')
    assert not event_tracker.in_recompute and not stack


def test_compact_nodes_keep_timer_detail_without_duplicate_nvtx(nvtx_capture):
    _, ranges, labels = nvtx_capture
    layer = _Layer()
    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    node = SimpleNamespace(
        chunk_state=SimpleNamespace(microbatch_id=3, vp_stage=1),
        layer_number=1,
        name='pre_dispatch_computation',
    )
    assert event_tracker.begin(2)
    with event_tracker.nvtx_scope('layer parent'):
        with event_tracker.schedule_node(node, 'forward'):
            output = layer(x, residual=x)
        with event_tracker.schedule_node(node, 'backward'):
            output.sum().backward()
    assert ('layer parent', 'attn forward') in ranges
    assert ('layer parent', 'attn backward') in ranges
    assert labels == []
    assert 'batch3-chunk1-layer1-mlp-forward/start' in event_tracker.events
    assert 'batch3-chunk1-layer1-mlp-backward/end' in event_tracker.events
    assert (
        'batch3-chunk1-layer1-schedule-pre_dispatch_computation-forward/start'
        in event_tracker.events
    )
    assert event_tracker.node_depth == 0
    event_tracker.finish()


def test_verbose_nvtx_keeps_timer_names(nvtx_capture, monkeypatch):
    _, _, labels = nvtx_capture
    monkeypatch.setenv('NVTX_EVENT_TRACKER_VERBOSE', '1')
    assert event_tracker.begin(2)
    with event_tracker.scope('batch0-chunk0-layer1-attn-forward'):
        pass
    assert 'iteration_2/batch0-chunk0-layer1-attn-forward' in labels
    event_tracker.finish()


def test_schedule_node_context_closes_on_exception(nvtx_capture):
    stack, _, _ = nvtx_capture
    node = SimpleNamespace(
        chunk_state=SimpleNamespace(microbatch_id=3, vp_stage=1),
        layer_number=1,
        name='pre_dispatch_computation',
    )
    assert event_tracker.begin(2)
    with pytest.raises(ValueError, match='node failed'):
        with event_tracker.schedule_node(node, 'backward'):
            assert event_tracker.node_depth == 1
            raise ValueError('node failed')
    assert event_tracker.node_depth == 0 and not stack
    assert (event_tracker.batch, event_tracker.chunk, event_tracker.layer) == (0, 0, None)
    event_tracker.finish()


class _Layer(torch.nn.Module):
    layer_number = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))

    @trace_module('mlp')
    def forward(self, x, *, residual):
        return (x @ self.weight).square() + residual


def test_boundaries_preserve_gradients_and_pool(tmp_path, monkeypatch):
    layer = _Layer()
    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    reference = layer(x, residual=residual)
    grads = torch.autograd.grad(reference.sum(), (x, residual, layer.weight))
    assert not event_tracker.begin(1)
    assert event_tracker.begin(2)
    result = layer(x, residual=residual)
    observed_grads = torch.autograd.grad(result.sum(), (x, residual, layer.weight))
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for observed, expected in zip(observed_grads, grads):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    pool = dict(event_tracker.pool)
    event_tracker.finish()
    payload = json.loads(next(tmp_path.glob('event_times_iter2_*.json')).read_text())
    for direction in ('forward', 'backward'):
        name = f'batch0-chunk0-layer1-mlp-{direction}'
        assert payload['events'][f'{name}/end'] >= payload['events'][f'{name}/start']
    assert event_tracker.begin(3)
    layer(x, residual=residual).sum().backward()
    event_tracker.finish()
    assert all(event_tracker.pool[key] is value for key, value in pool.items())
    assert not event_tracker.begin(4)
    monkeypatch.setenv('ENABLE_PP_TIMER_CPU', '0')
    assert event_tracker.begin(3)
    layer(x, residual=residual).sum().backward()
    event_tracker.finish()
    assert not list(tmp_path.glob('event_times_cpu_iter3_*.json'))
    payload = json.loads(next(tmp_path.glob('event_times_iter3_*.json')).read_text())
    assert not payload['cpu_enabled']


def test_repeated_recomputed_calls_are_not_overwritten(tmp_path):
    layer = _Layer()
    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    assert event_tracker.begin(2)
    a = checkpoint(lambda value: layer(value, residual=value), x, use_reentrant=True)
    b = layer(x, residual=x)
    event_tracker.in_backward = True
    (a + b).sum().backward()
    event_tracker.finish()
    payload = json.loads(next(tmp_path.glob('event_times_iter2_*.json')).read_text())
    assert any('recompute-forward' in name for name in payload['events'])
    assert any('#1' in name for name in payload['events'])
    assert torch.isfinite(x.grad).all()


def test_pipeline_output_deallocation():
    layer = _Layer()
    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    reference = layer(x, residual=x)
    expected = torch.autograd.grad(reference.sum(), x)[0]
    assert event_tracker.begin(2)
    output = layer(x, residual=x)
    assert output._base is None
    gradient = torch.ones_like(output)
    deallocate_output_tensor(output, deallocate_pipeline_outputs=True)
    assert output.numel() == 1
    custom_backward(output, gradient)
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    event_tracker.finish()


def test_async_collective_completion(tmp_path):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group('nccl')
    values = torch.ones(1024, device='cuda')
    assert event_tracker.begin(2)
    with tracked_coalescing_manager(
        'test-allreduce', torch.distributed.group.WORLD, async_ops=True
    ) as work:
        torch.distributed.all_reduce(values, async_op=True)
    work.wait()
    consumed = values.square()
    event_tracker.finish()
    assert torch.equal(values, torch.full_like(values, torch.distributed.get_world_size()))
    assert torch.equal(consumed, torch.full_like(consumed, torch.distributed.get_world_size() ** 2))
    payload = json.loads(next(tmp_path.glob('event_times_iter2_*.json')).read_text())
    assert payload['events']['test-allreduce/end'] >= payload['events']['test-allreduce/start']
