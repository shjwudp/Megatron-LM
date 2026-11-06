"""Sampled eager-training CPU/CUDA timelines, compatible with internal PP Timer dumps.

Module boundaries describe elapsed stream time, including stalls, rather than sums of
kernel durations. No synchronization is performed at a module boundary. CUDA graphs
are intentionally unsupported: Python annotations would otherwise describe capture.
"""

import functools
import inspect
import json
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.distributed import _coalescing_manager
from torch.utils._pytree import tree_flatten, tree_unflatten

_MODULE_EVENT = re.compile(
    r'^batch\d+-chunk\d+-layer\d+-'
    r'(attn|mlp|routed_experts_compute|shared_experts_compute|moe_dispatch|moe_combine)'
    r'(?:#\d+)?-(forward|backward|recompute-forward)(?:#\d+)?$'
)
_NODE_NAMES = {
    'pre_dispatch_computation': 'attn',
    'moe_dispatch': 'dispatch',
    'moe_combine': 'combine',
    'mtp_post_process': 'mtp loss',
    'pre_process': 'preprocess',
    'post_process': 'output/loss',
    'loss_func': 'loss',
}


def _recompute_operation(operation):
    """Use the configured recompute names instead of Python closure paths."""
    leaf = operation.rsplit('.', 1)[-1]
    return {
        '_fc1_act': 'dense_fc1_act',
        'qkv_up_proj_and_rope_apply': 'mla_up_proj',
        '_grouped_output_projection': 'csa_o_proj',
        'RMSNorm': 'layernorm',
        'LayerNorm': 'layernorm',
        'TEColumnParallelGroupedLinear': 'moe_fc1',
        'bias_act_func': 'moe_act',
        'SharedExpertMLP': 'shared_experts',
        'full_layer': 'layer',
    }.get(leaf, leaf)


class EventTracker:
    """One tracker per process. Events are reused only after collection completes."""

    def __init__(self):
        self.enabled = False
        self.timer_enabled = False
        self.dump_only = False
        self.cpu_enabled = False
        self.nvtx_enabled = False
        self.nvtx_verbose = False
        self.node_depth = 0
        self.pool = {}
        self.events = {}
        self.cpu_events = {}
        self.ranges = {}
        self.nvtx_messages = {}
        self.counts = defaultdict(int)
        self.batch = 0
        self.chunk = 0
        self.layer = None
        self.in_backward = False
        self.in_recompute = False
        self.generation = 0
        self.observer_streams = {}

    @property
    def nvtx_active(self):
        """NVTX hierarchy also follows --nvtx-ranges outside Timer samples."""
        from megatron.core.utils import is_nvtx_profiling_enabled

        return (self.enabled and self.nvtx_enabled) or is_nvtx_profiling_enabled()

    @contextmanager
    def nvtx_scope(self, name):
        """Nest CPU submission ranges without adding CUDA events or synchronization."""
        if not self.nvtx_active or name is None:
            yield
            return
        name = self.nvtx_messages.setdefault(name, name)
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def recompute_label(self, function, *, state=None, layer=None):
        """Capture strings only: a deferred label must not retain a model or tensor."""
        if not self.nvtx_active:
            return None
        while isinstance(function, functools.partial):
            function = function.func
        operation = (
            function
            if isinstance(function, str)
            else getattr(function, '__qualname__', type(function).__name__)
        )
        batch = getattr(state, 'microbatch_id', self.batch)
        chunk = getattr(state, 'vp_stage', self.chunk) or 0
        layer = self.layer if layer is None else layer
        if self.nvtx_verbose:
            return f'recompute/{operation}/mb={batch},vpp={chunk},layer={layer}'
        return f'recompute {_recompute_operation(operation)} [L{layer},mb{batch},vpp{chunk}]'

    @contextmanager
    def recompute_scope(self, label):
        """Mark only the forward replay, excluding the following backward traversal."""
        previous = self.in_recompute
        self.in_recompute = True
        try:
            with self.nvtx_scope(label):
                yield
        finally:
            self.in_recompute = previous

    def unique(self, name):
        """Keep repeated calls (including recomputation) instead of overwriting them."""
        count = self.counts[name]
        self.counts[name] += 1
        return name if count == 0 else f'{name}#{count}'

    def _nvtx_event_name(self, name):
        """Keep Timer keys intact while avoiding duplicate NVTX compute envelopes."""
        if self.nvtx_verbose:
            return f'iteration_{self.iteration}/{name}'
        if name.startswith('optimizer-'):
            return name.split('#', 1)[0].replace('-', ' ')
        if self.node_depth == 0:
            # Ordinary schedules have no ScheduleNodes to mark their module work.
            match = _MODULE_EVENT.fullmatch(name)
            if match:
                operation, direction = match.groups()
                return f'{_NODE_NAMES.get(operation, operation)} {direction}'
        return None

    def record(self, name, endpoint, *, stream=None, cpu=True):
        """Record an endpoint on the caller's stream; never wait for its completion."""
        if not self.enabled:
            return
        key = f'{name}/{endpoint}'
        if self.timer_enabled:
            event = self.pool.get(key)
            if event is None:
                event = self.pool[key] = torch.cuda.Event(enable_timing=True)
            event.record(stream)
            self.events[key] = event
        if cpu and self.cpu_enabled:
            self.cpu_events[key] = (time.perf_counter_ns() - self.cpu_reference) / 1e6
        if cpu and self.nvtx_enabled:
            if endpoint == 'start':
                message = self._nvtx_event_name(name)
                if message is not None:
                    message = self.nvtx_messages.setdefault(message, message)
                    self.ranges[name] = torch.cuda.nvtx.range_start(message)
            elif name in self.ranges:
                torch.cuda.nvtx.range_end(self.ranges.pop(name))

    def begin(self, iteration):
        """Sample CUDA timers independently from the complete NVTX capture window."""
        from megatron.core.utils import is_nvtx_profiling_enabled

        # Explicit local dumps cover every completed iteration, even when the
        # ordinary timer is disabled or configured to sample a narrower window.
        self.dump_only = os.getenv('ENABLE_PP_TIMER_DUMP_ONLY', '0') == '1'
        timer_enabled = self.dump_only
        if not self.dump_only and os.getenv('ENABLE_PP_TIMER', '0') == '1':
            selected = os.getenv('ENABLE_PP_TIMER_ITER')
            if selected is not None:
                bounds = selected.split('-')
                if len(bounds) not in (1, 2):
                    raise ValueError('ENABLE_PP_TIMER_ITER must be N or START-END')
                first, last = int(bounds[0]), int(bounds[-1])
                if first < 0 or last < first:
                    raise ValueError('Invalid PP Timer iteration range')
                timer_enabled = first <= iteration <= last
            else:
                interval = int(os.getenv('ENABLE_PP_TIMER_INTERVAL', '20'))
                timer_enabled = interval > 0 and iteration % interval == 0
        nvtx_enabled = is_nvtx_profiling_enabled() or (
            timer_enabled and os.getenv('NVTX_EVENT_TRACKER_ENABLED', '0') == '1'
        )
        if not timer_enabled and not nvtx_enabled:
            return False
        self.iteration = iteration
        self.enabled = True
        self.timer_enabled = timer_enabled
        self.cpu_enabled = timer_enabled and os.getenv('ENABLE_PP_TIMER_CPU', '0') == '1'
        self.nvtx_enabled = nvtx_enabled
        self.nvtx_verbose = os.getenv('NVTX_EVENT_TRACKER_VERBOSE', '0') == '1'
        self.events.clear()
        self.cpu_events.clear()
        self.counts.clear()
        if timer_enabled:
            self.reference = self.pool.setdefault('reference', torch.cuda.Event(enable_timing=True))
            self.reference.record()
            self.reference.synchronize()
            self.reference_unix_ms = time.time_ns() / 1e6
            self.cpu_reference = time.perf_counter_ns()
        self.record('iteration', 'start')
        return True

    def finish(self):
        """Collect after training work, outside the forward/backward measurement."""
        self.record('iteration', 'end')
        if not self.timer_enabled:
            self.reset()
            return
        sync_start = time.perf_counter_ns()
        torch.cuda.synchronize()
        sync_ms = (time.perf_counter_ns() - sync_start) / 1e6
        events = {key: self.reference.elapsed_time(value) for key, value in self.events.items()}
        output = Path(
            os.getenv('ENABLE_PP_TIMER_DUMP_DIR', os.getenv('PP_TIMER_LOG_DIR', './pp_timer'))
        )
        output.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        stem = f'iter{self.iteration}_rank_{rank}.json'
        payload = {
            'reference_unix_ms': self.reference_unix_ms,
            'events': events,
            'synchronize_ms': sync_ms,
            'iteration': self.iteration,
            'rank': rank,
            'cpu_enabled': self.cpu_enabled,
            'dump_only': self.dump_only,
        }
        (output / f'event_times_{stem}').write_text(json.dumps(payload))
        cpu_path = output / f'event_times_cpu_{stem}'
        if self.cpu_enabled:
            cpu_path.write_text(json.dumps(self.cpu_events))
        else:
            # A rerun in the same directory must not expose CPU data from an older run.
            cpu_path.unlink(missing_ok=True)
        self.reset()

    def reset(self):
        """Close host ranges also on exceptions; do not manufacture GPU endpoints."""
        for handle in self.ranges.values():
            torch.cuda.nvtx.range_end(handle)
        self.ranges.clear()
        self.enabled = False
        self.timer_enabled = False
        self.nvtx_enabled = False
        self.nvtx_verbose = False
        self.node_depth = 0
        self.in_backward = False
        self.in_recompute = False
        self.batch = self.chunk = 0
        self.layer = None
        self.generation += 1

    def record_completion(self, name, work, group):
        """Observe async completion on a stream private to this communicator."""
        if not self.timer_enabled:
            # NVTX alone describes host submission; observe GPU completion only
            # on Timer samples, without adding streams or waits to other steps.
            self.record(name, 'end')
            return
        stream = self.observer_streams.get(group)
        if stream is None:
            stream = self.observer_streams[group] = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            stream.wait_event(self.events[f'{name}/start'])
            work.wait()
            self.record(name, 'end')

    @contextmanager
    def scope(self, name):
        """A CPU submission interval and a CUDA stream interval."""
        if not self.enabled:
            yield
            return
        name = self.unique(name)
        self.record(name, 'start')
        try:
            yield
        finally:
            self.record(name, 'end')

    @contextmanager
    def schedule_node(self, node, direction):
        """Measure a combined-schedule node on its already-selected execution stream."""
        if not self.enabled and not self.nvtx_active:
            yield
            return
        state = getattr(node, 'chunk_state', None)
        previous = self.batch, self.chunk, self.layer, self.in_backward, self.node_depth
        self.batch = getattr(state, 'microbatch_id', 0)
        self.chunk = getattr(state, 'vp_stage', 0) or 0
        self.in_backward = direction in ('backward', 'wgrad')
        layer = getattr(node, 'layer_number', None)
        self.layer = layer
        self.node_depth += 1
        prefix = f'batch{self.batch}-chunk{self.chunk}'
        if layer is not None:
            prefix += f'-layer{layer}'
        # Keep node intervals distinct from the finer module/autograd intervals.
        operation = node.name if direction == 'wgrad' else f'schedule-{node.name}'
        nvtx_name = f'{_NODE_NAMES.get(node.name, node.name)} {direction}'
        if layer is None:
            nvtx_name += f' [mb{self.batch},vpp{self.chunk}]'
        try:
            with self.nvtx_scope(nvtx_name), self.scope(f'{prefix}-{operation}-{direction}'):
                yield
        finally:
            self.batch, self.chunk, self.layer, self.in_backward, self.node_depth = previous


event_tracker = EventTracker()


def _schedule_label(kind, forward, backward):
    batches, chunks, layers = [], [], []
    paired = forward is not None and backward is not None
    for direction, plan in (('f', forward), ('b', backward)):
        if plan is None:
            continue
        state = plan.chunk_state if kind == 'layer' else plan._model_chunk_state
        batches.append(f'{state.microbatch_id}{direction if paired else ""}')
        chunks.append(f'{state.vp_stage or 0}{direction if paired else ""}')
        if kind == 'layer':
            node = getattr(plan, 'pre_dispatch_computation', None)
            layer_kind = 'mtp' if getattr(node, 'is_mtp', False) else ''
            layers.append(f'layer_{layer_kind}{plan.layer.layer_number}{direction}')
    iteration = getattr(event_tracker, 'iteration', -1) + 1
    prefix = f'iter-{iteration}-microbatch-{"-".join(batches)}'
    body = '-'.join(layers) if kind == 'layer' else 'schedule'
    return f'{prefix}_{body}_vpp-{"-".join(chunks)}'


def trace_layer_schedule(func):
    """Nest the actual layer pair, including unmatched forward/backward tails."""

    @functools.wraps(func)
    def wrapped(f_layer, b_layer, *args, **kwargs):
        if not event_tracker.nvtx_active:
            return func(f_layer, b_layer, *args, **kwargs)
        with event_tracker.nvtx_scope(_schedule_label('layer', f_layer, b_layer)):
            return func(f_layer, b_layer, *args, **kwargs)

    return wrapped


def trace_schedule_plan(func):
    """Bracket combined microbatches, whose compute may interleave on several streams."""

    @functools.wraps(func)
    def wrapped(f_schedule_plan, b_schedule_plan, *args, **kwargs):
        tracker = event_tracker
        if not tracker.enabled and not tracker.nvtx_active:
            return func(f_schedule_plan, b_schedule_plan, *args, **kwargs)
        names, coordinates = [], []
        for direction, plan in (('forward', f_schedule_plan), ('backward', b_schedule_plan)):
            if plan is None:
                coordinates.extend((-1, -1))
                continue
            state = plan._model_chunk_state
            chunk, batch = state.vp_stage or 0, state.microbatch_id
            coordinates.extend((chunk, batch))
            names.append(tracker.unique(f'{direction}-{chunk}-{batch}'))
        if f_schedule_plan is not None and b_schedule_plan is not None:
            names.append(tracker.unique('forward-backward-' + '-'.join(map(str, coordinates))))
        label = (
            _schedule_label('schedule', f_schedule_plan, b_schedule_plan)
            if tracker.nvtx_verbose
            else None
        )
        with tracker.nvtx_scope(label):
            for name in names:
                tracker.record(name, 'start')
            try:
                return func(f_schedule_plan, b_schedule_plan, *args, **kwargs)
            finally:
                for name in reversed(names):
                    tracker.record(name, 'end')

    return wrapped


class _BackwardBoundary(torch.autograd.Function):
    """An identity edge joining all differentiable inputs/outputs of one operation."""

    @staticmethod
    def forward(ctx, name, endpoint, generation, *tensors):
        """Keep values and retain the event identity for this graph invocation."""
        ctx.name, ctx.endpoint, ctx.generation = name, endpoint, generation
        ctx.set_materialize_grads(False)
        # A Function returning its inputs directly creates autograd views. Pipeline
        # output deallocation requires viewless TensorImpls. Detached aliases retain
        # storage/version sharing; this Function's backward restores the gradient edge.
        return tuple(tensor.detach() for tensor in tensors)

    @staticmethod
    def backward(ctx, *grads):
        """Record on the autograd node's stream and return gradients unchanged."""
        if event_tracker.enabled and ctx.generation == event_tracker.generation:
            event_tracker.record(ctx.name, ctx.endpoint)
        return None, None, None, *grads


def _boundary(value, name, endpoint):
    flat, spec = tree_flatten(value)
    positions = [
        i for i, item in enumerate(flat) if isinstance(item, torch.Tensor) and item.requires_grad
    ]
    if positions and torch.is_grad_enabled():
        tensors = _BackwardBoundary.apply(
            name, endpoint, event_tracker.generation, *(flat[i] for i in positions)
        )
        for i, tensor in zip(positions, tensors):
            flat[i] = tensor
    return tree_unflatten(flat, spec)


def trace_module(operation):
    """Instrument a layer method only on sampled iterations, including autograd edges."""

    def decorate(func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            tracker = event_tracker
            if not tracker.enabled:
                return func(self, *args, **kwargs)
            name = tracker.unique(
                f'batch{tracker.batch}-chunk{tracker.chunk}-layer{self.layer_number}-{operation}'
            )
            backward_name = f'{name}-backward'
            args, kwargs = _boundary((args, kwargs), backward_name, 'end')
            suffix = (
                'recompute-forward' if tracker.in_recompute or tracker.in_backward else 'forward'
            )
            previous_layer, tracker.layer = tracker.layer, self.layer_number
            try:
                with tracker.scope(f'{name}-{suffix}'):
                    result = func(self, *args, **kwargs)
            finally:
                tracker.layer = previous_layer
            return _boundary(result, backward_name, 'start')

        return wrapped

    return decorate


def trace_step(func):
    """Bracket the whole train_step, including cleanup on exceptions."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if (
            os.getenv('ENABLE_PP_TIMER', '0') != '1'
            and os.getenv('ENABLE_PP_TIMER_DUMP_ONLY', '0') != '1'
            and not event_tracker.nvtx_active
        ):
            return func(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        iteration = bound.arguments.get('iteration')
        if iteration is None or not event_tracker.begin(iteration):
            return func(*args, **kwargs)
        config = bound.arguments['config']
        if getattr(config, 'cuda_graph_impl', 'none') != 'none':
            event_tracker.reset()
            raise ValueError('PP Timer currently supports eager training only; disable CUDA graphs')
        try:
            with event_tracker.nvtx_scope(f'iter-{iteration + 1}'):
                result = func(*args, **kwargs)
            event_tracker.finish()
            return result
        finally:
            event_tracker.reset()

    return wrapped


@contextmanager
def tracked_coalescing_manager(name, *args, **kwargs):
    """Measure launch-to-completion without serializing asynchronous collectives.

    Async work is waited on a separate observer stream. The training stream never
    waits on the observer. CPU endpoints bracket submission, GPU endpoints include
    NCCL queuing and completion (not just kernel execution).
    """
    tracker = event_tracker
    if not tracker.enabled:
        with _coalescing_manager(*args, **kwargs) as manager:
            yield manager
        return
    name = tracker.unique(name)
    tracker.record(name, 'start')
    with _coalescing_manager(*args, **kwargs) as manager:
        yield manager
    if kwargs.get('async_ops', False):
        group = args[0] if args else kwargs.get('group')
        tracker.record_completion(name, manager, group)
    else:
        tracker.record(name, 'end')
