# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure state/lifetime contracts; physical numerics are covered by the GPU oracle."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.gradient_window import (
    GradientWindow,
)


class _Header:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return self.values.copy()


class _Stream:
    def __init__(self, name, log):
        self.name, self.log, self.device = name, log, 'cuda:0'

    def wait_event(self, event):
        self.log.append(('wait_event', self.name, event))

    def wait_stream(self, stream):
        self.log.append(('wait_stream', self.name, stream.name))

    def record_event(self):
        event = (self.name, len(self.log))
        self.log.append(('record', event))
        return event


class _Buffer:
    def __init__(self, dtype, value=0):
        self.dtype, self.value = dtype, value

    def cast(self, dtype):
        return _Buffer(dtype, self.value)


class _Group:
    requires_grad = True
    fsdp_parameters = ()

    def __init__(self, context, charge=10, communication_dtype=torch.float32, network=True):
        self.context, self.gradient_window_storage_bytes = context, charge
        self.main_grad = SimpleNamespace(dtype=torch.float32)
        self.grad_comm_dtype, self.has_inner_network_reduction = communication_dtype, network
        self.inner, self.outer, self.incoming = [], [], None

    def allocate_partial_grad_buffer(self, *, dtype=None):
        return _Buffer(self.grad_comm_dtype if dtype is None else dtype)

    def copy_gradients_to_partial_buffer(self, buffer, *, accumulate=False):
        if self.incoming is None:
            raise RuntimeError('Missing local contribution')
        buffer.value = buffer.value + self.incoming if accumulate else self.incoming
        self.incoming = None

    def reduce_inner_gradients(self, buffer):
        self.inner.append((buffer.value, buffer.dtype))
        self.context.log.append(('inner', id(self), buffer.value))

    def reduce_outer_gradients(self):
        self.outer.append(sum(value for value, _ in self.inner))
        self.context.log.append(('outer', id(self)))


class _Owner:
    def __init__(self, *groups):
        self.parameter_groups = groups


class _Context:
    def __init__(self):
        self.log, self.is_last_microbatch = [], False
        self.current = _Stream('compute_a', self.log)
        self.inner_reduction_stream = _Stream('inner', self.log)
        self.outer_reduction_stream = _Stream('outer', self.log)
        self.allgather_stream = _Stream('ag', self.log)
        self._prefetch_process_group = object()
        self.policy = None

    def current_stream(self):
        return self.current

    def _flush_inner_gradient_window(self, group, *, reason, partial_grad=None):
        if partial_grad is None:
            self.policy.flush(group, reason=reason)
        else:
            group.reduce_inner_gradients(partial_grad)
            self.policy.legacy_flush(reason=reason)

    def _finalize_gradient_outer(self, group):
        group.reduce_outer_gradients()
        self.policy.outer_finished(group)


class TestGradientWindow(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        prefix = 'megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.gradient_window'
        self.stack.enter_context(
            patch(prefix + '.torch.cuda.stream', side_effect=lambda _: nullcontext())
        )
        self.stack.enter_context(
            patch(prefix + '.torch.cuda.is_current_stream_capturing', return_value=False)
        )
        self.stack.enter_context(
            patch(prefix + '.torch.tensor', side_effect=lambda values, **_: _Header(values))
        )
        self.reduce = self.stack.enter_context(patch(prefix + '.dist.all_reduce'))

    def _build(
        self, window=4, budget=20, groups=1, communication_dtype=torch.float32, network=True
    ):
        context = _Context()
        owners = tuple(
            _Owner(_Group(context, communication_dtype=communication_dtype, network=network))
            for _ in range(groups)
        )
        policy = GradientWindow(context, owners, owners, window, budget)
        context.policy = policy
        return context, owners, policy

    def _use(self, context, owner, value=1, *, last=False):
        group = owner.parameter_groups[0]
        context.is_last_microbatch, group.incoming = last, value
        consumed = context.policy.contribute(owner, group, 0)
        if not consumed:
            buffer = group.allocate_partial_grad_buffer()
            group.copy_gradients_to_partial_buffer(buffer)
            context._flush_inner_gradient_window(
                group, reason=context.policy.reason(group), partial_grad=buffer
            )
            if last:
                context._finalize_gradient_outer(group)
        self.assertEqual(
            context.policy.reserved_bytes,
            sum(lease.charge for lease in context.policy.pending.values())
            + sum(lease.charge for lease in context.policy.retiring),
        )

    def _record(self, context, owners, count=5, *, last=True):
        for index in range(count):
            for owner in owners:
                self._use(context, owner, index + 1, last=last and index == count - 1)
        context.policy.finish_step()
        for owner in owners:
            owner.parameter_groups[0].inner.clear()
            owner.parameter_groups[0].outer.clear()
        context.log.clear()

    def test_recording_then_window_sums_and_tail(self):
        context, (owner,), policy = self._build()
        self._record(context, (owner,))
        self.assertFalse(policy.recording)
        for index in range(5):
            self._use(context, owner, index + 1, last=index == 4)
        policy.finish_step()
        group = owner.parameter_groups[0]
        self.assertEqual([value for value, _ in group.inner], [10, 5])
        self.assertEqual(group.outer, [15])
        self.assertEqual(policy.last_step_statistics['contributions'], 5)
        self.assertEqual(policy.last_step_statistics['inner_reductions'], 2)
        self.assertEqual(policy.reserved_bytes, 0)

    def test_optimizer_tail_submits_all_inner_before_outer(self):
        context, owners, policy = self._build(groups=2)
        self._record(context, owners, count=3, last=False)
        for index in range(3):
            for owner in owners:
                self._use(context, owner, index + 1)
        policy.finish_step()
        actions = [row[0] for row in context.log if row[0] in ('inner', 'outer')]
        self.assertEqual(actions, ['inner', 'inner', 'outer', 'outer'])
        self.assertTrue(all(owner.parameter_groups[0].outer == [6] for owner in owners))

    def test_fifo_eviction_includes_retiring_inputs_in_budget(self):
        context, owners, policy = self._build(groups=3)
        self._record(context, owners, count=3, last=False)
        for index in range(3):
            for owner in owners:
                self._use(context, owner, index + 1)
        policy.finish_step()
        self.assertLessEqual(policy.peak_reserved_bytes, 20)
        self.assertGreater(policy.last_step_statistics['budget_evictions'], 0)
        self.assertTrue(all(owner.parameter_groups[0].outer == [6] for owner in owners))

    def test_original_communication_dtype_is_preserved(self):
        context, (owner,), policy = self._build(communication_dtype=torch.bfloat16)
        self._record(context, (owner,), count=2)
        self._use(context, owner, 1)
        self._use(context, owner, 2, last=True)
        policy.finish_step()
        self.assertEqual(owner.parameter_groups[0].inner, [(3, torch.bfloat16)])

    def test_window_one_and_singletons_need_no_window_budget(self):
        context, owners, policy = self._build(window=4, budget=0, network=False)
        self._record(context, owners, count=2)
        for index in range(2):
            self._use(context, owners[0], index + 1, last=index == 1)
        policy.finish_step()
        self.assertEqual(len(owners[0].parameter_groups[0].inner), 2)
        self.assertEqual(policy.reserved_bytes, 0)
        context, owners, policy = self._build(window=1, budget=0)
        self._record(context, owners, count=2)
        self._use(context, owners[0], 1)
        self._use(context, owners[0], 2, last=True)
        policy.finish_step()
        self.assertEqual(len(owners[0].parameter_groups[0].inner), 2)

    def test_policy_is_pinned_immediately_after_eager_consensus(self):
        context, owners, policy = self._build()
        self._record(context, owners, count=1)
        policy.window_size = 2
        with self.assertRaisesRegex(RuntimeError, 'Compiled gradient-window'):
            self._use(context, owners[0])

    def test_frozen_off_and_on_policies_reject_mutation(self):
        for window in (1, 4):
            context, owners, policy = self._build(window=window)
            self._record(context, owners, count=1)
            with patch(
                'megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.gradient_window'
                '.torch.cuda.is_current_stream_capturing',
                return_value=True,
            ):
                with self.assertRaisesRegex(RuntimeError, 'captured FSDP'):
                    self._use(context, owners[0], last=True)
            policy.prepare_capture()
            policy.prepare_capture()
            with self.assertRaises(TypeError):
                policy.capacities[owners[0].parameter_groups[0]] = 11
            group = owners[0].parameter_groups[0]
            group.grad_comm_dtype = torch.bfloat16
            with self.assertRaisesRegex(RuntimeError, 'dtype changed'):
                policy.validate_capture()
            group.grad_comm_dtype = torch.float32
            policy.budget_bytes += 1
            with self.assertRaisesRegex(RuntimeError, 'policy or trace changed'):
                policy.validate_capture()

    def test_changed_dtype_and_trace_fail_before_reduction(self):
        context, owners, policy = self._build()
        self._record(context, owners, count=1)
        with self.assertRaisesRegex(RuntimeError, 'lifecycle diverged'):
            self._use(context, owners[0], last=False)
        context, owners, policy = self._build()
        self._record(context, owners, count=1)
        owners[0].parameter_groups[0].grad_comm_dtype = torch.bfloat16
        with self.assertRaisesRegex(RuntimeError, 'dtype changed'):
            self._use(context, owners[0], last=True)

    def test_repeated_clean_finish_is_idempotent(self):
        context, owners, policy = self._build()
        self._record(context, owners, count=1)
        calls = self.reduce.call_count
        policy.finish_step()
        policy.finish_step()
        self.assertEqual(self.reduce.call_count, calls)
        self.assertFalse(policy.pending or policy.retiring or policy.pending_outer)

    def test_cross_stream_accumulation_and_retirement_dependencies(self):
        context, owners, policy = self._build(window=2)
        self._record(context, owners, count=2)
        self._use(context, owners[0], 1)
        context.current = _Stream('compute_b', context.log)
        self._use(context, owners[0], 2, last=True)
        policy.finish_step()
        waits = [row for row in context.log if row[0] == 'wait_event']
        self.assertTrue(any(row[1] == 'compute_b' and row[2][0] == 'compute_a' for row in waits))
        self.assertTrue(any(row[1] == 'inner' and row[2][0] == 'compute_b' for row in waits))
        self.assertTrue(any(row[1] == 'compute_a' and row[2][0] == 'inner' for row in waits))

    def test_distributed_consensus_rejects_mismatched_trace(self):
        context, owners, policy = self._build()
        self._use(context, owners[0], last=True)
        self.reduce.side_effect = lambda header, **_: header.values.__setitem__(0, 99)
        with self.assertRaisesRegex(RuntimeError, 'differs across ranks'):
            policy.finish_step()

    def test_invalid_policy_and_empty_context(self):
        context = _Context()
        with self.assertRaises(ValueError):
            GradientWindow(context, (), (), 0, 0)
        with self.assertRaises(ValueError):
            GradientWindow(context, (), (), 4, -1)
        with self.assertRaises(ValueError):
            self._build(window=4, budget=9)
        empty = GradientWindow(context, (), (), 4, 0)
        empty.finish_step()
        self.assertTrue(empty.recording)
        empty.prepare_capture()
        empty.validate_capture()
