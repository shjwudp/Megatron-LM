# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only contract checks; run with unittest, without Megatron's GPU fixtures."""

import unittest

from . import Chunk, Combined1f1bFsdpScheduler, FsdpActions, Node, Step, VppFsdpScheduler


class Runtime:
    """Small fake runtime that checks materialization and explicit gradient readiness."""

    def __init__(self):
        self.loaded = set()
        self.ready = set()
        self.events = []
        self.actions = FsdpActions(self.demand, self.prefetch, self.release, self.finish_backward)

    def prefetch(self, unit):
        if unit not in self.loaded:
            self.loaded.add(unit)
            self.events.append(('gather', unit))

    def demand(self, unit):
        self.prefetch(unit)
        self.events.append(('wait', unit))

    def release(self, unit):
        self.loaded.remove(unit)
        self.events.append(('release', unit))

    def finish_backward(self, unit):
        assert unit in self.ready, 'Backward finalized before gradients were ready.'
        self.ready.remove(unit)
        self.events.append(('reduce', unit))
        self.release(unit)


class TestSchedulers(unittest.TestCase):
    def test_grouped_vpp_passes_and_validation(self):
        runtime = Runtime()
        first, layer0, second, layer1 = (object() for _ in range(4))
        chunks = {
            0: Chunk((first, layer0), (first, layer0)),
            1: Chunk((second, layer1), (second, layer1)),
        }
        scheduler = VppFsdpScheduler(chunks, runtime.actions)

        def run(step):
            chunk = chunks[step.chunk]
            order = chunk.forward if step.phase == 'forward' else chunk.backward
            with scheduler.run_pass(step):
                for unit in order:
                    scheduler.demand(unit)
            # Stand in for existing post-forward/gradient-ready hooks, outside adapter logic.
            for unit in order:
                if step.phase == 'backward':
                    runtime.ready.add(unit)
                    runtime.finish_backward(unit)
                else:
                    runtime.release(unit)

        with scheduler.iteration():
            run(Step(0, 0, 'forward'))
            self.assertNotIn(('gather', second), runtime.events)
            run(Step(0, 1, 'forward'))
            run(Step(1, 0, 'forward'))
            run(Step(1, 0, 'backward'))
            run(Step(1, 1, 'forward'))
            run(Step(0, 0, 'backward'))
            run(Step(1, 1, 'backward'))
            run(Step(0, 1, 'backward'))
        waits = [unit for action, unit in runtime.events if action == 'wait']
        self.assertEqual(waits[:6], [first, layer0, first, layer0, second, layer1])
        with scheduler.iteration(forward_only=True):
            run(Step(1, 0, 'forward'))
            run(Step(0, 0, 'forward'))
        with scheduler.iteration():
            run(Step(0, 0, 'forward'))
            run(Step(0, 0, 'backward'))
        self.assertFalse(runtime.loaded)

    def test_combined_nodes_keep_delayed_wgrad_explicit(self):
        runtime = Runtime()
        first, second = object(), object()
        scheduler = Combined1f1bFsdpScheduler(runtime.actions)
        nodes = [
            Node(Step(1, 0, 'forward'), 'B.forward', demand=(second,), release=(second,)),
            Node(Step(0, 1, 'forward'), 'A.attention', demand=(first,), prefetch=(second,)),
            Node(Step(1, 0, 'backward'), 'B.mlp', demand=(second,)),
            Node(Step(0, 1, 'forward'), 'A.mlp', demand=(first,), release=(first,)),
            Node(Step(1, 0, 'backward'), 'B.attention', demand=(second,)),
            Node(Step(0, 1, 'backward'), 'A.backward', demand=(first,), finish_backward=(first,)),
            Node(Step(1, 0, 'weight_grad'), 'B.wgrad', finish_backward=(second,)),
        ]
        with scheduler.iteration():
            for node in nodes:
                with scheduler.run_node(node):
                    runtime.events.append(('compute', node.name))
                    runtime.ready.update(node.finish_backward)
                    # The adapter must not reduce just because a backward node returned.
                    if node.name == 'B.attention':
                        self.assertNotIn(('reduce', second), runtime.events)
        self.assertLess(
            runtime.events.index(('gather', second), 3), runtime.events.index(('compute', 'B.mlp'))
        )
        self.assertGreater(
            runtime.events.index(('reduce', second)), runtime.events.index(('compute', 'B.wgrad'))
        )
        self.assertEqual(runtime.events.count(('reduce', second)), 1)
        self.assertFalse(runtime.loaded)

    def test_invalid_windows_fail_before_unrelated_actions(self):
        unit = object()
        runtime = Runtime()
        vpp = VppFsdpScheduler({0: Chunk((unit,), (unit,))}, runtime.actions)
        with self.assertRaisesRegex(RuntimeError, 'completed forward'):
            with vpp.iteration():
                with vpp.run_pass(Step(0, 0, 'backward')):
                    pass
        with self.assertRaisesRegex(RuntimeError, 'Unexpected FSDP'):
            with vpp.iteration(forward_only=True):
                with vpp.run_pass(Step(0, 0, 'forward')):
                    vpp.demand(object())
        self.assertEqual(runtime.events, [])
        with self.assertRaisesRegex(RuntimeError, 'all units'):
            with vpp.iteration(forward_only=True):
                with vpp.run_pass(Step(0, 0, 'forward')):
                    pass
        combined = Combined1f1bFsdpScheduler(runtime.actions)
        with self.assertRaisesRegex(RuntimeError, 'backward finalization'):
            with combined.iteration():
                with combined.run_node(Node(Step(0, 0, 'backward'), 'unfinished', demand=(unit,))):
                    pass
        self.assertNotIn(('reduce', unit), runtime.events)
        with self.assertRaisesRegex(RuntimeError, 'previous backward window'):
            with combined.iteration():
                with combined.run_node(Node(Step(0, 0, 'backward'), 'deferred', demand=(unit,))):
                    pass
                with combined.run_node(Node(Step(0, 1, 'backward'), 'overlapping', demand=(unit,))):
                    pass
        with self.assertRaisesRegex(ValueError, 'finalizer owns'):
            with combined.iteration():
                with combined.run_node(
                    Node(
                        Step(0, 0, 'backward'),
                        'duplicate release',
                        release=(unit,),
                        finish_backward=(unit,),
                    )
                ):
                    pass


if __name__ == '__main__':
    unittest.main()
