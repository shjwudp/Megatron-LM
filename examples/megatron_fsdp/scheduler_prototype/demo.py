# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Print explicit communication boundaries; this demo does not execute GPU work."""

from . import Chunk, Combined1f1bFsdpScheduler, FsdpActions, Node, Step, VppFsdpScheduler


def log(action):
    def callback(unit):
        print(f'  {action}: {unit}')

    return callback


def main():
    actions = FsdpActions(
        log('demand/wait'), log('prefetch'), log('release'), log('finish backward')
    )
    chunks = {0: Chunk(('A', 'a'), ('A', 'a')), 1: Chunk(('B', 'b'), ('B', 'b'))}
    vpp = VppFsdpScheduler(chunks, actions)
    print('Conventional VPP: the pipeline scheduler selects each pass')
    with vpp.iteration():
        for phase, chunk, microbatch in [
            ('forward', 0, 0),
            ('forward', 0, 1),
            ('forward', 1, 0),
            ('backward', 1, 0),
            ('backward', 0, 0),
            ('backward', 0, 1),
        ]:
            step = Step(chunk, microbatch, phase)
            print(step)
            units = chunks[chunk].forward if phase == 'forward' else chunks[chunk].backward
            with vpp.run_pass(step):
                for unit in units:
                    vpp.demand(unit)
                    print(f'  compute: {unit}')
            for unit in units:
                # These callbacks represent the existing runtime's completion hooks.
                actions.release(unit) if phase == 'forward' else actions.finish_backward(unit)

    print('Combined 1F1B: the compute plan selects fine-grained nodes')
    combined = Combined1f1bFsdpScheduler(actions)
    nodes = [
        Node(Step(1, 0, 'forward'), 'B.forward', demand=('B',), release=('B',)),
        Node(Step(0, 1, 'forward'), 'A.attention', demand=('A',), prefetch=('B',)),
        Node(Step(1, 0, 'backward'), 'B.mlp', demand=('B',)),
        Node(Step(0, 1, 'forward'), 'A.mlp', demand=('A',), release=('A',)),
        Node(Step(1, 0, 'backward'), 'B.attention', demand=('B',)),
        Node(Step(0, 1, 'backward'), 'A.backward', demand=('A',), finish_backward=('A',)),
        Node(Step(1, 0, 'weight_grad'), 'B.wgrad', finish_backward=('B',)),
    ]
    with combined.iteration():
        for node in nodes:
            print(node.step, node.name)
            with combined.run_node(node):
                print(f'  compute: {node.name}')


if __name__ == '__main__':
    main()
