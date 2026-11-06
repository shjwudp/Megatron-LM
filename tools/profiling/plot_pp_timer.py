"""Plot PP Timer CPU/GPU intervals with microbatches and separate FSDP collectives."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

MICROBATCH = re.compile(r'^(forward|backward)-(\d+)-(\d+)(?:#\d+)?$')
ROWS = (
    'Microbatch F/B',
    'Attention F/B',
    'MLP F/B',
    'Routed experts F/B',
    'EP dispatch/combine',
    'Weight gradients',
    'HFSDP param AG',
    'HFSDP grad RS',
    'Muon gather/scatter',
    'Other optimizer comm',
)
FORWARD, BACKWARD = '#2878b5', '#d78d22'


def lane(name):
    """Keep collectives separate and avoid treating layer calls as microbatches."""
    if MICROBATCH.fullmatch(name):
        return 0
    if name.split('#')[0].endswith('-wgrad'):
        return 5
    for index, operation in ((1, 'attn'), (2, 'mlp'), (3, 'routed_experts_compute')):
        if f'-{operation}-' in name:
            return index
    if '-moe_dispatch-' in name or '-moe_combine-' in name:
        return 4
    if name.startswith('optimizer-mfsdp-param-gather'):
        return 6
    if name.startswith('optimizer-mfsdp-grads-reduce-scatter'):
        return 7
    if name.startswith('optimizer-muon-'):
        return 8
    if name.startswith('optimizer-') and not name.startswith('optimizer-megatron-step'):
        return 9
    return None


def intervals(events):
    """Pair endpoints, rejecting malformed ranges instead of drawing partial data."""
    for name in sorted({key.rsplit('/', 1)[0] for key in events}):
        start, end = events.get(f'{name}/start'), events.get(f'{name}/end')
        if start is None or end is None or end < start:
            raise ValueError(f'Incomplete or negative range: {name}')
        yield name, start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--iteration', type=int, required=True, help='Zero-based timer iteration')
    parser.add_argument('--ranks', type=int, nargs='+', default=[0, 4])
    parser.add_argument('--output', type=Path, required=True, help='PNG, SVG or PDF output')
    parser.add_argument('--start-ms', type=float, default=0)
    parser.add_argument('--end-ms', type=float)
    args = parser.parse_args()
    data = []
    for rank in args.ranks:
        path = args.directory / f'event_times_iter{args.iteration}_rank_{rank}.json'
        payload = json.loads(path.read_text())
        cpu_path = path.with_name(path.name.replace('event_times_', 'event_times_cpu_', 1))
        cpu = (
            json.loads(cpu_path.read_text())
            if payload.get('cpu_enabled', True) and cpu_path.exists()
            else {}
        )
        data.append((rank, payload, cpu))
    reference = min(payload['reference_unix_ms'] for _, payload, _ in data)
    end_ms = args.end_ms
    if end_ms is None:
        end_ms = max(
            payload['reference_unix_ms'] - reference + payload['events']['iteration/end']
            for _, payload, _ in data
        )
    if end_ms <= args.start_ms:
        parser.error('--end-ms must be larger than --start-ms')

    fig, axes = plt.subplots(
        len(data),
        1,
        figsize=(15, 4.7 * len(data)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (rank, payload, cpu) in zip(axes[:, 0], data):
        shift = payload['reference_unix_ms'] - reference
        counts = [0] * len(ROWS)
        for events, offset, alpha in ((payload['events'], -0.24, 1.0), (cpu, 0.08, 0.35)):
            for name, start, end in intervals(events):
                row = lane(name)
                if row is None:
                    continue
                if alpha == 1.0:
                    counts[row] += 1
                start, end = start + shift, end + shift
                if end < args.start_ms or start > end_ms:
                    continue
                color = BACKWARD if 'backward' in name or 'wgrad' in name else FORWARD
                if row >= 6:
                    color = ('#7954ad', '#bf4f54', '#4d9a74', '#6e7580')[row - 6]
                match = MICROBATCH.fullmatch(name)
                bar_offset, height = offset, 0.24
                if match:
                    # Combined F/B spans cover the same scheduling call. Separate
                    # directions vertically so neither hides the other.
                    bar_offset = -0.42 if match[1] == 'forward' else 0.04
                    bar_offset += 0.20 if alpha != 1.0 else 0
                    height = 0.17
                ax.broken_barh(
                    [(start / 1000, (end - start) / 1000)],
                    (row + bar_offset, height),
                    facecolors=color,
                    edgecolors='white' if row == 0 else color,
                    linewidth=0.5 if row == 0 else 0.35,
                    alpha=alpha,
                )
                if match and alpha == 1.0:
                    left, right = max(start, args.start_ms), min(end, end_ms)
                    if right - left > 0.012 * (end_ms - args.start_ms):
                        direction, chunk, microbatch = match.groups()
                        ax.text(
                            (left + right) / 2000,
                            row + bar_offset + height / 2,
                            f'{direction[0].upper()}{chunk}:{microbatch}',
                            ha='center',
                            va='center',
                            color='white',
                            fontsize=7,
                        )
        ax.set_yticks(range(len(ROWS)), [f'{label} ({n})' for label, n in zip(ROWS, counts)])
        ax.set_ylim(len(ROWS) - 0.5, -0.6)
        ax.set_xlim(args.start_ms / 1000, end_ms / 1000)
        ax.set_title(f'Rank {rank} | counts include all GPU ranges in this iteration', loc='left')
        ax.grid(axis='x', alpha=0.2)
    axes[-1, 0].set_xlabel('Seconds from earliest rank reference (approximate clock alignment)')
    fig.suptitle(
        f'Timer iteration {args.iteration} | dark: GPU; light: CPU | '
        'blue: forward; orange: backward\nF/B labels = direction + chunk:microbatch'
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
