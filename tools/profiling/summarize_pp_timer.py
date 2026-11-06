"""Validate PP Timer dumps and export a Chrome/Perfetto CPU + GPU timeline.

Usage: python tools/profiling/summarize_pp_timer.py DUMP_DIR --output REPORT_DIR
GPU ranges may overlap; category sums are not a decomposition of iteration time.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def category(name):
    """Group event names while retaining their full identity in the timeline."""
    for operation in (
        'routed_experts_compute',
        'shared_experts_compute',
        'moe_dispatch',
        'moe_combine',
        'attn',
        'mlp',
    ):
        if f'-{operation}' in name:
            return f'{operation}-{name.rsplit("-", 1)[-1]}'
    if name.startswith('optimizer-'):
        return name.split('#')[0]
    if name.startswith('forward-backward-'):
        return 'pipeline-pair'
    if name.startswith('forward-') and not name.startswith('forward-backward'):
        return 'microbatch-forward'
    if name.startswith('backward-'):
        return 'microbatch-backward'
    return name.split('#')[0]


def collect(directory):
    """Return a summary and timeline, reporting missing endpoints explicitly."""
    traces = []
    reports = []
    lanes = {}
    named_lanes = set()
    for path in sorted(directory.glob('event_times_iter*.json')):
        payload = json.loads(path.read_text())
        cpu_path = path.with_name(path.name.replace('event_times_', 'event_times_cpu_', 1))
        sources = {'GPU': payload['events']}
        if cpu_path.exists():
            sources['CPU'] = json.loads(cpu_path.read_text())
        report = {
            'file': path.name,
            'rank': payload['rank'],
            'iteration': payload['iteration'],
            'synchronize_ms': payload['synchronize_ms'],
            'clocks': {},
        }
        for clock, events in sources.items():
            incomplete = []
            negative = []
            counts = Counter()
            durations = defaultdict(float)
            names = sorted({name.rsplit('/', 1)[0] for name in events})
            for name in names:
                start, end = events.get(f'{name}/start'), events.get(f'{name}/end')
                if start is None or end is None:
                    incomplete.append(name)
                    continue
                if end < start:
                    negative.append(name)
                    continue
                group = category(name)
                counts[group] += 1
                durations[group] += end - start
                lane = lanes.setdefault((clock, group), len(lanes) + 1)
                if (payload['rank'], lane) not in named_lanes:
                    named_lanes.add((payload['rank'], lane))
                    traces.append(
                        {
                            'name': 'thread_name',
                            'ph': 'M',
                            'pid': payload['rank'],
                            'tid': lane,
                            'args': {'name': f'{clock} {group}'},
                        }
                    )
                traces.append(
                    {
                        'name': name,
                        'cat': group,
                        'ph': 'X',
                        'pid': payload['rank'],
                        'tid': lane,
                        'ts': (payload['reference_unix_ms'] + start) * 1000,
                        'dur': (end - start) * 1000,
                        'args': {'iteration': payload['iteration'], 'clock': clock},
                    }
                )
            report['clocks'][clock] = {
                'complete_ranges': sum(counts.values()),
                'counts': dict(counts),
                'sum_ms_including_overlap': dict(durations),
                'incomplete': incomplete,
                'negative': negative,
            }
        reports.append(report)
    return reports, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, traces = collect(args.directory)
    if not reports:
        parser.error('No event_times_iter*.json dumps found')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(reports, indent=2))
    (args.output / 'trace.json').write_text(json.dumps({'traceEvents': traces}))
    invalid = sum(
        len(clock['incomplete']) + len(clock['negative'])
        for report in reports
        for clock in report['clocks'].values()
    )
    ranges = sum(event['ph'] == 'X' for event in traces)
    print(f'{len(reports)} rank/iterations, {ranges} ranges, {invalid} invalid ranges')
    raise SystemExit(1 if invalid else 0)


if __name__ == '__main__':
    main()
