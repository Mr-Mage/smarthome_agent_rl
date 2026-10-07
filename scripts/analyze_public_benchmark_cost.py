"""Read-only latency/cost attribution; cached logical tokens are not GPU work."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.runner import valid_usage


def distribution(values):
    values = list(values)
    if not values:
        return {'count': 0, 'p50': None, 'p95': None, 'p99': None, 'max': None}
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Latency must be finite and nonnegative')
    values.sort()
    # Nearest-rank quantiles, declared rather than library-version dependent.
    return {'count': len(values), **{f'p{q}': values[max(0, math.ceil(q / 100 * len(values)) - 1)]
            for q in (50, 95, 99)}, 'max': values[-1]}


def analyze(directory):
    paths = {name: directory / name for name in ('report.json', 'episodes.jsonl')}
    report = json.loads(paths['report.json'].read_text(encoding='utf-8'))
    if report['status'] != 'complete':
        raise ValueError('Only complete runs admit final cost analysis')
    rows = [json.loads(line) for line in paths['episodes.jsonl'].read_text(encoding='utf-8').splitlines()]
    requests = {}
    for row in rows:
        if 'shared_request' in row:
            continue
        key = row['request_receipt']
        if key in requests:
            raise ValueError('Duplicate real request attribution')
        requests[key] = row
    for row in rows:
        if 'shared_request' in row:
            original = requests.get(row['request_receipt'])
            if original is None or original['task_id'] != row['task_id'] or original['arm'] != row['shared_request']:
                raise ValueError('Shared output has no matching real request')
    if len(requests) != report['cost']['actual_actor_requests']:
        raise ValueError('Request accounting differs from experiment report')
    actual = list(requests.values())
    tokens = sum(r['usage']['total_tokens'] for r in actual if valid_usage(r.get('usage')))
    if tokens != report['cost']['total_tokens']:
        raise ValueError('Token accounting differs from experiment report')
    arms = {}
    for arm in sorted({r['arm'] for r in actual}):
        own = [r for r in actual if r['arm'] == arm]
        arms[arm] = {name: distribution([r[name] for r in own])
                     for name in ('request_seconds', 'queue_seconds', 'episode_seconds')}
        arms[arm]['logical_tokens'] = sum(r['usage']['total_tokens'] for r in own if valid_usage(r.get('usage')))
    service = report.get('service_cost', {})
    samples_path = directory.parent / (directory.name + '-services') / 'gpu-samples.jsonl'
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
    utilization = {}
    for sample in samples:
        for line in sample.get('gpus', []):
            parts = line.split(',')
            if len(parts) < 2 or not parts[1].strip().isdigit():
                continue
            utilization.setdefault(parts[0].strip(), []).append(int(parts[1]))
    duration = service.get('seconds_including_cleanup')
    return {'schema': 'public-benchmark-cost-v1', 'input': str(directory),
            'input_sha256': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()},
            'actual_requests': len(actual), 'logical_tokens': tokens,
            'missing_usage': report['cost']['missing_usage_requests'],
            'failed_requests': report['cost']['failed_actor_requests'],
            'request_seconds': distribution([r['request_seconds'] for r in actual]),
            'queue_seconds': distribution([r['queue_seconds'] for r in actual]),
            'episode_seconds': distribution([r['episode_seconds'] for r in actual]),
            'arms': arms, 'inference_stage_seconds': report['seconds'],
            'service_seconds': duration,
            'non_inference_service_seconds': None if duration is None else duration - report['seconds'],
            'startup_ready_seconds': service.get('ready_seconds'),
            'reserved_h100_gpu_seconds': service.get('reserved_h100_gpu_seconds'),
            'startup_probes': service.get('probes'),
            'sampled_gpu_utilization': {g: {'samples': len(v), 'mean_percent': sum(v) / len(v), 'max_percent': max(v)}
                                        for g, v in utilization.items()},
            'quantiles': 'nearest rank; seconds; per-request latency includes transport/prefill/decode; no phase instrumentation',
            'scope': 'read-only cost diagnostic; queue includes bulk benchmark submission; not production SLO or physical token throughput; GPU means unweighted and include startup/cleanup; shared outputs counted once; failed starts outside this run remain separately reported'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
