"""Replay unchanged development HTTP requests; this is throughput, not task evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from threading import Lock, Event, Thread
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def corpus(source):
    groups = {}
    for path in sorted(source.glob('worker*/*/B0/lightning/model_calls.json')):
        records = json.loads(path.read_text())
        category = path.parents[2].name.rsplit('_seed_', 1)[0]
        for record in records:
            if record['status'] == 200:
                groups.setdefault(category, []).append(record)
    selected = []
    for category, records in sorted(groups.items()):
        records.sort(key=lambda row: row['response']['usage']['prompt_tokens'])
        for quantile in (0, .5, .95):
            record = records[int((len(records) - 1) * quantile)]
            body = record['request']
            selected.append({'category': category, 'quantile': quantile, 'request': body,
                'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                'historical_usage': record['response']['usage']})
    if len(groups) != 12:
        raise ValueError('Need all twelve development categories')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', default='runs/harness-mvp/primary-v1/dev')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--concurrency', nargs='+', type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument('--seconds', type=int, default=180)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    packets = corpus(ROOT / args.source)
    (run / 'corpus.json').write_text(json.dumps(packets, ensure_ascii=False))
    (run / 'protocol.json').write_text(json.dumps({'config': config, 'source': args.source,
        'seconds': args.seconds, 'concurrency': args.concurrency, 'purpose': 'fixed_request_capacity_only',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}, indent=2))
    endpoints = [f"http://127.0.0.1:{w['actor_port']}" for w in config['workflows']]
    def snapshot():
        data = {'at': time.time()}
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        data['gpu'] = result.stdout.strip()
        with httpx.Client(trust_env=False, timeout=10) as client:
            for i, url in enumerate(endpoints):
                response = client.get(url + '/metrics')
                response.raise_for_status()
                data[f'actor{i}'] = [line for line in response.text.splitlines() if not line.startswith('#')
                    and any(key in line for key in ('num_requests_running', 'num_requests_waiting',
                        'cache_usage_perc', 'num_preemptions', 'prefix_cache_hits', 'prefix_cache_queries'))]
        return data
    # Warm each engine with unchanged packets before starting the timed sweep.
    def warm(url):
        with httpx.Client(trust_env=False, timeout=300) as client:
            response = client.post(url + '/v1/chat/completions', json=packets[0]['request'])
            response.raise_for_status()
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        list(pool.map(warm, endpoints))
    summaries = []
    for concurrency in args.concurrency:
        if concurrency < len(endpoints) or concurrency % len(endpoints):
            raise ValueError('Concurrency must be a positive multiple of actor count')
        directory = run / str(concurrency)
        directory.mkdir()
        lock, failed, monitor_stop = Lock(), Event(), Event()
        records, samples, sequence = [], [], 0
        started = time.monotonic()
        deadline = started + args.seconds
        def monitor():
            while not monitor_stop.is_set():
                try:
                    samples.append(snapshot())
                except Exception as exc:
                    samples.append({'monitor_error': str(exc), 'at': time.time()})
                monitor_stop.wait(5)
        thread = Thread(target=monitor, daemon=True)
        thread.start()
        def worker(worker_id):
            nonlocal sequence
            url = endpoints[worker_id % len(endpoints)]
            with httpx.Client(trust_env=False, timeout=300) as client:
                while time.monotonic() < deadline and not failed.is_set():
                    with lock:
                        index = sequence
                        sequence += 1
                    packet = packets[index % len(packets)]
                    before = time.monotonic()
                    record = {'sequence': index, 'actor': worker_id % len(endpoints),
                        'corpus_index': index % len(packets), 'request_sha256': packet['request_sha256'],
                        'started_seconds': before - started}
                    try:
                        response = client.post(url + '/v1/chat/completions', json=packet['request'])
                        response.raise_for_status()
                        result = response.json()
                        record.update({'usage': result['usage'], 'choices': result['choices'], 'status': response.status_code})
                    except Exception as exc:
                        record['error'] = str(exc)
                        failed.set()
                    record['duration_seconds'] = time.monotonic() - before
                    record['finished_seconds'] = time.monotonic() - started
                    with lock:
                        records.append(record)
                        with (directory / 'responses.jsonl').open('a') as output:
                            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(worker, range(concurrency)))
        elapsed = time.monotonic() - started
        monitor_stop.set()
        thread.join(timeout=15)
        (directory / 'samples.json').write_text(json.dumps(samples, indent=2))
        successful = [r for r in records if 'error' not in r]
        latencies = sorted(r['duration_seconds'] for r in successful)
        summary = {'concurrency': concurrency, 'elapsed_seconds': elapsed, 'measurement_seconds': args.seconds,
            'drain_seconds': max(0, elapsed - args.seconds), 'requests': len(records),
            'errors': len(records) - len(successful), 'requests_per_second': len(successful) / elapsed,
            'input_tokens_per_second': sum(r['usage']['prompt_tokens'] for r in successful) / elapsed,
            'output_tokens_per_second': sum(r['usage']['completion_tokens'] for r in successful) / elapsed,
            'latency_p50': statistics.median(latencies) if latencies else None,
            'latency_p95': latencies[int((len(latencies) - 1) * .95)] if latencies else None}
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        (run / 'report.json').write_text(json.dumps({'purpose': 'fixed_request_capacity_only', 'stages': summaries}, indent=2))
        print(json.dumps(summary), flush=True)
        if failed.is_set():
            raise RuntimeError('Capacity stage failed; preserve evidence and do not retry requests')


if __name__ == '__main__':
    main()
