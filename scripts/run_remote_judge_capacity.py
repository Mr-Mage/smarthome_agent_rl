"""Background end-to-end capacity pilots; own H100 services, never the external judge."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, Thread
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import execution_slots, external_judge, judge_endpoint
from scripts.benchmark_capacity import pressure


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def queue_peaks(samples):
    peaks = {'actor_running': 0.0, 'actor_waiting': 0.0, 'judge_running': 0.0, 'judge_waiting': 0.0}
    for sample in samples:
        totals = dict.fromkeys(peaks, 0.0)
        for name, lines in sample.items():
            if not isinstance(lines, list):
                continue
            role = 'actor' if name.startswith('actor') else 'judge' if name == 'judge' else None
            if role is None:
                continue
            for line in lines:
                for metric, field in (('num_requests_running', 'running'), ('num_requests_waiting', 'waiting')):
                    if metric in line:
                        totals[role + '_' + field] += float(line.rsplit(' ', 1)[1])
        peaks = {key: max(peaks[key], totals[key]) for key in peaks}
    return peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/remote-judge-capacity.json')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    plan = config['capacity_pilot']
    levels = plan['concurrency']
    actors = len(config['workflows'])
    if not external_judge(config) or actors != 4:
        raise ValueError('Capacity pilot requires four actors and an external judge')
    if len(set(levels)) != len(levels) or any(n % actors or not actors <= n <= 32 * actors for n in levels):
        raise ValueError('Distinct levels must allocate 1-32 slots per actor')
    service_config = copy.deepcopy(config)
    service_config['slots_per_actor'] = max(levels) // actors
    execution_slots(service_config)
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    service_dir = run / 'services'
    service_dir.mkdir()
    runtime_path = run / 'service-config.json'
    write(runtime_path, service_config)
    manifest = ROOT / plan['manifest']
    tasks = json.loads(manifest.read_text())['tasks']
    expected = len(tasks) * len(plan['variants']) * len(config['actor_seeds'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    write(run / 'protocol.json', {'purpose': 'capacity_diagnostic_only', 'commit': commit,
        'config': config, 'config_sha256': digest(config_path), 'manifest_sha256': digest(manifest),
        'tasks': len(tasks), 'episodes_per_level': expected, 'order': levels,
        'selection': 'Highest verified episodes/minute; all startup and monitoring costs retained',
        'quality_results': 'Diagnostics only; original real-time simulator retained; no final tuning',
        'judge_managed': False})
    (run / 'driver.pid').write_text(str(os.getpid()))
    command_log = (run / 'commands.jsonl').open('a')
    supervisor = None
    reports = []
    began = time.monotonic()

    def state(stage, **extra):
        value = {'stage': stage, 'pid': os.getpid(), 'at': time.time(), **extra}
        write(run / 'state.json', value)
        print(json.dumps(value), flush=True)

    def command(script, *arguments):
        argv = [sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)]
        before = time.monotonic()
        result = subprocess.run(argv, cwd=ROOT)
        command_log.write(json.dumps({'argv': argv, 'seconds': time.monotonic() - before,
            'exit_code': result.returncode}) + '\n')
        command_log.flush()
        if result.returncode:
            raise RuntimeError(f'{script} exited {result.returncode}')

    def snapshot():
        result = {'at': time.time()}
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        result['gpu'] = gpu.stdout.strip()
        urls = [(f"actor{w['id']}", f"http://127.0.0.1:{w['actor_port']}") for w in config['workflows']]
        urls.append(('judge', judge_endpoint(config).removesuffix('/v1')))
        def fetch(pair):
            name, url = pair
            try:
                with httpx.Client(trust_env=False, timeout=5) as client:
                    response = client.get(url + '/metrics')
                    response.raise_for_status()
                return name, [line for line in response.text.splitlines() if not line.startswith('#') and
                    any(key in line for key in ('num_requests_running', 'num_requests_waiting',
                                               'cache_usage_perc', 'num_preemptions'))]
            except Exception as exc:
                return name, {'error': str(exc)}
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            result.update(pool.map(fetch, urls))
        return result

    def monitor(directory, stop, samples):
        with (directory / 'metrics.jsonl').open('w') as output:
            while not stop.is_set():
                try:
                    sample = snapshot()
                except Exception as exc:
                    sample = {'at': time.time(), 'monitor_error': str(exc)}
                samples.append(sample)
                output.write(json.dumps(sample) + '\n')
                output.flush()
                stop.wait(10)

    try:
        with (service_dir / 'supervisor.log').open('w') as log:
            supervisor = subprocess.Popen([sys.executable, ROOT / 'scripts/harness_services.py', 'start',
                '--config', runtime_path, '--run-dir', service_dir], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state('waiting_services', supervisor=supervisor.pid)
        deadline = time.monotonic() + config['startup_timeout']
        while not (service_dir / 'ready').exists():
            if supervisor.poll() is not None or (service_dir / 'failure.json').exists():
                raise RuntimeError('H100 services failed; inspect services/supervisor.log')
            if time.monotonic() > deadline:
                raise TimeoutError('H100 service readiness timeout')
            time.sleep(3)
        command('harness_services.py', 'probe', '--config', runtime_path, '--run-dir', service_dir)
        startup = time.monotonic() - began
        for level in levels:
            directory = run / f'pilot-{level}'
            directory.mkdir()
            current = copy.deepcopy(config)
            current['slots_per_actor'] = level // actors
            current_path = directory / 'config.json'
            write(current_path, current)
            samples, stop = [], Event()
            thread = Thread(target=monitor, args=(directory, stop, samples), daemon=True)
            thread.start()
            state('episode_pilot', concurrency=level, expected=expected)
            before = time.monotonic()
            try:
                command('run_benchmark_suite.py', '--phase', 'smoke', '--variants', *plan['variants'],
                    '--manifest', manifest, '--config', current_path, '--run-dir', directory / 'episodes')
                command('report_benchmark.py', directory / 'episodes')
                command('verify_benchmark.py', directory / 'episodes')
            finally:
                stop.set()
                thread.join(timeout=30)
            completion = json.loads((directory / 'episodes/completion.json').read_text())
            report = json.loads((directory / 'episodes/report.json').read_text())
            if not report['verified'] or completion['episodes'] != expected:
                raise ValueError('Pilot coverage or verification failed')
            elapsed = completion['elapsed_seconds']
            row = {'concurrency': level, 'verified': True, 'episodes': expected,
                'elapsed_seconds': elapsed, 'startup_seconds': completion['startup_seconds'],
                'execution_seconds': elapsed - completion['startup_seconds'],
                'episodes_per_minute': expected / elapsed * 60,
                'execution_episodes_per_minute': expected / (elapsed - completion['startup_seconds']) * 60,
                'total_stage_seconds': time.monotonic() - before,
                **pressure([{k: v for k, v in sample.items() if isinstance(v, list)} for sample in samples]),
                'request_queue_peaks': queue_peaks(samples),
                'monitor_errors': sum('monitor_error' in s or any(isinstance(v, dict) and 'error' in v
                                     for v in s.values()) for s in samples),
                'arms': report['arms']}
            reports.append(row)
            write(run / 'pilot-report.json', reports)
        selected = max(reports, key=lambda row: row['episodes_per_minute'])
        write(run / 'report.json', {'verified': True, 'pilots': reports,
            'selected_concurrency': selected['concurrency'], 'shared_model_startup_seconds': startup,
            'total_seconds_before_cleanup': time.monotonic() - began,
            'limitation': 'One ordered sweep; 144 episodes per level; capacity and live-clock diagnostics only'})
        state('complete', selected_concurrency=selected['concurrency'])
    except Exception as exc:
        write(run / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'completed_levels': reports})
        state('failed', error=str(exc))
        raise
    finally:
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=120)
        write(run / 'driver-timing.json', {'total_seconds_including_cleanup': time.monotonic() - began,
                                        'external_judge_stopped': False})
        command_log.close()


if __name__ == '__main__':
    main()
