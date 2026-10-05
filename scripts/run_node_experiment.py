"""Run frozen development stages with four actors and permanent external judge."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import execution_slots, external_judge, judge_endpoint
from smarthome_agent_rl.node_selection import select_guard, select_time_plan
from benchmark_capacity import pressure
from run_remote_judge_capacity import queue_peaks, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    slots = execution_slots(config)
    if len(slots) != 64 or len(config['workflows']) != 4 or not external_judge(config):
        raise ValueError('Node execution requires four actors, 64 slots and external judge')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit the frozen experiment before execution')
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    services = run / 'services'
    services.mkdir()
    write(run / 'protocol.json', {'commit': commit, 'config': config, 'config_sha256': digest(config_path),
        'stages': [{**s, 'manifest_sha256': digest(ROOT / s['manifest'])} for s in config['node_experiment']['stages']],
        'exclusions': 'None; task failures retained. Infrastructure failure invalidates entire stage; no selective reruns.',
        'resources': {'actors': 4, 'total_slots': 64, 'judge_managed': False},
        'gates': config['node_experiment']['gates']})
    (run / 'driver.pid').write_text(str(os.getpid()))
    began, supervisor = time.monotonic(), None
    timings, samples = [], []
    stop = Event()
    thread = None

    def state(stage, **extra):
        value = {'stage': stage, 'pid': os.getpid(), 'at': time.time(), **extra}
        write(run / 'state.json', value)
        print(json.dumps(value), flush=True)

    def command(script, *values):
        argv = [sys.executable, str(ROOT / 'scripts' / script), *map(str, values)]
        started = time.monotonic()
        result = subprocess.run(argv, cwd=ROOT)
        timings.append({'argv': argv, 'seconds': time.monotonic() - started, 'exit_code': result.returncode})
        write(run / 'commands.json', timings)
        if result.returncode:
            raise RuntimeError(f'{script} exited {result.returncode}')

    def monitor():
        urls = [(f"actor{w['id']}", f"http://127.0.0.1:{w['actor_port']}") for w in config['workflows']]
        urls.append(('judge', judge_endpoint(config).removesuffix('/v1')))
        def fetch(pair):
            name, url = pair
            try:
                with httpx.Client(trust_env=False, timeout=5) as client:
                    response = client.get(url + '/metrics')
                    response.raise_for_status()
                return name, [l for l in response.text.splitlines() if not l.startswith('#') and
                    any(k in l for k in ('num_requests_running', 'num_requests_waiting', 'cache_usage_perc', 'num_preemptions'))]
            except Exception as exc:
                return name, {'error': str(exc)}
        with (run / 'metrics.jsonl').open('w') as output:
            while not stop.is_set():
                sample = {'at': time.time()}
                try:
                    sample['stage'] = json.loads((run / 'state.json').read_text())['stage']
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
                    sample['gpu'] = gpu.stdout.strip()
                    with ThreadPoolExecutor(max_workers=5) as pool:
                        sample.update(pool.map(fetch, urls))
                except Exception as exc:
                    sample['monitor_error'] = str(exc)
                samples.append(sample)
                output.write(json.dumps(sample) + '\n')
                output.flush()
                stop.wait(10)

    try:
        with (services / 'supervisor.log').open('w') as log:
            supervisor = subprocess.Popen([sys.executable, ROOT / 'scripts/harness_services.py', 'start',
                '--config', config_path, '--run-dir', services], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state('waiting_services', supervisor=supervisor.pid)
        deadline = time.monotonic() + config['startup_timeout']
        while not (services / 'ready').exists():
            if supervisor.poll() is not None or (services / 'failure.json').exists():
                raise RuntimeError('H100 service startup failed; inspect supervisor log')
            if time.monotonic() > deadline:
                raise TimeoutError('H100 service startup timeout')
            time.sleep(3)
        command('harness_services.py', 'probe', '--config', config_path, '--run-dir', services)
        if config['node_experiment'].get('preflight_episode'):
            command('preflight_time_plan.py', '--config', config_path, '--output', run / 'metadata-preflight.json')
        if config['node_experiment'].get('reasoning_preflight'):
            command('preflight_reasoning.py','--config',config_path,'--output',run/'reasoning-preflight.json')
        write(run / 'startup.json', {'seconds': time.monotonic() - began})
        thread = Thread(target=monitor, daemon=True)
        thread.start()
        reports = {}
        for stage in config['node_experiment']['stages']:
            name=stage.get('name',stage['phase'])
            state(name)
            directory = run / name
            seeds=['--actor-seeds',*stage['actor_seeds']] if 'actor_seeds' in stage else []
            command('run_benchmark_suite.py', '--phase', stage['phase'], '--variants', *stage['variants'],
                '--manifest', stage['manifest'], '--config', config_path, '--run-dir', directory,*seeds)
            command('report_benchmark.py', directory)
            command('verify_benchmark.py', directory)
            reports[name] = json.loads((directory / 'report.json').read_text())
            completion = json.loads((directory / 'completion.json').read_text())
            write(run / (name + '-resources.json'), {'allocated_actor_gpu_seconds': 4 * completion['elapsed_seconds'],
                'scope': 'Four reserved H100 actors × suite elapsed; not active GPU compute or monetary cost'})
        if config['node_experiment']['node'] == 'N15':
            write(run / 'selection.json', select_guard(reports['dev'], config['node_experiment']['gates']))
        elif config['node_experiment']['node'] == 'N16':
            write(run / 'selection.json', select_time_plan(reports['dev'], config['node_experiment']['gates']))
        state('complete')
    except BaseException as exc:
        write(run / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        state('failed', error=str(exc))
        raise
    finally:
        stop.set()
        if thread:
            thread.join(timeout=30)
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=120)
        seconds = time.monotonic() - began
        gpu_values = [float(line.split(',')[1]) for s in samples for line in s.get('gpu', '').splitlines()]
        write(run / 'resource-cost.json', {'total_seconds_including_cleanup': seconds,
            'allocated_actor_gpu_seconds': 4 * seconds, 'external_judge_stopped': False,
            'monitor_errors': sum('monitor_error' in s or any(isinstance(v, dict) and 'error' in v
                                 for v in s.values()) for s in samples),
            'sampled_gpu_utilization_mean': sum(gpu_values) / len(gpu_values) if gpu_values else None,
            'sampled_gpu_utilization_peak': max(gpu_values) if gpu_values else None,
            'queue_peaks': queue_peaks(samples),
            'pressure': pressure([{k: v for k, v in s.items() if isinstance(v, list)} for s in samples]),
            'scope': 'Reserved H100 GPU wall time, including startup/failures/cleanup; excludes A800 shared-service allocation'})


if __name__ == '__main__':
    main()
