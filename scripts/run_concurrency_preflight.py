"""Measure inference and isolated episode throughput before freezing a timed rerun."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/concurrency-capacity/preflight-v1')
    parser.add_argument('--config', default='configs/concurrency-capacity.json')
    parser.add_argument('--attach-services', help='Existing authorized eager supervisor to take ownership of')
    args = parser.parse_args()
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / args.config).read_text())
    log = (run / 'commands.jsonl').open('a')
    def state(stage, **extra):
        value = {'stage': stage, 'pid': os.getpid(), 'at': time.time(), **extra}
        temporary = run / 'state.tmp'
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(run / 'state.json')
        print(json.dumps(value), flush=True)
    def command(script, *arguments):
        argv = [sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)]
        before = time.monotonic()
        result = subprocess.run(argv, cwd=ROOT)
        log.write(json.dumps({'argv': argv, 'seconds': time.monotonic() - before,
            'exit_code': result.returncode}) + '\n')
        log.flush()
        if result.returncode:
            raise RuntimeError(f'Command failed: {script}')
    supervisor = None
    service_dir = None
    def stop():
        nonlocal supervisor
        if supervisor is not None:
            try:
                os.kill(supervisor, signal.SIGTERM)
            except ProcessLookupError:
                supervisor = None
                return
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                try:
                    os.kill(supervisor, 0)
                    stat = Path(f'/proc/{supervisor}/stat')
                    if stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'):
                        break
                except ProcessLookupError:
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Supervisor did not stop; do not start colliding services')
            supervisor = None
    def start(label, current, attached=None):
        nonlocal supervisor, service_dir
        service_dir = ROOT / attached if attached else run / label / 'services'
        config_path = run / label / 'config.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(current, indent=2))
        if attached:
            supervisor = int((service_dir / 'supervisor.pid').read_text())
            cmdline = Path(f'/proc/{supervisor}/cmdline').read_bytes()
            if b'harness_services.py' not in cmdline or str(service_dir.relative_to(ROOT)).encode() not in cmdline:
                raise RuntimeError('Attached supervisor identity mismatch')
        else:
            service_dir.mkdir(parents=True)
            with (service_dir / 'supervisor.log').open('w') as output:
                process = subprocess.Popen([sys.executable, ROOT / 'scripts/harness_services.py', 'start',
                    '--config', config_path, '--run-dir', service_dir], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            supervisor = process.pid
        state('waiting_services', label=label, supervisor=supervisor)
        deadline = time.monotonic() + current['startup_timeout']
        while not (service_dir / 'ready').exists():
            if (service_dir / 'failure.json').exists():
                raise RuntimeError((service_dir / 'failure.json').read_text())
            os.kill(supervisor, 0)
            status = Path(f'/proc/{supervisor}/stat')
            if not status.exists() or status.read_text().split(') ', 1)[1].startswith('Z'):
                raise RuntimeError('Service supervisor exited before readiness: ' +
                    (service_dir / 'supervisor.log').read_text(errors='replace')[-3000:])
            if time.monotonic() > deadline:
                raise TimeoutError('Services did not become ready')
            time.sleep(3)
        command('harness_services.py', 'probe', '--config', config_path, '--run-dir', service_dir)
        return config_path
    stages = []
    try:
        eager_path = start('eager', config, args.attach_services)
        state('capacity', label='eager')
        command('benchmark_capacity.py', '--config', eager_path, '--run-dir', run / 'eager/replay', '--try64')
        eager = json.loads((run / 'eager/replay/report.json').read_text())['stages']
        best = max(eager, key=lambda row: row['output_tokens_per_second'])
        levels = sorted({max(2, best['concurrency'] // 2), best['concurrency']})
        stages.append({'label': 'eager', 'config': copy.deepcopy(config), 'capacity': best})
        stop()
        for label, graph in [('prefix', False), ('graph-prefix', True)]:
            candidate = copy.deepcopy(config)
            candidate['inference'].update(actor_prefix_caching=True, actor_enforce_eager=not graph)
            try:
                current_path = start(label, candidate)
                state('capacity', label=label, levels=levels)
                command('benchmark_capacity.py', '--config', current_path, '--run-dir', run / label / 'replay',
                    '--concurrency', *levels)
                rows = json.loads((run / label / 'replay/report.json').read_text())['stages']
                stages.append({'label': label, 'config': candidate,
                    'capacity': max(rows, key=lambda row: row['output_tokens_per_second'])})
            except Exception as exc:
                (run / label / 'rejected.json').write_text(json.dumps({'error': str(exc),
                    'eligible': False, 'config': candidate}, indent=2))
                state('candidate_rejected', label=label, error=str(exc))
            finally:
                stop()
        # Only successful configurations are eligible; all measured stages stay in their own directories.
        stages.sort(key=lambda row: row['capacity']['output_tokens_per_second'], reverse=True)
        if stages[0]['capacity']['output_tokens_per_second'] < 770 or stages[0]['capacity']['input_tokens_per_second'] < 19800:
            # Test the same judge weights/context on one H100; an OOM rejects this allocation.
            candidate = copy.deepcopy(stages[0]['config'])
            actor = {**candidate['workflows'][0], 'id': 2, 'gpus': [2],
                'actor_port': 20002, 'simulator_port': 20082, 'gateway_port': 20183}
            candidate['workflows'].append(actor)
            candidate['judge_gpus'] = [3]
            candidate['inference'].update(judge_gpu_memory_utilization=.94, judge_max_num_seqs=12)
            cache = ROOT / candidate['kernel_cache_root']
            import shutil
            shutil.copytree(cache / 'actor0', cache / 'actor2', dirs_exist_ok=True)
            try:
                current_path = start('three-actors', candidate)
                state('capacity', label='three-actors', levels=[6, 12, 24, 48])
                command('benchmark_capacity.py', '--config', current_path, '--run-dir', run / 'three-actors/replay',
                    '--concurrency', 6, 12, 24, 48)
                rows = json.loads((run / 'three-actors/replay/report.json').read_text())['stages']
                stages.append({'label': 'three-actors', 'config': candidate,
                    'capacity': max(rows, key=lambda row: row['output_tokens_per_second'])})
            except Exception as exc:
                (run / 'three-actors/rejected.json').write_text(json.dumps({'error': str(exc),
                    'eligible': False, 'config': candidate}, indent=2))
                state('candidate_rejected', label='three-actors', error=str(exc))
            finally:
                stop()
            stages.sort(key=lambda row: row['capacity']['output_tokens_per_second'], reverse=True)
        chosen = stages[0]
        actor_count = len(chosen['config']['workflows'])
        pilot_levels = sorted({max(actor_count, chosen['capacity']['concurrency'] // 2), chosen['capacity']['concurrency']})
        pilot_config = copy.deepcopy(chosen['config'])
        pilot_config['slots_per_actor'] = max(pilot_levels) // len(pilot_config['workflows'])
        current_path = start('pilot-services', pilot_config)
        pilot_reports = []
        for level in pilot_levels:
            current = copy.deepcopy(pilot_config)
            current['slots_per_actor'] = level // len(current['workflows'])
            path = run / f'pilot-{level}/config.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(current, indent=2))
            directory = run / f'pilot-{level}/episodes'
            state('episode_pilot', concurrency=level)
            command('run_benchmark_suite.py', '--phase', 'smoke', '--variants', *current['variants_dev'],
                '--config', path, '--run-dir', directory)
            command('report_benchmark.py', directory)
            command('verify_benchmark.py', directory)
            completion = json.loads((directory / 'completion.json').read_text())
            report = json.loads((directory / 'report.json').read_text())
            elapsed = completion['elapsed_seconds']
            pilot_reports.append({'concurrency': level, 'elapsed_seconds': elapsed,
                'episodes_per_minute': completion['episodes'] / elapsed * 60,
                'predicted_dev_final_seconds': elapsed * (1368 / 120),
                'arms': {k: {'successes': v['successes'], 'episodes': v['episodes'],
                    'unfinished': v['unfinished']} for k, v in report['arms'].items()}, 'config': current})
            (run / 'pilot-report.json').write_text(json.dumps(pilot_reports, indent=2))
        selected = min(pilot_reports, key=lambda row: row['predicted_dev_final_seconds'])
        stop()
        output = {'inference_stages': stages, 'pilots': pilot_reports, 'selected': selected,
            'within_50_minute_projection': selected['predicted_dev_final_seconds'] <= 3000,
            'actual_full_run_required': True}
        (run / 'report.json').write_text(json.dumps(output, indent=2))
        (run / 'selected-config.json').write_text(json.dumps(selected['config'], indent=2))
        state('preflight_complete', projected_minutes=selected['predicted_dev_final_seconds'] / 60)
    except Exception as exc:
        state('failed', error=str(exc))
        raise
    finally:
        stop()
        log.close()


if __name__ == '__main__':
    main()
