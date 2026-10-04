"""Time fresh service startup, complete dev/final, freezing, reporting and verification."""
import argparse
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
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Timed acceptance requires a clean committed project')
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages = []
    supervisor = None
    def state(stage, **fields):
        record = {'stage': stage, 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started, **fields}
        temporary = run / 'state.tmp'
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(run / 'state.json')
        print(json.dumps(record), flush=True)
    def command(label, script, *arguments):
        state(label)
        before = time.monotonic()
        subprocess.run([sys.executable, ROOT / 'scripts' / script, *map(str, arguments)], cwd=ROOT, check=True)
        stages.append({'stage': label, 'seconds': time.monotonic() - before})
    service_dir = run / 'services'
    service_dir.mkdir()
    try:
        with (service_dir / 'supervisor.log').open('w') as output:
            supervisor = subprocess.Popen([sys.executable, ROOT / 'scripts/harness_services.py', 'start',
                '--config', config_path, '--run-dir', service_dir], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        state('service_startup', supervisor=supervisor.pid)
        deadline = time.monotonic() + config['startup_timeout']
        while not (service_dir / 'ready').exists():
            if supervisor.poll() is not None:
                raise RuntimeError('Fresh model service supervisor exited')
            if time.monotonic() > deadline:
                raise TimeoutError('Fresh service startup timed out')
            time.sleep(3)
        stages.append({'stage': 'service_startup', 'seconds': time.monotonic() - started})
        command('inference_probe', 'harness_services.py', 'probe', '--config', config_path, '--run-dir', service_dir)
        # Re-hash identities for this acceptance; never replace the historical inventories.
        command('model_inventory', 'harness_services.py', 'inventory', '--config', config_path, '--run-dir', run / 'inventory')
        command('dev600', 'run_benchmark_suite.py', '--phase', 'dev', '--config', config_path, '--run-dir', run / 'dev')
        command('dev_report', 'report_benchmark.py', run / 'dev')
        command('dev_verify', 'verify_benchmark.py', run / 'dev')
        command('freeze', 'freeze_experiment.py', '--config', config_path, '--dev-run', run / 'dev',
            '--services', service_dir, '--inventory', run / 'inventory', '--output', run / 'final-freeze.json')
        command('final768', 'run_benchmark_suite.py', '--phase', 'final', '--config', config_path,
            '--run-dir', run / 'final', '--freeze', run / 'final-freeze.json')
        command('final_report', 'report_benchmark.py', run / 'final')
        command('final_verify', 'verify_benchmark.py', run / 'final')
        elapsed = time.monotonic() - started
        result = {'complete': True, 'episodes': 1368, 'elapsed_seconds': elapsed,
            'within_one_hour': elapsed <= 3600, 'timing_scope': 'fresh service start through final verification',
            'stages': stages, 'config': config, 'rerun_purpose': 'performance acceptance; original final remains historical'}
        (run / 'timing-report.json').write_text(json.dumps(result, indent=2))
        state('complete', within_one_hour=elapsed <= 3600)
    except Exception as exc:
        state('failed', error=str(exc))
        (run / 'failure.json').write_text(json.dumps({'error': str(exc), 'stages': stages,
            'elapsed_seconds': time.monotonic() - started}, indent=2))
        raise
    finally:
        if supervisor is not None and supervisor.poll() is None:
            os.kill(supervisor.pid, signal.SIGTERM)
            supervisor.wait(timeout=90)


if __name__ == '__main__':
    main()
