"""One shared four-GPU deployment across v2 smoke, module, integration and final gates."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.experiment_v2 import module_selection, integration_selection
from scripts.harness_services import validate_resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stage', choices=['modules', 'finalize'], required=True)
    parser.add_argument('--config', default='configs/harness-v2-modules.json')
    parser.add_argument('--after-acceptance', help='Wait for complete N8 acceptance and released service ports')
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Clean committed source required')
    run = ROOT / args.run_dir
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    gates_path = ROOT / 'configs/harness-v2-protocol.json'
    gates = json.loads(gates_path.read_text())
    service_dir = run / 'services'
    inventory = run / 'inventory'
    def save(path, data):
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    def state(stage, **fields):
        save(run / 'state.json', {'stage': stage, 'pid': os.getpid(), **fields})
        print(json.dumps({'stage': stage, **fields}), flush=True)
    def command(script, *arguments):
        subprocess.run([sys.executable, ROOT / 'scripts' / script, *map(str, arguments)], cwd=ROOT, check=True)
    def suite(phase, path, cfg, *extra):
        state(path.name)
        command('run_benchmark_suite.py', '--phase', phase, '--run-dir', path, '--config', cfg, *extra)
        command('report_benchmark.py', path)
        command('verify_benchmark.py', path)
        return json.loads((path / 'report.json').read_text())
    def stop_services():
        pid_path = service_dir / 'supervisor.pid'
        if pid_path.exists():
            pid = int(pid_path.read_text())
            proc = Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'harness_services.py' in proc.read_bytes():
                os.kill(pid, signal.SIGTERM)
    inventory_process = None
    try:
        if args.stage == 'modules':
            if args.after_acceptance:
                acceptance = ROOT / args.after_acceptance
                while not (acceptance / 'timing-report.json').exists():
                    if (acceptance / 'failure.json').exists():
                        raise RuntimeError('N8 acceptance failed; v2 was not started')
                    owner = json.loads((acceptance / 'state.json').read_text())['pid']
                    if not Path(f'/proc/{owner}').exists():
                        raise RuntimeError('N8 owner exited without complete acceptance')
                    time.sleep(5)
                timing = json.loads((acceptance / 'timing-report.json').read_text())
                if not timing['complete'] or not timing['within_one_hour']:
                    raise ValueError('N8 has not passed the one-hour gate')
                deadline = time.monotonic() + 120
                while True:
                    try:
                        validate_resources(config)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('N8 did not release shared ports')
                        time.sleep(3)
            run.mkdir(parents=True, exist_ok=False)
            # Do not silently compete with another suite on the shared ports.
            validate_resources(config)
            save(run / 'protocol-gates.json', gates)
            service_dir.mkdir(exist_ok=True)
            with (service_dir / 'supervisor.log').open('w') as stream:
                process = subprocess.Popen([sys.executable, ROOT / 'scripts/harness_services.py', 'start',
                    '--config', str(config_path), '--run-dir', str(service_dir)], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state('service_startup', supervisor=process.pid)
            deadline = time.monotonic() + config['startup_timeout']
            while not (service_dir / 'ready').exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Service startup failed; inspect supervisor.log')
                time.sleep(3)
            command('harness_services.py', 'probe', '--config', config_path, '--run-dir', service_dir)
            with (run / 'inventory.log').open('w') as stream:
                inventory_process = subprocess.Popen([sys.executable, ROOT / 'scripts/harness_services.py',
                    'inventory', '--config', str(config_path), '--run-dir', str(inventory)], cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT)
            suite('smoke', run / 'smoke', config_path, '--variants', 'G', 'GV2', 'GC2', '--actor-seeds', '42')
            report = suite('dev', run / 'modules', config_path)
            if inventory_process.wait() != 0:
                raise RuntimeError('Model inventory failed')
            decision = module_selection(report, gates)
            save(run / 'module-selection.json', {'decision': decision, 'report_sha256': digest(run / 'modules/report.json'),
                'gates_sha256': digest(gates_path)})
            integrated = {**config, 'variants_dev': ['B0', 'G', 'Full', 'Candidate'],
                'variant_policies': {**config['variant_policies'], 'Candidate': decision['candidate_policy']}}
            save(run / 'integration-config.json', integrated)
            state('modules_complete', decision=decision, services_remain_resident=True)
        else:
            if json.loads((run / 'state.json').read_text())['stage'] != 'modules_complete':
                raise ValueError('Verified modules must precede integration')
            if json.loads((run / 'protocol-gates.json').read_text()) != gates:
                raise ValueError('Predeclared gates changed')
            selected = json.loads((run / 'module-selection.json').read_text())
            if selected['report_sha256'] != digest(run / 'modules/report.json') or selected['decision'] != module_selection(
                    json.loads((run / 'modules/report.json').read_text()), gates):
                raise ValueError('Module evidence/selection changed')
            integration_path = run / 'integration-config.json'
            integrated = json.loads(integration_path.read_text())
            expected = {**config, 'variants_dev': ['B0', 'G', 'Full', 'Candidate'],
                'variant_policies': {**config['variant_policies'], 'Candidate': selected['decision']['candidate_policy']}}
            if integrated != expected or not (service_dir / 'ready').exists():
                raise ValueError('Candidate policy or resident services changed')
            report = suite('dev', run / 'integration', integration_path)
            decision = integration_selection(report, integrated, gates)
            selection_path = run / 'integration-selection.json'
            save(selection_path, {'decision': decision, 'report_sha256': digest(run / 'integration/report.json'),
                'gates_sha256': digest(gates_path)})
            formal_path = run / 'formal-config.json'
            save(formal_path, {**integrated, 'variants_final': decision['formal_variants']})
            state('freeze', decision=decision)
            command('freeze_experiment.py', '--config', formal_path, '--dev-run', run / 'integration',
                '--services', service_dir, '--inventory', inventory, '--selection', selection_path,
                '--output', run / 'final-freeze.json')
            suite('final', run / 'final', formal_path, '--freeze', run / 'final-freeze.json')
            save(run / 'accepted.json', {'complete': True, 'decision': decision,
                'reports': ['smoke/report.json', 'modules/report.json', 'integration/report.json', 'final/report.json']})
            state('complete', decision=decision)
            stop_services()
    except Exception as exc:
        if run.exists():
            state('failed', error=str(exc))
            save(run / 'failure.json', {'error': str(exc), 'stage': args.stage})
        stop_services()
        raise
    finally:
        if inventory_process is not None and inventory_process.poll() is None:
            inventory_process.terminate()
            inventory_process.wait(timeout=30)


if __name__ == '__main__':
    main()
