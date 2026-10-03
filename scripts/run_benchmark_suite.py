"""Two isolated Lightning workflows; each task's paired arms stay on one actor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest, schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['align', 'smoke', 'dev', 'final'], required=True)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', default='configs/harness-mvp.json')
    parser.add_argument('--freeze', help='Required frozen protocol for final')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    manifest_name = 'smoke' if args.phase in ('align', 'smoke') else args.phase
    manifest_path = ROOT / f'configs/benchmark-mvp/{manifest_name}.json'
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['tasks'][:2] if args.phase == 'align' else manifest['tasks']
    variants = args.variants or (config['variants_dev'] if args.phase == 'dev' else
        config['variants_final'] if args.phase == 'final' else ['B0'])
    if len(variants) != len(set(variants)) or set(variants) - set(config['variants_dev']):
        raise ValueError('Unknown or repeated variant')
    if args.phase == 'align' and variants != ['B0']:
        raise ValueError('Baseline alignment only')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if args.phase == 'final':
        if not args.freeze:
            raise ValueError('Final requires a frozen protocol')
        freeze = json.loads((ROOT / args.freeze).read_text())
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
        if dirty or freeze['commit'] != commit or freeze['config_sha256'] != digest(ROOT / args.config) or (
            freeze['manifest_sha256'] != digest(manifest_path)) or variants != config['variants_final']:
            raise ValueError('Final code/config/manifest/arms differ from frozen protocol')
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (run / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    job_schedule = schedule(rows, variants)
    save('protocol.json', {'phase': args.phase, 'config': config, 'commit': commit,
        'manifest_sha256': digest(manifest_path), 'variants': variants,
        'expected_episodes': len(rows) * len(variants) * (2 if args.phase == 'align' else 1),
        'retry_failed_episodes': False, 'schedule': job_schedule})
    # Capture the exact working sources, including any not-yet-committed development changes.
    save('source_identity.json', {str(path.relative_to(ROOT)): digest(path) for directory in
        ('scripts', 'smarthome_agent_rl') for path in (ROOT / directory).rglob('*.py')})
    (run / 'working.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    for directory in ('scripts', 'smarthome_agent_rl'):
        shutil.copytree(ROOT / directory, run / 'code' / directory,
                        ignore=shutil.ignore_patterns('__pycache__'))
    sim = ROOT / 'deps/SimuHome'
    upstream_commit = subprocess.check_output(['git', '-C', str(sim), 'rev-parse', 'HEAD'], text=True).strip()
    upstream_dirty = subprocess.check_output(['git', '-C', str(sim), 'status', '--porcelain'], text=True).strip()
    if upstream_commit != manifest['simuhome_commit'] or upstream_dirty:
        raise RuntimeError('Official evaluator/simulator must remain pristine at the manifest revision')
    services, handles, commands = [], [], []
    key = 'smarthome-local-rollout'
    client = httpx.Client(trust_env=False, timeout=30)
    def launch(name, command, cwd, environment=None):
        log = (run / f'{name}.log').open('w')
        handles.append(log)
        process = subprocess.Popen(list(map(str, command)), cwd=cwd,
            env={**os.environ, 'PYTHONNOUSERSITE': '1', **(environment or {})},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        services.append(process)
        commands.append({'name': name, 'pid': process.pid, 'argv': list(map(str, command))})
        save('services.json', commands)
        return process
    def wait(url, process, timeout=90):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(f'Service exited: {url}')
            try:
                if client.get(url).is_success:
                    return
            except httpx.TransportError:
                pass
            time.sleep(1)
        raise TimeoutError(url)
    def api(workflow, method, path, value=None):
        response = client.request(method, f"http://127.0.0.1:{workflow['gateway_port']}" + path,
            headers={'Authorization': f'Bearer {key}'}, json=value)
        response.raise_for_status()
        return response.json()
    def execute(workflow, task, variant, mode):
        output = run / f"worker{workflow['id']}" / task['id'] / variant / mode
        output.parent.mkdir(parents=True, exist_ok=True)
        episode_config = {**config, 'served_model': config['actor_model'], 'variant': variant,
            'simulator_url': f"http://127.0.0.1:{workflow['simulator_port']}/api",
            'model_endpoint': f"http://127.0.0.1:{workflow['actor_port']}/v1",
            'judge_endpoint': f"http://127.0.0.1:{config['judge_port']}/v1",
            'embedding_endpoint': f"http://127.0.0.1:{config['embedding_port']}"}
        if args.phase == 'final':
            episode_config['protocol_frozen_commit'] = commit
        if mode == 'direct':
            with output.with_suffix('.log').open('w') as log:
                try:
                    process = subprocess.run([ROOT / '.venv-baseline/bin/python',
                        ROOT / 'scripts/run_benchmark_episode.py', '--mode', 'direct'], cwd=ROOT,
                        env={**os.environ, 'PYTHONNOUSERSITE': '1', 'SMARTHOME_TASK': json.dumps(task),
                             'SMARTHOME_CONFIG': json.dumps(episode_config), 'SMARTHOME_OUTPUT': str(output)},
                        stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                    if process.returncode:
                        raise RuntimeError(f'Direct episode failed {process.returncode}: {output}')
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f'Direct episode timed out: {output}') from exc
        else:
            created = api(workflow, 'POST', '/api/rollouts', [{
                'input': {'task': task, 'config': episode_config, 'output': str(output)}, 'is_train': False,
                'metadata': {'phase': args.phase, 'variant': variant, 'task_id': task['id']},
                'config': {'timeout_seconds': 1800, 'local': {
                    'agent_class': 'smarthome_agent_rl.benchmark_worker:BenchmarkAgent',
                    'env_map': {'SMARTHOME_TASK': 'input.task', 'SMARTHOME_CONFIG': 'input.config',
                                'SMARTHOME_OUTPUT': 'input.output'}}}}])
            rid = created[0]['rollout_id']
            output.with_suffix('.rollout.json').write_text(json.dumps(created, indent=2))
            deadline = time.monotonic() + 1840
            while time.monotonic() < deadline:
                detail = api(workflow, 'GET', f'/api/rollouts/{rid}')
                state = detail['rollout']['status']['state']
                if state in ('succeeded', 'failed'):
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f'Lightning rollout {rid}')
            event_rows = api(workflow, 'GET', f'/api/rollouts/{rid}/events')
            output.mkdir(exist_ok=True)
            (output / 'lightning_rollout.json').write_text(json.dumps(detail, indent=2), encoding='utf-8')
            (output / 'lightning_events.json').write_text(json.dumps(event_rows, indent=2), encoding='utf-8')
            if state != 'succeeded':
                raise RuntimeError(f'Lightning infrastructure failure {rid}: {output}')
            summary = json.loads((output / 'summary.json').read_text())
            rewards = [row for row in event_rows if row['event_type'] == 'reward']
            requests = [row for row in event_rows if row['event_type'] == 'model_request']
            if len(rewards) != 1 or len(requests) != summary['actor_model_calls']:
                raise RuntimeError('Lightning request/reward accounting mismatch')
            model_calls = json.loads((output / 'model_calls.json').read_text())
            for call, event in zip(model_calls, requests):
                if event['data']['request'] != {**call['request'], 'return_token_ids': True}:
                    raise RuntimeError('Gateway altered actor generation policy or messages')
        summary = json.loads((output / 'summary.json').read_text())
        print(json.dumps({'completed': task['id'], 'variant': variant, 'mode': mode,
                          'success': summary['success']}), flush=True)
        return summary
    def workflow_jobs(workflow):
        results = []
        for item in job_schedule:
            if item['workflow'] != workflow['id']:
                continue
            for variant in item['variants']:
                if args.phase == 'align':
                    execute(workflow, item['task'], variant, 'direct')
                results.append(execute(workflow, item['task'], variant, 'lightning'))
                if args.phase == 'align':
                    parent = run / f"worker{workflow['id']}" / item['task']['id'] / variant
                    calls = {mode: json.loads((parent / mode / 'model_calls.json').read_text())
                             for mode in ('direct', 'lightning')}
                    gateway_events = json.loads((parent / 'lightning/lightning_events.json').read_text())
                    bodies = {'direct': [row['request'] for row in calls['direct']],
                        'lightning': [row['data']['request'] for row in gateway_events if row['event_type'] == 'model_request']}
                    # Authorization differs; actual message/generation bodies must match.
                    identical = bodies['direct'] == bodies['lightning']
                    first_identical = bodies['direct'][0] == bodies['lightning'][0]
                    if not first_identical:
                        raise RuntimeError(f'First direct/Gateway request divergence: {parent}')
                    # Live environment dynamics can change later observations. Verify transport
                    # separately with the exact same frozen request, without any simulator calls.
                    frozen_records = fixed_replay(workflow, bodies['direct'][0], parent)
                    (parent / 'alignment.json').write_text(json.dumps({
                        'first_backend_request_identical': first_identical,
                        'trajectory_requests_identical': identical,
                        'model_calls': {k: len(v) for k, v in calls.items()},
                        'frozen_request_pairs': len(frozen_records), 'frozen_pairs_identical': True,
                        'live_simulator_clock_unchanged': True}))
        return results
    def fixed_replay(workflow, packet, parent):
        records = []
        gateway_packet = {k: v for k, v in packet.items() if k != 'return_token_ids'}
        for repeat in range(3):
            response = client.post(f"http://127.0.0.1:{workflow['actor_port']}/v1/chat/completions", json=packet)
            response.raise_for_status()
            direct = response.json()
            reply = parent / f'frozen-{repeat}.json'
            created = api(workflow, 'POST', '/api/rollouts', [{
                'input': {'request': gateway_packet, 'output': str(reply)}, 'is_train': False,
                'metadata': {'purpose': 'transport_diagnostic_only'},
                'config': {'timeout_seconds': 180, 'local': {
                    'agent_class': 'smarthome_agent_rl.p1_replay:FrozenRequestAgent',
                    'env_map': {'P1_REQUEST': 'input.request', 'P1_OUTPUT': 'input.output'}}}}])
            rid = created[0]['rollout_id']
            deadline = time.monotonic() + 200
            while time.monotonic() < deadline:
                detail = api(workflow, 'GET', f'/api/rollouts/{rid}')
                if detail['rollout']['status']['state'] in ('succeeded', 'failed'):
                    break
                time.sleep(1)
            else:
                raise TimeoutError(rid)
            if detail['rollout']['status']['state'] != 'succeeded':
                raise RuntimeError(f'Frozen replay failed: {rid}')
            events = api(workflow, 'GET', f'/api/rollouts/{rid}/events')
            triplets = api(workflow, 'GET', f'/api/rollouts/{rid}/events?format=triplet')
            event = next(e for e in events if e['event_type'] == 'model_request')
            token_event = next(e for e in triplets if e['event_type'] == 'model_request')
            gateway = json.loads(reply.read_text())['response']
            record = {'repeat': repeat, 'direct': direct, 'gateway': gateway,
                'gateway_request': event['data']['request'], 'token_event': token_event['data']}
            records.append(record)
            (parent / 'frozen_pairs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            if event['data']['request'] != packet or (
                direct['choices'][0]['message']['content'] != gateway['choices'][0]['message']['content']) or (
                direct['prompt_token_ids'] != token_event['data']['prompt_token_ids']) or (
                direct['choices'][0]['token_ids'] != token_event['data']['response_token_ids']):
                raise RuntimeError(f'Frozen direct/Gateway token or text divergence: {parent}')
        return records
    try:
        for workflow in config['workflows']:
            for port in (workflow['simulator_port'], workflow['gateway_port']):
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind(('127.0.0.1', port))
            simulator = launch(f"simulator{workflow['id']}", [ROOT / '.venv-simuhome/bin/python',
                '-m', 'uvicorn', 'src.simulator.api.app:app', '--host', '127.0.0.1',
                '--port', workflow['simulator_port']], ROOT / 'deps/SimuHome')
            wait(f"http://127.0.0.1:{workflow['simulator_port']}/api/__health__", simulator)
            gateway = launch(f"gateway{workflow['id']}", [Path(sys.executable).parent / 'agl-server',
                'host=127.0.0.1', f"port={workflow['gateway_port']}", f'key={key}',
                f"default_proxy.model_name={config['actor_model']}",
                f"default_proxy.val.temperature={config['generation']['temperature']}"], ROOT)
            wait(f"http://127.0.0.1:{workflow['gateway_port']}/healthz", gateway)
            api(workflow, 'POST', '/api/models', [{'model': config['actor_model'],
                'endpoint': f"http://127.0.0.1:{workflow['actor_port']}/v1", 'version': 0}])
            launch(f"controller{workflow['id']}", [Path(sys.executable).parent / 'agl-controller',
                'runner_type=local', 'local_runner.maximum_size=1', 'local_runner.poll_interval=1',
                f"agl_server.url=http://127.0.0.1:{workflow['gateway_port']}", f'agl_server.key={key}'], ROOT,
                {'PYTHONPATH': str(ROOT), 'OPENAI_API_KEY': 'local-unused'})
        with ThreadPoolExecutor(max_workers=2) as pool:
            all_results = list(pool.map(workflow_jobs, config['workflows']))
        results = [row for workflow_result in all_results for row in workflow_result]
        save('completion.json', {'complete': True, 'episodes': len(results),
            'successes': sum(row['success'] for row in results), 'results': results})
    except Exception as exc:
        save('failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        for process in reversed(services):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        for handle in handles:
            handle.close()
        client.close()


if __name__ == '__main__':
    main()
