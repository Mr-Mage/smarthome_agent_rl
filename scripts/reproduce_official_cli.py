"""Run pristine SimuHome CLI against local inference, with auditable external config.

This is a protocol reproduction harness, not a paper-score reproduction claim.
No agent, simulator, evaluator, or benchmark source is patched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/reproduction-smoke.json')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    sim = ROOT / 'deps/SimuHome'
    model = ROOT.parent / config['model_path']
    baseline = ROOT / '.venv-baseline/bin/python'
    model_python = config.get('model_python', sys.executable)
    save = lambda name, value: (run / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    owned, handles, commands = [], [], []
    dirty = subprocess.check_output(['git', '-C', str(sim), 'status', '--porcelain'], text=True)
    if dirty:
        raise RuntimeError('Upstream must be pristine')
    if config['case'] != 'feasible' or config['qt'] not in {'qt2', 'qt3', 'qt4'}:
        raise ValueError('Local reproduction currently allows simulator-verified feasible subsets only; judge-dependent tasks need separate configuration')
    selected = [sim / 'data/benchmark' / f"{config['qt']}_feasible_seed_{seed}.json" for seed in config['seeds']]
    if not all(path.exists() for path in selected):
        raise FileNotFoundError('Requested official benchmark seeds not present')
    checkpoint_validation = {'required': bool(config.get('model_revision')), 'complete': model.is_dir()}
    if config.get('model_revision'):
        manifest = json.loads((ROOT / 'configs/reproduction-model-qwen3-32b.json').read_text())
        marker_path = model / 'download-verification.json'
        marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
        invalid_sizes = [item['path'] for item in manifest['files']
            if not (model / item['path']).is_file() or (model / item['path']).stat().st_size != item['bytes']]
        checkpoint_validation = {'required': True, 'marker_path': str(marker_path), 'invalid_or_missing_sizes': invalid_sizes,
            'complete': not invalid_sizes and marker.get('verified') is True and not marker.get('invalid_or_missing')
                and marker.get('repository') == config['model_repo'] and marker.get('revision') == config['model_revision'],
            'marker': marker, 'method': 'Prior full hash verification marker plus current all-file sizes; use downloader --verify-only for full rehash'}
    api = f"http://127.0.0.1:{config['model_port']}/v1"
    spec = {
        'schema': 'simuhome-eval-spec-v1',
        'run': {'id': 'official', 'output_root': str(run)},
        'episode': {'dir': str(sim / 'data/benchmark'), 'qt': config['qt'], 'case': config['case'], 'seed': ','.join(map(str, config['seeds']))},
        'strategy': {'name': 'react', 'timeout': config['timeout'], 'temperature': 0.0, 'max_steps': 20},
        'orchestration': {'max_workers': 1, 'simulator_start_timeout': 60, 'simulator_start_retries': 1, 'evaluation_retries': 0, 'allow_partial_start': False},
        'api': {'base': api, 'key': None},
        # Provider instances are initialized by upstream, but these evaluators do not call the judge.
        'judge': {'model': 'unused-simulator-only-judge', 'api_base': api, 'api_key': None},
        'models': [{'model': config['served_model'], 'api_base': api, 'api_key': None}],
    }
    save('evaluation_spec.yaml', spec)  # JSON content is valid YAML; original CLI requires YAML suffix.
    save('config.json', config)
    save('preflight.json', {
        'simuhome_commit': subprocess.check_output(['git', '-C', str(sim), 'rev-parse', 'HEAD'], text=True).strip(),
        'upstream_pristine_before': True, 'model_path': str(model), 'model_available': model.is_dir(),
        'model_config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest() if (model / 'config.json').exists() else None,
        'selected_episode_sha256': {str(p.relative_to(sim)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected},
        'gpu': config['gpu'], 'tensor_parallel_size': config['tensor_parallel_size'],
        'checkpoint_validation': checkpoint_validation,
        'source_sha256': {p: hashlib.sha256((sim / p).read_bytes()).hexdigest() for p in
            ['src/cli/main.py', 'src/cli/parallel_model_evaluation.py', 'src/cli/episode_evaluator.py',
             'src/pipelines/episode_evaluation/qt3/feasible.py', 'src/agents/strategies/react_agent.py',
             'src/agents/providers/openai_provider.py', 'src/agents/tools.py', 'prompts/agents/react.py']},
        'official_cli': True, 'official_benchmark': True, 'official_evaluator': True,
        'paper_exact_reproduction': False, 'training': False,
        'limitations': config['limitations'],
        'simulator_ports': 'Original CLI allocates independent ephemeral free ports; no hardcoded development simulator port',
        'judge_calls_expected': 0, 'paid_api_calls_allowed': False,
    })
    if args.preflight_only:
        return
    if not model.is_dir():
        raise FileNotFoundError(model)
    if not checkpoint_validation['complete']:
        raise RuntimeError('Pinned checkpoint is not completely downloaded and verified')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', config['model_port']))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': config['gpu'], 'NO_PROXY': 'localhost,127.0.0.1', 'no_proxy': 'localhost,127.0.0.1',
           'OPENAI_API_KEY': 'local-unused', 'OPENAI_API_BASE': api, 'OPENAI_BASE_URL': api, 'PYTHONPATH': str(sim),
           'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        env.pop(key, None)
    def launch(name, command, cwd):
        command = list(map(str, command))
        commands.append({'name': name, 'argv': command, 'cwd': str(cwd), 'env': {k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OPENAI_BASE_URL', 'PYTHONPATH']}})
        save('commands.json', commands)
        handle = (run / f'{name}.log').open('w')
        handles.append(handle)
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append(proc)
        return proc
    try:
        (run / 'model-dependencies.txt').write_text(subprocess.check_output([model_python, '-m', 'pip', 'freeze'], text=True))
        backend = launch('model', [model_python, '-m', 'vllm.entrypoints.openai.api_server', '--model', model,
            '--served-model-name', config['served_model'], '--host', '127.0.0.1', '--port', config['model_port'],
            '--max-model-len', config['model_context'], '--max-num-seqs', 1,
            '--gpu-memory-utilization', config['gpu_memory_utilization'], '--tensor-parallel-size', config['tensor_parallel_size'], '--enforce-eager'], ROOT)
        with httpx.Client(trust_env=False, timeout=10) as client:
            deadline = time.monotonic() + config.get('model_startup_timeout', 600)
            while time.monotonic() < deadline:
                if backend.poll() is not None:
                    raise RuntimeError('Local inference exited during startup')
                try:
                    response = client.get(api + '/models')
                    if response.is_success:
                        save('model_service.json', response.json())
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Model startup')
        cli = launch('official-cli', [baseline, '-m', 'src.cli.main', 'eval-start', '--spec', run / 'evaluation_spec.yaml'], sim)
        code = cli.wait()
        if code:
            raise RuntimeError(f'Original CLI exited with {code}')
        aggregate = launch('official-aggregate', [baseline, '-m', 'src.cli.main', 'aggregate-all', '--dir', run / 'official'], sim)
        if aggregate.wait():
            raise RuntimeError('Original aggregation failed')
        records = []
        for path in sorted((run / 'official').rglob('*_seed_*.json')):
            item = json.loads(path.read_text())
            if 'evaluation_result' in item:
                records.append({'path': str(path.relative_to(run)), 'seed': item.get('seed'), 'score': item['evaluation_result'].get('score'),
                    'error_type': item['evaluation_result'].get('error_type'), 'required_actions': item['evaluation_result'].get('required_actions'),
                    'error_category': item['evaluation_result'].get('error_category'),
                    'targets_eval': item['evaluation_result'].get('targets_eval'), 'steps': len(item.get('steps', [])),
                    'judge': item['evaluation_result'].get('judge', [])})
        if len(records) != len(selected):
            raise RuntimeError(f'Expected {len(selected)} records, found {len(records)}')
        if any(item['judge'] for item in records):
            raise RuntimeError('Unexpected judge use in simulator-only subset')
        dirty_after = subprocess.check_output(['git', '-C', str(sim), 'status', '--porcelain'], text=True)
        if dirty_after:
            raise RuntimeError('Upstream changed during run')
        save('verification.json', {'records': records, 'episodes': len(records), 'successes': sum(r['score'] == 1 for r in records),
            'failures': sum(r['score'] == 0 for r in records), 'infra_or_schema_errors': sum(r['score'] == -1 for r in records),
            'upstream_pristine_after': True, 'official_cli_and_aggregation': True, 'paper_exact_reproduction': False,
            'model_replaced': config['model_replaced'], 'judge_used': False})
        print(json.dumps({'episodes': len(records), 'successes': sum(r['score'] == 1 for r in records), 'artifacts': str(run)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        save('failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        for proc in reversed(owned):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
