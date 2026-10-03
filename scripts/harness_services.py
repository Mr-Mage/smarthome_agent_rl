"""Own the isolated four-H100 inference services; never modify Conda environments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_resources(config):
    gpus = config['judge_gpus'] + [gpu for w in config['workflows'] for gpu in w['gpus']]
    ports = [config['judge_port'], config['embedding_port']] + [
        w[k] for w in config['workflows'] for k in ('actor_port', 'simulator_port', 'gateway_port')]
    if sorted(gpus) != [0, 1, 2, 3] or len(set(ports)) != len(ports):
        raise ValueError('Expected disjoint four-GPU and port allocation')
    for port in ports:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))


def inventory(model):
    model = model.resolve()
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    required = sorted(set(shards + [p.name for p in model.iterdir() if p.is_file() and (
        p.suffix in ('.json', '.jinja', '.txt') and p.name != 'download-verification.json')]))
    files = []
    for name in required:
        path = model / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        files.append({'name': name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    # This establishes local identity, not provenance against an upstream manifest.
    return {'model': str(model), 'shards': len(shards), 'files': files,
            'provenance_verified': False, 'identity': hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'probe', 'inventory'])
    parser.add_argument('--config', default='configs/harness-mvp.json')
    parser.add_argument('--run-dir', default='work/harness-mvp/services')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    run = ROOT / args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if args.action == 'inventory':
        save(run / 'judge-inventory.json', inventory(Path(config['judge_path'])))
        save(run / 'actor-inventory.json', inventory(ROOT / config['actor_path']))
        return
    if args.action == 'probe':
        probe(config, run)
        return
    validate_resources(config)
    if (run / 'services.json').exists():
        raise FileExistsError('Use a fresh run directory; keep previous failure evidence')
    owned, handles, state = [], [], []
    def launch(name, command, environment, url):
        log = (run / f'{name}.log').open('w')
        handles.append(log)
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT,
            env={**os.environ, 'PYTHONNOUSERSITE': '1', **environment}, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        owned.append(process)
        state.append({'name': name, 'pid': process.pid, 'argv': list(map(str, command)),
                      'environment': environment, 'url': url, 'ready': False})
        save(run / 'services.json', state)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save(run / 'config.json', config)
    (run / 'supervisor.pid').write_text(str(os.getpid()))
    try:
        retrieval = json.loads((ROOT / 'configs/p1-qwen35-9b-retrieval.json').read_text())['retrieval']
        launch('embedding', [retrieval['python'], ROOT / 'scripts/serve_doc_embeddings.py',
            '--model', retrieval['model_path'], '--port', config['embedding_port']],
            {'CUDA_VISIBLE_DEVICES': ''}, f"http://127.0.0.1:{config['embedding_port']}/health")
        deployments = [('actor' + str(w['id']), config['actor_path'], config['actor_model'],
                        w['gpus'], w['actor_port'], config['actor_context'], 1)
                       for w in config['workflows']]
        deployments.append(('judge', config['judge_path'], config['judge_model'],
                            config['judge_gpus'], config['judge_port'], config['judge_context'], 6))
        for name, path, model, gpus, port, context, sequences in deployments:
            environment = {'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)),
                'CUDA_HOME': '/usr/local/cuda-12.8', 'CUDA_PATH': '/usr/local/cuda-12.8',
                'FLASHINFER_WORKSPACE_BASE': str(run / name / 'flashinfer'),
                'VLLM_CACHE_ROOT': str(run / name / 'vllm'),
                'TORCHINDUCTOR_CACHE_DIR': str(run / name / 'torchinductor'), 'MAX_JOBS': '8'}
            launch(name, [config['model_python'], '-m', 'vllm.entrypoints.openai.api_server',
                '--model', (ROOT / path).resolve(), '--served-model-name', model,
                '--host', '127.0.0.1', '--port', port, '--tensor-parallel-size', len(gpus),
                '--max-model-len', context, '--max-num-seqs', sequences,
                '--gpu-memory-utilization', '0.72' if name == 'judge' else '0.60',
                '--seed', config['engine_seed'], '--language-model-only', '--enforce-eager',
                '--no-enable-log-requests'], environment, f'http://127.0.0.1:{port}/v1/models')
        deadline = time.monotonic() + config['startup_timeout']
        with httpx.Client(trust_env=False, timeout=3) as client:
            while not stopped:
                for item, process in zip(state, owned):
                    if process.poll() is not None:
                        raise RuntimeError(f"{item['name']} exited {process.returncode}; inspect its log")
                    if not item['ready']:
                        try:
                            item['ready'] = client.get(item['url']).is_success
                        except httpx.TransportError:
                            pass
                save(run / 'services.json', state)
                if all(item['ready'] for item in state):
                    (run / 'ready').touch()
                elif time.monotonic() > deadline:
                    raise TimeoutError('Service startup timeout')
                time.sleep(3)
    except Exception as exc:
        save(run / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        for process in reversed(owned):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        for log in handles:
            log.close()
        (run / 'ready').unlink(missing_ok=True)


def probe(config, run):
    def request(name, port, model, generation, prompt, seed, expected=None):
        body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                'seed': seed, **{k: v for k, v in generation.items() if k != 'extra_body'},
                **generation.get('extra_body', {})}
        with httpx.Client(trust_env=False, timeout=240) as client:
            response = client.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=body)
            response.raise_for_status()
            result = response.json()
        text = result['choices'][0]['message']['content']
        assert result['usage']['total_tokens'] > 0 and isinstance(text, str) and text.strip()
        if name == 'judge':
            assert text.strip().upper()[:1] == expected and '<think>' not in text
        return {'service': name, 'request': body, 'response': result}
    tasks = [('actor' + str(w['id']), w['actor_port'], config['actor_model'], config['generation'],
              'Reply with the single word READY.', 42) for w in config['workflows']]
    for seed in config['judge_seeds']:
        for answer, expected in [('4', 'A'), ('5', 'B')]:
            tasks.append(('judge', config['judge_port'], config['judge_model'], config['judge_generation'],
                f'Evaluate the answer: question 2+2, answer {answer}. Reply A if correct, B otherwise. Only one letter.', seed, expected))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda args: request(*args), tasks))
    save(run / 'inference-probe.json', {'passed': True, 'results': results})
    print('Two actors and the three-vote local judge passed inference probes', flush=True)


if __name__ == '__main__':
    main()
