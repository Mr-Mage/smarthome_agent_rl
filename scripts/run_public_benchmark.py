"""Frozen official single-turn ablations with isolated actor-only services."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.homebench import HomeBenchAdapter
from smarthome_agent_rl.benchmarks.runner import completion, digest, run, save, valid_usage


@contextmanager
def actors(config, directory):
    directory.mkdir(parents=True, exist_ok=False)
    processes, handles, deployments = [], [], []
    monitoring_stop = threading.Event()
    monitor = None
    start = time.monotonic()
    lifecycle = {'started_unix': time.time(), 'error': None, 'ready_seconds': None}
    save(directory/'config.json', config)
    try:
        if sorted(a['gpu'] for a in config['actors']) != [0, 1, 2, 3]:
            raise ValueError('Expected four independent GPU actors')
        allocation = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                               '--format=csv,noheader,nounits'], text=True)
        lifecycle['initial_gpu_memory'] = allocation
        if any(int(line.split(',')[1]) > 256 for line in allocation.strip().splitlines()):
            raise RuntimeError('GPUs occupied; preserve other processes')
        for actor in config['actors']:
            parsed = urlparse(actor['endpoint'])
            if parsed.hostname != '127.0.0.1' or parsed.port is None:
                raise ValueError('Owned actors must listen on explicit local ports')
            with socket.socket() as probe:
                probe.bind((parsed.hostname, parsed.port))
        # Existing identity checker; hash once, not once per GPU.
        from scripts.harness_services import inventory
        save(directory/'actor-inventory.json', inventory(ROOT/config['actor_path']))
        for actor in config['actors']:
            name = 'actor'+str(actor['id'])
            parsed = urlparse(actor['endpoint'])
            cache = ROOT/config.get('kernel_cache_root', str(directory/'kernel-cache'))/name
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(actor['gpu']),
                           'CUDA_HOME': '/usr/local/cuda-12.8', 'CUDA_PATH': '/usr/local/cuda-12.8',
                           'VLLM_CACHE_ROOT': str(cache/'vllm'), 'FLASHINFER_WORKSPACE_BASE': str(cache/'flashinfer'),
                           'TORCHINDUCTOR_CACHE_DIR': str(cache/'torchinductor'), 'MAX_JOBS': '8',
                           'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONNOUSERSITE': '1',
                           'HF_HUB_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1'}
            command = [config['model_python'], '-m', 'vllm.entrypoints.openai.api_server',
                       '--model', str((ROOT/config['actor_path']).resolve()), '--served-model-name', config['model'],
                       '--host', '127.0.0.1', '--port', str(parsed.port), '--tensor-parallel-size', '1',
                       '--max-model-len', str(config['context']), '--max-num-seqs', str(config['max_num_seqs']),
                       '--gpu-memory-utilization', str(config['gpu_memory_utilization']),
                       '--seed', str(config['engine_seed']), '--language-model-only',
                       '--enable-prefix-caching', '--no-enable-log-requests']
            command += config.get('actor_extra_args', [])
            log = (directory/(name+'.log')).open('w', encoding='utf-8')
            handles.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            deployments.append({'id': actor['id'], 'pid': process.pid, 'argv': command})
            save(directory/'deployments.json', deployments)
        def sample_resources():
            with (directory/'gpu-samples.jsonl').open('w', encoding='utf-8') as log:
                while not monitoring_stop.is_set():
                    sample = {'unix': time.time()}
                    try:
                        result = subprocess.run(['nvidia-smi',
                            '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
                        sample['gpus'] = result.stdout.strip().splitlines()
                    except Exception as exc:
                        sample['error'] = str(exc)
                    log.write(json.dumps(sample)+'\n')
                    log.flush()
                    monitoring_stop.wait(10)
        monitor = threading.Thread(target=sample_resources, daemon=True)
        monitor.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic()+config['startup_timeout']
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('Actor exited during startup; inspect retained logs')
            ready = []
            for actor in config['actors']:
                try:
                    with opener.open(actor['endpoint']+'/models', timeout=2) as response:
                        data = json.load(response)
                    ready.append(any(m['id'] == config['model'] and m.get('max_model_len', 0) >= config['context']
                                     for m in data['data']))
                except Exception:
                    ready.append(False)
            if all(ready):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Actor startup exceeded frozen budget')
            time.sleep(1)
        lifecycle['ready_seconds'] = time.monotonic()-start
        probes = []
        for actor in config['actors']:
            messages = [{'role': 'system', 'content': 'Reply OK.'}]
            if config.get('chat_transport') == 'append_empty_user':
                messages.append({'role': 'user', 'content': ''})
            generation = config['generation']
            body = {'model': config['model'], 'messages': messages, 'seed': config['model_seed'],
                    **{k: v for k, v in generation.items() if k != 'extra_body'},
                    **generation.get('extra_body', {}), 'max_tokens': 8}
            call = completion(actor['endpoint'], body, config['request_timeout'])
            probes.append(call)
            save(directory/'probe-receipts.json', probes)
            lifecycle['probes'] = {'requests': len(probes),
                'failed': sum(p['error'] is not None for p in probes),
                'tokens': sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
                'missing_usage': sum(not valid_usage(p['usage']) for p in probes)}
            if call['error'] or not valid_usage(call['usage']):
                raise RuntimeError('Actor chat/usage probe failed before benchmark requests; retain receipt')
        save(directory/'lifecycle.json', lifecycle)
        yield
    except BaseException as exc:
        lifecycle['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for handle in handles:
            handle.close()
        monitoring_stop.set()
        if monitor is not None:
            monitor.join(timeout=12)
        lifecycle.update(seconds_including_cleanup=time.monotonic()-start,
                         reserved_h100_gpu_seconds=(time.monotonic()-start)*len(processes),
                         stopped_pids=[p.pid for p in processes], exit_codes=[p.returncode for p in processes])
        save(directory/'lifecycle.json', lifecycle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['HomeBench'], default='HomeBench')
    parser.add_argument('--config', default='configs/homebench-ablation.json', type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=['calibration', 'full'], default='calibration')
    parser.add_argument('--launch-actors', action='store_true')
    parser.add_argument('--calibration', type=Path, help='Frozen successful calibration directory required for full')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    lock = json.loads((ROOT/'configs/public-benchmarks.json').read_text(encoding='utf-8'))['HomeBench']
    if lock['commit'] != config['source_commit']:
        raise ValueError('Experiment/source commit differs')
    if args.stage == 'full':
        if args.calibration is None:
            raise ValueError('Full requires the completed calibration evidence')
        report = json.loads((args.calibration/'report.json').read_text(encoding='utf-8'))
        freeze = json.loads((args.calibration/'freeze.json').read_text(encoding='utf-8'))
        if report['status'] != 'complete' or not report['engineering_gate']['passed'] or freeze['config_sha256'] != digest(config):
            raise ValueError('Frozen calibration engineering gate not satisfied')
    adapter = HomeBenchAdapter(args.source, lock)
    # Freeze execution identity before service startup; later documentation commits cannot rewrite it.
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    def execute():
        report = run(adapter, config, args.output, config['selection'][args.stage])
        report['provenance'] = {'git_commit': commit,
                                'source': lock, 'stage': args.stage}
        save(args.output/'report.json', report)
        print(json.dumps({'state': report['status'], 'episodes': report['arms']['B0']['episodes'],
                          'native_em': {a: r['exact_match'] for a, r in report['arms'].items()},
                          'engineering_gate': report['engineering_gate']}), flush=True)
    # SIGTERM also unwinds the context manager and stops only owned processes.
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Experiment interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    if args.launch_actors:
        directory = args.output.parent/(args.output.name+'-services')
        with actors(config, directory):
            execute()
        report = json.loads((args.output/'report.json').read_text(encoding='utf-8'))
        report['service_cost'] = json.loads((directory/'lifecycle.json').read_text(encoding='utf-8'))
        save(args.output/'report.json', report)
    else:
        execute()


if __name__ == '__main__':
    main()
