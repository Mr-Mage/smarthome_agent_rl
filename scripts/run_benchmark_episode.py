"""Official episode runner: original reset, time semantics, evaluators and judge panel."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess
from threading import Lock

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deps/SimuHome'))
from src.agents.providers import OpenAIChatProvider
from src.agents.tools import ToolConfig, set_tool_config
from src.pipelines.episode_evaluation import runner
from smarthome_agent_rl.generation import install_generation_options
from smarthome_agent_rl.retrieval import load_retrieval
from smarthome_agent_rl.benchmark import task_failure_kind


def main(mode):
    config = json.loads(os.environ['SMARTHOME_CONFIG'])
    if config.get('protocol_frozen_commit'):
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        if commit != config['protocol_frozen_commit'] or dirty:
            raise RuntimeError('Final source changed after protocol freeze')
    task = json.loads(os.environ['SMARTHOME_TASK'])
    output = Path(os.environ['SMARTHOME_OUTPUT'])
    output.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    endpoint = os.environ['AGL_OPENAI_BASE_URL'] if mode == 'lightning' else config['model_endpoint']
    key = os.environ['AGL_KEY'] if mode == 'lightning' else 'local-unused'
    client = httpx.Client(trust_env=False, timeout=30)
    def emit(kind, data):
        if mode == 'lightning':
            response = client.post(os.environ['AGL_EVENT_URL'], headers={'Authorization': f'Bearer {key}'},
                                   json={'event_type': kind, 'data': data})
            response.raise_for_status()
    path = ROOT / 'deps/SimuHome/data/benchmark' / task['path']
    if path.name != task['path'] or hashlib.sha256(path.read_bytes()).hexdigest() != task['sha256']:
        raise ValueError('Frozen official case identity mismatch')
    calls, judges, events, starts = [], [], [], {}
    calls_lock = Lock()
    def hook(label, collection):
        def capture(response):
            response.read()
            record = {'provider': label, 'request': json.loads(response.request.content),
                'response': response.json(), 'status': response.status_code,
                'duration_seconds': time.monotonic() - starts.pop(id(response.request), time.monotonic())}
            with calls_lock:
                collection.append(record)
                save('judge_calls.json' if collection is judges else 'model_calls.json', collection)
        return capture
    def provider(model, url, seed, generation, collection, label, api_key='local-unused'):
        p = OpenAIChatProvider(model=model, api_base=url, api_key=api_key,
            seed=seed, temperature=generation['temperature'], timeout=300)
        install_generation_options(p, {'generation': generation})
        p._client._client.event_hooks['request'].append(lambda request: starts.update({id(request): time.monotonic()}))
        p._client._client.event_hooks['response'].append(hook(label, collection))
        return p
    actor = provider(config['served_model'], endpoint, config['model_seed'], config['generation'], calls, 'actor', key)
    if mode == 'direct':
        create = actor._client.chat.completions.create
        def include_token_ids(**kwargs):
            kwargs['extra_body'] = {**kwargs.get('extra_body', {}), 'return_token_ids': True}
            return create(**kwargs)
        actor._client.chat.completions.create = include_token_ids
    panel = [provider(config['judge_model'], config['judge_endpoint'], seed,
        config['judge_generation'], judges, f'judge-{seed}') for seed in config['judge_seeds']]
    retrieval = json.loads((ROOT / 'configs/p1-qwen35-9b-retrieval.json').read_text())['retrieval']
    retrieval['endpoint'] = config['embedding_endpoint']
    set_tool_config(ToolConfig(base_url=config['simulator_url'], timeout=30,
        db=load_retrieval(retrieval, output / 'retrieval_calls.json')))
    variant = config['variant']
    def trace(kind, payload):
        event = {'event': kind, 'payload': payload}
        events.append(event)
        save('agent_events.json', events)
        if kind in ('observation', 'finish', 'rejected_action'):
            emit('environment_step', event)
    original = runner._build_agent
    tokenization_calls = []
    def count_tokens(messages):
        started = time.monotonic()
        request = {'model': config['served_model'], 'messages': [asdict(m) for m in messages],
                  'add_generation_prompt': True, 'chat_template_kwargs':
                  config['generation'].get('extra_body', {}).get('chat_template_kwargs', {})}
        response = client.post(config['model_endpoint'].removesuffix('/v1') + '/tokenize',
            json=request)
        response.raise_for_status()
        tokenization_calls.append({'count': response.json()['count'], 'status': response.status_code,
            'duration_seconds': time.monotonic() - started,
            'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()})
        save('tokenization_calls.json', tokenization_calls)
        return response.json()['count']
    def build(llm, *, max_steps, strategy):
        if variant == 'B0':
            agent = original(llm, max_steps=max_steps, strategy=strategy)
            agent.config.trace_fn = trace
        else:
            from smarthome_agent_rl.harness_agent import HarnessAgent
            agent = HarnessAgent(llm, variant=variant, max_steps=max_steps, trace_fn=trace,
                audit_fn=lambda data: save('harness_audit.json', data),
                repair_limit=config['recovery_per_action'], query_limit=config['extra_queries_max'],
                policy=config.get('variant_policies', {}).get(variant), token_count_fn=count_tokens)
        return agent
    runner._build_agent = build
    save('contract.json', {'config': config, 'task_identity': task, 'mode': mode,
        'evaluator': 'original run_single_config', 'upstream_source_modified': False,
        'agent_inputs': ['query', 'user_location', 'current_time'], 'training': False})
    error, result = None, None
    started = time.monotonic()
    try:
        result = runner.run_single_config(cfg_path=str(path), base_url=config['simulator_url'], timeout=30,
            max_steps=config['max_steps'], agent_strategy='react', main_llm=actor, judge_llms=panel)
        save('official_result.json', result)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        save('error.json', error)
    finally:
        runner._build_agent = original
        failure_kind = task_failure_kind(error, calls)
        task_failure = failure_kind is not None
        score = result['evaluation_result']['score'] if result else None
        tokens = lambda rows: sum(r['response'].get('usage', {}).get('total_tokens', 0) for r in rows)
        summary = {'task_id': task['id'], 'variant': variant, 'mode': mode, 'actor_seed': config['model_seed'],
            'official_score': score, 'evaluator_called': result is not None,
            'success': score == 1, 'task_failure': task_failure, 'error': error,
            'task_failure_kind': failure_kind,
            'infrastructure_error': (error is not None and not task_failure) or score == -1,
            'actor_model_calls': len(calls), 'actor_tokens': tokens(calls),
            'judge_model_calls': len(judges), 'judge_tokens': tokens(judges),
            'duration_seconds': time.monotonic() - started}
        save('summary.json', summary)
        if (error is None and score != -1) or task_failure:
            emit('reward', {'value': float(score == 1), 'source': 'official_simuhome_evaluator' if result else failure_kind,
                'evaluator_called': result is not None})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        actor._client.close()
        for judge in panel:
            judge._client.close()
        client.close()
    if summary['infrastructure_error']:
        raise RuntimeError(f'Infrastructure failure: {error or result["evaluation_result"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['direct', 'lightning'], required=True)
    main(parser.parse_args().mode)
