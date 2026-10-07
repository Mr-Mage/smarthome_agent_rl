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
import importlib
from threading import Lock

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deps/SimuHome'))
from src.agents.providers import OpenAIChatProvider
from src.agents.tools import ToolConfig, set_tool_config
from src.pipelines.episode_evaluation import runner
from smarthome_agent_rl.generation import install_generation_options, record_generation_errors, use_direct_service_transport, audit_response_payload
from smarthome_agent_rl.retrieval import load_retrieval
from smarthome_agent_rl.benchmark import task_failure_kind
from smarthome_agent_rl.profiling import PhaseProfile, TimedTime
from smarthome_agent_rl.benchmarks.simuhome import SimuHomeAdapter


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
    profile = PhaseProfile(save)
    endpoint = os.environ['AGL_OPENAI_BASE_URL'] if mode == 'lightning' else config['model_endpoint']
    key = os.environ['AGL_KEY'] if mode == 'lightning' else 'local-unused'
    client = httpx.Client(trust_env=False, timeout=30)
    def emit(kind, data):
        if mode == 'lightning':
            response = client.post(os.environ['AGL_EVENT_URL'], headers={'Authorization': f'Bearer {key}'},
                                   json={'event_type': kind, 'data': data})
            response.raise_for_status()
    benchmark = SimuHomeAdapter(ROOT / 'deps/SimuHome/data/benchmark', {'tasks': [task]})
    path = benchmark.task_path(task['id'])
    calls, judges, events, starts = [], [], [], {}
    calls_lock = Lock()
    provider_errors = []
    def record_provider_error(row):
        with calls_lock:
            provider_errors.append(row)
            save('provider_errors.json', provider_errors)
    def hook(label, collection):
        def capture(response):
            response.read()
            record = {'provider': label, 'request': json.loads(response.request.content),
                'response': audit_response_payload(response), 'status': response.status_code,
                'duration_seconds': time.monotonic() - starts.pop(id(response.request), time.monotonic())}
            with calls_lock:
                collection.append(record)
                save('judge_calls.json' if collection is judges else 'model_calls.json', collection)
        return capture
    def provider(model, url, seed, generation, collection, label, api_key='local-unused'):
        p = OpenAIChatProvider(model=model, api_base=url, api_key=api_key,
            seed=seed, temperature=generation['temperature'], timeout=300)
        use_direct_service_transport(p)
        install_generation_options(p, {'generation': generation})
        record_generation_errors(p, label, record_provider_error)
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
    attached_runtime = None
    from smarthome_agent_rl.report_semantic_review import attach_report_reviewer
    agent_policy, report_reviewer = attach_report_reviewer(config, variant,
        audit_fn=lambda rows: save('semantic_review_calls.json', rows))
    if report_reviewer is not None:
        report_reviewer.verify = profile.wrap(report_reviewer.verify, 'semantic_review')
    def trace(kind, payload):
        event = {'event': kind, 'payload': payload, 'at_seconds': time.monotonic() - profile.origin}
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
        nonlocal attached_runtime
        if variant == 'B0':
            agent = original(llm, max_steps=max_steps, strategy=strategy)
            agent.config.trace_fn = trace
        else:
            from smarthome_agent_rl.harness_agent import HarnessAgent
            agent = HarnessAgent(llm, variant=variant, max_steps=max_steps, trace_fn=trace,
                audit_fn=lambda data: save('harness_audit.json', data),
                repair_limit=config['recovery_per_action'], query_limit=config['extra_queries_max'],
                policy=agent_policy, token_count_fn=count_tokens)
        if config.get('variant_policies', {}).get(variant, {}).get('task_runtime'):
            if variant not in ('GTM', 'GTME', 'GTMEC') or type(agent.executor).__name__ != 'GuardedExecutor':
                raise ValueError('Native task-runtime ablation must use an isolated legacy Guard arm')
            policies = config['variant_policies']
            base_policy = {k: v for k, v in policies[variant].items()
                           if k not in ('task_runtime', 'task_runtime_tolerance', 'task_runtime_clock', 'task_runtime_context')}
            if base_policy != policies.get('G'):
                raise ValueError('GTM must preserve the frozen G policy and isolate runtime attachment')
            from smarthome_agent_rl.execution.episode import EpisodeRuntime
            from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter
            from smarthome_agent_rl.guard import command_contracts, public_power_rules
            signatures, _ = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
            power, _ = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices')
            clock_policy = policies[variant].get('task_runtime_clock', 'poll')
            if clock_policy == 'public_events' and variant == 'GTMEC' and policies[variant].get('task_runtime_context'):
                from smarthome_agent_rl.execution.resource_context import ResourceEpisodeRuntime
                runtime_class = ResourceEpisodeRuntime
            elif clock_policy == 'public_events' and variant == 'GTME':
                from smarthome_agent_rl.execution.events import EventEpisodeRuntime
                runtime_class = EventEpisodeRuntime
            elif clock_policy == 'poll' and variant == 'GTM':
                runtime_class = EpisodeRuntime
            else:
                raise ValueError('Runtime variant/clock policy differs from frozen arm identity')
            attached_runtime = agent.task_runtime = runtime_class(agent.executor,
                SimuHomeContractAdapter(signatures, power), output / 'task-runtime.sqlite3',
                tolerance=config['variant_policies'][variant]['task_runtime_tolerance'], save=save)
            attached_runtime.supervise = profile.wrap(attached_runtime.supervise, 'runtime_supervision')
        import src.agents.strategies.react_agent as react_module
        return benchmark.bind_agent(task['id'], profile.agent(agent, react_module))
    runner._build_agent = build
    import src.agents.strategies.react_agent as react_module
    evaluation_module = importlib.import_module(runner._EVALUATOR_REGISTRY[(task['query_type'], task['case'])])
    original_evaluate = evaluation_module.evaluate
    evaluation_module.evaluate = profile.wrap(original_evaluate, 'evaluator')
    replaced_times = []
    for module in (runner, react_module, evaluation_module):
        if hasattr(module, 'time'):
            replaced_times.append((module, module.time))
            module.time = TimedTime(module.time, profile)
    client_methods = []
    for name in ('health', 'reset_simulation', 'get_home_state', 'fast_forward_to', 'get_workflow_status'):
        if hasattr(runner.SmartHomeClient, name):
            method = getattr(runner.SmartHomeClient, name)
            client_methods.append((name, method))
            setattr(runner.SmartHomeClient, name, profile.wrap(method, 'simulator_client', name))
    original_fast_forward = runner.SmartHomeClient.fast_forward_to
    def observed_fast_forward(*args, **kwargs):
        response = original_fast_forward(*args, **kwargs)
        # Never pass evaluator deadlines, labels or result payloads to the
        # runtime. Read the current public clock through its budgeted tools.
        if attached_runtime is not None:
            if hasattr(attached_runtime, 'observe_native_response'):
                attached_runtime.observe_native_response(response)
            else:
                attached_runtime.supervise(phase='native_virtual_time_advanced')
        return response
    runner.SmartHomeClient.fast_forward_to = observed_fast_forward
    save('contract.json', {'config': config, 'task_identity': task, 'mode': mode,
        'evaluator': 'original run_single_config', 'upstream_source_modified': False,
        'agent_inputs': ['query', 'user_location', 'current_time'],
        'public_context': benchmark.public_input(task['id']), 'training': False})
    error, result = None, None
    started = time.monotonic()
    try:
        result = runner.run_single_config(cfg_path=str(path), base_url=config['simulator_url'], timeout=30,
            max_steps=config['max_steps'], agent_strategy='react', main_llm=actor, judge_llms=panel)
        if attached_runtime is not None:
            attached_runtime.finish()
        benchmark.score(task['id'], result)  # Validate native episode identity; never replace its evaluator.
        save('official_result.json', result)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        save('error.json', error)
    finally:
        runner._build_agent = original
        evaluation_module.evaluate = original_evaluate
        for module, original_time in replaced_times:
            module.time = original_time
        for name, method in client_methods:
            setattr(runner.SmartHomeClient, name, method)
        if attached_runtime is not None:
            attached_runtime.finish(supervise=False)
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
            'semantic_review': report_reviewer.costs() if report_reviewer is not None else None,
            'duration_seconds': time.monotonic() - started}
        save('summary.json', summary)
        profile.flush(summary['duration_seconds'], started - profile.origin)
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
