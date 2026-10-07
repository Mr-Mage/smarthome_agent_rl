"""Frozen public goal extraction/review on previously exposed official tasks.

No simulator invocation, benchmark generation, training records or native scores
are involved. All responses and errors are retained; there are no retries.
"""
import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.runner import completion, save, valid_usage
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.execution.goals import (extraction_schema, parse_proposal, parse_review,
    public_messages, review_messages, review_schema)
from scripts.run_public_benchmark import actors
from scripts.verify_benchmark import verify


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(config):
    stage = ROOT / config['source_stage']
    verify(stage)
    protocol = read(stage / 'protocol.json')
    if protocol['commit'] != config['source_commit'] or sha(stage / 'artifact_manifest.json') != config['source_artifact_sha256']:
        raise ValueError('Exposed source stage differs from frozen identity')
    expected = read(ROOT / config['manifest'])['tasks']
    inputs, identity = {}, {}
    for item in protocol['schedule']:
        ep = episode_directory(stage, item, 'G')
        contract = read(ep / 'contract.json')
        if contract['task_identity'] != item['task']:
            raise ValueError('Source public task identity differs')
        public = contract['public_context']
        if set(public) != {'query', 'current_time', 'user_location'}:
            raise ValueError('Source includes fields outside the public actor boundary')
        key = item['task']['id']
        if key in inputs and inputs[key] != public:
            raise ValueError('Repeated source public inputs differ')
        inputs[key] = public
        identity[key] = item['task']
    if identity != {r['id']: r for r in expected} or len(inputs) != config['gates']['public_tasks']:
        raise ValueError('Exposed task scope differs from frozen manifest')
    result = [{'task': task, 'public': inputs[task['id']]} for task in expected]
    if config.get('reference_run'):
        reference = ROOT / config['reference_run']
        if sha(reference / 'artifact_manifest.json') != config['reference_artifact_sha256']:
            raise ValueError('Retained reference evidence identity differs')
        original = read(reference / 'protocol.json')
        if original['inputs'] != result or original['extraction_schema'] != extraction_schema() or \
                original['review_schema'] != review_schema() or any(
                    original['config'][key] != config[key] for key in
                    ('generation', 'review_generation', 'review_seed', 'actor_seeds', 'gates', 'manual_review')):
            raise ValueError('Model-only diagnosis must retain public inputs,schemas,generation and gates')
    return result


@contextmanager
def external_actor(config, services):
    """Borrow the permanent service; never restart or reserve H100s for it."""
    services.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    lifecycle = {'managed': False, 'reserved_h100_gpu_seconds': 0,
        'shared_a800_allocation': 'Unallocated permanent service; request costs recorded by role',
        'external_service_stopped': False}
    try:
        body = {'model': config['model'], 'seed': 42, 'temperature': 0.0, 'max_tokens': 32,
            'messages': [{'role': 'user', 'content': 'Reply OK.'}],
            'chat_template_kwargs': {'enable_thinking': False}}
        call = completion(config['actors'][0]['endpoint'], body, config['request_timeout'])
        save(services / 'probe-receipts.json', [call])
        lifecycle['probes'] = {'requests': 1, 'failed': int(call['error'] is not None),
            'tokens': call['usage']['total_tokens'] if valid_usage(call['usage']) else 0,
            'missing_usage': int(not valid_usage(call['usage']))}
        if call['error'] is not None or not valid_usage(call['usage']) or call['response']['model'] != config['model']:
            raise ValueError('Permanent extractor model/usage probe failed')
        lifecycle['ready_seconds'] = time.monotonic() - started
        yield
    finally:
        lifecycle['seconds_including_cleanup'] = time.monotonic() - started
        save(services / 'lifecycle.json', lifecycle)


def request(config, messages, schema, *, seed, review=False):
    generation = config['review_generation' if review else 'generation']
    return {'model': config['review_model'] if review else config['model'], 'seed': seed,
        'messages': messages, 'response_format': schema,
        **{k: v for k, v in generation.items() if k != 'extra_body'}, **generation.get('extra_body', {})}


def evaluate(config, records):
    gates = config['gates']
    dimensions = ('coverage', 'fidelity', 'targets', 'dependencies')
    structured = lambda r: r.get('proposal') is not None
    faithful = lambda r: r.get('review') is not None and all(r['review'][d] == 'YES' for d in dimensions)
    by_seed = {str(seed): {'records': sum(r['seed'] == seed for r in records),
        'valid_proposals': sum(r['seed'] == seed and structured(r) for r in records),
        'review_all_yes': sum(r['seed'] == seed and faithful(r) for r in records)} for seed in config['actor_seeds']}
    calls = [call for r in records for call in r['calls']]
    http_errors = sum(call['error'] is not None for call in calls)
    missing = sum(not valid_usage(call['usage']) for call in calls)
    complete = (len(records) == gates['records'] and
        len({(r['task_id'], r['seed']) for r in records}) == gates['records'] and
        len({r['task_id'] for r in records}) == gates['public_tasks'] and
        all(row['records'] == gates['public_tasks'] for row in by_seed.values()))
    checks = {'complete_records': complete, 'http_errors': http_errors == 0, 'usage_complete': missing == 0,
        'structural_validity': all(r['valid_proposals'] / max(1, r['records']) >= gates['valid_proposal_ratio_min'] for r in by_seed.values()),
        'model_review_fidelity': all(r['review_all_yes'] / max(1, r['records']) >= gates['review_all_yes_ratio_min'] for r in by_seed.values()),
        'manual_semantics': False}
    # Roles remain distinct even when a shared model performs both calls.
    role_calls = {role: [r['calls'][index] for r in records if len(r['calls']) > index]
                  for role, index in (('extractor', 0), ('reviewer', 1))}
    costs = {role: {'requests': len(rows), 'tokens': sum(c['usage']['total_tokens'] for c in rows if valid_usage(c['usage']))}
             for role, rows in role_calls.items()}
    return {'checks': checks, 'by_seed': by_seed, 'cost': costs, 'http_errors': http_errors,
        'missing_usage': missing, 'ready_for_native_integration': False,
        'time_graph_rows': sum(len(r['proposal']['time_graph']) for r in records if structured(r)),
        'time_graph_resolved': sum(g['status'] == 'RESOLVED_PROPOSED' for r in records if structured(r) for g in r['proposal']['time_graph']),
        'scope': 'Public source/DAG/model agreement only; no semantic truth or Task completion proof. '
            'A800 reviewer is the same service as the official judge; retain shared-model bias. '
            'No native SR, independent holdout, training, new benchmark or state-grounding claims.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    external = config.get('actor_deployment') == 'external'
    if external and (config['node'] != 'N70' or len(config['actors']) != 1 or config['slots_per_actor'] != 16 or
            config['actors'][0]['endpoint'] != config['review_endpoint'] or config['model'] != config['review_model']):
        raise ValueError('N70 external diagnosis requires the frozen shared A800 endpoint/model and16 slots')
    if not external and (len(config['actors']) != 4 or config['slots_per_actor'] != 16):
        raise ValueError('Use four isolated actors and 64 total slots')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit the frozen source/config before diagnosis')
    inputs = sources(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    save(output / 'protocol.json', {'source_commit': source_commit, 'config': config,
        'config_sha256': sha(args.config), 'inputs': inputs, 'extraction_schema': extraction_schema(),
        'review_schema': review_schema(), 'retry': False})
    services = output / 'services'
    started = time.monotonic()
    records = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def state(name, **kwargs):
        save(output / 'state.json', {'stage': name, **kwargs})
        print(json.dumps({'stage': name, **kwargs}), flush=True)

    def worker(item, seed, actor):
        task, public = item['task'], item['public']
        directory = output / 'records' / f'seed{seed}' / task['id']
        directory.mkdir(parents=True, exist_ok=False)
        result = {'task_id': task['id'], 'seed': seed, 'actor_id': actor['id'],
            'calls': [], 'proposal': None, 'review': None, 'errors': []}
        body = request(config, public_messages(public['query'], public['current_time']), extraction_schema(), seed=seed)
        call = completion(actor['endpoint'], body, config['request_timeout'])
        result['calls'].append(call)
        save(directory / 'evidence.json', result)
        if call['error'] is None:
            try:
                result['proposal'] = parse_proposal(public['query'], public['current_time'], call['text'])
            except (ValueError, TypeError, KeyError) as exc:
                result['errors'].append({'phase': 'proposal', 'error': str(exc)})
        if result['proposal'] is not None:
            body = request(config, review_messages(public['query'], public['current_time'], result['proposal']),
                           review_schema(), seed=config['review_seed'], review=True)
            call = completion(config['review_endpoint'], body, config['request_timeout'])
            result['calls'].append(call)
            if call['error'] is None:
                try:
                    result['review'] = parse_review(call['text'])
                except (ValueError, TypeError, KeyError) as exc:
                    result['errors'].append({'phase': 'review', 'error': str(exc)})
        save(directory / 'evidence.json', result)
        return result

    try:
        state('loading_actors')
        with (external_actor(config, services) if external else actors(config, services)):
            state('public_goal_diagnosis')
            pools = [ThreadPoolExecutor(max_workers=config['slots_per_actor']) for _ in config['actors']]
            try:
                count = len(config['actors'])
                futures = [pools[index % count].submit(worker, item, seed, config['actors'][index % count])
                    for seed in config['actor_seeds'] for index, item in enumerate(inputs)]
                for future in as_completed(futures):
                    records.append(future.result())
                    if len(records) % 30 == 0:
                        state('public_goal_diagnosis', completed=len(records), expected=config['gates']['records'])
            finally:
                for pool in pools:
                    pool.shutdown(wait=True)
        records.sort(key=lambda r: (r['seed'], r['task_id']))
        report = evaluate(config, records)
        lifecycle = read(services / 'lifecycle.json')
        report['resources'] = lifecycle
        report['inference_workflow_wall_seconds'] = time.monotonic() - started
        report['probes'] = lifecycle.get('probes')
        save(output / 'report.json', report)
        ids = set(config['manual_review']['task_ids'])
        public_by_id = {item['task']['id']: item['public'] for item in inputs}
        save(output / 'manual-review.json', {'criteria': config['manual_review']['criteria'],
            'status': 'pending', 'cases': [{**r, 'public': public_by_id[r['task_id']]}
                for r in records if r['task_id'] in ids]})
        state('awaiting_manual_audit', completed=len(records), checks=report['checks'])
    except BaseException as exc:
        save(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        state('failed', completed=len(records))
        raise
    finally:
        save(output / 'artifact_manifest.json', {str(p.relative_to(output)): sha(p)
            for p in output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})


if __name__ == '__main__':
    main()
