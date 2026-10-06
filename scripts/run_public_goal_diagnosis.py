"""Frozen public goal extraction/review on previously exposed official tasks.

No simulator invocation, benchmark generation, training records or native scores
are involved. All responses and errors are retained; there are no retries.
"""
import argparse
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
    return [{'task': task, 'public': inputs[task['id']]} for task in expected]


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
    costs = {role: {'requests': len([c for c in calls if c['body']['model'] == model]),
        'tokens': sum(c['usage']['total_tokens'] for c in calls if c['body']['model'] == model and valid_usage(c['usage']))}
        for role, model in (('extractor', config['model']), ('reviewer', config['review_model']))}
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
    if len(config['actors']) != 4 or config['slots_per_actor'] != 16:
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
        with actors(config, services):
            state('public_goal_diagnosis')
            pools = [ThreadPoolExecutor(max_workers=config['slots_per_actor']) for _ in config['actors']]
            try:
                futures = [pools[index % 4].submit(worker, item, seed, config['actors'][index % 4])
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
