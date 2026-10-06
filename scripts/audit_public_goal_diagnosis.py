"""Independently replay public proposal validation against retained HTTP evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.runner import save, valid_usage
from smarthome_agent_rl.execution.goals import (parse_proposal, parse_review, public_messages,
    review_messages, extraction_schema, review_schema)
from scripts.run_public_goal_diagnosis import evaluate, request


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_record(config, item, record, *, seed, actor):
    problems = []
    if (record['task_id'], record['seed'], record['actor_id']) != (item['task']['id'], seed, actor['id']):
        problems.append('Record task/seed/actor identity differs')
    public = item['public']
    calls = record['calls']
    if not calls:
        return problems + ['Extraction request missing']
    body = request(config, public_messages(public['query'], public['current_time']), extraction_schema(), seed=seed)
    expected = [(actor['endpoint'], body)]
    parsed = None
    if calls[0]['error'] is None:
        try:
            parsed = parse_proposal(public['query'], public['current_time'], calls[0]['text'])
        except (ValueError, TypeError, KeyError):
            pass  # Invalid model output is retained, not relabelled as an audit error.
    if record['proposal'] != parsed:
        problems.append('Exported proposal differs from raw model output/public clock')
    parsed_review = None
    if parsed is not None:
        body = request(config, review_messages(public['query'], public['current_time'], parsed),
                       review_schema(), seed=config['review_seed'], review=True)
        expected.append((config['review_endpoint'], body))
        if len(calls) > 1 and calls[1]['error'] is None:
            try:
                parsed_review = parse_review(calls[1]['text'])
            except (ValueError, TypeError, KeyError):
                pass
    if record['review'] != parsed_review:
        problems.append('Exported review differs from raw review response')
    if len(calls) != len(expected):
        problems.append('Missing review request or unexpected extra model call')
    for call, (endpoint, body) in zip(calls, expected):
        if call['endpoint'] != endpoint or call['body'] != body:
            problems.append('HTTP request differs from frozen public input/schema/model/seed/generation')
        if call['error'] is None:
            response = call['response']
            raw = json.loads(call['raw_response'])
            if response != raw or response['model'] != body['model'] or \
                    response['choices'][0]['message']['content'] != call['text'] or response.get('usage') != call['usage']:
                problems.append('Parsed response/content/usage differs from raw HTTP receipt')
    return problems


def audit(run):
    run = Path(run)
    manifest = read(run / 'artifact_manifest.json')
    problems = []
    for name, expected in manifest.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append('Artifact missing/outside root or SHA mismatch: ' + name)
    protocol = read(run / 'protocol.json')
    config = protocol['config']
    records = []
    expected_paths = set()
    for seed in config['actor_seeds']:
        for index, item in enumerate(protocol['inputs']):
            path = run / 'records' / f'seed{seed}' / item['task']['id'] / 'evidence.json'
            expected_paths.add(path.resolve())
            record = read(path)
            records.append(record)
            problems.extend(audit_record(config, item, record, seed=seed, actor=config['actors'][index % 4]))
    if expected_paths != {p.resolve() for p in (run / 'records').rglob('evidence.json')}:
        problems.append('Record coverage differs from frozen task/seed schedule')
    calculated = evaluate(config, records)
    report = read(run / 'report.json')
    if any(report[key] != value for key, value in calculated.items()):
        problems.append('Reported counts/cost/gates differ from complete HTTP records')
    manual = read(run / 'manual-review.json')
    expected_cases = [(r['task_id'], r['seed']) for r in records if r['task_id'] in config['manual_review']['task_ids']]
    if [(r['task_id'], r['seed']) for r in manual['cases']] != expected_cases:
        # records traversal order is manifest then seed; exported manual order
        # is sorted task IDs. Compare sets, retaining duplicate detection.
        actual_cases = [(r['task_id'], r['seed']) for r in manual['cases']]
        if len(actual_cases) != len(set(actual_cases)) or set(actual_cases) != set(expected_cases):
            problems.append('Manual review selection differs from frozen cases')
    probes = read(run / 'services/probe-receipts.json')
    cost = {'requests': len(probes), 'tokens': sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
            'errors': sum(p['error'] is not None for p in probes), 'missing_usage': sum(not valid_usage(p['usage']) for p in probes)}
    result = {'verified': not problems, 'problems': problems, 'files': len(manifest),
        'records': len(records), 'unique_public_tasks': len(protocol['inputs']), 'probe_cost': cost,
        'model_cost': calculated['cost'], 'source_commit': protocol['source_commit'],
        'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'HTTP identity,raw output/provenance/graph,coverage and cost only; manual semantic review still required'}
    save(run / 'independent-audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    print(json.dumps(result))
    if not result['verified']:
        raise SystemExit(1)
