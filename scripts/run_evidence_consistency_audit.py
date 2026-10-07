"""Frozen read-only audit of every N76/N78 reviewer output; no inference."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from scripts.run_action_semantic_diagnosis import read, sha
from smarthome_agent_rl.benchmarks.runner import save, valid_usage
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.evidence_consistency import audit_claims
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_diagnosis import parse
from smarthome_agent_rl.typed_semantic_review import request


def check_manifest(parent, name, expected):
    parent = Path(parent).resolve();path = parent / name
    if sha(path) != expected:
        raise ValueError('Frozen parent manifest differs: ' + str(path))
    manifest = read(path)
    for relative, checksum in manifest.items():
        original = (parent / relative).resolve()
        if not original.is_relative_to(parent) or not original.is_file() or sha(original) != checksum:
            raise ValueError('Original parent artifact differs: ' + relative)
    return len(manifest)


def prepare(config, n76=None, n78=None):
    parents = {name: Path(value or ROOT / config['parents'][name]['run']) for name, value in
               (('n76', n76), ('n78', n78))}
    counts = {name: check_manifest(parent, config['parents'][name]['manifest'],
                                  config['parents'][name]['manifest_sha256']) for name, parent in parents.items()}
    inputs = [];old = read(parents['n76'] / 'protocol.json')
    for index, item in enumerate(old['inputs']):
        path = parents['n76'] / 'records' / f'{index:03d}' / '9b_typed.json';row = read(path)
        inputs.append({'id': 'n76:' + item['id'], 'origin': 'n76', 'task_id': item['id'].split(':')[0],
                       'context': item['context'], 'call': row['call'], 'decision': row['decision'],
                       'source_file': str(path.relative_to(parents['n76'])).replace('\\', '/'),
                       'source_sha256': sha(path)})
    stage = parents['n78'] / 'calibration';native = read(stage / 'protocol.json')
    review = native['config']['semantic_review']
    if any(review[key] != old['config'][key] for key in ('model', 'model_seed', 'generation', 'request_timeout')):
        raise ValueError('Frozen historical reviewer generation settings differ')
    for item in native['schedule']:
        path = episode_directory(stage, item, 'Candidate') / 'semantic_review_calls.json'
        if not path.exists():
            continue  # Frozen zero-mutation episodes; parent coverage retained below.
        for row in read(path):
            inputs.append({'id': f"n78:{item['task']['id']}:seed{item['actor_seed']}:r{row['sequence']:06d}",
                           'origin': 'n78', 'task_id': item['task']['id'], 'context': row['context'],
                           'call': row['call'], 'decision': row['model_decision'],
                           'source_file': str(path.relative_to(parents['n78'])).replace('\\', '/'),
                           'source_sha256': sha(path)})
    if dict(Counter(r['origin'] for r in inputs)) != config['records_by_origin'] or \
            len({r['id'] for r in inputs}) != len(inputs):
        raise ValueError('Original cohort has missing or duplicated records')
    for item in inputs:
        call = item['call'];decision = None
        if call['body'] != request(old['config'], {'context': item['context']}):
            raise ValueError('Historical context/schema/prompt/generation differs')
        if call['error'] is None:
            raw = json.loads(call['raw_response']);choice = raw['choices'][0]
            if raw != call['response'] or choice['message']['content'] != call['text'] or \
                    raw.get('usage') != call['usage'] or choice.get('finish_reason') != call['finish_reason']:
                raise ValueError('Original raw HTTP differs')
            try:
                decision = parse(call['text'])
            except (ValueError, TypeError, KeyError):
                pass
        if decision != item['decision']:
            raise ValueError('Original model decision changed')
    if digest(inputs) != config['inputs_sha256']:
        raise ValueError('Frozen reviewer cohort differs')
    coverage = {'n76_source_episodes': len(old['coverage']),
                'n78_source_episodes': native['expected_episodes'],
                'n78_candidate_zero_review_episodes': sum(not (
                    episode_directory(stage, item, 'Candidate') / 'semantic_review_calls.json').exists()
                    for item in native['schedule']),
                'tasks_with_reviews': len({r['task_id'] for r in inputs}),
                'distinct_contexts': len({digest(r['context']) for r in inputs}),
                'parent_artifacts_verified': counts}
    return inputs, coverage


def evaluate(inputs):
    records = [{'id': r['id'], 'origin': r['origin'], 'context_sha256': digest(r['context']),
                'audit': audit_claims(r['context'], r['decision'])} for r in inputs]
    arms = {}
    for origin in ('n76', 'n78'):
        rows = [r for r in records if r['origin'] == origin]
        source = [r for r in inputs if r['origin'] == origin]
        arms[origin] = {'records': len(rows), 'parsed_outputs': sum(r['decision'] is not None for r in source),
            'schema_conformant': sum(r['audit']['schema_conformant'] for r in rows),
            'literal_evidence_valid': sum(r['audit']['literal_evidence']['valid'] for r in rows),
            'flagged_records': sum(bool(r['audit']['flags']) for r in rows),
            'flag_kinds': dict(Counter(f['kind'] for r in rows for f in r['audit']['flags'])),
            'flagged_records_by_class': {kind: sum(any(f['class'] == kind for f in r['audit']['flags']) for r in rows)
                for kind in ('internal_conflict', 'unsubstantiated_dimension', 'arithmetic_binding_review')},
            'anchor_statuses': dict(Counter(s['status'] for r in rows for s in r['audit']['time_anchors'])),
            'original_verdicts': dict(Counter(r['decision']['verdict'] if r['decision'] else 'INVALID' for r in source)),
            'reused_review_tokens': sum(r['call']['usage']['total_tokens'] for r in source if valid_usage(r['call']['usage']))}
    result = {'records': len(records), 'origins': arms, 'new_model_requests': 0, 'new_tokens': 0,
              'reserved_gpu_seconds': 0, 'changed_verdicts': 0, 'native_admitted': False,
              'scope': 'Step-label consistency and explicitly assumed clock arithmetic only. '
                       'No independent semantic accuracy,false rejection rate,corrected labels,Task completion or default adoption.'}
    return result, records


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True);parser.add_argument('--n76', type=Path)
    parser.add_argument('--n78', type=Path);args = parser.parse_args()
    config = read(args.config);started = time.monotonic()
    inputs, coverage = prepare(config, args.n76, args.n78)
    preparation = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=False)
    report, records = evaluate(inputs)
    save(args.output / 'protocol.json', {'source_commit': subprocess.check_output(
         ['git', '-c', 'safe.directory='+str(ROOT), 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'config': config, 'coverage': coverage, 'preparation_seconds': preparation})
    save(args.output / 'inputs.json', inputs);save(args.output / 'records.json', records)
    save(args.output / 'report.json', report)
    save(args.output / 'timing.json', {'total_cpu_wall_seconds': time.monotonic() - started,
         'scope': 'Includes hash/parse/audit/write;does not include tests or transfer/archive'})
    save(args.output / 'artifact_manifest.json', {str(p.relative_to(args.output)).replace('\\', '/'): sha(p)
         for p in args.output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    print(json.dumps(report))


if __name__ == '__main__':
    main()
