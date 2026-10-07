"""Reconstruct N80 requests, raw HTTP, representation checks and all costs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_effect_evidence_ablation import prepare
from scripts.run_evidence_consistency_audit import check_manifest
from smarthome_agent_rl.benchmarks.runner import save, valid_usage
from smarthome_agent_rl.effect_evidence_ablation import ARMS, PROMPT_BY_ARM, request, evaluate
from smarthome_agent_rl.semantic_diagnosis import parse


def verify_record(row, item, arm, actor, config):
    call = row['call']
    if (row['id'], row['arm'], row['actor_id'], call['endpoint']) != (item['id'], arm, actor['id'], actor['endpoint']) or call['body'] != request(config, item, arm):
        raise ValueError('Request identity/body differs')
    decision, error = None, None
    if call['error'] is None:
        raw = json.loads(call['raw_response']); choice = raw['choices'][0]
        if raw != call['response'] or raw['model'] != config['model'] or choice['message']['content'] != call['text'] or raw.get('usage') != call['usage'] or choice.get('finish_reason') != call['finish_reason']:
            raise ValueError('Original raw HTTP/text/usage/finish differs')
        try: decision = parse(call['text'])
        except (ValueError, TypeError, KeyError) as exc: error = str(exc)
    if decision != row['decision'] or error != row['parse_error']:
        raise ValueError('Derived decision/parse failure differs')


def audit(run, parent=None, n76=None, n78=None, *, write_receipt=True):
    run = Path(run)
    count = check_manifest(run, 'artifact_manifest.json', sha(run/'artifact_manifest.json'))
    protocol = read(run/'protocol.json'); config = protocol['config']
    if config != read(ROOT/'configs/effect-evidence-ablation.json'):
        raise ValueError('Frozen repository config differs')
    inputs, labels, coverage, schemas, checked = prepare(config, parent, n76, n78)
    for name, value in {'inputs': inputs, 'labels': labels, 'coverage': coverage, 'schemas': schemas,
                        'parent_audit': checked, 'prompts': PROMPT_BY_ARM}.items():
        if protocol[name] != value: raise ValueError('Frozen source/representation differs: '+name)
    records = []; expected = set()
    for index, item in enumerate(inputs):
        for arm in ARMS:
            path = run/'records'/f'{index:03d}'/(arm+'.json'); expected.add(path.resolve())
            row = read(path); verify_record(row, item, arm, config['actors'][index % 4], config); records.append(row)
    if expected != {p.resolve() for p in (run/'records').rglob('*.json')}:
        raise ValueError('Missing/unexpected records')
    calculated = evaluate(config, inputs, records, labels); report = read(run/'report.json')
    if any(report[key] != value for key, value in calculated.items()):
        raise ValueError('Reported metrics/screen/cost differ')
    resources = read(run/'services/lifecycle.json'); deployments = read(run/'services/deployments.json')
    if report['resources'] != resources or resources['error'] is not None or resources['stopped_pids'] != [r['pid'] for r in deployments] or [r['id'] for r in deployments] != [a['id'] for a in config['actors']]:
        raise ValueError('Service lifecycle/cleanup differs')
    probes = read(run/'services/probe-receipts.json')
    probe_cost = {'requests': len(probes), 'failed': sum(p['error'] is not None for p in probes),
                  'tokens': sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
                  'missing_usage': sum(not valid_usage(p['usage']) for p in probes)}
    if probe_cost != resources['probes'] or probe_cost['requests'] != 4 or probe_cost['failed'] or probe_cost['missing_usage']:
        raise ValueError('Probe cost differs')
    receipt = {'verified': True, 'artifacts': count, 'records': len(records), 'coverage': coverage,
               'new_tokens': calculated['new_tokens'], 'probes': probe_cost, 'source_commit': protocol['source_commit'],
               'parent_audit': checked, 'auditor_sha256': sha(Path(__file__)),
               'scope': 'Source/prompt/schema/HTTP/engineering consistency only;not semantic truth or native gain'}
    if write_receipt: save(run/'independent-audit.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', type=Path); parser.add_argument('--n76', type=Path); parser.add_argument('--n78', type=Path)
    args = parser.parse_args(); print(json.dumps(audit(args.run, args.parent, args.n76, args.n78)))
