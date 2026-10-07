"""Frozen all-arm N80 receipt diagnosis; no inference or simulator access."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_effect_evidence_ablation import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_evidence_consistency_audit import check_manifest
from smarthome_agent_rl.benchmarks.runner import save
from smarthome_agent_rl.declared_evidence_report import declared_evidence_report
from smarthome_agent_rl.effect_evidence_ablation import ARMS
from smarthome_agent_rl.semantic_context import digest


def prepare(config, parent=None, n79=None, n76=None, n78=None):
    parent = Path(parent or ROOT/config['parent_run'])
    count = check_manifest(parent, 'collection-manifest.json', config['parent_collection_sha256'])
    checked = audit_parent(parent, n79, n76, n78, write_receipt=False)
    inputs = read(parent/'protocol.json')['inputs']
    if digest(inputs) != config['inputs_sha256'] or len(inputs) != config['contexts'] or config['arms'] != ['historical', *ARMS]:
        raise ValueError('Frozen cohort differs')
    sources = []
    for index, item in enumerate(inputs):
        sources.append({'id': item['id'], 'origin': item['origin'], 'arm': 'historical',
                        'context': item['context'], 'decision': item['decision'],
                        'source_file': 'protocol.json', 'source_sha256': sha(parent/'protocol.json')})
        for arm in ARMS:
            path = parent/'records'/f'{index:03d}'/(arm+'.json'); row = read(path)
            sources.append({'id': item['id'], 'origin': item['origin'], 'arm': arm,
                            'context': item['context'], 'decision': row['decision'],
                            'source_file': str(path.relative_to(parent)).replace('\\', '/'), 'source_sha256': sha(path)})
    if len(sources) != config['records'] or digest(sources) != config['sources_sha256']:
        raise ValueError('Frozen all-arm receipts differ')
    return sources, {'parent_artifacts': count, 'parent_audit': checked}


def evaluate(config, sources):
    records = [{k: item[k] for k in ('id', 'origin', 'arm', 'source_file', 'source_sha256')} |
               {'context_sha256': digest(item['context']), 'report': declared_evidence_report(item['context'], item['decision'], item['arm'])}
               for item in sources]
    arms = {}
    for arm in config['arms']:
        rows = [row['report'] for row in records if row['arm'] == arm]
        original = lambda row: row['original']['verdict'] if row['original'] else 'INVALID'
        arms[arm] = {'records': len(rows), 'available': sum(row['available'] for row in rows),
            'unavailable': sum(not row['available'] for row in rows),
            'projected_label_change_records': sum(bool(row['changed_dimensions']) for row in rows),
            'projected_label_changes': dict(Counter(name for row in rows for name in row['changed_dimensions'])),
            'original_to_declared_projection': dict(Counter(original(row)+'->'+(row['declared_evidence_verdict'] or 'UNAVAILABLE') for row in rows)),
            'shared_target_quote_records': sum(bool(row['shared_target_quotes']) for row in rows),
            'explicit_equivalent_clock_disagreement_records': sum(any(a['assumed_encodings_disagree'] for a in row['explicit_clock_equivalences']) for row in rows),
            'cited_quote_dual_anchor_disagreement_records': sum(any(len({a['expected_encoding'] for a in step['anchors']}) > 1 for step in row['checks']['time_anchors']) for row in rows),
            'literal_catalog_room_coverage': {'steps': sum(len(row['identity_panels']) for row in rows),
                'room_word_present': sum(panel['quote_mentions_observed_room'] is True for row in rows for panel in row['identity_panels']),
                'room_word_absent': sum(panel['quote_mentions_observed_room'] is False for row in rows for panel in row['identity_panels']),
                'room_unobserved': sum(panel['quote_mentions_observed_room'] is None for row in rows for panel in row['identity_panels'])},
            'cost_origin': 'immutable historical receipts;no new inference'}
    complete = len(records) == config['records'] and len({(r['id'], r['arm']) for r in records}) == len(records)
    result = {'complete': complete, 'records': len(records), 'arms': arms, 'new_model_requests': 0,
              'new_tokens': 0, 'reserved_gpu_seconds': 0, 'changed_original_decisions': 0,
              'native_admitted': False, 'semantic_status': 'UNVERIFIED',
              'scope': 'Post-hoc engineering diagnosis only. Projection consistency is by construction,not semantic improvement,accuracy,false-rejection rate,SR or admission. Invalid evidence remains unavailable.'}
    return result, records


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path); args = parser.parse_args()
    started = time.monotonic(); config = read(args.config); sources, checked = prepare(config)
    preparation = time.monotonic()-started
    args.output.mkdir(parents=True, exist_ok=False); report, records = evaluate(config, sources)
    save(args.output/'protocol.json', {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
         'config': config, 'parent_checks': checked, 'preparation_seconds': preparation})
    save(args.output/'records.json', records); save(args.output/'report.json', report)
    save(args.output/'timing.json', {'cpu_wall_seconds': time.monotonic()-started,
         'scope': 'Parent hashes/audit,receipt parse,projection,diagnosis and writes;tests/archive/transfer separate'})
    save(args.output/'artifact_manifest.json', {str(p.relative_to(args.output)).replace('\\','/'):sha(p)
         for p in args.output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    print(report)


if __name__ == '__main__': main()
