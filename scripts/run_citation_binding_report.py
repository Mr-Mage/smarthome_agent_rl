"""Frozen all156 N83 receipts; no model/simulator/new benchmark access."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_citation_review import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_evidence_consistency_audit import check_manifest
from smarthome_agent_rl.benchmarks.runner import save
from smarthome_agent_rl.citation_binding_report import evaluate
from smarthome_agent_rl.semantic_context import digest


def prepare(config, parent=None, n82=None, n80=None, n79=None, n76=None, n78=None):
    parent = Path(parent or ROOT / config['parent_run'])
    count = check_manifest(parent, 'collection-manifest.json', config['parent_collection_sha256'])
    checked = audit_parent(parent, n82, n80, n79, n76, n78, write_receipt=False)
    protocol = read(parent / 'protocol.json'); inputs = protocol['inputs']
    if digest(inputs) != config['inputs_sha256'] or len(inputs) != config['paired_inputs']:
        raise ValueError('Frozen public contexts differ')
    sources = []
    for index, item in enumerate(inputs):
        for arm in config['arms']:
            path = parent / 'records' / f'{index:03d}' / (arm + '.json'); row = read(path)
            if (row['id'], row['arm']) != (item['id'], arm):
                raise ValueError('Source identity differs')
            sources.append({'id': item['id'], 'task_id': item['task_id'], 'origin': item['origin'], 'arm': arm,
                'context': item['context'], 'row': row,
                'source_file': str(path.relative_to(parent)).replace('\\', '/'), 'source_sha256': sha(path)})
    if digest(sources) != config['sources_sha256']:
        raise ValueError('Frozen raw sources differ')
    return sources, {'artifacts': count, 'audit': checked}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    began = time.monotonic(); config = read(args.config)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit source/config before diagnosis')
    sources, checked = prepare(config); preparation = time.monotonic() - began
    args.output.mkdir(parents=True, exist_ok=False); report, records = evaluate(config, sources)
    save(args.output / 'protocol.json', {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'config': config, 'parent_checks': checked, 'preparation_seconds': preparation})
    save(args.output / 'records.json', records); save(args.output / 'report.json', report)
    save(args.output / 'timing.json', {'cpu_wall_seconds': time.monotonic() - began,
        'scope': 'Parent hashes/audit,parse,diagnosis and writes;tests/archive/transfer separate'})
    save(args.output / 'artifact_manifest.json', {str(p.relative_to(args.output)).replace('\\', '/'): sha(p)
        for p in args.output.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    print(report)


if __name__ == '__main__': main()
