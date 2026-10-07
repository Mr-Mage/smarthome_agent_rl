"""Recompute all source-bound N84 lexical panels without changing parents."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_citation_binding_report import prepare
from scripts.run_evidence_consistency_audit import check_manifest
from smarthome_agent_rl.benchmarks.runner import save
from smarthome_agent_rl.citation_binding_report import evaluate


def audit(run, parent=None, n82=None, n80=None, n79=None, n76=None, n78=None, *, write_receipt=True):
    run = Path(run); protocol = read(run / 'protocol.json'); config = protocol['config']
    if config != read(ROOT / 'configs/citation-binding-report.json'):
        raise ValueError('Frozen repository config differs')
    count = check_manifest(run, 'artifact_manifest.json', sha(run / 'artifact_manifest.json'))
    sources, checked = prepare(config, parent, n82, n80, n79, n76, n78)
    report, records = evaluate(config, sources)
    if report != read(run / 'report.json') or records != read(run / 'records.json') or checked != protocol['parent_checks']:
        raise ValueError('Source/coverage/diagnostic panels differ')
    receipt = {'verified': True, 'artifacts': count, 'records': len(records), 'parent_checks': checked,
        'source_commit': protocol['source_commit'], 'auditor_sha256': sha(Path(__file__)),
        'new_model_requests': 0, 'scope': 'Source-bound mechanical coverage audit,not independent semantic truth'}
    if write_receipt: save(run / 'independent-audit.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args(); print(audit(args.run))
