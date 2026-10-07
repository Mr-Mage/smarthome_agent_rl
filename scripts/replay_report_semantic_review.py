"""Read-only N76 HTTP replay; no new inference or source outcome labels."""
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
from smarthome_agent_rl.benchmarks.runner import save
from smarthome_agent_rl.report_semantic_review import ReportSemanticReviewer
from smarthome_agent_rl.semantic_verifier import VerificationContext


def replay(parent):
    parent = Path(parent)
    manifest = read(parent / 'artifact_manifest.json')
    for name, expected in manifest.items():
        path = (parent / name).resolve()
        if not path.is_relative_to(parent.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError('Original N76 artifact differs: ' + name)
    protocol = read(parent / 'protocol.json');config = protocol['config']
    records = [];started = time.monotonic()
    for index, item in enumerate(protocol['inputs']):
        original = read(parent / 'records' / f'{index:03d}' / '9b_typed.json')
        captured = []
        def transport(endpoint, body, timeout):
            captured.append(body)
            if endpoint != original['call']['endpoint'] or body != original['call']['body'] or timeout != config['request_timeout']:
                raise AssertionError('Replay request differs from frozen N76 HTTP')
            return copy.deepcopy(original['call'])
        reviewer = ReportSemanticReviewer(config, original['call']['endpoint'], transport=transport)
        reviewer.verify(VerificationContext(**item['context']))
        row = reviewer.records[0]
        if len(captured) != 1 or row['call'] != original['call'] or row['model_decision'] != original['decision']:
            raise ValueError('Replay changed original response/decision')
        records.append({'id': item['id'], **row})
    result = {'parent_manifest_sha256': sha(parent / 'artifact_manifest.json'),
              'original_artifacts': len(manifest), 'proposals': len(records),
              'accepted_model_outputs': sum(r['accepted_model_output'] for r in records),
              'model_verdicts': dict(Counter(r['model_decision']['verdict'] if r['model_decision'] else 'INVALID' for r in records)),
              'report_verdicts': dict(Counter(r['result']['verdict'] for r in records)),
              'fallbacks': dict(Counter(r['fallback_reason'] for r in records if r['fallback_reason'])),
              'reused_tokens': sum(r['call']['usage']['total_tokens'] for r in records),
              'new_model_requests': 0, 'new_tokens': 0, 'reserved_gpu_seconds': 0,
              'replay_cpu_wall_seconds': time.monotonic() - started,
              'scope': 'Receipt/failure-isolation replay only;historical tokens not new cost;not semantic accuracy or N76 screen admission'}
    return result, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True);args = parser.parse_args()
    report, rows = replay(args.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'protocol.json', {'source_commit': subprocess.check_output(['git', '-c', 'safe.directory='+str(ROOT), 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'parent': str(args.parent), 'parent_manifest_sha256': report['parent_manifest_sha256']})
    save(args.output / 'report.json', report);save(args.output / 'records.json', rows)
    print(json.dumps(report))
