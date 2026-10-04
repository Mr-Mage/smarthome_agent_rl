"""Verify complete stages and archive receipts; retain all raw and failed episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest


def usage(row):
    response = row.get('response', {})
    value = response.get('usage', {}) or {}
    return {'input_tokens': value.get('prompt_tokens', 0), 'output_tokens': value.get('completion_tokens', 0),
            'total_tokens': value.get('total_tokens', 0)}


def costs(run):
    result = {'run': str(run.relative_to(ROOT)), 'actor': {'calls': 0}, 'judge': {'calls': 0}, 'preflight_tokens': 0}
    for role, name in (('actor', 'model_calls.json'), ('judge', 'judge_calls.json')):
        total = {'calls': 0, 'calls_without_usage': 0,
                 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        for path in run.rglob(name):
            for row in json.loads(path.read_text()):
                total['calls'] += 1
                if not row.get('response', {}).get('usage'):
                    total['calls_without_usage'] += 1
                for key, value in usage(row).items():
                    total[key] += value
        result[role] = total
    preflight = run / 'metadata-preflight.json'
    if preflight.exists():
        result['preflight_tokens'] = usage(json.loads(preflight.read_text()))['total_tokens']
    probe = run / 'services/inference-probe.json'
    result['service_probe'] = {'actor_tokens': 0, 'judge_tokens': 0}
    if probe.exists():
        for row in json.loads(probe.read_text())['results']:
            key = 'judge_tokens' if row['service'] == 'judge' else 'actor_tokens'
            result['service_probe'][key] += usage(row)['total_tokens']
    resource = run / 'resource-cost.json'
    result['resource'] = json.loads(resource.read_text()) if resource.exists() else None
    result['a800_allocation'] = 'Shared permanent service; not allocated to this round; request tokens above, resource cost unallocated'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--stages', nargs='+', required=True)
    parser.add_argument('--extra', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    receipts, verified = set(), []
    for name in args.stages:
        stage = ROOT / name
        result = subprocess.run([sys.executable, ROOT / 'scripts/verify_benchmark.py', stage],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        verified.append({'stage': name, **json.loads(result.stdout)})
        for filename in ('report.json', 'report.md', 'artifact_manifest.json', 'protocol.json',
                         'source_identity.json', 'simulator_identity.json'):
            receipts.add(stage / filename)
    all_costs = []
    for name in args.runs:
        run = ROOT / name
        all_costs.append(costs(run))
        receipts.update(p for p in run.glob('*.json') if p.is_file())
        for filename in ('config.json', 'external-judge.json', 'timing.json', 'failure.json', 'services.json'):
            if (run / 'services' / filename).exists():
                receipts.add(run / 'services' / filename)
    receipts.update(ROOT / name for name in args.extra)
    for path in receipts:
        if not path.is_file():
            raise FileNotFoundError(path)
    cost_path = output / 'all-costs.json'
    cost_path.write_text(json.dumps({'runs': all_costs, 'raw_partial_calls_included': True,
        'allocated_actor_gpu_seconds': sum(r['resource']['allocated_actor_gpu_seconds'] for r in all_costs if r['resource']),
        'scope': 'Recorded costs of all declared node attempts, including partial failed trajectories, startup, probes, preflight and driver cleanup; no monetary conversion',
        'unmeasured': ['Manual cleanup extra CPU time', 'Shared A800 resource allocation',
                       'Tokens of calls without response usage, if any; known token sums are lower bounds in that case']}, indent=2))
    receipts.add(cost_path)
    review = output / 'review-evidence.tar.gz'
    with tarfile.open(review, 'w:gz') as archive:
        for path in sorted(receipts):
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    index = {'verified_stages': verified, 'total_verified_files': sum(v['files'] for v in verified),
        'receipts_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(receipts)},
        'archive_sha256': digest(review), 'archive_bytes': review.stat().st_size,
        'raw_evidence_preserved': True, 'runs': args.runs,
        'archive_scope': 'Review receipts only; every raw/failed episode remains in runs',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (output / 'archive-verification.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'receipts': len(receipts), 'verified_files': index['total_verified_files'], 'archive_bytes': review.stat().st_size}))
