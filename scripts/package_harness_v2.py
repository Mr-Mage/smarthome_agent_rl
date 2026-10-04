"""Verify preserved rounds and package compact review receipts; raw episodes stay in runs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import episode_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/harness-v2')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    rounds = ['runs/concurrency-capacity/acceptance-v2/' + phase for phase in ('dev', 'final')]
    rounds += ['runs/harness-v2/primary-v1/' + phase for phase in ('smoke', 'modules')]
    rounds += ['runs/harness-v2/primary-v2/' + phase for phase in ('smoke', 'modules', 'integration', 'final')]
    receipts, verified = [], []
    for name in rounds:
        run = ROOT / name
        result = subprocess.run([sys.executable, ROOT / 'scripts/verify_benchmark.py', run],
            cwd=ROOT, check=True, capture_output=True, text=True)
        row = json.loads(result.stdout.strip())
        verified.append({'run': name, **row, 'report_sha256': digest(run / 'report.json')})
        for filename in ('report.json', 'report.md', 'artifact_manifest.json', 'protocol.json',
                         'source_identity.json', 'simulator_identity.json'):
            receipts.append(run / filename)
    primary = ROOT / 'runs/harness-v2/primary-v2'
    frozen = json.loads((primary / 'final-freeze.json').read_text())
    sim = ROOT / frozen['upstream']['SimuHome']['path']
    lightning = ROOT / frozen['upstream']['agent-lightning']['path']
    def git(path, *arguments):
        return subprocess.check_output(['git', '-C', str(path), *arguments])
    if git(sim, 'rev-parse', 'HEAD').decode().strip() != frozen['upstream']['SimuHome']['commit'] or git(sim, 'status', '--porcelain'):
        raise RuntimeError('Simulator differs from frozen identity')
    if git(lightning, 'rev-parse', 'HEAD').decode().strip() != frozen['upstream']['agent-lightning']['commit'] or (
            hashlib.sha256(git(lightning, 'diff', 'HEAD')).hexdigest() != frozen['lightning_user_patch_sha256']):
        raise RuntimeError('Lightning or user patch differs from frozen identity')
    final = primary / 'final'
    protocol = json.loads((final / 'protocol.json').read_text())
    successes = defaultdict(dict)
    categories = defaultdict(lambda: {'episodes': 0, 'successes': 0})
    for item in protocol['schedule']:
        for variant in item['variants']:
            summary = json.loads((episode_directory(final, item, variant) / 'summary.json').read_text())
            seed, task = item['actor_seed'], item['task']
            successes[variant, task['id']][seed] = summary['success']
            key = variant + ':' + str(seed) + ':' + task['query_type'] + ':' + task['case']
            categories[key]['episodes'] += 1
            categories[key]['successes'] += summary['success']
    reliability = {v: {'independent_tasks': 192,
        'success_all_three': sum(all(successes[v, task][seed] for seed in protocol['actor_seeds'])
                                 for arm, task in successes if arm == v),
        'success_any_seed': sum(any(successes[v, task].values()) for arm, task in successes if arm == v)}
        for v in protocol['variants']}
    readout = {'primary_actor_seed': 42, 'categories': dict(categories), 'reliability': reliability,
        'note': 'All-three success is empirical repeat reliability, not 576 independent tasks or a new primary test.'}
    (output / 'final-readout.json').write_text(json.dumps(readout, indent=2))
    receipts += [output / 'final-readout.json']
    for root in (ROOT / 'runs/concurrency-capacity/acceptance-v2', ROOT / 'runs/harness-v2/primary-v1', primary):
        receipts.extend(root.glob('*.json'))
        for sub in ('inventory', 'module-audit'):
            receipts.extend((root / sub).glob('*.json'))
    receipts.extend((ROOT / 'runs/harness-v2/N9').glob('*.json'))
    for directory in ('configs/benchmark-v2', 'docs/nodes', 'patches/SimuHome'):
        receipts.extend((ROOT / directory).glob('*'))
    receipts += [ROOT / 'dependencies.lock.json', ROOT / 'configs/harness-v2-modules.json',
        ROOT / 'configs/harness-v2-protocol.json', ROOT / 'docs/实验报表.md', ROOT / 'AGENTS.md']
    receipts = sorted({p for p in receipts if p.is_file()})
    review = output / 'review-evidence.tar.gz'
    with tarfile.open(review, 'w:gz') as archive:
        for path in receipts:
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    index = {'complete': True, 'rounds': verified,
        'total_verified_files': sum(r['files'] for r in verified),
        'raw_episodes_preserved': True, 'raw_root': str(ROOT / 'runs'),
        'archive_scope': 'Review receipts/configs/identities/metrics; raw episode evidence remains at recorded run paths',
        'receipts_sha256': {str(p.relative_to(ROOT)): digest(p) for p in receipts},
        'archive': review.name, 'archive_sha256': digest(review), 'archive_bytes': review.stat().st_size,
        'upstream_and_lightning_user_patch_unchanged': True}
    (output / 'archive-verification.json').write_text(json.dumps(index, indent=2))
    print(json.dumps({'complete': True, 'files': index['total_verified_files'],
        'archive_bytes': index['archive_bytes'], 'reliability': reliability}))


if __name__ == '__main__':
    main()
