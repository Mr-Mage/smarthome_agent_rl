"""Freeze a clean project commit and verified model/runtime/manifests before final."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest


def git(directory, *arguments):
    return subprocess.check_output(['git', '-C', str(directory), *arguments], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-run', required=True)
    parser.add_argument('--output', default='work/harness-mvp/final-freeze.json')
    parser.add_argument('--services', default='work/harness-mvp/services-v3')
    parser.add_argument('--inventory', default='work/harness-mvp/inventory-v2')
    parser.add_argument('--config', default='configs/harness-mvp.json')
    parser.add_argument('--final-manifest')
    args = parser.parse_args()
    if git(ROOT, 'status', '--porcelain'):
        raise RuntimeError('Commit all implementation/configuration changes before freezing')
    config = json.loads((ROOT / args.config).read_text())
    dev = ROOT / args.dev_run
    manifest_dir = config.get('manifest_dir', 'configs/benchmark-mvp')
    final_manifest = ROOT / (args.final_manifest or f'{manifest_dir}/final.json')
    dev_manifest = ROOT / f'{manifest_dir}/dev.json'
    seeds = config.get('actor_seeds', [config['model_seed']])
    dev_count = len(json.loads(dev_manifest.read_text())['tasks']) * len(seeds)
    report = json.loads((dev / 'report.json').read_text())
    if not report['verified'] or report['phase'] != 'dev' or set(report['arms']) != set(config['variants_dev']) or any(
            arm['episodes'] != dev_count or arm['evaluator_errors'] for arm in report['arms'].values()):
        raise ValueError('Complete verified dev protocol required before final')
    protocol = json.loads((dev / 'protocol.json').read_text())
    if protocol['config'] != config or protocol['manifest_sha256'] != digest(dev_manifest) or protocol.get('actor_seeds', seeds) != seeds:
        raise ValueError('Dev configuration or task manifest changed')
    source_identity = json.loads((dev / 'source_identity.json').read_text())
    source_changes = {name: {'dev_sha256': expected, 'frozen_sha256': digest(ROOT / name)}
        for name, expected in source_identity.items() if digest(ROOT / name) != expected}
    reporting_only = {'scripts/report_benchmark.py', 'scripts/freeze_experiment.py'}
    if set(source_changes) - reporting_only:
        raise RuntimeError('Runtime policy changed after dev started')
    services = ROOT / args.services
    if not (services / 'ready').exists():
        raise RuntimeError('Four-card service supervisor is not ready')
    service_config = json.loads((services / 'config.json').read_text())
    if service_config != config:
        raise ValueError('Running service allocation/model configuration differs from protocol')
    lock = json.loads((ROOT / 'dependencies.lock.json').read_text())
    sim = ROOT / lock['SimuHome']['path']
    lightning = ROOT / lock['agent-lightning']['path']
    if git(sim, 'rev-parse', 'HEAD') != lock['SimuHome']['commit'] or git(sim, 'status', '--porcelain'):
        raise RuntimeError('SimuHome must remain at its pristine locked revision')
    if git(lightning, 'rev-parse', 'HEAD') != lock['agent-lightning']['commit']:
        raise RuntimeError('Lightning revision changed')
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Never overwrite an existing frozen experiment')
    output.parent.mkdir(parents=True, exist_ok=True)
    inventories = {name: json.loads((ROOT / args.inventory / f'{name}-inventory.json').read_text())
                   for name in ('actor', 'judge')}
    record = {'schema': 'harness-final-freeze-v1', 'commit': git(ROOT, 'rev-parse', 'HEAD'),
        'config_sha256': digest(ROOT / args.config),
        'manifest_sha256': digest(final_manifest), 'manifest_path': str(final_manifest.relative_to(ROOT)),
        'actor_seeds': seeds,
        'dev_report_sha256': digest(dev / 'report.json'), 'dev_run': args.dev_run,
        'dev_commit': protocol['commit'], 'dev_runtime_source_identity_matches': True,
        'reporting_only_source_changes': source_changes,
        'model_identities': {k: v['identity'] for k, v in inventories.items()},
        'upstream': lock, 'lightning_user_patch_sha256': hashlib.sha256(
            subprocess.check_output(['git', '-C', str(lightning), 'diff', 'HEAD'])).hexdigest(),
        'services_config_sha256': digest(services / 'config.json'),
        'judge_probe_sha256': digest(services / 'inference-probe.json'),
        'primary_comparisons': [v + '-' + config['variants_final'][0] for v in config['variants_final'][1:]], 'multiplicity': 'Holm',
        'unfinished_task_score': 0, 'infrastructure_policy': 'Stop; preserve entire failed round; no selective rerun',
        'judge_panel_independent_models': False,
        'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version',
                                            '--format=csv,noheader'], text=True).splitlines()}
    output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'frozen_commit': record['commit'], 'final_tasks': 192, 'arms': config['variants_final']}))


if __name__ == '__main__':
    main()
