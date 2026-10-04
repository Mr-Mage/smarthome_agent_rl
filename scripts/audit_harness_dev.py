"""Evidence-only dev audit and metadata-only freezing of an unused official holdout."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import STRATA, KNOWN_EXPOSURES, digest


def freeze_holdout(root, output, seed=20261005):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Never overwrite a frozen holdout')
    exposed = set(KNOWN_EXPOSURES)
    evidence = []
    for path in sorted((root / 'runs').rglob('protocol.json')):
        for row in json.loads(path.read_text()) .get('schedule', []):
            task = row.get('task', {})
            if isinstance(task, dict) and task.get('id'):
                exposed.add(task['id'])
        evidence.append({'path': str(path.relative_to(root)), 'sha256': digest(path)})
    previous = root / 'configs/benchmark-mvp'
    for split in ('dev', 'final', 'smoke'):
        path = previous / (split + '.json')
        exposed.update(row['id'] for row in json.loads(path.read_text())['tasks'])
        evidence.append({'path': str(path.relative_to(root)), 'sha256': digest(path)})
    benchmark = root / 'deps/SimuHome/data/benchmark'
    excluded_hashes = {digest(benchmark / (name + '.json')) for name in exposed
                       if (benchmark / (name + '.json')).exists()}
    groups = defaultdict(list)
    for path in sorted(benchmark.glob('*.json')):
        # Read metadata and establish task identity; never export query/evaluation contents.
        meta = json.loads(path.read_text(encoding='utf-8'))['meta']
        sha = digest(path)
        if path.stem not in exposed and sha not in excluded_hashes:
            groups[(meta['query_type'], meta['case'])].append({'id': path.stem,
                'path': path.name, 'sha256': sha, 'query_type': meta['query_type'],
                'case': meta['case'], 'seed': meta['seed']})
    rng = random.Random(seed)
    tasks, available = [], {}
    for stratum in STRATA:
        candidates = groups[stratum][:]
        available[':'.join(stratum)] = len(candidates)
        if len(candidates) < 16:
            raise ValueError(f'Insufficient unused official tasks: {stratum}')
        rng.shuffle(candidates)
        tasks.extend(candidates[:16])
    if len({row['sha256'] for row in tasks}) != 192:
        raise ValueError('Duplicate task content in holdout')
    output.mkdir(parents=True)
    for split in ('dev', 'smoke'):
        (output / (split + '.json')).write_bytes((previous / (split + '.json')).read_bytes())
    original = json.loads((previous / 'final.json').read_text())
    record = {**original, 'sampling_seed': seed, 'tasks': tasks,
              'purpose': 'untouched-holdout-v2', 'exposure_policy': 'Exclude every historical protocol, old split, known exposure and content duplicate'}
    (output / 'final.json').write_text(json.dumps(record, indent=2) + '\n')
    (output / 'selection.json').write_text(json.dumps({'sampling_seed': seed,
        'excluded_ids': sorted(exposed), 'remaining_counts': available, 'historical_evidence': evidence,
        'dev_manifest_unchanged': digest(output / 'dev.json') == digest(previous / 'dev.json'),
        'final_manifest_sha256': digest(output / 'final.json')}, indent=2) + '\n')
    return {'tasks': len(tasks), 'available': available, 'final_sha256': digest(output / 'final.json')}


def audit(run):
    run = Path(run)
    report = json.loads((run / 'report.json').read_text())
    if not report['verified'] or report['phase'] != 'dev':
        raise ValueError('Only a verified development run is eligible for diagnosis')
    counts, failures = defaultdict(Counter), []
    for path in sorted(run.glob('worker*/*/*/lightning/summary.json')):
        row = json.loads(path.read_text())
        arm = counts[row['variant']]
        arm['episodes'] += 1
        calls = json.loads((path.parent / 'model_calls.json').read_text())
        for call in calls:
            usage = call['response'].get('usage', {})
            arm['input_tokens'] += usage.get('prompt_tokens', 0)
            arm['output_tokens'] += usage.get('completion_tokens', 0)
        labels = set()
        if row['task_failure_kind']:
            labels.add(row['task_failure_kind'])
        audit_path = path.parent / 'harness_audit.json'
        if audit_path.exists():
            data = json.loads(audit_path.read_text())
            arm['extra_queries'] += data['extra_queries']
            for proposal in data['proposals']:
                if proposal.get('blocked'):
                    arm['guard_' + proposal['layer']] += 1
                    labels.add('guard_' + proposal['layer'])
                arm['repair_attempts'] += bool(proposal.get('repair_attempt'))
                arm['recovered_actions'] += bool(proposal.get('recovered'))
                verification = proposal.get('verification')
                if verification:
                    status = verification['status']
                    arm['verification_' + status] += 1
                    arm['verification_false'] += verification.get('verified') is False
                    if status == 'uncovered':
                        args = proposal['arguments']
                        signature = ':'.join(str(args.get(k, '')) for k in ('cluster_id', 'command_id', 'attribute_id'))
                        arm['uncovered_' + signature] += 1
            for context in data['context']:
                arm['context_turns'] += 1
                arm['context_used'] += context['used']
                arm['raw_characters'] += context['original_characters']
                arm['rendered_characters'] += context['managed_characters'] if context['used'] else context['original_characters']
            arm['format_rejections'] += sum('validation_error' in x for x in data['structured'])
        if not row['success']:
            failures.append({'task_id': row['task_id'], 'variant': row['variant'],
                'observed_events': sorted(labels), 'root_cause': 'unclassified',
                'evidence': str(path.parent.relative_to(run)),
                'note': 'Events are evidence, not an automatic causal attribution.'})
    return {'source': str(run), 'report_sha256': digest(run / 'report.json'),
        'arms': {k: dict(v) for k, v in counts.items()}, 'failed_episodes': failures,
        'limitations': 'Character counts are diagnostic estimates; HTTP usage supplies actual total input/output tokens. No final case content was inspected.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-run', default='runs/harness-mvp/primary-v1/dev')
    parser.add_argument('--output', required=True)
    parser.add_argument('--holdout-dir')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    result = audit(ROOT / args.dev_run)
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    if args.holdout_dir:
        result['holdout'] = freeze_holdout(ROOT, ROOT / args.holdout_dir)
        (output / 'holdout-receipt.json').write_text(json.dumps(result['holdout'], indent=2) + '\n')
    print(json.dumps({'arms': result['arms'], 'holdout': result.get('holdout')}))


if __name__ == '__main__':
    main()
