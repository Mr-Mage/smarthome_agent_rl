"""Replay existing public tool receipts against old/new executors; no model/simulator IO."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deps/SimuHome'))
from smarthome_agent_rl.harness_agent import GuardedExecutor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable(value):
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items()
                if k not in ('action_id', 'duration_seconds')}
    if isinstance(value, list):
        return [comparable(v) for v in value]
    return value


def replay(audit, executor_class):
    rows = audit['actual_observations']
    consumed = []
    def dispatch(tool, arguments):
        if len(consumed) >= len(rows):
            raise AssertionError('New tool call not present in frozen trace')
        expected = rows[len(consumed)]
        assert (tool, arguments) == (expected['tool'], expected['arguments'])
        consumed.append(copy.deepcopy(expected))
        return copy.deepcopy(expected['response'])
    executor = executor_class(dispatch=dispatch)
    replies = []
    for proposal in audit['proposals']:
        executor.structured_audit = [{'turn': proposal['turn']}]
        replies.append(executor.execute(proposal['tool'], copy.deepcopy(proposal['arguments'])))
    assert len(consumed) == len(rows), 'Frozen receipt was not consumed'
    assert comparable(executor.observations) == comparable(rows)
    assert comparable(executor.audit) == comparable(audit['proposals'])
    return executor, replies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--baseline-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/action-lifecycle.json'))
    args = parser.parse_args()
    source, baseline, output = map(Path, (args.source_run, args.baseline_source, args.output))
    config = json.loads(Path(args.config).read_text())
    assert digest(baseline) == config['baseline_source_sha256']
    simulator_commit = subprocess.check_output(['git', '-C', str(ROOT / 'deps/SimuHome'),
                                                'rev-parse', 'HEAD'], text=True).strip()
    assert simulator_commit == config['simulator_commit']
    spec = importlib.util.spec_from_file_location('baseline_harness_replay', baseline)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    paths = sorted(source.rglob('harness_audit.json'))
    if not paths:
        raise ValueError('No public harness traces found')
    assert len(paths) == config['expected']['episodes']
    output.mkdir(parents=True, exist_ok=False)
    # Freeze all input identities before executing either replay arm.
    manifest = {str(path.relative_to(source)): digest(path) for path in paths}
    (output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    report = {'schema': 'action-lifecycle-replay-v1', 'baseline_source_sha256': digest(baseline),
        'config': config, 'config_sha256': digest(Path(args.config)),
        'source_manifest_sha256': digest(output / 'source-manifest.json'),
        'new_source_sha256': digest(ROOT / 'smarthome_agent_rl/harness_agent.py'),
        'state_source_sha256': digest(ROOT / 'smarthome_agent_rl/action_state.py'),
        'episodes': [], 'model_calls': 0, 'network_calls': 0,
        'limits': 'Frozen public receipt replay. No online success-rate or latency claim.'}
    total = {'proposals': 0, 'actual_calls': 0, 'extra_queries': 0, 'rejections': 0}
    states = {}
    for index, path in enumerate(paths):
        audit = json.loads(path.read_text())
        old, old_replies = replay(audit, old_module.GuardedExecutor)
        new, new_replies = replay(audit, GuardedExecutor)
        assert new_replies == old_replies
        assert comparable(new.audit) == comparable(old.audit)
        assert comparable(new.observations) == comparable(old.observations)
        assert new.extra_queries == old.extra_queries
        snapshot = new.actions.snapshot()
        main_actions = [r for r in snapshot['actions'] if not r['extra_query']]
        assert len(main_actions) == len(audit['proposals'])
        dispatched = [r for r in snapshot['actions'] if any(
            t['to'] == 'dispatched' for t in r['transitions'])]
        assert len(dispatched) == len(audit['actual_observations'])
        for record in snapshot['actions']:
            states[record['state']] = states.get(record['state'], 0) + 1
        target = output / f'episode-{index:04d}.json'
        target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report['episodes'].append({'source': str(path.relative_to(source)),
            'source_sha256': digest(path), 'output': target.name, 'output_sha256': digest(target),
            'proposals': len(audit['proposals']), 'actual_calls': len(audit['actual_observations']),
            'behavior_parity': True})
        total['proposals'] += len(audit['proposals'])
        total['actual_calls'] += len(audit['actual_observations'])
        total['extra_queries'] += new.extra_queries
        total['rejections'] += sum(p.get('blocked', False) for p in new.audit)
    assert total == {key: value for key, value in config['expected'].items() if key != 'episodes'}
    assert manifest == {str(path.relative_to(source)): digest(path) for path in paths}
    report.update({'totals': total, 'final_states': states, 'verified': True})
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'verified': True, 'episodes': len(paths), 'totals': total, 'final_states': states}))


if __name__ == '__main__':
    main()
