"""Evaluate repair-budget decisions on frozen public traces, without counterfactual rollouts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.guard import GuardError
from smarthome_agent_rl.recovery import RecoveryPolicy, public_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/recovery-policy.json'))
    args = parser.parse_args()
    source, output = Path(args.source_run), Path(args.output)
    config = json.loads(Path(args.config).read_text())
    paths = sorted(source.rglob('harness_audit.json'))
    assert len(paths) == config['expected_episodes']
    output.mkdir(parents=True, exist_ok=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {str(p.relative_to(source)): digest(p) for p in paths}
    (output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    report = {'schema': 'recovery-policy-audit-v1', 'config': config,
        'config_sha256': digest(Path(args.config)), 'episodes': [], 'arms': {},
        'model_calls': 0, 'network_calls': 0, 'verified': True}
    for path in paths:
        audit = json.loads(path.read_text())
        policy, observed, decisions = RecoveryPolicy(config['repair_limit'], config['total_failure_limit']), [], []
        rows = audit['actual_observations']
        for index, proposal in enumerate(audit['proposals']):
            start = proposal['actual_calls_before']
            end = start + proposal['actual_calls']
            assert len(observed) == start
            pending = rows[start:end]
            mains = [row for row in pending if not row['extra_query']]
            assert len(mains) <= 1
            if mains:
                main = mains[0]
                assert (main['tool'], main['arguments']) == (proposal['tool'], proposal['arguments'])
                position = next(i for i, row in enumerate(pending) if not row['extra_query'])
                observed.extend(pending[:position])
                context = public_context(main['tool'], main['arguments'], observed)
                decision = 'allow'
                try:
                    policy.check(main['tool'], main['arguments'], context)
                except GuardError as exc:
                    decision = exc.layer
                decisions.append({'proposal_index': index, 'turn': proposal['turn'], 'decision': decision,
                                  'context': context})
                # Preserve actual historical receipts, even after a hypothetical block.
                # This is a policy audit, not a simulated alternative Agent trajectory.
                policy.observe(main['tool'], main['arguments'], main['response'], context, f'p{index}')
                observed.extend(pending[position:])
            else:
                observed.extend(pending)
        assert len(observed) == len(rows)
        relative = str(path.relative_to(source))
        arm = next(v for v in (config['reference'], config['historical_negative_probe']) if f'/{v}/' in relative)
        counts = Counter(row['decision'] for row in decisions)
        total = report['arms'].setdefault(arm, {'episodes': 0, 'decisions': {}, 'triggered_episodes': 0})
        total['episodes'] += 1
        total['triggered_episodes'] += int(any(r['decision'] != 'allow' for r in decisions))
        for kind, count in counts.items():
            total['decisions'][kind] = total['decisions'].get(kind, 0) + count
        report['episodes'].append({'source': relative, 'sha256': manifest[relative],
                                   'decisions': decisions, 'policy': policy.snapshot()})
    assert manifest == {str(p.relative_to(source)): digest(p) for p in paths}
    report['online_admitted'] = False  # A trigger alone does not freeze an online protocol.
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'verified': True, 'arms': report['arms'], 'online_admitted': False}))


if __name__ == '__main__':
    main()
