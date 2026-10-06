"""Frozen N47 coverage gate and incremental paired runtime diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.paired_stats import mcnemar, bootstrap_ci
from smarthome_agent_rl.benchmark import STRATA


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def analyze(run, stage):
    directory = run / stage
    protocol, report = read(directory / 'protocol.json'), read(directory / 'report.json')
    frozen = protocol['config']['node_experiment']
    diagnostics, episodes = {}, {}
    for variant in protocol['variants']:
        counts, reasons = Counter(), Counter()
        for item in protocol['schedule']:
            episode = episode_directory(directory, item, variant)
            summary = read(episode / 'summary.json')
            seed = item.get('actor_seed', protocol['config']['model_seed'])
            episodes[(item['task']['id'], seed, variant)] = summary
            if variant not in ('RC', 'RCV'):
                continue
            audit = read(episode / 'harness_audit.json') if (episode / 'harness_audit.json').exists() else None
            if audit is None:
                counts['missing_audit_episodes'] += 1
                continue
            if audit['runtime_verify'] != (variant == 'RCV'):
                raise ValueError('Runtime arm switch differs from frozen protocol')
            if audit['extra_queries'] > protocol['config']['extra_queries_max']:
                raise ValueError('Auxiliary query budget exceeded')
            trace = {row['invocation_id']: row for row in audit['tool_trace']}
            if len(trace) != len(audit['actual_observations']):
                raise ValueError('Tool trace and actual-call accounting differ')
            if any('state' in row for row in trace.values()):
                raise ValueError('Tool trace must not invent query lifecycles')
            blocked_codes = []
            for proposal in audit['proposals']:
                if proposal['tool'] not in ('execute_command', 'write_attribute'):
                    continue
                counts['mutation_proposals'] += 1
                mutation = proposal.get('mutation', {})
                if proposal['blocked']:
                    code = proposal['response']['error']['reason_code']
                    blocked_codes.append(code)
                    reasons[code] += 1
                    counts['blocked_mutations'] += 1
                    continue
                invocation = mutation['mutation_invocation_id']
                if invocation not in trace or trace[invocation]['arguments'] != proposal['arguments']:
                    raise ValueError('Executed mutation lost its immutable trace')
                dispatched = [row for row in trace.values() if row['parent_step'] == str(proposal['turn'])
                              and row['tool'] in ('execute_command', 'write_attribute')]
                if len(dispatched) != 1:
                    raise ValueError('A single proposal must not replay a mutation')
                counts['executed_mutations'] += 1
                counts[mutation['status']] += 1
                reasons[mutation['verification'].get('reason') or mutation['status']] += 1
            if 'CONTRACT_UNCOVERED' in blocked_codes:
                counts['uncovered_precondition_episodes'] += 1
            if any(code.endswith('BUDGET_EXHAUSTED') for code in blocked_codes):
                counts['budget_blocked_episodes'] += 1
        executed = counts['executed_mutations']
        diagnostics[variant] = {'counts': dict(counts), 'reasons': dict(reasons),
            'unverified_action_rate': counts['UNVERIFIED'] / executed if executed else None,
            'definition': 'UNVERIFIED / actually dispatched mutations; blocked proposals excluded; legacy G/B0 not instrumented'}
    rc = diagnostics['RCV']['counts']
    smoke_gate = frozen['gates']['smoke']
    coverage = rc.get('uncovered_precondition_episodes', 0) / report['arms']['RCV']['episodes']
    gate = {'can_continue': rc.get('missing_audit_episodes', 0) == 0 and
            rc.get('executed_mutations', 0) >= smoke_gate['executed_mutations_min'] and
            coverage <= smoke_gate['uncovered_episode_ratio_max'],
            'uncovered_episode_ratio': coverage, 'scope': 'engineering coverage gate, not efficacy selection'}
    result = {'stage': stage, 'commit': protocol['commit'], 'diagnostics': diagnostics, 'gate': gate}
    if stage == 'dev':
        seed = protocol['config']['primary_actor_seed']
        items = {item['task']['id']: item for item in protocol['schedule']}
        baseline = [episodes[(key, seed, 'RC')]['success'] for key in items]
        treatment = [episodes[(key, seed, 'RCV')]['success'] for key in items]
        differences = [[int(episodes[(key, seed, 'RCV')]['success']) - int(episodes[(key, seed, 'RC')]['success'])
            for key, item in items.items() if (item['task']['query_type'], item['task']['case']) == stratum]
            for stratum in STRATA]
        result['RCV_vs_RC_primary'] = {**mcnemar(baseline, treatment), 'ci95': bootstrap_ci(differences), 'seed': seed}
        checks = {}
        reference = report['arms']['G']
        for variant in ('RC', 'RCV'):
            arm = report['arms'][variant]
            checks[variant] = {'sr_nondecreasing': arm['success_rate'] >= reference['success_rate'],
                'illegal_nonincreasing': arm['all_totals']['invalid_reached_executor'] <= reference['all_totals']['invalid_reached_executor'],
                'token_budget': arm['all_totals']['actor_tokens'] <= frozen['gates']['actor_token_ratio_max'] * reference['all_totals']['actor_tokens']}
        result['efficacy_checks'] = checks
        result['decision'] = 'retain G; pilot alone cannot establish stable public benchmark benefit'
    write(run / (stage + '-runtime-analysis.json'), result)
    write(run / (stage + '-runtime-gate.json'), gate)
    print(json.dumps({'stage': stage, 'gate': gate, 'diagnostics': diagnostics}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    analyze(Path(args.run), args.stage)


if __name__ == '__main__':
    main()
