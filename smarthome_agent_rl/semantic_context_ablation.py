"""Independent public context interventions on frozen real pre-action slices."""
from collections import Counter
from dataclasses import asdict
import statistics

from .benchmarks.runner import valid_usage
from .semantic_context import digest
from .semantic_context_enriched import build_enriched_context
from .semantic_diagnosis import contexts

ARMS = ('9b_references', '9b_workflow', '9b_both')


def enrich(item, public, proposal, observations, rules):
    original, cutoff = contexts(public, proposal, observations)
    if cutoff != item['dispatch_observation_index'] or original != item['contexts']:
        raise ValueError('Frozen parent public boundary differs')
    if digest(observations[:cutoff]) != item['preaction_receipts_sha256']:
        raise ValueError('Frozen parent receipt prefix differs')
    versions = {}
    for arm in ARMS:
        suffix = arm.split('_', 1)[1]
        value = asdict(build_enriched_context(public['query'], original['public']['proposed_action'],
            observations[:cutoff], user_location=public['user_location'], initial_time=public['current_time'],
            contract=proposal.get('contract'), references=suffix in ('references', 'both'),
            workflow_rules=rules if suffix in ('workflow', 'both') else None))
        if any(value[key] != original['public'][key] for key in value if key != 'environment_state'):
            raise ValueError('Intervention changed task/action/recent attempts/contract')
        versions[suffix] = value
    return {**item, 'contexts': versions, 'historical_public_context_sha256': digest(original['public'])}


def evaluate(config, records, baseline, labels):
    expected = config['proposals']
    ids = {r['id'] for r in baseline}
    if len(baseline) != expected or len(ids) != expected or any(r['arm'] != '9b_public' for r in baseline):
        raise ValueError('Historical baseline identity/coverage differs')
    categories = {r['id']: r['category'] for r in labels['cases']}
    if set(categories) != ids or len(categories) != len(labels['cases']):
        raise ValueError('Developer case coverage differs')
    totals = {}
    for arm in ('9b_public',) + ARMS:
        rows = baseline if arm == '9b_public' else [r for r in records if r['arm'] == arm]
        clear = [r for r in rows if categories[r['id']] == 'CERTAIN_CONFLICT']
        controls = [r for r in rows if categories[r['id']] == 'CONSISTENT_CONTROL']
        verdict = lambda r: r['decision']['verdict'] if r['decision'] is not None else 'INVALID'
        totals[arm] = {'records': len(rows), 'valid': sum(r['decision'] is not None for r in rows),
            'verdicts': dict(Counter(verdict(r) for r in rows)),
            'tokens': sum(r['call']['usage']['total_tokens'] for r in rows if valid_usage(r['call']['usage'])),
            'http_errors': sum(r['call']['error'] is not None for r in rows),
            'missing_usage': sum(not valid_usage(r['call']['usage']) for r in rows),
            'parse_errors': sum(r['parse_error'] is not None for r in rows),
            'median_request_seconds': statistics.median(r['call']['request_seconds'] for r in rows) if rows else None,
            'developer_conflicts': dict(Counter(verdict(r) for r in clear)),
            'developer_controls': dict(Counter(verdict(r) for r in controls)),
            'cost_origin': 'reused immutable N72; not new cost' if arm == '9b_public' else 'new inference'}
    mapped = {(r['id'], r['arm']): r for r in baseline + records}
    pairs = {}
    for left, right in [('9b_public', arm) for arm in ARMS] + [('9b_references', '9b_both'), ('9b_workflow', '9b_both')]:
        transitions = Counter()
        for identity in ids:
            a, b = mapped.get((identity, left)), mapped.get((identity, right))
            transitions['MISSING' if a is None or b is None else verdict(a) + '->' + verdict(b)] += 1
        pairs[left + '__' + right] = dict(transitions)
    counts = Counter(categories.values())
    if dict(counts) != config['developer_counts']:
        raise ValueError('Frozen developer diagnostic categories differ')
    checks = {'complete_records': len(records) == expected * len(ARMS) and
        {(r['id'], r['arm']) for r in records} == {(i, a) for i in ids for a in ARMS},
        'http_errors': all(totals[a]['http_errors'] == 0 for a in ARMS),
        'usage_complete': all(totals[a]['missing_usage'] == 0 for a in ARMS),
        'structural': all(totals[a]['valid'] / expected >= config['valid_ratio_min'] for a in ARMS)}
    case_gates = {a: totals[a]['valid'] / expected >= config['valid_ratio_min'] and
        totals[a]['developer_conflicts'].get('DENY', 0) == counts['CERTAIN_CONFLICT'] and
        totals[a]['developer_controls'].get('ALLOW', 0) == counts['CONSISTENT_CONTROL'] for a in ARMS}
    return {'checks': checks, 'engineering_passed': all(checks.values()), 'arms': totals, 'pairs': pairs,
        'developer_case_gate': case_gates, 'developer_counts': dict(counts),
        'new_model_requests': len(records), 'new_model_tokens': sum(totals[a]['tokens'] for a in ARMS),
        'native_integration_admitted': False,
        'scope': 'Developer cases are retained diagnostics,not independent truth or accuracy. '
                 'No simulator rerun,Task completion,native blocking,SR claim or default adoption.'}
