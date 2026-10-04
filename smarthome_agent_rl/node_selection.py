"""Predeclared development gates; no selection from final or smoke."""


def select_guard(report, gates):
    if not report['verified'] or report['phase'] != 'dev':
        raise ValueError('Selection requires complete verified dev evidence')
    baseline = report['arms']['G']
    decisions = {}
    accepted = ['G']
    for variant in ('GD', 'GW', 'GDW'):
        arm = report['arms'][variant]
        illegal = baseline['all_totals']['invalid_reached_executor']
        reduction = 1 - arm['all_totals']['invalid_reached_executor'] / illegal if illegal else None
        ratio = arm['all_totals']['actor_tokens'] / baseline['all_totals']['actor_tokens']
        checks = {'success_rate': arm['success_rate'] >= baseline['success_rate'],
            'illegal_execution': reduction is not None and reduction >= gates['illegal_reduction_min'],
            'actor_tokens': ratio <= gates['actor_token_ratio_max']}
        decisions[variant] = {'accepted': all(checks.values()), 'checks': checks,
            'illegal_reduction': reduction, 'actor_token_ratio': ratio}
        if all(checks.values()):
            accepted.append(variant)
    winner = min(accepted, key=lambda v: (-report['arms'][v]['successes'],
        report['arms'][v]['all_totals']['actor_tokens'], v))
    return {'winner': winner, 'decisions': decisions, 'basis': 'all dev runs; task-clustered CIs reported separately'}
