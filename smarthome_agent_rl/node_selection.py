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


def select_time_plan(report, gates):
    if not report['verified'] or report['phase'] != 'dev':
        raise ValueError('Selection requires complete verified dev evidence')
    reference = report['reference']
    baseline, arm = report['arms'][reference], report['arms']['TimePlan']
    category = 'qt4-1:feasible'
    ratio = arm['all_totals']['actor_tokens'] / baseline['all_totals']['actor_tokens']
    checks = {'success_rate': arm['success_rate'] - baseline['success_rate'] >= gates['sr_delta_min'],
        'temporal_success': arm['categories'][category]['successes'] > baseline['categories'][category]['successes'],
        'actor_tokens': ratio <= gates['actor_token_ratio_max'],
        'illegal_execution': arm['all_totals']['invalid_reached_executor'] <= baseline['all_totals']['invalid_reached_executor']}
    return {'winner': 'TimePlan' if all(checks.values()) else reference, 'checks': checks,
            'actor_token_ratio': ratio, 'basis': 'all dev runs; qt4-1 feasible gate; smoke diagnostic only'}


def assess_sft(report, gates):
    """Assess the one frozen adapter; this does not select a training checkpoint."""
    if not report['verified'] or report['phase'] != 'dev':
        raise ValueError('Assessment requires complete verified development evidence')
    reference = report['reference']
    baseline, arm = report['arms'][reference], report['arms']['SFT9B']
    if not baseline['episodes'] or baseline['episodes'] != arm['episodes']:
        raise ValueError('Assessment requires equal paired episode coverage')
    ratio = arm['all_totals']['actor_tokens'] / baseline['all_totals']['actor_tokens']
    checks = {'success_rate': arm['success_rate'] - baseline['success_rate'] >= gates['sr_delta_min'],
        'actor_tokens': ratio <= gates['actor_token_ratio_max'],
        'illegal_execution': arm['all_totals']['invalid_reached_executor'] <= baseline['all_totals']['invalid_reached_executor']}
    return {'winner': 'SFT9B' if all(checks.values()) else reference, 'checks': checks,
        'actor_token_ratio': ratio, 'checkpoint_selected_from_evaluation': False,
        'basis': 'One frozen adapter; all development runs including failures; task-clustered CIs reported separately'}
