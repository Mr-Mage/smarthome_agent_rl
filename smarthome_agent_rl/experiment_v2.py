"""Predeclared dev gates and deterministic selection; never read final outcomes."""
import copy

DEFAULT_POLICIES = {
    'B0': None,
    'G': {'verify': False, 'verification_version': 1, 'context_version': 0},
    'GV': {'verify': True, 'verification_version': 1, 'context_version': 0},
    'GC': {'verify': False, 'verification_version': 1, 'context_version': 1},
    'Full': {'verify': True, 'verification_version': 1, 'context_version': 1},
}


def policy(config, variant):
    value = copy.deepcopy(config.get('variant_policies', {}).get(variant, DEFAULT_POLICIES.get(variant)))
    if value is not None and not value['verify']:
        value['verification_version'] = 1
    return value


def service_identity(config):
    keys = ('model_python', 'actor_path', 'actor_model', 'actor_context', 'judge_path',
            'judge_model', 'judge_context', 'judge_gpus', 'judge_port', 'embedding_port',
            'workflows', 'engine_seed', 'inference', 'kernel_cache_root')
    return {key: config.get(key) for key in keys}


def unique_policies(config, variants):
    representatives, aliases = [], {}
    for variant in variants:
        match = next((old for old in representatives if policy(config, old) == policy(config, variant)), None)
        aliases[variant] = match or variant
        if match is None:
            representatives.append(variant)
    return representatives, aliases


def module_selection(report, gates):
    if not report['verified'] or report['phase'] != 'dev':
        raise ValueError('Verified dev evidence required')
    arms = report['arms']
    if set(arms) != set(gates['module_variants']):
        raise ValueError('Complete module comparison required')
    g, gv, gv2, gc2 = (arms[v] for v in ('G', 'GV', 'GV2', 'GC2'))
    vg, cg = gates['verification_gate'], gates['context_gate']
    accepted_v = (gv2['success_rate'] - g['success_rate'] >= vg['sr_delta_min'] and
        gv2['all_totals']['extra_queries'] <= gv['all_totals']['extra_queries'] and
        gv2['all_totals']['actor_tokens'] <= vg['actor_token_ratio_max'] * g['all_totals']['actor_tokens'])
    accepted_c = (gc2['success_rate'] - g['success_rate'] >= cg['sr_delta_min'] and
        gc2['all_totals']['actor_tokens'] <= cg['actor_token_ratio_max'] * g['all_totals']['actor_tokens'])
    return {'verification_accepted': accepted_v, 'context_accepted': accepted_c,
        'candidate_policy': {'verify': accepted_v, 'verification_version': 2 if accepted_v else 1,
                             'context_version': 2 if accepted_c else 0}}


def integration_selection(report, config, gates):
    if not report['verified'] or report['phase'] != 'dev':
        raise ValueError('Verified dev evidence required')
    variants = gates['integration_variants']
    rows = report['arms']
    aliases = {v: v if v in rows else next((old for old in rows if policy(config, old) == policy(config, v)), None)
               for v in variants}
    if None in aliases.values() or 'B0' not in rows:
        raise ValueError('Integration must include B0 calibration for final freeze')
    rows = {**rows, **{v: rows[aliases[v]] for v in variants}}
    best = max(rows[v]['success_rate'] for v in variants)
    eligible = [v for v in variants if best - rows[v]['success_rate'] <=
                gates['integration_selection']['success_tie_tolerance'] + 1e-12]
    winner = min(eligible, key=lambda v: (rows[v]['all_totals']['actor_tokens'],
                rows[v]['all_totals']['extra_queries'], v))
    desired = ['B0', 'G', 'Full', winner]
    representatives, formal_aliases = unique_policies(config, desired)
    return {'winner': winner, 'selected_policy': policy(config, winner),
            'formal_variants': representatives, 'aliases': formal_aliases, 'integration_aliases': aliases,
            'selection_scope': 'dev only; no final tuning'}
