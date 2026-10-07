"""Read-only lexical binding panels, never a device/phase truth oracle."""
from collections import Counter
import copy
import re

from .citation_review import ARMS, catalog, binding_checks
from .declared_evidence_report import identity_panel
from .effect_evidence_ablation import checks
from .semantic_context import digest


def mentions(text, identifier):
    if not isinstance(text, str) or not isinstance(identifier, str) or not identifier:
        return None
    phrase = identifier.replace('_', ' ')
    return bool(re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text, re.I))


def quote_spans(goal, quote):
    """Return all literal occurrences; do not choose an ambiguous phase."""
    if quote is None:
        return {'status': 'NULL', 'spans': []}
    if not isinstance(quote, str) or not quote:
        return {'status': 'INVALID', 'spans': []}
    spans = []
    position = 0
    while True:
        start = goal.find(quote, position)
        if start < 0:
            break
        spans.append([start, start + len(quote)])
        position = start + 1
    return {'status': 'LITERAL' if spans else 'NONLITERAL', 'spans': spans}


def public_identity(context, target):
    """Room/type word candidates come only from observed catalog receipts."""
    panel = identity_panel(context, target)
    room = panel['observed_room_id']; types = panel['observed_device_types']
    observed = context['environment_state'].get('devices', {}).get(target['device_id'])
    # Ambiguous/truncated observations are unavailable, not a chosen identity.
    available = panel['observed_identity_available'] and not panel['catalog_truncated'] and len(types) == 1
    candidates = [row for row in catalog(context['user_goal']) if available and
                  mentions(row['text'], room) is True and any(mentions(row['text'], kind) is True for kind in types)]
    quote = target['support_quote']
    quote_literal = quote_spans(context['user_goal'], quote)
    joint = None if not available or quote_literal['status'] != 'LITERAL' else \
        mentions(quote, room) is True and any(mentions(quote, kind) is True for kind in types)
    return {**panel, 'identity_word_panel_available': available,
            'public_room_type_window_candidates': candidates,
            'goal_mentions_observed_room': mentions(context['user_goal'], room),
            'goal_mentions_observed_types': {kind: mentions(context['user_goal'], kind) for kind in types},
            'support_quote_literal': quote_literal,
            'support_quote_joint_room_type_words': joint,
            'unsupported_with_public_identity_words': target['support'] == 'unsupported' and bool(candidates),
            'requested_without_joint_quote_words': target['support'] == 'requested' and joint is not True,
            'structure_source': copy.deepcopy(observed.get('structure_source')) if isinstance(observed, dict) else None,
            'scope': 'Observed room/type lexical windows only;not exact numbered-device binding,identity truth or entailment. '
                     'Anaphora can omit words;matching never overrides model support or decodes a device ID.'}


def binding_report(context, row):
    """Keep failures and original decisions, without projecting any labels."""
    if row['arm'] not in ARMS:
        raise ValueError('Unknown frozen arm')
    before = digest(row)
    decision = row['decision']; checked = checks(context, decision, 'combined')
    result = {'original_decision_sha256': digest(decision), 'raw_model_decision_sha256': digest(row['model_decision']),
              'schema_conformant': checked['schema_conformant'],
              'literal_evidence_valid': checked['literal_evidence_valid'],
              'evidence_issues': checked['evidence_issues'],
              'aggregate_mismatches': checked['aggregate_mismatches'],
              'original_verdict': decision['verdict'] if decision else 'INVALID',
              'identity_panels': [], 'effect_panels': [],
              'semantic_status': 'UNVERIFIED', 'native_admitted': False,
              'dispatch_verdict': None, 'task_completed': False}
    if checked['schema_conformant']:
        result['identity_panels'] = [public_identity(context, target) for target in decision['correct_target']['evidence']['steps']]
        steps = {step['step_index']: step for step in decision['goal_consistent']['evidence']['steps']}
        for diagnostic in binding_checks(context, decision):
            step = steps[diagnostic['step_index']]
            effect = quote_spans(context['user_goal'], step['effect_quote'])
            time = quote_spans(context['user_goal'], step['requested_quote'])
            literal = effect['status'] in ('NULL', 'LITERAL') and time['status'] in ('NULL', 'LITERAL')
            result['effect_panels'].append({**diagnostic, 'effect_quote_literal': effect, 'time_quote_literal': time,
                'literal_panel_available': literal,
                'usable_percentage_status': diagnostic['status'] if literal else 'invalid_literal_evidence',
                'literal_offset_words_disjoint': diagnostic['offset_binding_needs_review'] if literal else None,
                'scope': 'Literal parameter/offset-word coverage,not phase truth. Multi-value,null andnonliteral evidence remain unverified.'})
    result['flags'] = {
        'unavailable_schema': not checked['schema_conformant'],
        'invalid_literal_evidence': not checked['literal_evidence_valid'],
        'aggregate_mismatch': bool(checked['aggregate_mismatches']),
        'unsupported_with_public_identity_words': any(p['unsupported_with_public_identity_words'] for p in result['identity_panels']),
        'requested_without_joint_quote_words': any(p['requested_without_joint_quote_words'] for p in result['identity_panels']),
        'multiple_percentage_references': any(p['usable_percentage_status'] == 'multiple_percentage_references' for p in result['effect_panels']),
        'no_explicit_percentage': any(p['usable_percentage_status'] == 'no_explicit_percentage' for p in result['effect_panels']),
        'disjoint_literal_offset_words': any(p['literal_offset_words_disjoint'] is True for p in result['effect_panels'])}
    result['scope'] = ('Read-only coverage/tension flags,not accuracy,false-rejection rate,semantic errors or goal completion. '
                       'No chosen desired value,phase,corrected label or dispatch decision.')
    if digest(row) != before:
        raise AssertionError('Diagnosis mutated original record')
    return result


def evaluate(config, sources):
    expected = {(item['id'], arm) for item in sources for arm in config['arms']}
    actual = {(item['id'], item['arm']) for item in sources}
    if config['arms'] != list(ARMS) or config['paired_inputs'] <= 0 or config['records'] != config['paired_inputs'] * 2 or \
            len(sources) != config['records'] or len(actual) != len(sources) or actual != expected or \
            len({item['id'] for item in sources}) != config['paired_inputs']:
        raise ValueError('All-arm paired receipt denominator differs')
    records = [{key: item[key] for key in ('id', 'task_id', 'origin', 'arm', 'source_file', 'source_sha256')} |
               {'context_sha256': digest(item['context']), 'report': binding_report(item['context'], item['row'])} for item in sources]
    arms = {}
    for arm in config['arms']:
        reports = [item['report'] for item in records if item['arm'] == arm]
        identities = [p for report in reports for p in report['identity_panels']]
        effects = [p for report in reports for p in report['effect_panels']]
        arms[arm] = {'records': len(reports), 'schema_conformant': sum(r['schema_conformant'] for r in reports),
            'literal_evidence_valid': sum(r['literal_evidence_valid'] for r in reports),
            'original_verdicts': dict(Counter(r['original_verdict'] for r in reports)),
            'record_flags': {flag: sum(r['flags'][flag] for r in reports) for flag in reports[0]['flags']},
            'identity_steps': len(identities), 'identity_word_panel_available': sum(p['identity_word_panel_available'] for p in identities),
            'declared_support': dict(Counter(p['declared_support'] for p in identities)),
            'unsupported_with_public_identity_words_steps': sum(p['unsupported_with_public_identity_words'] for p in identities),
            'requested_without_joint_quote_words_steps': sum(p['requested_without_joint_quote_words'] for p in identities),
            'percentage_steps': len(effects),
            'percentage_statuses': dict(Counter(p['usable_percentage_status'] for p in effects)),
            'disjoint_literal_offset_words_steps': sum(p['literal_offset_words_disjoint'] is True for p in effects)}
    indexed = {(r['id'], r['arm']): r for r in records}
    paired = {flag: dict(Counter(str(indexed[(identity, 'quotes')]['report']['flags'][flag]) + '->' +
                   str(indexed[(identity, 'citations')]['report']['flags'][flag]) for identity in {s['id'] for s in sources}))
              for flag in records[0]['report']['flags']}
    report = {'complete': True, 'records': len(records), 'arms': arms, 'paired_flags': paired,
        'coverage': {'task_count': len({s['task_id'] for s in sources}),
                     'distinct_contexts': len({digest(s['context']) for s in sources}),
                     'paired_inputs': len(sources) // 2},
        'new_model_requests': 0, 'new_tokens': 0, 'reserved_gpu_seconds': 0,
        'original_decisions_changed': 0, 'semantic_status': 'UNVERIFIED', 'native_admitted': False,
        'scope': 'All156 original N83 receipts including failures. Mechanical post-hoc coverage diagnosis;'
                 'repeated task trajectories not independent benefits. No label projection,blocking orTask completion.'}
    return report, records
