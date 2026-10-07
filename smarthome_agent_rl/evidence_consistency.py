"""Audit agreement between reviewer labels and its own cited step evidence.

These are engineering diagnostics, not semantic truth labels or admission
rules. Literal provenance, internal consistency and entailment are separate.
"""
from datetime import datetime, timedelta
import re

from .semantic_prompt_ablation import evidence_check
from .semantic_verifier import VerificationContext
from .typed_semantic_review import schema_structure_check

FROM_NOW = re.compile(r'\b(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?)\s+from\s+now\b', re.I)
CLOCK = re.compile(r'\b(?:at\s+)?(\d{1,2}):(\d{2})\s*(AM|PM)\b', re.I)
DEPENDENT = re.compile(r'\b(?:before|after|when|until|previous|finishes?|finishing)\b', re.I)


def cited_anchors(context, quote, encoded):
    """Arithmetic projections only; does not identify the requested phase."""
    if not isinstance(quote, str) or not quote or quote not in context['user_goal']:
        return {'status': 'no_literal_quote', 'anchors': []}
    initial = context['environment_state'].get('initial_public_time')
    try:
        now, planned = datetime.fromisoformat(initial), datetime.fromisoformat(encoded)
        if now.tzinfo is not None or planned.tzinfo is not None:
            raise ValueError('No implicit timezone conversion')
    except (ValueError, TypeError):
        return {'status': 'unavailable_clock_encoding', 'anchors': []}
    values = []
    for match in FROM_NOW.finditer(quote):
        unit = match.group(2).lower().rstrip('s')
        amount = float(match.group(1))
        try:
            expected = now + timedelta(seconds=amount * {'second': 1, 'minute': 60, 'hour': 3600}[unit])
        except (OverflowError, ValueError):
            continue
        values.append({'kind': 'initial_public_time_plus_offset', 'quote': match.group(0),
                       'expected_encoding': expected.isoformat(sep=' '),
                       'delta_seconds': (planned - expected).total_seconds(),
                       'assumption': 'from now refers to initial public timestamp;not actual dispatch freshness'})
    if not re.search(r'\b(?:tomorrow|yesterday|next\s+day)\b', quote, re.I):
        for match in CLOCK.finditer(quote):
            hour, minute = int(match.group(1)), int(match.group(2))
            if not 1 <= hour <= 12 or not 0 <= minute <= 59:
                continue
            hour = hour % 12 + (12 if match.group(3).upper() == 'PM' else 0)
            expected = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            values.append({'kind': 'same_date_clock', 'quote': match.group(0),
                           'expected_encoding': expected.isoformat(sep=' '),
                           'delta_seconds': (planned - expected).total_seconds(),
                           'assumption': 'same date as initial timestamp;no inferred date,timezone or event binding'})
    return {'status': 'arithmetic_projection' if values else
            'needs_previous_action_or_event_binding' if DEPENDENT.search(quote) else 'unhandled_quote',
            'anchors': values}


def audit_claims(context, decision):
    VerificationContext(**context)  # Hidden evaluator data remains forbidden.
    item = {'context': context};row = {'decision': decision, 'arm': '9b_stepwise'}
    structure = schema_structure_check(item, row)
    provenance = evidence_check(item, row)
    flags, time_rows = [], []
    if structure:
        target, goal = decision['correct_target'], decision['goal_consistent']
        targets, times = target['evidence']['steps'], goal['evidence']['steps']
        unsupported = [r['step_index'] for r in targets if r['support'] == 'unsupported']
        if target['label'] == 'YES' and unsupported:
            flags.append({'kind': 'target_yes_with_unsupported_steps', 'class': 'internal_conflict',
                          'dimension': 'correct_target', 'steps': unsupported})
        if target['label'] == 'NO' and targets and not unsupported:
            flags.append({'kind': 'target_no_without_declared_unsupported_step', 'class': 'unsubstantiated_dimension',
                          'dimension': 'correct_target', 'steps': [r['step_index'] for r in targets],
                          'supports': [r['support'] for r in targets]})
        conflict = [r['step_index'] for r in times if r['relation'] == 'conflict']
        if goal['label'] == 'YES' and conflict:
            flags.append({'kind': 'goal_yes_with_declared_time_conflict', 'class': 'internal_conflict',
                          'dimension': 'goal_consistent', 'steps': conflict})
        if goal['label'] == 'NO' and times and not conflict:
            flags.append({'kind': 'goal_no_without_declared_time_conflict', 'class': 'unsubstantiated_dimension',
                          'dimension': 'goal_consistent', 'steps': [r['step_index'] for r in times],
                          'relations': [r['relation'] for r in times],
                          'scope': 'Time-only evidence cannot justify other command/value conflicts;not proof that NO is false'})
        for step in times:
            anchors = cited_anchors(context, step['requested_quote'], step['encoded_execution_time'])
            time_rows.append({'step_index': step['step_index'], 'relation': step['relation'], **anchors})
            if step['relation'] == 'conflict' and anchors['anchors'] and all(
                    value['delta_seconds'] == 0 for value in anchors['anchors']):
                flags.append({'kind': 'time_conflict_despite_cited_anchor_arithmetic_match',
                              'class': 'arithmetic_binding_review', 'dimension': 'goal_consistent',
                              'steps': [step['step_index']],
                              'scope': 'Cited encoding matches under explicit anchor assumptions;does not prove phase binding,entailment or valid action'})
    return {'schema_conformant': structure, 'literal_evidence': provenance, 'flags': flags,
            'time_anchors': time_rows,
            'scope': 'Internal labels/evidence and stated arithmetic assumptions only. Flags are not error truth,corrected verdicts,semantic blocking or Task completion.'}
