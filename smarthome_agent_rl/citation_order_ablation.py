"""Change public citation display order without changing IDs or source evidence."""
from collections import Counter
import json

from .bounded_review_ablation import DIMENSIONS, bounded_panels
from .identity_presentation_ablation import evaluate as paired_evaluate
from .review_factorial import request as parent_request, parse_response as parent_parse

ARMS = ('original', 'reverse')


def actor_for(config, index, arm):
    if arm not in ARMS:
        raise ValueError('Unknown citation-order arm')
    return config['actors'][index % 4]


def request(config, item, arm):
    if arm not in ARMS:
        raise ValueError('Unknown citation-order arm')
    body = parent_request(config, item, 'plain_history')
    found = 0
    for message in body['messages']:
        if message['role'] != 'user':
            continue
        try:
            value = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and 'public_citations' in value:
            found += 1
            if arm == 'reverse':
                value['public_citations'].reverse()
                message['content'] = json.dumps(value, ensure_ascii=False)
    if found != 1:
        raise ValueError('Expected exactly one public citation presentation')
    return body


def parse_response(context, text, arm):
    if arm not in ARMS:
        raise ValueError('Unknown citation-order arm')
    return parent_parse(context, text, 'plain_history')


def evaluate(config, inputs, records, historical, labels):
    if any(row['arm'] not in ARMS for row in records):
        raise ValueError('Unexpected citation-order arm')
    mapped = [{**row, 'arm': 'control' if row['arm'] == 'original' else 'identity'}
              for row in records]
    result = paired_evaluate(config, inputs, mapped, historical, labels)
    aliases = {'control': 'original', 'identity': 'reverse', 'historical_citations': 'historical_history_compact'}
    for key in ('arms', 'diagnostic_screen_passed', 'binding_coverage'):
        result[key] = {aliases[arm]: value for arm, value in result[key].items()}
    result['transitions'] = {
        'original->reverse': result['transitions']['control->identity'],
        'historical_history_compact->original': result['transitions']['historical_citations->control']}
    contexts = {item['id']: item['context'] for item in inputs}
    result['bounded_dimensions'] = {}
    result['strict_parser_diagnostics'] = {}
    result['paired_changes'] = {}
    by_key = {(row['id'], row['arm']): row for row in records}
    for arm in ARMS:
        selected = [row for row in records if row['arm'] == arm]
        panels = [bounded_panels(contexts[row['id']], row['decision']) for row in selected]
        result['bounded_dimensions'][arm] = {name: {
            'records': len(panels), 'unavailable': sum(p is None for p in panels),
            **{key: sum(p[name][key] for p in panels if p is not None) for key in (
                'label_mismatch', 'reference_count', 'empty_reference_steps', 'duplicate_reference_steps')}}
            for name in DIMENSIONS}
        result['strict_parser_diagnostics'][arm] = dict(Counter(
            row['parse_error'] for row in selected if row['parse_error'] is not None))
    for name in ('verdict', 'correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute'):
        changes = []
        for item in inputs:
            pair = [by_key.get((item['id'], arm)) for arm in ARMS]
            if any(row is None for row in pair):
                continue
            def value(row):
                decision = row['decision']
                return 'INVALID' if decision is None else decision[name] if name == 'verdict' else decision[name]['label']
            left, right = map(value, pair)
            if left != right:
                changes.append({'id': item['id'], 'original': left, 'reverse': right})
        result['paired_changes'][name] = changes
    result['scope'] = ('Fresh original/reversed public_citations presentation only; same actor per input pair. '
        'IDs, source spans, context, prompt, schema enum order, compact xgrammar, generation and semantic gates unchanged. '
        'Exposed diagnostic cohort; no independent accuracy, native adoption, SR, training or default change.')
    return result
