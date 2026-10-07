"""Separate operation and time evidence layout, preserving predicted decisions."""
import copy
import json

from .bounded_review_ablation import strict_json
from .citation_order_ablation import evaluate as parent_evaluate
from .review_factorial import request as parent_request, parse_response as parent_parse
from .semantic_diagnosis import parse

ARMS = ('original', 'partitioned')
TIME_FIELDS = ('step_index', 'encoded_execution_time', 'requested_ref', 'relation')
EFFECT_FIELDS = ('step_index', 'proposed_effect', 'effect_ref', 'effect_relation')
LAYOUT_PROMPT = '''Output layout only: goal_consistent.evidence uses effect_steps and
time_steps instead of steps. effect_steps contains the original per-step
step_index,proposed_effect,effect_ref,effect_relation fields. time_steps contains
the original per-step step_index,encoded_execution_time,requested_ref,relation
fields. Match the supplied step indices in both arrays. All original evidence,
binding,uncertainty and aggregation rules still apply; predict one unchanged
goal_consistent label and probability. Other dimensions keep their original layout.'''


def schema(context):
    # Build from the exact frozen request schema, without decoding device IDs.
    from .identity_evidence_ablation import schema as identity_schema
    from .bounded_review_ablation import DIMENSIONS, evidence_schema
    value = identity_schema(context, 'control')
    for name in DIMENSIONS:
        value['json_schema']['schema']['properties'][name]['properties']['evidence'] = evidence_schema(context)
    evidence = value['json_schema']['schema']['properties']['goal_consistent']['properties']['evidence']
    original = evidence['properties']['steps']
    groups = {}
    for name, fields in (('effect_steps', EFFECT_FIELDS), ('time_steps', TIME_FIELDS)):
        array = copy.deepcopy(original)
        for entry in array['prefixItems']:
            entry['properties'] = {field: entry['properties'][field] for field in fields}
            entry['required'] = list(fields)
        groups[name] = array
    evidence['properties'] = groups; evidence['required'] = list(groups)
    return value


def actor_for(config, index, arm):
    if arm not in ARMS: raise ValueError('Unknown partitioned-goal arm')
    return config['actors'][index % 4]


def request(config, item, arm):
    if arm not in ARMS: raise ValueError('Unknown partitioned-goal arm')
    body = parent_request(config, item, 'plain_history')
    if arm == 'partitioned':
        body['response_format'] = schema(item['context'])
        body['messages'][0]['content'] += '\n' + LAYOUT_PROMPT
    return body


def project(value, count):
    """Lossless field layout conversion. No reference/label/relation repair."""
    result = copy.deepcopy(value)
    evidence = result['goal_consistent']['evidence']
    if not isinstance(evidence, dict) or set(evidence) != {'effect_steps', 'time_steps'}:
        raise ValueError('Missing/extra grouped evidence')
    for name, fields in (('effect_steps', EFFECT_FIELDS), ('time_steps', TIME_FIELDS)):
        rows = evidence[name]
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError('Grouped step denominator differs')
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict) or set(row) != set(fields) or type(row['step_index']) is not int or row['step_index'] != index:
                raise ValueError('Grouped step identity/fields differ')
    rows = []
    for timing, effect in zip(evidence['time_steps'], evidence['effect_steps']):
        rows.append({**timing, **{key: effect[key] for key in EFFECT_FIELDS if key != 'step_index'}})
    result['goal_consistent']['evidence'] = {'steps': rows}
    return result


def parse_response(context, text, arm):
    if arm not in ARMS: raise ValueError('Unknown partitioned-goal arm')
    if arm == 'original': return parent_parse(context, text, 'plain_history')
    model = None
    try:
        value = strict_json(text); model = parse(text)
        from .effect_evidence_ablation import action_steps
        projected = project(value, len(action_steps(context)))
        _, decision, error = parent_parse(context, json.dumps(projected, ensure_ascii=False), 'plain_history')
        if error is not None: return model, None, error
        for name in ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute'):
            if (model[name]['label'], model[name]['probability']) != (decision[name]['label'], decision[name]['probability']):
                raise AssertionError('Projection changed predicted labels')
        if model['verdict'] != decision['verdict']: raise AssertionError('Projection changed verdict')
        return model, decision, None
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return model, None, str(exc)


def evaluate(config, inputs, records, historical, labels):
    if any(row['arm'] not in ARMS for row in records): raise ValueError('Unexpected partitioned-goal arm')
    mapped = [{**row, 'arm': 'reverse' if row['arm'] == 'partitioned' else 'original'} for row in records]
    result = parent_evaluate(config, inputs, mapped, historical, labels)
    aliases = {'original': 'original', 'reverse': 'partitioned', 'historical_history_compact': 'historical_original'}
    for key in ('arms', 'diagnostic_screen_passed', 'binding_coverage', 'bounded_dimensions', 'strict_parser_diagnostics'):
        result[key] = {aliases[arm]: value for arm, value in result[key].items()}
    result['transitions'] = {'original->partitioned': result['transitions']['original->reverse'],
                            'historical_original->original': result['transitions']['historical_history_compact->original']}
    result['paired_changes'] = {name: [{**{key: value for key, value in row.items() if key != 'reverse'},
        'partitioned': row['reverse']} for row in rows] for name, rows in result['paired_changes'].items()}
    result['scope'] = ('Single representation factor: per-step mixed time/effect evidence versus two field groups '
        'with mechanical schema-layout note. Target, original full context/citation order, generation, total2048 budget, '
        'compact xgrammar and original semantic gates unchanged. Raw grouped predictions retained; lossless projection '
        'and citation dereference never repair labels/relations. Exposed diagnosis only, no native adoption, SR or default change.')
    return result
