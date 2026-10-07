"""Isolate four reviewer decodes; preserve raw components and original gates."""
from collections import Counter
import copy
import json
import statistics

from .benchmarks.runner import valid_usage
from .bounded_review_ablation import strict_json
from .citation_order_ablation import evaluate as parent_evaluate
from .evidence_order_ablation import DIMENSIONS
from .review_factorial import request as parent_request, parse_response as parent_parse

ARMS = ('joint', 'independent')
COMPONENTS = ('joint',) + DIMENSIONS
FOCUS_PROMPT = '''Output scope only: review and return ONLY {name}, using its
original evidence, binding, uncertainty and aggregation rules. Return an object
with exactly the {name} key. Do not generate other dimension decisions. Other
model decisions are not supplied; use the entire original public context.'''


def actor_for(config, index):
    return config['actors'][index % 4]


def request(config, item, component):
    if component not in COMPONENTS: raise ValueError('Unknown independent-review component')
    body = parent_request(config, item, 'plain_history')
    if component != 'joint':
        root = body['response_format']['json_schema']['schema']
        root['properties'] = {component: root['properties'][component]}
        root['required'] = [component]
        body['messages'][0]['content'] += '\n' + FOCUS_PROMPT.format(name=component)
        body['max_tokens'] = config['dimension_tokens'][component]
    return body


def assemble(context, calls, arm):
    """Join fields, then use the original parser/aggregation; never repair labels."""
    if arm not in ARMS: raise ValueError('Unknown independent-review arm')
    expected = ('joint',) if arm == 'joint' else DIMENSIONS
    if set(calls) != set(expected): raise ValueError('Missing/extra review calls')
    value = None; text = ''; model = decision = None
    try:
        if any(calls[name]['error'] is not None for name in expected):
            raise ValueError('Retained component HTTP failure')
        if arm == 'joint':
            text = calls['joint']['text']; value = strict_json(text)
        else:
            value = {}
            for name in DIMENSIONS:
                partial = strict_json(calls[name]['text'])
                if not isinstance(partial, dict) or set(partial) != {name}:
                    raise ValueError('Unexpected component scope: ' + name)
                row = partial[name]
                if not isinstance(row, dict) or set(row) != {'evidence', 'label', 'probability'}:
                    raise ValueError('Missing/extra component decision fields: ' + name)
                value[name] = copy.deepcopy(row)
            # Derived text is an assembly, never an original HTTP response.
            text = json.dumps(value, ensure_ascii=False)
        model, decision, error = parent_parse(context, text, 'plain_history')
        return value, text, model, decision, error
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return value, text, model, None, str(exc)


def costs(calls):
    usages = [call['usage'] for call in calls if valid_usage(call.get('usage'))]
    return {'requests': len(calls), 'http_errors': sum(call['error'] is not None for call in calls),
        'missing_usage': len(calls)-len(usages),
        **{key: sum(usage[key] for usage in usages) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'truncations': sum(call['finish_reason'] == 'length' for call in calls),
        'sum_http_seconds': sum(call['request_seconds'] for call in calls),
        'median_http_seconds': statistics.median(call['request_seconds'] for call in calls) if calls else None}


def metrics_row(row):
    """In-memory adapter for existing gates, not a fabricated HTTP receipt."""
    cost = costs(list(row['calls'].values()))
    usage = {key: cost[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')} if cost['missing_usage'] == 0 else None
    return {**row, 'arm': 'original' if row['arm'] == 'joint' else 'reverse',
        'call': {'text': row['assembled_text'], 'usage': usage,
                 'error': None if cost['http_errors'] == 0 else {'type': 'ComponentHTTPFailure'},
                 'finish_reason': 'length' if cost['truncations'] else 'stop',
                 'request_seconds': cost['sum_http_seconds'], 'derived_metrics_only': True}}


def evaluate(config, inputs, records, historical, labels):
    if any(row['arm'] not in ARMS for row in records): raise ValueError('Unexpected independent-review arm')
    expected = {(item['id'], arm) for item in inputs for arm in ARMS}
    actual = [(row['id'], row['arm']) for row in records]
    expected_components = lambda arm: {'joint'} if arm == 'joint' else set(DIMENSIONS)
    scope_valid = all(set(row['calls']) == expected_components(row['arm']) for row in records)
    legacy = {**config, 'new_requests': config['review_records']}
    result = parent_evaluate(legacy, inputs, [metrics_row(row) for row in records], historical, labels)
    aliases = {'original': 'joint', 'reverse': 'independent', 'historical_history_compact': 'historical_joint'}
    for key in ('arms', 'diagnostic_screen_passed', 'binding_coverage', 'bounded_dimensions', 'strict_parser_diagnostics'):
        result[key] = {aliases[arm]: value for arm, value in result[key].items()}
    result['transitions'] = {'joint->independent': result['transitions']['original->reverse'],
        'historical_joint->joint': result['transitions']['historical_history_compact->original']}
    result['paired_changes'] = {name: [{**{key: value for key, value in row.items() if key not in ('original', 'reverse')},
        'joint': row['original'], 'independent': row['reverse']} for row in rows] for name, rows in result['paired_changes'].items()}
    complete = (len(actual) == config['review_records'] and len(set(actual)) == len(actual)
        and set(actual) == expected and scope_valid
        and sum(len(row['calls']) for row in records) == config['new_requests'])
    result['complete'] = complete
    result['component_costs'] = {}
    result['per_review_costs'] = {}
    for arm in ARMS:
        rows = [row for row in records if row['arm'] == arm]
        calls = [call for row in rows for call in row['calls'].values()]
        cost = costs(calls); panel = result['arms'][arm]
        panel.update(tokens=cost['total_tokens'], prompt_tokens=cost['prompt_tokens'], completion_tokens=cost['completion_tokens'],
            http_errors=cost['http_errors'], missing_usage=cost['missing_usage'], truncations=cost['truncations'],
            actual_http_requests=cost['requests'], median_request_seconds=cost['median_http_seconds'])
        result['per_review_costs'][arm] = {'sum_http_seconds': cost['sum_http_seconds'],
            'median_sum_http_seconds': statistics.median(costs(list(row['calls'].values()))['sum_http_seconds'] for row in rows) if rows else None,
            'median_review_seconds_including_queue': statistics.median(row['review_seconds'] for row in rows) if rows else None,
            'scope': 'Actual HTTP medians per call; sum HTTP times is not wall time. Review span includes frozen queue/start/end receipts.'}
        result['diagnostic_screen_passed'][arm] &= complete and cost['http_errors'] == 0 and cost['missing_usage'] == 0
    for component in COMPONENTS:
        result['component_costs'][component] = costs([row['calls'][component] for row in records if component in row['calls']])
    result['new_model_requests'] = sum(panel['requests'] for panel in result['component_costs'].values())
    result['new_tokens'] = sum(result['arms'][arm]['tokens'] for arm in ARMS)
    result['new_review_records'] = len(records)
    result['scope'] = ('Joint one-call/four-decision review vs four isolated full-context decodes plus lossless assembly. '
        'Same total2048 output cap, explicit per-dimension allocation changes individual caps; same actor/input pair and '
        'unchanged original evidence/reason gates. Raw component HTTP outputs retained; no label/relation repair. '
        'Repeated input cost included, exposed diagnosis only; no independent truth, SR, native adoption or default change.')
    return result
