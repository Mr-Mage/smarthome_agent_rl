"""Repeat observed identity beside each proposed step, without deciding support."""
import copy

from .citation_review import request as citation_request, schema as citation_schema, resolve
from .effect_evidence_ablation import action_steps, metadata_equal
from .identity_presentation_ablation import evaluate as presentation_evaluate
from .semantic_diagnosis import parse

ARMS = ('control', 'identity')


def observed_identity(context, device_id):
    """Catalog-derived public metadata; never decode IDs or match user wishes."""
    observed = context['environment_state'].get('devices', {}).get(device_id)
    if not isinstance(observed, dict):
        return None
    catalogs = observed.get('catalog', [])
    return {'room_id': copy.deepcopy(observed.get('room_id')),
            'catalog_total': observed.get('catalog_total'),
            'catalog_truncated': observed.get('catalog_truncated'),
            'catalog': [{'room_id': copy.deepcopy(row.get('room_id')),
                         'device_type': copy.deepcopy(row.get('metadata', {}).get('device_type')) if isinstance(row.get('metadata'), dict) else None,
                         'source': copy.deepcopy(row.get('source'))} for row in catalogs]}


def schema(context, arm):
    if arm not in ARMS:
        raise ValueError('Unknown identity-evidence arm')
    value = citation_schema(context, 'citations')
    if arm == 'identity':
        entries = value['json_schema']['schema']['properties']['correct_target']['properties']['evidence']['properties']['steps']['prefixItems']
        for entry in entries:
            fields = entry['properties']; device = fields['device_id']['const']
            # Evidence-first overall; public identity immediately precedes the
            # support reference/relation. All predicted fields stay unconstrained.
            entry['properties'] = {key: child for key, child in fields.items() if key in ('step_index', 'device_id')}
            entry['properties']['observed_identity'] = {'const': observed_identity(context, device)}
            entry['properties'].update({key: child for key, child in fields.items() if key not in ('step_index', 'device_id')})
            entry['required'] = list(entry['properties'])
    return value


def identity_panel(context):
    return [{'step_index': index + 1, 'device_id': step['arguments'].get('device_id'),
             'observed_identity': observed_identity(context, step['arguments'].get('device_id'))}
            for index, step in enumerate(action_steps(context))]


def request(config, item, arm):
    body = citation_request(config, item, 'citations')
    body['response_format'] = schema(item['context'], arm)
    return body


def parse_response(context, text, arm):
    """Keep raw model output. Validate/remove auxiliary metadata, then dereference."""
    if arm not in ARMS:
        raise ValueError('Unknown identity-evidence arm')
    model = None
    try:
        model = parse(text); value = copy.deepcopy(model)
        rows = value['correct_target']['evidence'].get('steps')
        if not isinstance(rows, list):
            raise ValueError('Missing identity steps')
        for row in rows:
            if arm == 'identity':
                if 'observed_identity' not in row or not metadata_equal(row['observed_identity'], observed_identity(context, row.get('device_id'))):
                    raise ValueError('Generated identity differs from public receipt')
                row.pop('observed_identity')
            elif 'observed_identity' in row:
                raise ValueError('Unexpected identity metadata in control')
        decision = resolve(context, value, 'citations')
        for name in ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute'):
            if decision[name]['label'] != model[name]['label'] or decision[name]['probability'] != model[name]['probability']:
                raise AssertionError('Metadata projection changed predicted decision')
        if decision['verdict'] != model['verdict']:
            raise AssertionError('Metadata projection changed verdict')
        return model, decision, None
    except (ValueError, TypeError, KeyError) as exc:
        return model, None, str(exc)


def evaluate(config, inputs, records, historical, labels):
    result = presentation_evaluate(config, inputs, records, historical, labels)
    result['arms']['historical_control'] = result['arms'].pop('historical_citations')
    result['transitions']['historical_control->control'] = result['transitions'].pop('historical_citations->control')
    result['raw_identity_metadata_valid'] = {arm: sum(row['parse_error'] is None and row['decision'] is not None
        for row in records if row['arm'] == arm) for arm in ARMS}
    result['scope'] = ('Single-factor target evidence schema;public identity const fields precede support references. '
                       'Messages,prompt,citation catalog,context,generation unchanged. Raw metadata retained;'
                       'only verified auxiliary metadata removed andreferences dereferenced forlegacy checks;labels unchanged. '
                       'Const identity is not model identity understanding orsupport truth,no native admission,newSR orTask completion.')
    return result
