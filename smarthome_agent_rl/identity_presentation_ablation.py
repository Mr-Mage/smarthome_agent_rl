"""Duplicate already-observed identity fields in a compact receipt panel."""
from collections import Counter
import copy
import json

from .citation_review import request as citation_request, schema as citation_schema, evaluate as citation_evaluate
from .citation_binding_report import binding_report

ARMS = ('control', 'identity')
SCOPE = ('Existing public pre-action identity receipts only. No requested-device,numbered-device orphase labels;'
         'catalog observations can be incomplete/stale. Device IDs are references,not natural-language identity proofs.')


def identity_panel(context):
    action = context['proposed_action']
    targets = [step.get('args', {}).get('device_id') for step in action['steps']] if action['tool'] == 'schedule_workflow' else [action.get('device_id')]
    devices = context['environment_state'].get('devices', {})
    rows = []
    for device_id in dict.fromkeys(v for v in targets if isinstance(v, str) and v):
        observed = devices.get(device_id)
        if not isinstance(observed, dict):
            rows.append({'proposed_device_id': device_id, 'observed_identity': None})
            continue
        catalogs = [{'room_id': copy.deepcopy(c.get('room_id')),
                     'device_type': copy.deepcopy(c.get('metadata', {}).get('device_type')) if isinstance(c.get('metadata'), dict) else None,
                     'source': copy.deepcopy(c.get('source'))} for c in observed.get('catalog', [])]
        structure = observed.get('structure')
        rows.append({'proposed_device_id': device_id, 'observed_identity': {
            'room_id': copy.deepcopy(observed.get('room_id')), 'catalog': catalogs,
            'catalog_total': observed.get('catalog_total'), 'catalog_truncated': observed.get('catalog_truncated'),
            'structure_device_type': copy.deepcopy(structure.get('device_type')) if isinstance(structure, dict) else None,
            'structure_source': copy.deepcopy(observed.get('structure_source')),
            'state_freshness': copy.deepcopy(observed.get('state_freshness'))}})
    return {'public_proposed_device_identity_receipts': rows, 'scope': SCOPE}


def schema(context, arm):
    if arm not in ARMS: raise ValueError('Unknown identity arm')
    return citation_schema(context, 'citations')


def request(config, item, arm):
    if arm not in ARMS: raise ValueError('Unknown identity arm')
    body = citation_request(config, item, 'citations')
    if arm == 'identity':
        body['messages'].append({'role': 'user', 'content': json.dumps(identity_panel(item['context']), ensure_ascii=False)})
    return body


def evaluate(config, inputs, records, historical, labels):
    # N83 gates are reused intact on both evidence-first, ID-based arms.
    mapped = [{**row, 'arm': 'quotes' if row['arm'] == 'control' else 'citations'} for row in records]
    if any(row['arm'] not in ARMS for row in records): raise ValueError('Unknown actual arm')
    original = citation_evaluate(config, inputs, mapped, historical, labels)
    result = {**original, 'arms': {name: original['arms'][key] for name, key in
              (('control', 'quotes'), ('identity', 'citations'), ('historical_citations', 'historical_evidence_first'))},
              'diagnostic_screen_passed': {name: original['diagnostic_screen_passed'][key] for name, key in
              (('control', 'quotes'), ('identity', 'citations'))},
              'transitions': {'control->identity': original['transitions']['quotes->citations'],
                              'historical_citations->control': original['transitions']['historical_evidence_first->quotes']}}
    contexts = {item['id']: item['context'] for item in inputs}
    panels = {arm: [binding_report(contexts[row['id']], {**row, 'arm': 'citations'}) for row in records if row['arm'] == arm] for arm in ARMS}
    result['binding_coverage'] = {arm: {
        'record_flags': {flag: sum(p['flags'][flag] for p in rows) for flag in rows[0]['flags']} if rows else {},
        'unsupported_with_public_identity_words_steps': sum(panel['unsupported_with_public_identity_words'] for p in rows for panel in p['identity_panels']),
        'percentage_statuses': dict(Counter(panel['usable_percentage_status'] for p in rows for panel in p['effect_panels']))}
        for arm, rows in panels.items()}
    result['scope'] = ('Single-factor appended identity-receipt presentation;unchanged original context,prompt,citation schema/catalog,generation. '
                       'No selected user identity orphase truth. Frozen screening only,no native adoption,newSR orTask completion.')
    return result
