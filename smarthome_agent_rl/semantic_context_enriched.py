"""Independent optional reference receipts and public workflow semantics.

Keep the frozen v1 builder unchanged. Reference candidates are observed read
targets, not model-extracted dependencies or current/future state guarantees.
"""
import copy
import json

from .semantic_context import build_context, _source, _targets, _ok, MAX_ENVIRONMENT_CHARS
from .semantic_verifier import VerificationContext
from .workflow_semantics import workflow_context

READS = frozenset({'get_device_structure', 'get_attribute', 'get_all_attributes'})
MAX_REFERENCES, MAX_FACTS = 8, 4


def reference_receipts(action, observations):
    targets = set(_targets(action))
    devices, rooms = {}, {}
    for index, row in enumerate(observations):
        tool, args, response = row['tool'], row['arguments'], row['response']
        device = args.get('device_id')
        data = response.get('data') if isinstance(response, dict) else None
        if tool == 'get_room_devices' and _ok(response) and isinstance(data, dict):
            for key, metadata in data.items():
                if isinstance(key, str) and isinstance(metadata, dict):
                    rooms.setdefault(key, {})[args.get('room_id')] = {'room_id': args.get('room_id'),
                        'metadata': copy.deepcopy(metadata), 'source': _source(index, row)}
        if tool not in READS or not isinstance(device, str) or not device or device in targets:
            continue
        record = devices.setdefault(device, {'device_id': device, 'latest_observation_ordinal': index + 1,
                                            'facts': {}, 'failed_queries_total': 0})
        record['latest_observation_ordinal'] = index + 1
        if not _ok(response) or (tool == 'get_device_structure' and (
                not isinstance(data, dict) or data.get('device_id') != device)):
            record['failed_queries_total'] += 1
            record['latest_failed_query'] = _source(index, row)
            continue
        key = json.dumps([tool, args], sort_keys=True, ensure_ascii=False)
        record['facts'][key] = {'response': copy.deepcopy(response), 'source': _source(index, row),
            'observations_after_read': len(observations) - index - 1,
            'binding': 'response device_id checked' if tool == 'get_device_structure' else
                       'device identity from public tool request; response may not carry identity'}
    ordered = sorted(devices.values(), key=lambda d: (-d['latest_observation_ordinal'], d['device_id']))
    total_facts = sum(len(d['facts']) for d in ordered)
    total_catalog = sum(len(rooms.get(d['device_id'], {})) for d in ordered)
    selected = []
    for record in ordered[:MAX_REFERENCES]:
        facts = sorted(record.pop('facts').values(), key=lambda r: -r['source']['observation_ordinal'])
        memberships = list(rooms.get(record['device_id'], {}).values())
        selected.append({**record, 'role': 'observed reference candidate; not a bound user-goal dependency',
            'state_freshness': 'UNKNOWN', 'facts': facts[:MAX_FACTS], 'facts_total': len(facts),
            'room_id': memberships[0]['room_id'] if len(memberships) == 1 else None,
            'catalog': memberships[-MAX_REFERENCES:], 'catalog_total': len(memberships)})
    return {'devices': selected, 'devices_total': len(ordered), 'facts_total': total_facts,
            'devices_omitted': len(ordered) - len(selected),
            'facts_omitted': total_facts - sum(len(d['facts']) for d in selected),
            'catalog_entries_total': total_catalog,
            'catalog_entries_omitted': total_catalog - sum(len(d['catalog']) for d in selected),
            'semantics': 'Prior public read receipts only. No new query, dependency extraction, '
                         'clock/countdown alignment, predicted finish time or current-state certification.'}


def build_enriched_context(user_goal, proposed_action, observations, *, user_location=None,
                           initial_time=None, contract=None, references=False, workflow_rules=None):
    if not references and workflow_rules is None:
        raise ValueError('Enriched context needs an explicit independent intervention')
    original = build_context(user_goal, proposed_action, observations, user_location=user_location,
                             initial_time=initial_time, contract=contract)
    environment = copy.deepcopy(original.environment_state)
    if references:
        environment['schema'] = 'public-preaction-context-v2'
        environment['observed_reference_receipts'] = reference_receipts(proposed_action, observations)
    if workflow_rules is not None:
        # Rules are public source-bound metadata; hidden keys are checked even
        # when the current action is not a workflow and the projection is None.
        VerificationContext(user_goal, workflow_rules, proposed_action)
        value = workflow_context(proposed_action, workflow_rules)
        if value is not None:
            environment['schema'] = 'public-preaction-context-v2'
            environment['workflow_execution_semantics'] = value
    size = lambda: len(json.dumps(environment, ensure_ascii=False, separators=(',', ':')))
    if references:
        receipt = environment['observed_reference_receipts']
        while receipt['devices'] and size() > MAX_ENVIRONMENT_CHARS:
            removed = receipt['devices'].pop()  # oldest retained candidate first
            receipt['devices_omitted'] += 1
            receipt['facts_omitted'] += len(removed['facts'])
            receipt['catalog_entries_omitted'] += len(removed['catalog'])
        receipt['size_truncated'] = receipt['devices_omitted'] > 0 or receipt['facts_omitted'] > 0
    if size() > MAX_ENVIRONMENT_CHARS:
        raise ValueError('Public reference/workflow context exceeds budget; do not truncate essential task inputs')
    return VerificationContext(original.user_goal, environment, copy.deepcopy(original.proposed_action),
                               copy.deepcopy(original.recent_actions), copy.deepcopy(original.contract))
