"""Bounded verifier input from receipts already observed before dispatch.

This is a partial public snapshot, not a goal extractor, fresh-state oracle or
future-state prediction. No simulator calls, inference or evaluator access.
"""
import copy
import hashlib
import json

from .semantic_verifier import VerificationContext

MUTATIONS = frozenset({'execute_command', 'write_attribute', 'schedule_workflow',
                      'cancel_workflow', 'add_device', 'remove_device', 'set_tick_interval'})
MAX_DEVICES, MAX_ACTIONS, MAX_ENVIRONMENT_CHARS = 8, 8, 24000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _ok(response):
    return (isinstance(response, dict) and isinstance(response.get('status'), dict)
            and response['status'].get('code') == 200 and response.get('error') is None)


def _source(index, row):
    return {'observation_ordinal': index + 1, 'turn': row.get('turn'), 'tool': row['tool'],
            'arguments': copy.deepcopy(row['arguments']), 'response_sha256': digest(row['response'])}


def _targets(action):
    values = [action.get('device_id')]
    if action['tool'] == 'schedule_workflow':
        values.extend(step.get('args', {}).get('device_id') for step in action.get('steps', []))
    return list(dict.fromkeys(v for v in values if isinstance(v, str) and v))


def build_context(user_goal, proposed_action, observations, *, user_location=None,
                  initial_time=None, contract=None):
    # Validate the entire supplied receipt boundary, including rows that may be
    # omitted by selection/limits. Never silently strip a hidden evaluator key.
    VerificationContext(user_goal, {'observations': observations, 'user_location': user_location,
                                  'initial_time': initial_time}, proposed_action, contract=contract)
    targets = _targets(proposed_action)
    sources, catalogs, issues, clocks, recent = {}, {}, {}, [], []
    for index, row in enumerate(observations):
        tool, args, response = row['tool'], row['arguments'], row['response']
        source = _source(index, row)
        device = args.get('device_id')
        if tool in MUTATIONS:
            recent.append({'tool': tool, **copy.deepcopy(args), 'receipt_source': source,
                           'response': copy.deepcopy(response),
                           'semantics': 'attempt receipt; acknowledgement is not verified effect'})
        if tool == 'get_device_structure' and device in targets:
            data = response.get('data') if isinstance(response, dict) else None
            if _ok(response) and isinstance(data, dict) and data.get('device_id') == device:
                sources[device] = (index, copy.deepcopy(data), source)
            else:
                issues.setdefault(device, []).append({'reason': 'failed_or_mismatched_structure', 'source': source})
        if tool == 'get_room_devices' and _ok(response) and isinstance(response.get('data'), dict):
            for selected in targets:
                metadata = response['data'].get(selected)
                if isinstance(metadata, dict):
                    catalogs.setdefault(selected, {})[args.get('room_id')] = {
                        'room_id': args.get('room_id'), 'metadata': copy.deepcopy(metadata), 'source': source}
        field = 'now' if tool == 'get_current_time' else 'current_time' if tool == 'get_home_state' else None
        if field and _ok(response) and isinstance(response.get('data'), dict):
            value = response['data'].get(field)
            if isinstance(value, str) and value:
                clocks.append({'value': value, 'field': 'data.' + field, 'source': source})
    devices = {}
    for device in targets[:MAX_DEVICES]:
        memberships = list(catalogs.get(device, {}).values())
        # Multiple room observations stay ambiguous; don't infer room from ID.
        row = {'device_id': device, 'room_id': memberships[0]['room_id'] if len(memberships) == 1 else None,
               'catalog': memberships[-MAX_DEVICES:], 'catalog_total': len(memberships),
               'catalog_truncated': len(memberships) > MAX_DEVICES,
               'structure': None, 'structure_source': None, 'state_freshness': 'UNKNOWN',
               'issues': issues.get(device, [])[-MAX_DEVICES:], 'issues_total': len(issues.get(device, []))}
        if device in sources:
            index, data, source = sources[device]
            row.update(structure=data, structure_source=source,
                       observations_after_read=len(observations) - index - 1,
                       later_mutation_ordinals=[i + 1 for i in range(index + 1, len(observations))
                                               if observations[i]['tool'] in MUTATIONS][-MAX_ACTIONS:])
        devices[device] = row
    environment = {'schema': 'public-preaction-context-v1', 'user_location': copy.deepcopy(user_location),
        'initial_public_time': copy.deepcopy(initial_time),
        'latest_observed_public_clock': clocks[-1] if clocks else None,
        'clock_freshness': 'UNKNOWN', 'observation_count': len(observations),
        'devices': devices, 'requested_devices_total': len(targets),
        'devices_omitted': max(0, len(targets) - MAX_DEVICES),
        'missing_structure_total': sum(device not in sources for device in targets),
        'recent_mutations_total': len(recent), 'recent_mutations_truncated': len(recent) > MAX_ACTIONS,
        'semantics': 'Partial public pre-action evidence only. Initial/observed clocks may be stale. '
                     'No current freshness, future state, user-goal satisfaction or action effect is certified.',
        'oversized_device_rows': 0, 'oversized_clock': False}
    # Keep source IDs and absence explicit. Drop whole large fields/rows, never
    # slice JSON into an invalid or misleading partial fact.
    size = lambda: len(json.dumps(environment, ensure_ascii=False, separators=(',', ':')))
    for device in reversed(list(devices)):
        if size() <= MAX_ENVIRONMENT_CHARS:
            break
        devices[device] = {'device_id': device, 'structure': None,
                           'uncovered': 'public_device_row_exceeds_context_budget',
                           'original_row_sha256': digest(devices[device])}
        environment['oversized_device_rows'] += 1
    if size() > MAX_ENVIRONMENT_CHARS and environment['latest_observed_public_clock'] is not None:
        environment['latest_observed_public_clock'] = None
        environment['oversized_clock'] = True
    if size() > MAX_ENVIRONMENT_CHARS:
        raise ValueError('Essential public context exceeds budget; do not truncate task inputs')
    # Large recent receipts are omitted whole but counted. Actions themselves
    # are never synthesized, and failed attempts retain their actual receipts.
    recent = recent[-MAX_ACTIONS:]
    while len(json.dumps(recent, ensure_ascii=False, separators=(',', ':'))) > MAX_ENVIRONMENT_CHARS:
        recent.pop(0)
        environment['recent_mutations_truncated'] = True
    return VerificationContext(user_goal, environment, copy.deepcopy(proposed_action),
                               recent, copy.deepcopy(contract))
