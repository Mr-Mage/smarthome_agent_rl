"""Model-proposed task IR with deterministic provenance checks, not a success oracle."""
import copy
import hashlib
import json

KINDS = ('query', 'change', 'schedule', 'unsupported')
GOAL_FIELDS = {'goal_id', 'source_text', 'kind', 'target_text', 'condition_text',
               'time_text', 'depends_on', 'interpretation'}
CONTRACT = '''TASK SPEC CONTRACT
On the first valid response, task_spec contains goals extracted only from the actual
user request. Use IDs g1, g2, ...; source_text is a unique verbatim excerpt, not a
paraphrase. target_text, condition_text and time_text are verbatim parts of that
excerpt, or empty when unresolved. Keep distinct requested goals separate, including
settings, running and future actions. Do not turn an event condition into a new user
goal. Preserve event/time constraints literally; do not invent a timestamp.
depends_on references other extracted goals; use unresolved for ambiguous meaning.
Later task_spec must be null (the extracted goals are frozen).
goal_refs links this one proposed tool call to zero or more extracted goal IDs.
goal_bindings lists updated goal_id/device_ids; IDs must come from successful public
observations in this episode. Empty device_ids clears an unresolved binding.
These are model interpretations. Tool acknowledgement or workflow registration does
not prove a requested condition is satisfied. Do not invent completion evidence.
'''


def observed_devices(observations):
    catalog = {}
    for ordinal, row in enumerate(observations, 1):
        response = row['response']
        if not isinstance(response, dict) or response.get('status', {}).get('code') != 200 or response.get('error') is not None:
            continue
        tool, args, data = row['tool'], row['arguments'], response.get('data')
        if tool in ('add_device', 'remove_device'):
            catalog.clear()  # Require fresh discovery after topology mutations.
            continue
        if not isinstance(data, dict):
            continue
        devices = []
        if tool == 'get_device_structure' and data.get('device_id') == args.get('device_id'):
            if isinstance(data.get('device_id'), str):
                devices = [data['device_id']]
        elif tool == 'get_room_devices':
            catalog = {key: value for key, value in catalog.items() if value.get('room_id') != args.get('room_id')}
            devices = [key for key, value in data.items() if isinstance(value, dict) and 'device_type' in value]
        elif tool == 'get_home_state' and isinstance(data.get('rooms'), dict):
            catalog.clear()
            devices = [device['device_id'] for room in data['rooms'].values() if isinstance(room, dict)
                       for device in room.get('devices', []) if isinstance(device, dict)
                       and isinstance(device.get('device_id'), str)]
        for device in devices:
            catalog[device] = {'observation_index': ordinal, 'tool': tool,
                               'room_id': args.get('room_id', catalog.get(device, {}).get('room_id')),
                               'response_sha256': hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()}
    return catalog


def string_list(value, label):
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value) or len(set(value)) != len(value):
        raise ValueError(f'{label} must be unique strings')


class TaskSpec:
    def __init__(self, executor):
        self.executor = executor
        self.query = None
        self.goals = []
        self.bindings = {}
        self.turns = []
        self.errors = []

    def initialize(self, query):
        if self.query is not None:
            raise ValueError('TaskSpec is episode-local; do not reuse across tasks')
        self.query = query

    def augment_schema(self, response_format):
        result = copy.deepcopy(response_format)
        schema = result['json_schema']['schema']
        goal = {'type': 'object', 'properties': {
            key: {'type': 'string'} for key in sorted(GOAL_FIELDS - {'depends_on'})},
            'required': sorted(GOAL_FIELDS), 'additionalProperties': False}
        goal['properties']['kind']['enum'] = list(KINDS)
        goal['properties']['interpretation']['enum'] = ['interpreted', 'unresolved']
        goal['properties']['depends_on'] = {'type': 'array', 'items': {'type': 'string'}}
        schema['properties']['task_spec'] = ({'type': 'null'} if self.goals else {
            'type': 'object', 'properties': {'goals': {'type': 'array', 'items': goal, 'minItems': 1, 'maxItems': 16}},
            'required': ['goals'], 'additionalProperties': False})
        schema['properties']['goal_refs'] = {'type': 'array', 'items': {'type': 'string'}}
        schema['properties']['goal_bindings'] = {'type': 'array', 'items': {
            'type': 'object', 'properties': {'goal_id': {'type': 'string'},
                'device_ids': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['goal_id', 'device_ids'], 'additionalProperties': False}}
        schema['required'] += ['task_spec', 'goal_refs', 'goal_bindings']
        result['json_schema']['name'] = 'smart_home_task_spec_v1'
        return result

    def validate_goals(self, spec):
        if not isinstance(spec, dict) or set(spec) != {'goals'} or not isinstance(spec['goals'], list) or not 1 <= len(spec['goals']) <= 16:
            raise ValueError('Initial task_spec requires 1..16 goals')
        goals = copy.deepcopy(spec['goals'])
        for index, goal in enumerate(goals, 1):
            if not isinstance(goal, dict) or set(goal) != GOAL_FIELDS:
                raise ValueError('Invalid goal fields')
            if not all(isinstance(goal[key], str) for key in GOAL_FIELDS - {'depends_on'}):
                raise ValueError('Goal text fields must be strings')
            if goal['goal_id'] != f'g{index}' or goal['kind'] not in KINDS or goal['interpretation'] not in ('interpreted', 'unresolved'):
                raise ValueError('Invalid goal ID, kind or interpretation')
            text = goal['source_text']
            start = self.query.find(text)
            if not text or start < 0 or self.query.find(text, start + 1) >= 0:
                raise ValueError('source_text must be a unique verbatim excerpt; expand ambiguous quotes')
            for key in ('target_text', 'condition_text', 'time_text'):
                if goal[key] and goal[key] not in text:
                    raise ValueError(f'{key} must be copied from source_text')
            string_list(goal['depends_on'], 'depends_on')
            goal['source_span'] = {'start': start, 'end': start + len(text)}
        by_id = {g['goal_id']: g for g in goals}
        visited, active = set(), set()
        def visit(goal_id):
            if goal_id not in by_id or goal_id in active:
                raise ValueError('Goal dependencies must reference known IDs and form a DAG')
            if goal_id in visited:
                return
            active.add(goal_id)
            for dependency in by_id[goal_id]['depends_on']:
                visit(dependency)
            active.remove(goal_id)
            visited.add(goal_id)
        for goal_id in by_id:
            visit(goal_id)
        return goals

    def prepare(self, body):
        if self.query is None:
            raise ValueError('TaskSpec was not initialized from the actual request')
        if not isinstance(body, dict) or not {'task_spec', 'goal_refs', 'goal_bindings'} <= set(body):
            raise ValueError('Missing task_spec metadata')
        if self.goals:
            if body['task_spec'] is not None:
                raise ValueError('Extracted goals are frozen; task_spec must be null')
            goals = self.goals
        else:
            goals = self.validate_goals(body['task_spec'])
        ids = {g['goal_id'] for g in goals}
        refs = body['goal_refs']
        string_list(refs, 'goal_refs')
        if not set(refs) <= ids:
            raise ValueError('Unknown goal_refs')
        catalog = observed_devices(self.executor.observations)
        bindings = {key: value for key, value in self.bindings.items()
                    if all(device in catalog for device in value['device_ids'])}
        updates = body['goal_bindings']
        if not isinstance(updates, list):
            raise ValueError('goal_bindings must be an array')
        seen = set()
        for binding in updates:
            if not isinstance(binding, dict) or set(binding) != {'goal_id', 'device_ids'}:
                raise ValueError('Invalid binding fields')
            goal_id, devices = binding['goal_id'], binding['device_ids']
            if not isinstance(goal_id, str) or goal_id not in ids or goal_id in seen:
                raise ValueError('Unknown or duplicated binding goal')
            seen.add(goal_id)
            string_list(devices, 'device_ids')
            if not set(devices) <= set(catalog):
                raise ValueError('Device bindings require successful public observations')
            bindings[goal_id] = {'device_ids': devices, 'sources': [catalog[d] for d in devices],
                                 'semantic_match': 'model_proposed_unverified'}
        action_body = {k: v for k, v in body.items() if k not in ('task_spec', 'goal_refs', 'goal_bindings')}
        return action_body, {'goals': goals, 'bindings': bindings, 'goal_refs': refs}

    def commit(self, draft, action, turn):
        self.goals, self.bindings = copy.deepcopy(draft['goals']), copy.deepcopy(draft['bindings'])
        self.turns.append({'turn': turn,
                           'goal_refs': list(draft['goal_refs']), 'tool': action['action'],
                           'bindings': copy.deepcopy(self.bindings),
                           'scope': 'model_proposed_association'})

    def prompt(self):
        catalog = observed_devices(self.executor.observations)
        bindings = {key: value for key, value in self.bindings.items()
                    if all(device in catalog for device in value['device_ids'])}
        context = {'goals': self.goals, 'bindings': bindings,
                   'coverage': 'extraction_completeness_and_satisfaction_unverified'}
        if self.errors:
            context['last_metadata_error'] = self.errors[-1]
        return CONTRACT + '\n' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))

    def snapshot(self):
        catalog = observed_devices(self.executor.observations)
        records = self.executor.actions.snapshot()['actions']
        goals = []
        for goal in self.goals:
            binding = self.bindings.get(goal['goal_id'])
            if binding and not all(device in catalog for device in binding['device_ids']):
                binding = None
            related = [r for r in records if not r['extra_query'] and goal['goal_id'] in r['goal_ids']]
            goals.append({**goal, 'binding': binding, 'satisfaction': 'unverified',
                          'action_evidence': [{'action_id': r['action_id'], 'state': r['state'],
                                               'scope': 'action_only'} for r in related]})
        return {'schema': 'episode-task-spec-v1', 'status': 'extracted' if self.goals else 'unresolved',
                'query_sha256': hashlib.sha256((self.query or '').encode()).hexdigest(),
                'coverage': 'model_extraction_unverified', 'goals': goals,
                'turns': copy.deepcopy(self.turns), 'errors': list(self.errors)}
