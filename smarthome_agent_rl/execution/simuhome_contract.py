"""Translate public SimuHome schema and state into device-agnostic contracts.

Cluster rules are declarations in configs/device-contract-rules.json. Endpoint
and cluster identity are part of the function key, so names cannot collide.
"""
import copy
from pathlib import Path
import json

from ..device_contract import DeviceContract

RULES = Path(__file__).resolve().parents[2] / 'configs/device-contract-rules.json'


def function_key(tool, arguments):
    return ':'.join(map(str, (tool, arguments['endpoint_id'], arguments['cluster_id'],
                             arguments.get('command_id', arguments.get('attribute_id')))))


class SimuHomeContractAdapter:
    def __init__(self, signatures, power_rules, rules=None):
        self.signatures, self.power_rules = signatures, power_rules
        self.rules = copy.deepcopy(rules if rules is not None else json.loads(RULES.read_text(encoding='utf-8')))

    def build(self, tool, arguments, structure, *, future=False):
        endpoint = str(arguments['endpoint_id'])
        cid = arguments['cluster_id']
        cluster = structure.get('endpoints', {}).get(endpoint, {}).get('clusters', {}).get(cid)
        name = function_key(tool, arguments)
        capability = f'{endpoint}.{cid}'
        spec = {'capabilities': [capability] if cluster is not None else [], 'functions': {}}
        function = {'capability': capability, 'args': {}, 'assertions': [], 'postconditions': []}
        attrs = cluster.get('attributes', {}) if cluster else {}
        attribute_path = lambda attribute: f'endpoints.{endpoint}.clusters.{cid}.attributes.{attribute}.value'
        public_rule = self.power_rules.get(structure.get('device_type'))
        if tool == 'execute_command' and cluster and arguments['command_id'] in cluster.get('commands', []):
            command = arguments['command_id']
            signature = self.signatures.get((cid, command))
            if signature is None:
                function['assertions'].append({'path': '__uncovered.command_signature', 'value': True})
            else:
                for arg, row in signature['properties'].items():
                    function['args'][arg] = {'type': row.get('type') or 'any',
                                             'nullable': bool(row.get('nullable', False)),
                                             'required': arg in signature.get('required', [])}
            rule = self.rules.get('commands', {}).get(f'{cid}.{command}', {})
            for arg, constraints in rule.get('args', {}).items():
                constraints = copy.deepcopy(constraints)
                for bound in ('min', 'max'):
                    attribute = constraints.pop(f'{bound}_attribute', None)
                    if attribute and attribute in attrs:
                        constraints[bound] = attrs[attribute]['value']
                function['args'].setdefault(arg, {}).update(constraints)
            for row in rule.get('postconditions', []):
                # Optional alternatives produce an expectation only when supplied.
                if row.get('when_arg') and arguments.get('args', {}).get(row['when_arg']) is None:
                    continue
                if row.get('unless_arg') and arguments.get('args', {}).get(row['unless_arg']) is not None:
                    continue
                function['postconditions'].append({'path': attribute_path(row['attribute']), 'value': row['value']})
            dependent = public_rule and (cid in public_rule.get('commands', []) or
                (int(endpoint) == 1 and [cid, command] in public_rule.get('exact_commands', [])))
        elif tool == 'write_attribute' and cluster and arguments['attribute_id'] in attrs:
            attribute = arguments['attribute_id']
            function['readonly'] = bool(attrs[attribute].get('readonly'))
            constraints = copy.deepcopy(self.rules.get('attributes', {}).get(f'{cid}.{attribute}', {}))
            paired = constraints.pop('paired_attribute', None)
            relation, gap = constraints.pop('relation', None), constraints.pop('min_gap', None)
            function['args']['value'] = constraints
            function['postconditions'] = [{'path': attribute_path(attribute), 'value': '$args.value'}]
            if paired and relation == 'cooling_minus_heating' and not future:
                # Resolve the requested value at validation time, not tomorrow's
                # state. This declaration never assumes future preconditions.
                requested = arguments['value']
                op = 'ge' if attribute.endswith('HeatingSetpoint') else 'le'
                if type(requested) in (int, float):
                    expected = requested + gap if op == 'ge' else requested - gap
                    function['assertions'].append({'path': attribute_path(paired), 'op': op, 'value': expected})
            dependent = public_rule and int(endpoint) == 1 and attribute in public_rule.get('attributes', {}).get(cid, [])
        else:
            return DeviceContract.from_dict(str(structure.get('device_type', 'unknown')), spec), name, {}
        if not future:
            if public_rule is None:
                function['assertions'].append({'path': '__uncovered.device_preconditions', 'value': True})
            elif dependent:
                function['assertions'].append({'path': 'endpoints.1.clusters.OnOff.attributes.OnOff.value', 'value': True})
        function['verification'] = {'tool': 'get_device_structure', 'args': {'device_id': arguments['device_id']}}
        function['verification']['pending_paths'] = [attribute_path(attribute) for attribute in
            self.rules.get('pending_attributes', {}).get(cid, []) if attribute in attrs]
        spec['functions'][name] = function
        args = arguments.get('args', {}) if tool == 'execute_command' else {'value': arguments.get('value')}
        return DeviceContract.from_dict(str(structure.get('device_type', 'unknown')), spec), name, args
