"""Deterministic validation from public tool schemas and versioned cluster code."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime


class GuardError(ValueError):
    def __init__(self, layer, detail, **evidence):
        super().__init__(detail)
        self.layer, self.detail, self.evidence = layer, detail, evidence

    def response(self):
        return {'status': {'code': 422, 'message': 'Harness validation failed'}, 'data': {},
                'error': {'type': 'harness_guard', 'layer': self.layer, 'detail': self.detail,
                          'evidence': self.evidence, 'repair': 'Replan using public capabilities; no action was executed.'}}


def validate_object(schema, arguments, label):
    if not isinstance(arguments, dict):
        raise GuardError('schema', f'{label} arguments must be an object')
    missing = set(schema.get('required', [])) - set(arguments)
    extra = set(arguments) - set(schema['properties']) if not schema.get('additionalProperties', False) else set()
    if missing or extra:
        raise GuardError('schema', f'{label} argument keys invalid', missing=sorted(missing), extra=sorted(extra),
                         expected=schema)
    for key, value in arguments.items():
        rule = schema['properties'].get(key, {})
        kind = rule.get('type')
        valid = value is None and rule.get('nullable') or {
            'string': isinstance(value, str), 'integer': type(value) is int,
            'number': type(value) in (int, float), 'boolean': type(value) is bool,
            'object': isinstance(value, dict), 'array': isinstance(value, list), None: True}[kind]
        if not valid:
            raise GuardError('schema', f'{label}.{key} has invalid type', expected=rule)


def command_contracts(directory):
    """Read signatures without executing constructors or any hidden benchmark data."""
    contracts, sources = {}, {}
    for path in sorted(Path(directory).glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            functions = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            init = functions.get('__init__')
            if init is None:
                continue
            cluster, bindings = None, {}
            for node in ast.walk(init):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '__init__':
                    values = node.args + [kw.value for kw in node.keywords if kw.arg == 'cluster_id']
                    if values and isinstance(values[0], ast.Constant) and isinstance(values[0].value, str):
                        cluster = values[0].value
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'commands'
                                                        for t in node.targets) and isinstance(node.value, ast.Dict):
                    for key, value in zip(node.value.keys, node.value.values):
                        if isinstance(key, ast.Constant) and isinstance(value, ast.Attribute):
                            bindings[key.value] = value.attr
            if not cluster:
                continue
            for command, function_name in bindings.items():
                function = functions.get(function_name)
                if function is None:
                    continue
                properties, required = {}, []
                positional = function.args.posonlyargs + function.args.args
                defaults = [None] * (len(positional) - len(function.args.defaults)) + function.args.defaults
                arguments = list(zip(positional, defaults)) + list(zip(function.args.kwonlyargs, function.args.kw_defaults))
                for argument, default in arguments:
                    if argument.arg == 'self':
                        continue
                    annotation = ast.unparse(argument.annotation) if argument.annotation else 'Any'
                    optional = annotation.startswith('Optional[')
                    kind = annotation[9:-1] if optional else annotation
                    rule = {'type': {'int': 'integer', 'float': 'number', 'str': 'string',
                                     'bool': 'boolean', 'dict': 'object', 'list': 'array'}.get(kind)}
                    # Python annotations are not runtime constraints. Only reject a type when
                    # public implementation checks prove it invalid; keep annotations as guidance.
                    rule = {'type': None, 'declared_type': rule['type']}
                    if optional:
                        rule['nullable'] = True
                    properties[argument.arg] = rule
                    if default is None:
                        required.append(argument.arg)
                contracts[(cluster, command)] = {'properties': properties, 'required': required,
                    'additionalProperties': function.args.kwarg is not None, 'source': path.name}
    return contracts, sources


class ToolGuard:
    def __init__(self, schemas, contracts, power_rules=None):
        self.schemas, self.contracts = schemas, contracts
        self.power_rules = power_rules or {}

    def schema(self, tool, arguments):
        if tool not in self.schemas:
            raise GuardError('schema', f'Unknown tool {tool}')
        validate_object(self.schemas[tool], arguments, tool)
        if tool == 'schedule_workflow':
            if not arguments['steps']:
                raise GuardError('schema', 'Workflow steps must not be empty')
            try:
                datetime.strptime(arguments['start_time'], '%Y-%m-%d %H:%M:%S')
            except ValueError as exc:
                raise GuardError('schema', 'Workflow start_time requires YYYY-MM-DD HH:MM:SS') from exc
            for step in arguments['steps']:
                if not isinstance(step, dict) or set(step) != {'tool', 'args'}:
                    raise GuardError('schema', 'Workflow step requires tool and args')
                if step['tool'] not in ('execute_command', 'write_attribute'):
                    raise GuardError('schema', 'Workflow step must be execute_command or write_attribute')
                self.schema(step['tool'], step['args'])
        if tool == 'execute_command':
            contract = self.contracts.get((arguments['cluster_id'], arguments['command_id']))
            if contract:
                validate_object(contract, arguments['args'], arguments['command_id'])

    def capability(self, tool, arguments, structure, *, state=True):
        if tool not in ('execute_command', 'write_attribute', 'get_attribute'):
            return ['capability_not_applicable']
        endpoint = structure.get('endpoints', {}).get(str(arguments['endpoint_id']))
        if endpoint is None:
            raise GuardError('capability', 'Endpoint does not exist', endpoint=arguments['endpoint_id'])
        cluster = endpoint.get('clusters', {}).get(arguments['cluster_id'])
        if cluster is None:
            raise GuardError('capability', 'Cluster not supported', cluster=arguments['cluster_id'])
        attributes = cluster.get('attributes', {})
        if tool == 'execute_command':
            if arguments['command_id'] not in cluster.get('commands', []):
                raise GuardError('capability', 'Command not supported', commands=cluster.get('commands', []))
            self.known_command_rules(arguments, attributes)
            uncovered = [] if (arguments['cluster_id'], arguments['command_id']) in self.contracts else ['command_signature_uncovered']
            if state:
                uncovered.extend(self.power_precondition(tool, arguments, structure))
            return uncovered
        attribute = attributes.get(arguments['attribute_id'])
        if attribute is None:
            raise GuardError('capability', 'Attribute does not exist', attribute=arguments['attribute_id'])
        if tool == 'write_attribute':
            if attribute.get('readonly'):
                raise GuardError('capability', 'Attribute is readonly', attribute=arguments['attribute_id'])
            self.known_write_rules(arguments, attributes, state=state)
            if state:
                return self.power_precondition(tool, arguments, structure)
        return []

    def power_precondition(self, tool, arguments, structure):
        rule = self.power_rules.get(structure.get('device_type'))
        if rule is None:
            return ['device_preconditions_uncovered']
        cid = arguments['cluster_id']
        applies = (tool == 'execute_command' and cid in rule['commands']) or (
            tool == 'write_attribute' and arguments['endpoint_id'] == 1 and
            arguments['attribute_id'] in rule['attributes'].get(cid, []))
        if applies:
            power = structure.get('endpoints', {}).get('1', {}).get('clusters', {}).get('OnOff', {}).get(
                'attributes', {}).get('OnOff', {}).get('value')
            if power is False:
                raise GuardError('precondition', 'Power is OFF; issue On before this operation', source=rule['source'])
            if power is None:
                return ['power_state_unknown']
        return ['other_device_preconditions_uncovered']

    @staticmethod
    def known_command_rules(arguments, attributes):
        cid, command, args = arguments['cluster_id'], arguments['command_id'], arguments['args']
        if cid == 'LevelControl' and command in ('MoveToLevel', 'MoveToLevelWithOnOff'):
            if not isinstance(args['Level'], (int, float)) or not 1 <= args['Level'] <= 254:
                raise GuardError('capability', 'Level must be between 1 and 254', source='level_control.py')
        if cid == 'TemperatureControl' and command == 'SetTemperature':
            target = args.get('target_temperature')
            if target is not None and 'MinTemperature' in attributes and 'MaxTemperature' in attributes:
                low, high = attributes['MinTemperature']['value'], attributes['MaxTemperature']['value']
                if not isinstance(target, (int, float)) or not low <= target <= high:
                    raise GuardError('capability', 'Temperature outside device range', minimum=low, maximum=high)

    @staticmethod
    def known_write_rules(arguments, attributes, *, state):
        if arguments['cluster_id'] == 'Thermostat' and arguments['attribute_id'] in (
                'OccupiedHeatingSetpoint', 'OccupiedCoolingSetpoint'):
            value = arguments['value']
            if type(value) is not int or not 700 <= value <= 3200:
                raise GuardError('capability', 'Thermostat setpoint requires integer 700..3200', source='thermostat.py')
            if state:
                heating = value if arguments['attribute_id'] == 'OccupiedHeatingSetpoint' else attributes['OccupiedHeatingSetpoint']['value']
                cooling = value if arguments['attribute_id'] == 'OccupiedCoolingSetpoint' else attributes['OccupiedCoolingSetpoint']['value']
                if cooling - heating < 25:
                    raise GuardError('precondition', 'Cooling - Heating must be >= 25', heating=heating, cooling=cooling)


def public_power_rules(directory):
    """Extract the narrow explicit OnOff helper pattern; unknown patterns stay uncovered."""
    rules, sources = {}, {}
    for path in sorted(Path(directory).glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        sources['devices/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            functions = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            check = functions.get('_check_power_dependency')
            if check is None:
                continue
            # Only the known endpoint 1 OnOff read is supported by this deterministic rule.
            reads = [n for n in ast.walk(check) if isinstance(n, ast.Call) and
                isinstance(n.func, ast.Attribute) and n.func.attr == 'get_attribute']
            if not any([ast.literal_eval(a) for a in n.args] == [1, 'OnOff', 'OnOff'] for n in reads):
                continue
            attributes = {}
            attr_fn = functions.get('_check_power_dependency_for_attribute')
            if attr_fn:
                for n in ast.walk(attr_fn):
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'power_dependent_attributes'
                                                         for t in n.targets):
                        attributes = ast.literal_eval(n.value)
            commands = []
            helper = functions.get('_execute_power_dependent_command')
            execute = functions.get('execute_command')
            if helper and execute and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                    n.func.attr == '_check_power_dependency' for n in ast.walk(helper)):
                for branch in [n for n in ast.walk(execute) if isinstance(n, ast.If)]:
                    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                               n.func.attr == '_execute_power_dependent_command' for b in branch.body for n in ast.walk(b)):
                        continue
                    for n in ast.walk(branch.test):
                        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == 'cluster_id':
                            if len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq) and isinstance(n.comparators[0], ast.Constant):
                                commands.append(n.comparators[0].value)
                            elif len(n.ops) == 1 and isinstance(n.ops[0], ast.In):
                                commands.extend(ast.literal_eval(n.comparators[0]))
            rules[path.stem] = {'commands': commands, 'attributes': attributes, 'source': 'devices/' + path.name}
    return rules, sources


def harness_schemas(base):
    """Include optional public API fields absent from the tool docstring table."""
    from src.simulator.api import schemas as api
    result = copy.deepcopy(base)
    mappings = {'add_device': api.AddDeviceRequest, 'set_tick_interval': api.SetTickIntervalRequest,
        'execute_command': api.ExecuteCommandRequest, 'write_attribute': api.WriteAttributeRequest,
        'schedule_workflow': api.ScheduleWorkflowRequest}
    for tool, model in mappings.items():
        for key, rule in model.model_json_schema()['properties'].items():
            if key not in result[tool]['properties']:
                result[tool]['properties'][key] = rule
    return result
