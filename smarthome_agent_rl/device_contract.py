"""Executable, device-agnostic contracts for deterministic runtime validation.

The contract layer deliberately knows only capabilities, functions, arguments and
public state assertions.  It does not contain device-type-specific branches.
"""
from dataclasses import dataclass, field
from typing import Any, Mapping


def _get_path(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for part in path.split('.') if path else ():
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


def _type_ok(value: Any, expected: str) -> bool:
    return {
        'any': True, 'null': value is None, 'string': isinstance(value, str),
        'integer': type(value) is int, 'number': type(value) in (int, float),
        'boolean': type(value) is bool, 'object': isinstance(value, dict),
        'array': isinstance(value, list),
    }.get(expected, False)


@dataclass(frozen=True)
class ArgumentContract:
    name: str
    type: str = 'any'
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    enum: tuple[Any, ...] = ()

    @classmethod
    def from_dict(cls, name: str, spec: Mapping[str, Any]) -> 'ArgumentContract':
        return cls(name=name, type=spec.get('type', 'any'),
                   required=spec.get('required', True), minimum=spec.get('min', spec.get('minimum')),
                   maximum=spec.get('max', spec.get('maximum')),
                   enum=tuple(spec.get('enum', spec.get('values', ()))))

    def validate(self, value: Any) -> tuple[str, dict[str, Any]] | None:
        if not _type_ok(value, self.type):
            return 'ARGUMENT_TYPE', {'argument': self.name, 'expected': self.type, 'actual_type': type(value).__name__}
        if self.enum and value not in self.enum:
            return 'ARGUMENT_ENUM', {'argument': self.name, 'allowed': list(self.enum), 'actual': value}
        if self.minimum is not None and value < self.minimum:
            return 'ARGUMENT_RANGE', {'argument': self.name, 'minimum': self.minimum, 'actual': value}
        if self.maximum is not None and value > self.maximum:
            return 'ARGUMENT_RANGE', {'argument': self.name, 'maximum': self.maximum, 'actual': value}
        return None


@dataclass(frozen=True)
class Assertion:
    path: str
    op: str
    value: Any = None

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> 'Assertion':
        return cls(spec['path'], spec.get('op', 'eq'), spec.get('value'))

    def check(self, state: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
        found, actual = _get_path(state, self.path)
        if not found:
            return 'UNCOVERED_ASSERTION', {'path': self.path, 'reason': 'public state field missing'}
        checks = {'eq': actual == self.value, 'ne': actual != self.value,
                  'in': actual in self.value, 'not_in': actual not in self.value,
                  'ge': actual >= self.value, 'le': actual <= self.value}
        if self.op not in checks:
            return 'UNCOVERED_ASSERTION', {'path': self.path, 'reason': f'unsupported operator {self.op}'}
        if not checks[self.op]:
            return 'PRECONDITION_FAILED', {'path': self.path, 'expected': {self.op: self.value}, 'actual': actual}
        return None


@dataclass(frozen=True)
class FunctionContract:
    name: str
    capability: str
    arguments: dict[str, ArgumentContract] = field(default_factory=dict)
    assertions: tuple[Assertion, ...] = ()

    @classmethod
    def from_dict(cls, name: str, spec: Mapping[str, Any]) -> 'FunctionContract':
        args = {key: ArgumentContract.from_dict(key, value) for key, value in spec.get('args', {}).items()}
        assertions = tuple(Assertion.from_dict(row) for row in spec.get('assertions', ()))
        return cls(name=name, capability=spec.get('capability', name), arguments=args, assertions=assertions)


@dataclass(frozen=True)
class DeviceContract:
    device_type: str
    capabilities: frozenset[str]
    functions: dict[str, FunctionContract]

    @classmethod
    def from_dict(cls, device_type: str, spec: Mapping[str, Any]) -> 'DeviceContract':
        functions = {name: FunctionContract.from_dict(name, row)
                     for name, row in spec.get('functions', {}).items()}
        return cls(device_type=device_type, capabilities=frozenset(spec.get('capabilities', ())), functions=functions)

    def validate(self, action: Mapping[str, Any], state: Mapping[str, Any]) -> 'ContractResult':
        function_name = action.get('function') or action.get('command_id') or action.get('attribute_id')
        if function_name not in self.functions:
            return ContractResult.reject('UNSUPPORTED_FUNCTION', {'function': function_name,
                'available': sorted(self.functions)}, repair_hint={'select_function': sorted(self.functions)})
        function = self.functions[function_name]
        if function.capability not in self.capabilities:
            return ContractResult.reject('UNSUPPORTED_CAPABILITY', {'capability': function.capability,
                'available': sorted(self.capabilities)})
        arguments = action.get('args', action.get('arguments', {})) or {}
        missing = [name for name, spec in function.arguments.items() if spec.required and name not in arguments]
        extra = sorted(set(arguments) - set(function.arguments))
        if missing or extra:
            return ContractResult.reject('ARGUMENT_SCHEMA', {'missing': missing, 'extra': extra,
                'expected': sorted(function.arguments)})
        for name, spec in function.arguments.items():
            if name in arguments:
                failure = spec.validate(arguments[name])
                if failure:
                    code, evidence = failure
                    return ContractResult.reject(code, evidence)
        uncovered = []
        for assertion in function.assertions:
            failure = assertion.check(state)
            if failure:
                code, evidence = failure
                if code == 'UNCOVERED_ASSERTION':
                    uncovered.append(evidence)
                else:
                    return ContractResult.reject(code, evidence)
        return ContractResult.allow(uncovered=uncovered)


@dataclass(frozen=True)
class ContractResult:
    status: str
    reason_code: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    repair_hint: dict[str, Any] = field(default_factory=dict)
    uncovered: tuple[dict[str, Any], ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status == 'allowed'

    @classmethod
    def allow(cls, *, uncovered=()):
        return cls('allowed', uncovered=tuple(uncovered))

    @classmethod
    def reject(cls, reason_code, evidence=None, repair_hint=None):
        return cls('rejected', reason_code, evidence or {}, repair_hint or {})

    def response(self) -> dict[str, Any]:
        if self.accepted:
            return {'status': 'allowed', 'uncovered': list(self.uncovered)}
        return {'status': 'rejected', 'layer': 'deterministic', 'reason_code': self.reason_code,
                'evidence': self.evidence, 'repair_hint': self.repair_hint}


class ContractRegistry:
    def __init__(self, contracts: Mapping[str, DeviceContract] | None = None):
        self.contracts = dict(contracts or {})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'ContractRegistry':
        return cls({name: DeviceContract.from_dict(name, spec) for name, spec in value.items()})

    def resolve(self, state: Mapping[str, Any]) -> DeviceContract | None:
        device_type = state.get('device_type') or state.get('type')
        return self.contracts.get(device_type)

    def validate(self, action: Mapping[str, Any], state: Mapping[str, Any]) -> ContractResult:
        contract = self.resolve(state)
        if contract is None:
            return ContractResult.reject('UNSUPPORTED_DEVICE_CONTRACT',
                {'device_type': state.get('device_type', state.get('type'))},
                {'load_contract': True})
        return contract.validate(action, state)


def contract_from_structure(structure: Mapping[str, Any]) -> DeviceContract:
    """Create a conservative public contract from a SimuHome structure response."""
    functions: dict[str, dict[str, Any]] = {}
    capabilities: set[str] = set()
    for endpoint in structure.get('endpoints', {}).values():
        for cluster_name, cluster in endpoint.get('clusters', {}).items():
            capabilities.add(cluster_name)
            for command in cluster.get('commands', []):
                functions.setdefault(command, {'capability': cluster_name, 'args': {}})
    return DeviceContract(str(structure.get('device_type', 'unknown')), frozenset(capabilities),
                          {name: FunctionContract.from_dict(name, spec) for name, spec in functions.items()})
