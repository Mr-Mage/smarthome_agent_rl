"""Contract -> execute once -> public read-back -> postcondition decision.

Receipts do not decide business success. Uncertain dispatch is never replayed;
read-back may prove the intended state even after a timeout.
"""
import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from ..device_contract import DeviceContract, FunctionContract, _get_path
from .trace import ToolTrace


class VerificationStatus(str, Enum):
    VERIFIED_SUCCESS = 'VERIFIED_SUCCESS'
    VERIFIED_FAILURE = 'VERIFIED_FAILURE'
    UNVERIFIED = 'UNVERIFIED'


def resolve(value, *, action, state):
    """Restricted data references only; no expressions or executable strings."""
    if isinstance(value, str) and value.startswith('$'):
        roots = {'$args': action.get('args', {}), '$action': action, '$state': state}
        prefix, _, path = value.partition('.')
        if prefix not in roots:
            raise ValueError(f'Unknown contract reference {value}')
        found, result = _get_path(roots[prefix], path)
        if not found:
            raise ValueError(f'Missing contract reference {value}')
        return copy.deepcopy(result)
    if isinstance(value, Mapping):
        return {key: resolve(child, action=action, state=state) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve(child, action=action, state=state) for child in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class PostconditionResult:
    status: VerificationStatus
    evidence: tuple[dict[str, Any], ...] = ()
    reason: str | None = None

    def as_dict(self):
        return asdict(self)


def check_postconditions(function: FunctionContract, action, before, after) -> PostconditionResult:
    if not function.postconditions:
        return PostconditionResult(VerificationStatus.UNVERIFIED, reason='POSTCONDITION_UNCOVERED')
    evidence = []
    missing = False
    mismatch = False
    for condition in function.postconditions:
        path = condition['path']
        found, actual = _get_path(after, path)
        expected = resolve(condition.get('value'), action=action, state=before)
        tolerance = condition.get('tolerance', 0)
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Postcondition tolerance must be finite and nonnegative')
        if not found:
            matched = None
            missing = True
        elif type(expected) in (int, float) and type(actual) in (int, float):
            matched = math.isfinite(expected) and math.isfinite(actual) and abs(actual - expected) <= tolerance
        else:
            matched = type(actual) is type(expected) and actual == expected
        mismatch |= matched is False
        evidence.append({'path': path, 'expected': expected, 'observed': actual,
                         'found': found, 'tolerance': tolerance, 'matched': matched})
    status = (VerificationStatus.VERIFIED_FAILURE if mismatch else
              VerificationStatus.UNVERIFIED if missing else VerificationStatus.VERIFIED_SUCCESS)
    return PostconditionResult(status, tuple(evidence), 'PUBLIC_STATE_MISSING' if missing else None)


@dataclass(frozen=True)
class MutationResult:
    status: VerificationStatus
    mutation_invocation_id: str | None = None
    readback_invocation_id: str | None = None
    guard: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)

    def as_dict(self):
        return copy.deepcopy(asdict(self))


class MutationExecutor:
    def __init__(self, trace: ToolTrace):
        self.trace = trace

    def execute(self, tool, arguments, *, contract: DeviceContract, state,
                function_name=None, contract_args=None, task_id=None, workflow_id=None, parent_step=None):
        name = function_name or arguments.get('function') or arguments.get('command_id') or arguments.get('attribute_id')
        action = {**copy.deepcopy(arguments), 'function': name,
                  'args': copy.deepcopy(contract_args if contract_args is not None else arguments.get('args', {}))}
        guard = contract.validate(action, state)
        if not guard.accepted or guard.uncovered:
            return MutationResult(VerificationStatus.UNVERIFIED, guard=guard.response(),
                                  verification={'reason': 'REJECTED_OR_UNCOVERED_PRECONDITION', 'executed': False})
        function = contract.functions[name]
        # Resolve all references before dispatch; malformed contracts must not
        # cause mutations followed by a failed verifier configuration.
        query = resolve(function.verification, action=action, state=state)
        for condition in function.postconditions:
            resolve(condition.get('value'), action=action, state=state)
            tolerance = condition.get('tolerance', 0)
            if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError('Invalid postcondition tolerance')
        if query and (not isinstance(query.get('tool'), str) or not isinstance(query.get('args', {}), dict)):
            raise ValueError('Invalid verification query')
        linkage = {'task_id': task_id, 'workflow_id': workflow_id, 'parent_step': parent_step}
        mutation = self.trace.call(tool, arguments, **linkage)
        if not query or not function.postconditions:
            return MutationResult(VerificationStatus.UNVERIFIED, mutation.invocation_id,
                                  guard=guard.response(), verification={'reason': 'VERIFICATION_UNCOVERED'})
        readback = self.trace.call(query['tool'], query.get('args', {}), **linkage)
        response = readback.response
        if readback.error or not isinstance(response, dict) or response.get('status', {}).get('code') != 200:
            return MutationResult(VerificationStatus.UNVERIFIED, mutation.invocation_id, readback.invocation_id,
                                  guard.response(), {'reason': 'READBACK_UNAVAILABLE'})
        after = response.get('data', {})
        verification = check_postconditions(function, action, state, after)
        return MutationResult(verification.status, mutation.invocation_id, readback.invocation_id,
                              guard.response(), verification.as_dict())
