"""Small, explicit semantic verification interface.

The verifier sees public task text, public state and the proposed action.  It
never receives the evaluator's hidden target or judge output.
"""
from dataclasses import dataclass, asdict, replace
import re
from typing import Any, Callable, Mapping, Sequence

LABELS = ('YES', 'NO', 'UNCERTAIN')
HIDDEN_KEYS = frozenset({'evaluator', 'evaluator_goal', 'judge', 'judge_output',
                         'official_score', 'evaluation_result', 'ground_truth', 'hidden_goal'})


def _find_hidden(value: Any, path='') -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = str(key).casefold()
            child_path = f'{path}.{key}' if path else str(key)
            if key_name in HIDDEN_KEYS:
                return child_path
            found = _find_hidden(child, child_path)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _find_hidden(child, f'{path}[{index}]')
            if found:
                return found
    return None


@dataclass(frozen=True)
class Decision:
    label: str
    probability: float
    evidence: dict[str, Any]

    def __post_init__(self):
        if self.label not in LABELS or not 0 <= self.probability <= 1:
            raise ValueError('Invalid semantic decision')


@dataclass(frozen=True)
class VerificationResult:
    correct_target: Decision
    goal_consistent: Decision
    trajectory_consistent: Decision
    safe_to_execute: Decision
    reason_code: str | None = None

    @property
    def verdict(self) -> str:
        labels = [self.correct_target.label, self.goal_consistent.label,
                  self.trajectory_consistent.label, self.safe_to_execute.label]
        if 'NO' in labels:
            return 'DENY'
        if 'UNCERTAIN' in labels:
            return 'UNCERTAIN'
        return 'ALLOW'

    def as_dict(self):
        return {'verdict': self.verdict, 'reason_code': self.reason_code,
                **{name: asdict(getattr(self, name)) for name in
                   ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute')}}


@dataclass(frozen=True)
class VerificationContext:
    user_goal: str
    environment_state: Mapping[str, Any]
    proposed_action: Mapping[str, Any]
    recent_actions: Sequence[Mapping[str, Any]] = ()
    contract: Mapping[str, Any] | None = None

    def __post_init__(self):
        for name, value in (('environment_state', self.environment_state),
                            ('proposed_action', self.proposed_action),
                            ('recent_actions', self.recent_actions),
                            ('contract', self.contract)):
            hidden = _find_hidden(value)
            if hidden:
                raise ValueError(f'Hidden evaluator information is forbidden: {name}.{hidden}')


def _decision(label, probability, **evidence):
    return Decision(label, probability, evidence)


class RuleSemanticVerifier:
    """Conservative baseline used to validate the benchmark and integration path."""
    room_pattern = re.compile(r'([\w-]+(?:room|bedroom|kitchen|bathroom|living|study))', re.I)

    def verify(self, context: VerificationContext) -> VerificationResult:
        goal = context.user_goal.lower()
        action = context.proposed_action
        state = context.environment_state
        selected_device = action.get('device_id') or action.get('device')
        devices = state.get('devices', {})
        if isinstance(devices, list):
            devices = {row.get('device_id'): row for row in devices if row.get('device_id')}
        selected = devices.get(selected_device, {}) if isinstance(devices, Mapping) else {}
        rooms = [m.group(1).lower() for m in self.room_pattern.finditer(goal)]
        selected_room = str(selected.get('room_id', selected.get('room', ''))).lower()
        if rooms and selected_room:
            target = _decision('YES' if selected_room in rooms else 'NO', .95,
                               requested_rooms=rooms, selected_room=selected_room)
        else:
            target = _decision('UNCERTAIN', .5, requested_rooms=rooms, selected_room=selected_room)
        command = str(action.get('command_id', action.get('function', ''))).lower()
        conflict = (('turn on' in goal or 'switch on' in goal or '开启' in goal) and command in {'off', 'stop'}) or \
                   (('turn off' in goal or 'switch off' in goal or '关闭' in goal) and command in {'on', 'start'})
        goal_decision = _decision('NO' if conflict else 'YES' if command else 'UNCERTAIN', .93 if conflict else .65,
                                  command=command)
        normalized = (str(action.get('device_id')), command, repr(action.get('args', action.get('arguments', {}))))
        previous = [((str(row.get('device_id')), str(row.get('command_id', row.get('function', ''))).lower(),
                      repr(row.get('args', row.get('arguments', {}))))) for row in context.recent_actions]
        trajectory = _decision('NO' if normalized in previous else 'YES', .9 if normalized in previous else .65,
                               repeated=normalized in previous)
        safe = _decision('NO' if conflict else 'UNCERTAIN' if target.label == 'UNCERTAIN' else 'YES',
                         .9 if conflict else .55 if target.label == 'UNCERTAIN' else .7,
                         deterministic_contract_present=context.contract is not None)
        reason = 'TARGET_MISMATCH' if target.label == 'NO' else 'GOAL_CONFLICT' if goal_decision.label == 'NO' else \
            'TRAJECTORY_CONFLICT' if trajectory.label == 'NO' else None
        return VerificationResult(target, goal_decision, trajectory, safe, reason)


class ConfidenceGate:
    def __init__(self, verifier, *, high=0.85, escalation: Callable | None = None):
        if not 0 < high <= 1:
            raise ValueError('high confidence threshold must be in (0, 1]')
        self.verifier, self.high, self.escalation = verifier, high, escalation

    def _high_confidence(self, result: VerificationResult, label: str) -> bool:
        decisions = (result.correct_target, result.goal_consistent,
                     result.trajectory_consistent, result.safe_to_execute)
        return all(decision.label != label or decision.probability >= self.high
                   for decision in decisions) and any(decision.label == label for decision in decisions)

    def _gate(self, result: VerificationResult) -> VerificationResult:
        # A deny is enforceable only when every negative decision is confident;
        # an allow requires all four dimensions to be confident YES. Anything
        # else is escalated or observed as UNCERTAIN by the caller.
        if result.verdict == 'DENY' and self._high_confidence(result, 'NO'):
            return result
        if result.verdict == 'ALLOW' and all(decision.label == 'YES' and
                                             decision.probability >= self.high for decision in
                                             (result.correct_target, result.goal_consistent,
                                              result.trajectory_consistent, result.safe_to_execute)):
            return result
        def uncertain(decision):
            if decision.label == 'NO' and decision.probability < self.high:
                return Decision('UNCERTAIN', decision.probability,
                                {**decision.evidence, 'confidence_gate': self.high})
            if decision.label == 'YES' and decision.probability < self.high:
                return Decision('UNCERTAIN', decision.probability,
                                {**decision.evidence, 'confidence_gate': self.high})
            return decision
        return VerificationResult(*(uncertain(decision) for decision in (
            result.correct_target, result.goal_consistent, result.trajectory_consistent,
            result.safe_to_execute)), result.reason_code or 'SEMANTIC_UNCERTAIN')

    def verify(self, context: VerificationContext) -> VerificationResult:
        result = self.verifier.verify(context)
        gated = self._gate(result)
        if gated.verdict == 'UNCERTAIN' and self.escalation:
            escalated = self.escalation(context, gated)
            if escalated is not None:
                return escalated
        return gated


def feedback(result: VerificationResult) -> dict[str, Any]:
    if result.verdict == 'ALLOW':
        return {'status': 'allowed', 'verifier': result.as_dict()}
    reason = result.reason_code or 'SEMANTIC_UNCERTAIN'
    failed = next((name for name in ('correct_target', 'goal_consistent',
                                     'trajectory_consistent', 'safe_to_execute')
                   if getattr(result, name).label == 'NO'), None)
    repair = {
        'correct_target': 'Re-select a device from the requested room.',
        'goal_consistent': 'Replan an action that matches the user goal.',
        'trajectory_consistent': 'Remove the repeated or conflicting action.',
        'safe_to_execute': 'Escalate for an independent safety check.',
    }.get(failed, 'Replan using public state and contract evidence.')
    return {'status': 'rejected' if result.verdict == 'DENY' else 'uncertain',
            'layer': 'semantic', 'reason_code': reason,
            'failed_dimension': failed, 'evidence': result.as_dict(), 'repair_hint': {
                'action': 'replan', 'repair': repair,
                'ask_for_public_state': result.verdict == 'UNCERTAIN'}}
