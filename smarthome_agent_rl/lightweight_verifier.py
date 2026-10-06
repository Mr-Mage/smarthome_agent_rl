"""Tiny decision verifier for the narrow semantic benchmark.

This is intentionally a small CPU-friendly model.  It is a decision model,
not a language model and never receives hidden evaluator information.
"""
import json
import math
from typing import Any, Mapping

from .semantic_verifier import Decision, VerificationContext, VerificationResult


FEATURES = ('target_known', 'target_matches', 'goal_conflict', 'command_present',
            'trajectory_repeat', 'has_contract')


def features(context: VerificationContext) -> list[float]:
    state = context.environment_state if isinstance(context.environment_state, Mapping) else {}
    devices = state.get('devices', {})
    if isinstance(devices, list):
        devices = {row.get('device_id'): row for row in devices if row.get('device_id')}
    action = context.proposed_action
    selected = devices.get(action.get('device_id'), {}) if isinstance(devices, Mapping) else {}
    goal = context.user_goal.lower()
    room = str(selected.get('room_id', selected.get('room', ''))).lower()
    target_known = bool(selected)
    target_matches = bool(room and room in goal)
    command = str(action.get('command_id', action.get('function', ''))).lower()
    goal_conflict = (('turn on' in goal or '开启' in goal) and command in {'off', 'stop'}) or \
        (('turn off' in goal or '关闭' in goal) and command in {'on', 'start'})
    normalized = (str(action.get('device_id')), command, repr(action.get('args', action.get('arguments', {}))))
    repeated = any((str(row.get('device_id')), str(row.get('command_id', row.get('function', ''))).lower(),
                    repr(row.get('args', row.get('arguments', {})))) == normalized
                    for row in context.recent_actions if isinstance(row, Mapping))
    return [float(target_known), float(target_matches), float(not goal_conflict),
            float(bool(command)), float(not repeated), float(context.contract is not None)]


class LightweightDecisionVerifier:
    def __init__(self, weights=None, bias=0.0, threshold=.5):
        self.weights = list([0.0] * len(FEATURES) if weights is None else weights)
        self.bias, self.threshold = float(bias), float(threshold)
        if len(self.weights) != len(FEATURES):
            raise ValueError('Verifier weight dimension mismatch')
        if not 0 < self.threshold < 1 or any(not math.isfinite(v) for v in [*self.weights, self.bias]):
            raise ValueError('Invalid verifier parameters')

    def probability(self, context: VerificationContext) -> float:
        score = self.bias + sum(w * x for w, x in zip(self.weights, features(context)))
        score = max(-30.0, min(30.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    def verify(self, context: VerificationContext) -> VerificationResult:
        probability = self.probability(context)
        if abs(probability - self.threshold) < .15:
            label = 'UNCERTAIN'
        else:
            label = 'YES' if probability >= self.threshold else 'NO'
        safe = Decision(label, max(probability, 1 - probability), {
            'model': 'lightweight-decision-v1', 'features': dict(zip(FEATURES, features(context)))})
        # This checkpoint learns only safe_to_execute. Other dimensions remain
        # uncovered; rule decisions must not inflate learned-model metrics.
        unknown = Decision('UNCERTAIN', .5, {'uncovered': 'not trained for this dimension'})
        return VerificationResult(unknown, unknown, unknown, safe,
                                  'SMALL_VERIFIER_REJECT' if label == 'NO' else None)

    def fit(self, examples, *, epochs=30, learning_rate=.1, l2=.001):
        if type(epochs) is not int or epochs <= 0 or learning_rate <= 0 or l2 < 0:
            raise ValueError('Invalid training parameters')
        rows = []
        for row in examples:
            label = row.get('labels', {}).get('safe_to_execute')
            if label not in ('YES', 'NO'):
                continue
            context = VerificationContext(row['user_goal'], row['environment_state'],
                                          row['proposed_action'], row.get('recent_actions', ()), row.get('contract'))
            rows.append((features(context), 1.0 if label == 'YES' else 0.0))
        if not rows:
            raise ValueError('No binary safe_to_execute examples')
        for _ in range(epochs):
            for vector, target in rows:
                prediction = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0,
                    self.bias + sum(w * x for w, x in zip(self.weights, vector))))))
                error = prediction - target
                self.bias -= learning_rate * error
                self.weights = [w - learning_rate * (error * x + l2 * w) for w, x in zip(self.weights, vector)]
        return {'examples': len(rows), 'epochs': epochs, 'features': list(FEATURES)}

    def to_dict(self):
        return {'schema': 'lightweight-decision-verifier-v1', 'features': list(FEATURES),
                'weights': self.weights, 'bias': self.bias, 'threshold': self.threshold}

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != 'lightweight-decision-verifier-v1' or value.get('features') != list(FEATURES):
            raise ValueError('Unsupported verifier checkpoint')
        return cls(value['weights'], value['bias'], value['threshold'])


def evaluate(verifier, examples):
    true_bad = blocked_bad = false_reject = true_good = uncertain = 0
    for row in examples:
        context = VerificationContext(row['user_goal'], row['environment_state'], row['proposed_action'],
                                      row.get('recent_actions', ()), row.get('contract'))
        result = verifier.verify(context)
        decision = result.safe_to_execute.label
        uncertain += decision == 'UNCERTAIN'
        expected = row.get('labels', {}).get('safe_to_execute')
        if expected == 'NO':
            true_bad += 1
            blocked_bad += decision == 'NO'
        elif expected == 'YES':
            true_good += 1
            false_reject += decision == 'NO'
    return {'bad_action_recall': blocked_bad / true_bad if true_bad else None,
            'false_reject_rate': false_reject / true_good if true_good else None,
            'true_bad': true_bad, 'true_good': true_good,
            'uncertain': uncertain, 'evaluated': true_bad + true_good,
            'scope': 'learned safe_to_execute only; no rule verdicts'}


def save(verifier, path):
    path.write_text(json.dumps(verifier.to_dict(), indent=2) + '\n', encoding='utf-8')


def load(path):
    return LightweightDecisionVerifier.from_dict(json.loads(path.read_text(encoding='utf-8')))
