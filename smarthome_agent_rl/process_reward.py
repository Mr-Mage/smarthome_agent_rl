"""Frozen process-reward accounting for future Agent Lightning/GRPO rollouts."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ProcessRewardWeights:
    task_success: float = 1.0
    hard_violation: float = 0.10
    semantic_reject: float = 0.05
    excessive_step: float = 0.01

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Reward weights must be finite and nonnegative')


@dataclass(frozen=True)
class RolloutFeedback:
    task_success: bool
    hard_violations: int = 0
    semantic_rejects: int = 0
    steps: int = 0
    max_steps: int = 20

    def __post_init__(self):
        if type(self.task_success) is not bool:
            raise ValueError('task_success must be a boolean')
        if any(type(v) is not int or v < 0 for v in
               (self.hard_violations, self.semantic_rejects, self.steps, self.max_steps)):
            raise ValueError('Rollout counts must be nonnegative integers')

    def reward(self, weights: ProcessRewardWeights = ProcessRewardWeights()) -> float:
        excess = max(0, self.steps - self.max_steps)
        return (weights.task_success * int(self.task_success)
                - weights.hard_violation * self.hard_violations
                - weights.semantic_reject * self.semantic_rejects
                - weights.excessive_step * excess)

    def as_dict(self, weights=ProcessRewardWeights()):
        return {**asdict(self), 'reward': self.reward(weights), 'weights': asdict(weights)}


def rollout_feedback(*, success, audit, steps, max_steps=20):
    hard = sum(1 for row in audit if row.get('blocked') is True and
               row.get('layer') in ('schema', 'capability', 'precondition', 'deterministic'))
    semantic = sum(1 for row in audit if row.get('blocked') is True and row.get('layer') == 'semantic')
    return RolloutFeedback(bool(success), hard, semantic, steps, max_steps)
