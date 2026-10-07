"""Frozen verifier/reward adapters for an Agent Lightning training driver."""
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .process_reward import ProcessRewardWeights, RolloutFeedback, rollout_feedback


@dataclass(frozen=True)
class RLVariant:
    name: str
    include_hard_feedback: bool
    include_semantic_feedback: bool


RL_VARIANTS = {
    'RL-A': RLVariant('RL-A', False, False),
    'RL-B': RLVariant('RL-B', True, False),
    'RL-C': RLVariant('RL-C', True, True),
}


class FrozenVerifierRewardAdapter:
    def __init__(self, variant='RL-A', weights=ProcessRewardWeights()):
        if variant not in RL_VARIANTS:
            raise ValueError(variant)
        self.variant, self.weights = RL_VARIANTS[variant], weights

    def reward(self, *, success: bool, audit: list[Mapping[str, Any]], steps: int, max_steps=20):
        feedback = rollout_feedback(success=success, audit=audit, steps=steps, max_steps=max_steps)
        if not self.variant.include_hard_feedback:
            feedback = RolloutFeedback(feedback.task_success, steps=feedback.steps, max_steps=feedback.max_steps)
        if not self.variant.include_semantic_feedback:
            feedback = RolloutFeedback(feedback.task_success, hard_violations=feedback.hard_violations,
                                       steps=feedback.steps, max_steps=feedback.max_steps)
        # RL-A is the sparse final-task baseline; no process penalties.
        weights = replace(self.weights, excessive_step=0.0) if self.variant.name == 'RL-A' else self.weights
        return feedback.as_dict(weights)


class AlternatingSchedule:
    """Explicit freeze/update schedule; no simultaneous actor/verifier updates."""
    def __init__(self, actor_steps: int, verifier_refresh_every: int):
        if actor_steps <= 0 or verifier_refresh_every <= 0:
            raise ValueError('Schedule values must be positive')
        self.actor_steps = actor_steps
        self.verifier_refresh_every = verifier_refresh_every

    def phase(self, update_index: int) -> str:
        if type(update_index) is not int or not 0 <= update_index < self.actor_steps:
            raise ValueError('Update index outside frozen schedule')
        return 'actor_frozen_verifier_update' if update_index and update_index % self.verifier_refresh_every == 0 else 'actor_update_verifier_frozen'

    def as_dict(self):
        return {'actor_steps': self.actor_steps, 'verifier_refresh_every': self.verifier_refresh_every,
                'simultaneous_updates': False}
