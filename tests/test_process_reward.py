import unittest

from smarthome_agent_rl.process_reward import ProcessRewardWeights, RolloutFeedback, rollout_feedback
from smarthome_agent_rl.agentic_rl import AlternatingSchedule, FrozenVerifierRewardAdapter


class ProcessRewardTests(unittest.TestCase):
    def test_reward_keeps_task_success_primary(self):
        result = RolloutFeedback(True, hard_violations=2, semantic_rejects=1, steps=20)
        self.assertAlmostEqual(result.reward(ProcessRewardWeights()), .75)

    def test_feedback_counts_layers_without_hidden_goal(self):
        result = rollout_feedback(success=False, steps=21, audit=[
            {'layer': 'deterministic', 'blocked': True}, {'layer': 'semantic', 'blocked': True},
            {'layer': 'semantic', 'blocked': True}, {'layer': 'semantic', 'blocked': False}])
        self.assertEqual(result.hard_violations, 1)
        self.assertEqual(result.semantic_rejects, 2)
        self.assertLess(result.reward(), 0)

    def test_rl_variants_keep_verifier_frozen_and_alternate_updates(self):
        adapter = FrozenVerifierRewardAdapter('RL-C')
        value = adapter.reward(success=True, audit=[{'layer': 'deterministic', 'blocked': True},
            {'layer': 'semantic', 'blocked': True}], steps=3)
        self.assertEqual(value['hard_violations'], 1)
        self.assertEqual(value['semantic_rejects'], 1)
        self.assertFalse(AlternatingSchedule(4, 8).as_dict()['simultaneous_updates'])

    def test_sparse_baseline_has_no_step_penalty(self):
        self.assertEqual(FrozenVerifierRewardAdapter('RL-A').reward(success=True,
            audit=[{'layer': 'semantic', 'blocked': True}], steps=100)['reward'], 1.0)
        self.assertEqual(FrozenVerifierRewardAdapter('RL-B').reward(success=True,
            audit=[{'layer': 'semantic', 'blocked': True}], steps=3)['reward'], 1.0)

    def test_schedule_starts_with_frozen_verifier_and_rejects_out_of_bounds(self):
        schedule = AlternatingSchedule(10, 4)
        self.assertEqual(schedule.phase(0), 'actor_update_verifier_frozen')
        self.assertEqual(schedule.phase(4), 'actor_frozen_verifier_update')
        with self.assertRaises(ValueError):
            schedule.phase(10)

    def test_invalid_reward_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            RolloutFeedback(True, hard_violations=-1)
        with self.assertRaises(ValueError):
            ProcessRewardWeights(semantic_reject=float('nan'))


if __name__ == '__main__':
    unittest.main()
