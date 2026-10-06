import unittest

from smarthome_agent_rl.semantic_verifier import (ConfidenceGate, RuleSemanticVerifier,
    VerificationContext, feedback)


class SemanticVerifierTests(unittest.TestCase):
    def context(self, goal, action, recent=()):
        return VerificationContext(goal, {'devices': {
            'bedroom_ac': {'device_id': 'bedroom_ac', 'room_id': 'bedroom'},
            'living_ac': {'device_id': 'living_ac', 'room_id': 'living_room'},
        }}, action, recent)

    def test_wrong_target_is_detected_without_hidden_evaluator_state(self):
        result = RuleSemanticVerifier().verify(self.context('turn on the bedroom light',
            {'device_id': 'living_ac', 'command_id': 'On', 'args': {}}))
        self.assertEqual(result.correct_target.label, 'NO')
        self.assertEqual(result.reason_code, 'TARGET_MISMATCH')
        self.assertEqual(feedback(result)['reason_code'], 'TARGET_MISMATCH')

    def test_goal_conflict_and_repeated_action_are_detected(self):
        action = {'device_id': 'bedroom_ac', 'command_id': 'Off', 'args': {}}
        result = RuleSemanticVerifier().verify(self.context('turn on the bedroom device', action, [action]))
        self.assertEqual(result.goal_consistent.label, 'NO')
        self.assertEqual(result.trajectory_consistent.label, 'NO')
        self.assertEqual(result.verdict, 'DENY')

    def test_uncertain_cases_can_escalate(self):
        base = RuleSemanticVerifier()
        called = []
        def escalation(ctx, old):
            called.append(old.verdict)
            return base.verify(self.context('turn on bedroom',
                {'device_id': 'bedroom_ac', 'command_id': 'On', 'args': {}}))
        result = ConfidenceGate(base, escalation=escalation).verify(self.context('change it',
            {'device_id': 'bedroom_ac', 'command_id': 'On', 'args': {}}))
        self.assertTrue(called)
        self.assertIn(result.verdict, {'ALLOW', 'UNCERTAIN'})


if __name__ == '__main__':
    unittest.main()
