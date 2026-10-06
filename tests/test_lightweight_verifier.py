import unittest

from smarthome_agent_rl.lightweight_verifier import LightweightDecisionVerifier, evaluate
from smarthome_agent_rl.semantic_verifier import VerificationContext


def rows():
    state = {'devices': {'bedroom_light': {'device_id': 'bedroom_light', 'room_id': 'bedroom'},
                         'living_light': {'device_id': 'living_light', 'room_id': 'living_room'}}}
    common = {'user_goal': 'turn on the bedroom light', 'environment_state': state,
              'recent_actions': [], 'proposed_action': {'device_id': 'bedroom_light', 'command_id': 'On', 'args': {}}}
    return [{**common, 'labels': {'safe_to_execute': 'YES'}},
            {**common, 'proposed_action': {'device_id': 'living_light', 'command_id': 'On', 'args': {}},
             'labels': {'safe_to_execute': 'NO'}}]


class LightweightVerifierTests(unittest.TestCase):
    def test_fit_save_roundtrip_and_metrics(self):
        examples = rows()
        model = LightweightDecisionVerifier()
        model.fit(examples, epochs=20)
        result = evaluate(model, examples)
        self.assertIsNotNone(result['bad_action_recall'])
        restored = LightweightDecisionVerifier.from_dict(model.to_dict())
        ctx = VerificationContext(examples[0]['user_goal'], examples[0]['environment_state'], examples[0]['proposed_action'])
        self.assertEqual(restored.probability(ctx), model.probability(ctx))

    def test_untrained_model_cannot_borrow_wrong_target_rule_recall(self):
        result = evaluate(LightweightDecisionVerifier(), rows())
        self.assertEqual(result['bad_action_recall'], 0.0)
        self.assertEqual(result['false_reject_rate'], 0.0)
        self.assertEqual(result['uncertain'], 2)


if __name__ == '__main__':
    unittest.main()
