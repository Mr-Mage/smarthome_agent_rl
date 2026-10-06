import unittest

from smarthome_agent_rl.reflection import SelfReflectionVerifier, parse_reflection
from smarthome_agent_rl.semantic_verifier import VerificationContext


def response(label='YES'):
    return '{"correct_target":{"label":"%s","probability":0.9,"evidence":{}},"goal_consistent":{"label":"%s","probability":0.9,"evidence":{}},"trajectory_consistent":{"label":"YES","probability":0.9,"evidence":{}},"safe_to_execute":{"label":"%s","probability":0.9,"evidence":{}}}' % (label, label, label)


class ReflectionTests(unittest.TestCase):
    def test_parser_rejects_chain_of_thought_or_extra_fields(self):
        self.assertEqual(parse_reflection(response()).verdict, 'ALLOW')
        with self.assertRaises(ValueError):
            parse_reflection('{"extra": 1}')

    def test_reflection_preserves_input_and_returns_decisions(self):
        captured = []
        verifier = SelfReflectionVerifier(lambda prompt: (captured.append(prompt) or response('NO')))
        result = verifier.verify(VerificationContext('turn on bedroom', {'x': 1}, {'device_id': 'x'}))
        self.assertEqual(result.verdict, 'DENY')
        self.assertIn('user_goal', captured[0])
        self.assertIn('Do not provide chain-of-thought', captured[0])


if __name__ == '__main__':
    unittest.main()
