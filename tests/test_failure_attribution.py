import json
import unittest
from scripts.audit_failure_attribution import public_errors, required_actions


class AttributionTests(unittest.TestCase):
    def test_error_context_excludes_demonstrations_and_ignores_non_observations(self):
        error = {'status': {'code': 404}, 'error': {'detail': 'not found'}}
        messages = [{'role': 'user', 'content': 'observation:' + json.dumps(error)},
                    {'role': 'user', 'content': 'This is your actual task.'},
                    {'role': 'assistant', 'content': 'observation:' + json.dumps(error)},
                    {'role': 'user', 'content': 'observation:invalid'},
                    {'role': 'user', 'content': 'observation:' + json.dumps({'status': {'code': 200}, 'error': None})},
                    {'role': 'user', 'content': 'observation:' + json.dumps(error)}]
        self.assertEqual(public_errors(messages), [{'message_index': 5, 'response': error}])

    def test_nested_evaluator_receipts_count_explicit_missing_only(self):
        missing = {'tool': 'get_room_devices', 'invoked': False}
        value = {'details': [{'required_actions': [missing, {'tool': 'get_rooms', 'invoked': True},
                              {'tool': 'finish'}]}], 'other': {'invoked': False}}
        self.assertEqual(required_actions(value), [missing])


if __name__ == '__main__':
    unittest.main()
