import copy
from dataclasses import asdict
import json
import unittest

from smarthome_agent_rl.semantic_context import build_context, digest, MAX_ENVIRONMENT_CHARS


def receipt(tool, arguments, data, *, code=200):
    return {'turn': 3, 'tool': tool, 'arguments': arguments,
            'response': {'status': {'code': code}, 'data': data,
                         'error': None if code == 200 else {'type': 'failed'}}}


class PublicSemanticContextTests(unittest.TestCase):
    action = {'tool': 'schedule_workflow', 'start_time': '2030-01-01 12:01:00', 'steps': [
        {'tool': 'execute_command', 'args': {'device_id': 'a', 'command_id': 'On'}},
        {'tool': 'execute_command', 'args': {'device_id': 'b', 'command_id': 'Off'}}]}

    def build(self, observations, action=None):
        return build_context('Switch the specified devices at noon.', action or self.action, observations,
                             user_location='living', initial_time='2030-01-01 11:59:00')

    def test_workflow_retains_public_catalog_and_structure_with_exact_provenance(self):
        observations = [receipt('get_room_devices', {'room_id': 'living'}, {'a': {'device_type': 'light'},
                                                                                  'b': {'device_type': 'fan'}}),
                        receipt('get_device_structure', {'device_id': 'a'}, {'device_id': 'a', 'endpoints': {}})]
        original = copy.deepcopy(observations)
        context = self.build(observations)
        state = context.environment_state
        self.assertEqual(state['requested_devices_total'], 2)
        self.assertEqual(state['missing_structure_total'], 1)
        self.assertEqual(state['devices']['a']['structure'], observations[1]['response']['data'])
        source = state['devices']['a']['structure_source']
        self.assertEqual(source['observation_ordinal'], 2)
        self.assertEqual(source['response_sha256'], digest(observations[1]['response']))
        self.assertEqual(state['devices']['b']['room_id'], 'living')
        self.assertIsNone(state['devices']['b']['structure'])
        self.assertIsNone(state['latest_observed_public_clock'])
        context.environment_state['devices']['a']['structure']['new'] = 'local'
        self.assertEqual(observations, original)

    def test_future_or_failed_observation_never_substitutes_a_fresh_verified_state(self):
        rows = [receipt('get_device_structure', {'device_id': 'a'}, {'device_id': 'a', 'state': 'Off'}),
                receipt('execute_command', {'device_id': 'a', 'command_id': 'On'}, {'ack': True}),
                receipt('get_device_structure', {'device_id': 'a'}, {}, code=404)]
        context = self.build(rows)
        device = context.environment_state['devices']['a']
        self.assertEqual(device['structure']['state'], 'Off')
        self.assertEqual(device['state_freshness'], 'UNKNOWN')
        self.assertEqual(device['later_mutation_ordinals'], [2])
        self.assertEqual(device['issues'][0]['reason'], 'failed_or_mismatched_structure')
        self.assertEqual(context.recent_actions[0]['response'], rows[1]['response'])
        self.assertNotIn('verified', context.recent_actions[0])

    def test_mismatched_device_structure_and_conflicting_catalog_remain_unresolved(self):
        rows = [receipt('get_room_devices', {'room_id': room}, {'a': {'device_type': 'light'}})
                for room in ('bedroom', 'living')]
        rows.append(receipt('get_device_structure', {'device_id': 'a'}, {'device_id': 'other'}))
        device = self.build(rows).environment_state['devices']['a']
        self.assertIsNone(device['room_id'])
        self.assertEqual(device['catalog_total'], 2)
        self.assertIsNone(device['structure'])

    def test_observed_clock_is_separate_from_initial_and_has_receipt_source(self):
        row = receipt('get_current_time', {}, {'now': '2030-01-01 12:00:00'})
        state = self.build([row]).environment_state
        self.assertEqual(state['initial_public_time'], '2030-01-01 11:59:00')
        self.assertEqual(state['latest_observed_public_clock']['value'], '2030-01-01 12:00:00')
        self.assertEqual(state['latest_observed_public_clock']['source']['observation_ordinal'], 1)
        self.assertEqual(state['clock_freshness'], 'UNKNOWN')

    def test_hidden_fields_rejected_even_in_irrelevant_or_oversized_receipts(self):
        row = receipt('unrelated', {}, {'padding': 'x' * 50000, 'nested': [{'judge_output': 'YES'}]})
        with self.assertRaisesRegex(ValueError, 'Hidden evaluator'):
            self.build([row])
        with self.assertRaisesRegex(ValueError, 'Hidden evaluator'):
            self.build([], {**self.action, 'hidden_goal': 'leak'})

    def test_device_and_receipt_budgets_preserve_exact_counts_and_drop_whole_rows(self):
        action = {**self.action, 'steps': [{'tool': 'execute_command', 'args': {'device_id': str(i)}}
                                          for i in range(11)]}
        rows = [receipt('get_device_structure', {'device_id': str(i)}, {'device_id': str(i),
                'public_data': 'x' * 30000}) for i in range(11)]
        rows += [receipt('execute_command', {'device_id': '0', 'command_id': 'On'}, {'ack': True}) for _ in range(10)]
        context = self.build(rows, action)
        state = context.environment_state
        self.assertEqual(state['requested_devices_total'], 11)
        self.assertEqual(state['devices_omitted'], 3)
        self.assertEqual(state['missing_structure_total'], 0)
        self.assertEqual(state['oversized_device_rows'], 8)
        self.assertEqual(state['recent_mutations_total'], 10)
        self.assertTrue(state['recent_mutations_truncated'])
        self.assertEqual(len(context.recent_actions), 8)
        self.assertLessEqual(len(json.dumps(state, ensure_ascii=False, separators=(',', ':'))), MAX_ENVIRONMENT_CHARS)
        self.assertIn('uncovered', state['devices']['0'])

    def test_oversized_recent_receipt_omitted_without_certifying_missing_action(self):
        context = self.build([receipt('execute_command', {'device_id': 'a'}, {'public_blob': 'x' * 30000})])
        self.assertEqual(context.recent_actions, [])
        self.assertEqual(context.environment_state['recent_mutations_total'], 1)
        self.assertTrue(context.environment_state['recent_mutations_truncated'])

    def test_preaction_slice_has_no_later_receipts_and_preserves_action_and_user_text(self):
        rows = [receipt('get_device_structure', {'device_id': 'a'}, {'device_id': 'a', 'state': 'Off'}),
                receipt('execute_command', {'device_id': 'a'}, {'changed': True}),
                receipt('get_device_structure', {'device_id': 'a'}, {'device_id': 'a', 'state': 'On'})]
        context = self.build(rows[:1])
        self.assertEqual(context.environment_state['observation_count'], 1)
        self.assertEqual(context.environment_state['devices']['a']['structure']['state'], 'Off')
        self.assertEqual(context.recent_actions, [])
        self.assertEqual(context.proposed_action, self.action)
        self.assertEqual(asdict(context)['user_goal'], 'Switch the specified devices at noon.')


if __name__ == '__main__':
    unittest.main()
