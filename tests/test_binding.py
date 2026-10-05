import copy
import json
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.binding import catalog_prompt, visible_catalog


def task():
    return [{'role': 'user', 'content': 'This is your actual task. Control a light.'}]


def observation(tool, args, data, *, error=None):
    return [{'role': 'assistant', 'content': json.dumps({'action': tool, 'action_input': json.dumps(args)})},
            {'role': 'user', 'content': 'observation:' + json.dumps(
                {'status': {'code': 200 if error is None else 404}, 'data': data, 'error': error})}]


class BindingTests(unittest.TestCase):
    def test_catalog_ignores_fewshot_failed_observations_and_task_text_ids(self):
        messages = observation('get_rooms', {}, {'rooms': [{'room_id': 'fewshot'}]}) + task()
        messages += observation('get_rooms', {}, {'rooms': [{'room_id': 'study_room', 'display_name': 'Study Room'}]})
        messages += observation('get_room_devices', {'room_id': 'study'}, {'invented_device': {'device_type': 'light'}}, error={'detail': 'not found'})
        before = copy.deepcopy(messages)
        catalog = visible_catalog(messages)
        self.assertEqual(catalog['rooms'], {'study_room': 'Study Room'})
        self.assertEqual(catalog['devices'], {})
        self.assertEqual(messages, before)
        self.assertEqual(len(catalog['sources']), 1)
        self.assertNotIn('fewshot', catalog_prompt(catalog))

    def test_partial_device_catalog_does_not_assert_absence_or_infer_room_from_id(self):
        messages = task() + observation('get_room_devices', {'room_id': 'study_room'},
            {'study_light_1': {'device_type': 'light'}})
        messages += observation('get_device_structure', {'device_id': 'other_id'},
            {'device_id': 'other_id', 'device_type': 'fan'})
        catalog = visible_catalog(messages)
        self.assertFalse(catalog['complete_devices'])
        self.assertEqual(catalog['devices']['other_id']['room_id'], None)
        self.assertIn('not proof of absence', catalog_prompt(catalog))
        self.assertEqual(catalog['devices']['study_light_1']['room_id'], 'study_room')

    def test_topology_changes_invalidate_and_catalog_refresh_replaces_old_ids(self):
        messages = task() + observation('get_home_state', {},
            {'rooms': {'study_room': {'devices': [{'device_id': 'old', 'device_type': 'light'}]}}})
        self.assertTrue(visible_catalog(messages)['complete_devices'])
        messages += observation('remove_device', {'device_id': 'old'}, {})
        self.assertIsNone(catalog_prompt(visible_catalog(messages)))
        messages += observation('get_room_devices', {'room_id': 'study_room'}, {'new': {'device_type': 'light'}})
        messages += observation('get_room_devices', {'room_id': 'study_room'}, {})
        self.assertEqual(visible_catalog(messages)['devices'], {})

    def test_provider_preserves_history_schema_and_output_without_new_tool_calls(self):
        from src.agents.types import ChatMessage
        from smarthome_agent_rl.binding import BindingProvider
        calls = []
        class Inner:
            def generate(self, messages, response_format=None):
                calls.append((messages, response_format))
                return 'unchanged-output'
        executor = SimpleNamespace(binding_audit=[], save_audit=lambda: None)
        provider = BindingProvider(Inner(), executor)
        schema = {'type': 'json_schema'}
        messages = [ChatMessage(**m) for m in task()]
        provider.generate(messages, schema)
        self.assertIs(calls[0][0], messages)
        messages += [ChatMessage(**m) for m in observation('get_rooms', {}, {'rooms': [{'room_id': 'study_room'}]})]
        before = copy.deepcopy(messages)
        self.assertEqual(provider.generate(messages, schema), 'unchanged-output')
        self.assertEqual(calls[1][0][:-1], messages)
        self.assertIs(calls[1][1], schema)
        self.assertEqual(messages, before)
        self.assertTrue(executor.binding_audit[-1]['used'])


if __name__ == '__main__':
    unittest.main()
