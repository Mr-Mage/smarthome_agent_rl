import copy
import json
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.start_semantics import observed_cycle_devices, semantics_prompt
from tests.test_binding import observation, task


def structure(device='washer', start=True):
    return {'device_id': device, 'commands': {'1.OnOff': ['On', 'Off'],
        '1.OperationalState': ['Start', 'Pause'] if start else ['Pause']}}


class StartSemanticsTests(unittest.TestCase):
    def test_metric_rejects_failed_start_and_distinguishes_registration(self):
        from scripts.analyze_start_semantics import missing_start
        messages = observation('get_device_structure', {'device_id': 'washer'},
            {**structure(), 'attributes': {'1.OperationalState.OperationalState': 0}})
        on = {'tool': 'execute_command', 'arguments': {'device_id': 'washer', 'cluster_id': 'OnOff',
              'command_id': 'On'}, 'reached_executor': True, 'simulator_error': False}
        failed_start = {'tool': 'execute_command', 'arguments': {'device_id': 'washer',
            'cluster_id': 'OperationalState', 'command_id': 'Start'}, 'blocked': True}
        audit = {'proposals': [on, failed_start]}
        self.assertTrue(missing_start(audit, messages, 'washer')['omission'])
        audit['proposals'].append({'tool': 'schedule_workflow', 'reached_executor': True,
            'arguments': {'steps': [{'tool': 'execute_command', 'args': failed_start['arguments']}]}})
        result = missing_start(audit, messages, 'washer')
        self.assertFalse(result['omission'])
        self.assertTrue(result['accepted_or_registered_start'])
        self.assertTrue(result['observed_stopped'])

    def test_actual_successful_supported_same_endpoint_only(self):
        demo = observation('get_device_structure', {'device_id': 'demo'}, structure('demo'))
        messages = demo + task() + observation('get_device_structure', {'device_id': 'washer'}, structure(), error={'message': 'not found'})
        self.assertIsNone(semantics_prompt(observed_cycle_devices(messages)))
        messages += observation('get_device_structure', {'device_id': 'washer'}, structure())
        devices = observed_cycle_devices(messages)
        self.assertEqual([d['device_id'] for d in devices], ['washer'])
        self.assertEqual(devices[0]['endpoints'], [1])
        wrong_endpoint = structure('dryer')
        wrong_endpoint['commands']['2.OnOff'] = wrong_endpoint['commands'].pop('1.OnOff')
        messages += observation('get_device_structure', {'device_id': 'dryer'}, wrong_endpoint)
        self.assertEqual(len(observed_cycle_devices(messages)), 1)
        messages += observation('get_device_structure', {'device_id': 'washer'}, structure(start=False))
        self.assertEqual(observed_cycle_devices(messages), [])

    def test_topology_changes_and_mismatched_ids_do_not_keep_contract(self):
        messages = task() + observation('get_device_structure', {'device_id': 'washer'}, structure())
        messages += observation('remove_device', {'device_id': 'washer'}, {})
        messages += observation('get_device_structure', {'device_id': 'washer'}, structure('different'))
        self.assertEqual(observed_cycle_devices(messages), [])

    def test_provider_preserves_history_output_schema_without_actions(self):
        from src.agents.types import ChatMessage
        from smarthome_agent_rl.start_semantics import StartSemanticsProvider
        calls = []
        class Inner:
            def generate(self, messages, response_format=None):
                calls.append((messages, response_format))
                return 'unaltered'
        executor = SimpleNamespace(start_semantics_audit=[], save_audit=lambda: None)
        provider = StartSemanticsProvider(Inner(), executor)
        messages = [ChatMessage(**m) for m in task()]
        schema = {'type': 'json_schema'}
        provider.generate(messages, schema)
        self.assertIs(calls[0][0], messages)
        messages += [ChatMessage(**m) for m in observation('get_device_structure', {'device_id': 'washer'}, structure())]
        original = copy.deepcopy(messages)
        self.assertEqual(provider.generate(messages, schema), 'unaltered')
        self.assertEqual(calls[1][0][:-1], original)
        self.assertIs(calls[1][1], schema)
        self.assertEqual(messages, original)
        self.assertEqual(executor.start_semantics_audit[-1]['devices'][0]['device_id'], 'washer')


if __name__ == '__main__':
    unittest.main()
