import copy
import json
from pathlib import Path
import unittest

from smarthome_agent_rl.guard import public_power_rules, ToolGuard, GuardError, command_contracts
from smarthome_agent_rl.harness_agent import GuardedExecutor
from smarthome_agent_rl.structured import tool_schemas
from src.simulator.domain.devices.dishwasher import Dishwasher
from src.simulator.domain.devices.laundry_washer import LaundryWasher
from src.simulator.domain.devices.laundry_dryer import LaundryDryer

ROOT = Path(__file__).resolve().parents[1]


class GuardV3Tests(unittest.TestCase):
    def test_real_dead_front_start_replay_and_no_false_rejection(self):
        directory = ROOT / 'deps/SimuHome/src/simulator/domain/devices'
        legacy, _ = public_power_rules(directory)
        rules, _ = public_power_rules(directory, dead_front=True)
        contracts, _ = command_contracts(directory.parent / 'clusters')
        guard = ToolGuard(tool_schemas(), contracts, rules)
        for cls in (Dishwasher, LaundryWasher, LaundryDryer):
            with self.subTest(device=cls.__name__):
                device = cls('test-cycle')
                structure = json.loads(json.dumps(device.get_structure()))
                self.assertNotIn(structure['device_type'], legacy)
                action = {'device_id': 'test-cycle', 'endpoint_id': 1,
                    'cluster_id': 'OperationalState', 'command_id': 'Start', 'args': {}}
                with self.assertRaises(GuardError):
                    guard.capability('execute_command', action, structure)
                self.assertFalse(device.execute_command(1, 'OperationalState', 'Start', {}).success)
                guard.capability('execute_command', action, structure, state=False)
                guard.capability('execute_command', {**action, 'command_id': 'Stop'}, structure)
                unknown = copy.deepcopy(structure)
                unknown['endpoints']['1']['clusters']['OnOff']['attributes']['OnOff']['value'] = None
                self.assertIn('power_state_unknown', guard.capability('execute_command', action, unknown))
                self.assertTrue(device.execute_command(1, 'OnOff', 'On', {}).success)
                powered = json.loads(json.dumps(device.get_structure()))
                guard.capability('execute_command', action, powered)
                self.assertTrue(device.execute_command(1, 'OperationalState', 'Start', {}).success)

    def test_second_device_static_capability_is_checked_and_unique_query_budget(self):
        structure = {'endpoints': {'1': {'clusters': {'OnOff': {'commands': ['On'], 'attributes': {}}}}}}
        action = {'device_id': 'a', 'endpoint_id': 1, 'cluster_id': 'OnOff', 'command_id': 'On', 'args': {}}
        workflow = {'start_time': '2030-01-01 12:00:00', 'steps': [
            {'tool': 'execute_command', 'args': action},
            {'tool': 'execute_command', 'args': {**action, 'device_id': 'b'}},
            {'tool': 'execute_command', 'args': {**action, 'device_id': 'b'}}]}
        calls = []
        def dispatch(tool, arguments):
            calls.append((tool, arguments))
            data = copy.deepcopy(structure)
            if arguments.get('device_id') == 'b':
                data['endpoints']['1']['clusters'] = {}
            return {'status': {'code': 200}, 'data': data, 'error': None}
        legacy = GuardedExecutor(dispatch=dispatch)
        self.assertEqual(legacy.execute('schedule_workflow', workflow)['status']['code'], 200)
        calls.clear()
        executor = GuardedExecutor(dispatch=dispatch, workflow_all_devices=True)
        self.assertEqual(executor.execute('schedule_workflow', workflow)['status']['code'], 422)
        self.assertEqual([a['device_id'] for t, a in calls], ['a', 'b'])
        self.assertFalse(any(t == 'schedule_workflow' for t, a in calls))
        calls.clear()
        budget = GuardedExecutor(dispatch=dispatch, workflow_all_devices=True, query_limit=1)
        self.assertEqual(budget.execute('schedule_workflow', workflow)['status']['code'], 200)
        self.assertEqual(budget.extra_queries, 1)
        self.assertIn('prequery_budget_exhausted', budget.audit[-1]['uncovered'])
        calls.clear()
        valid = GuardedExecutor(dispatch=lambda t, a: {'status': {'code': 200}, 'data': structure, 'error': None},
                                workflow_all_devices=True)
        self.assertEqual(valid.execute('schedule_workflow', workflow)['status']['code'], 200)
        self.assertEqual(valid.extra_queries, 2)
