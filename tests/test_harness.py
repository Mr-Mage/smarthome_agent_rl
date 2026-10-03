import copy
import json
from pathlib import Path
import unittest

from src.simulator.domain.clusters.level_control import LevelControlCluster
from src.simulator.domain.clusters.onoff import OnOffCluster
from smarthome_agent_rl.guard import ToolGuard, GuardError, command_contracts, public_power_rules, harness_schemas
from smarthome_agent_rl.structured import tool_schemas
from smarthome_agent_rl.harness_agent import GuardedExecutor
from smarthome_agent_rl.verification import expected_effect, verify_effect
from smarthome_agent_rl.context import build_ledger, flatten, LedgerProvider
from src.agents.types import ChatMessage
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.contracts, self.sources = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
        self.guard = ToolGuard(tool_schemas(), self.contracts)
        self.structure = {'device_id': 'test-light', 'endpoints': {'1': {'clusters': {
            'LevelControl': LevelControlCluster().get_structure(), 'OnOff': OnOffCluster().get_structure()}}}}
        self.action = {'device_id': 'test-light', 'endpoint_id': 1, 'cluster_id': 'LevelControl',
                       'command_id': 'MoveToLevelWithOnOff', 'args': {'Level': 100}}

    def test_real_public_signature_missing_nested_args_is_rejected_without_mutation(self):
        bad = copy.deepcopy(self.action)
        bad['args'] = {'level': 100}
        snapshot = copy.deepcopy(bad)
        with self.assertRaises(GuardError):
            self.guard.schema('execute_command', bad)
        self.assertEqual(bad, snapshot)
        self.guard.schema('execute_command', self.action)

    def test_readonly_unsupported_command_range_and_endpoint_are_rejected(self):
        for change in ({'endpoint_id': 8}, {'command_id': 'Unknown'}, {'args': {'Level': 255}}):
            action = {**self.action, **change}
            with self.assertRaises(GuardError):
                self.guard.capability('execute_command', action, self.structure)
        with self.assertRaises(GuardError):
            self.guard.capability('write_attribute', {**self.action, 'attribute_id': 'CurrentLevel', 'value': 100}, self.structure)

    def test_unknown_command_signature_is_uncovered_not_rejected(self):
        structure = copy.deepcopy(self.structure)
        structure['endpoints']['1']['clusters']['NewCluster'] = {'attributes': {}, 'commands': ['NewCommand']}
        action = {**self.action, 'cluster_id': 'NewCluster', 'command_id': 'NewCommand', 'args': {'unknown': 9}}
        self.guard.schema('execute_command', action)
        self.assertIn('command_signature_uncovered', self.guard.capability('execute_command', action, structure))

    def test_public_power_precondition_and_optional_api_fields(self):
        from src.simulator.domain.devices.air_conditioner import AirConditioner
        rules, _ = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices')
        guard = ToolGuard(harness_schemas(tool_schemas()), self.contracts, rules)
        device = AirConditioner('test-ac')
        structure = json.loads(json.dumps(device.get_structure()))
        action = {'device_id': 'test-ac', 'endpoint_id': 1, 'cluster_id': 'Thermostat',
                  'attribute_id': 'OccupiedCoolingSetpoint', 'value': 3000}
        with self.assertRaises(GuardError) as blocked:
            guard.capability('write_attribute', action, structure)
        self.assertEqual(blocked.exception.layer, 'precondition')
        self.assertFalse(device.write_attribute(1, 'Thermostat', 'OccupiedCoolingSetpoint', 3000).success)
        guard.capability('write_attribute', action, structure, state=False)
        guard.schema('add_device', {'device_id': 'x', 'device_type': 'fan', 'room_id': 'room', 'attributes': {'x': 1}})

    def test_workflow_failed_prequery_does_not_exceed_one_query(self):
        calls = []
        def dispatch(tool, arguments):
            calls.append(tool)
            return {'status': {'code': 404}, 'data': {}, 'error': {'type': 'not-found'}}
        executor = GuardedExecutor(dispatch=dispatch)
        executor.execute('schedule_workflow', {'start_time': '2030-01-01 12:00:00',
            'steps': [{'tool': 'execute_command', 'args': self.action},
                      {'tool': 'execute_command', 'args': {**self.action, 'device_id': 'b'}}]})
        self.assertEqual(calls.count('get_device_structure'), 1)

    def test_deadband_future_state_is_not_checked_as_current_precondition(self):
        attributes = {'OccupiedHeatingSetpoint': {'value': 2800}, 'OccupiedCoolingSetpoint': {'value': 2000}}
        action = {'cluster_id': 'Thermostat', 'attribute_id': 'OccupiedCoolingSetpoint', 'value': 2100}
        with self.assertRaises(GuardError):
            self.guard.known_write_rules(action, attributes, state=True)
        self.guard.known_write_rules(action, attributes, state=False)

    def test_postcondition_transition_and_suppression_are_not_false_success(self):
        effects = expected_effect('execute_command', self.action, self.structure)
        result = {'data': {'duration': 4}}
        self.assertEqual(verify_effect(effects, self.structure, result)['status'], 'pending')
        self.assertIsNone(verify_effect(effects, self.structure, result)['verified'])
        self.assertFalse(verify_effect(effects, self.structure, {'data': {'suppressed': True}})['verified'])
        self.assertFalse(verify_effect(effects, self.structure, {'data': {}})['verified'])

    def test_blocked_proposal_does_not_reach_actual_executor(self):
        calls = []
        def dispatch(tool, arguments):
            calls.append(tool)
            return {'status': {'code': 200}, 'data': self.structure, 'error': None}
        executor = GuardedExecutor(dispatch=dispatch)
        response = executor.execute('execute_command', {**self.action, 'args': {'level': 50}})
        self.assertEqual(response['error']['layer'], 'schema')
        self.assertEqual(calls, [])
        self.assertEqual(executor.actual, [])

    def test_two_repairs_and_query_budget_are_enforced(self):
        calls = []
        def dispatch(tool, arguments):
            calls.append(tool)
            return {'status': {'code': 422}, 'data': {}, 'error': {'type': 'test-error'}}
        executor = GuardedExecutor(verify=True, dispatch=dispatch, query_limit=2)
        for _ in range(3):
            executor.execute('get_rooms', {})
        response = executor.execute('get_rooms', {})
        self.assertEqual(response['error']['layer'], 'recovery')
        self.assertEqual(len(calls), 3)
        executor.call('get_rooms', {}, extra=True)
        executor.call('get_rooms', {}, extra=True)
        self.assertIsNone(executor.call('get_rooms', {}, extra=True))
        self.assertEqual(executor.extra_queries, 2)

    def test_ledger_keeps_query_changes_errors_workflows_and_staleness(self):
        def obs(turn, tool, data, arguments=None):
            return {'turn': turn, 'tool': tool, 'arguments': arguments or {}, 'extra_query': False,
                    'response': {'status': {'code': 200}, 'data': data, 'error': None}}
        observations = [obs(1, 'get_attribute', {'value': 30}, {'device_id': 'a'}),
            obs(2, 'get_attribute', {'value': 40}, {'device_id': 'a'}),
            obs(3, 'schedule_workflow', {'workflow_id': 'w'}),
            obs(4, 'get_current_time', {'now': '2030-01-01 12:30:00'})]
        ledger = build_ledger(observations, [{'turn': 5, 'tool': 'execute_command', 'arguments': {},
            'blocked': True, 'response': {'error': 'readonly'}}])
        fact = next(v for v in ledger['facts'].values() if v['source']['tool'] == 'get_attribute')
        self.assertTrue(fact['stale'])
        self.assertEqual(fact['response']['data']['value'], 40)
        self.assertEqual(fact['previous_versions'][0]['previous_leaf_values']['/data/value']['value'], 30)
        self.assertTrue(ledger['action_receipts'][0]['workflow_registration_is_not_future_success'])
        self.assertEqual(len(ledger['errors']), 1)
        self.assertEqual(flatten({'a/b': {'~x': 2}}), {'/a~1b/~0x': 2})

    def test_context_retains_task_recent_pairs_and_repeated_query_provenance(self):
        document = 'public device manual ' * 200
        response = {'status': {'code': 200}, 'data': {'manual': document}, 'error': None}
        observations = [{'turn': i, 'tool': 'get_cluster_doc', 'arguments': {'query': 'manual', 'top_k': 1},
            'response': response, 'extra_query': False} for i in range(1, 10)]
        class Inner:
            def generate(self, messages, response_format=None):
                self.messages, self.schema = messages, response_format
                return 'reply'
        inner = Inner()
        executor = SimpleNamespace(observations=observations, audit=[], structured_audit=[
            {'turn': 2, 'validation_error': 'missing command args'}], context_audit=[])
        provider = LedgerProvider(inner, executor)
        task = ChatMessage(role='user', content='This is your actual task. QUERY_MARKER')
        history = [message for _ in observations for message in (
            ChatMessage(role='assistant', content='action'),
            ChatMessage(role='user', content='observation: ' + json.dumps(response)))]
        messages = [ChatMessage(role='system', content='full tool registry'), task, *history]
        provider.generate(messages, response_format={'original_schema': True})
        self.assertTrue(executor.context_audit[-1]['used'])
        self.assertIn(task, inner.messages)
        self.assertEqual(inner.messages[-4:], history[-4:])
        self.assertEqual(inner.schema, {'original_schema': True})
        ledger = build_ledger(observations, [], executor.structured_audit)
        fact = next(iter(ledger['facts'].values()))
        self.assertEqual(len(fact['observed_at']), 9)
        self.assertEqual(fact['response']['data']['manual'], document)
        self.assertEqual(ledger['errors'][0]['output_schema_error'], 'missing command args')
        provider.generate(messages[:2])
        self.assertFalse(executor.context_audit[-1]['used'])
        self.assertEqual(inner.messages, messages[:2])

    def test_time_or_workflow_query_invalidates_earlier_state_without_agent_write(self):
        for tool, data in [('get_current_time', {'now': '2030-01-01 12:31:00'}),
                           ('get_workflow_status', {'status': 'completed'})]:
            observations = [{'turn': 1, 'tool': 'get_attribute', 'arguments': {'device_id': 'x'},
                'extra_query': False, 'response': {'status': {'code': 200}, 'error': None, 'data': {'value': 1}}},
                {'turn': 2, 'tool': tool, 'arguments': {}, 'extra_query': False,
                 'response': {'status': {'code': 200}, 'error': None, 'data': data}}]
            ledger = build_ledger(observations, [])
            old = next(v for v in ledger['facts'].values() if v['source']['tool'] == 'get_attribute')
            self.assertTrue(old['stale'])
            self.assertEqual(old['response']['data']['value'], 1)


if __name__ == '__main__':
    unittest.main()
