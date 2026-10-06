import copy
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.execution.harness import RuntimeExecutor
from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter
from smarthome_agent_rl.guard import ToolGuard


class RuntimeHarnessTests(unittest.TestCase):
    def setUp(self):
        self.state = {'device_id': 'lamp', 'device_type': 'lamp', 'endpoints': {'1': {'clusters': {
            'OnOff': {'commands': ['On', 'Off'], 'attributes': {'OnOff': {'value': False}}}}}}}
        self.action = {'device_id': 'lamp', 'endpoint_id': 1, 'cluster_id': 'OnOff', 'command_id': 'On', 'args': {}}
        self.calls = []
        self.timeout = self.ack_only = self.bad_identity = self.pending = False
        self.signatures = {('OnOff', 'On'): {'properties': {}, 'required': []},
                           ('OnOff', 'Off'): {'properties': {}, 'required': []}}
        self.power = {'lamp': {'commands': [], 'attributes': {}}}
        schemas = {name: {'properties': {}, 'additionalProperties': True}
                   for name in ('get_rooms', 'get_device_structure', 'execute_command', 'schedule_workflow')}
        self.guard = ToolGuard(schemas, self.signatures, self.power)

    def dispatch(self, tool, arguments):
        self.calls.append((tool, copy.deepcopy(arguments)))
        if tool == 'execute_command':
            if not self.ack_only:
                self.state['endpoints']['1']['clusters']['OnOff']['attributes']['OnOff']['value'] = True
            if self.timeout:
                raise TimeoutError('response lost after device commit')
        after = copy.deepcopy(self.state)
        if self.bad_identity and len(self.calls) > 1:
            after['device_id'] = 'other-lamp'
        return {'status': {'code': 200}, 'data': after if tool.startswith('get_') else {}}

    def executor(self, budget=40):
        return RuntimeExecutor(guard=self.guard, adapter=SimuHomeContractAdapter(self.signatures, self.power),
            dispatch=self.dispatch, invocation_factory=SimpleNamespace, query_limit=budget)

    def test_timeout_after_real_change_has_one_mutation_and_verified_evidence(self):
        self.timeout = True
        executor = self.executor()
        result = executor.execute('execute_command', self.action)
        self.assertEqual(result['harness_verification']['status'], 'VERIFIED_SUCCESS')
        self.assertEqual(result['status']['state'], 'unknown')
        self.assertEqual([t for t, _ in self.calls], ['get_device_structure', 'execute_command', 'get_device_structure'])
        self.assertEqual(len(executor.actual), 3)
        self.assertEqual(executor.extra_queries, 2)
        self.assertEqual(executor.trace.invocations[1].error['type'], 'TimeoutError')

    def test_api_success_without_state_change_has_failure_evidence(self):
        self.ack_only = True
        result = self.executor().execute('execute_command', self.action)
        self.assertEqual(result['status']['code'], 200)
        self.assertEqual(result['harness_verification']['status'], 'VERIFIED_FAILURE')

    def test_unknown_precondition_is_blocked_without_mutation(self):
        self.power.clear()
        executor = self.executor()
        result = executor.execute('execute_command', self.action)
        self.assertEqual(result['error']['reason_code'], 'CONTRACT_UNCOVERED')
        self.assertEqual([t for t, _ in self.calls], ['get_device_structure'])
        self.assertFalse(executor.audit[0]['reached_executor'])

    def test_missing_budget_does_not_create_a_fake_tool_trace(self):
        executor = self.executor(budget=1)
        result = executor.execute('execute_command', self.action)
        self.assertEqual(result['harness_verification']['reason'], 'READBACK_BUDGET_EXHAUSTED')
        self.assertEqual(len(executor.trace.invocations), 2)
        second = executor.execute('execute_command', self.action)
        self.assertEqual(second['error']['reason_code'], 'PREQUERY_BUDGET_EXHAUSTED')
        self.assertEqual(len(executor.trace.invocations), 2)

    def test_wrong_readback_identity_cannot_prove_success(self):
        self.bad_identity = True
        result = self.executor().execute('execute_command', self.action)
        self.assertEqual(result['harness_verification']['reason'], 'READBACK_IDENTITY_MISMATCH')
        self.assertEqual(result['harness_verification']['status'], 'UNVERIFIED')

    def test_query_has_only_trace_and_audit_has_no_query_lifecycle(self):
        outputs = []
        executor = self.executor()
        executor.audit_fn = outputs.append
        executor.execute('get_rooms', {})
        self.assertEqual(len(outputs[-1]['tool_trace']), 1)
        self.assertEqual(outputs[-1]['action_lifecycle'], {})
        self.assertNotIn('state', outputs[-1]['tool_trace'][0])

    def test_future_workflow_does_not_assert_todays_power_or_completion(self):
        executor = self.executor()
        result = executor.execute('schedule_workflow', {'start_time': '2030-01-01 12:00:00',
            'steps': [{'tool': 'execute_command', 'args': self.action}]})
        self.assertEqual(result['harness_verification']['status'], 'REGISTRATION_ONLY')
        self.assertFalse(result['harness_verification']['future_success_verified'])
        self.assertEqual([t for t, _ in self.calls], ['get_device_structure', 'schedule_workflow'])

    def test_contract_only_arm_does_not_read_back_or_claim_success(self):
        executor = self.executor()
        executor.verify_mutations = False
        result = executor.execute('execute_command', self.action)
        self.assertEqual(result['harness_verification']['reason'], 'VERIFICATION_DISABLED')
        self.assertIsNone(result['harness_verification']['verified'])
        self.assertEqual([t for t, _ in self.calls], ['get_device_structure', 'execute_command'])


if __name__ == '__main__':
    unittest.main()
