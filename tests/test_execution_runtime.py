import copy
import unittest

from smarthome_agent_rl.device_contract import DeviceContract
from smarthome_agent_rl.execution.mutation import MutationExecutor, VerificationStatus
from smarthome_agent_rl.execution.trace import ToolTrace
from smarthome_agent_rl.execution.workflow import Workflow, WorkflowStatus


def contract():
    return DeviceContract.from_dict('temperature-device', {
        'capabilities': ['temperature'], 'functions': {'SetTemperature': {
            'capability': 'temperature',
            'args': {'temperature': {'type': 'float', 'min': 16, 'max': 30}},
            'preconditions': [{'path': 'power', 'op': 'eq', 'value': True}],
            'postconditions': [{'path': 'target', 'value': '$args.temperature', 'tolerance': .1}],
            'verification': {'tool': 'read_target', 'args': {'device': '$action.device_id'}},
        }}})


class ExecutionRuntimeTests(unittest.TestCase):
    def run_action(self, receipt, target=24, read_error=False):
        calls = []
        def dispatch(tool, args):
            calls.append((tool, copy.deepcopy(args)))
            if tool == 'set_target':
                if isinstance(receipt, Exception):
                    raise receipt
                return receipt
            return {'status': {'code': 503 if read_error else 200}, 'data': {'target': target}}
        trace = ToolTrace(dispatch)
        args = {'device_id': 'bedroom_ac', 'args': {'temperature': 24}}
        snapshot = copy.deepcopy(args)
        result = MutationExecutor(trace).execute('set_target', args, contract=contract(),
            state={'power': True}, function_name='SetTemperature', task_id='t1')
        self.assertEqual(args, snapshot)
        self.assertEqual(calls[-1][1], {'device': 'bedroom_ac'})
        return result, trace, calls

    def test_timeout_then_matching_readback_proves_success_without_replay(self):
        result, trace, calls = self.run_action(TimeoutError('uncertain dispatch'))
        self.assertEqual(result.status, VerificationStatus.VERIFIED_SUCCESS)
        self.assertEqual([t for t, _ in calls], ['set_target', 'read_target'])
        self.assertEqual(trace.invocations[0].error['type'], 'TimeoutError')
        self.assertEqual(trace.invocations[1].task_id, 't1')

    def test_acknowledgement_with_wrong_world_state_is_failure(self):
        result, _, _ = self.run_action({'status': {'code': 200}}, target=27)
        self.assertEqual(result.status, VerificationStatus.VERIFIED_FAILURE)

    def test_missing_readback_stays_unverified(self):
        result, _, _ = self.run_action({'status': {'code': 200}}, read_error=True)
        self.assertEqual(result.status, VerificationStatus.UNVERIFIED)

    def test_unknown_precondition_and_invalid_range_never_dispatch(self):
        trace = ToolTrace(lambda *args: self.fail('Rejected action must not dispatch'))
        for state, temperature in [({}, 24), ({'power': True}, 99), ({'power': True}, float('nan'))]:
            result = MutationExecutor(trace).execute('set_target', {'args': {'temperature': temperature}},
                contract=contract(), state=state, function_name='SetTemperature')
            self.assertEqual(result.status, VerificationStatus.UNVERIFIED)
            self.assertIsNone(result.mutation_invocation_id)

    def test_query_has_a_trace_without_mutation_lifecycle(self):
        trace = ToolTrace(lambda *args: {'status': {'code': 200}, 'data': {'target': 24}})
        row = trace.call('read_target', {'device': 'x'})
        self.assertIn('response', row.as_dict())
        self.assertNotIn('state', row.as_dict())

    def test_workflow_receipt_cannot_mark_completion(self):
        workflow = Workflow('t1', 1, 'device_native')
        workflow.transition(WorkflowStatus.SCHEDULED, {'registration_receipt': '200'})
        with self.assertRaises(ValueError):
            workflow.transition(WorkflowStatus.VERIFIED_SUCCESS, {'http': 200})
        workflow.transition(WorkflowStatus.ACTIVE, {'target_time_reached': True})
        result, _, _ = self.run_action({'status': {'code': 200}})
        from smarthome_agent_rl.execution.mutation import PostconditionResult
        workflow.verify(PostconditionResult(result.status, tuple(result.verification['evidence'])))
        self.assertEqual(workflow.status, WorkflowStatus.VERIFIED_SUCCESS)


if __name__ == '__main__':
    unittest.main()
