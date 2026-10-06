import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.execution.episode import EpisodeRuntime, virtual_seconds
from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter


class EpisodeRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.state = {'device_id': 'lamp', 'device_type': 'lamp', 'endpoints': {'1': {'clusters': {
            'OnOff': {'commands': ['On', 'Off'], 'attributes': {'OnOff': {'value': False}}}}}}}
        self.now = '2030-01-01 12:00:00'
        self.due = '2030-01-01 12:00:10'
        self.workflow_status = 'pending'
        self.bad_identity = False
        self.timeout = False
        self.calls, self.actual = [], []
        self.receipts = []
        executor = SimpleNamespace(dispatch=self.dispatch, extra_queries=0, query_limit=40,
                                   save_audit=lambda: None)
        def call(tool, args, *, extra=False):
            if extra:
                if executor.extra_queries >= executor.query_limit:
                    return None
                executor.extra_queries += 1
            value = executor.dispatch(tool, args)
            self.actual.append((tool, copy.deepcopy(args), copy.deepcopy(value)))
            return value
        executor.call = call
        self.executor = executor
        adapter = SimuHomeContractAdapter({('OnOff', name): {'properties': {}, 'required': []}
                                          for name in ('On', 'Off')}, {'lamp': {}})
        self.runtime = EpisodeRuntime(executor, adapter, self.root / 'runtime.sqlite3',
                                      save=lambda *args: self.receipts.append(args))
        self.runtime.start('Turn on lamp at 12:00:10.', current_time=self.now)
        self.arguments = {'start_time': self.due, 'steps': [{'tool': 'execute_command', 'args': {
            'device_id': 'lamp', 'endpoint_id': 1, 'cluster_id': 'OnOff', 'command_id': 'On', 'args': {}}}]}

    def dispatch(self, tool, args):
        self.calls.append((tool, copy.deepcopy(args)))
        if tool == 'schedule_workflow':
            # Durable intent exists before remote side effects.
            job = self.runtime.store.list('job')[0]
            self.assertEqual(job['status'], 'REGISTERING')
            self.assertEqual(self.runtime.store.list('workflow')[0]['status'], 'CREATED')
            if self.timeout:
                raise TimeoutError('after native registration')
            data = {'workflow_id': 'native-1'}
        elif tool == 'get_device_structure':
            data = copy.deepcopy(self.state)
            if self.bad_identity:
                data['device_id'] = 'other-device'
        elif tool == 'get_current_time':
            data = {'now': self.now}
        elif tool == 'cancel_workflow':
            data = {'workflow_id': args['workflow_id'], 'status': 'cancelled'}
        elif tool == 'get_workflow_status':
            data = {'workflow_id': 'native-1', 'status': self.workflow_status}
        else:
            data = {}
        return {'status': {'code': 200}, 'data': data}

    def register(self):
        return self.executor.call('schedule_workflow', self.arguments)

    def complete(self, *, state=True, now=None):
        self.now = now or self.due
        self.workflow_status = 'completed'
        self.state['endpoints']['1']['clusters']['OnOff']['attributes']['OnOff']['value'] = state
        return self.runtime.supervise(phase='test-public-clock')

    def test_single_registration_durable_before_dispatch_and_receipt_preserved(self):
        response = self.register()
        self.assertEqual(response, {'status': {'code': 200}, 'data': {'workflow_id': 'native-1'}})
        self.assertEqual(sum(tool == 'schedule_workflow' for tool, _ in self.calls), 1)
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'SCHEDULED')
        self.runtime.finish()
        self.assertEqual(self.runtime.store.list('task')[0]['status'], 'WAITING')
        self.assertEqual(self.runtime.store.list('task')[0]['expected_postconditions'], [])

    def test_real_due_reads_verify_action_but_never_claim_user_goal_complete(self):
        self.register()
        self.assertEqual(self.runtime.supervise(phase='early'), [])
        outcomes = self.complete()
        self.assertEqual(outcomes[0]['status'], 'VERIFIED_SUCCESS')
        self.runtime.finish()
        self.assertEqual(self.runtime.store.list('task')[0]['status'], 'WAITING')
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'DONE')
        self.assertEqual(self.runtime.supervisor_queries, self.executor.extra_queries)
        self.assertEqual(len(self.actual), len(self.calls))
        self.assertFalse(any(tool == 'execute_command' for tool, _ in self.calls))
        self.assertEqual(outcomes[0]['observed_at'], virtual_seconds(self.due))

    def test_completed_ack_with_mismatching_state_is_failed_intention(self):
        self.register()
        self.assertEqual(self.complete(state=False)[0]['status'], 'VERIFIED_FAILURE')
        self.runtime.finish()
        self.assertEqual(self.runtime.store.list('task')[0]['status'], 'WAITING')

    def test_late_clock_does_not_read_device_or_replay_action(self):
        self.register()
        before = len(self.calls)
        self.assertEqual(self.complete(now='2030-01-01 12:00:13')[0]['status'], 'UNVERIFIED')
        self.assertEqual([t for t, _ in self.calls[before:]], ['get_current_time'])
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'UNKNOWN')

    def test_budget_exhaustion_leaves_pending_and_has_no_fake_readback(self):
        self.register()
        self.executor.query_limit = self.executor.extra_queries
        before = len(self.calls)
        self.assertEqual(self.complete(), [])
        self.assertEqual(len(self.calls), before)
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'SCHEDULED')

    def test_many_native_callbacks_after_budget_exhaustion_do_not_rewrite_evidence(self):
        self.register()
        self.executor.query_limit = self.executor.extra_queries
        before, calls = len(self.receipts), len(self.calls)
        for _ in range(100):
            self.runtime.supervise(phase='native_virtual_time_advanced')
        self.assertEqual(len(self.receipts), before + 1)
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(self.runtime.skipped_supervisions, 100)
        self.runtime.flush()
        self.assertEqual(self.receipts[-1][1]['skipped_supervisions_query_budget'], 100)
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'SCHEDULED')

    def test_registration_timeout_retains_original_exception_and_never_replays(self):
        self.timeout = True
        with self.assertRaises(TimeoutError):
            self.register()
        self.complete()
        self.complete()
        self.assertEqual(sum(tool == 'schedule_workflow' for tool, _ in self.calls), 1)
        self.assertTrue(self.runtime.store.list('job')[0]['registration_uncertain'])

    def test_null_data_native_rejection_is_preserved_and_durable(self):
        original = self.runtime.raw_dispatch
        rejection = {'status': {'code': 400}, 'data': None, 'error': {'type': 'HTTP_ERROR'}}
        def rejected(tool, arguments):
            if tool == 'schedule_workflow':
                return copy.deepcopy(rejection)
            return original(tool, arguments)
        self.runtime.raw_dispatch = rejected
        self.assertEqual(self.register(), rejection)
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'SCHEDULED')
        self.assertEqual(self.runtime.store.list('workflow')[0]['evidence'][-1]['kind'], 'registration')
        self.assertEqual(self.complete()[0]['status'], 'UNVERIFIED')
        self.assertEqual(len(self.actual), len(self.runtime.trace.invocations))

    def test_wrong_device_id_cannot_verify_intention(self):
        self.register()
        self.bad_identity = True
        self.assertEqual(self.complete()[0]['status'], 'UNVERIFIED')

    def test_cancelled_receipt_updates_persistent_mapping_without_second_call(self):
        self.register()
        self.executor.call('cancel_workflow', {'workflow_id': 'native-1'})
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'CANCELLED')
        self.assertEqual(self.runtime.store.list('workflow')[0]['status'], 'CANCELLED')
        self.assertEqual(sum(tool == 'cancel_workflow' for tool, _ in self.calls), 1)

    def test_missing_postcondition_and_invalid_clock_never_invent_success(self):
        self.arguments['steps'][0]['args']['command_id'] = 'Unknown'
        self.register()
        self.assertEqual(self.complete()[0]['status'], 'UNVERIFIED')
        with self.assertRaises(ValueError):
            virtual_seconds('invalid')


if __name__ == '__main__':
    unittest.main()
