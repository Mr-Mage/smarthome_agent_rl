import copy
import unittest

from smarthome_agent_rl.action_state import ActionLedger


def reply(data=None, code=200, error=None):
    return {'status': {'code': code}, 'data': data or {}, 'error': error}


class ActionStateTests(unittest.TestCase):
    def setUp(self):
        self.ledger = ActionLedger()

    def call(self, tool, args, response, index):
        aid = self.ledger.propose(tool, args, turn=index, observation_version=index - 1)
        self.ledger.dispatched(aid)
        self.ledger.observed(aid, response, index)
        return aid

    def test_full_parameters_and_roles_identify_attempts_without_mutating_inputs(self):
        args = {'args': {'Level': 20}, 'device_id': 'light'}
        original = copy.deepcopy(args)
        a = self.ledger.propose('execute_command', args, turn=1, observation_version=0)
        b = self.ledger.propose('execute_command', dict(reversed(list(args.items()))),
                                turn=2, observation_version=0)
        c = self.ledger.propose('execute_command', {**args, 'args': {'Level': 21}},
                                turn=3, observation_version=0)
        d = self.ledger.propose('execute_command', args, turn=3, observation_version=0,
                                extra_query=True, parent_action_id=c)
        self.assertEqual([r.attempt for r in self.ledger.records.values()], [1, 2, 1, 1])
        self.assertEqual(self.ledger.records[a].request_sha256, self.ledger.records[b].request_sha256)
        self.assertNotEqual(self.ledger.records[a].request_sha256, self.ledger.records[c].request_sha256)
        args['args']['Level'] = 999
        self.assertEqual(self.ledger.records[a].arguments, original)
        self.assertEqual(self.ledger.records[d].parent_action_id, c)
        self.assertIsNone(self.ledger.records[a].goal_id)

    def test_guard_rejection_is_never_a_dispatch(self):
        aid = self.ledger.propose('execute_command', {}, turn=1, observation_version=0)
        self.ledger.rejected(aid, reply(code=422, error={'type': 'harness_guard'}))
        self.assertEqual(self.ledger.records[aid].state, 'rejected')
        self.assertNotIn('dispatched', [t['to'] for t in self.ledger.records[aid].transitions])
        with self.assertRaises(ValueError):
            self.ledger.dispatched(aid)

    def test_mutation_acknowledgement_is_not_completion(self):
        for tool in ('execute_command', 'write_attribute', 'cancel_workflow', 'add_device'):
            aid = self.call(tool, {}, reply(), len(self.ledger.records) + 1)
            self.assertEqual(self.ledger.records[aid].state, 'acknowledged')

    def test_registration_requires_id_and_later_public_status_is_workflow_scoped(self):
        missing = self.call('schedule_workflow', {}, reply(), 1)
        self.assertEqual(self.ledger.records[missing].state, 'acknowledged')
        aid = self.call('schedule_workflow', {}, reply({'workflow_id': 'wf'}), 2)
        self.assertEqual(self.ledger.records[aid].state, 'registered')
        self.call('get_workflow_status', {'workflow_id': 'other'}, reply({'status': 'completed'}), 3)
        self.call('get_workflow_status', {'workflow_id': 'wf'}, reply({'status': 'completed'}, code=500), 4)
        self.call('get_workflow_status', {'workflow_id': 'wf'}, reply({'workflow_id': 'other', 'status': 'completed'}), 5)
        self.assertEqual(self.ledger.records[aid].state, 'registered')
        self.call('get_workflow_status', {'workflow_id': 'wf'}, reply({'status': 'running'}), 6)
        self.call('get_workflow_status', {'workflow_id': 'wf'}, reply({'status': 'completed'}), 7)
        self.assertEqual(self.ledger.records[aid].state, 'completed')
        self.assertEqual(self.ledger.records[aid].evidence[-1]['scope'], 'workflow_only')
        self.call('get_workflow_status', {'workflow_id': 'wf'}, reply({'status': 'pending'}), 8)
        self.call('get_workflow_status', {'workflow_id': 'wf'}, reply({'status': 'running'}), 9)
        self.assertEqual(self.ledger.records[aid].state, 'completed')

    def test_cancellation_updates_registration_only_from_explicit_status(self):
        aid = self.call('schedule_workflow', {}, reply({'workflow_id': 'wf'}), 1)
        self.call('cancel_workflow', {'workflow_id': 'wf'}, reply({'workflow_id': 'wf'}), 2)
        self.assertEqual(self.ledger.records[aid].state, 'registered')
        self.call('cancel_workflow', {'workflow_id': 'wf'}, reply({'status': 'cancelled'}), 3)
        self.assertEqual(self.ledger.records[aid].state, 'cancelled')

    def test_timeouts_servers_and_wrapped_transport_errors_keep_unknown(self):
        for response in (None, reply(code=408), reply(code=503),
                         reply(code=400, error={'type': 'HTTP_ERROR'})):
            aid = self.call('execute_command', {}, response, len(self.ledger.records) + 1)
            self.assertEqual(self.ledger.records[aid].state, 'unknown')
        aid = self.call('execute_command', {}, reply(code=400, error={'type': 'INVALID_STATE'}), 8)
        self.assertEqual(self.ledger.records[aid].state, 'failed')
        aid = self.call('execute_command', {}, reply(code=400, error={
            'type': 'HTTP_ERROR', 'detail': '400: Bad Request from simulator'}), 9)
        self.assertEqual(self.ledger.records[aid].state, 'failed')
        aid = self.call('execute_command', {}, reply(code=400, error={
            'type': 'HTTP_ERROR', 'detail': '503: Service Unavailable'}), 10)
        self.assertEqual(self.ledger.records[aid].state, 'unknown')

    def test_exception_before_dispatch_is_aborted_after_dispatch_unknown(self):
        a = self.ledger.propose('execute_command', {}, turn=1, observation_version=0)
        self.ledger.interrupted(a, RuntimeError('prequery failed'))
        b = self.ledger.propose('execute_command', {}, turn=2, observation_version=0)
        self.ledger.dispatched(b)
        self.ledger.interrupted(b, TimeoutError())
        self.assertEqual(self.ledger.records[a].state, 'aborted')
        self.assertEqual(self.ledger.records[b].state, 'unknown')
        snapshot = self.ledger.snapshot()
        snapshot['actions'][0]['arguments']['changed'] = True
        self.assertEqual(self.ledger.records[a].arguments, {})


if __name__ == '__main__':
    unittest.main()
