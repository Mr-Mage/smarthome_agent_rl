import copy
import unittest

from src.simulator.domain.clusters.onoff import OnOffCluster
from smarthome_agent_rl.harness_agent import GuardedExecutor


class ActionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.action = {'device_id': 'light', 'endpoint_id': 1, 'cluster_id': 'OnOff',
                       'command_id': 'On', 'args': {}}
        self.structure = {'device_id': 'light', 'endpoints': {'1': {'clusters': {
            'OnOff': OnOffCluster().get_structure()}}}}

    def test_prequery_is_linked_but_not_confused_with_main_action(self):
        calls, snapshots = [], []
        def dispatch(tool, args):
            calls.append((tool, copy.deepcopy(args)))
            return {'status': {'code': 200}, 'error': None,
                    'data': self.structure if tool == 'get_device_structure' else {}}
        executor = GuardedExecutor(dispatch=dispatch, audit_fn=lambda x: snapshots.append(copy.deepcopy(x)))
        response = executor.execute('execute_command', self.action)
        records = snapshots[-1]['action_lifecycle']['actions']
        self.assertEqual(len(calls), 2)
        self.assertEqual([r['state'] for r in records], ['acknowledged', 'completed'])
        self.assertEqual(records[1]['parent_action_id'], records[0]['action_id'])
        self.assertTrue(records[1]['extra_query'])
        self.assertEqual(executor.audit[0]['action_id'], records[0]['action_id'])
        self.assertEqual(executor.actual[-1].observation, response)
        self.assertEqual(executor.observations[-1]['response'], response)

    def test_schema_and_capability_rejections_do_not_dispatch_main_action(self):
        calls = []
        def dispatch(tool, args):
            calls.append(tool)
            return {'status': {'code': 200}, 'error': None, 'data': self.structure}
        executor = GuardedExecutor(dispatch=dispatch)
        executor.execute('execute_command', {**self.action, 'extra': True})
        self.assertEqual(calls, [])
        executor.execute('execute_command', {**self.action, 'command_id': 'Missing'})
        self.assertEqual(calls, ['get_device_structure'])
        records = executor.actions.snapshot()['actions']
        self.assertEqual([r['state'] for r in records], ['rejected', 'rejected', 'completed'])

    def test_budget_denial_creates_no_phantom_auxiliary_call(self):
        executor = GuardedExecutor(query_limit=0, dispatch=lambda *args: {
            'status': {'code': 200}, 'error': None, 'data': {}})
        executor.execute('execute_command', self.action)
        self.assertEqual(len(executor.actual), 1)
        self.assertEqual(len(executor.actions.records), 1)
        self.assertEqual(executor.extra_queries, 0)

    def test_dispatch_exception_is_preserved_and_saved_as_unknown(self):
        snapshots = []
        def dispatch(tool, args):
            raise TimeoutError('fake transport timeout')
        executor = GuardedExecutor(dispatch=dispatch, query_limit=0,
                                   audit_fn=lambda x: snapshots.append(copy.deepcopy(x)))
        with self.assertRaisesRegex(TimeoutError, 'fake transport timeout'):
            executor.execute('execute_command', self.action)
        self.assertEqual(snapshots[-1]['action_lifecycle']['actions'][0]['state'], 'unknown')
        self.assertEqual(executor.actual, [])  # Upstream-compatible receipt accounting.
        self.assertIsNone(executor.active_action_id)

    def test_auxiliary_exception_aborts_main_without_claiming_it_executed(self):
        def dispatch(*args):
            raise TimeoutError('prequery timeout')
        executor = GuardedExecutor(dispatch=dispatch)
        with self.assertRaises(TimeoutError):
            executor.execute('execute_command', self.action)
        self.assertEqual([r.state for r in executor.actions.records.values()], ['aborted', 'unknown'])


if __name__ == '__main__':
    unittest.main()
