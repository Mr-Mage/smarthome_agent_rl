"""Native engineering fixtures, never generated benchmark tasks."""
import copy
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest

try:
    from smarthome_agent_rl.harness_agent import GuardedExecutor, HarnessAgent
except ModuleNotFoundError as exc:
    if exc.name != 'src':
        raise
    GuardedExecutor = HarnessAgent = None

from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_verifier import Decision, VerificationResult


@unittest.skipIf(GuardedExecutor is None, 'requires existing SimuHome server environment')
class NativePublicSemanticContextTests(unittest.TestCase):
    def setUp(self):
        from src.simulator.domain.clusters.onoff import OnOffCluster
        self.structure = {'device_id': 'lamp', 'endpoints': {'1': {'clusters': {
            'OnOff': OnOffCluster().get_structure()}}}}
        self.action = {'device_id': 'lamp', 'endpoint_id': 1, 'cluster_id': 'OnOff', 'command_id': 'On', 'args': {}}
        self.workflow = {'start_time': '2030-01-01 12:01:00', 'steps': [
            {'tool': 'execute_command', 'args': self.action},
            {'tool': 'execute_command', 'args': {**self.action, 'device_id': 'other'}}]}
        self.calls, self.contexts = [], []
        self.verifier = SimpleNamespace(verify=self.verify)

    def verify(self, context):
        self.contexts.append(context)
        self.assertFalse(self.calls and self.calls[-1][0] == context.proposed_action['tool'])
        no = Decision('NO', .99, {'scope': 'fixture only'})
        return VerificationResult(no, no, no, no, 'FIXTURE_DENY')

    def dispatch(self, tool, args):
        self.calls.append((tool, copy.deepcopy(args)))
        data = self.structure if tool == 'get_device_structure' else {'workflow_id': 'fixture-workflow'}
        return {'status': {'code': 200}, 'data': copy.deepcopy(data), 'error': None}

    def test_version0_preserves_empty_workflow_state_and_version1_exposes_existing_guard_receipt(self):
        calls = []
        for version in (0, 1):
            self.calls.clear()
            executor = GuardedExecutor(dispatch=self.dispatch, reflection_verifier=self.verifier,
                                       semantic_context_version=version)
            executor.user_goal = 'Switch lamps at 12:01.'
            executor.initial_public_time = '2030-01-01 12:00:00'
            executor.execute('schedule_workflow', self.workflow)
            calls.append(copy.deepcopy(self.calls))
            context = self.contexts[-1]
            if version == 0:
                self.assertEqual(context.environment_state, {})
                self.assertNotIn('semantic_context', executor.audit[-1])
            else:
                state = context.environment_state
                self.assertEqual(state['devices']['lamp']['structure'], self.structure)
                self.assertIsNone(state['devices']['other']['structure'])
                self.assertEqual(state['observation_count'], 1)
                self.assertEqual(executor.audit[-1]['semantic_context'], asdict(context))
                self.assertEqual(executor.audit[-1]['semantic_context_sha256'], digest(asdict(context)))
            self.assertFalse(executor.audit[-1]['blocked'])  # report-only DENY
        self.assertEqual(calls[0], calls[1])
        self.assertEqual([name for name, _ in calls[1]], ['get_device_structure', 'schedule_workflow'])

    def test_version1_does_not_override_existing_explicit_blocking_policy(self):
        executor = GuardedExecutor(dispatch=self.dispatch, semantic_verifier=self.verifier,
                                   semantic_context_version=1, semantic_blocking=True)
        response = executor.execute('execute_command', self.action)
        self.assertTrue(executor.audit[-1]['blocked'])
        self.assertEqual(response['error']['layer'], 'semantic')
        self.assertEqual([tool for tool, _ in self.calls], ['get_device_structure'])

    def test_native_react_supplies_original_public_inputs_and_no_extra_tool_calls(self):
        outputs = iter([{'thought': 'Schedule.', 'call': {'tool': 'schedule_workflow', 'arguments': self.workflow}},
                        {'thought': 'Only registered.', 'call': {'tool': 'finish', 'arguments': {'answer': 'Registered.'}}}])
        llm = SimpleNamespace(generate=lambda *a, **k: json.dumps(next(outputs)))
        agent = HarnessAgent(llm, variant='G', max_steps=3,
            policy={'verify': False, 'verification_version': 1, 'context_version': 0,
                    'reflection_verifier': self.verifier, 'semantic_context_version': 1})
        agent.executor.dispatch = self.dispatch
        result = agent.run('Switch lamps at 12:01.', user_location='living', current_time='2030-01-01 12:00:00')
        context = self.contexts[-1]
        self.assertEqual(context.user_goal, 'Switch lamps at 12:01.')
        self.assertEqual(context.environment_state['user_location'], 'living')
        self.assertEqual(context.environment_state['initial_public_time'], '2030-01-01 12:00:00')
        self.assertEqual(result.final_answer, 'Registered.')
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual([name for name, _ in self.calls], ['get_device_structure', 'schedule_workflow'])

    def test_invalid_versions_are_rejected(self):
        for version in (True, -1, 2, '1'):
            with self.assertRaises(ValueError):
                GuardedExecutor(semantic_context_version=version)


if __name__ == '__main__':
    unittest.main()
