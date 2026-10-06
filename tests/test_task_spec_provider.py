import copy
import json
from types import SimpleNamespace
import unittest

from src.agents.types import ChatMessage
from smarthome_agent_rl.action_state import ActionLedger
from smarthome_agent_rl.structured import StructuredProvider
from smarthome_agent_rl.task_spec import TaskSpec
from smarthome_agent_rl.harness_agent import HarnessAgent
from src.simulator.domain.clusters.onoff import OnOffCluster
from test_task_spec import body, goal


class TaskSpecProviderTests(unittest.TestCase):
    def make(self, response):
        executor = SimpleNamespace(observations=[], actions=ActionLedger())
        spec = TaskSpec(executor)
        spec.initialize('Start the washer now')
        class Fake:
            def __init__(self):
                self.calls = []
                self.response = response
            def generate(self, messages, response_format=None):
                self.calls.append((copy.deepcopy(messages), copy.deepcopy(response_format)))
                return json.dumps(self.response)
        fake = Fake()
        return StructuredProvider(fake, finish_guard=False, recovery=False, guidance=False,
                                  task_spec=spec), spec, fake

    def test_one_model_call_per_turn_and_metadata_removed_from_upstream_action(self):
        provider, spec, fake = self.make(body([goal()], refs=['g1']))
        messages = [ChatMessage('system', 'tools'), ChatMessage('user', 'This is your actual task. Start the washer now')]
        original = copy.deepcopy(messages)
        result = json.loads(provider.generate(messages))
        self.assertEqual(set(result), {'thought', 'action', 'action_input'})
        fake.response = body(refs=['g1'])
        provider.generate(messages)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([r['turn'] for r in spec.turns], [1, 2])
        self.assertEqual(messages, original)
        self.assertEqual(fake.calls[1][1]['json_schema']['schema']['properties']['task_spec'], {'type': 'null'})
        self.assertEqual(provider.audit[-1]['goal_refs'], ['g1'])

    def test_invalid_tool_or_metadata_does_not_publish_half_valid_spec(self):
        response = body([goal()])
        response['call']['tool'] = 'invented'
        provider, spec, fake = self.make(response)
        messages = [ChatMessage('system', 'tools'), ChatMessage('user', 'This is your actual task.')]
        self.assertEqual(provider.generate(messages), '{}')
        self.assertEqual(spec.goals, [])
        fake.response = body([goal()], refs=['unknown'])
        self.assertEqual(provider.generate(messages), '{}')
        self.assertEqual(spec.goals, [])
        self.assertEqual(len(spec.errors), 2)
        self.assertEqual(len(fake.calls), 2)

    def test_executor_links_main_and_auxiliary_without_certifying_running_from_on(self):
        agent = HarnessAgent(None, variant='GTS', max_steps=20)
        executor, spec = agent.executor, agent.executor.task_spec
        spec.initialize('Start the washer now')
        _, draft = spec.prepare(body([goal()], refs=['g1']))
        spec.commit(draft, {'action': 'execute_command'}, 1)
        executor.record_structured([{'turn': 1, 'goal_refs': ['g1']}])
        structure = {'device_id': 'washer', 'endpoints': {'1': {'clusters': {
            'OnOff': OnOffCluster().get_structure()}}}}
        calls = []
        def dispatch(tool, args):
            calls.append(tool)
            return {'status': {'code': 200}, 'error': None,
                    'data': structure if tool == 'get_device_structure' else {}}
        executor.dispatch = dispatch
        executor.execute('execute_command', {'device_id': 'washer', 'endpoint_id': 1,
            'cluster_id': 'OnOff', 'command_id': 'On', 'args': {}})
        records = executor.actions.snapshot()['actions']
        self.assertEqual(calls, ['get_device_structure', 'execute_command'])
        self.assertTrue(all(r['goal_ids'] == ['g1'] and r['goal_id'] == 'g1' for r in records))
        result = spec.snapshot()['goals'][0]
        self.assertEqual(result['action_evidence'][0]['state'], 'acknowledged')
        self.assertEqual(result['satisfaction'], 'unverified')


if __name__ == '__main__':
    unittest.main()
