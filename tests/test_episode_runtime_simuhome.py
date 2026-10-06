"""Engineering checks against unmodified native ReAct and Home, not a benchmark."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    from src.simulator.domain.home import Home
    from src.simulator.domain.devices.dimmable_light import DimmableLight
except ModuleNotFoundError as exc:
    if exc.name != 'src':
        raise
    Home = None

from smarthome_agent_rl.execution.episode import EpisodeRuntime
from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter
from smarthome_agent_rl.execution.store import RuntimeStore

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(Home is None, 'requires existing SimuHome server environment')
class NativeEpisodeRuntimeTests(unittest.TestCase):
    def setUp(self):
        from smarthome_agent_rl.guard import command_contracts, public_power_rules
        from smarthome_agent_rl.harness_agent import HarnessAgent
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'episode.sqlite3'
        self.home = Home(tick_interval=1, fast_forward=True, base_time='2030-01-01 12:00:00')
        self.device = DimmableLight('lamp')
        self.device.execute_command(1, 'OnOff', 'Off')
        self.assertTrue(self.home._add_device('living', self.device).success)
        self.calls = []
        self.arguments = {'start_time': '2030-01-01 12:00:10', 'steps': [{'tool': 'execute_command',
            'args': {'device_id': 'lamp', 'endpoint_id': 1, 'cluster_id': 'OnOff', 'command_id': 'On', 'args': {}}}]}
        self.outputs = [
            {'thought': 'Check public capabilities.', 'call': {'tool': 'get_device_structure', 'arguments': {'device_id': 'lamp'}}},
            {'thought': 'Delegate the fixed future action.', 'call': {'tool': 'schedule_workflow', 'arguments': self.arguments}},
            {'thought': 'Registration is not completion.', 'call': {'tool': 'finish', 'arguments': {'answer': 'Scheduled.'}}}]
        self.index = 0
        def generate(messages, response_format=None):
            result = json.dumps(self.outputs[self.index])
            self.index += 1
            return result
        self.agent = HarnessAgent(SimpleNamespace(generate=generate), variant='GTM', max_steps=4,
            policy={'verify': False, 'verification_version': 1, 'context_version': 0})
        self.agent.executor.dispatch = self.dispatch
        signatures, _ = command_contracts(ROOT / 'deps/SimuHome/src/simulator/domain/clusters')
        power, _ = public_power_rules(ROOT / 'deps/SimuHome/src/simulator/domain/devices')
        self.runtime = self.agent.task_runtime = EpisodeRuntime(self.agent.executor,
            SimuHomeContractAdapter(signatures, power), self.path)

    def dispatch(self, tool, args):
        self.calls.append((tool, copy.deepcopy(args)))
        if tool == 'get_current_time':
            return {'status': {'code': 200}, 'data': {'now': self.home.get_virtual_now_str()}}
        if tool == 'get_device_structure':
            result = self.home._get_structure(args['device_id'])
        elif tool == 'schedule_workflow':
            self.assertEqual(self.runtime.store.list('job')[0]['status'], 'REGISTERING')
            result = self.home.schedule_workflow(**args)
        elif tool == 'get_workflow_status':
            result = self.home.get_workflow_status(**args)
        else:
            self.fail(f'Unexpected tool {tool}')
        return {'status': {'code': 200 if result.success else 400},
                'data': json.loads(json.dumps(result.data or {})),
                'error': None if result.success else {'message': result.error_message}}

    def run_agent(self):
        return self.agent.run('Turn on the lamp at 12:00:10.', current_time='2030-01-01 12:00:00')

    def test_native_react_delegates_once_and_real_clock_verifies_after_conversation(self):
        import src.agents.strategies.react_agent as react
        original = react.run_tool
        result = self.run_agent()
        self.assertEqual(result.final_answer, 'Scheduled.')
        self.assertIs(react.run_tool, original)
        self.assertEqual(len(result.tool_calls), len(self.calls))
        self.assertFalse(self.device.get_attribute(1, 'OnOff', 'OnOff'))
        self.assertEqual(self.runtime.store.list('task')[0]['status'], 'WAITING')
        self.assertTrue(self.home._fast_forward_to(11).success)
        outcomes = self.runtime.supervise(phase='native_virtual_time_advanced')
        self.runtime.finish()
        self.assertTrue(self.device.get_attribute(1, 'OnOff', 'OnOff'))
        self.assertEqual(outcomes[0]['status'], 'VERIFIED_SUCCESS')
        reloaded = RuntimeStore(self.path)
        self.assertEqual(reloaded.list('job')[0]['status'], 'DONE')
        self.assertEqual(reloaded.list('task')[0]['status'], 'WAITING')
        self.assertEqual(sum(t == 'schedule_workflow' for t, _ in self.calls), 1)
        self.assertFalse(any(t == 'execute_command' for t, _ in self.calls))
        self.assertTrue(all(row['task_id'] for row in reloaded.list('trace')))
        self.assertEqual(len(self.agent.executor.actual), len(self.calls))

    def test_native_fast_forward_that_misses_window_stays_unknown(self):
        self.run_agent()
        self.assertTrue(self.home._fast_forward_to(14).success)
        outcomes = self.runtime.supervise(phase='native_virtual_time_advanced')
        self.assertEqual(outcomes[0]['status'], 'UNVERIFIED')
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'UNKNOWN')
        self.assertEqual(sum(t == 'schedule_workflow' for t, _ in self.calls), 1)

    def test_guard_rejection_never_registers_a_runtime_or_native_job(self):
        self.arguments['steps'][0]['args']['command_id'] = 'Unsupported'
        self.run_agent()
        self.assertEqual(self.runtime.store.list('job'), [])
        self.assertEqual(self.home.workflows_by_id, {})
        self.assertFalse(any(t == 'schedule_workflow' for t, _ in self.calls))

    def test_event_supervision_uses_real_public_fast_forward_response_and_reconciles_native_start(self):
        from smarthome_agent_rl.execution.events import EventEpisodeRuntime
        self.agent.executor.dispatch = self.runtime.raw_dispatch
        self.runtime = self.agent.task_runtime = EventEpisodeRuntime(self.agent.executor,
            self.runtime.adapter, self.path.with_name('event.sqlite3'))
        self.run_agent()
        self.assertFalse(any(t == 'get_current_time' for t, _ in self.calls))
        self.assertTrue(self.home._fast_forward_to(10).success)
        response = {'status': {'code': 200}, 'data': self.home._get_home_state().data}
        result = self.runtime.observe_native_response(response)
        self.assertEqual(result[0]['status'], 'UNVERIFIED')
        self.assertTrue(self.home._fast_forward_to(11).success)
        response = {'status': {'code': 200}, 'data': self.home._get_home_state().data}
        result = self.runtime.observe_native_response(response)
        self.assertEqual(result[0]['status'], 'VERIFIED_SUCCESS')
        self.assertEqual(self.runtime.store.list('job')[0]['status'], 'DONE')
        self.assertEqual(sum(t == 'schedule_workflow' for t, _ in self.calls), 1)
        self.runtime.finish()
        self.assertEqual(self.runtime.store.list('task')[0]['status'], 'WAITING')

    def test_resource_context_reaches_real_react_and_native_conflicts_without_automatic_cancellation(self):
        from smarthome_agent_rl.execution.resource_context import ResourceEpisodeRuntime
        from smarthome_agent_rl.harness_agent import HarnessAgent
        captured = []
        off = copy.deepcopy(self.arguments)
        off['steps'][0]['args']['command_id'] = 'Off'
        self.outputs.insert(2, {'thought': 'Conflicting second intention.',
            'call': {'tool': 'schedule_workflow', 'arguments': off}})
        def generate(messages, response_format=None):
            captured.append(copy.deepcopy(messages))
            result = json.dumps(self.outputs[self.index])
            self.index += 1
            return result
        policy = {'verify': False, 'verification_version': 1, 'context_version': 0,
                  'task_runtime': True, 'task_runtime_clock': 'public_events', 'task_runtime_context': True}
        self.agent = HarnessAgent(SimpleNamespace(generate=generate), variant='GTMEC', max_steps=5, policy=policy)
        self.agent.executor.dispatch = self.dispatch
        self.runtime = self.agent.task_runtime = ResourceEpisodeRuntime(self.agent.executor,
            self.runtime.adapter, self.path.with_name('resource.sqlite3'))
        result = self.run_agent()
        self.assertEqual(result.final_answer, 'Scheduled.')
        self.assertFalse(any('PUBLIC NATIVE JOB / RESOURCE CONTEXT' in m.content for m in captured[0]))
        contexts = [json.loads(m.content.split('\n', 1)[1]) for turn in captured
                    for m in turn if m.content.startswith('PUBLIC NATIVE JOB / RESOURCE CONTEXT')]
        self.assertEqual([c['jobs_total'] for c in contexts], [1, 2])
        self.assertEqual(contexts[-1]['conflicts_total'], 1)
        self.assertEqual(len(self.home.workflows_by_id), 2)
        self.assertEqual(sum(t == 'schedule_workflow' for t, _ in self.calls), 2)
        self.assertFalse(any(t in ('cancel_workflow', 'get_current_time') for t, _ in self.calls))
        self.assertEqual(self.runtime.store.list('task')[0]['status'], 'WAITING')
        # Summary is freshly copied, never added permanently to native history.
        self.assertEqual(len([e for e in self.runtime.events if e['kind'] == 'runtime_context_exposed']), 2)


if __name__ == '__main__':
    unittest.main()
