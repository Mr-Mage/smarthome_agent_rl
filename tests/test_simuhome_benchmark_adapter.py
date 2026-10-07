import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.benchmarks.actors import ChatActor, ReActActor
from smarthome_agent_rl.benchmarks.simuhome import SimuHomeAdapter
from scripts.run_official_benchmark import command


class SimuHomeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.episode = {'query': 'turn off the light', 'user_location': 'living room',
            'initial_home_config': {'base_time': '2025-01-01 12:00:00', 'secret_world_setup': 'DO_NOT_SEND'},
            'meta': {'query_type': 'qt1', 'case': 'feasible', 'seed': 10},
            'goal_state': 'HIDDEN_GOAL', 'evaluation': {'secret': 'HIDDEN_LABEL'}}
        self.path = self.root / 'case.json'
        self.path.write_text(json.dumps(self.episode))
        self.row = {'id': 'case', 'path': 'case.json', 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    **self.episode['meta']}
        self.adapter = SimuHomeAdapter(self.root, {'tasks': [self.row]})

    def test_native_agent_boundary_excludes_goal_world_and_category(self):
        public = self.adapter.public_input('case')
        calls = []
        def native_run(user_query, /, *, user_location=None, current_time=None):
            calls.append({'query': user_query, 'user_location': user_location, 'current_time': current_time})
            return 'native-artifact'
        agent = SimpleNamespace(run=native_run, extra='native')
        bound = self.adapter.bind_agent('case', agent)
        self.assertEqual(bound.run(**public), 'native-artifact')
        self.assertEqual(bound.extra, 'native')
        self.assertEqual(set(calls[0]), {'query', 'user_location', 'current_time'})
        self.assertNotIn('HIDDEN', json.dumps(calls))
        self.assertNotIn('DO_NOT_SEND', json.dumps(calls))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            bound.run(public['query'] + ' HIDDEN_GOAL', user_location=public['user_location'],
                      current_time=public['current_time'])
        self.assertEqual(len(calls), 1)

    def test_native_score_is_separate_from_runtime_success_and_keeps_errors(self):
        result = {'query': self.episode['query'], 'query_type': 'qt1', 'case': 'feasible',
                  'evaluation_result': {'score': -1}, 'task_runtime': {'status': 'COMPLETED'}}
        self.assertTrue(self.adapter.score('case', result)['evaluator_error'])
        self.assertFalse(self.adapter.score('case', result)['success'])
        result['query'] = 'different task'
        with self.assertRaises(ValueError):
            self.adapter.score('case', result)

    def test_source_mutation_and_path_escape_are_rejected(self):
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.adapter.public_input('case')
        self.adapter.tasks['case']['path'] = '../case.json'
        with self.assertRaisesRegex(ValueError, 'escaped'):
            self.adapter.public_input('case')

    def test_both_actor_types_accept_only_their_public_transport(self):
        calls = []
        options = {'model': 'fixture', 'seed': 42}
        actor = ChatActor(lambda *args: calls.append(args) or 'response', 'http://actor/v1', options, 30)
        self.assertEqual(actor.invoke([{'role': 'system', 'content': 'public'}]), 'response')
        self.assertEqual(calls[0][1], {'model': 'fixture', 'seed': 42,
                                     'messages': [{'role': 'system', 'content': 'public'}]})
        with self.assertRaises(ValueError):
            ReActActor(SimpleNamespace()).invoke({'query': 'q', 'goal_state': 'hidden'})

    def test_common_entry_retains_native_environment_and_stages(self):
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps({'driver_python': '/existing/lightning/python',
                                   'node_experiment': {'stages': [{'name': 'calibration'}]}}))
        argv = command('SimuHome', cfg, self.root / 'output')
        self.assertEqual(argv[0], '/existing/lightning/python')
        self.assertIn('run_node_experiment.py', argv[1])
        with self.assertRaises(ValueError):
            command('SimuHome', cfg, self.root / 'output', stage='full')
        with self.assertRaises(ValueError):
            command('SAGE', cfg, self.root / 'output')


if __name__ == '__main__':
    unittest.main()
