from types import SimpleNamespace
import unittest
from smarthome_agent_rl.profiling import PhaseProfile, cost_summary


class ProfileTests(unittest.TestCase):
    def test_exception_records_agent_and_restores_tools(self):
        clock = iter([0, 1, 3]).__next__
        saved = {}
        profile = PhaseProfile(lambda name, value: saved.update({name: value}), clock=clock)
        dispatch = lambda *a: 'unchanged'
        react = SimpleNamespace(run_tool=dispatch)
        def fail(*a):
            raise ValueError('task failure')
        agent = SimpleNamespace(run=fail)
        with self.assertRaises(ValueError):
            profile.agent(agent, react).run('query')
        self.assertIs(react.run_tool, dispatch)
        self.assertEqual(profile.spans[0]['outcome'], 'error')
        self.assertEqual(profile.spans[0]['end_seconds'], 3)
        profile.flush(3, .5)
        self.assertEqual(saved['phase_profile.json']['episode_start_seconds'], .5)

    def test_failed_cost_and_legacy_missing_phases_are_preserved(self):
        rows = [{'success': True, 'actor_tokens': 10, 'judge_tokens': 2, 'duration_seconds': 4, 'agent_seconds': 1},
                {'success': False, 'actor_tokens': 30, 'judge_tokens': 3, 'duration_seconds': 10}]
        result = cost_summary(rows)
        self.assertEqual(result['actor_tokens_per_success'], 40)
        self.assertEqual(result['failed_actor_tokens'], 30)
        self.assertEqual(result['agent_seconds']['observed'], 1)
        self.assertEqual(result['duration_seconds']['p50'], 7)
        self.assertIsNone(result['post_agent_seconds']['p95'])
