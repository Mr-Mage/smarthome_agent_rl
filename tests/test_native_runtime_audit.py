import copy
import json
import unittest
from unittest.mock import patch

from scripts.analyze_native_task_runtime import analyze, audit_episode
from tests import test_episode_runtime as fixtures


class NativeRuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EpisodeRuntimeTests('test_single_registration_durable_before_dispatch_and_receipt_preserved')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.register()
        f.complete()
        f.runtime.finish()
        self.directory = f.root
        self.runtime = copy.deepcopy(f.receipts[-1][1])
        self.write('task_runtime.json', self.runtime)
        self.write('contract.json', {'public_context': {'query': f.runtime.task.goal},
                                    'config': {'extra_queries_max': 40}})
        self.write('summary.json', {'task_id': 'public-test', 'official_score': 0})
        self.write('official_result.json', {'evaluation_result': {'score': 0}})
        self.audit = {'extra_queries': f.executor.extra_queries, 'actual_observations': [
            {'tool': tool, 'arguments': args, 'response': response} for tool, args, response in f.actual]}
        self.write('harness_audit.json', self.audit)
        # The online runner uses this exact durable evidence filename.
        f.runtime.store.path.rename(f.root / 'task-runtime.sqlite3')

    def write(self, name, data):
        (self.directory / name).write_text(json.dumps(data), encoding='utf-8')

    def test_verified_action_and_failed_native_task_score_remain_separate(self):
        result = audit_episode(self.directory)
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['job_statuses'], {'DONE': 1})

    def test_missing_supervisor_cost_and_in_memory_only_completion_are_rejected(self):
        self.audit['actual_observations'].pop()
        self.write('harness_audit.json', self.audit)
        self.runtime['tasks'][0]['status'] = 'COMPLETED'
        self.write('task_runtime.json', self.runtime)
        problems = audit_episode(self.directory)['problems']
        self.assertTrue(any('conservative completion' in p for p in problems))
        self.assertTrue(any('Durable task' in p for p in problems))
        self.assertTrue(any('supervisor cost' in p for p in problems))

    def test_unfinished_model_task_is_retained_without_inventing_native_score(self):
        (self.directory / 'official_result.json').unlink()
        self.write('summary.json', {'task_id': 'public-test', 'official_score': None,
                                    'task_failure': True, 'task_failure_kind': 'max_steps'})
        self.assertEqual(audit_episode(self.directory)['problems'], [])

    def test_incomplete_registration_is_reported_as_failure_without_dropping_episode(self):
        self.runtime['workflows'][0]['evidence'] = [
            e for e in self.runtime['workflows'][0]['evidence'] if e.get('kind') != 'registration']
        self.write('task_runtime.json', self.runtime)
        problems = audit_episode(self.directory)['problems']
        self.assertTrue(any('completed receipt linkage' in p for p in problems))

    def test_event_completion_requires_untampered_public_clock_receipt(self):
        self.write('contract.json', {'public_context': {'query': self.fixture.runtime.task.goal},
            'config': {'extra_queries_max': 40, 'variant_policies': {
                'GTME': {'task_runtime_clock': 'public_events'}}}})
        self.assertEqual(audit_episode(self.directory)['problems'], [])
        job = self.runtime['jobs'][0]
        evidence = job['evidence'][-1]['evidence']
        clock = next(e for e in evidence if e.get('kind') == 'public_observation_clock')
        clock['invocation_id'] = 'missing-public-receipt'
        self.write('task_runtime.json', self.runtime)
        self.assertTrue(any('clock does not match' in p for p in audit_episode(self.directory)['problems']))
        evidence.remove(clock)
        self.write('task_runtime.json', self.runtime)
        self.assertTrue(any('lacks clock receipt' in p for p in audit_episode(self.directory)['problems']))


class NativeRuntimeStageAuditTests(unittest.TestCase):
    def test_repeated_seeds_count_unique_tasks_and_require_actual_verification(self):
        # Six candidate runs are only two public tasks, not six independent tasks.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            stage = run / 'dev'
            stage.mkdir()
            config = {'variant_policies': {'G': {'verify': False}, 'GTME': {
                'verify': False, 'task_runtime': True, 'task_runtime_clock': 'public_events'}},
                'node_experiment': {'candidate': 'GTME', 'gates': {'complete_episodes': 12,
                    'public_tasks': 2, 'candidate_executions': 6, 'minimum_native_jobs': 1,
                    'minimum_native_time_callbacks': 1, 'minimum_verified_native_jobs': 1}}}
            for name, data in {'protocol.json': {'config': config, 'commit': 'frozen',
                    'variants': ['G', 'GTME'], 'expected_episodes': 12, 'schedule': [{}] * 6},
                    'report.json': {'arms': {'G': {'successes': 0}, 'GTME': {'successes': 0}}, 'paired': []},
                    'artifact_manifest.json': {}}.items():
                (stage / name).write_text(json.dumps(data), encoding='utf-8')
            rows = [{'task_id': f'public-{i % 2}', 'problems': [], 'jobs': 1,
                'job_statuses': {'UNKNOWN': 1}, 'native_clock_callbacks': 1, 'supervisor_queries': 2}
                for i in range(6)]
            with patch('scripts.analyze_native_task_runtime.verify', return_value={'expected_episodes': 12}), \
                    patch('scripts.analyze_native_task_runtime.episode_directory', return_value=stage), \
                    patch('scripts.analyze_native_task_runtime.audit_episode', side_effect=rows):
                result = analyze(run, 'dev')
            self.assertTrue(result['checks']['public_tasks'])
            self.assertTrue(result['checks']['candidate_executions'])
            self.assertFalse(result['engineering_accepted'])
            rows[0]['job_statuses'] = {'DONE': 1}
            with patch('scripts.analyze_native_task_runtime.verify', return_value={'expected_episodes': 12}), \
                    patch('scripts.analyze_native_task_runtime.episode_directory', return_value=stage), \
                    patch('scripts.analyze_native_task_runtime.audit_episode', side_effect=rows):
                self.assertTrue(analyze(run, 'dev')['engineering_accepted'])


if __name__ == '__main__':
    unittest.main()
