import copy
import json
import unittest

from scripts.analyze_native_task_runtime import audit_episode
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


if __name__ == '__main__':
    unittest.main()
