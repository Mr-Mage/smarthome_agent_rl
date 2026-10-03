import json
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.benchmark import select, schedule, STRATA, task_failure_kind


class BenchmarkTests(unittest.TestCase):
    def test_official_selection_is_disjoint_balanced_and_excludes_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for qt, case in STRATA:
                for seed in range(1, 51):
                    (root / f'{qt}_{case}_seed_{seed}.json').write_text(json.dumps({
                        'meta': {'query_type': qt, 'case': case, 'seed': seed}}))
            first = select(root)
            self.assertEqual(first, select(root))
            self.assertEqual(len(first['dev']), 120)
            self.assertEqual(len(first['final']), 192)
            self.assertFalse({r['id'] for r in first['dev']} & {r['id'] for r in first['final']})
            self.assertFalse(set(first['excluded_from_final']) & {r['id'] for r in first['final']})
            for qt, case in STRATA:
                self.assertEqual(sum((r['query_type'], r['case']) == (qt, case) for r in first['final']), 16)

    def test_paired_arms_never_cross_actors_and_order_is_counterbalanced(self):
        rows = [{'id': str(i)} for i in range(8)]
        jobs = schedule(rows, ['B0', 'G'])
        self.assertEqual([j['workflow'] for j in jobs], [0, 1] * 4)
        for workflow in (0, 1):
            orders = [j['variants'] for j in jobs if j['workflow'] == workflow]
            self.assertEqual(orders, [['B0', 'G'], ['G', 'B0']] * 2)

    def test_actor_context_budget_is_task_data_and_service_failures_are_infrastructure(self):
        error = {'type': 'AgentExecutionError', 'message': 'Episode failed: maximum context length exceeded'}
        self.assertEqual(task_failure_kind(error, [{'status': 400}]), 'actor_context_limit')
        self.assertIsNone(task_failure_kind(error, [{'status': 500}]))
        self.assertIsNone(task_failure_kind({'type': 'AgentExecutionError', 'message': 'LLM connection failed'}, []))
        self.assertEqual(task_failure_kind({'type': 'AgentExecutionError',
            'message': 'Episode failed: explicit finish action is required.'}), 'step_limit')


if __name__ == '__main__':
    unittest.main()
