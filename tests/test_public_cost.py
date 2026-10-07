import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_public_benchmark_cost import analyze, distribution


class PublicCostTests(unittest.TestCase):
    def test_shared_outputs_do_not_multiply_cost_and_bulk_queue_is_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {'status': 'complete', 'seconds': 10,
                      'cost': {'actual_actor_requests': 1, 'total_tokens': 12,
                               'missing_usage_requests': 0, 'failed_actor_requests': 0}}
            row = {'arm': 'B1', 'task_id': 'x', 'request_receipt': 'requests/x.json',
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}, 'request_seconds': 2,
                   'queue_seconds': 5, 'episode_seconds': 7}
            (root / 'report.json').write_text(json.dumps(report))
            shared = {**row, 'arm': 'WG', 'shared_request': 'B1'}
            (root / 'episodes.jsonl').write_text('\n'.join(map(json.dumps, (row, shared))))
            result = analyze(root)
            self.assertEqual(result['logical_tokens'], 12)
            self.assertEqual(result['request_seconds']['p95'], 2)
            self.assertEqual(result['queue_seconds']['p95'], 5)
            self.assertIsNone(result['reserved_h100_gpu_seconds'])
            shared['request_receipt'] = 'missing'
            (root / 'episodes.jsonl').write_text('\n'.join(map(json.dumps, (row, shared))))
            with self.assertRaisesRegex(ValueError, 'matching real request'):
                analyze(root)

    def test_quantiles_and_invalid_observations(self):
        self.assertEqual(distribution(list(range(1, 101)))['p95'], 95)
        self.assertIsNone(distribution([])['p50'])
        for values in ([float('nan')], [-1], [True]):
            with self.assertRaises(ValueError):
                distribution(values)


if __name__ == '__main__':
    unittest.main()
