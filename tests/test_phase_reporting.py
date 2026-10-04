import json
from pathlib import Path
import tempfile
import unittest
from scripts.report_benchmark import episode_metrics


class PhaseReportingTests(unittest.TestCase):
    def test_episode_boundary_and_inclusive_tool_cost_are_not_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def save(name, value):
                (root / name).write_text(json.dumps(value))
            save('summary.json', {'actor_tokens': 10, 'judge_tokens': 0, 'infrastructure_error': False,
                 'official_score': 0, 'success': False, 'duration_seconds': 31})
            save('model_calls.json', [{'response': {'usage': {'total_tokens': 10}}, 'duration_seconds': 4}])
            save('agent_events.json', [{'event': 'action', 'payload': 'get_rooms'},
                {'event': 'observation', 'payload': json.dumps({'status': {'code': 200}})}])
            spans = [{'kind': kind, 'phase': phase, 'start_seconds': a, 'end_seconds': b}
                for kind, phase, a, b in [('agent', 'agent', 3, 13), ('tool_dispatch', 'agent', 7, 10),
                    ('waiting', 'agent', 11, 12), ('evaluator', 'post_agent', 14, 33),
                    ('simulator_client', 'post_agent', 15, 25)]]
            save('phase_profile.json', {'episode_seconds': 31, 'episode_start_seconds': 2, 'spans': spans})
            row = episode_metrics(root)
            self.assertEqual(row['agent_seconds'], 10)
            self.assertEqual(row['post_agent_seconds'], 20)
            self.assertEqual(row['evaluator_seconds'], 19)
            self.assertEqual(row['agent_residual_seconds'], 2)
            self.assertEqual(row['evaluation_client_seconds'], 10)
            (root / 'phase_profile.json').unlink()
            self.assertNotIn('agent_seconds', episode_metrics(root))
