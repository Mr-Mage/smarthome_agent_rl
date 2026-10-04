import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.report_benchmark import report
from smarthome_agent_rl.benchmark import STRATA
from smarthome_agent_rl.concurrency import seeded_schedule
from smarthome_agent_rl.experiment_v2 import module_selection, integration_selection, service_identity, unique_policies


class V2ReportingTests(unittest.TestCase):
    def test_primary_seed_and_task_cluster_unit_and_reject_seed_corruption(self):
        rows = [{'id': str(n), 'query_type': qt, 'case': case} for n, (qt, case) in enumerate(STRATA)]
        items = seeded_schedule(rows, ['G', 'GV2'], [42, 43, 44])
        numeric = ('actor_tokens judge_tokens actor_model_calls judge_model_calls invalid_proposed '
            'invalid_reached_executor structured_rejections executed_tool_calls guard_blocked extra_queries '
            'verification_failures recovered_actions recovery_budget_blocked duration_seconds retrieval_tokens '
            'actor_latency judge_latency extra_query_latency tokenization_calls tokenization_latency').split()
        def metrics(path):
            seed = int(path.parts[-5][4:])
            variant, task = path.parts[-2], path.parts[-3]
            return {**dict.fromkeys(numeric, 1), 'task_id': task, 'variant': variant, 'actor_seed': seed,
                'success': variant == 'GV2' and seed == 42, 'task_failure': False, 'official_score': 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text(json.dumps({'phase': 'dev', 'variants': ['G', 'GV2'],
                'config': {'model_seed': 42, 'primary_actor_seed': 42}, 'actor_seeds': [42, 43, 44],
                'schedule': items, 'expected_episodes': 72, 'commit': 'test'}))
            (root / 'completion.json').write_text(json.dumps({'complete': True, 'episodes': 72}))
            with patch('scripts.report_benchmark.episode_metrics', side_effect=metrics):
                result = report(root)
            self.assertEqual(result['arms']['GV2']['unique_tasks'], 12)
            self.assertEqual(result['paired']['GV2']['wins'], 12)
            self.assertEqual(result['repeated_seed_diagnostics']['GV2']['independent_tasks'], 12)
            self.assertEqual(result['repeated_seed_diagnostics']['GV2']['by_actor_seed']['43']['wins'], 0)
            with patch('scripts.report_benchmark.episode_metrics', side_effect=lambda path: {**metrics(path), 'actor_seed': 99}):
                with self.assertRaises(ValueError):
                    report(root)

    def test_dev_gates_and_policy_deduplication(self):
        root = Path(__file__).resolve().parents[1]
        gates = json.loads((root / 'configs/harness-v2-protocol.json').read_text())
        def arm(sr, tokens, queries):
            return {'success_rate': sr, 'all_totals': {'actor_tokens': tokens, 'extra_queries': queries}}
        r = {'verified': True, 'phase': 'dev', 'arms': {'G': arm(.5, 100, 20),
            'GV': arm(.5, 110, 30), 'GC': arm(.5, 110, 20), 'GV2': arm(.5, 105, 25), 'GC2': arm(.49, 89, 20)}}
        decision = module_selection(r, gates)
        self.assertTrue(decision['verification_accepted'])
        self.assertTrue(decision['context_accepted'])
        r['arms']['GC2']['all_totals']['actor_tokens'] = 91
        self.assertFalse(module_selection(r, gates)['context_accepted'])
        config = {'variant_policies': {'Candidate': {'verify': False, 'verification_version': 2, 'context_version': 0}}}
        integration = {'verified': True, 'phase': 'dev', 'arms': {'B0': arm(.4, 100, 0),
            'G': arm(.5, 100, 20), 'Full': arm(.505, 120, 30), 'Candidate': arm(.5, 90, 20)}}
        selected = integration_selection(integration, config, gates)
        self.assertEqual(selected['winner'], 'Candidate')
        self.assertEqual(selected['formal_variants'], ['B0', 'G', 'Full'])
        self.assertEqual(selected['aliases']['Candidate'], 'G')
        unique, aliases = unique_policies(config, ['B0', 'G', 'Full', 'Candidate'])
        self.assertEqual(unique, ['B0', 'G', 'Full'])
        deduplicated = copy.deepcopy(integration)
        deduplicated['arms']['G'] = deduplicated['arms'].pop('Candidate')
        reused = integration_selection(deduplicated, config, gates)
        self.assertEqual(reused['integration_aliases']['Candidate'], 'G')
        self.assertEqual(reused['formal_variants'], ['B0', 'G', 'Full'])
        changed = copy.deepcopy(config)
        changed['actor_path'] = 'another-model'
        self.assertNotEqual(service_identity(config), service_identity(changed))
