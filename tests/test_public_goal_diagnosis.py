from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_public_goal_diagnosis import evaluate, external_actor, read


class PublicGoalDiagnosisTests(unittest.TestCase):
    def test_borrowed_service_retains_probe_cost_and_is_not_stopped_on_failure(self):
        config = {'model': 'shared', 'actors': [{'endpoint': 'http://public-service/v1'}], 'request_timeout': 10}
        call = {'error': None, 'response': {'model': 'shared'},
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        with tempfile.TemporaryDirectory() as temp, patch('scripts.run_public_goal_diagnosis.completion', return_value=call):
            service = Path(temp) / 'success'
            with self.assertRaises(RuntimeError):
                with external_actor(config, service):
                    raise RuntimeError('Later diagnosis failure')
            lifecycle = read(service / 'lifecycle.json')
            self.assertEqual(lifecycle['probes']['tokens'], 2)
            self.assertEqual(lifecycle['reserved_h100_gpu_seconds'], 0)
            self.assertFalse(lifecycle['managed'])
            self.assertFalse(lifecycle['external_service_stopped'])
            self.assertEqual(read(service / 'probe-receipts.json'), [call])

    def test_shared_model_roles_are_counted_once_without_double_counting_tokens(self):
        config = {'actor_seeds': [42], 'model': 'shared', 'review_model': 'shared',
            'gates': {'public_tasks': 1, 'records': 1, 'valid_proposal_ratio_min': .95, 'review_all_yes_ratio_min': .95}}
        call = lambda total: {'body': {'model': 'shared'}, 'error': None,
            'usage': {'prompt_tokens': total - 1, 'completion_tokens': 1, 'total_tokens': total}}
        rows = [{'task_id': 'public', 'seed': 42, 'proposal': None, 'review': None, 'calls': [call(7), call(11)]}]
        cost = evaluate(config, rows)['cost']
        self.assertEqual(cost, {'extractor': {'requests': 1, 'tokens': 7}, 'reviewer': {'requests': 1, 'tokens': 11}})

    def test_repeats_missing_calls_and_model_agreement_do_not_prove_goal_completion(self):
        config = {'actor_seeds': [42, 43], 'model': 'extractor', 'review_model': 'reviewer',
            'gates': {'public_tasks': 1, 'records': 2, 'valid_proposal_ratio_min': .95,
                      'review_all_yes_ratio_min': .95}}
        review = {key: 'YES' for key in ('coverage', 'fidelity', 'targets', 'dependencies')}
        call = {'body': {'model': 'extractor'}, 'error': None,
                'usage': {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 4}}
        rows = [{'task_id': 'public', 'seed': seed, 'proposal': {'time_graph': []},
                 'review': review, 'calls': [dict(call)]} for seed in (42, 43)]
        report = evaluate(config, rows)
        self.assertTrue(report['checks']['complete_records'])
        self.assertTrue(report['checks']['model_review_fidelity'])
        self.assertFalse(report['ready_for_native_integration'])
        self.assertFalse(report['checks']['manual_semantics'])
        rows[1]['seed'] = 42
        self.assertFalse(evaluate(config, rows)['checks']['complete_records'])
        rows[1]['seed'] = 43
        rows[1]['calls'][0]['usage'] = None
        self.assertFalse(evaluate(config, rows)['checks']['usage_complete'])
        rows[1]['review'] = None
        self.assertFalse(evaluate(config, rows)['checks']['model_review_fidelity'])


if __name__ == '__main__':
    unittest.main()
