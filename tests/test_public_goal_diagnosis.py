import unittest

from scripts.run_public_goal_diagnosis import evaluate


class PublicGoalDiagnosisTests(unittest.TestCase):
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
