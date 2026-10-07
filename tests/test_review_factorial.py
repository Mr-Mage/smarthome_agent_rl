import copy
import unittest

from smarthome_agent_rl.bounded_review_ablation import request as bounded_request
from smarthome_agent_rl.review_factorial import (ARMS, HISTORY_PROMPT, evaluate,
                                               known_empty_history, request,
                                               parse_response)
from tests import test_bounded_review_ablation as fixtures


class ReviewFactorialTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BoundedReviewTests()
        self.fixture.setUp()
        self.context = self.fixture.context
        self.item = self.fixture.item
        self.config = self.fixture.config

    def test_factor_changes_are_separable_and_control_matches_n87(self):
        original = request(self.config, self.item, ARMS[0])
        self.assertEqual(original, bounded_request(self.config, self.item, 'bounded'))
        history = request(self.config, self.item, ARMS[2])
        history['messages'][0]['content'] = history['messages'][0]['content'].removesuffix('\n' + HISTORY_PROMPT)
        self.assertEqual(history, original)
        plain = request(self.config, self.item, ARMS[1])
        self.assertEqual(plain['messages'], original['messages'])
        for arm in (ARMS[0], ARMS[2]):
            repeated = request(self.config, self.item, arm)
            for entry in repeated['response_format']['json_schema']['schema']['properties']['correct_target']['properties']['evidence']['properties']['steps']['prefixItems']:
                entry['properties'].pop('observed_identity')
                entry['required'].remove('observed_identity')
            self.assertEqual(repeated, request(self.config, self.item, arm.replace('metadata_', 'plain_')))

    def test_empty_history_requires_exact_observed_counters_and_nontruncation(self):
        context = copy.deepcopy(self.context)
        context['environment_state'].update(recent_mutations_total=0, recent_mutations_truncated=False)
        context['recent_actions'] = []
        self.assertTrue(known_empty_history(context))
        for field, value in [('recent_mutations_total', False), ('recent_mutations_total', None),
                             ('recent_mutations_total', 1), ('recent_mutations_truncated', True),
                             ('recent_mutations_truncated', None)]:
            changed = copy.deepcopy(context); changed['environment_state'][field] = value
            self.assertFalse(known_empty_history(changed))
        context['recent_actions'] = None
        self.assertFalse(known_empty_history(context))

    def test_metadata_projection_keeps_relations_and_labels_and_rejects_wrong_arm(self):
        raw = self.fixture.raw()
        raw['trajectory_consistent']['evidence']['steps'][0]['relation'] = 'unknown'
        text = self.fixture.fixture.text(raw)
        model, decision, error = parse_response(self.context, text, ARMS[2])
        self.assertIsNone(error)
        self.assertEqual(model['trajectory_consistent'], decision['trajectory_consistent'])
        self.assertIsNone(parse_response(self.context, text, ARMS[1])[1])
        for row in raw['correct_target']['evidence']['steps']:
            row.pop('observed_identity')
        text = self.fixture.fixture.text(raw)
        self.assertIsNone(parse_response(self.context, text, ARMS[1])[2])
        self.assertIsNone(parse_response(self.context, text, ARMS[0])[1])
        text = text.replace('"relation": "unknown"', '"relation": "unknown", "relation": "agrees"', 1)
        for arm in ARMS:
            self.assertIn('Duplicate JSON key', parse_response(self.context, text, arm)[2])

    def test_factorial_failures_keep_all_cost_and_full_denominator(self):
        raw = self.fixture.raw(); text = self.fixture.fixture.text(raw)
        model, decision, error = parse_response(self.context, text, ARMS[0])
        call = copy.deepcopy(self.fixture.fixture.fixture.fixture.row('evidence_first')['call'])
        call['text'] = text
        rows = [{'id': self.item['id'], 'arm': arm, 'call': call, 'model_decision': model,
                 'decision': decision, 'parse_error': error} for arm in ARMS]
        config = {**self.config, 'records': 1, 'new_requests': 4, 'valid_ratio_min': .95,
                  'evidence_valid_ratio_min': .95, 'developer_counts': {'CONSISTENT_CONTROL': 1},
                  'developer_conflict_groups': {'time': [], 'target': []}}
        labels = [{'id': self.item['id'], 'category': 'CONSISTENT_CONTROL'}]
        rows[-1] = {**rows[-1], 'decision': None, 'model_decision': None, 'parse_error': 'invalid'}
        report = evaluate(config, [self.item], rows, [rows[0]], labels)
        self.assertTrue(report['complete'])
        self.assertEqual(report['new_tokens'], 20)
        self.assertEqual(report['arms'][ARMS[-1]]['records'], 1)
        self.assertFalse(report['diagnostic_screen_passed'][ARMS[-1]])
        report = evaluate(config, [self.item], rows[:-1], [rows[0]], labels)
        self.assertFalse(report['complete'])
        self.assertEqual(report['new_tokens'], 15)
        self.assertFalse(any(report['diagnostic_screen_passed'].values()))


if __name__ == '__main__':
    unittest.main()
