import copy
import unittest

from smarthome_agent_rl.decoder_whitespace_ablation import (ARMS, actor_for, evaluate,
                                                          request, whitespace_panel)
from tests import test_bounded_review_ablation as fixtures


class DecoderWhitespaceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BoundedReviewTests()
        self.fixture.setUp()

    def test_decoder_pair_has_identical_http_body_and_fixed_disjoint_replicas(self):
        config = {**self.fixture.config, 'actors': [{'id': i, 'gpu': i} for i in range(4)]}
        for prompt in ('original', 'history'):
            self.assertEqual(request(config, self.fixture.item, prompt + '_allow'),
                             request(config, self.fixture.item, prompt + '_compact'))
            for index in range(78):
                self.assertEqual(actor_for(config, index, prompt + '_allow')['id'], index % 2)
                self.assertEqual(actor_for(config, index, prompt + '_compact')['id'], 2 + index % 2)

    def test_whitespace_scan_preserves_literal_spaces_escapes_and_incomplete_outputs(self):
        panel = whitespace_panel('{"x":"a b\\\" c", "y":1}\n')
        self.assertEqual(panel['outside_string_whitespace'], 2)
        self.assertEqual(whitespace_panel('{"x":"a b"}')['outside_string_whitespace'], 0)
        panel = whitespace_panel('{"x":' + ' ' * 600)
        self.assertEqual(panel['longest_outside_string_run'], 600)
        self.assertEqual(whitespace_panel('{"x":"unfinished  ')['outside_string_whitespace'], 0)

    def test_invalid_calls_remain_in_denominator_cost_and_matching_prompt_history(self):
        from smarthome_agent_rl.review_factorial import parse_response
        raw = self.fixture.raw()
        for row in raw['correct_target']['evidence']['steps']:
            row.pop('observed_identity')
        text = self.fixture.fixture.text(raw)
        model, decision, error = parse_response(self.fixture.context, text, 'plain_original')
        call = copy.deepcopy(self.fixture.fixture.fixture.fixture.row('evidence_first')['call'])
        call['text'] = text
        rows = [{'id': self.fixture.item['id'], 'arm': arm, 'call': call, 'model_decision': model,
                 'decision': decision, 'parse_error': error} for arm in ARMS]
        config = {**self.fixture.config, 'records': 1, 'new_requests': 4, 'valid_ratio_min': .95,
                  'evidence_valid_ratio_min': .95, 'developer_counts': {'CONSISTENT_CONTROL': 1},
                  'developer_conflict_groups': {'time': [], 'target': []}}
        labels = [{'id': self.fixture.item['id'], 'category': 'CONSISTENT_CONTROL'}]
        history = {'original': [rows[0]], 'history': [{**rows[0], 'decision': None}]}
        rows[-1] = {**rows[-1], 'decision': None, 'model_decision': None, 'parse_error': 'invalid'}
        result = evaluate(config, [self.fixture.item], rows, history, labels)
        self.assertTrue(result['complete'])
        self.assertEqual(result['new_tokens'], 20)
        self.assertEqual(result['historical_transitions']['history'], {'INVALID->ALLOW': 1})
        self.assertEqual(result['arms'][ARMS[-1]]['records'], 1)
        self.assertFalse(result['diagnostic_screen_passed'][ARMS[-1]])
        self.assertFalse(result['native_admitted'])


if __name__ == '__main__':
    unittest.main()
