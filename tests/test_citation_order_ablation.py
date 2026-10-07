import copy
import json
import unittest

from smarthome_agent_rl.citation_order_ablation import ARMS, actor_for, request, parse_response, evaluate
from smarthome_agent_rl.review_factorial import request as parent_request
from tests import test_bounded_review_ablation as fixtures


class CitationOrderTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BoundedReviewTests(); self.fixture.setUp()
        self.item = self.fixture.item; self.context = self.fixture.context
        self.config = {**self.fixture.config, 'actors': [{'id': i, 'gpu': i} for i in range(4)]}

    def test_only_display_order_changes_exact_text_ids_schema_and_context_preserved(self):
        before = copy.deepcopy(self.item)
        left = request(self.config, self.item, 'original'); right = request(self.config, self.item, 'reverse')
        self.assertEqual(left, parent_request(self.config, self.item, 'plain_history'))
        differences = 0
        for a, b in zip(left['messages'], right['messages']):
            if a == b: continue
            differences += 1
            original = json.loads(a['content']); reversed_value = json.loads(b['content'])
            self.assertEqual(reversed_value['public_citations'], list(reversed(original['public_citations'])))
            reversed_value['public_citations'].reverse()
            self.assertEqual(json.dumps(reversed_value, ensure_ascii=False), a['content'])
            b['content'] = a['content']
        self.assertEqual(differences, 1); self.assertEqual(left, right); self.assertEqual(self.item, before)
        for index in range(78):
            self.assertEqual(actor_for(self.config, index, 'original'), actor_for(self.config, index, 'reverse'))
            self.assertEqual(actor_for(self.config, index, 'reverse')['id'], index % 4)
        with self.assertRaises(ValueError): request(self.config, self.item, 'chosen_truth')

    def row(self, arm):
        raw = self.fixture.raw()
        for step in raw['correct_target']['evidence']['steps']: step.pop('observed_identity')
        raw['safe_to_execute']['evidence']['steps'][0]['relation'] = 'conflict'
        text = self.fixture.fixture.text(raw)
        model, decision, error = parse_response(self.context, text, arm)
        self.assertIsNone(error); self.assertEqual(decision['safe_to_execute']['label'], 'YES')
        self.assertEqual(model['verdict'], decision['verdict'])
        call = copy.deepcopy(self.fixture.fixture.fixture.fixture.row('evidence_first')['call']); call['text'] = text
        return {'id': self.item['id'], 'arm': arm, 'call': call, 'model_decision': model,
                'decision': decision, 'parse_error': error}

    def test_invalid_and_missing_responses_remain_in_cost_denominator_no_adoption(self):
        rows = [self.row(arm) for arm in ARMS]
        config = {**self.config, 'records': 1, 'new_requests': 2, 'valid_ratio_min': .95,
            'evidence_valid_ratio_min': .95, 'developer_counts': {'CONSISTENT_CONTROL': 1},
            'developer_conflict_groups': {'time': [], 'target': []}}
        labels = [{'id': self.item['id'], 'category': 'CONSISTENT_CONTROL'}]
        result = evaluate(config, [self.item], rows, [rows[0]], labels)
        self.assertTrue(result['complete']); self.assertFalse(result['native_admitted'])
        self.assertEqual(result['new_tokens'], 10)
        self.assertEqual(result['bounded_dimensions']['reverse']['safe_to_execute']['label_mismatch'], 1)
        rows[1] = {**rows[1], 'decision': None, 'model_decision': None, 'parse_error': 'invalid'}
        result = evaluate(config, [self.item], rows, [rows[0]], labels)
        self.assertEqual(result['new_tokens'], 10); self.assertFalse(result['diagnostic_screen_passed']['reverse'])
        self.assertEqual(result['paired_changes']['verdict'][0]['reverse'], 'INVALID')
        self.assertFalse(evaluate(config, [self.item], rows[:1], [rows[0]], labels)['complete'])


if __name__ == '__main__': unittest.main()
