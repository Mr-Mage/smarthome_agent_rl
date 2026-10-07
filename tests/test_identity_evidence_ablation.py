import copy
import json
import unittest

from smarthome_agent_rl.identity_evidence_ablation import observed_identity, schema, request, parse_response, evaluate
from smarthome_agent_rl.citation_review import request as original_request
from tests import test_citation_review as fixtures


class IdentityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CitationReviewTests(); self.fixture.setUp()
        self.context = self.fixture.item['context']; self.item = self.fixture.item; self.config = self.fixture.config
        device = self.context['proposed_action']['steps'][0]['args']['device_id']
        self.context['environment_state']['devices'] = {device: {'room_id': 'utility_room', 'catalog_total': 1,
            'catalog_truncated': False, 'catalog': [{'room_id': 'utility_room', 'metadata': {'device_type': 'dehumidifier'},
                'source': {'observation_ordinal': 2, 'response_sha256': 'receipt'}}]}}

    def raw(self):
        raw = self.fixture.citation_decision()
        for row in raw['correct_target']['evidence']['steps']:
            row['observed_identity'] = observed_identity(self.context, row['device_id'])
        return raw

    def text(self, raw):
        return json.dumps({name: raw[name] for name in ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute')})

    def test_only_schema_changes_all_messages_and_predicted_labels_unconstrained(self):
        original = original_request(self.config, self.item, 'citations')
        self.assertEqual(request(self.config, self.item, 'control'), original)
        changed = request(self.config, self.item, 'identity')
        changed['response_format'] = original['response_format']; self.assertEqual(changed, original)
        fields = schema(self.context, 'identity')['json_schema']['schema']['properties']['correct_target']
        self.assertNotIn('const', fields['properties']['label'])
        entry = fields['properties']['evidence']['properties']['steps']['prefixItems'][0]['properties']
        self.assertEqual(list(entry), ['step_index', 'device_id', 'observed_identity', 'support_ref', 'support'])
        self.assertNotIn('const', entry['support']); self.assertEqual(entry['observed_identity']['const']['room_id'], 'utility_room')

    def test_unobserved_identity_and_ambiguity_never_decode_id_or_select_room_type(self):
        self.assertIsNone(observed_identity(self.context, 'living_room_fan_1'))
        device = next(iter(self.context['environment_state']['devices']))
        row = self.context['environment_state']['devices'][device]
        row.update(room_id=None, catalog_total=2, catalog_truncated=True)
        row['catalog'].append({'room_id': 'other_room', 'metadata': {'device_type': 'fan'}, 'source': {'response_sha256': 'other'}})
        value = observed_identity(self.context, device)
        self.assertIsNone(value['room_id']); self.assertTrue(value['catalog_truncated'])
        self.assertEqual([r['device_type'] for r in value['catalog']], ['dehumidifier', 'fan'])
        changed = copy.deepcopy(self.context); changed['user_goal'] = 'Do something else'
        self.assertEqual(observed_identity(changed, device), value)

    def test_metadata_projection_preserves_wrong_model_support_labels_and_raw_response(self):
        raw = self.raw(); raw['correct_target']['evidence']['steps'][0]['support'] = 'unsupported'
        text = self.text(raw); model, decision, error = parse_response(self.context, text, 'identity')
        self.assertIsNone(error); self.assertEqual(model['correct_target']['evidence']['steps'][0]['observed_identity'], raw['correct_target']['evidence']['steps'][0]['observed_identity'])
        self.assertNotIn('observed_identity', decision['correct_target']['evidence']['steps'][0])
        self.assertEqual(decision['correct_target']['evidence']['steps'][0]['support'], 'unsupported')
        for name in ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute'):
            self.assertEqual(decision[name]['label'], model[name]['label']); self.assertEqual(decision[name]['probability'], model[name]['probability'])
        self.assertEqual(model['verdict'], decision['verdict']); self.assertEqual(text, self.text(raw))

    def test_missing_forged_and_boolean_metadata_or_bad_reference_is_retained_failure(self):
        for change in (lambda r: r.pop('observed_identity'),
                       lambda r: r['observed_identity'].update(room_id='living_room'),
                       lambda r: r['observed_identity'].update(catalog_total=True),
                       lambda r: r.update(support_ref=999)):
            raw = self.raw(); change(raw['correct_target']['evidence']['steps'][0])
            model, decision, error = parse_response(self.context, self.text(raw), 'identity')
            self.assertIsNotNone(model); self.assertIsNone(decision); self.assertIsNotNone(error)
        model, decision, error = parse_response(self.context, self.text(self.raw()), 'control')
        self.assertIsNone(decision); self.assertIsNotNone(error)

    def test_failed_metadata_keeps_cost_and_full_paired_denominator_without_admission(self):
        config = {**self.config, 'records': 1, 'new_requests': 2, 'valid_ratio_min': .95, 'evidence_valid_ratio_min': .95,
            'developer_counts': {'CONSISTENT_CONTROL': 1}, 'developer_conflict_groups': {'target': [], 'time': []}}
        rows = []
        for arm in ('control', 'identity'):
            raw = self.fixture.citation_decision() if arm == 'control' else self.raw()
            if arm == 'identity': raw['correct_target']['evidence']['steps'][0].pop('observed_identity')
            text = self.text(raw); model, decision, error = parse_response(self.context, text, arm)
            call = copy.deepcopy(self.fixture.fixture.row('evidence_first')['call']); call['text'] = text
            rows.append({'id': self.item['id'], 'arm': arm, 'model_decision': model, 'decision': decision,
                'parse_error': error, 'call': call})
        result = evaluate(config, [self.item], rows, [rows[0]], [{'id': self.item['id'], 'category': 'CONSISTENT_CONTROL'}])
        self.assertTrue(result['complete']); self.assertEqual(result['new_tokens'], 10)
        self.assertEqual(result['arms']['identity']['records'], 1)
        self.assertEqual(result['arms']['identity']['resolution_errors'], 1)
        self.assertEqual(result['raw_identity_metadata_valid']['identity'], 0)
        self.assertFalse(result['diagnostic_screen_passed']['identity']); self.assertFalse(result['native_admitted'])


if __name__ == '__main__': unittest.main()
