import copy
import json
import unittest

from smarthome_agent_rl.effect_evidence_ablation import ARMS, schema, request, checks, evaluate, action_steps
from smarthome_agent_rl.semantic_diagnosis import parse
from smarthome_agent_rl.typed_semantic_review import schema as original_schema, request as original_request
from tests import test_prompt_semantic_ablation as fixtures
from tests.test_report_semantic_review import fixture_decision


class EffectEvidenceAblationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PromptSemanticAblationTests('test_literal_evidence_is_not_claimed_to_prove_entailment')
        fixture.setUp();self.item = fixture.item;self.config = fixture.config
        self.item['context']['proposed_action']['steps'][1]['args']['value'] = 60
        self.item['context']['user_goal'] = 'Turn on lamp, then raise it to 60 after ten minutes.'

    def decision(self, arm='combined'):
        value = fixture_decision(self.item['context'], 'YES')
        for row in value['correct_target']['evidence']['steps']:
            row.update(support='requested', support_quote='lamp')
        for row, effect in zip(value['goal_consistent']['evidence']['steps'], action_steps(self.item['context'])):
            row.update(relation='not_applicable')
            if arm in ('effects', 'combined'):
                row.update(proposed_effect=effect, effect_quote='Turn on lamp' if row['step_index'] == 1 else 'raise it to 60',
                           effect_relation='agrees')
        return parse(json.dumps(value))

    def test_coherence_only_keeps_exact_schema_and_all_arms_preserve_context_and_generation(self):
        body = original_request(self.config, self.item)
        self.assertEqual(schema(self.item['context'], 'coherence'), original_schema(self.item['context']))
        for arm in ARMS:
            candidate = request(self.config, self.item, arm)
            self.assertEqual(candidate['messages'][1:], body['messages'][1:])
            self.assertEqual({k: v for k, v in candidate.items() if k not in ('messages', 'response_format')},
                             {k: v for k, v in body.items() if k not in ('messages', 'response_format')})

    def test_schema_fixes_only_actual_effect_metadata_not_requested_effect_or_verdict(self):
        context = self.item['context'];before = original_schema(context)
        value = schema(context, 'combined')['json_schema']['schema']['properties']
        rows = value['goal_consistent']['properties']['evidence']['properties']['steps']['prefixItems']
        self.assertEqual(rows[1]['properties']['proposed_effect']['const']['arguments']['value'], 60)
        self.assertNotIn('const', rows[1]['properties']['effect_relation'])
        self.assertNotIn('const', value['goal_consistent']['properties']['label'])
        changed = copy.deepcopy(context);changed['user_goal'] = 'A different desired setting.'
        self.assertEqual(schema(context, 'combined'), schema(changed, 'combined'))
        self.assertEqual(original_schema(context), before)

    def test_provider_ignoring_proposed_value_missing_quote_or_extra_field_rejected(self):
        value = self.decision();self.assertTrue(checks(self.item['context'], value, 'combined')['literal_evidence_valid'])
        changed = copy.deepcopy(value);changed['goal_consistent']['evidence']['steps'][1]['proposed_effect']['arguments']['value'] = 40
        self.assertFalse(checks(self.item['context'], changed, 'combined')['schema_conformant'])
        changed = copy.deepcopy(value);changed['goal_consistent']['evidence']['steps'][1]['effect_quote'] = None
        result = checks(self.item['context'], changed, 'combined')
        self.assertTrue(result['schema_conformant']);self.assertIn('effect:missing_operation_clause', result['evidence_issues'])
        changed = copy.deepcopy(value);changed['goal_consistent']['evidence']['steps'][1]['extra'] = 'invented'
        self.assertFalse(checks(self.item['context'], changed, 'combined')['schema_conformant'])

    def test_operation_conflict_can_support_goal_no_without_inventing_time_conflict(self):
        value = self.decision();value['goal_consistent']['label'] = 'NO'
        value['goal_consistent']['evidence']['steps'][1]['effect_relation'] = 'conflict'
        result = checks(self.item['context'], value, 'combined')
        self.assertFalse(result['aggregate_mismatches'])
        self.assertFalse(any(f['kind'] == 'goal_no_without_declared_time_conflict' for f in result['flags']))
        self.assertIn('not entailment', result['scope'])
        value['goal_consistent']['label'] = 'YES'
        self.assertIn('goal_yes_with_declared_effect_conflict', {f['kind'] for f in checks(self.item['context'], value, 'combined')['flags']})

    def test_aggregate_unknown_and_unsupported_are_checked_without_changing_model_verdict(self):
        value = self.decision();before = copy.deepcopy(value)
        value['correct_target']['evidence']['steps'][1]['support'] = 'unknown'
        result = checks(self.item['context'], value, 'combined')
        self.assertEqual(result['aggregate_mismatches'][0]['evidence_projection'], 'UNCERTAIN')
        self.assertEqual(value['correct_target']['label'], 'YES')
        self.assertEqual(before['verdict'], value['verdict'])
        value['correct_target']['evidence']['steps'][1]['support'] = 'unsupported'
        self.assertIn('target_yes_with_unsupported_steps', {f['kind'] for f in checks(self.item['context'], value, 'combined')['flags']})

    def test_nonliteral_effect_quote_and_missing_output_remain_invalid(self):
        value = self.decision();value['goal_consistent']['evidence']['steps'][0]['effect_quote'] = 'a different command'
        self.assertIn('effect:nonliteral_quote', checks(self.item['context'], value, 'combined')['evidence_issues'])
        self.assertFalse(checks(self.item['context'], None, 'combined')['schema_conformant'])

    def test_numeric_const_equality_does_not_treat_boolean_as_integer(self):
        value = self.decision()
        value['goal_consistent']['evidence']['steps'][1]['proposed_effect']['arguments']['value'] = 60.0
        self.assertTrue(checks(self.item['context'], value, 'combined')['schema_conformant'])
        value['goal_consistent']['evidence']['steps'][1]['proposed_effect']['arguments']['value'] = True
        self.assertFalse(checks(self.item['context'], value, 'combined')['schema_conformant'])

    def test_screen_and_cost_keep_failed_and_missing_requests_in_frozen_denominator(self):
        config = {**self.config, 'records': 1, 'new_requests': 3, 'valid_ratio_min': .95,
                  'evidence_valid_ratio_min': .95, 'developer_counts': {'CONSISTENT_CONTROL': 1},
                  'developer_conflict_groups': {'target': [], 'time': []}}
        inputs = [{**self.item, 'origin': 'n76', 'decision': self.decision('coherence'),
                   'call': {'error': None, 'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
                            'finish_reason': 'stop', 'request_seconds': .1}}]
        records = [{'id': 'workflow', 'arm': arm, 'decision': self.decision(arm), 'call': inputs[0]['call']} for arm in ARMS]
        labels = [{'id': 'workflow', 'category': 'CONSISTENT_CONTROL'}]
        result = evaluate(config, inputs, records, labels)
        self.assertTrue(result['complete_records']);self.assertEqual(result['new_tokens'], 15)
        self.assertTrue(all(result['diagnostic_screen_passed'].values()));self.assertFalse(result['native_admitted'])
        records[-1]['decision'] = None
        result = evaluate(config, inputs, records, labels)
        self.assertFalse(result['diagnostic_screen_passed']['combined'])
        self.assertEqual(result['arms']['combined']['tokens'], 5)
        self.assertFalse(evaluate(config, inputs, records[:-1], labels)['complete_records'])


if __name__ == '__main__':
    unittest.main()
