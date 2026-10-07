import copy
import unittest

from smarthome_agent_rl.evidence_consistency import audit_claims, cited_anchors
from smarthome_agent_rl.semantic_diagnosis import parse
from tests import test_prompt_semantic_ablation as fixtures
from tests.test_report_semantic_review import fixture_decision
import json


class EvidenceConsistencyTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PromptSemanticAblationTests('test_literal_evidence_is_not_claimed_to_prove_entailment')
        fixture.setUp();self.context = fixture.item['context']
        self.context['user_goal'] = 'Turn on lamp 9 minutes from now, then raise it 29 minutes after the previous action.'
        self.context['environment_state']['initial_public_time'] = '2030-01-01 12:00:00'
        self.context['proposed_action']['start_time'] = '2030-01-01 12:09:00'
        self.decision = parse(json.dumps(fixture_decision(self.context, 'YES')))

    def kinds(self, decision):
        return {r['kind'] for r in audit_claims(self.context, decision)['flags']}

    def test_target_yes_cannot_hide_declared_unsupported_final_step(self):
        self.decision['correct_target']['evidence']['steps'][-1]['support'] = 'unsupported'
        self.assertIn('target_yes_with_unsupported_steps', self.kinds(self.decision))
        self.decision['correct_target']['label'] = 'UNCERTAIN'
        self.assertNotIn('target_yes_with_unsupported_steps', self.kinds(self.decision))

    def test_denial_without_negative_step_is_audit_flag_not_corrected_verdict_or_truth(self):
        self.decision['correct_target']['label'] = 'NO'
        self.decision['goal_consistent']['label'] = 'NO'
        before = copy.deepcopy(self.decision)
        result = audit_claims(self.context, self.decision)
        self.assertEqual(self.decision, before)
        self.assertIn('target_no_without_declared_unsupported_step', self.kinds(self.decision))
        self.assertIn('goal_no_without_declared_time_conflict', self.kinds(self.decision))
        self.assertTrue(all(r['class'] == 'unsubstantiated_dimension' for r in result['flags']))
        self.assertNotIn('verdict', result)

    def test_goal_yes_conflicts_with_step_evidence_and_exact_anchor_has_distinct_scope(self):
        for row in self.decision['goal_consistent']['evidence']['steps']:
            row.update(requested_quote='9 minutes from now', relation='conflict')
        kinds = self.kinds(self.decision)
        self.assertIn('goal_yes_with_declared_time_conflict', kinds)
        self.assertIn('time_conflict_despite_cited_anchor_arithmetic_match', kinds)
        self.context['proposed_action']['start_time'] = '2030-01-01 12:09:02'
        for row in self.decision['goal_consistent']['evidence']['steps']:
            row['encoded_execution_time'] = self.context['proposed_action']['start_time']
        self.assertNotIn('time_conflict_despite_cited_anchor_arithmetic_match', self.kinds(self.decision))

    def test_relative_previous_event_and_nonliteral_quotes_are_not_invented_clock_anchors(self):
        for quote in ('29 minutes after the previous action', 'not in original request'):
            result = cited_anchors(self.context, quote, '2030-01-01 12:38:00')
            self.assertEqual(result['anchors'], [])
        self.assertEqual(cited_anchors(self.context, '29 minutes after the previous action',
                                      '2030-01-01 12:38:00')['status'], 'needs_previous_action_or_event_binding')

    def test_multiple_clock_clauses_and_unknown_date_remain_explicit_assumptions(self):
        self.context['user_goal'] = 'At 12:09 PM, which is 10 minutes from now, turn on lamp.'
        result = cited_anchors(self.context, self.context['user_goal'], '2030-01-01 12:09:00')
        self.assertEqual(sorted(v['delta_seconds'] for v in result['anchors']), [-60.0, 0.0])
        self.context['user_goal'] = 'Tomorrow at 12:09 PM, turn on lamp.'
        self.assertEqual(cited_anchors(self.context, self.context['user_goal'], '2030-01-01 12:09:00')['anchors'], [])

    def test_invalid_schema_or_output_is_retained_without_claim_flags(self):
        self.assertFalse(audit_claims(self.context, None)['schema_conformant'])
        self.decision['correct_target']['evidence']['steps'].pop()
        result = audit_claims(self.context, self.decision)
        self.assertFalse(result['schema_conformant']);self.assertEqual(result['flags'], [])

    def test_hidden_fields_are_not_swallowed_as_unknown_evidence(self):
        self.context['environment_state']['judge_output'] = 'leak'
        with self.assertRaisesRegex(ValueError, 'Hidden evaluator'):
            audit_claims(self.context, self.decision)


if __name__ == '__main__':
    unittest.main()
