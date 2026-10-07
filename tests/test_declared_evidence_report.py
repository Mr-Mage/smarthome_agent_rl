import copy
import unittest

from smarthome_agent_rl.declared_evidence_report import declared_evidence_report, explicit_clock_equivalences
from tests import test_effect_evidence_ablation as fixtures


class DeclaredEvidenceReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EffectEvidenceAblationTests();self.fixture.setUp()
        self.context=self.fixture.item['context'];self.decision=self.fixture.decision()

    def test_report_preserves_conflicting_model_output_and_never_dispatches_or_completes(self):
        self.decision['correct_target']['evidence']['steps'][1]['support']='unsupported'
        original=copy.deepcopy(self.decision)
        value=declared_evidence_report(self.context,self.decision,'combined')
        self.assertEqual(self.decision,original);self.assertEqual(value['original'],original)
        self.assertEqual(value['original']['verdict'],'ALLOW')
        self.assertEqual(value['projected_labels']['correct_target'],'NO')
        self.assertEqual(value['declared_evidence_verdict'],'DENY')
        self.assertIsNone(value['dispatch_verdict']);self.assertFalse(value['ready_for_blocking'])
        self.assertFalse(value['task_completed']);self.assertEqual(value['semantic_status'],'UNVERIFIED')

    def test_nonliteral_missing_clause_or_corrupted_model_verdict_has_no_projection(self):
        for modify in (lambda d:d['goal_consistent']['evidence']['steps'][0].update(effect_quote='invented'),
                       lambda d:d['goal_consistent']['evidence']['steps'][0].update(effect_quote=None),
                       lambda d:d.update(verdict='DENY'),lambda d:d['correct_target'].update(probability=True)):
            changed=copy.deepcopy(self.decision);modify(changed)
            value=declared_evidence_report(self.context,changed,'combined')
            self.assertFalse(value['available']);self.assertIsNone(value['projected_labels'])
            self.assertIsNone(value['declared_evidence_verdict']);self.assertEqual(value['original'],changed)
        self.assertFalse(declared_evidence_report(self.context,None,'combined')['available'])

    def test_other_dimensions_are_preserved_and_unknown_is_not_treated_as_error(self):
        self.decision['correct_target']['evidence']['steps'][1]['support']='unknown'
        value=declared_evidence_report(self.context,self.decision,'combined')
        self.assertEqual(value['declared_evidence_verdict'],'UNCERTAIN')
        self.assertEqual(value['projected_labels']['safe_to_execute'],self.decision['safe_to_execute']['label'])

    def test_shared_quote_does_not_override_user_binding_or_infer_room_from_id(self):
        self.decision['correct_target']['evidence']['steps'][1]['device_id']='utility_room_dehumidifier_2'
        self.context['proposed_action']['steps'][1]['args']['device_id']='utility_room_dehumidifier_2'
        self.decision['goal_consistent']['evidence']['steps'][1]['proposed_effect']['arguments']['device_id']='utility_room_dehumidifier_2'
        self.context['environment_state']['devices']={}
        value=declared_evidence_report(self.context,self.decision,'combined')
        self.assertEqual(len(value['shared_target_quotes']),1)
        self.assertIsNone(value['identity_panels'][1]['observed_room_id'])
        self.assertIsNone(value['identity_panels'][1]['quote_mentions_observed_room'])
        self.assertEqual(value['projected_labels']['correct_target'],'YES')
        self.assertEqual(value['semantic_status'],'UNVERIFIED')

    def test_explicit_clock_equivalence_keeps_both_anchors_and_staleness_assumptions(self):
        self.context['user_goal']='At 4:37 PM, that is 11 minutes from now, turn on fan.'
        self.context['environment_state']['initial_public_time']='2025-08-23 16:44:17'
        rows=explicit_clock_equivalences(self.context)
        self.assertEqual(len(rows),1);self.assertTrue(rows[0]['assumed_encodings_disagree'])
        self.assertEqual({a['expected_encoding'] for a in rows[0]['anchors']},
                         {'2025-08-23 16:37:00','2025-08-23 16:55:17'})
        start,end=rows[0]['quote_span'];self.assertEqual(rows[0]['quote'],self.context['user_goal'][start:end])
        self.assertIn('not fresh clock',rows[0]['scope'])

    def test_separate_phases_and_previous_action_never_become_equivalent_clock_conflicts(self):
        for goal in ('At 4:37 PM turn on fan and 11 minutes from now close curtain.',
                     'At 4:37 PM turn on fan, 11 minutes after the previous action turn it off.'):
            self.context['user_goal']=goal;self.assertEqual(explicit_clock_equivalences(self.context),[])
        self.context['user_goal']='At 4:37 PM, that is 11 minutes from now, turn on fan.'
        self.context['environment_state']['initial_public_time']=None
        rows=explicit_clock_equivalences(self.context)
        self.assertEqual(rows[0]['status'],'unavailable_clock_encoding');self.assertFalse(rows[0]['assumed_encodings_disagree'])


if __name__=='__main__':unittest.main()
