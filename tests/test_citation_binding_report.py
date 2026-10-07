import copy
import unittest

from smarthome_agent_rl.citation_binding_report import public_identity, quote_spans, mentions, binding_report, evaluate
from smarthome_agent_rl.citation_review import resolve
from tests import test_citation_review as fixtures


class CitationBindingReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CitationReviewTests(); self.fixture.setUp()
        self.context = self.fixture.item['context']
        self.decision = resolve(self.context, self.fixture.citation_decision(), 'citations')
        self.row = {'arm': 'citations', 'model_decision': self.fixture.citation_decision(), 'decision': self.decision}

    def observed(self, target, room='dining_room', kind='dehumidifier'):
        self.context['environment_state']['devices'] = {target['device_id']: {
            'device_id': target['device_id'], 'room_id': room, 'catalog_truncated': False,
            'catalog': [{'room_id': room, 'metadata': {'device_type': kind}, 'source': {'response_sha256': 'receipt'}}],
            'structure_source': {'response_sha256': 'structure'}}}

    def test_lexical_boundaries_and_all_occurrences_do_not_choose_a_phase(self):
        self.assertTrue(mentions('In the UTILITY ROOM', 'utility_room'))
        self.assertFalse(mentions('in the utility rooms', 'utility_room'))
        self.assertIsNone(mentions('room', None))
        self.assertEqual(quote_spans('50% then 50%', '50%'), {'status': 'LITERAL', 'spans': [[0, 3], [9, 12]]})
        self.assertEqual(quote_spans('50%', None)['status'], 'NULL')
        self.assertEqual(quote_spans('50%', '80%')['status'], 'NONLITERAL')

    def test_unsupported_room_type_words_are_tension_only_never_exact_device_truth(self):
        target = self.decision['correct_target']['evidence']['steps'][0]
        self.observed(target); self.context['user_goal'] = 'Turn on dehumidifier 2 in the dining room.'
        target.update(support='unsupported', support_quote=None)
        original = copy.deepcopy(target)
        panel = public_identity(self.context, target)
        self.assertTrue(panel['unsupported_with_public_identity_words'])
        self.assertEqual(panel['public_room_type_window_candidates'][0]['text'], self.context['user_goal'])
        self.assertEqual(target, original); self.assertIn('not exact numbered-device', panel['scope'])
        self.assertEqual(panel['observed_catalog_sources'], [{'response_sha256': 'receipt'}])

    def test_absent_ambiguous_or_truncated_catalog_never_decodes_informative_device_id(self):
        target = self.decision['correct_target']['evidence']['steps'][0]
        target.update(device_id='dining_room_dehumidifier_2', support='unsupported', support_quote=None)
        self.context['user_goal'] = 'Turn on dehumidifier 2 in the dining room.'
        self.context['environment_state']['devices'] = {}
        panel = public_identity(self.context, target)
        self.assertFalse(panel['identity_word_panel_available']); self.assertFalse(panel['unsupported_with_public_identity_words'])
        self.observed(target); device = self.context['environment_state']['devices'][target['device_id']]
        for change in ({'catalog_truncated': True}, {'room_id': None}):
            changed = copy.deepcopy(self.context); changed['environment_state']['devices'][target['device_id']].update(change)
            self.assertFalse(public_identity(changed, target)['identity_word_panel_available'])
        device['catalog'].append({'metadata': {'device_type': 'fan'}, 'source': {}})
        self.assertFalse(public_identity(self.context, target)['identity_word_panel_available'])

    def test_anaphoric_or_nonliteral_support_stays_unverified_instead_of_relabelled(self):
        target = self.decision['correct_target']['evidence']['steps'][0]
        self.observed(target); self.context['user_goal'] = 'Turn on dehumidifier 2 in the dining room, then set it to 50%.'
        target.update(support='requested', support_quote='then set it to 50%')
        panel = public_identity(self.context, target)
        self.assertTrue(panel['requested_without_joint_quote_words'])
        self.assertFalse(panel['support_quote_joint_room_type_words'])
        target['support_quote'] = 'dehumidifier in dining room'
        panel = public_identity(self.context, target)
        self.assertIsNone(panel['support_quote_joint_room_type_words'])
        self.assertEqual(panel['support_quote_literal']['status'], 'NONLITERAL')

    def test_missing_nonliteral_and_multivalue_effects_preserve_failure_denominators(self):
        effect = self.decision['goal_consistent']['evidence']['steps'][1]
        effect.update(effect_quote='set fan 50%', requested_quote='22 minutes after the previous action')
        original = copy.deepcopy(self.row); report = binding_report(self.context, self.row)
        self.assertTrue(report['flags']['disjoint_literal_offset_words'] is False)  # No offset in effect quote.
        self.assertEqual(self.row, original); self.assertIsNone(report['dispatch_verdict'])
        self.assertFalse(report['native_admitted']); self.assertFalse(report['task_completed'])
        effect['effect_quote'] = 'invented 80%'
        report = binding_report(self.context, self.row)
        self.assertFalse(report['literal_evidence_valid'])
        self.assertEqual(report['effect_panels'][0]['usable_percentage_status'], 'invalid_literal_evidence')
        effect['effect_quote'] = self.context['user_goal']
        self.assertTrue(binding_report(self.context, self.row)['flags']['multiple_percentage_references'])
        effect['effect_quote'] = None; effect['effect_relation'] = 'unknown'
        self.assertTrue(binding_report(self.context, self.row)['flags']['no_explicit_percentage'])

    def test_disjoint_offset_words_are_not_an_automatic_phase_conflict(self):
        step = self.decision['goal_consistent']['evidence']['steps'][1]
        step.update(effect_quote='set fan 50% 24 minutes from now', requested_quote='22 minutes after the previous action')
        report = binding_report(self.context, self.row)
        self.assertTrue(report['flags']['disjoint_literal_offset_words'])
        self.assertEqual(report['semantic_status'], 'UNVERIFIED')
        self.assertEqual(self.decision['goal_consistent']['label'], 'YES')

    def test_complete_paired_summary_keeps_missing_decisions_and_rejects_duplicate_or_missing_arms(self):
        config = {'records': 2, 'paired_inputs': 1, 'arms': ['quotes', 'citations']}
        sources = [{'id': 'same', 'task_id': 'task', 'origin': 'n76', 'arm': arm,
            'source_file': arm + '.json', 'source_sha256': arm, 'context': self.context,
            'row': {**self.row, 'arm': arm}} for arm in config['arms']]
        sources[0]['row'] = {**sources[0]['row'], 'decision': None, 'model_decision': None}
        result, records = evaluate(config, sources)
        self.assertEqual(result['records'], 2); self.assertEqual(result['arms']['quotes']['schema_conformant'], 0)
        self.assertEqual(result['arms']['quotes']['original_verdicts'], {'INVALID': 1})
        self.assertEqual(result['coverage']['task_count'], 1); self.assertEqual(result['new_tokens'], 0)
        self.assertEqual(records[0]['report']['identity_panels'], [])
        with self.assertRaises(ValueError): evaluate(config, sources[:1])
        with self.assertRaises(ValueError): evaluate(config, [sources[0], sources[0]])


if __name__ == '__main__': unittest.main()
