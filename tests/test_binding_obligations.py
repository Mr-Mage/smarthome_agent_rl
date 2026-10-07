import copy
import unittest

from smarthome_agent_rl.binding_obligations import (identity,identity_words,
    operation_words,time_words,interval_links,quote_panel,report_record,evaluate)
from smarthome_agent_rl.citation_review import resolve
from tests import test_citation_review as fixtures


class BindingObligationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CitationReviewTests();self.fixture.setUp()
        self.context=self.fixture.item['context'];self.decision=resolve(self.context,self.fixture.citation_decision(),'citations')
        self.row={'model_decision':self.fixture.citation_decision(),'decision':self.decision}
        device=self.decision['correct_target']['evidence']['steps'][0]['device_id']
        self.device=device
        self.context['environment_state']['devices']={device:{'room_id':'dining_room','catalog_total':1,'catalog_truncated':False,
            'catalog':[{'room_id':'dining_room','metadata':{'device_type':'dehumidifier'},'source':{'tool':'get_room_devices','arguments':{'room_id':'dining_room'},'observation_ordinal':1,'response_sha256':'a'*64}}]}}

    def test_identity_requires_public_receipt_without_decoding_device_identifier(self):
        self.assertEqual(identity(self.context,self.device)['status'],'public_catalog_words_available')
        missing=copy.deepcopy(self.context);missing['environment_state']['devices']={}
        self.assertEqual(identity(missing,'dining_room_dehumidifier_2')['status'],'unavailable')
        for change in ({'catalog_truncated':True},{'catalog_total':2},{'room_id':'other_room'}):
            changed=copy.deepcopy(self.context);changed['environment_state']['devices'][self.device].update(change)
            self.assertNotEqual(identity(changed,self.device)['status'],'public_catalog_words_available')
        changed=copy.deepcopy(self.context);changed['environment_state']['devices'][self.device]['catalog'][0]['source']['arguments']={'room_id':'other_room'}
        self.assertEqual(identity(changed,self.device)['status'],'invalid_catalog_receipt')
        panel=identity(self.context,self.device)
        self.assertFalse(identity_words('Turn on dehumidifier 2.',panel)['joint_words'])
        self.assertTrue(identity_words('In the DINING ROOM switch on dehumidifier 2.',panel)['joint_words'])

    def test_operation_literals_keep_conflicting_values_negation_and_uncovered_names(self):
        step={'tool':'write_attribute','arguments':{'cluster_id':'FanControl','attribute_id':'PercentSetting','value':50}}
        result=operation_words('fan 50% then eighty percent',step)
        self.assertEqual([r['value'] for r in result['literals']],[50,80]);self.assertEqual(result['proposed_value'],50)
        command={'tool':'execute_command','arguments':{'cluster_id':'OnOff','command_id':'On'}}
        result=operation_words('Do not switch on; turn off.',command)
        self.assertEqual([r['command_word'] for r in result['literals']],['On','Off'])
        self.assertIn('negation',result['scope'])
        command['arguments']['cluster_id']='Unknown'
        self.assertEqual(operation_words('turn on',command)['status'],'uncovered_operation')

    def test_repeated_quotes_keep_all_occurrence_pairs_no_chosen_phase(self):
        goal='fan 50% then fan 50%';panel=identity(self.context,self.device)
        step={'tool':'write_attribute','arguments':{'cluster_id':'FanControl','attribute_id':'PercentSetting','value':50}}
        cited=quote_panel(goal,'fan 50%',panel,step)
        spans=cited['literal']['spans'];self.assertEqual(spans,[[0,7],[13,20]])
        links=interval_links(spans,spans);self.assertEqual(len(links),4)
        self.assertEqual([r['relation'] for r in links],['same_span','disjoint','disjoint','same_span'])
        self.assertEqual(quote_panel(goal,'invented 50%',panel,step)['status'],'unavailable')

    def test_previous_action_and_clock_anchors_remain_unbound_not_invented_arithmetic(self):
        text='24 minutes from now, then 22 minutes after the previous action at 2:30 PM'
        result=time_words(text)
        self.assertEqual([r['anchor'] for r in result['literals']],['initial_public_time','previous_action_unbound','date_timezone_unbound'])
        self.assertNotIn('expected_time',str(result))

    def test_all_window_and_selected_edges_preserve_bad_labels_and_cross_device_quotes(self):
        self.context['user_goal']='Switch on dehumidifier 2 in the dining room 9 minutes from now and fan 50%. Also switch on air purifier 1 in the bathroom 24 minutes from now and fan 80%.'
        for t in self.decision['correct_target']['evidence']['steps']:t.update(support='unsupported',support_quote=None)
        for c in self.decision['goal_consistent']['evidence']['steps']:
            c.update(requested_quote='switch on air purifier 1 in the bathroom 24 minutes from now',
                     effect_quote='Switch on dehumidifier 2 in the dining room 9 minutes from now and fan 50%',relation='conflict',effect_relation='conflict')
        before=copy.deepcopy((self.context,self.row));result=report_record(self.context,self.row)
        self.assertEqual((self.context,self.row),before)
        self.assertTrue(all(s['flags']['selected_operation_time_all_disjoint'] for s in result['steps']))
        self.assertTrue(all(s['flags']['selected_time_without_device_words'] for s in result['steps']))
        self.assertTrue(all(s['flags']['unsupported_with_identity_windows'] for s in result['steps']))
        self.assertTrue(any(s['joint_word_window_ids'] for s in result['steps']))
        # Fixture command lacks cluster_id; never infer its capability family.
        self.assertTrue(result['steps'][0]['flags']['uncovered_operation'])
        self.assertEqual(result['semantic_status'],'UNVERIFIED');self.assertIsNone(result['dispatch_verdict'])
        self.assertFalse(result['native_admitted']);self.assertFalse(result['task_completed'])

    def test_failures_arms_hidden_fields_and_denominators_are_not_dropped(self):
        arms=['n93_joint','n93_evidence_only','n92_independent'];config={'arms':arms,'paired_inputs':1,'records':3}
        sources=[{'id':'case','task_id':'task','origin':'n76','arm':arm,'context':self.context,'row':copy.deepcopy(self.row),
                  'source_file':arm+'.json','source_sha256':arm,'source_run':arm} for arm in arms]
        sources[-1]['row'].update(decision=None,model_decision=None)
        report,records=evaluate(config,sources)
        self.assertEqual(report['records'],3);self.assertEqual(report['arms']['n92_independent']['records'],1)
        self.assertEqual(report['arms']['n92_independent']['original_verdicts'],{'INVALID':1})
        self.assertEqual(records[-1]['report']['steps'],[]);self.assertEqual(report['new_model_requests'],0)
        with self.assertRaises(ValueError):evaluate(config,sources[:2])
        with self.assertRaises(ValueError):evaluate(config,[sources[0],sources[0],sources[2]])
        self.context['environment_state']['judge_output']='hidden'
        with self.assertRaisesRegex(ValueError,'Hidden evaluator'):report_record(self.context,self.row)


if __name__=='__main__':unittest.main()
