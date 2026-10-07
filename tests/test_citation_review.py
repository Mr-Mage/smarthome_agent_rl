import copy
import json
import unittest

from smarthome_agent_rl.citation_review import catalog,request,schema,resolve,binding_checks,evaluate,ARMS
from smarthome_agent_rl.evidence_order_ablation import request as original_request
from smarthome_agent_rl.effect_evidence_ablation import action_steps
from tests import test_evidence_order_ablation as fixtures


class CitationReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EvidenceOrderAblationTests();self.fixture.setUp()
        self.config=self.fixture.config;self.item=self.fixture.item
        self.item['context']['proposed_action']['steps'][1]['args'].update(cluster_id='FanControl',attribute_id='PercentSetting')
        self.item['context']['user_goal']='Turn on lamp, and set fan 50% 24 minutes from now, then increase fan to 80% 22 minutes after the previous action.'

    def citation_decision(self):
        value=copy.deepcopy(self.fixture.row('evidence_first')['decision'])
        for row in value['correct_target']['evidence']['steps']:
            row.pop('support_quote');row['support_ref']=0
        for row,effect in zip(value['goal_consistent']['evidence']['steps'],action_steps(self.item['context'])):
            row.pop('requested_quote');row.pop('effect_quote');row.update(requested_ref=3,effect_ref=3)
            row['proposed_effect']=effect
        return value

    def test_all_catalog_entries_are_contiguous_original_spans_with_no_hidden_desired_values(self):
        goal=self.item['context']['user_goal'];rows=catalog(goal)
        self.assertEqual([row['id'] for row in rows],list(range(len(rows))))
        self.assertTrue(all(row['text']==goal[row['start']:row['end']] for row in rows))
        self.assertTrue(all(set(row)=={'id','start','end','text'} for row in rows))
        changed=copy.deepcopy(self.item);changed['context']['proposed_action']['steps'][1]['args']['value']=99
        self.assertEqual(catalog(goal),catalog(changed['context']['user_goal']))
        self.assertEqual(request(self.config,self.item,'quotes'),original_request(self.config,self.item,'evidence_first'))
        self.assertEqual(request(self.config,self.item,'citations')['messages'][1],original_request(self.config,self.item,'evidence_first')['messages'][1])

    def test_resolve_keeps_labels_confidence_verdict_and_raw_evidence_without_repair(self):
        raw=self.citation_decision();before=copy.deepcopy(raw)
        resolved=resolve(self.item['context'],raw,'citations')
        self.assertEqual(raw,before);self.assertEqual(resolved['verdict'],raw['verdict'])
        self.assertEqual(resolved['goal_consistent']['label'],raw['goal_consistent']['label'])
        self.assertEqual(resolved['goal_consistent']['evidence']['steps'][0]['effect_quote'],catalog(self.item['context']['user_goal'])[3]['text'])
        for illegal in (True,999,'3'):
            changed=copy.deepcopy(raw);changed['goal_consistent']['evidence']['steps'][0]['effect_ref']=illegal
            with self.assertRaises(ValueError):resolve(self.item['context'],changed,'citations')

    def test_valid_reference_can_still_have_wrong_parameter_or_unresolved_phase(self):
        value=resolve(self.item['context'],self.citation_decision(),'citations')
        row=value['goal_consistent']['evidence']['steps'][1]
        row.update(effect_quote='increase fan to 80%',effect_relation='agrees',requested_quote='24 minutes from now')
        result=binding_checks(self.item['context'],value)
        self.assertTrue(result[0]['agreement_conflict'])  # Proposed60 vs quoted80.
        row['effect_relation']='conflict';self.assertFalse(binding_checks(self.item['context'],value)[0]['agreement_conflict'])
        row.update(effect_quote='22 minutes after the previous action',effect_relation='unknown')
        result=binding_checks(self.item['context'],value)
        self.assertEqual(result[0]['status'],'no_explicit_percentage');self.assertTrue(result[0]['offset_binding_needs_review'])
        self.assertIn('not independent',result[0]['scope'])

    def test_schema_references_only_public_catalog_not_selected_truth_or_labels(self):
        value=schema(self.item['context'],'citations')['json_schema']['schema']['properties']
        fields=value['goal_consistent']['properties']['evidence']['properties']['steps']['prefixItems'][0]['properties']
        self.assertNotIn('effect_quote',fields);self.assertEqual(fields['effect_ref']['anyOf'][0]['enum'],[row['id'] for row in catalog(self.item['context']['user_goal'])])
        self.assertNotIn('const',value['goal_consistent']['properties']['label'])
        self.assertEqual(list(value['correct_target']['properties']),['evidence','label','probability'])

    def test_valid_citation_with_wrong_agreement_cannot_pass_screen_and_cost_keeps_failures(self):
        config={**self.config,'records':1,'new_requests':2,'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
                'developer_counts':{'CONSISTENT_CONTROL':1},'developer_conflict_groups':{'target':[],'time':[]}}
        raw=self.citation_decision();decision=resolve(self.item['context'],raw,'citations')
        call=self.fixture.row('evidence_first')['call']
        call['text']=json.dumps({name:raw[name] for name in ('correct_target','goal_consistent','trajectory_consistent','safe_to_execute')})
        rows=[{'id':self.item['id'],'arm':arm,'decision':decision,'call':call,'parse_error':None} for arm in ARMS]
        inputs=[self.item];labels=[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}]
        result=evaluate(config,inputs,rows,[rows[0]],labels)
        self.assertTrue(result['complete']);self.assertEqual(result['new_tokens'],10)
        self.assertEqual(result['arms']['citations']['percentage_agreement_conflict_records'],1)
        self.assertFalse(result['diagnostic_screen_passed']['citations'])
        changed=copy.deepcopy(rows);changed[-1].update(decision=None,parse_error='unknown reference')
        result=evaluate(config,inputs,changed,[rows[0]],labels)
        self.assertEqual(result['arms']['citations']['tokens'],5);self.assertEqual(result['arms']['citations']['resolution_errors'],1)
        self.assertFalse(result['native_admitted'])


if __name__=='__main__':unittest.main()
