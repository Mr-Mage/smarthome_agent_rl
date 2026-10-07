import copy
import json
import unittest

from scripts.audit_evidence_order_ablation import verify_record
from smarthome_agent_rl.evidence_order_ablation import ARMS,DIMENSIONS,ordered_digest,request,schema,output_order,evaluate
from smarthome_agent_rl.effect_evidence_ablation import request as original_request
from smarthome_agent_rl.semantic_context import digest
from tests import test_effect_evidence_audit as fixtures


class EvidenceOrderAblationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EffectEvidenceAuditTests();self.fixture.setUp()
        self.config=self.fixture.config;self.item=self.fixture.item;self.actor=self.fixture.actor

    def row(self,arm):
        row=copy.deepcopy(self.fixture.row);row['arm']=arm
        row['call']['body']=request(self.config,self.item,arm)
        content=json.loads(row['call']['text'])
        if arm=='evidence_first':
            content={name:{key:content[name][key] for key in ('evidence','label','probability')} for name in DIMENSIONS}
        row['call']['text']=json.dumps(content)
        row['call']['response']['choices'][0]['message']['content']=row['call']['text']
        row['call']['raw_response']=json.dumps(row['call']['response'])
        return row

    def test_only_property_order_changes_semantics_prompt_context_and_constraints_match(self):
        original=original_request(self.config,self.item,'combined')
        for arm in ARMS:self.assertEqual(request(self.config,self.item,arm),original)
        control=schema(self.item['context'],'label_first');candidate=schema(self.item['context'],'evidence_first')
        self.assertEqual(control,candidate);self.assertEqual(digest(control),digest(candidate))
        self.assertNotEqual(ordered_digest(control),ordered_digest(candidate))
        self.assertEqual(list(control['json_schema']['schema']['properties']['correct_target']['properties']),['label','probability','evidence'])
        self.assertEqual(list(candidate['json_schema']['schema']['properties']['correct_target']['properties']),['evidence','label','probability'])

    def test_auditor_rejects_reordered_body_even_when_dictionary_equality_passes(self):
        row=self.row('evidence_first');verify_record(row,self.item,'evidence_first',self.actor,self.config)
        changed=copy.deepcopy(row);changed['call']['body']=request(self.config,self.item,'label_first')
        self.assertEqual(row['call']['body'],changed['call']['body'])
        with self.assertRaises(ValueError):verify_record(changed,self.item,'evidence_first',self.actor,self.config)
        changed=copy.deepcopy(row);changed['decision']['verdict']='DENY'
        with self.assertRaises(ValueError):verify_record(changed,self.item,'evidence_first',self.actor,self.config)

    def test_raw_output_not_schema_confirms_actual_order_and_duplicates_are_invalid(self):
        self.assertTrue(output_order(self.row('evidence_first')['call']['text'])['evidence_first'])
        self.assertTrue(output_order(self.row('label_first')['call']['text'])['label_first'])
        text=self.row('evidence_first')['call']['text'].replace('"label": "YES"','"label": "NO", "label": "YES"',1)
        self.assertFalse(output_order(text)['valid'])
        self.assertFalse(output_order('incomplete')['valid'])

    def test_cost_denominator_order_compliance_and_failures_are_not_selected_away(self):
        config={**self.config,'records':1,'new_requests':2,'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
                'developer_counts':{'CONSISTENT_CONTROL':1},'developer_conflict_groups':{'target':[],'time':[]}}
        inputs=[{**self.item,'origin':'n76'}];rows=[self.row(arm) for arm in ARMS]
        historical=[self.fixture.row];labels=[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}]
        value=evaluate(config,inputs,rows,historical,labels)
        self.assertTrue(value['complete']);self.assertTrue(all(value['diagnostic_screen_passed'].values()))
        self.assertEqual(value['new_tokens'],10);self.assertFalse(value['native_admitted'])
        rows[-1]['call']['text']=rows[0]['call']['text']
        value=evaluate(config,inputs,rows,historical,labels)
        self.assertFalse(value['diagnostic_screen_passed']['evidence_first'])
        rows[-1]['decision']=None;rows[-1]['call']['error']='timeout'
        value=evaluate(config,inputs,rows,historical,labels)
        self.assertEqual(value['arms']['evidence_first']['tokens'],5)
        self.assertEqual(value['arms']['evidence_first']['verdicts'],{'INVALID':1})
        self.assertFalse(evaluate(config,inputs,rows[:-1],historical,labels)['complete'])


if __name__=='__main__':unittest.main()
