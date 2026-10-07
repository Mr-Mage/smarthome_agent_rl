import copy
import json
import unittest

from smarthome_agent_rl.bounded_review_ablation import request,schema,reference_catalog,parse_response,evaluate
from smarthome_agent_rl.identity_evidence_ablation import request as original_request
from tests import test_identity_evidence_ablation as fixtures


class BoundedReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.IdentityEvidenceTests();self.fixture.setUp()
        self.context=self.fixture.context;self.item=self.fixture.item;self.config=self.fixture.config
        device=next(iter(self.context['environment_state']['devices']))
        self.context['environment_state']['devices'][device]['catalog'][0]['source']['response_sha256']='a'*64

    def raw(self):
        raw=self.fixture.raw();pointer=reference_catalog(self.context)[0]['pointer']
        for name in ('trajectory_consistent','safe_to_execute'):
            raw[name]['evidence']={'steps':[{'step_index':index+1,'public_refs':[pointer],'relation':'agrees'}
                for index in range(len(raw['correct_target']['evidence']['steps']))]}
        return raw

    def test_only_two_evidence_schemas_change_no_predicted_label_or_reference_truth_const(self):
        original=original_request(self.config,self.item,'identity');self.assertEqual(request(self.config,self.item,'control'),original)
        changed=request(self.config,self.item,'bounded');value=changed['response_format']['json_schema']['schema']['properties']
        source=original['response_format']['json_schema']['schema']['properties']
        for name in ('correct_target','goal_consistent'):self.assertEqual(value[name],source[name])
        for name in ('trajectory_consistent','safe_to_execute'):
            self.assertEqual(value[name]['properties']['label'],source[name]['properties']['label'])
            value[name]['properties']['evidence']=source[name]['properties']['evidence']
        self.assertEqual(changed,original)

    def test_catalog_only_includes_visible_receipts_with_no_device_id_decoding_or_goal_matching(self):
        before=copy.deepcopy(self.context);refs=reference_catalog(self.context)
        self.assertEqual(self.context,before);self.assertEqual(len(refs),1)
        self.assertEqual(refs[0]['receipt']['response_sha256'],'a'*64)
        changed=copy.deepcopy(self.context);changed['user_goal']='Another device in another room'
        self.assertEqual(reference_catalog(changed),refs)
        changed['environment_state']['devices']={};self.assertEqual(reference_catalog(changed),[])
        fields=schema(changed,'bounded')['json_schema']['schema']['properties']['safe_to_execute']['properties']['evidence']['properties']['steps']['prefixItems'][0]['properties']
        self.assertEqual(fields['public_refs']['maxItems'],0)

    def test_strict_duplicate_rejection_is_common_to_control_and_bounded_at_any_depth(self):
        text=self.fixture.text(self.raw());text=text.replace('"relation": "agrees"','"relation": "agrees", "relation": "unknown"',1)
        for arm in ('control','bounded'):
            model,decision,error=parse_response(self.context,text,arm)
            self.assertIsNone(model);self.assertIsNone(decision);self.assertIn('Duplicate JSON key',error)

    def test_bounded_reference_and_denominator_failures_are_retained_without_repair(self):
        for change in (lambda r:r.update(public_refs=['/hidden/evaluator']),lambda r:r.update(public_refs=['not-a-pointer']),
                       lambda r:r.update(public_refs=[reference_catalog(self.context)[0]['pointer']]*4),
                       lambda r:r.update(step_index=True),lambda r:r.update(explanation='expanded text')):
            raw=self.raw();change(raw['safe_to_execute']['evidence']['steps'][0])
            model,decision,error=parse_response(self.context,self.fixture.text(raw),'bounded')
            self.assertIsNotNone(model);self.assertIsNone(decision);self.assertIsNotNone(error)
        raw=self.raw();raw['trajectory_consistent']['evidence']['steps'].pop()
        self.assertIsNone(parse_response(self.context,self.fixture.text(raw),'bounded')[1])

    def test_relation_label_disagreement_remains_original_and_missing_outputs_keep_cost(self):
        raw=self.raw();raw['safe_to_execute']['evidence']['steps'][0]['relation']='conflict'
        text=self.fixture.text(raw);model,decision,error=parse_response(self.context,text,'bounded')
        self.assertIsNone(error);self.assertEqual(decision['safe_to_execute']['label'],'YES')
        self.assertEqual(model['verdict'],decision['verdict']);self.assertEqual(model['safe_to_execute']['evidence'],decision['safe_to_execute']['evidence'])
        config={**self.config,'records':1,'new_requests':2,'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
            'developer_counts':{'CONSISTENT_CONTROL':1},'developer_conflict_groups':{'target':[],'time':[]}}
        call=copy.deepcopy(self.fixture.fixture.fixture.row('evidence_first')['call']);call['text']=text
        rows=[{'id':self.item['id'],'arm':arm,'call':call,'model_decision':model,'decision':decision,'parse_error':None} for arm in ('control','bounded')]
        result=evaluate(config,[self.item],rows,[rows[0]],[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}])
        self.assertEqual(result['bounded_dimensions']['safe_to_execute']['label_mismatch_records'],1)
        self.assertFalse(result['native_admitted']);self.assertEqual(result['new_tokens'],10)
        rows[1]={**rows[1],'model_decision':None,'decision':None,'parse_error':'invalid'}
        result=evaluate(config,[self.item],rows,[rows[0]],[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}])
        self.assertEqual(result['arms']['bounded']['records'],1);self.assertEqual(result['arms']['bounded']['tokens'],5)
        self.assertFalse(result['diagnostic_screen_passed']['bounded'])


if __name__=='__main__':unittest.main()
