import copy
import json
import unittest

from smarthome_agent_rl.identity_presentation_ablation import identity_panel,request,schema,evaluate,ARMS
from smarthome_agent_rl.citation_review import request as citation_request,resolve
from tests import test_citation_review as fixtures


class IdentityPresentationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CitationReviewTests();self.fixture.setUp()
        self.context=self.fixture.item['context'];self.item=self.fixture.item;self.config=self.fixture.config

    def test_only_appended_public_identity_changes_request_not_schema_prompt_context_or_catalog(self):
        original=citation_request(self.config,self.item,'citations')
        self.assertEqual(request(self.config,self.item,'control'),original)
        changed=request(self.config,self.item,'identity');extra=changed['messages'].pop()
        self.assertEqual(changed,original)
        self.assertEqual(json.loads(extra['content']),identity_panel(self.context))
        self.assertEqual(schema(self.context,'control'),schema(self.context,'identity'))

    def test_unobserved_device_keeps_null_and_never_decodes_device_id_or_goal(self):
        self.context['environment_state']['devices']={}
        self.context['proposed_action']['steps'][0]['args']['device_id']='bathroom_air_purifier_1'
        panel=identity_panel(self.context)
        self.assertIsNone(panel['public_proposed_device_identity_receipts'][0]['observed_identity'])
        changed=copy.deepcopy(self.context);changed['user_goal']='Contradictory unrelated goal'
        self.assertEqual(identity_panel(changed),panel)
        self.assertIn('not natural-language identity proofs',panel['scope'])

    def test_multiple_catalogs_and_missing_type_preserve_provenance_and_ambiguity(self):
        device=self.context['proposed_action']['steps'][0]['args']['device_id']
        self.context['environment_state']['devices']={device:{'room_id':None,'catalog_total':2,'catalog_truncated':True,
            'catalog':[{'room_id':'one','metadata':{'device_type':'fan'},'source':{'response_sha256':'a'}},
                       {'room_id':'two','metadata':{},'source':{'response_sha256':'b'}}],
            'structure':{'device_type':'fan'},'structure_source':{'response_sha256':'c'},'state_freshness':'UNKNOWN'}}
        before=copy.deepcopy(self.context);value=identity_panel(self.context)['public_proposed_device_identity_receipts'][0]['observed_identity']
        self.assertEqual(self.context,before);self.assertIsNone(value['room_id']);self.assertTrue(value['catalog_truncated'])
        self.assertEqual([r['source']['response_sha256'] for r in value['catalog']],['a','b'])
        self.assertIsNone(value['catalog'][1]['device_type']);self.assertEqual(value['structure_source']['response_sha256'],'c')

    def test_gate_retains_bad_percentage_missing_outputs_and_cost_no_identity_truth_override(self):
        config={**self.config,'records':1,'new_requests':2,'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
            'developer_counts':{'CONSISTENT_CONTROL':1},'developer_conflict_groups':{'target':[],'time':[]}}
        raw=self.fixture.citation_decision();decision=resolve(self.context,raw,'citations')
        call=copy.deepcopy(self.fixture.fixture.row('evidence_first')['call'])
        call['text']=json.dumps({name:raw[name] for name in ('correct_target','goal_consistent','trajectory_consistent','safe_to_execute')})
        rows=[{'id':self.item['id'],'arm':arm,'decision':decision,'model_decision':raw,'call':call,'parse_error':None} for arm in ARMS]
        labels=[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}]
        report=evaluate(config,[self.item],rows,[rows[0]],labels)
        self.assertTrue(report['complete']);self.assertEqual(report['new_tokens'],10)
        self.assertFalse(report['diagnostic_screen_passed']['identity']);self.assertFalse(report['native_admitted'])
        changed=copy.deepcopy(rows);changed[-1].update(decision=None,model_decision=None,parse_error='bad')
        report=evaluate(config,[self.item],changed,[rows[0]],labels)
        self.assertEqual(report['arms']['identity']['records'],1);self.assertEqual(report['arms']['identity']['tokens'],5)
        self.assertEqual(report['binding_coverage']['identity']['record_flags']['unavailable_schema'],1)


if __name__=='__main__':unittest.main()
