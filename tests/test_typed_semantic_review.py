import copy
import unittest

from tests import test_prompt_semantic_ablation as prompt_fixture
from smarthome_agent_rl.semantic_diagnosis import schema as original_schema
from smarthome_agent_rl.semantic_prompt_ablation import request as prompt_request
from smarthome_agent_rl.typed_semantic_review import schema,request,schema_structure_check,evaluate


class TypedSemanticReviewTests(unittest.TestCase):
    def setUp(self):
        fixture=prompt_fixture.PromptSemanticAblationTests('test_literal_evidence_is_not_claimed_to_prove_entailment')
        fixture.setUp();self.item=fixture.item;self.config=fixture.config;self.row=fixture.valid_row()
        self.row['arm']='9b_typed'

    def test_schema_binds_only_proposed_metadata_and_keeps_all_verdicts_available(self):
        value=schema(self.item['context']);original=original_schema()
        properties=value['json_schema']['schema']['properties']
        for name in properties:
            self.assertEqual(properties[name]['properties']['label'],original['json_schema']['schema']['properties'][name]['properties']['label'])
            self.assertEqual(properties[name]['properties']['probability'],original['json_schema']['schema']['properties'][name]['properties']['probability'])
        rows=properties['correct_target']['properties']['evidence']['properties']['steps']
        self.assertFalse(rows['items']);self.assertEqual(rows['minItems'],2);self.assertEqual(rows['maxItems'],2)
        self.assertEqual(rows['prefixItems'][1]['properties']['step_index']['const'],2)
        self.assertEqual(rows['prefixItems'][1]['properties']['device_id']['const'],'lamp')
        self.assertEqual(properties['trajectory_consistent']['properties']['evidence'],{'type':'object'})
        self.assertEqual(properties['safe_to_execute']['properties']['evidence'],{'type':'object'})
        self.assertEqual(original_schema(),original)
        changed=copy.deepcopy(self.item['context']);changed['user_goal']='A different requested outcome.'
        self.assertEqual(value,schema(changed))

    def test_request_changes_only_response_schema(self):
        original=prompt_request(self.config,self.item,'9b_stepwise');candidate=request(self.config,self.item)
        self.assertEqual({k:v for k,v in original.items() if k!='response_format'},
                         {k:v for k,v in candidate.items() if k!='response_format'})
        self.assertNotEqual(original['response_format'],candidate['response_format'])

    def test_provider_ignoring_const_index_or_extra_fields_is_detected(self):
        self.assertTrue(schema_structure_check(self.item,self.row))
        changed=copy.deepcopy(self.row);changed['decision']['correct_target']['evidence']['steps'][1]['step_index']=0
        self.assertFalse(schema_structure_check(self.item,changed))
        changed=copy.deepcopy(self.row);changed['decision']['goal_consistent']['evidence']['steps'][0]['hidden_extra']='anything'
        self.assertFalse(schema_structure_check(self.item,changed))

    def test_literal_quote_failure_is_separate_from_schema_conformance(self):
        changed=copy.deepcopy(self.row)
        changed['decision']['goal_consistent']['evidence']['steps'][0]['requested_quote']='invented clause'
        self.assertTrue(schema_structure_check(self.item,changed))
        config={**self.config,'proposals':1,'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
                'developer_counts':{'CERTAIN_CONFLICT':1}}
        labels={'cases':[{'id':'workflow','category':'CERTAIN_CONFLICT'}]}
        baseline=[{**self.row,'arm':'9b_stepwise'}]
        result=evaluate(config,[self.item],[changed],baseline,labels)
        self.assertTrue(result['checks']['schema_structure']);self.assertFalse(result['diagnostic_screen_passed'])
        self.assertEqual(result['new_model_tokens'],5)
        self.assertFalse(result['native_integration_admitted'])


if __name__=='__main__':unittest.main()
