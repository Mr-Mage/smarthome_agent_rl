import copy
import json
import unittest

from smarthome_agent_rl.evidence_reducer_ablation import (ARMS, PROTOCOL_PROMPT,
    request, parse_response, wire_check, evaluate)
from tests import test_partitioned_goal_ablation as fixtures


class EvidenceReducerTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PartitionedGoalTests();self.fixture.setUp()
        self.context=self.fixture.context;self.item=self.fixture.item
        self.config={**self.fixture.config,'records':1,'new_requests':2,
                     'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
                     'developer_counts':{'CONSISTENT_CONTROL':1},'developer_conflict_groups':{'time':[],'target':[]}}

    def raw(self):
        value=json.loads(self.fixture.text(self.fixture.raw()))
        for row in value.values():row.pop('label')
        return value

    def test_context_rules_sampling_and_evidence_subtrees_preserved(self):
        left=request(self.config,self.item,'joint');right=request(self.config,self.item,'evidence_only')
        right['messages'][0]['content']=right['messages'][0]['content'].removesuffix('\n'+PROTOCOL_PROMPT)
        for name,row in right['response_format']['json_schema']['schema']['properties'].items():
            original=left['response_format']['json_schema']['schema']['properties'][name]
            self.assertEqual(row['properties'],{k:v for k,v in original['properties'].items() if k!='label'})
        right['response_format']=left['response_format'];self.assertEqual(right,left)

    def test_claim_reduction_is_explicit_preserves_wrong_claims_and_model_confidence(self):
        value=self.raw();value['goal_consistent']['evidence']['steps'][0]['effect_relation']='conflict'
        value['goal_consistent']['probability']=.17;before=copy.deepcopy(value)
        raw,decision,error=parse_response(self.context,json.dumps(value),'evidence_only')
        self.assertIsNone(error);self.assertEqual(raw,before);self.assertNotIn('label',raw['goal_consistent'])
        self.assertEqual(decision['goal_consistent']['label'],'NO');self.assertEqual(decision['verdict'],'DENY')
        self.assertEqual(decision['goal_consistent']['probability'],.17)
        self.assertEqual(decision['goal_consistent']['evidence']['steps'][0]['effect_relation'],'conflict')
        # An incorrect claim still produces NO; reduction never establishes truth.

    def test_unknown_and_not_applicable_are_distinct_conflict_has_priority(self):
        value=self.raw();value['safe_to_execute']['evidence']['steps'][0]['relation']='unknown'
        self.assertEqual(parse_response(self.context,json.dumps(value),'evidence_only')[1]['safe_to_execute']['label'],'UNCERTAIN')
        value['safe_to_execute']['evidence']['steps'][0]['relation']='not_applicable'
        self.assertEqual(parse_response(self.context,json.dumps(value),'evidence_only')[1]['safe_to_execute']['label'],'YES')
        value['goal_consistent']['evidence']['steps'][0].update(relation='unknown',effect_relation='conflict')
        self.assertEqual(parse_response(self.context,json.dumps(value),'evidence_only')[1]['goal_consistent']['label'],'NO')

    def test_malformed_scope_refs_metadata_booleans_nonfinite_and_duplicates_fail(self):
        for change in [lambda v:v['goal_consistent'].update(label='YES'),
                       lambda v:v['safe_to_execute'].update(probability=True),
                       lambda v:v['safe_to_execute'].update(probability=float('nan')),
                       lambda v:v['safe_to_execute']['evidence']['steps'][0].update(public_refs=['/hidden']),
                       lambda v:v['goal_consistent']['evidence']['steps'][0].update(effect_ref=True),
                       lambda v:v['goal_consistent']['evidence']['steps'][0].update(step_index=True),
                       lambda v:v['goal_consistent']['evidence']['steps'].pop()]:
            value=self.raw();change(value);text=json.dumps(value)
            self.assertIsNone(parse_response(self.context,text,'evidence_only')[1])
            self.assertFalse(wire_check(self.context,text,'evidence_only')['schema'])
        text=json.dumps(self.raw()).replace('"probability":','"probability": 0, "probability":',1)
        self.assertIsNotNone(parse_response(self.context,text,'evidence_only')[2])

    def rows(self):
        rows=[]
        for arm,text in [('joint',self.fixture.text(self.fixture.raw())),('evidence_only',json.dumps(self.raw()))]:
            model,decision,error=parse_response(self.context,text,arm)
            call={'text':text,'error':None,'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},
                  'request_seconds':1,'finish_reason':'stop'}
            rows.append({'id':self.item['id'],'arm':arm,'call':call,'model_decision':model,'decision':decision,'parse_error':error})
        return rows

    def test_original_reason_literal_cost_and_failure_gates_remain(self):
        rows=self.rows();labels=[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}]
        report=evaluate(self.config,[self.item],rows,[rows[0]],labels)
        self.assertTrue(report['complete']);self.assertEqual(report['new_tokens'],30);self.assertFalse(report['native_admitted'])
        self.assertTrue(report['diagnostic_screen_passed']['evidence_only'])
        changed=copy.deepcopy(rows);value=self.raw();value['goal_consistent']['evidence']['steps'][0].update(effect_ref=None,effect_relation='agrees')
        changed[1]['call']['text']=json.dumps(value)
        changed[1]['model_decision'],changed[1]['decision'],changed[1]['parse_error']=parse_response(self.context,changed[1]['call']['text'],'evidence_only')
        self.assertFalse(evaluate(self.config,[self.item],changed,[rows[0]],labels)['diagnostic_screen_passed']['evidence_only'])
        changed=copy.deepcopy(rows);changed[1]['call']['usage']=None
        self.assertEqual(evaluate(self.config,[self.item],changed,[rows[0]],labels)['new_tokens'],15)
        self.assertFalse(evaluate(self.config,[self.item],changed,[rows[0]],labels)['diagnostic_screen_passed']['evidence_only'])
        self.assertFalse(evaluate(self.config,[self.item],rows[:1],[rows[0]],labels)['complete'])
        with self.assertRaises(ValueError):evaluate(self.config,[self.item],rows,[rows[0]],labels*2)


if __name__=='__main__':unittest.main()
