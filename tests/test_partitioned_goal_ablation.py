import copy
import json
import unittest

from smarthome_agent_rl.partitioned_goal_ablation import (ARMS, EFFECT_FIELDS, TIME_FIELDS,
    LAYOUT_PROMPT, actor_for, request, project, parse_response, evaluate)
from smarthome_agent_rl.review_factorial import request as original_request
from tests import test_bounded_review_ablation as fixtures


class PartitionedGoalTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.BoundedReviewTests();self.fixture.setUp()
        self.context=self.fixture.context;self.item=self.fixture.item
        self.config={**self.fixture.config,'actors':[{'id':i} for i in range(4)]}

    def raw(self):
        value=self.fixture.raw()
        for row in value['correct_target']['evidence']['steps']:row.pop('observed_identity')
        return value

    def grouped(self,value):
        result=copy.deepcopy(value);rows=result['goal_consistent']['evidence']['steps']
        result['goal_consistent']['evidence']={name:[{key:row[key] for key in fields} for row in rows]
            for name,fields in [('effect_steps',EFFECT_FIELDS),('time_steps',TIME_FIELDS)]}
        return result

    def text(self,value):
        return json.dumps({name:{key:value[name][key] for key in ('evidence','label','probability')}
            for name in ('correct_target','goal_consistent','trajectory_consistent','safe_to_execute')})

    def test_one_layout_factor_and_mechanical_note_only(self):
        left=request(self.config,self.item,'original');right=request(self.config,self.item,'partitioned')
        self.assertEqual(left,original_request(self.config,self.item,'plain_history'))
        right['messages'][0]['content']=right['messages'][0]['content'].removesuffix('\n'+LAYOUT_PROMPT)
        original=left['response_format']['json_schema']['schema']['properties']['goal_consistent']['properties']['evidence']
        changed=right['response_format']['json_schema']['schema']['properties']['goal_consistent']['properties']['evidence']
        self.assertEqual(list(changed['properties']),['effect_steps','time_steps'])
        for name,fields in [('effect_steps',EFFECT_FIELDS),('time_steps',TIME_FIELDS)]:
            for old,new in zip(original['properties']['steps']['prefixItems'],changed['properties'][name]['prefixItems']):
                self.assertEqual(new['properties'],{key:old['properties'][key] for key in fields})
        right['response_format']['json_schema']['schema']['properties']['goal_consistent']['properties']['evidence']=original
        self.assertEqual(left,right)
        for index in range(78):self.assertEqual(actor_for(self.config,index,'original'),actor_for(self.config,index,'partitioned'))

    def test_lossless_projection_preserves_wrong_or_uncertain_predicted_relations(self):
        raw=self.raw();raw['goal_consistent']['evidence']['steps'][0]['relation']='conflict'
        raw['goal_consistent']['label']='YES'
        grouped=self.grouped(raw);before=copy.deepcopy(grouped)
        self.assertEqual(project(grouped,len(raw['goal_consistent']['evidence']['steps'])),raw)
        model,decision,error=parse_response(self.context,self.text(grouped),'partitioned')
        self.assertIsNone(error);self.assertEqual(model['goal_consistent']['evidence'],grouped['goal_consistent']['evidence'])
        self.assertEqual(decision['goal_consistent']['label'],'YES')
        self.assertEqual(decision['goal_consistent']['evidence']['steps'][0]['relation'],'conflict')
        self.assertEqual(model['verdict'],decision['verdict']);self.assertEqual(grouped,before)

    def test_misaligned_missing_duplicate_and_extra_groups_fail_without_repair(self):
        for change in [lambda v:v['effect_steps'].pop(),lambda v:v['time_steps'][0].update(step_index=True),
            lambda v:v['effect_steps'][0].update(step_index=2),lambda v:v.update(hidden=[]),
            lambda v:v['effect_steps'][0].update(effect_ref=99999)]:
            raw=self.grouped(self.raw());change(raw['goal_consistent']['evidence'])
            model,decision,error=parse_response(self.context,self.text(raw),'partitioned')
            self.assertIsNotNone(model);self.assertIsNone(decision);self.assertIsNotNone(error)
        text=self.text(self.grouped(self.raw())).replace('"effect_steps":','"effect_steps": [], "effect_steps":',1)
        self.assertIsNone(parse_response(self.context,text,'partitioned')[1])

    def test_all_original_gates_denominators_and_costs_retained(self):
        self.context['user_goal']=self.context['user_goal'].replace('50%','60%')
        raw=self.raw();grouped=self.grouped(raw)
        call=copy.deepcopy(self.fixture.fixture.fixture.fixture.row('evidence_first')['call'])
        rows=[]
        for arm,value in zip(ARMS,[raw,grouped]):
            text=self.text(value);model,decision,error=parse_response(self.context,text,arm)
            rows.append({'id':self.item['id'],'arm':arm,'call':{**call,'text':text},'model_decision':model,'decision':decision,'parse_error':error})
        config={**self.config,'records':1,'new_requests':2,'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
            'developer_counts':{'CONSISTENT_CONTROL':1},'developer_conflict_groups':{'time':[],'target':[]}}
        labels=[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}]
        result=evaluate(config,[self.item],rows,[rows[0]],labels)
        self.assertTrue(result['complete']);self.assertFalse(result['native_admitted']);self.assertEqual(result['new_tokens'],10)
        self.assertTrue(result['diagnostic_screen_passed']['partitioned'])
        rows[1]={**rows[1],'decision':None,'parse_error':'retained failure'}
        result=evaluate(config,[self.item],rows,[rows[0]],labels)
        self.assertEqual(result['new_tokens'],10);self.assertFalse(result['diagnostic_screen_passed']['partitioned'])
        self.assertFalse(evaluate(config,[self.item],rows[:1],[rows[0]],labels)['complete'])


if __name__=='__main__':unittest.main()
