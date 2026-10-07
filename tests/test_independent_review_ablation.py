import copy
import json
import unittest

from smarthome_agent_rl.evidence_order_ablation import DIMENSIONS
from smarthome_agent_rl.independent_review_ablation import (COMPONENTS,FOCUS_PROMPT,request,assemble,costs,evaluate)
from scripts.run_independent_review_ablation import build_record
from tests import test_partitioned_goal_ablation as fixtures


class IndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PartitionedGoalTests();self.fixture.setUp()
        self.context=self.fixture.context;self.item=self.fixture.item
        self.context['user_goal']=self.context['user_goal'].replace('50%','60%')
        self.config={**self.fixture.config,'records':1,'new_requests':5,'review_records':2,
            'dimension_tokens':dict(zip(DIMENSIONS,[512,1024,256,256])),
            'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,'developer_counts':{'CONSISTENT_CONTROL':1},
            'developer_conflict_groups':{'time':[],'target':[]}}

    def calls(self):
        raw=self.fixture.raw();text=self.fixture.text(raw)
        value=json.loads(text);call=copy.deepcopy(self.fixture.fixture.fixture.fixture.fixture.row('evidence_first')['call'])
        return {component:{**call,'text':text if component=='joint' else json.dumps({component:value[component]})}
            for component in COMPONENTS}

    def test_isolated_requests_keep_full_context_schema_subtree_generation_and_total_budget(self):
        joint=request(self.config,self.item,'joint')
        for name in DIMENSIONS:
            child=request(self.config,self.item,name)
            self.assertEqual(child['response_format']['json_schema']['schema']['properties'],
                {name:joint['response_format']['json_schema']['schema']['properties'][name]})
            self.assertEqual(child['response_format']['json_schema']['schema']['required'],[name])
            child['messages'][0]['content']=child['messages'][0]['content'].removesuffix('\n'+FOCUS_PROMPT.format(name=name))
            child['response_format']=joint['response_format'];child['max_tokens']=joint['max_tokens']
            self.assertEqual(child,joint)
        self.assertEqual(sum(self.config['dimension_tokens'].values()),joint['max_tokens'])

    def test_assembly_preserves_wrong_label_and_relation_no_repair(self):
        calls=self.calls();v=json.loads(calls['goal_consistent']['text']);v['goal_consistent']['evidence']['steps'][0]['relation']='conflict'
        calls['goal_consistent']['text']=json.dumps(v);before=copy.deepcopy(calls)
        value,text,model,decision,error=assemble(self.context,{name:calls[name] for name in DIMENSIONS},'independent')
        self.assertIsNone(error);self.assertEqual(model['goal_consistent']['label'],'YES')
        self.assertEqual(decision['goal_consistent']['evidence']['steps'][0]['relation'],'conflict')
        self.assertEqual(value['goal_consistent'],v['goal_consistent']);self.assertEqual(calls,before)
        self.assertEqual(model['verdict'],decision['verdict'])

    def test_missing_duplicate_cross_scope_and_http_failure_keep_original_cost(self):
        for change in [lambda c:c['correct_target'].update(text='{}'),
            lambda c:c['correct_target'].update(text=c['joint']['text']),
            lambda c:c['goal_consistent'].update(text=c['goal_consistent']['text'].replace('"goal_consistent":','"goal_consistent": {}, "goal_consistent":',1)),
            lambda c:c['safe_to_execute'].update(error={'type':'TimeoutError'})]:
            calls=self.calls();change(calls);selected={name:calls[name] for name in DIMENSIONS}
            self.assertIsNone(assemble(self.context,selected,'independent')[3]);self.assertIsNotNone(assemble(self.context,selected,'independent')[4])
            self.assertEqual(costs(list(selected.values()))['total_tokens'],20)
        calls=self.calls();calls['safe_to_execute']['usage']=None
        result=costs([calls[name] for name in DIMENSIONS]);self.assertEqual(result['missing_usage'],1);self.assertEqual(result['total_tokens'],15)

    def rows(self):
        calls=self.calls();wrappers={name:{'id':self.item['id'],'component':name,'actor_id':0,
            'submitted_monotonic':10.,'dispatch_monotonic':11.,'finished_monotonic':13.,'call':call} for name,call in calls.items()}
        rows=[]
        for arm,names in [('joint',('joint',)),('independent',DIMENSIONS)]:
            stored,row=build_record(self.item,0,arm,{name:wrappers[name] for name in names})
            self.assertNotIn('call',stored);self.assertNotIn('calls',stored);self.assertEqual(row['review_seconds'],3)
            rows.append(row)
        return rows

    def test_denominators_raw_call_cost_and_old_gates_preserved(self):
        rows=self.rows();labels=[{'id':self.item['id'],'category':'CONSISTENT_CONTROL'}]
        joint=self.calls()['joint'];_,_,model,decision,error=assemble(self.context,{'joint':joint},'joint')
        historical=[{'id':self.item['id'],'call':joint,'model_decision':model,'decision':decision,'parse_error':error}]
        result=evaluate(self.config,[self.item],rows,historical,labels)
        self.assertTrue(result['complete']);self.assertEqual(result['new_model_requests'],5);self.assertEqual(result['new_tokens'],25)
        self.assertEqual(result['arms']['independent']['actual_http_requests'],4)
        self.assertTrue(result['diagnostic_screen_passed']['independent']);self.assertFalse(result['native_admitted'])
        rows[1]['calls']['safe_to_execute']['usage']=None
        result=evaluate(self.config,[self.item],rows,historical,labels)
        self.assertEqual(result['new_tokens'],20);self.assertEqual(result['arms']['independent']['missing_usage'],1)
        self.assertFalse(result['diagnostic_screen_passed']['independent'])
        self.assertFalse(evaluate(self.config,[self.item],rows[:1],historical,labels)['complete'])


if __name__=='__main__':unittest.main()
