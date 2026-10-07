import copy
import unittest

from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_context_ablation import ARMS, enrich, evaluate
from smarthome_agent_rl.semantic_diagnosis import contexts


class ContextSemanticAblationTests(unittest.TestCase):
    def setUp(self):
        self.public={'query':'Set fan40,then60 after29 minutes.','user_location':'living','current_time':'2030-01-01 12:00:00'}
        self.proposal={'tool':'schedule_workflow','arguments':{'start_time':'2030-01-01 12:01:00','steps':[]},
            'reached_executor':True,'blocked':False,'actual_calls_before':1,'actual_calls':1,'action_id':'a1'}
        self.observations=[{'tool':'get_device_structure','arguments':{'device_id':'washer'},'turn':1,
            'response':{'status':{'code':200},'data':{'device_id':'washer','countdown':600}}},
            {'tool':'schedule_workflow','arguments':self.proposal['arguments'],'turn':2,
             'response':{'status':{'code':200},'data':{'outcome':'FORBIDDEN_CURRENT_RESULT'}}},
            {'tool':'get_device_structure','arguments':{'device_id':'dryer'},'turn':3,
             'response':{'status':{'code':200},'data':{'device_id':'dryer','future':'FORBIDDEN_FUTURE'}}}]
        values,cutoff=contexts(self.public,self.proposal,self.observations)
        self.item={'id':'case','contexts':values,'dispatch_observation_index':cutoff,
                   'preaction_receipts_sha256':digest(self.observations[:cutoff])}
        self.rules={'source_sha256':{},'facts':{'per_step_wait_supported':False}}

    def test_real_cutoff_excludes_current_result_and_future_reference(self):
        result=enrich(self.item,self.public,self.proposal,self.observations,self.rules)
        for context in result['contexts'].values():
            self.assertNotIn('FORBIDDEN',str(context))
            self.assertNotIn('dryer',str(context))
        self.assertEqual(result['contexts']['references']['environment_state']['observed_reference_receipts']['devices'][0]['device_id'],'washer')

    def test_workflow_and_reference_changes_are_independent(self):
        result=enrich(self.item,self.public,self.proposal,self.observations,self.rules)
        for suffix in ('references','workflow','both'):
            state=result['contexts'][suffix]['environment_state']
            self.assertEqual('observed_reference_receipts' in state,suffix in ('references','both'))
            self.assertEqual('workflow_execution_semantics' in state,suffix in ('workflow','both'))
            self.assertEqual(result['contexts'][suffix]['proposed_action'],self.item['contexts']['public']['proposed_action'])

    def test_forged_parent_context_rejected_even_with_new_input_digest(self):
        changed=copy.deepcopy(self.item)
        changed['contexts']['public']['environment_state']['forged_result']='success'
        with self.assertRaisesRegex(ValueError,'boundary'):enrich(changed,self.public,self.proposal,self.observations,self.rules)

    def test_wrong_dispatch_and_receipt_identity_rejected(self):
        for field,value in (('dispatch_observation_index',2),('preaction_receipts_sha256','forged')):
            changed={**self.item,field:value}
            with self.assertRaises(ValueError):enrich(changed,self.public,self.proposal,self.observations,self.rules)

    def results(self):
        categories={'conflict':'CERTAIN_CONFLICT','control':'CONSISTENT_CONTROL','uncertain':'UNCERTAIN'}
        config={'proposals':3,'valid_ratio_min':.95,'developer_counts':{v:1 for v in categories.values()}}
        labels={'cases':[{'id':key,'category':value} for key,value in categories.items()]}
        def row(identity,arm,verdict):return {'id':identity,'arm':arm,'decision':{'verdict':verdict},'parse_error':None,
            'call':{'error':None,'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5},'request_seconds':.1}}
        baseline=[row(i,'9b_public','ALLOW') for i in categories]
        records=[row(i,a,'DENY' if i=='conflict' else 'ALLOW') for i in categories for a in ARMS]
        return config,records,baseline,labels

    def test_historical_tokens_not_charged_as_new_requests(self):
        config,records,baseline,labels=self.results()
        result=evaluate(config,records,baseline,labels)
        self.assertEqual(result['new_model_tokens'],45)
        self.assertEqual(result['arms']['9b_public']['tokens'],15)
        self.assertEqual(result['new_model_requests'],9)
        self.assertTrue(all(result['developer_case_gate'].values()))
        self.assertFalse(result['native_integration_admitted'])

    def test_invalid_control_not_excluded_and_missing_request_fails(self):
        config,records,baseline,labels=self.results()
        records[3]['decision']=None;records[3]['parse_error']='length'
        result=evaluate(config,records,baseline,labels)
        self.assertFalse(result['developer_case_gate']['9b_references'])
        self.assertEqual(result['arms']['9b_references']['developer_controls'],{'INVALID':1})
        self.assertFalse(evaluate(config,records[:-1],baseline,labels)['checks']['complete_records'])


if __name__=='__main__':unittest.main()
