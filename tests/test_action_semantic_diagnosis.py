import copy
import json
import unittest

from smarthome_agent_rl.semantic_diagnosis import contexts, messages, parse
from scripts.run_action_semantic_diagnosis import evaluate, ARMS


class ActionSemanticDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.public={'query':'Turn on the lamp later.','user_location':'living','current_time':'2030-01-01 12:00:00'}
        self.proposal={'tool':'schedule_workflow','arguments':{'steps':[{'tool':'execute_command',
            'args':{'device_id':'lamp'}}]},'action_id':'a1','actual_calls_before':0,'actual_calls':2,
            'blocked':False,'reached_executor':True}
        self.rows=[{'tool':'get_device_structure','arguments':{'device_id':'lamp'},'turn':1,
            'response':{'status':{'code':200},'data':{'device_id':'lamp','state':'Off'}}},
            {'tool':'schedule_workflow','arguments':self.proposal['arguments'],'turn':1,
             'response':{'status':{'code':200},'data':{'workflow_id':'future-secret'}}},
            {'tool':'get_device_structure','arguments':{'device_id':'lamp'},'turn':2,
             'response':{'status':{'code':200},'data':{'device_id':'lamp','state':'On'}}}]

    def test_cutoff_includes_guard_receipt_and_excludes_action_result_and_later_state(self):
        before=copy.deepcopy(self.rows)
        result,cutoff=contexts(self.public,self.proposal,self.rows)
        self.assertEqual(cutoff,1)
        self.assertEqual(result['legacy']['environment_state'],{})
        self.assertEqual(result['public']['environment_state']['devices']['lamp']['structure']['state'],'Off')
        self.assertNotIn('future-secret',json.dumps(result))
        self.assertEqual(result['public']['recent_actions'],[])
        self.assertEqual(self.rows,before)

    def test_nonmatching_last_call_and_blocked_or_readback_proposals_rejected(self):
        with self.assertRaisesRegex(ValueError,'do not guess'):
            contexts(self.public,{**self.proposal,'actual_calls':3},self.rows)
        with self.assertRaises(ValueError):
            contexts(self.public,{**self.proposal,'blocked':True},self.rows)

    def test_ordinary_mutation_legacy_state_comes_only_from_own_prequery(self):
        proposal={**self.proposal,'tool':'execute_command','arguments':{'device_id':'lamp'},
                  'actual_calls_before':1,'actual_calls':1}
        rows=[self.rows[0],{'tool':'execute_command','arguments':{'device_id':'lamp'},'response':{},'turn':2}]
        result,cutoff=contexts(self.public,proposal,rows)
        self.assertEqual(result['legacy']['environment_state'],{})
        self.assertEqual(result['public']['environment_state']['devices']['lamp']['structure']['state'],'Off')
        self.assertEqual(cutoff,1)

    def test_prompt_boundary_rejects_hidden_payloads(self):
        result,_=contexts(self.public,self.proposal,self.rows)
        result['public']['environment_state']['judge_output']='YES'
        with self.assertRaises(ValueError): messages(result['public'])

    def test_parser_checks_types_and_preserves_uncertainty(self):
        value={name:{'label':'UNCERTAIN','probability':.5,'evidence':{'receipt':1}}
               for name in ('correct_target','goal_consistent','trajectory_consistent','safe_to_execute')}
        self.assertEqual(parse(json.dumps(value))['verdict'],'UNCERTAIN')
        for invalid in (True,'0.5'):
            changed=copy.deepcopy(value);changed['correct_target']['probability']=invalid
            with self.assertRaises(ValueError): parse(json.dumps(changed))
        value['correct_target']['evidence']='narrative'
        with self.assertRaises(ValueError): parse(json.dumps(value))

    def test_report_preserves_failed_cost_and_never_admits_native_verifier(self):
        config={'proposals':1,'valid_ratio_min':.95}
        rows=[{'id':'real-action','arm':arm,'decision':{'verdict':'UNCERTAIN'},'parse_error':None,
               'call':{'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5},
                       'error':None,'request_seconds':.1}} for arm in ARMS]
        report=evaluate(config,rows)
        self.assertTrue(report['engineering_passed'])
        self.assertFalse(report['native_integration_admitted'])
        rows[0]['decision']=None;rows[0]['parse_error']='JSON error'
        report=evaluate(config,rows)
        self.assertFalse(report['engineering_passed'])
        self.assertEqual(report['arms']['9b_legacy']['tokens'],5)
        self.assertEqual(report['arms']['9b_legacy']['parse_errors'],1)


if __name__=='__main__': unittest.main()
