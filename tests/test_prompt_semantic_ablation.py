import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_prompt_semantic_ablation as driver
from smarthome_agent_rl.semantic_diagnosis import messages, schema
from smarthome_agent_rl.semantic_prompt_ablation import ARMS, PROMPTS, request, evidence_check, evaluate


class PromptSemanticAblationTests(unittest.TestCase):
    def setUp(self):
        self.item={'id':'workflow','context':{'user_goal':'Turn on lamp,then raise it after ten minutes.',
            'environment_state':{},'recent_actions':[],'contract':None,
            'proposed_action':{'tool':'schedule_workflow','start_time':'2030-01-01 12:00:00','steps':[
                {'tool':'execute_command','args':{'device_id':'lamp','command_id':'On'}},
                {'tool':'write_attribute','args':{'device_id':'lamp','value':60}}]}}}
        self.config={'model':'9b','model_seed':42,'generation':{'temperature':0.0,'max_tokens':2048,
            'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}}}

    def valid_row(self,arm='9b_stepwise'):
        target={'steps':[{'step_index':i,'device_id':'lamp','support_quote':'lamp','support':'requested'} for i in (1,2)]}
        time={'steps':[{'step_index':i,'encoded_execution_time':'2030-01-01 12:00:00',
                       'requested_quote':'after ten minutes','relation':'conflict'} for i in (1,2)]}
        return {'id':'workflow','arm':arm,'decision':{'verdict':'DENY','correct_target':{'evidence':target},
                                                    'goal_consistent':{'evidence':time}},'parse_error':None,
                'call':{'error':None,'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5},'request_seconds':.1}}

    def test_prompt_only_keeps_context_schema_and_generation_identical(self):
        original=messages(self.item['context'])
        for arm in ARMS:
            body=request(self.config,self.item,arm)
            self.assertEqual(body['messages'][1:],original[1:])
            self.assertEqual(body['messages'][0]['content'],PROMPTS[arm])
            self.assertEqual(body['response_format'],schema())
            self.assertEqual(body['seed'],42);self.assertEqual(body['max_tokens'],2048)
            self.assertEqual(body['chat_template_kwargs'],{'enable_thinking':False})

    def test_literal_evidence_is_not_claimed_to_prove_entailment(self):
        result=evidence_check(self.item,self.valid_row())
        self.assertTrue(result['valid'])
        self.assertIn('not entailment',result['scope'])

    def test_omitted_final_step_wrong_target_and_invented_delay_detected(self):
        row=self.valid_row();row['decision']['correct_target']['evidence']['steps'].pop()
        self.assertIn('target:missing_or_extra_steps',evidence_check(self.item,row)['issues'])
        row=self.valid_row();row['decision']['correct_target']['evidence']['steps'][1]['device_id']='other'
        self.assertIn('target:device_identity',evidence_check(self.item,row)['issues'])
        row=self.valid_row();row['decision']['goal_consistent']['evidence']['steps'][1]['encoded_execution_time']='2030-01-01 12:10:00'
        self.assertIn('time:invented_encoded_time',evidence_check(self.item,row)['issues'])

    def test_nonliteral_quote_and_bool_step_index_rejected(self):
        row=self.valid_row();step=row['decision']['correct_target']['evidence']['steps'][0]
        step['support_quote']='Turn on OTHER';step['step_index']=True
        self.assertFalse(evidence_check(self.item,row)['valid'])
        row=self.valid_row();row['decision']['goal_consistent']['evidence']['steps'][0]['requested_quote']='after twenty minutes'
        self.assertIn('time:nonliteral_quote',evidence_check(self.item,row)['issues'])

    def test_costs_and_evidence_gate_include_invalid_or_missing_records(self):
        config={**self.config,'proposals':1,'valid_ratio_min':.95,'evidence_valid_ratio_min':.95,
                'developer_counts':{'CERTAIN_CONFLICT':1}}
        records=[self.valid_row(a) for a in ARMS];baseline=[self.valid_row('9b_both')]
        baseline[0]['decision']['verdict']='ALLOW'
        labels={'cases':[{'id':'workflow','category':'CERTAIN_CONFLICT'}]}
        result=evaluate(config,[self.item],records,baseline,labels)
        self.assertEqual(result['new_model_tokens'],15)
        self.assertEqual(result['arms']['9b_historical_both']['tokens'],5)
        self.assertTrue(all(result['diagnostic_screen_passed'].values()))
        records[-1]['decision']['goal_consistent']['evidence']['steps'].pop()
        self.assertFalse(evaluate(config,[self.item],records,baseline,labels)['diagnostic_screen_passed']['9b_stepwise'])
        self.assertFalse(evaluate(config,[self.item],records[:-1],baseline,labels)['checks']['complete_records'])

    def test_historical_auditor_writes_only_to_disposable_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();parent=root/'evidence';parent.mkdir()
            original=parent/'receipt.json';original.write_text('original')
            def mutating_audit(copied):
                self.assertNotEqual(copied.resolve(),parent)
                (copied/'receipt.json').write_text('new auditor receipt')
                return {'verified':True}
            with patch.object(driver,'ROOT',root):
                self.assertTrue(driver.audit_readonly(parent,mutating_audit)['verified'])
            self.assertEqual(original.read_text(),'original')
            self.assertEqual(list((root/'work').iterdir()),[])


if __name__=='__main__':unittest.main()
