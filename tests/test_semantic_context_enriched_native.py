"""Unmodified native Home/ReAct engineering checks, not benchmark tasks."""
import copy
import json
from types import SimpleNamespace
import unittest

try:
    from src.simulator.domain.home import Home
    from src.simulator.domain.devices.fan import Fan
    from src.simulator.api.schemas import ScheduleWorkflowStep
    from smarthome_agent_rl.harness_agent import HarnessAgent
except ModuleNotFoundError as exc:
    if exc.name != 'src':
        raise
    Home = None

from smarthome_agent_rl.workflow_semantics import load_workflow_semantics
from tests import test_semantic_context_native as native_fixtures


@unittest.skipIf(Home is None,'requires existing SimuHome server environment')
class NativeEnrichedSemanticContextTests(unittest.TestCase):
    def test_native_schema_and_clock_prove_no_per_step_wait(self):
        rules=load_workflow_semantics()
        step=ScheduleWorkflowStep.model_validate({'tool':'write_attribute','args':{},'delay_seconds':1740})
        self.assertEqual(set(step.model_dump()),{'tool','args'})
        home=Home(tick_interval=1,fast_forward=True,base_time='2030-01-01 12:00:00')
        fan=Fan('fan');self.assertTrue(home._add_device('living',fan).success)
        self.assertTrue(fan.execute_command(1,'OnOff','On').success)
        rows=[];original=home._write_attribute
        def write(**args):
            result=original(**args)
            rows.append({'virtual_time':home.get_virtual_now_str(),'args':copy.deepcopy(args),'success':result.success})
            return result
        home._write_attribute=write
        args={'device_id':'fan','endpoint_id':1,'cluster_id':'FanControl','attribute_id':'PercentSetting'}
        steps=[{'tool':'write_attribute','args':{**args,'value':value}} for value in (40,60)]
        registered=home.schedule_workflow('2030-01-01 12:00:10',steps)
        self.assertTrue(registered.success)
        self.assertTrue(home._fast_forward_to(11).success)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['virtual_time'],rows[1]['virtual_time'])
        self.assertEqual([r['args']['value'] for r in rows],[40,60])
        self.assertTrue(all(r['success'] for r in rows))
        self.assertEqual(fan.get_attribute(1,'FanControl','PercentSetting'),60)
        self.assertFalse(rules['facts']['per_step_wait_supported'])
        self.native_timing={'source_rules':rules,'steps':rows,'final_setting':60,
                            'status':home.get_workflow_status(registered.data['workflow_id']).data}

    def test_failed_native_step_halts_next_steps(self):
        home=Home(tick_interval=1,fast_forward=True,base_time='2030-01-01 12:00:00')
        fan=Fan('fan');self.assertTrue(home._add_device('living',fan).success)
        steps=[{'tool':'execute_command','args':{'device_id':'fan','endpoint_id':1,'cluster_id':'OnOff','command_id':command}}
               for command in ('Unsupported','On')]
        registered=home.schedule_workflow('2030-01-01 12:00:10',steps)
        self.assertTrue(registered.success)
        self.assertTrue(home._fast_forward_to(11).success)
        self.assertEqual(home.get_workflow_status(registered.data['workflow_id']).data['status'],'failed')
        self.assertFalse(fan.get_attribute(1,'OnOff','OnOff'))

    def test_native_react_supplies_references_and_rules_without_new_tool_calls(self):
        fixture=native_fixtures.NativePublicSemanticContextTests('test_native_react_supplies_original_public_inputs_and_no_extra_tool_calls')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        rows=[]
        calls=[]
        for references,workflow in ((True,False),(False,True),(True,True)):
            fixture.calls.clear();fixture.contexts.clear()
            outputs=iter([{'thought':'Read prior public reference.','call':{'tool':'get_device_structure','arguments':{'device_id':'washer'}}},
                          {'thought':'Schedule.','call':{'tool':'schedule_workflow','arguments':fixture.workflow}},
                          {'thought':'Only registered.','call':{'tool':'finish','arguments':{'answer':'Registered.'}}}])
            llm=SimpleNamespace(generate=lambda *a,**k:json.dumps(next(outputs)))
            agent=HarnessAgent(llm,variant='G',max_steps=4,policy={'verify':False,'verification_version':1,'context_version':0,
                'reflection_verifier':fixture.verifier,'semantic_context_version':2,
                'semantic_context_references':references,'semantic_context_workflow':workflow})
            def dispatch(tool,args):
                if tool=='get_device_structure' and args['device_id']=='washer':
                    fixture.calls.append((tool,copy.deepcopy(args)))
                    return {'status':{'code':200},'data':{'device_id':'washer','countdown':600},'error':None}
                return fixture.dispatch(tool,args)
            agent.executor.dispatch=dispatch
            result=agent.run('Switch lamps before washer finishes.',user_location='living',current_time='2030-01-01 12:00:00')
            state=fixture.contexts[-1].environment_state
            self.assertEqual('observed_reference_receipts' in state,references)
            self.assertEqual('workflow_execution_semantics' in state,workflow)
            self.assertEqual(state['observation_count'],2)
            if references:
                self.assertEqual(state['observed_reference_receipts']['devices'][0]['device_id'],'washer')
                self.assertEqual(state['observed_reference_receipts']['devices'][0]['facts'][0]['source']['observation_ordinal'],1)
            self.assertEqual(result.final_answer,'Registered.')
            self.assertEqual([name for name,_ in fixture.calls],['get_device_structure','get_device_structure','schedule_workflow'])
            self.assertEqual(len(result.tool_calls),3)
            rows.append(copy.deepcopy(agent.executor.audit[-1]));calls.append(copy.deepcopy(fixture.calls))
        self.assertEqual(calls[0],calls[1]);self.assertEqual(calls[1],calls[2])
        self.native_context={'proposals':rows,'actual_calls':calls,
                             'scope':'Scripted native ReAct engineering case;no model requests or benchmark score'}


if __name__=='__main__':unittest.main()
