import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_action_semantic_diagnosis import audit
from scripts.run_action_semantic_diagnosis import ARMS, evaluate, request, sha
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_diagnosis import contexts, parse, schema


class ActionSemanticAuditTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.run=self.root/'run';self.run.mkdir()
        write=lambda p,v: (p.parent.mkdir(parents=True,exist_ok=True),p.write_text(json.dumps(v),encoding='utf-8'))
        self.write=write
        task={'id':'fixture','path':'fixture.json'}
        public={'query':'Turn on lamp.','user_location':'living','current_time':'2030-01-01 12:00:00'}
        selection=self.root/'manifest.json';write(selection,{'tasks':[task]})
        scheduled={'task':task,'actor_seed':42,'workflow':0,'variants':['G']}
        # Match the production schedule-to-directory utility, not an invented path.
        from smarthome_agent_rl.concurrency import episode_directory
        source=self.run/'source-evidence';source.mkdir()
        prefix=str(episode_directory(Path('source-stage'),scheduled,'G').relative_to('source-stage')).replace('\\','/')
        proposal={'tool':'execute_command','arguments':{'device_id':'lamp','command_id':'On'},
                  'reached_executor':True,'blocked':False,'actual_calls_before':0,'actual_calls':2,'action_id':'a1'}
        observations=[{'tool':'get_device_structure','arguments':{'device_id':'lamp'},'turn':1,
            'response':{'status':{'code':200},'data':{'device_id':'lamp','state':'Off'}}},
            {'tool':'execute_command','arguments':proposal['arguments'],'turn':1,
             'response':{'status':{'code':200},'data':{'result':'unseen-after-cutoff'}}}]
        trace={'proposals':[proposal],'actual_observations':observations}
        write(source/prefix/'harness_audit.json',trace)
        write(source/prefix/'contract.json',{'task_identity':task,'public_context':public,
                                          'config':{'variant_policies':{'G':{'verify':False}}}})
        write(source/'protocol.json',{'commit':'retained-source','schedule':[scheduled]})
        write(source/'artifact_manifest.json',{str(p.relative_to(source)).replace('\\','/'):sha(p)
                                              for p in source.rglob('*.json')})
        self.config={'source_commit':'retained-source','source_artifact_sha256':sha(source/'artifact_manifest.json'),
            'manifest':str(selection),'model':'9b','review_model':'35b','review_endpoint':'http://shared/v1',
            'model_seed':42,'generation':{'temperature':0.0,'max_tokens':2048},'proposals':1,'valid_ratio_min':.95,
            'actors':[{'id':i,'endpoint':f'http://actor{i}/v1'} for i in range(4)]}
        values,cutoff=contexts(public,proposal,observations)
        item={'id':'fixture:seed42:a1','task_id':'fixture','source_seed':42,'contexts':values,
            'source_episode':prefix,'source_audit_sha256':sha(source/prefix/'harness_audit.json'),
            'dispatch_observation_index':cutoff,'proposal_action_id':'a1',
            'preaction_receipts_sha256':digest(observations[:cutoff])}
        self.config['inputs_sha256']=digest([item])
        coverage=[{'task_id':'fixture','source_seed':42,'selected':1,'all_proposals':1,'excluded_nonmutation_or_guard_blocked':0}]
        self.protocol={'source_commit':'fixture-code','config':self.config,'inputs':[item],'coverage':coverage,'schema':schema()}
        write(self.run/'protocol.json',self.protocol)
        value={name:{'label':'UNCERTAIN','probability':.5,'evidence':{}} for name in
               ('correct_target','goal_consistent','trajectory_consistent','safe_to_execute')}
        self.rows=[]
        for arm in ARMS:
            body=request(self.config,item,arm);actor_id=0 if arm.startswith('9b_') else None
            response={'model':body['model'],'choices':[{'message':{'content':json.dumps(value)}}],
                      'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}
            call={'endpoint':'http://actor0/v1' if actor_id==0 else self.config['review_endpoint'],
                  'body':body,'response':response,'raw_response':json.dumps(response),'text':json.dumps(value),
                  'usage':response['usage'],'error':None,'request_seconds':.1}
            row={'id':item['id'],'arm':arm,'actor_id':actor_id,'call':call,'decision':parse(call['text']),'parse_error':None}
            self.rows.append(row);write(self.run/'records/000'/(arm+'.json'),row)
        write(self.run/'report.json',evaluate(self.config,self.rows))
        write(self.run/'services/probe-receipts.json',[])
        write(self.run/'review-probe.json',{'usage':response['usage'],'error':None})
        self.freeze()

    def freeze(self):
        self.write(self.run/'artifact_manifest.json',{str(p.relative_to(self.run)).replace('\\','/'):sha(p)
            for p in self.run.rglob('*.json') if p.name not in ('artifact_manifest.json','independent-audit.json')})

    def test_replays_real_boundary_requests_cost_without_observing_mutation_result(self):
        result=audit(self.run)
        self.assertTrue(result['verified'],result['problems'])
        self.assertEqual(result['review_requests'],4)
        self.assertEqual(result['probes']['tokens'],5)
        self.assertNotIn('unseen-after-cutoff',json.dumps(self.protocol['inputs']))

    def test_later_state_in_context_detected_even_after_forged_input_and_artifact_hash(self):
        changed=copy.deepcopy(self.protocol)
        changed['inputs'][0]['contexts']['public']['environment_state']['devices']['lamp']['structure']['state']='On'
        changed['config']['inputs_sha256']=digest(changed['inputs'])
        self.write(self.run/'protocol.json',changed);self.freeze()
        result=audit(self.run)
        self.assertFalse(result['verified'])
        self.assertTrue(any('cutoffs' in p for p in result['problems']))


if __name__=='__main__':unittest.main()
