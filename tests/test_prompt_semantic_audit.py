import copy
import unittest
from unittest.mock import patch

from tests import test_context_semantic_audit as previous_fixture
from scripts import audit_prompt_semantic_ablation as auditor
from scripts import run_prompt_semantic_ablation as driver
from scripts.run_action_semantic_diagnosis import read, sha
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_diagnosis import schema
from smarthome_agent_rl.semantic_prompt_ablation import ARMS, PROMPTS, request, evaluate


class PromptSemanticAuditTests(unittest.TestCase):
    def setUp(self):
        source=previous_fixture.ContextSemanticAuditTests('test_replays_original_parent_and_three_candidate_costs')
        source.setUp();self.addCleanup(source.doCleanups)
        self.root=source.root;self.write=source.write;parent=source.run
        self.run=self.root/'prompt-candidate';self.run.mkdir()
        protocol=read(parent/'protocol.json')
        self.config={**protocol['config'],'parent_artifact_sha256':sha(parent/'artifact_manifest.json'),
            'parent_inputs_sha256':protocol['config']['inputs_sha256'],
            'arms':list(ARMS),'prompt_sha256':{a:digest(PROMPTS[a]) for a in ARMS},'evidence_valid_ratio_min':.95}
        self.root_driver=patch.object(driver,'ROOT',self.root);self.root_driver.start();self.addCleanup(self.root_driver.stop)
        inputs,baseline,labels,coverage,checked=driver.prepare(self.config,parent)
        self.config['inputs_sha256']=digest(inputs)
        import shutil
        shutil.copytree(parent,self.run/'n74-reference')
        self.protocol={'source_commit':'prompt-fixture','config':self.config,'inputs':inputs,'coverage':coverage,
                       'schema':schema(),'prompts':PROMPTS,'parent_audit':checked}
        self.write(self.run/'protocol.json',self.protocol)
        records=[]
        for arm in ARMS:
            row=copy.deepcopy(baseline[0]);row['arm']=arm;row['call']['body']=request(self.config,inputs[0],arm)
            records.append(row);self.write(self.run/'records/000'/(arm+'.json'),row)
        resources={'error':None,'stopped_pids':[1,2,3,4]}
        report=evaluate(self.config,inputs,records,baseline,labels);report['resources']=resources
        self.write(self.run/'report.json',report);self.write(self.run/'services/lifecycle.json',resources)
        self.write(self.run/'services/probe-receipts.json',[])
        self.write(self.root/'configs/public-prompt-semantic-ablation.json',self.config)
        self.root_auditor=patch.object(auditor,'ROOT',self.root);self.root_auditor.start();self.addCleanup(self.root_auditor.stop)
        self.freeze()

    def freeze(self):
        self.write(self.run/'artifact_manifest.json',{str(p.relative_to(self.run)).replace('\\','/'):sha(p)
            for p in self.run.rglob('*.json') if p.name not in ('artifact_manifest.json','independent-audit.json')})

    def test_recursive_replay_does_not_rewrite_frozen_parent_receipts(self):
        before={str(p):sha(p) for p in (self.run/'n74-reference').rglob('*') if p.is_file()}
        result=auditor.audit(self.run,write_receipt=False)
        self.assertTrue(result['verified'],result['problems'])
        self.assertEqual(result['new_review_requests'],3);self.assertEqual(result['reused_review_requests'],1)
        self.assertEqual(result['new_model_tokens'],15)
        self.assertEqual(before,{str(p):sha(p) for p in (self.run/'n74-reference').rglob('*') if p.is_file()})
        self.assertFalse((self.run/'independent-audit.json').exists())

    def test_forged_action_and_rebuilt_config_rejected_against_source(self):
        forged=copy.deepcopy(self.protocol)
        forged['inputs'][0]['context']['proposed_action']['device_id']='other'
        forged['config']['inputs_sha256']=digest(forged['inputs'])
        self.write(self.root/'configs/public-prompt-semantic-ablation.json',forged['config'])
        self.write(self.run/'protocol.json',forged);self.freeze()
        result=auditor.audit(self.run,write_receipt=False)
        self.assertFalse(result['verified'])
        self.assertTrue(any('cutoffs' in p for p in result['problems']))

    def test_unfrozen_prompt_rejected_even_with_rebuilt_manifest(self):
        path=self.run/'records/000/9b_stepwise.json';row=read(path)
        row['call']['body']['messages'][0]['content']+='Unfrozen instruction.'
        self.write(path,row);self.freeze()
        result=auditor.audit(self.run,write_receipt=False)
        self.assertFalse(result['verified'])
        self.assertTrue(any('Request identity/body' in p for p in result['problems']))


if __name__=='__main__':unittest.main()
