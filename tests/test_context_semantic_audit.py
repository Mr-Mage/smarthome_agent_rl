"""Auditor rejects forged context/report even when the outer manifest is rebuilt."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_action_semantic_audit as parent_fixture
from scripts import audit_context_semantic_ablation as auditor
from scripts.run_action_semantic_diagnosis import read, sha, request
from scripts.run_context_semantic_ablation import prepare
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_context_ablation import ARMS, evaluate
from smarthome_agent_rl.semantic_diagnosis import schema


class ContextSemanticAuditTests(unittest.TestCase):
    def setUp(self):
        source=parent_fixture.ActionSemanticAuditTests('test_replays_real_boundary_requests_cost_without_observing_mutation_result')
        source.setUp();self.addCleanup(source.doCleanups)
        self.write=source.write;self.root=source.root;self.run=self.root/'candidate';self.run.mkdir()
        parent=source.run;protocol=read(parent/'protocol.json')
        protocol['config']['request_timeout']=240;self.write(parent/'protocol.json',protocol)
        labels={'cases':[{'id':protocol['inputs'][0]['id'],'category':'UNCERTAIN'}]}
        self.write(parent/'developer-semantic-audit.json',labels);source.freeze()
        self.config={**protocol['config'],'episodes':1,'parent_artifact_sha256':sha(parent/'artifact_manifest.json'),
            'parent_inputs_sha256':protocol['config']['inputs_sha256'],
            'developer_labels_sha256':sha(parent/'developer-semantic-audit.json'),'developer_counts':{'UNCERTAIN':1}}
        rules={'source_sha256':{},'facts':{'per_step_wait_supported':False}}
        self.start_patch=patch('scripts.run_context_semantic_ablation.load_workflow_semantics',return_value=rules)
        self.start_patch.start();self.addCleanup(self.start_patch.stop)
        inputs,baseline,labels,coverage,checked=prepare(self.config,parent)
        self.config['inputs_sha256']=digest(inputs)
        import shutil
        shutil.copytree(parent,self.run/'n72-reference')
        self.protocol={'source_commit':'candidate-fixture','config':self.config,'inputs':inputs,
                       'coverage':coverage,'schema':schema(),'parent_audit':checked}
        self.write(self.run/'protocol.json',self.protocol)
        records=[]
        for arm in ARMS:
            row=copy.deepcopy(baseline[0]);row['arm']=arm
            row['call']['body']=request(self.config,inputs[0],arm)
            records.append(row);self.write(self.run/'records/000'/(arm+'.json'),row)
        resources={'error':None,'stopped_pids':[1,2,3,4]}
        report=evaluate(self.config,records,baseline,labels);report['resources']=resources
        self.write(self.run/'report.json',report)
        self.write(self.run/'services/lifecycle.json',resources)
        self.write(self.run/'services/probe-receipts.json',[])
        self.write(self.root/'configs/public-context-semantic-ablation.json',self.config)
        self.root_patch=patch.object(auditor,'ROOT',self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)
        self.freeze()

    def freeze(self):
        self.write(self.run/'artifact_manifest.json',{str(p.relative_to(self.run)).replace('\\','/'):sha(p)
            for p in self.run.rglob('*.json') if p.name not in ('artifact_manifest.json','independent-audit.json')})

    def test_replays_original_parent_and_three_candidate_costs(self):
        result=auditor.audit(self.run)
        self.assertTrue(result['verified'],result['problems'])
        self.assertEqual(result['new_review_requests'],3)
        self.assertEqual(result['reused_review_requests'],1)
        self.assertEqual(result['new_model_tokens'],15)

    def test_forged_future_context_rejected_despite_rebuilt_config_and_manifest(self):
        forged=copy.deepcopy(self.protocol)
        forged['inputs'][0]['contexts']['both']['environment_state']['later_state']='FORGED'
        forged['config']['inputs_sha256']=digest(forged['inputs'])
        self.write(self.root/'configs/public-context-semantic-ablation.json',forged['config'])
        self.write(self.run/'protocol.json',forged);self.freeze()
        result=auditor.audit(self.run)
        self.assertFalse(result['verified'])
        self.assertTrue(any('pre-action' in p for p in result['problems']))

    def test_forged_cost_detected_despite_rebuilt_manifest(self):
        report=read(self.run/'report.json');report['new_model_tokens']+=100
        self.write(self.run/'report.json',report);self.freeze()
        result=auditor.audit(self.run)
        self.assertFalse(result['verified'])
        self.assertTrue(any('cost' in p for p in result['problems']))


if __name__=='__main__':unittest.main()
