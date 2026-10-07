import copy
import unittest
from unittest.mock import patch

from tests import test_prompt_semantic_audit as previous_fixture
from scripts import audit_typed_semantic_review as auditor
from scripts import run_typed_semantic_review as driver
from scripts.run_action_semantic_diagnosis import read,sha
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_prompt_ablation import PROMPTS
from smarthome_agent_rl.typed_semantic_review import schema,request,evaluate


class TypedSemanticAuditTests(unittest.TestCase):
    def setUp(self):
        source=previous_fixture.PromptSemanticAuditTests('test_recursive_replay_does_not_rewrite_frozen_parent_receipts')
        source.setUp();self.addCleanup(source.doCleanups)
        self.root=source.root;self.write=source.write;parent=source.run
        self.run=self.root/'typed-candidate';self.run.mkdir();protocol=read(parent/'protocol.json')
        self.config={**protocol['config'],'parent_artifact_sha256':sha(parent/'artifact_manifest.json'),
            'prompt_sha256':digest(PROMPTS['9b_stepwise']),
            'schemas_sha256':digest([schema(r['context']) for r in protocol['inputs']])}
        inputs,baseline,labels,coverage,checked,schemas=driver.prepare(self.config,parent)
        import shutil
        shutil.copytree(parent,self.run/'n75-reference')
        self.protocol={'source_commit':'typed-fixture','config':self.config,'inputs':inputs,'coverage':coverage,
                       'schemas':schemas,'prompt':PROMPTS['9b_stepwise'],'parent_audit':checked}
        self.write(self.run/'protocol.json',self.protocol)
        row=copy.deepcopy(baseline[0]);row['arm']='9b_typed';row['call']['body']=request(self.config,inputs[0])
        self.write(self.run/'records/000/9b_typed.json',row)
        resources={'error':None,'stopped_pids':[1,2,3,4]}
        report=evaluate(self.config,inputs,[row],baseline,labels);report['resources']=resources
        self.write(self.run/'report.json',report);self.write(self.run/'services/lifecycle.json',resources)
        self.write(self.run/'services/probe-receipts.json',[])
        self.write(self.root/'configs/public-typed-semantic-review.json',self.config)
        self.root_auditor=patch.object(auditor,'ROOT',self.root);self.root_auditor.start();self.addCleanup(self.root_auditor.stop)
        self.freeze()

    def freeze(self):
        self.write(self.run/'artifact_manifest.json',{str(p.relative_to(self.run)).replace('\\','/'):sha(p)
            for p in self.run.rglob('*.json') if p.name not in ('artifact_manifest.json','independent-audit.json')})

    def test_ignoring_schema_is_visible_and_parent_receipts_remain_immutable(self):
        before={str(p):sha(p) for p in (self.run/'n75-reference').rglob('*') if p.is_file()}
        result=auditor.audit(self.run,write_receipt=False)
        self.assertTrue(result['verified'],result['problems'])
        report=read(self.run/'report.json')
        self.assertFalse(report['checks']['schema_structure'])
        self.assertFalse(report['diagnostic_screen_passed'])
        self.assertEqual(result['new_review_requests'],1)
        self.assertEqual(result['new_model_tokens'],5)
        self.assertEqual(before,{str(p):sha(p) for p in (self.run/'n75-reference').rglob('*') if p.is_file()})

    def test_weakened_response_schema_detected_despite_rebuilt_manifest(self):
        path=self.run/'records/000/9b_typed.json';row=read(path)
        row['call']['body']['response_format']['json_schema']['schema']['properties']['correct_target']['properties']['evidence']={'type':'object'}
        self.write(path,row);self.freeze()
        result=auditor.audit(self.run,write_receipt=False)
        self.assertFalse(result['verified'])
        self.assertTrue(any('Request identity/body' in p for p in result['problems']))


if __name__=='__main__':unittest.main()
