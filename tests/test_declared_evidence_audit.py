import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_declared_evidence_report import audit, ROOT
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_declared_evidence_report import evaluate
from smarthome_agent_rl.benchmarks.runner import save
from tests import test_effect_evidence_ablation as fixtures


class DeclaredEvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.EffectEvidenceAblationTests();fixture.setUp()
        self.config={'records':4,'arms':['historical','effects','coherence','combined']}
        self.sources=[{'id':'one','origin':'n76','arm':arm,'context':fixture.item['context'],
                       'decision':fixture.decision('coherence' if arm=='historical' else arm),
                       'source_file':'original.json','source_sha256':'immutable'} for arm in self.config['arms']]
        self.checked={'parent_artifacts':1,'parent_audit':{'verified':True}}

    def test_failures_remain_unavailable_and_originals_preserved_in_full_denominator(self):
        sources=copy.deepcopy(self.sources);sources[-1]['decision']=None
        report,records=evaluate(self.config,sources)
        self.assertTrue(report['complete']);self.assertEqual(report['records'],4)
        self.assertEqual(report['arms']['combined']['original_to_declared_projection'],{'INVALID->UNAVAILABLE':1})
        self.assertEqual(report['changed_original_decisions'],0);self.assertFalse(report['native_admitted'])
        self.assertEqual(report['new_tokens'],0)
        self.assertIsNone(records[-1]['report']['original'])

    def test_rehashed_report_cannot_rewrite_projection_original_or_admission(self):
        report,records=evaluate(self.config,self.sources)
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)
            save(run/'protocol.json',{'config':self.config,'parent_checks':self.checked,'source_commit':'frozen'})
            def manifest():
                save(run/'artifact_manifest.json',{p.name:sha(p) for p in run.iterdir() if p.name!='artifact_manifest.json'})
            def reader(path):
                return self.config if path==ROOT/'configs/declared-evidence-report.json' else read(path)
            with patch('scripts.audit_declared_evidence_report.prepare',return_value=(self.sources,self.checked)), \
                 patch('scripts.audit_declared_evidence_report.read',side_effect=reader):
                save(run/'report.json',report);save(run/'records.json',records);manifest()
                self.assertTrue(audit(run,write_receipt=False)['verified'])
                for field,value in [('original',None),('projected_labels',{'correct_target':'NO'}),('ready_for_blocking',True)]:
                    changed=copy.deepcopy(records);changed[0]['report'][field]=value
                    save(run/'records.json',changed);manifest()
                    with self.assertRaises(ValueError):audit(run,write_receipt=False)
                save(run/'records.json',records)
                changed=copy.deepcopy(report);changed['new_tokens']=123
                save(run/'report.json',changed);manifest()
                with self.assertRaises(ValueError):audit(run,write_receipt=False)


if __name__=='__main__':unittest.main()
