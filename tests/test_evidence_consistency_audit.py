import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_evidence_consistency_audit import check_manifest, evaluate
from tests import test_evidence_consistency as fixtures


class EvidenceConsistencyAuditTests(unittest.TestCase):
    def test_missing_or_changed_parent_artifact_rejected_without_writing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'record.json';path.write_text('{}',encoding='utf-8')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'record.json':hashlib.sha256(path.read_bytes()).hexdigest()}))
            expected=hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(check_manifest(root,'manifest.json',expected),1)
            path.write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaises(ValueError):check_manifest(root,'manifest.json',expected)
            path.unlink()
            with self.assertRaises(ValueError):check_manifest(root,'manifest.json',expected)
            self.assertEqual(list(root.iterdir()),[manifest])

    def test_invalid_outputs_stay_in_denominator_and_flags_do_not_change_verdicts(self):
        fixture=fixtures.EvidenceConsistencyTests('test_target_yes_cannot_hide_declared_unsupported_final_step');fixture.setUp()
        inputs=[{'id':'old','origin':'n76','context':fixture.context,'decision':fixture.decision,
                 'call':{'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}},
                {'id':'failed','origin':'n78','context':fixture.context,'decision':None,
                 'call':{'usage':{'prompt_tokens':1,'completion_tokens':4,'total_tokens':5}}}]
        report,records=evaluate(inputs)
        self.assertEqual(report['records'],2);self.assertEqual(len(records),2)
        self.assertEqual(report['origins']['n78']['original_verdicts'],{'INVALID':1})
        self.assertEqual(report['origins']['n78']['reused_review_tokens'],5)
        self.assertEqual(report['changed_verdicts'],0);self.assertFalse(report['native_admitted'])
        self.assertEqual(report['new_model_requests'],0)


if __name__=='__main__':unittest.main()
