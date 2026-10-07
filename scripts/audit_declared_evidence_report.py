"""Recompute N81 source references, projections, panels and totals."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_declared_evidence_report import prepare,evaluate
from scripts.run_evidence_consistency_audit import check_manifest
from smarthome_agent_rl.benchmarks.runner import save


def audit(run,parent=None,n79=None,n76=None,n78=None,*,write_receipt=True):
    run=Path(run);protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/declared-evidence-report.json'):
        raise ValueError('Frozen repository configuration differs')
    count=check_manifest(run,'artifact_manifest.json',sha(run/'artifact_manifest.json'))
    sources,checked=prepare(config,parent,n79,n76,n78);report,records=evaluate(config,sources)
    if checked!=protocol['parent_checks'] or report!=read(run/'report.json') or records!=read(run/'records.json'):
        raise ValueError('Source/diagnosis/projection/counts changed')
    receipt={'verified':True,'artifacts':count,'records':len(records),'parent_checks':checked,
             'source_commit':protocol['source_commit'],'auditor_sha256':sha(Path(__file__)),
             'new_model_requests':0,'scope':'Source-bound mechanical diagnosis only;no semantic truth or changed original decisions'}
    if write_receipt:save(run/'independent-audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(audit(args.run))
