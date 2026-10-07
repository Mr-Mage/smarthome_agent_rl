"""Recompute the complete frozen cohort and every reported consistency flag."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_evidence_consistency_audit import prepare, evaluate, check_manifest
from smarthome_agent_rl.benchmarks.runner import save


def audit(run, n76=None, n78=None, *, write_receipt=True):
    run=Path(run);protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/evidence-consistency-audit.json'):
        raise ValueError('Frozen configuration differs')
    manifest=read(run/'artifact_manifest.json')
    files=check_manifest(run,'artifact_manifest.json',sha(run/'artifact_manifest.json'))
    inputs,coverage=prepare(config,n76,n78)
    report,records=evaluate(inputs)
    if inputs!=read(run/'inputs.json') or records!=read(run/'records.json') or report!=read(run/'report.json') or coverage!=protocol['coverage']:
        raise ValueError('Source cohort/claims/counts/costs changed')
    receipt={'verified':True,'artifacts':files,'records':len(inputs),'coverage':coverage,
             'source_commit':protocol['source_commit'],'new_model_requests':0,
             'scope':'All historical failures preserved;engineering consistency only,not semantic truth'}
    if write_receipt:save(run/'independent-audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--n76',type=Path);parser.add_argument('--n78',type=Path)
    args=parser.parse_args();print(audit(args.run,args.n76,args.n78))
