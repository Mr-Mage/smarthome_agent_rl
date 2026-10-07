"""Independently recompute N94 source-bound coverage; never certify semantics."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_binding_obligations import prepare
from scripts.run_evidence_consistency_audit import check_manifest
from smarthome_agent_rl.benchmarks.runner import save
from smarthome_agent_rl.binding_obligations import evaluate


def audit(run,parent=None,n92=None,*ancestors,write_receipt=True):
    run=Path(run);protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/binding-obligations.json'):raise ValueError('Frozen repository config differs')
    count=check_manifest(run,'artifact_manifest.json',sha(run/'artifact_manifest.json'))
    sources,checked=prepare(config,parent,n92,*ancestors);report,records=evaluate(config,sources)
    if (report,records,checked)!=(read(run/'report.json'),read(run/'records.json'),protocol['parent_checks']):
        raise ValueError('Binding coverage/source graphs differ')
    receipt={'verified':True,'artifacts':count,'records':len(records),'parent_checks':checked,
        'source_commit':protocol['source_commit'],'auditor_sha256':sha(Path(__file__)),
        'new_model_requests':0,'new_tokens':0,'scope':'Exact original source andlexical/interval graph audit only;not device/phase semantic truth'}
    if write_receipt:save(run/'independent-audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();print(audit(args.run))
