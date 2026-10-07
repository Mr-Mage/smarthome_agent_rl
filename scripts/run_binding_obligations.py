"""All retained N93 arms and N92 independent reviews; read-only, no services."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_evidence_reducer_ablation import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from smarthome_agent_rl.benchmarks.runner import save
from smarthome_agent_rl.binding_obligations import evaluate
from smarthome_agent_rl.semantic_context import digest


def prepare(config,parent=None,n92=None,*ancestors):
    parent=Path(parent or ROOT/config['parent_run']);n92=Path(n92 or ROOT/config['secondary_run'])
    count=check_manifest(parent,'collection-manifest.json',config['parent_collection_sha256'])
    secondary_count=check_manifest(n92,'collection-manifest.json',config['secondary_collection_sha256'])
    checked=audit_parent(parent,n92,*ancestors,write_receipt=False)
    protocol=read(parent/'protocol.json');older=read(n92/'protocol.json')
    inputs=protocol['inputs']
    if digest(inputs)!=config['inputs_sha256'] or inputs!=older['inputs'] or len(inputs)!=config['paired_inputs']:
        raise ValueError('Frozen identical public contexts required')
    sources=[]
    for index,item in enumerate(inputs):
        for arm,run,original in [('n93_joint',parent,'joint'),('n93_evidence_only',parent,'evidence_only'),('n92_independent',n92,'independent')]:
            relative=f'records/{index:03d}/{original}.json';path=run/relative;row=read(path)
            if (row['id'],row['arm'])!=(item['id'],original):raise ValueError('Source row identity differs')
            sources.append({'id':item['id'],'task_id':item['task_id'],'origin':item['origin'],'arm':arm,
                'context':item['context'],'row':row,'source_run':'n93' if run==parent else 'n92',
                'source_file':relative,'source_sha256':sha(path)})
    if digest(sources)!=config['sources_sha256']:raise ValueError('Frozen raw source rows differ')
    return sources,{'artifacts':count,'secondary_artifacts':secondary_count,'audit':checked}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    began=time.monotonic();config=read(args.config)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit source/config before audit')
    sources,checked=prepare(config);prepared=time.monotonic()-began
    report,records=evaluate(config,sources)
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':config,'parent_checks':checked,'preparation_seconds':prepared})
    save(args.output/'report.json',report);save(args.output/'records.json',records)
    save(args.output/'timing.json',{'cpu_wall_seconds':time.monotonic()-began,
        'scope':'Parent manifest/source/HTTP audit,coverage calculation andwrites;tests,archive,transfer andlocal audit notfullymetered'})
    save(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)).replace('\\','/'):sha(p) for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print(report)


if __name__=='__main__':main()
