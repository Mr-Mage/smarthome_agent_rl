"""Four actors/64slots, paired same instance, single frozen156-request run."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_identity_evidence_ablation import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import completion,save
from smarthome_agent_rl.evidence_order_ablation import ordered_digest
from smarthome_agent_rl.bounded_review_ablation import ARMS,reference_catalog,schema,request,evaluate,parse_response
from smarthome_agent_rl.semantic_context import digest


def prepare(config,parent=None,n85=None,n84=None,n83=None,n82=None,n80=None,n79=None,n76=None,n78=None):
    parent=Path(parent or ROOT/config['parent_run']); n83=Path(n83 or ROOT/config['historical_run'])
    count=check_manifest(parent,'collection-manifest.json',config['parent_collection_sha256'])
    checked=audit_parent(parent,n85,n84,n83,n82,n80,n79,n76,n78,write_receipt=False)
    protocol=read(n83/'protocol.json');inputs=protocol['inputs'];labels=protocol['labels']
    if digest(inputs)!=config['inputs_sha256'] or config['arms']!=list(ARMS) or len(inputs)!=config['records']:
        raise ValueError('Frozen source cohort/arms differ')
    for key in ('model','model_seed','generation','actors','request_timeout','developer_counts','developer_conflict_groups'):
        if config[key]!=protocol['config'][key]:raise ValueError('Parent resources/generation/labels differ: '+key)
    if len(config['actors'])!=4 or config['slots_per_actor']!=16 or config['new_requests']!=len(inputs)*2:
        raise ValueError('Four actors/64slots/156unique requests required')
    schemas={arm:[schema(item['context'],arm) for item in inputs] for arm in ARMS}
    requests={arm:[request(config,item,arm) for item in inputs] for arm in ARMS}
    references=[reference_catalog(item['context']) for item in inputs]
    if ordered_digest(schemas)!=config['ordered_schemas_sha256'] or ordered_digest(requests)!=config['ordered_requests_sha256'] or digest(references)!=config['references_sha256']:
        raise ValueError('Frozen schema/request/identity presentation differs')
    historical=[read(parent/'records'/f'{index:03d}'/'identity.json') for index in range(len(inputs))]
    evaluate(config,inputs,[],historical,labels)
    return inputs,labels,schemas,requests,references,historical,{'artifacts':count,'audit':checked}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    began=time.monotonic();config=read(args.config);inputs,labels,schemas,requests,references,historical,checked=prepare(config)
    preparation=time.monotonic()-began
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit source/config before inference')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':config,'inputs':inputs,'labels':labels,'schemas':schemas,'requests':requests,'references':references,
        'parent_checks':checked,'preparation_seconds':preparation})
    records=[];started=time.monotonic();limits={a['id']:threading.Semaphore(16) for a in config['actors']}
    def one(index,item,arm):
        actor=config['actors'][index%4]
        with limits[actor['id']]:call=completion(actor['endpoint'],requests[arm][index],config['request_timeout'])
        model,decision,error=(None,None,None) if call['error'] is not None else parse_response(item['context'],call['text'],arm)
        row={'id':item['id'],'arm':arm,'actor_id':actor['id'],'call':call,'model_decision':model,'decision':decision,'parse_error':error}
        directory=args.output/'records'/f'{index:03d}';directory.mkdir(parents=True,exist_ok=True);save(directory/(arm+'.json'),row)
        return row
    def interrupted(*unused):raise KeyboardInterrupt('Interrupted;completed calls retained')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with actors(config,args.output/'services'):
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures=[pool.submit(one,index,item,arm) for index,item in enumerate(inputs) for arm in (ARMS if index%2==0 else ARMS[::-1])]
                for future in as_completed(futures):
                    records.append(future.result())
                    if len(records)%10==0:print({'completed':len(records),'expected':config['new_requests']},flush=True)
        report=evaluate(config,inputs,records,historical,labels)
        report.update(wall_seconds=time.monotonic()-started,resources=read(args.output/'services/lifecycle.json'))
        save(args.output/'report.json',report);print({'complete':report['complete'],'screens':report['diagnostic_screen_passed']},flush=True)
    except BaseException as exc:
        save(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(records)});raise
    finally:
        save(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)).replace('\\','/'):sha(p)
            for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


if __name__=='__main__':main()
