"""Frozen evidence-first quote versus citation-ID paired reviews."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_evidence_order_ablation import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import completion,save
from smarthome_agent_rl.citation_review import ARMS,PROMPT,catalog,schema,request,parse_response,evaluate
from smarthome_agent_rl.effect_evidence_ablation import PROMPT_BY_ARM
from smarthome_agent_rl.evidence_order_ablation import ordered_digest
from smarthome_agent_rl.semantic_context import digest


def prepare(config,parent=None,n80=None,n79=None,n76=None,n78=None):
    parent=Path(parent or ROOT/config['parent_run'])
    count=check_manifest(parent,'collection-manifest.json',config['parent_collection_sha256'])
    checked=audit_parent(parent,n80,n79,n76,n78,write_receipt=False)
    protocol=read(parent/'protocol.json');inputs=protocol['inputs'];labels=protocol['labels']
    if digest(inputs)!=config['inputs_sha256'] or len(inputs)!=config['records'] or config['arms']!=list(ARMS):
        raise ValueError('Frozen cohort/arms differ')
    for key in ('model','model_seed','generation','actors','request_timeout','developer_counts','developer_conflict_groups'):
        if config[key]!=protocol['config'][key]:raise ValueError('Frozen resource/generation/developer scope differs: '+key)
    schemas={arm:[schema(item['context'],arm) for item in inputs] for arm in ARMS}
    catalogs=[catalog(item['context']['user_goal']) for item in inputs]
    prompts={'quotes':PROMPT_BY_ARM['combined'],'citations':PROMPT_BY_ARM['combined']+'\n'+PROMPT}
    if ordered_digest(schemas)!=config['ordered_schemas_sha256'] or digest(prompts)!=config['prompts_sha256'] or digest(catalogs)!=config['catalogs_sha256']:
        raise ValueError('Frozen representation/catalog differs')
    if len(config['actors'])!=4 or config['slots_per_actor']!=16 or config['new_requests']!=len(inputs)*2:
        raise ValueError('Four actors/64slots and complete paired requests required')
    historical=[read(parent/'records'/f'{index:03d}'/'evidence_first.json') for index in range(len(inputs))]
    evaluate(config,inputs,[],historical,labels)
    return inputs,labels,schemas,catalogs,prompts,historical,{'artifacts':count,'audit':checked}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=read(args.config);began=time.monotonic();inputs,labels,schemas,catalogs,prompts,historical,checked=prepare(config)
    preparation=time.monotonic()-began
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit source/config before inference')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'config':config,'inputs':inputs,'labels':labels,'schemas':schemas,'catalogs':catalogs,'prompts':prompts,
         'parent_checks':checked,'preparation_seconds':preparation})
    records=[];started=time.monotonic();limits={actor['id']:threading.Semaphore(16) for actor in config['actors']}
    def one(index,item,arm):
        actor=config['actors'][index%4]
        with limits[actor['id']]:call=completion(actor['endpoint'],request(config,item,arm),config['request_timeout'])
        model,decision,error=(None,None,None) if call['error'] is not None else parse_response(item['context'],call['text'],arm)
        row={'id':item['id'],'arm':arm,'actor_id':actor['id'],'call':call,'model_decision':model,'decision':decision,'parse_error':error}
        directory=args.output/'records'/f'{index:03d}';directory.mkdir(parents=True,exist_ok=True)
        save(directory/(arm+'.json'),row);return row
    def interrupted(*unused):raise KeyboardInterrupt('Interrupted;all completed calls retained')
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
        save(args.output/'report.json',report);save(args.output/'state.json',{'stage':'awaiting_independent_audit','records':len(records)})
        print({'complete':report['complete'],'screens':report['diagnostic_screen_passed']},flush=True)
    except BaseException as exc:
        save(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(records)});raise
    finally:
        save(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)).replace('\\','/'):sha(p)
             for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


if __name__=='__main__':main()
