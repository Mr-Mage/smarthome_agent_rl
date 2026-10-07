"""Frozen joint/independent review:390 real calls,156 derived review records."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_partitioned_goal_ablation import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import completion,save
from smarthome_agent_rl.evidence_order_ablation import DIMENSIONS,ordered_digest
from smarthome_agent_rl.independent_review_ablation import ARMS,COMPONENTS,actor_for,request,assemble,evaluate
from smarthome_agent_rl.semantic_context import digest


def prepare(config,parent=None,*ancestors):
    parent=Path(parent or ROOT/config['parent_run'])
    count=check_manifest(parent,'collection-manifest.json',config['parent_collection_sha256'])
    checked=audit_parent(parent,*ancestors,write_receipt=False)
    protocol=read(parent/'protocol.json');inputs=protocol['inputs'];labels=protocol['labels']
    if digest(inputs)!=config['inputs_sha256'] or config['arms']!=list(ARMS) or config['components']!=list(COMPONENTS) or len(inputs)!=config['records']:
        raise ValueError('Frozen source/cohort/components differ')
    for key in ('model','model_seed','generation','actors','request_timeout','developer_counts','developer_conflict_groups','decoder_source','actor_extra_args'):
        if config[key]!=protocol['config'][key]:raise ValueError('Parent resources/generation/labels differ:'+key)
    if (len(config['actors']),config['slots_per_actor'],config['new_requests'],config['review_records'])!=(4,16,len(inputs)*5,len(inputs)*2):
        raise ValueError('Fouractors/64slots/390calls/156reviews required')
    if config['dimension_tokens']!={'correct_target':512,'goal_consistent':1024,'trajectory_consistent':256,'safe_to_execute':256} or sum(config['dimension_tokens'].values())!=config['generation']['max_tokens']:
        raise ValueError('Frozen2048 total output allocation differs')
    requests={component:[request(config,item,component) for item in inputs] for component in COMPONENTS}
    schemas={component:[body['response_format'] for body in requests[component]] for component in COMPONENTS}
    if ordered_digest(schemas)!=config['ordered_schemas_sha256'] or ordered_digest(requests)!=config['ordered_requests_sha256']:
        raise ValueError('Frozen ordered schemas/requests differ')
    if requests['joint']!=protocol['requests']['original']:raise ValueError('Fresh joint control differs')
    historical=[read(parent/'records'/f'{index:03d}'/'original.json') for index in range(len(inputs))]
    evaluate(config,inputs,[],historical,labels)
    return inputs,labels,schemas,requests,historical,{'artifacts':count,'audit':checked}


def build_record(item,index,arm,wrappers):
    components=('joint',) if arm=='joint' else DIMENSIONS
    if set(wrappers)!=set(components):raise ValueError('Review component denominator differs')
    if any(w['id']!=item['id'] or w['component']!=name for name,w in wrappers.items()):raise ValueError('Review source identity differs')
    calls={name:wrappers[name]['call'] for name in components}
    value,text,model,decision,error=assemble(item['context'],calls,arm)
    began=min(w['submitted_monotonic'] for w in wrappers.values());ended=max(w['finished_monotonic'] for w in wrappers.values())
    stored={'id':item['id'],'arm':arm,'actor_id':next(iter(wrappers.values()))['actor_id'],
        'call_files':{name:f'calls/{index:03d}/{name}.json' for name in components},
        'assembled_value':value,'assembled_text':text,'model_decision':model,'decision':decision,'parse_error':error,
        'review_start_monotonic':began,'review_end_monotonic':ended,'review_seconds':ended-began,
        'assembly_scope':'Derived field assembly;call_files are the only original HTTP receipts.'}
    return stored,{**stored,'calls':calls}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    began=time.monotonic();config=read(args.config);inputs,labels,schemas,requests,historical,checked=prepare(config);preparation=time.monotonic()-began
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit source/config before inference')
    args.output.mkdir(parents=True,exist_ok=False);snapshots=args.output/'decoder-source';snapshots.mkdir()
    for index,(path,checksum) in enumerate(config['decoder_source']['source_sha256'].items()):
        if sha(Path(path))!=checksum:raise ValueError('Existing decoder source differs')
        shutil.copy2(path,snapshots/f'{index}.py')
    save(args.output/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':config,'inputs':inputs,'labels':labels,'schemas':schemas,'requests':requests,'parent_checks':checked,'preparation_seconds':preparation})
    started=time.monotonic();limits={a['id']:threading.Semaphore(16) for a in config['actors']};records=[];wrappers={};completed=0
    def one(index,item,component,submitted):
        actor=actor_for(config,index)
        with limits[actor['id']]:
            dispatch=time.monotonic();call=completion(actor['endpoint'],requests[component][index],config['request_timeout']);finished=time.monotonic()
        return index,component,{'id':item['id'],'component':component,'actor_id':actor['id'],'submitted_monotonic':submitted,
            'dispatch_monotonic':dispatch,'finished_monotonic':finished,'call':call}
    def interrupted(*unused):raise KeyboardInterrupt('Interrupted;all returned receipts retained')
    signal.signal(signal.SIGTERM,interrupted)
    futures={}
    def retain(result):
        nonlocal completed
        index,component,wrapper=result;key=(index,component)
        if key in wrappers:return
        wrappers[key]=wrapper;directory=args.output/'calls'/f'{index:03d}';directory.mkdir(parents=True,exist_ok=True)
        save(directory/(component+'.json'),wrapper);completed+=1
        for arm,names in [('joint',('joint',)),('independent',DIMENSIONS)]:
            if component in names and all((index,name) in wrappers for name in names):
                stored,row=build_record(inputs[index],index,arm,{name:wrappers[(index,name)] for name in names})
                target=args.output/'records'/f'{index:03d}';target.mkdir(parents=True,exist_ok=True);save(target/(arm+'.json'),stored);records.append(row)
    try:
        with actors(config,args.output/'services'):
            with ThreadPoolExecutor(max_workers=64) as pool:
                for index,item in enumerate(inputs):
                    for component in COMPONENTS[index%5:]+COMPONENTS[:index%5]:
                        futures[pool.submit(one,index,item,component,time.monotonic())]=(index,component)
                for future in as_completed(futures):
                    retain(future.result())
                    if completed%20==0:print({'completed_calls':completed,'expected_calls':config['new_requests'],'reviews':len(records)},flush=True)
        report=evaluate(config,inputs,records,historical,labels)
        report.update(wall_seconds=time.monotonic()-started,resources=read(args.output/'services/lifecycle.json'))
        save(args.output/'report.json',report);print({'complete':report['complete'],'screens':report['diagnostic_screen_passed']},flush=True)
    except BaseException as exc:
        # The executor has joined; retain successful futures not yet observed by
        # the interrupted consumer, including raw failed HTTP calls and cost.
        for future in futures:
            if future.done() and not future.cancelled() and future.exception() is None:retain(future.result())
        save(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed_calls':completed,'reviews':len(records)});raise
    finally:
        save(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)).replace('\\','/'):sha(p)
            for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


if __name__=='__main__':main()
