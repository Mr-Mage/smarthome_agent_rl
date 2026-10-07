"""One38-request evidence-schema ablation with fixed N75 combined prompt."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import json

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_prompt_semantic_ablation import audit as audit_parent
from scripts.run_prompt_semantic_ablation import audit_readonly
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import completion,save
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_diagnosis import parse
from smarthome_agent_rl.semantic_prompt_ablation import PROMPTS
from smarthome_agent_rl.typed_semantic_review import request,schema,evaluate


def prepare(config,parent=None):
    parent=Path(parent or ROOT/config['parent_run'])
    if sha(parent/'artifact_manifest.json')!=config['parent_artifact_sha256']:raise ValueError('Frozen N75 manifest differs')
    checked=audit_readonly(parent,audit_parent)
    if not checked['verified']:raise ValueError('Parent audit failed: '+str(checked['problems']))
    protocol=read(parent/'protocol.json')
    if digest(protocol['inputs'])!=config['inputs_sha256']:raise ValueError('Frozen N75 inputs differ')
    for key in ('model','model_seed','generation','actors','request_timeout'):
        if config[key]!=protocol['config'][key]:raise ValueError('Frozen review setting changed: '+key)
    if config['prompt_sha256']!=digest(PROMPTS['9b_stepwise']):raise ValueError('Frozen combined prompt changed')
    inputs=protocol['inputs'];schemas=[schema(r['context']) for r in inputs]
    if digest(schemas)!=config['schemas_sha256']:raise ValueError('Frozen evidence schemas differ')
    label_path=parent/'n74-reference/n72-reference/developer-semantic-audit.json'
    if sha(label_path)!=config['developer_labels_sha256']:raise ValueError('Developer readings changed')
    if len(inputs)!=config['proposals'] or len(protocol['coverage'])!=config['episodes']:raise ValueError('Frozen source coverage differs')
    baseline=[read(parent/'records'/f'{i:03d}'/'9b_stepwise.json') for i in range(len(inputs))]
    return inputs,baseline,read(label_path),protocol['coverage'],checked,schemas


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();config=read(args.config);prepared=time.monotonic()
    inputs,baseline,labels,coverage,checked,schemas=prepare(config);preparation_seconds=time.monotonic()-prepared
    if args.prepare_only:
        print(json.dumps({'proposals':len(inputs),'inputs_sha256':digest(inputs),'schemas_sha256':digest(schemas),
                          'parent_audit':checked,'preparation_seconds':preparation_seconds}));return
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit source/config before inference')
    if len(config['actors'])!=4 or config['slots_per_actor']!=16:raise ValueError('Four actors/64 slots required')
    args.output.mkdir(parents=True,exist_ok=False);copied=time.monotonic()
    shutil.copytree(ROOT/config['parent_run'],args.output/'n75-reference')
    save(args.output/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':config,'inputs':inputs,'coverage':coverage,'schemas':schemas,'prompt':PROMPTS['9b_stepwise'],
        'parent_audit':checked,'preparation_seconds':preparation_seconds,'snapshot_seconds':time.monotonic()-copied})
    records=[];started=time.monotonic();limits={a['id']:threading.Semaphore(config['slots_per_actor']) for a in config['actors']}
    def one(index,item):
        actor=config['actors'][index%len(config['actors'])]
        with limits[actor['id']]:call=completion(actor['endpoint'],request(config,item),config['request_timeout'])
        decision,error=None,None
        if call['error'] is None:
            try:decision=parse(call['text'])
            except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        row={'id':item['id'],'arm':'9b_typed','actor_id':actor['id'],'call':call,'decision':decision,'parse_error':error}
        path=args.output/'records'/f'{index:03d}';path.mkdir(parents=True,exist_ok=True);save(path/'9b_typed.json',row)
        return row
    def interrupted(*unused):raise KeyboardInterrupt('Interrupted;completed calls retained')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with actors(config,args.output/'services'):
            with ThreadPoolExecutor(max_workers=64) as pool:
                for future in as_completed([pool.submit(one,i,r) for i,r in enumerate(inputs)]):
                    records.append(future.result())
                    if len(records)%10==0:print(json.dumps({'completed':len(records),'expected':config['new_requests']}),flush=True)
        report=evaluate(config,inputs,records,baseline,labels);report['wall_seconds']=time.monotonic()-started
        report['resources']=read(args.output/'services/lifecycle.json');save(args.output/'report.json',report)
        save(args.output/'state.json',{'stage':'awaiting_independent_audit','records':len(records)})
        print(json.dumps({'stage':'awaiting_independent_audit','checks':report['checks'],'diagnostic_screen_passed':report['diagnostic_screen_passed']}),flush=True)
    except BaseException as exc:
        save(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(records)});raise
    finally:
        save(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)).replace('\\','/'):sha(p)
            for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


if __name__=='__main__':main()
