"""Frozen target/time/combined prompt ablation against immutable N74 both outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_context_semantic_ablation import audit as audit_parent
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_public_benchmark import actors
from smarthome_agent_rl.benchmarks.runner import completion, save
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_diagnosis import parse, schema
from smarthome_agent_rl.semantic_prompt_ablation import ARMS, PROMPTS, request, evaluate


def audit_readonly(parent, auditor=audit_parent):
    # Historical auditors write receipts; sandbox that write, preserving inputs.
    workspace=(ROOT/'work').resolve();workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='n75-audit-',dir=workspace) as directory:
        sandbox=Path(directory).resolve()
        if not sandbox.is_relative_to(workspace):raise ValueError('Audit scratch path escaped workspace')
        copied=sandbox/'parent';shutil.copytree(parent,copied)
        return auditor(copied)


def prepare(config,parent=None):
    parent=Path(parent or ROOT/config['parent_run'])
    if sha(parent/'artifact_manifest.json')!=config['parent_artifact_sha256']:
        raise ValueError('Frozen N74 manifest differs')
    checked=audit_readonly(parent)
    if not checked['verified']:raise ValueError('Parent audit failed: '+str(checked['problems']))
    protocol=read(parent/'protocol.json')
    if digest(protocol['inputs'])!=config['parent_inputs_sha256'] or protocol['schema']!=schema():
        raise ValueError('Frozen N74 inputs/schema differ')
    for key in ('model','model_seed','generation','actors','request_timeout'):
        if config[key]!=protocol['config'][key]:raise ValueError('Frozen review setting changed: '+key)
    if config['arms']!=list(ARMS) or config['prompt_sha256']!={a:digest(PROMPTS[a]) for a in ARMS}:
        raise ValueError('Frozen prompts/arms differ')
    labels_path=parent/'n72-reference/developer-semantic-audit.json'
    if sha(labels_path)!=config['developer_labels_sha256']:raise ValueError('Developer readings changed')
    inputs=[];baseline=[]
    for index,item in enumerate(protocol['inputs']):
        inputs.append({**{k:v for k,v in item.items() if k!='contexts'},'context':item['contexts']['both']})
        baseline.append(read(parent/'records'/f'{index:03d}'/'9b_both.json'))
    if len(inputs)!=config['proposals'] or len(protocol['coverage'])!=config['episodes']:
        raise ValueError('Frozen real source coverage differs')
    return inputs,baseline,read(labels_path),protocol['coverage'],checked


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();config=read(args.config)
    preparation_started=time.monotonic();inputs,baseline,labels,coverage,checked=prepare(config)
    preparation_seconds=time.monotonic()-preparation_started
    if args.prepare_only:
        print(json.dumps({'inputs_sha256':digest(inputs),'proposals':len(inputs),'parent_audit':checked,
                          'preparation_seconds':preparation_seconds}));return
    if digest(inputs)!=config['inputs_sha256']:raise ValueError('Inputs changed after freeze')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit source/config before inference')
    if len(config['actors'])!=4 or config['slots_per_actor']!=16:raise ValueError('Four actors/64 slots required')
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_started=time.monotonic();shutil.copytree(ROOT/config['parent_run'],args.output/'n74-reference')
    save(args.output/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':config,'inputs':inputs,'coverage':coverage,'schema':schema(),'prompts':PROMPTS,'parent_audit':checked,
        'preparation_seconds':preparation_seconds,'snapshot_seconds':time.monotonic()-snapshot_started})
    started=time.monotonic();records=[]
    limits={a['id']:threading.Semaphore(config['slots_per_actor']) for a in config['actors']}
    def one(index,item,arm):
        actor=config['actors'][index%len(config['actors'])]
        with limits[actor['id']]:call=completion(actor['endpoint'],request(config,item,arm),config['request_timeout'])
        decision,error=None,None
        if call['error'] is None:
            try:decision=parse(call['text'])
            except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        row={'id':item['id'],'arm':arm,'actor_id':actor['id'],'call':call,'decision':decision,'parse_error':error}
        path=args.output/'records'/f'{index:03d}';path.mkdir(parents=True,exist_ok=True);save(path/(arm+'.json'),row)
        return row
    def interrupted(*unused):raise KeyboardInterrupt('Interrupted;all completed calls retained')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with actors(config,args.output/'services'):
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures=[pool.submit(one,index,item,arm) for index,item in enumerate(inputs) for arm in ARMS]
                for future in as_completed(futures):
                    records.append(future.result())
                    if len(records)%20==0:print(json.dumps({'completed':len(records),'expected':config['new_requests']}),flush=True)
        report=evaluate(config,inputs,records,baseline,labels);report['wall_seconds']=time.monotonic()-started
        report['resources']=read(args.output/'services/lifecycle.json')
        save(args.output/'report.json',report);save(args.output/'state.json',{'stage':'awaiting_independent_audit','records':len(records)})
        print(json.dumps({'stage':'awaiting_independent_audit','checks':report['checks'],
                          'diagnostic_screen_passed':report['diagnostic_screen_passed']}),flush=True)
    except BaseException as exc:
        save(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(records)});raise
    finally:
        save(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)).replace('\\','/'):sha(p)
            for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


if __name__=='__main__':main()
