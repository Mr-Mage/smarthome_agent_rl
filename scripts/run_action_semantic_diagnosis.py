"""Frozen paired read-only reviews of retained official G action proposals."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmarks.runner import completion, save, valid_usage
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.semantic_context import MUTATIONS, digest
from smarthome_agent_rl.semantic_diagnosis import contexts, messages, schema, parse
from scripts.run_public_benchmark import actors
from scripts.verify_benchmark import verify

read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ARMS = ('9b_legacy','9b_public','35b_legacy','35b_public')


def prepare(config):
    stage = ROOT/config['source_stage']
    verify(stage)
    protocol = read(stage/'protocol.json')
    if protocol['commit']!=config['source_commit'] or sha(stage/'artifact_manifest.json')!=config['source_artifact_sha256']:
        raise ValueError('Retained source identity differs')
    selected = {r['id']:r for r in read(ROOT/config['manifest'])['tasks']}
    inputs, coverage = [], []
    for item in protocol['schedule']:
        task = item['task']
        if task['id'] not in selected:
            continue
        if task != selected[task['id']]:
            raise ValueError('Task manifest differs from source')
        episode = episode_directory(stage,item,'G')
        contract, audit = read(episode/'contract.json'),read(episode/'harness_audit.json')
        public=contract['public_context']
        if set(public)!={'query','user_location','current_time'} or contract['config']['variant_policies']['G']['verify']:
            raise ValueError('Only unchanged legacy G public input and non-readback boundary supported')
        selected_count=0
        for proposal in audit['proposals']:
            if proposal['tool'] not in MUTATIONS or proposal.get('blocked') or not proposal.get('reached_executor'):
                continue
            variants,cutoff=contexts(public,proposal,audit['actual_observations'])
            inputs.append({'id':f"{task['id']}:seed{item['actor_seed']}:{proposal['action_id']}",
                'task_id':task['id'],'source_seed':item['actor_seed'],'contexts':variants,
                'source_episode':str(episode.relative_to(stage)).replace('\\','/'),
                'source_audit_sha256':sha(episode/'harness_audit.json'), 'dispatch_observation_index':cutoff,
                'proposal_action_id':proposal['action_id'],
                'preaction_receipts_sha256':digest(audit['actual_observations'][:cutoff])})
            selected_count+=1
        coverage.append({'task_id':task['id'],'source_seed':item['actor_seed'],'selected':selected_count,
                         'all_proposals':len(audit['proposals']),
                         'excluded_nonmutation_or_guard_blocked':len(audit['proposals'])-selected_count})
    if len(coverage)!=config['episodes'] or {(r['task_id'],r['source_seed']) for r in coverage} != {
            (task,seed) for task in selected for seed in config['source_seeds']} or len(inputs)!=config['proposals']:
        raise ValueError('Frozen episode/action coverage differs')
    if len({r['id'] for r in inputs})!=len(inputs):
        raise ValueError('Duplicate proposal identity')
    return inputs,coverage


def request(config, item, arm):
    model, version = arm.split('_',1)
    generation = config['generation']
    return {'model':config['model'] if model=='9b' else config['review_model'],
        'seed':config['model_seed'], 'messages':messages(item['contexts'][version]),
        'response_format':schema(), **{k:v for k,v in generation.items() if k!='extra_body'},
        **generation.get('extra_body',{})}


def evaluate(config, records):
    from collections import Counter
    totals={}
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        totals[arm]={'records':len(rows), 'valid':sum(r['decision'] is not None for r in rows),
            'verdicts':dict(Counter(r['decision']['verdict'] for r in rows if r['decision'] is not None)),
            'tokens':sum(r['call']['usage']['total_tokens'] for r in rows if valid_usage(r['call']['usage'])),
            'http_errors':sum(r['call']['error'] is not None for r in rows),
            'missing_usage':sum(not valid_usage(r['call']['usage']) for r in rows),
            'parse_errors':sum(r['parse_error'] is not None for r in rows),
            'median_request_seconds':statistics.median(r['call']['request_seconds'] for r in rows) if rows else None}
    complete=len(records)==config['proposals']*len(ARMS) and len({(r['id'],r['arm']) for r in records})==len(records)
    checks={'complete_records':complete,
        'http_errors':all(r['http_errors']==0 for r in totals.values()),
        'usage_complete':all(r['missing_usage']==0 for r in totals.values()),
        'structural':all(r['valid']/max(1,config['proposals'])>=config['valid_ratio_min'] for r in totals.values())}
    pairs={}
    mapped={(r['id'],r['arm']):r for r in records}
    for left,right in (('9b_legacy','9b_public'),('35b_legacy','35b_public'),('9b_public','35b_public')):
        values=Counter()
        for proposal in {r['id'] for r in records}:
            a,b=mapped.get((proposal,left)),mapped.get((proposal,right))
            if a is None or b is None or a['decision'] is None or b['decision'] is None:
                values['invalid_pair']+=1
            else:
                values[a['decision']['verdict']+'->'+b['decision']['verdict']]+=1
        pairs[left+'__'+right]=dict(values)
    return {'checks':checks,'engineering_passed':all(checks.values()),'arms':totals,'pairs':pairs,
            'native_integration_admitted':False,
            'scope':'Retained real public proposals only; verdicts are uncalibrated model opinions,not accuracy or native SR. '
                    'Guard-admitted intermediate actions do not certify whole-task completion. Same-model self-review bias retained.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); config=read(args.config)
    inputs,coverage=prepare(config)
    if args.prepare_only:
        print(json.dumps({'proposals':len(inputs),'episodes':len(coverage),'inputs_sha256':digest(inputs),'coverage':coverage}))
        return
    if digest(inputs)!=config['inputs_sha256']:
        raise ValueError('Prepared public action inputs changed after freeze')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit source/config before inference')
    if len(config['actors'])!=4 or config['slots_per_actor']!=16 or config['review_slots']!=16:
        raise ValueError('Four isolated9B actors/64 slots and shared A80016 slots required')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                    'config':config,'inputs':inputs,'coverage':coverage,'schema':schema()})
    save(args.output/'review-inputs.json',{'assessor':'Pending Codex developer review,not independent human truth',
        'cases':[{'id':r['id'],'context':r['contexts']['public']} for r in inputs]})
    records=[]; started=time.monotonic()
    limits={a['id']:threading.Semaphore(config['slots_per_actor']) for a in config['actors']}
    review_limit=threading.Semaphore(config['review_slots'])
    def one(index,item,arm):
        actor=config['actors'][index%len(config['actors'])]
        endpoint=actor['endpoint'] if arm.startswith('9b_') else config['review_endpoint']
        limit=limits[actor['id']] if arm.startswith('9b_') else review_limit
        with limit:
            call=completion(endpoint,request(config,item,arm),config['request_timeout'])
        decision,error=None,None
        if call['error'] is None:
            try: decision=parse(call['text'])
            except (ValueError,TypeError,KeyError) as exc: error=str(exc)
        row={'id':item['id'],'arm':arm,'actor_id':actor['id'] if arm.startswith('9b_') else None,
             'call':call,'decision':decision,'parse_error':error}
        path=args.output/'records'/f'{index:03d}'
        path.mkdir(parents=True,exist_ok=True)
        save(path/(arm+'.json'),row)
        return row
    def interrupted(*args): raise KeyboardInterrupt('Interrupted; preserve all completed calls')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with ExitStack() as stack:
            stack.enter_context(actors(config,args.output/'services'))
            probe=completion(config['review_endpoint'],{'model':config['review_model'],'seed':42,'temperature':0.0,
                'max_tokens':8,'messages':[{'role':'user','content':'Reply OK.'}],
                'chat_template_kwargs':{'enable_thinking':False}},config['request_timeout'])
            save(args.output/'review-probe.json',probe)
            if probe['error'] is not None or not valid_usage(probe['usage']):
                raise ValueError('Shared reviewer probe failed; never stop shared service')
            with ThreadPoolExecutor(max_workers=64) as nine,ThreadPoolExecutor(max_workers=16) as large:
                futures=[(nine if arm.startswith('9b_') else large).submit(one,index,item,arm)
                         for index,item in enumerate(inputs) for arm in ARMS]
                for future in as_completed(futures):
                    records.append(future.result())
                    if len(records)%20==0:
                        print(json.dumps({'completed':len(records),'expected':len(futures)}),flush=True)
        report=evaluate(config,records)
        report['wall_seconds']=time.monotonic()-started
        report['resources']=read(args.output/'services/lifecycle.json')
        save(args.output/'report.json',report)
        save(args.output/'state.json',{'stage':'awaiting_developer_audit','records':len(records),'checks':report['checks']})
        print(json.dumps({'stage':'awaiting_developer_audit','records':len(records),'checks':report['checks']}),flush=True)
    except BaseException as exc:
        save(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed':len(records)})
        raise
    finally:
        save(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)).replace('\\','/'):sha(p)
             for p in args.output.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


if __name__=='__main__': main()
