"""Curate legal action targets from verified successful training-task episodes only."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.sft_data import action_target,text_hash,tokenized_target
from smarthome_agent_rl.concurrency import episode_directory
from scripts.verify_benchmark import verify

def curate(stages,train_rows,root=ROOT):
    import jsonschema
    tasks={r['id']:r for r in train_rows}
    candidates=[]; exclusions=Counter(); source_receipts={}
    validators={}
    for name in stages:
        stage=root/name
        verified=verify(stage)
        protocol=json.loads((stage/'protocol.json').read_text())
        source_receipts[name]={'files':verified['files'],'report_sha256':digest(stage/'report.json'),
                              'manifest_sha256':digest(stage/'artifact_manifest.json')}
        for item in protocol['schedule']:
            for variant in item['variants']:
                if variant not in ('G','Teacher'):continue
                if item['task']['id'] not in tasks:continue
                if item['task']['sha256']!=tasks[item['task']['id']]['sha256']:raise ValueError('Training task hash drift')
                episode=episode_directory(stage,item,variant)
                summary=json.loads((episode/'summary.json').read_text())
                if not summary['success'] or summary['infrastructure_error'] or summary['task_failure']:
                    exclusions['unsuccessful_episode']+=1;continue
                calls=json.loads((episode/'model_calls.json').read_text())
                audit=json.loads((episode/'harness_audit.json').read_text())
                if len(calls)!=len(audit['structured']):
                    exclusions['ambiguous_turn_http_alignment']+=1;continue
                proposal={row['turn']:row for row in audit['proposals']}
                calls_sha,summary_sha=digest(episode/'model_calls.json'),digest(episode/'summary.json')
                samples=[]; actions=[]
                for turn,(call,structured) in enumerate(zip(calls,audit['structured']),1):
                    evidence=proposal.get(turn,{})
                    if evidence.get('blocked') or evidence.get('simulator_error') or structured.get('validation_error'):
                        exclusions['incorrect_action_target']+=1;continue
                    try:
                        messages,target,action=action_target(call)
                        schema=call['request']['response_format']['json_schema']['schema']
                        schema_key=text_hash(json.dumps(schema,sort_keys=True))
                        if schema_key not in validators:
                            jsonschema.Draft202012Validator.check_schema(schema)
                            validators[schema_key]=jsonschema.Draft202012Validator(schema)
                        validators[schema_key].validate(json.loads(target))
                    except (ValueError,KeyError,jsonschema.ValidationError):
                        exclusions['invalid_or_truncated_target']+=1;continue
                    normalized=structured.get('normalized_action',{})
                    if normalized.get('action')!=action['tool']:
                        raise ValueError('Action audit mismatch')
                    if json.loads(normalized['action_input'])!=action['arguments']:
                        raise ValueError('Action arguments audit mismatch')
                    if action['tool']!='finish' and not evidence.get('reached_executor'):
                        exclusions['missing_execution_receipt']+=1;continue
                    samples.append({'task_id':item['task']['id'],'category':item['task']['query_type']+':'+item['task']['case'],
                        'messages':messages,'target':target,'source':str(episode.relative_to(root)),
                        'turn':turn,'model_calls_sha256':calls_sha,
                        'summary_sha256':summary_sha,'supervision':'Current assistant target only; public prefix masked',
                        'teacher_judge_shared_model':variant=='Teacher'})
                    actions.append(action)
                if not samples or actions[-1]['tool']!='finish':
                    exclusions['missing_valid_finish_target']+=1;continue
                candidates.append({'task_id':item['task']['id'],'samples':samples,
                    'signature':text_hash(json.dumps(actions,sort_keys=True)),
                    'rank':(sum(p.get('blocked',False) or p.get('simulator_error',False) for p in audit['proposals']),
                            len(calls),summary['actor_tokens'],str(episode))})
    chosen=[];per_task=Counter();signatures=set()
    for episode in sorted(candidates,key=lambda r:r['rank']):
        key=episode['task_id'],episode['signature']
        if key in signatures or per_task[episode['task_id']]>=2:
            exclusions['duplicate_or_task_cap']+=1;continue
        signatures.add(key);per_task[episode['task_id']]+=1
        chosen.extend(episode['samples'])
    return chosen,{'source_receipts':source_receipts,'exclusions':dict(exclusions),'task_episode_counts':dict(per_task)}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stages',nargs='+',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--model',default='../models/Qwen3.5-9B')
    args=parser.parse_args()
    from transformers import AutoTokenizer
    rows=json.loads((ROOT/'configs/sft-pilot/train.json').read_text())['tasks']
    samples,audit=curate(args.stages,rows)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/args.model,local_files_only=True)
    eligible=[];lengths=[];target_tokens=0
    for sample in samples:
        try: tokenized=tokenized_target(tokenizer,sample['messages'],sample['target'],16384)
        except ValueError:
            audit['exclusions']['overlength']=audit['exclusions'].get('overlength',0)+1;continue
        eligible.append(sample);lengths.append(len(tokenized['input_ids']));target_tokens+=tokenized['target_tokens']
    path=out/'train.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in eligible),encoding='utf-8')
    counts=Counter(r['category'] for r in eligible)
    independent=len({r['task_id'] for r in eligible})
    audit.update({'samples':len(eligible),'independent_tasks':independent,'category_targets':dict(sorted(counts.items())),
        'teacher_targets':sum(r['teacher_judge_shared_model'] for r in eligible),
        'target_tokens':target_tokens,'max_input_tokens':max(lengths,default=0),'total_input_tokens':sum(lengths),
        'admitted':independent>=24 and len(eligible)>=120 and any(':feasible' in c for c in counts) and any(':infeasible' in c for c in counts),
        'data_sha256':digest(path),'train_manifest_sha256':digest(ROOT/'configs/sft-pilot/train.json'),
        'quality_limitations':'Official success plus legal executed targets is a filter, not complete semantic correctness. Teacher and judge share model; generator templates may cross splits. No private reasoning, judge text, hidden goals or eval tasks exported.'})
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('samples','independent_tasks','category_targets','admitted','total_input_tokens')}))
