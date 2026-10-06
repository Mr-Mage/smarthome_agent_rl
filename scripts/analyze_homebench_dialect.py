"""Additional, read-only public enum dialect diagnosis; never requests models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmarks.homebench import HomeBenchAdapter,aggregate


def analyze(run,source):
    start=time.monotonic()
    config=json.loads((ROOT/'configs/homebench-dialect-diagnostic.json').read_text(encoding='utf-8'))
    lock=json.loads((ROOT/'configs/public-benchmarks.json').read_text(encoding='utf-8'))['HomeBench']
    if lock['commit']!=config['source_commit']:
        raise ValueError('Public source differs')
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    if report['status']!='complete' or 'service_cost' not in report:
        raise ValueError('Wait for complete source run and service cleanup')
    adapter=HomeBenchAdapter(source,lock)
    indexed={}
    for line in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['arm'] in ('F','FG'):
            key=(row['task_id'],row['arm'])
            if key in indexed:
                raise ValueError('Duplicate source episode')
            indexed[key]=row
    ids=sorted(t for t,a in indexed if a=='F')
    if len(ids)!=report['arms']['F']['episodes'] or any((t,'FG') not in indexed for t in ids):
        raise ValueError('Incomplete paired source')
    baseline,projected=[],[]
    wins,losses=[],[]
    counts={'public_bindings':0,'bound_episodes':0,'rejected_instructions':0,'uncovered_episodes':0}
    for task_id in ids:
        raw=indexed[task_id,'F']
        original=indexed[task_id,'FG']
        guarded=adapter.guard(task_id,raw['prediction'],symbolic_enums=True)
        score=adapter.score(task_id,guarded['prediction'])
        if raw['error']:
            score['exact_match']=False
        baseline.append(original['score'])
        projected.append(score)
        bindings=guarded.get('symbolic_bindings',[])
        counts['public_bindings']+=len(bindings)
        counts['bound_episodes']+=bool(bindings)
        counts['rejected_instructions']+=len(guarded['rejections'])
        counts['uncovered_episodes']+=bool(guarded['uncovered'])
        if score['exact_match']!=original['score']['exact_match']:
            case={'task_id':task_id,'category':raw['category'],'F':raw['prediction'],
                  'FG':original['prediction'],'dialect_projection':guarded['prediction'],
                  'public_bindings':bindings,'original_match':original['score']['exact_match'],
                  'projected_match':score['exact_match']}
            (wins if score['exact_match'] else losses).append(case)
    return {'config':config,'input_report_sha256':hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),
            'input_episodes_sha256':hashlib.sha256((run/'episodes.jsonl').read_bytes()).hexdigest(),
            'source':lock,'episodes':len(ids),'baseline_FG':aggregate(baseline),
            'additional_dialect_projection':aggregate(projected),'counts':counts,
            'wins':len(wins),'losses':len(losses),'win_cases':wins[:3],'loss_cases':losses[:3],
            'seconds':time.monotonic()-start,'actor_requests':0,'judge_requests':0,'tokens':0,'gpu_seconds':0,
            'scope':config['scope']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.run_dir,args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:
        file.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('episodes','counts','wins','losses','seconds')}))
