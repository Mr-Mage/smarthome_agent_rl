"""Replay prompt identity, actual source slices, evidence and separated costs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read, sha
from scripts.run_prompt_semantic_ablation import prepare
from smarthome_agent_rl.benchmarks.runner import save, valid_usage
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_diagnosis import parse, schema
from smarthome_agent_rl.semantic_prompt_ablation import ARMS, PROMPTS, request, evaluate


def audit(run,*,write_receipt=True):
    run=Path(run);problems=[];manifest=read(run/'artifact_manifest.json')
    for name,value in manifest.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or sha(path)!=value:
            problems.append('Artifact SHA/path differs: '+name)
    protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/public-prompt-semantic-ablation.json'):
        problems.append('Frozen repository configuration differs')
    try:inputs,baseline,labels,coverage,parent_audit=prepare(config,run/'n74-reference')
    except (ValueError,KeyError,StopIteration) as exc:
        return {'verified':False,'problems':problems+['Parent/source derivation failed: '+str(exc)]}
    if inputs!=protocol['inputs'] or digest(inputs)!=config['inputs_sha256'] or coverage!=protocol['coverage']:
        problems.append('Original context/selection/cutoffs differ')
    if protocol['schema']!=schema() or protocol['prompts']!=PROMPTS:
        problems.append('Frozen schema/prompt identity differs')
    records=[];expected_paths=set()
    for index,item in enumerate(inputs):
        for arm in ARMS:
            path=run/'records'/f'{index:03d}'/(arm+'.json');expected_paths.add(path.resolve())
            row=read(path);records.append(row);call=row['call'];actor=config['actors'][index%len(config['actors'])]
            if row['id']!=item['id'] or row['arm']!=arm or row['actor_id']!=actor['id'] or call['endpoint']!=actor['endpoint'] or call['body']!=request(config,item,arm):
                problems.append('Request identity/body differs: '+str(path))
            decision,error=None,None
            if call['error'] is None:
                raw=json.loads(call['raw_response'])
                if raw!=call['response'] or raw['model']!=config['model'] or raw['choices'][0]['message']['content']!=call['text'] or raw.get('usage')!=call['usage']:
                    problems.append('Raw response/text/usage differs: '+str(path))
                try:decision=parse(call['text'])
                except (ValueError,TypeError,KeyError) as exc:error=str(exc)
            if decision!=row['decision'] or error!=row['parse_error']:
                problems.append('Retained model decision differs')
    if expected_paths!={p.resolve() for p in (run/'records').rglob('*.json')}:
        problems.append('Missing/unexpected candidate records')
    calculated=evaluate(config,inputs,records,baseline,labels);report=read(run/'report.json')
    if any(report[key]!=value for key,value in calculated.items()):
        problems.append('Reported cost/count/gates/evidence/transitions differ')
    resources=read(run/'services/lifecycle.json')
    if report['resources']!=resources or resources['error'] is not None or len(resources['stopped_pids'])!=4:
        problems.append('Owned service cleanup differs')
    probes=read(run/'services/probe-receipts.json')
    result={'verified':not problems,'problems':problems,'original_artifacts':len(manifest),
        'source_episodes':len(coverage),'proposals':len(inputs),'new_review_requests':len(records),
        'reused_review_requests':len(baseline),'new_model_tokens':calculated['new_model_tokens'],
        'parent_audit':parent_audit,'source_commit':protocol['source_commit'],'auditor_sha256':sha(Path(__file__)),
        'probes':{'requests':len(probes),'tokens':sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
                  'errors':sum(p['error'] is not None for p in probes),'missing_usage':sum(not valid_usage(p['usage']) for p in probes)},
        'scope':'Source/prompt/HTTP/cost/literal-evidence checks;not independent semantics,quote entailment or native admission'}
    if write_receipt:save(run/'independent-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    result=audit(args.run);print(json.dumps(result))
    if not result['verified']:raise SystemExit(1)
