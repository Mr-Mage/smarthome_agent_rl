"""Recalculate N74 contexts, original cutoffs, HTTP bodies and new/reused costs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read, sha, request
from scripts.run_context_semantic_ablation import prepare
from smarthome_agent_rl.benchmarks.runner import save, valid_usage
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_context_ablation import ARMS, evaluate
from smarthome_agent_rl.semantic_diagnosis import parse, schema


def audit(run):
    run=Path(run);problems=[]
    manifest=read(run/'artifact_manifest.json')
    for name,value in manifest.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or sha(path)!=value:
            problems.append('Artifact SHA/path differs: '+name)
    protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/public-context-semantic-ablation.json'):
        problems.append('Frozen repository configuration differs')
    try:
        inputs,baseline,labels,coverage,parent_audit=prepare(config,run/'n72-reference')
    except (ValueError,KeyError,StopIteration) as exc:
        problems.append('Parent/source derivation failed: '+str(exc))
        return {'verified':False,'problems':problems}
    if inputs!=protocol['inputs'] or digest(inputs)!=config['inputs_sha256'] or coverage!=protocol['coverage'] or schema()!=protocol['schema']:
        problems.append('Retained pre-action source/input/schema differs')
    records=[];expected_paths=set()
    for index,item in enumerate(inputs):
        for arm in ARMS:
            path=run/'records'/f'{index:03d}'/(arm+'.json');expected_paths.add(path.resolve())
            row=read(path);records.append(row);call=row['call']
            actor=config['actors'][index%len(config['actors'])]
            if row['id']!=item['id'] or row['arm']!=arm or row['actor_id']!=actor['id'] or call['endpoint']!=actor['endpoint'] or call['body']!=request(config,item,arm):
                problems.append('Request identity/body differs: '+str(path))
            decision,error=None,None
            if call['error'] is None:
                raw=json.loads(call['raw_response'])
                if raw!=call['response'] or raw['model']!=config['model'] or raw['choices'][0]['message']['content']!=call['text'] or raw.get('usage')!=call['usage']:
                    problems.append('Raw HTTP response/text/usage differs: '+str(path))
                try:decision=parse(call['text'])
                except (ValueError,TypeError,KeyError) as exc:error=str(exc)
            if decision!=row['decision'] or error!=row['parse_error']:
                problems.append('Decision differs from original model response')
    if expected_paths!={p.resolve() for p in (run/'records').rglob('*.json')}:
        problems.append('Unexpected/missing candidate records')
    recalculated=evaluate(config,records,baseline,labels);report=read(run/'report.json')
    if any(report[k]!=v for k,v in recalculated.items()):
        problems.append('Reported new/reused cost/count/gates/transitions differ')
    resources=read(run/'services/lifecycle.json')
    if report['resources']!=resources or resources['error'] is not None or len(resources['stopped_pids'])!=4:
        problems.append('Owned service cleanup differs')
    probes=read(run/'services/probe-receipts.json')
    result={'verified':not problems,'problems':problems,'original_artifacts':len(manifest),
        'source_episodes':len(coverage),'proposals':len(inputs),'new_review_requests':len(records),
        'reused_review_requests':len(baseline),'new_model_tokens':recalculated['new_model_tokens'],
        'parent_audit':parent_audit,'source_commit':protocol['source_commit'],'auditor_sha256':sha(Path(__file__)),
        'probes':{'requests':len(probes),'tokens':sum(r['usage']['total_tokens'] for r in probes if valid_usage(r['usage'])),
                  'errors':sum(r['error'] is not None for r in probes),'missing_usage':sum(not valid_usage(r['usage']) for r in probes)},
        'scope':'Engineering provenance and developer diagnostic gates only;not independent semantics,SR or native admission'}
    save(run/'independent-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    result=audit(args.run);print(json.dumps(result))
    if not result['verified']:raise SystemExit(1)
