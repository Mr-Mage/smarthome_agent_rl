"""Replay frozen real-action inputs, request identity, response and cost."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmarks.runner import save, valid_usage
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.semantic_context import digest, MUTATIONS
from smarthome_agent_rl.semantic_diagnosis import contexts, parse, schema
from scripts.run_action_semantic_diagnosis import ARMS, evaluate, request, read, sha


def audit(run):
    run=Path(run); problems=[]
    manifest=read(run/'artifact_manifest.json')
    for name,value in manifest.items():
        path=(run/name).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or sha(path)!=value:
            problems.append('Artifact SHA/path: '+name)
    protocol=read(run/'protocol.json'); config=protocol['config']; inputs=protocol['inputs']
    if schema()!=protocol['schema'] or digest(inputs)!=config['inputs_sha256']:
        problems.append('Public input/schema fingerprint differs')
    # The original full manifest, copied without alteration, binds the selected
    # raw traces. No original episode score/judge output is required or read.
    original=read(run/'source-evidence/artifact_manifest.json')
    if sha(run/'source-evidence/artifact_manifest.json')!=config['source_artifact_sha256']:
        problems.append('Retained original source manifest differs')
    source_protocol=read(run/'source-evidence/protocol.json')
    if source_protocol['commit']!=config['source_commit'] or sha(run/'source-evidence/protocol.json')!=original['protocol.json']:
        problems.append('Original source protocol identity differs')
    expected=[]; coverage=[]
    public_tasks={r['id']:r for r in read(ROOT/config['manifest'])['tasks']}
    for item in source_protocol['schedule']:
        if item['task']['id'] not in public_tasks:
            continue
        prefix=str(episode_directory(Path('source-stage'),item,'G').relative_to('source-stage')).replace('\\','/')
        ep=run/'source-evidence'/prefix
        for filename in ('contract.json','harness_audit.json'):
            if sha(ep/filename)!=original[prefix+'/'+filename]:
                problems.append('Original source receipt differs: '+prefix+'/'+filename)
        contract,trace=read(ep/'contract.json'),read(ep/'harness_audit.json')
        if contract['task_identity']!=item['task'] or item['task']!=public_tasks[item['task']['id']]:
            problems.append('Frozen task identity differs')
        public=contract['public_context']
        if set(public)!={'query','user_location','current_time'} or contract['config']['variant_policies']['G']['verify']:
            problems.append('Unsupported public/source G boundary')
        count=0
        for proposal in trace['proposals']:
            if proposal['tool'] not in MUTATIONS or proposal.get('blocked') or not proposal.get('reached_executor'):
                continue
            variants,cutoff=contexts(public,proposal,trace['actual_observations'])
            expected.append({'id':f"{item['task']['id']}:seed{item['actor_seed']}:{proposal['action_id']}",
                'task_id':item['task']['id'],'source_seed':item['actor_seed'],'contexts':variants,
                'source_episode':prefix,'source_audit_sha256':sha(ep/'harness_audit.json'),
                'dispatch_observation_index':cutoff,'proposal_action_id':proposal['action_id'],
                'preaction_receipts_sha256':digest(trace['actual_observations'][:cutoff])})
            count+=1
        coverage.append({'task_id':item['task']['id'],'source_seed':item['actor_seed'],'selected':count,
            'all_proposals':len(trace['proposals']),'excluded_nonmutation_or_guard_blocked':len(trace['proposals'])-count})
    if expected!=inputs or coverage!=protocol['coverage']:
        problems.append('Public inputs/selection/cutoffs differ from retained pre-action receipts')
    records=[]; expected_paths=set()
    for index,item in enumerate(inputs):
        for arm in ARMS:
            path=run/'records'/f'{index:03d}'/(arm+'.json');expected_paths.add(path.resolve())
            row=read(path);records.append(row);call=row['call']
            actor=config['actors'][index%len(config['actors'])]
            endpoint=actor['endpoint'] if arm.startswith('9b_') else config['review_endpoint']
            actor_id=actor['id'] if arm.startswith('9b_') else None
            if row['id']!=item['id'] or row['arm']!=arm or row['actor_id']!=actor_id or call['endpoint']!=endpoint or call['body']!=request(config,item,arm):
                problems.append('HTTP identity/body differs: '+str(path))
            decision,error=None,None
            if call['error'] is None:
                raw=json.loads(call['raw_response'])
                if raw!=call['response'] or raw['model']!=call['body']['model'] or raw['choices'][0]['message']['content']!=call['text'] or raw.get('usage')!=call['usage']:
                    problems.append('Raw response/text/usage mismatch: '+str(path))
                try: decision=parse(call['text'])
                except (ValueError,TypeError,KeyError) as exc: error=str(exc)
            if decision!=row['decision'] or error!=row['parse_error']:
                problems.append('Decision differs from retained raw output')
    if expected_paths!={p.resolve() for p in (run/'records').rglob('*.json')}:
        problems.append('Unexpected or missing review records')
    recalculated=evaluate(config,records);report=read(run/'report.json')
    if any(report[k]!=v for k,v in recalculated.items()):
        problems.append('Reported count/cost/paired labels differ')
    probes=read(run/'services/probe-receipts.json')+[read(run/'review-probe.json')]
    result={'verified':not problems,'problems':problems,'original_artifacts':len(manifest),
        'source_episodes':len(coverage),'proposals':len(inputs),'review_requests':len(records),
        'source_commit':protocol['source_commit'],'auditor_sha256':sha(Path(__file__)),
        'probes':{'requests':len(probes),'tokens':sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
                  'errors':sum(p['error'] is not None for p in probes),'missing_usage':sum(not valid_usage(p['usage']) for p in probes)},
        'scope':'Source receipts/pre-action boundary/HTTP/cost only;model and developer opinions are not independent semantic truth'}
    save(run/'independent-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    result=audit(args.run);print(json.dumps(result))
    if not result['verified']:raise SystemExit(1)
