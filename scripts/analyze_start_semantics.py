"""Frozen N28 mechanism gates and complete prompt provenance from public traces."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import episode_directory
from smarthome_agent_rl.start_semantics import observed_cycle_devices, semantics_prompt


def missing_start(audit, messages, target):
    """Descriptive omission, not proof that a cycle should be forced on an infeasible task."""
    stopped = False
    for message in messages:
        if message['role'] != 'user' or not message['content'].startswith('observation:'):
            continue
        try:
            response = json.loads(message['content'].split(':',1)[1])
        except (ValueError, TypeError):
            continue
        data = response.get('data')
        if response.get('status',{}).get('code') == 200 and not response.get('error') and isinstance(data,dict) and data.get('device_id') == target:
            stopped |= data.get('endpoints',{}).get('1',{}).get('clusters',{}).get('OperationalState',{}).get('attributes',{}).get('OperationalState',{}).get('value') == 0
    accepted = []
    for row in audit['proposals']:
        if row.get('blocked') or row.get('simulator_error') or not row.get('reached_executor'):
            continue
        if row['tool']=='schedule_workflow':
            accepted.extend(row['arguments']['steps'])
        else:
            accepted.append({'tool':row['tool'],'args':row['arguments']})
    power_or_configured = start = False
    for action in accepted:
        args = action['args']
        if args.get('device_id') != target:
            continue
        cluster = args.get('cluster_id')
        start |= action['tool']=='execute_command' and cluster=='OperationalState' and args.get('command_id') in ('Start','Resume')
        power_or_configured |= (action['tool']=='execute_command' and cluster=='OnOff' and args.get('command_id')=='On') or (
            cluster in ('LaundryWasherMode','LaundryDryerControls','DishwasherMode') and action['tool'] in ('execute_command','write_attribute'))
    return {'observed_stopped':stopped,'accepted_power_or_configuration':power_or_configured,
        'accepted_or_registered_start':start,'omission':bool(stopped and power_or_configured and not start)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    run=ROOT/args.run
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    root_protocol=read(run/'protocol.json')
    config=root_protocol['config']
    targets=config['node_experiment']['motivating_targets']
    counts={v:{'omissions':0,'observed_stopped_targets':0,'motivating_feasible_successes':0} for v in ('G','GS')}
    details=[]
    provenance={'episodes':0,'model_calls':0,'prompts_used':0,'passed':True}
    for phase in ('smoke','dev'):
        stage=run/phase
        protocol=read(stage/'protocol.json')
        report=read(stage/'report.json')
        assert report['verified'] and set(report['arms'])=={'G','GS'}
        for item in protocol['schedule']:
            first_requests={}
            contracts={}
            for variant in ('G','GS'):
                path=episode_directory(stage,item,variant)
                calls=read(path/'model_calls.json')
                audit=read(path/'harness_audit.json')
                summary=read(path/'summary.json')
                contracts[variant]=read(path/'contract.json')['config']
                records=audit['start_semantics']
                assert len(records)==(len(calls) if variant=='GS' else 0)
                provenance['episodes']+=1
                provenance['model_calls']+=len(calls)
                first_requests[variant]=calls[0]['request']
                for i,call in enumerate(calls):
                    request=call['request']
                    assert request['model']==call['response']['model']==config['actor_model']
                    wire=request['messages']
                    devices=observed_cycle_devices(wire)
                    prompt=semantics_prompt(devices) if variant=='GS' else None
                    if variant=='GS':
                        assert records[i]['devices']==devices and records[i]['prompt']==prompt
                        if prompt:
                            assert wire[-1]=={'role':'user','content':prompt}
                            provenance['prompts_used']+=1
                        else:
                            assert not any('PUBLIC CYCLE COMMAND SEMANTICS' in m['content'] for m in wire)
                    else:
                        assert not any('PUBLIC CYCLE COMMAND SEMANTICS' in m['content'] for m in wire)
                task_id=summary['task_id']
                if phase=='dev' and task_id in targets:
                    messages=calls[-1]['request']['messages']
                    boundary=max(i for i,m in enumerate(messages) if 'This is your actual task.' in m['content'])
                    result=missing_start(audit,messages[boundary+1:],targets[task_id])
                    counts[variant]['omissions']+=int(result['omission'])
                    counts[variant]['observed_stopped_targets']+=int(result['observed_stopped'])
                    if '_feasible_' in task_id:
                        counts[variant]['motivating_feasible_successes']+=int(summary['success'])
                    details.append({'task_id':task_id,'variant':variant,'seed':summary['actor_seed'],
                        'success':summary['success'],**result,'path':str(path.relative_to(ROOT)),
                        'audit_sha256':digest(path/'harness_audit.json'),'calls_sha256':digest(path/'model_calls.json')})
            assert first_requests['G']==first_requests['GS']
            for key in ('model_endpoint','served_model','model_seed','generation','max_steps',
                        'recovery_per_action','extra_queries_max','judge_endpoint'):
                assert contracts['G'][key]==contracts['GS'][key],key
    assert len(details)==len(targets)*len(config['actor_seeds'])*2
    report=read(run/'dev/report.json')
    g,gs=report['arms']['G'],report['arms']['GS']
    gates=config['node_experiment']['gates']
    n=counts['G']['omissions']
    reduction=1-counts['GS']['omissions']/n if n else None
    ratio=gs['all_totals']['actor_tokens']/g['all_totals']['actor_tokens']
    checks={'mechanism_observed':n>=gates['control_omissions_min'],
        'public_evidence_coverage':counts['GS']['observed_stopped_targets']>=counts['G']['observed_stopped_targets'],
        'omission_reduction':reduction is not None and reduction>=gates['omission_reduction_min'],
        'success_rate':gs['success_rate']-g['success_rate']>=gates['sr_delta_min'],
        'motivating_feasible_success':counts['GS']['motivating_feasible_successes']-counts['G']['motivating_feasible_successes']>=gates['motivating_feasible_success_gain_min'],
        'illegal_execution':gs['all_totals']['invalid_reached_executor']<=g['all_totals']['invalid_reached_executor'],
        'actor_tokens':ratio<=gates['actor_token_ratio_max']}
    output={'counts':counts,'checks':checks,'winner':'GS' if all(checks.values()) else 'G',
        'n29_admitted':all(checks.values()),'omission_reduction':reduction,'actor_token_ratio':ratio,
        'provenance':provenance,'details':details,'config_sha256':root_protocol['config_sha256'],
        'limits':'Exposed enriched development tasks. Omission is descriptive; registered Start is not running proof; honest infeasible refusals may remove actions. All gates including feasible SR required. No corrected scores.'}
    (run/'selection.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:output[key] for key in ('counts','checks','winner','n29_admitted','omission_reduction','actor_token_ratio','provenance')}))


if __name__=='__main__':
    main()
