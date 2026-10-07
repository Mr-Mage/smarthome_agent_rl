"""Recompute exact frozen requests, raw outputs and all screening/cost totals."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from scripts.run_identity_evidence_ablation import prepare
from smarthome_agent_rl.benchmarks.runner import save,valid_usage
from smarthome_agent_rl.evidence_order_ablation import ordered_digest
from smarthome_agent_rl.identity_evidence_ablation import ARMS,request,evaluate,parse_response


def audit(run,parent=None,n84=None,n83=None,n82=None,n80=None,n79=None,n76=None,n78=None,*,write_receipt=True):
    run=Path(run);count=check_manifest(run,'artifact_manifest.json',sha(run/'artifact_manifest.json'))
    protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/identity-evidence-ablation.json'):raise ValueError('Frozen repository config differs')
    inputs,labels,schemas,requests,identities,historical,checked=prepare(config,parent,n84,n83,n82,n80,n79,n76,n78)
    for name,value in {'inputs':inputs,'labels':labels,'schemas':schemas,'requests':requests,'identities':identities,'parent_checks':checked}.items():
        if ordered_digest(protocol[name])!=ordered_digest(value):raise ValueError('Frozen ordered sources differ: '+name)
    records=[];expected=set()
    for index,item in enumerate(inputs):
        actor=config['actors'][index%4]
        for arm in ARMS:
            path=run/'records'/f'{index:03d}'/(arm+'.json');expected.add(path.resolve());row=read(path);call=row['call']
            if (row['id'],row['arm'],row['actor_id'],call['endpoint'])!=(item['id'],arm,actor['id'],actor['endpoint']) or ordered_digest(call['body'])!=ordered_digest(request(config,item,arm)):
                raise ValueError('Frozen request/endpoint/identity differs')
            model,decision,error=None,None,None
            if call['error'] is None:
                raw=json.loads(call['raw_response']);choice=raw['choices'][0]
                if raw!=call['response'] or raw['model']!=config['model'] or choice['message']['content']!=call['text'] or raw.get('usage')!=call['usage'] or choice.get('finish_reason')!=call['finish_reason']:
                    raise ValueError('Original HTTP/output/usage differs')
                model,decision,error=parse_response(item['context'],call['text'],arm)
            if (model,decision,error)!=(row['model_decision'],row['decision'],row['parse_error']):raise ValueError('Original decision differs')
            records.append(row)
    if expected!={p.resolve() for p in (run/'records').rglob('*.json')}:raise ValueError('Missing/unexpected output files')
    calculated=evaluate(config,inputs,records,historical,labels);report=read(run/'report.json')
    if any(report[key]!=value for key,value in calculated.items()):raise ValueError('Cost/counts/screen differ')
    resources=read(run/'services/lifecycle.json');deployments=read(run/'services/deployments.json')
    if report['resources']!=resources or resources['error'] is not None or resources['stopped_pids']!=[r['pid'] for r in deployments] or [r['id'] for r in deployments]!=[a['id'] for a in config['actors']]:
        raise ValueError('Owned actor lifecycle differs')
    probes=read(run/'services/probe-receipts.json');probe_cost={'requests':len(probes),'failed':sum(p['error'] is not None for p in probes),
        'tokens':sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
        'missing_usage':sum(not valid_usage(p['usage']) for p in probes)}
    if probe_cost!=resources['probes'] or probe_cost['requests']!=4 or probe_cost['failed'] or probe_cost['missing_usage']:
        raise ValueError('Probe cost differs')
    receipt={'verified':True,'artifacts':count,'records':len(records),'parent_checks':checked,'new_tokens':calculated['new_tokens'],
        'probes':probe_cost,'source_commit':protocol['source_commit'],'auditor_sha256':sha(Path(__file__)),
        'scope':'Original requests/HTTP/output;verified public metadata projection andtext-only dereference,labels unchanged;not semantic truth orSR'}
    if write_receipt:save(run/'independent-audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();print(audit(args.run))
