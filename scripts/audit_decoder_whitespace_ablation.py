"""Recompute exact frozen requests, raw outputs and all screening/cost totals."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from scripts.run_decoder_whitespace_ablation import prepare
from smarthome_agent_rl.benchmarks.runner import save,valid_usage
from smarthome_agent_rl.evidence_order_ablation import ordered_digest
from smarthome_agent_rl.decoder_whitespace_ablation import ARMS,actor_for,request,evaluate,parse_response


def audit(run,parent=None,*ancestors,write_receipt=True):
    run=Path(run);count=check_manifest(run,'artifact_manifest.json',sha(run/'artifact_manifest.json'))
    protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/decoder-whitespace-ablation.json'):raise ValueError('Frozen repository config differs')
    inputs,labels,schemas,requests,references,historical,checked=prepare(config,parent,*ancestors)
    for name,value in {'inputs':inputs,'labels':labels,'schemas':schemas,'requests':requests,'references':references,'parent_checks':checked}.items():
        if ordered_digest(protocol[name])!=ordered_digest(value):raise ValueError('Frozen ordered sources differ: '+name)
    records=[];expected=set()
    for index,item in enumerate(inputs):
        for arm in ARMS:
            actor=actor_for(config,index,arm)
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
    resources=read(run/'services/lifecycle.json')
    if report['resources']!=resources or resources['error'] is not None or set(resources['groups'])!={'allow','compact'}:
        raise ValueError('Owned decoder group lifecycle differs')
    aggregate={key:0 for key in ('requests','failed','tokens','missing_usage')}
    for policy,offset,compact in [('allow',0,False),('compact',2,True)]:
        directory=run/'services'/policy;group=read(directory/'lifecycle.json');deployments=read(directory/'deployments.json')
        expected=config['actors'][offset:offset+2]
        if resources['groups'][policy]!=group or group['error'] is not None or group['stopped_pids']!=[r['pid'] for r in deployments] or [r['id'] for r in deployments]!=[a['id'] for a in expected]:
            raise ValueError('Owned decoder actor IDs/lifecycle differ')
        service_config=read(directory/'config.json')
        if service_config!={**config,'actors':expected,'actor_extra_args':config['actor_extra_args']+['--structured-outputs-config',json.dumps({'backend':'xgrammar','disable_any_whitespace':compact},separators=(',',':'))]}:
            raise ValueError('Decoder service configuration differs')
        for deployment in deployments:
            argv=deployment['argv'];marker=argv.index('--structured-outputs-config')
            if marker!=len(argv)-2 or argv.count('--structured-outputs-config')!=1 or json.loads(argv[marker+1])!={'backend':'xgrammar','disable_any_whitespace':compact}:
                raise ValueError('Decoder executable arguments differ')
            log=(directory/('actor'+str(deployment['id'])+'.log')).read_text()
            if "StructuredOutputsConfig(backend='xgrammar', disable_any_whitespace="+str(compact) not in log:
                raise ValueError('Running engine did not acknowledge frozen decoder policy')
        probes=read(directory/'probe-receipts.json')
        cost={'requests':len(probes),'failed':sum(p['error'] is not None for p in probes),
              'tokens':sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
              'missing_usage':sum(not valid_usage(p['usage']) for p in probes)}
        if cost!=group['probes'] or cost['requests']!=2 or cost['failed'] or cost['missing_usage']:
            raise ValueError('Probe cost differs')
        for key,value in cost.items():aggregate[key]+=value
    if aggregate!=resources['probes'] or resources['reserved_h100_gpu_seconds']!=sum(g['reserved_h100_gpu_seconds'] for g in resources['groups'].values()):
        raise ValueError('Aggregate reserved GPU/probe costs differ')
    for index,(path,checksum) in enumerate(config['decoder_source']['source_sha256'].items()):
        if sha(run/'decoder-source'/f'{index}.py')!=checksum:
            raise ValueError('Frozen decoder source snapshot differs')
    probe_cost=aggregate
    receipt={'verified':True,'artifacts':count,'records':len(records),'parent_checks':checked,'new_tokens':calculated['new_tokens'],
        'probes':probe_cost,'source_commit':protocol['source_commit'],'auditor_sha256':sha(Path(__file__)),
        'scope':'Original requests/HTTP/output;frozen decoder executable flags,paired exactHTTP bodies,rawoutputs andcosts;labels unchanged,not semantic truth orSR'}
    if write_receipt:save(run/'independent-audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();print(audit(args.run))
