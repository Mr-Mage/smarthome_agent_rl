"""Verify390 original HTTP receipts,156 assemblies,unchanged gates and costs."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from scripts.run_independent_review_ablation import prepare,build_record
from smarthome_agent_rl.benchmarks.runner import save,valid_usage
from smarthome_agent_rl.evidence_order_ablation import DIMENSIONS,ordered_digest
from smarthome_agent_rl.independent_review_ablation import ARMS,COMPONENTS,actor_for,request,evaluate


def audit(run,parent=None,*ancestors,write_receipt=True):
    run=Path(run);count=check_manifest(run,'artifact_manifest.json',sha(run/'artifact_manifest.json'))
    protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/independent-review-ablation.json'):raise ValueError('Frozen repository config differs')
    inputs,labels,schemas,requests,historical,checked=prepare(config,parent,*ancestors)
    for name,value in {'inputs':inputs,'labels':labels,'schemas':schemas,'requests':requests,'parent_checks':checked}.items():
        if ordered_digest(protocol[name])!=ordered_digest(value):raise ValueError('Frozen ordered sources differ:'+name)
    records=[];calls_expected=set();records_expected=set()
    for index,item in enumerate(inputs):
        actor=actor_for(config,index);wrappers={}
        for component in COMPONENTS:
            path=run/'calls'/f'{index:03d}'/(component+'.json');calls_expected.add(path.resolve());wrapper=read(path);call=wrapper['call']
            if (wrapper['id'],wrapper['component'],wrapper['actor_id'],call['endpoint'])!=(item['id'],component,actor['id'],actor['endpoint']) or ordered_digest(call['body'])!=ordered_digest(request(config,item,component)):
                raise ValueError('Frozen component request/endpoint/identity differs')
            moments=[wrapper[k] for k in ('submitted_monotonic','dispatch_monotonic','finished_monotonic')]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in moments) or not moments[0]<=moments[1]<=moments[2] or not 0<=call['request_seconds']<=moments[2]-moments[1]+.001:
                raise ValueError('Original queue/HTTP timing receipt differs')
            if call['error'] is None:
                raw=json.loads(call['raw_response']);choice=raw['choices'][0]
                if raw!=call['response'] or raw['model']!=config['model'] or choice['message']['content']!=call['text'] or raw.get('usage')!=call['usage'] or choice.get('finish_reason')!=call['finish_reason']:
                    raise ValueError('Original HTTP/output/usage differs')
            wrappers[component]=wrapper
        for arm,components in [('joint',('joint',)),('independent',DIMENSIONS)]:
            path=run/'records'/f'{index:03d}'/(arm+'.json');records_expected.add(path.resolve())
            stored,row=build_record(item,index,arm,{name:wrappers[name] for name in components})
            if read(path)!=stored or row['actor_id']!=actor['id']:raise ValueError('Lossless assembly/model decision/timing differs')
            records.append(row)
    if calls_expected!={p.resolve() for p in (run/'calls').rglob('*.json')} or records_expected!={p.resolve() for p in (run/'records').rglob('*.json')}:
        raise ValueError('Missing/unexpected raw calls orreview files')
    calculated=evaluate(config,inputs,records,historical,labels);report=read(run/'report.json')
    if any(report[key]!=value for key,value in calculated.items()):raise ValueError('Cost/counts/screen differ')
    directory=run/'services';resources=read(directory/'lifecycle.json');deployments=read(directory/'deployments.json')
    if report['resources']!=resources or resources['error'] is not None or resources['stopped_pids']!=[r['pid'] for r in deployments] or [r['id'] for r in deployments]!=[a['id'] for a in config['actors']] or read(directory/'config.json')!=config:
        raise ValueError('Owned actors/configuration/lifecycle differ')
    for deployment in deployments:
        argv=deployment['argv'];marker=argv.index('--structured-outputs-config')
        if marker!=len(argv)-2 or argv.count('--structured-outputs-config')!=1 or json.loads(argv[marker+1])!={'backend':'xgrammar','disable_any_whitespace':True}:
            raise ValueError('Actual compact decoder arguments differ')
        if "StructuredOutputsConfig(backend='xgrammar', disable_any_whitespace=True" not in (directory/('actor'+str(deployment['id'])+'.log')).read_text(encoding='utf-8'):
            raise ValueError('Engine did not acknowledge compact policy')
    probes=read(directory/'probe-receipts.json')
    probe_cost={'requests':len(probes),'failed':sum(p['error'] is not None for p in probes),
        'tokens':sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
        'missing_usage':sum(not valid_usage(p['usage']) for p in probes)}
    if probe_cost!=resources['probes'] or probe_cost['requests']!=4 or probe_cost['failed'] or probe_cost['missing_usage']:
        raise ValueError('Probe accounting differs')
    for index,(_,checksum) in enumerate(config['decoder_source']['source_sha256'].items()):
        if sha(run/'decoder-source'/f'{index}.py')!=checksum:raise ValueError('Decoder snapshot differs')
    receipt={'verified':True,'artifacts':count,'reviews':len(records),'actual_http_requests':calculated['new_model_requests'],
        'parent_checks':checked,'new_tokens':calculated['new_tokens'],'probes':probe_cost,'source_commit':protocol['source_commit'],
        'auditor_sha256':sha(Path(__file__)),'scope':'Original raw component HTTP,lossless assembly,unchanged gates,actual engineflags andall costs;not independent semantic truth orSR'}
    if write_receipt:save(run/'independent-audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();print(audit(args.run))
