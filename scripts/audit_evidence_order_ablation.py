"""Order-sensitive request audit plus original HTTP and derived metrics."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_effect_evidence_ablation import verify_record as verify_original
from scripts.run_action_semantic_diagnosis import read,sha
from scripts.run_evidence_consistency_audit import check_manifest
from scripts.run_evidence_order_ablation import prepare
from smarthome_agent_rl.benchmarks.runner import save,valid_usage
from smarthome_agent_rl.effect_evidence_ablation import PROMPT_BY_ARM
from smarthome_agent_rl.evidence_order_ablation import ARMS,ordered_digest,request,evaluate


def verify_record(row,item,arm,actor,config):
    if row['arm']!=arm or ordered_digest(row['call']['body'])!=ordered_digest(request(config,item,arm)):
        raise ValueError('Frozen request arm or object order differs')
    # Both schemas are semantically identical to combined; reuse all original
    # HTTP/model/text/usage/finish/parse checks after checking our actual order.
    verify_original({**row,'arm':'combined'},item,'combined',actor,config)


def audit(run,parent=None,n79=None,n76=None,n78=None,*,write_receipt=True):
    run=Path(run);count=check_manifest(run,'artifact_manifest.json',sha(run/'artifact_manifest.json'))
    protocol=read(run/'protocol.json');config=protocol['config']
    if config!=read(ROOT/'configs/evidence-order-ablation.json'):raise ValueError('Frozen repository config differs')
    inputs,labels,schemas,historical,checked=prepare(config,parent,n79,n76,n78)
    for name,value in {'inputs':inputs,'labels':labels,'schemas':schemas,'parent_checks':checked,'prompt':PROMPT_BY_ARM['combined']}.items():
        if name=='schemas':
            equal=ordered_digest(protocol[name])==ordered_digest(value)
        else:equal=protocol[name]==value
        if not equal:raise ValueError('Source/prompt/ordered schema differs: '+name)
    records=[];expected=set()
    for index,item in enumerate(inputs):
        for arm in ARMS:
            path=run/'records'/f'{index:03d}'/(arm+'.json');expected.add(path.resolve())
            row=read(path);verify_record(row,item,arm,config['actors'][index%4],config);records.append(row)
    if expected!={p.resolve() for p in (run/'records').rglob('*.json')}:raise ValueError('Missing/unexpected requests')
    calculated=evaluate(config,inputs,records,historical,labels);report=read(run/'report.json')
    if any(report[key]!=value for key,value in calculated.items()):raise ValueError('Cost/order/counts/screen/transition differs')
    resources=read(run/'services/lifecycle.json');deployments=read(run/'services/deployments.json')
    if resources!=report['resources'] or resources['error'] is not None or resources['stopped_pids']!=[r['pid'] for r in deployments] or [r['id'] for r in deployments]!=[a['id'] for a in config['actors']]:
        raise ValueError('Owned service cleanup differs')
    probes=read(run/'services/probe-receipts.json')
    probe_cost={'requests':len(probes),'failed':sum(p['error'] is not None for p in probes),
                'tokens':sum(p['usage']['total_tokens'] for p in probes if valid_usage(p['usage'])),
                'missing_usage':sum(not valid_usage(p['usage']) for p in probes)}
    if probe_cost!=resources['probes'] or probe_cost['requests']!=4 or probe_cost['failed'] or probe_cost['missing_usage']:
        raise ValueError('Probe costs differ')
    receipt={'verified':True,'artifacts':count,'records':len(records),'parent_checks':checked,
             'new_tokens':calculated['new_tokens'],'probes':probe_cost,'source_commit':protocol['source_commit'],
             'auditor_sha256':sha(Path(__file__)),'scope':'Order-sensitive source/HTTP/engineering audit only;not independent semantic truth or native benefit'}
    if write_receipt:save(run/'independent-audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(audit(args.run))
