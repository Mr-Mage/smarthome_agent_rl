"""Real existing training-task request: verify native reasoning + structured actions."""
import argparse, copy, json, sys, time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.variant_runtime import runtime

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    train={r['id'] for r in json.loads((ROOT/'configs/sft-pilot/train.json').read_text())['tasks']}
    source=next(p for p in (ROOT/'runs/guard-v3/n15-v1/dev/seed42').rglob('model_calls.json')
                if p.parent.parent.name=='G' and p.parent.parent.parent.name in train)
    request=json.loads(source.read_text())[0]['request']
    records=[]
    with httpx.Client(trust_env=False,timeout=300) as client:
        for variant in ('G','GThinking'):
            options=runtime(config,variant,config['workflows'][0])
            body=copy.deepcopy(request)
            generation=options['generation']
            body.update({k:v for k,v in generation.items() if k!='extra_body'})
            body.update(generation.get('extra_body',{}))
            body['model']=options['served_model']
            started=time.monotonic()
            r=client.post(options['model_endpoint']+'/chat/completions',json=body)
            r.raise_for_status(); response=r.json()
            records.append({'variant':variant,'request':body,'response':response,'seconds':time.monotonic()-started})
            output=ROOT/args.output
            output.write_text(json.dumps({'source':str(source.relative_to(ROOT)),'source_sha256':digest(source),
                 'records':records,'scope':'No tools executed; existing train request only; count tokens separately'},indent=2))
            msg=response['choices'][0]['message']
            action=json.loads(msg['content'])
            if not isinstance(action.get('call',{}).get('arguments'),dict):raise ValueError('Invalid structured output')
            reasoning=msg.get('reasoning') or msg.get('reasoning_content')
            if variant=='GThinking' and not reasoning:raise ValueError('Thinking requested but no native reasoning evidence')
    print(json.dumps({'passed':True,'variants':[r['variant'] for r in records]}))
