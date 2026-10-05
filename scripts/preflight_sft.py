"""Validate adapter routing and actual base/LoRA request behavior before evaluation."""
import argparse,copy,json,sys,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.variant_runtime import runtime

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    receipt=json.loads((ROOT/'runs/sft-pilot/n24-v1/completion.json').read_text())
    if not receipt['complete']:raise ValueError('Training incomplete')
    for name,sha in receipt['adapter_sha256'].items():
        if digest(ROOT/'runs/sft-pilot/n24-v1'/name)!=sha:raise ValueError('Adapter identity drift')
    source=json.loads((ROOT/'runs/sft-pilot/n23-v1/train.jsonl').read_text().splitlines()[0])
    results=[]
    with httpx.Client(trust_env=False,timeout=300) as client:
        for workflow in config['workflows']:
            models=client.get(f"http://127.0.0.1:{workflow['actor_port']}/v1/models").json()['data']
            if not any(r['id']=='smarthome-qwen35-9b-sft' for r in models):raise ValueError('Adapter model not registered')
            for variant in ('G','SFT9B'):
                options=runtime(config,variant,workflow)
                body={'model':options['served_model'],'messages':source['messages'],'seed':42,'logprobs':True,
                      **{k:v for k,v in options['generation'].items() if k!='extra_body'},
                      **options['generation'].get('extra_body',{})}
                started=time.monotonic();r=client.post(options['model_endpoint']+'/chat/completions',json=body)
                r.raise_for_status();value=r.json()
                results.append({'actor_id':workflow['id'],'variant':variant,'request':body,'response':value,'seconds':time.monotonic()-started})
                (ROOT/args.output).write_text(json.dumps({'records':results,'source_task':source['task_id'],
                    'scope':'Existing train input; no tools, judge or eval task; costs separate'},indent=2))
                if value['model']!=options['served_model']:raise ValueError('Wrong model routed')
                if not value['choices'][0]['message'].get('content'):raise ValueError('Empty output')
                if not value['choices'][0].get('logprobs',{}).get('content'):raise ValueError('No token likelihood evidence')
    for actor_id in range(4):
        pair=[r for r in results if r['actor_id']==actor_id]
        base=pair[0]['response']['choices'][0]['logprobs']['content']
        sft=pair[1]['response']['choices'][0]['logprobs']['content']
        if base==sft:raise ValueError('Base and adapter likelihoods identical; verify adapter activation')
    print(json.dumps({'passed':True,'actors':4,'base_adapter_likelihoods_differ':True}))
