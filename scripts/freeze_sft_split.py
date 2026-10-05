"""Freeze training-task-isolated splits within the already exposed official dev set."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smarthome_agent_rl.sft_data import partition
from smarthome_agent_rl.benchmark import digest

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    args=p.parse_args()
    source=ROOT/'configs/benchmark-v2/dev.json'
    manifest=json.loads(source.read_text())
    queries={}
    for row in manifest['tasks']:
        path=ROOT/'deps/SimuHome/data/benchmark'/row['path']
        if digest(path)!=row['sha256']: raise ValueError('Task identity drift')
        queries[row['id']]=json.loads(path.read_text())['query']
    splits, audit=partition(manifest['tasks'],queries)
    frozen=set(json.loads((ROOT/'runs/holdout-inventory/n19-v1/inventory.json').read_text())['exposed_ids'])
    if set(queries)-frozen: raise ValueError('Split must use only historically exposed development tasks')
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=False)
    for name,rows in splits.items():
        (out/(name+'.json')).write_text(json.dumps({**{k:v for k,v in manifest.items() if k!='tasks'},
            'split':'dev','sft_split':name,'sampling_seed':20261005,'tasks':rows},indent=2)+'\n')
    diag=splits['train']+splits['calibration']
    smoke=[r for key in sorted({(r['query_type'],r['case']) for r in diag}) for r in
           [r for r in diag if (r['query_type'],r['case'])==key][:2]]
    for name,rows,phase in [('diagnostic',diag,'dev'),('smoke',smoke,'smoke')]:
        (out/(name+'.json')).write_text(json.dumps({**{k:v for k,v in manifest.items() if k!='tasks'},
            'split':phase,'sft_split':name,'tasks':rows},indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps({'source_sha256':digest(source),'counts':{k:len(v) for k,v in splits.items()},
        **audit,'reserved_unused_tasks':95,'manifests_sha256':{p.name:digest(p) for p in out.glob('*.json')}},indent=2)+'\n')
    print(json.dumps({'counts':{k:len(v) for k,v in splits.items()},**audit}))
