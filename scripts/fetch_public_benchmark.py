"""Download fixed public assets with SHA checks; no environment installation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]


def fetch(lock,output,proxy=None):
    output.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}) if proxy else urllib.request.ProxyHandler({}))
    def asset(pair):
        name,row=pair
        path=output/name
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError('Asset path escapes output directory')
        if path.exists():
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError(f'Existing asset differs; preserve and inspect: {path}')
        else:
            url=f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}/{name}"
            with opener.open(url,timeout=60) as response:
                data=response.read()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError(f'Download differs from pinned source: {name}')
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.'+uuid4().hex+'.partial')
            temporary.write_bytes(data)
            temporary.replace(path)
        return name,hashlib.sha256(data).hexdigest()
    with ThreadPoolExecutor(max_workers=5) as pool:
        hashes=dict(pool.map(asset,lock['files'].items()))
    receipt={'repository':lock['repository'],'commit':lock['commit'],'sha256':hashes,'verified':True}
    (output/'download-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'verified':True,'files':len(hashes),'commit':lock['commit']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',choices=['HomeBench'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--proxy')
    args=parser.parse_args()
    lock=json.loads((ROOT/'configs/public-benchmarks.json').read_text(encoding='utf-8'))[args.benchmark]
    fetch(lock,Path(args.output),args.proxy)
