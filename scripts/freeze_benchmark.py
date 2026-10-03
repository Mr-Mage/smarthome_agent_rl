"""Freeze official metadata/hash-only manifests, without showing final queries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import freeze

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='configs/benchmark-mvp')
parser.add_argument('--exposures', help='JSON list of official IDs observed in previous runs')
args = parser.parse_args()
sim = ROOT / 'deps/SimuHome'
exposed = json.loads(Path(args.exposures).read_text()) if args.exposures else []
selection = freeze(sim / 'data/benchmark', ROOT / args.output,
    upstream_commit=subprocess.check_output(['git', '-C', str(sim), 'rev-parse', 'HEAD'], text=True).strip(),
    exposed=exposed)
print(json.dumps({'counts': selection['source_counts'], 'dev': len(selection['dev']),
                  'final': len(selection['final']), 'smoke': len(selection['smoke'])}))
