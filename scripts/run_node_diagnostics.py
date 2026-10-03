"""Run gated node acceptance comparisons, separate from the frozen primary experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--start', choices=['N3', 'N4', 'N5'], default='N3')
parser.add_argument('--prefix', default='runs/harness-mvp/nodes-v1')
args = parser.parse_args()
nodes = [('N3', ['B0', 'G']), ('N4', ['G', 'GV']), ('N5', ['GC', 'Full'])]
for node, variants in nodes:
    if node < args.start:
        continue
    directory = f'{args.prefix}/{node}'
    subprocess.run([sys.executable, ROOT / 'scripts/run_benchmark_suite.py', '--phase', 'smoke',
        '--variants', *variants, '--run-dir', directory], cwd=ROOT, check=True)
    subprocess.run([sys.executable, ROOT / 'scripts/report_benchmark.py', directory], cwd=ROOT, check=True)
    (ROOT / args.prefix / f'{node}-accepted.json').write_text(json.dumps({
        'node': node, 'paired_evidence': directory, 'complete': True}, indent=2), encoding='utf-8')
