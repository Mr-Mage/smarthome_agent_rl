"""Sequential protocol gates; episodes execute in two parallel isolated actor workflows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='runs/harness-mvp/primary-v1')
args = parser.parse_args()
def command(script, *arguments):
    subprocess.run([sys.executable, ROOT / 'scripts' / script, *arguments], cwd=ROOT, check=True)
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
    raise RuntimeError('Primary experiment requires a clean committed project')
for node in ('N3', 'N4', 'N5'):
    accepted = ROOT / 'runs/harness-mvp/nodes-v1' / f'{node}-accepted.json'
    if not accepted.exists() or not json.loads(accepted.read_text())['complete']:
        raise RuntimeError(f'{node} diagnostics are not accepted')
command('run_benchmark_suite.py', '--phase', 'dev', '--run-dir', args.prefix + '/dev')
command('report_benchmark.py', args.prefix + '/dev')
command('freeze_experiment.py', '--dev-run', args.prefix + '/dev')
command('run_benchmark_suite.py', '--phase', 'final', '--run-dir', args.prefix + '/final',
        '--freeze', 'work/harness-mvp/final-freeze.json')
command('report_benchmark.py', args.prefix + '/final')
(ROOT / args.prefix / 'accepted.json').write_text(json.dumps({'complete': True,
    'dev_episodes': 600, 'final_episodes': 768, 'reports': ['dev/report.json', 'final/report.json']}, indent=2))
