"""Detach an authorized job with closed SSH descriptors and durable output."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
if not args.command:
    parser.error('A command is required')
log = ROOT / args.log
log.parent.mkdir(parents=True, exist_ok=True)
with log.open('ab') as stream:
    process = subprocess.Popen(args.command, cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, 'PYTHONNOUSERSITE': '1'})
print(process.pid, flush=True)
