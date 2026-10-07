"""Verify permanent baseline Git objects without checking out or changing upstream."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    commit = manifest['source_commit']
    for item in manifest['files']:
        content = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                          'show', f"{commit}:{item['path']}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError(f"Frozen baseline SHA mismatch: {item['path']}")
    return {'verified': True, 'source_commit': commit, 'files': len(manifest['files'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(ROOT / 'configs/1007-baseline.json'))
    print(json.dumps(verify(parser.parse_args().manifest)))
