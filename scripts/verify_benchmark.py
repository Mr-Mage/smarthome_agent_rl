"""Independent immutable artifact verification, suitable for checking a copied archive."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(run):
    run = Path(run).resolve()
    manifest = json.loads((run / 'artifact_manifest.json').read_text(encoding='utf-8'))
    failures = []
    for name, expected in manifest.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or not path.is_file():
            failures.append({'path': name, 'problem': 'missing or outside artifact root'})
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append({'path': name, 'problem': 'SHA256 mismatch'})
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    completion = json.loads((run / 'completion.json').read_text(encoding='utf-8'))
    expected_summaries = []
    for item in protocol['schedule']:
        for variant in item['variants']:
            expected_summaries.append(f"worker{item['workflow']}/{item['task']['id']}/{variant}/lightning/summary.json")
    if set(expected_summaries) != {name for name in manifest if name.endswith('/summary.json')}:
        failures.append({'problem': 'missing/extra episode summary'})
    if completion['episodes'] != protocol['expected_episodes'] or not completion['complete']:
        failures.append({'problem': 'episode coverage mismatch'})
    result = {'verified': not failures, 'files': len(manifest),
              'expected_episodes': len(expected_summaries), 'failures': failures}
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise ValueError('Artifact verification failed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    verify(parser.parse_args().run_dir)
