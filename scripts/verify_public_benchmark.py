"""Audit paired inputs, raw responses, labels, metrics and evidence hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.homebench import HomeBenchAdapter
from smarthome_agent_rl.benchmarks.runner import digest, messages_for, select_tasks, summarize


def verify(directory, source, lock=None):
    freeze = json.loads((directory/'freeze.json').read_text(encoding='utf-8'))
    config = freeze['config']
    if lock is None:
        lock = json.loads((ROOT/'configs/public-benchmarks.json').read_text(encoding='utf-8'))['HomeBench']
    adapter = HomeBenchAdapter(source, lock)
    ids = freeze['task_ids']
    if digest(config) != freeze['config_sha256'] or digest(ids) != freeze['task_ids_sha256']:
        raise AssertionError('Frozen identity differs')
    if ids != select_tasks(adapter, freeze['selection']):
        raise AssertionError('Selection differs')
    rows = [json.loads(line) for line in (directory/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    index = {(r['task_id'], r['arm']): r for r in rows}
    if len(index) != len(rows) or set(index) != {(t, a) for t in ids for a in ('B0', 'B1', 'B2')}:
        raise AssertionError('Missing/duplicate/unplanned episodes')
    references = set()
    for (task_id, arm), row in index.items():
        if row['category'] != adapter.category(task_id) or row['home_id'] != adapter.home_id(task_id):
            raise AssertionError('Evaluator metadata differs')
        receipt = directory/row['request_receipt']
        if not receipt.resolve().is_relative_to(directory.resolve()):
            raise AssertionError('Receipt escaped evidence directory')
        call = json.loads(receipt.read_text(encoding='utf-8'))
        references.add(receipt.resolve())
        parent = 'B1' if arm == 'B2' else arm
        messages = messages_for(adapter, task_id, parent, config.get('chat_transport', 'native'))
        generation = config['generation']
        body = {'model': config['model'], 'messages': messages, 'seed': config['model_seed'],
                **{k: v for k, v in generation.items() if k != 'extra_body'}, **generation.get('extra_body', {})}
        if call['body'] != body or row['input_sha256'] != digest(messages):
            raise AssertionError('Actor input/config drift or evaluator leakage')
        if row['error'] != call['error'] or row['usage'] != call['usage']:
            raise AssertionError('Cost/failure evidence differs')
        if arm == 'B2':
            guarded = adapter.guard(task_id, call['text'])
            if row['guard'] != guarded or row['prediction'] != guarded['prediction']:
                raise AssertionError('Guard replay differs')
            if row['request_receipt'] != index[(task_id, 'B1')]['request_receipt']:
                raise AssertionError('B2 did not share B1 request')
        elif row['prediction'] != call['text']:
            raise AssertionError('Raw prediction modified')
        score = adapter.score(task_id, row['prediction'])
        if row['error']:
            score['exact_match'] = False
        if row['score'] != score:
            raise AssertionError('Evaluator score differs')
    if len(references) != 2*len(ids):
        raise AssertionError('Actual request accounting differs')
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    reproduced = summarize(rows, len(ids), report['seconds'])
    for field in ('status', 'arms', 'paired_changes', 'cost'):
        if report[field] != reproduced[field]:
            raise AssertionError(f'Report differs: {field}')
    files = {}
    for root in (directory, directory.parent/(directory.name+'-services')):
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            # Kernel caches are reproducible performance artifacts; omit from evidence archive.
            if path.is_file() and 'cache' not in path.relative_to(root).parts and path.name != 'evidence-receipt.json':
                files[str(path.relative_to(directory.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'verified': True, 'episodes': len(rows), 'actual_requests': len(references),
            'paired_raw_input_checks': len(ids), 'source_commit': lock['commit'],
            'files': files, 'scope': 'integrity check; no causal or heldout benefit claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir, args.source)
    path = args.run_dir/'evidence-receipt.json'
    if path.exists():
        raise FileExistsError('Preserve previous receipt')
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v if k != 'files' else len(v) for k, v in result.items()}))
