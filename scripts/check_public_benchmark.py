"""Verify pinned assets and parity with public author code, without model imports."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.homebench import HomeBenchAdapter, aggregate, native_counts


def check(source, lock):
    start = time.monotonic()
    adapter = HomeBenchAdapter(source, lock)
    tasks = adapter.task_ids()
    representatives = {}
    for task_id in tasks:
        adapter.contract(task_id)
        representatives.setdefault(adapter.home_id(task_id), task_id)
    # Run the author's zero-shot Dataset class on one original row per home.
    # Only its formatter and Dataset class are compiled; no model or torch import.
    rows = [adapter._cases[t] for t in representatives.values()]
    files = {
        'test_data.jsonl': '\n'.join(json.dumps(r) for r in rows),
        'new_home_status_method.jsonl': (source/'dataset/home_status_method.jsonl').read_text(encoding='utf-8'),
        'system.txt': adapter.system,
    }
    def source_open(path, mode='r'):
        if mode != 'r' or Path(path).name not in files:
            raise ValueError('Unexpected pinned formatter file access')
        return io.StringIO(files[Path(path).name])
    tree = ast.parse((source/'code/model_test.py').read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
                n.name in ('chang_json2str', 'no_few_shot_home_assistant_dataset')]
    namespace = {'Dataset': object, 'json': json, 'open': source_open}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'pinned_zero_shot', 'exec'), namespace)
    author_dataset = namespace['no_few_shot_home_assistant_dataset'](None)
    for row, author in zip(rows, author_dataset.data):
        if adapter.public_input(row['id']) != [{'role': 'system', 'content': author['input']}]:
            raise AssertionError(f"Author prompt mismatch: {row['id']}")
    if len(author_dataset.data) != len(rows):
        raise AssertionError('Author dataset length differs')
    metric_tree = ast.parse((source/'code/eval.py').read_text(encoding='utf-8'))
    function = next(n for n in metric_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_accuracy')
    namespace = {'re': re, 'Counter': Counter}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'pinned_native_metric', 'exec'), namespace)
    # Include tuple comma splitting, duplicate errors, extra calls, multiple braces and empty output.
    predictions = ['{room.light.set_color((1,2,3)),error_input,error_input}',
                   '{error_input}{extra()}', 'no instructions']
    expected = ["'''error_input,error_input,room.light.set_color((1,2,3))'''",
                "'''error_input'''", "'''error_input'''" ]
    log = io.StringIO()
    with redirect_stdout(log):
        namespace['compute_accuracy'](predictions, expected)
    native = {}
    for line in log.getvalue().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            native[key] = float(value)
    ours = aggregate(native_counts(p, g) for p, g in zip(predictions, expected))
    for name, key in [('em', 'exact_match'), ('Precision', 'precision'), ('Recall', 'recall'), ('F1', 'f1')]:
        if native[name] != ours[key]:
            raise AssertionError(f'Native metric differs: {name}')
    return {'verified': True, 'commit': lock['commit'], 'tasks': len(tasks),
            'homes': len(representatives), 'contracts_constructed': len(adapter._contracts),
            'author_prompt_parity_homes': len(rows), 'native_metric_parity': native,
            'adaptation': lock['protocol_note'], 'seconds': time.monotonic()-start,
            'actor_calls': 0, 'judge_calls': 0, 'gpu_seconds': 0,
            'scope': 'source/adapter verification, no online benchmark result'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((ROOT/'configs/public-benchmarks.json').read_text(encoding='utf-8'))['HomeBench']
    result = check(args.source, lock)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
