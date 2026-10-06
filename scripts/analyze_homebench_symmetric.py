"""Compare contract prompts after applying identical frozen output transforms."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.homebench import HomeBenchAdapter, aggregate
from smarthome_agent_rl.benchmarks.instructions import serialize_instructions
from smarthome_agent_rl.benchmarks.wire import compile_wire


def compare(adapter, rows):
    index = {}
    for row in rows:
        if row['arm'] in ('B0', 'B1'):
            key = (row['task_id'], row['arm'])
            if key in index:
                raise ValueError('Duplicate real source episode')
            index[key] = row
    ids = sorted(t for t, a in index if a == 'B0')
    if not ids or set(index) != {(t, a) for t in ids for a in ('B0', 'B1')}:
        raise ValueError('Incomplete paired source')
    arms = {f'{a}_{v}': [] for a in ('B0', 'B1') for v in ('raw', 'F', 'FG', 'W', 'WG')}
    counts = {k: {'format_uncovered': 0, 'guard_uncovered': 0, 'rejections': 0} for k in arms}
    changes = {v: {'wins': 0, 'losses': 0, 'win_cases': [], 'loss_cases': []}
               for v in ('raw', 'F', 'FG', 'W', 'WG')}
    for task_id in ids:
        projected = {}
        _, shapes, spec = adapter.contract(task_id)
        for a in ('B0', 'B1'):
            row = index[task_id, a]
            raw = row['prediction']
            f = serialize_instructions(raw)
            w = compile_wire(raw, spec['functions'], shapes)
            fg = adapter.guard(task_id, f['prediction'])
            wg = adapter.guard(task_id, w['prediction'])
            for v, result, fmt, guard in (('raw', {'prediction': raw}, {}, {}),
                    ('F', f, f, {}), ('FG', fg, f, fg), ('W', w, w, {}), ('WG', wg, w, wg)):
                score = adapter.score(task_id, result['prediction'])
                if row.get('error'):
                    score['exact_match'] = False
                arms[a + '_' + v].append(score)
                count = counts[a + '_' + v]
                count['format_uncovered'] += bool(fmt.get('uncovered'))
                count['guard_uncovered'] += bool(guard.get('uncovered'))
                count['rejections'] += len(guard.get('rejections', []))
                projected[a, v] = (result['prediction'], score['exact_match'])
        for v, change in changes.items():
            before, after = projected['B0', v], projected['B1', v]
            if before[1] == after[1]:
                continue
            direction = 'win' if after[1] else 'loss'
            change[direction + 's'] += 1
            if len(change[direction + '_cases']) < 3:
                change[direction + '_cases'].append({'task_id': task_id,
                    'B0_prediction': before[0], 'B1_prediction': after[0]})
    return {'episodes': len(ids), 'arms': {a: aggregate(scores) for a, scores in arms.items()},
            'coverage': counts, 'contract_paired_changes': changes}


def analyze(directory, source):
    began = time.monotonic()
    config_path = ROOT / 'configs/homebench-symmetric-diagnostic.json'
    config = json.loads(config_path.read_text())
    lock = json.loads((ROOT / 'configs/public-benchmarks.json').read_text())['HomeBench']
    if config['source_commit'] != lock['commit']:
        raise ValueError('Pinned source differs')
    report = json.loads((directory / 'report.json').read_text())
    if report['status'] != 'complete' or 'service_cost' not in report:
        raise ValueError('Wait for completed inference and service cleanup')
    rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    result = compare(HomeBenchAdapter(source, lock), rows)
    if result['episodes'] != report['arms']['B0']['episodes']:
        raise ValueError('Source episode count differs')
    return {**result, 'config': config, 'seconds': time.monotonic() - began,
            'input_sha256': {n: hashlib.sha256((directory / n).read_bytes()).hexdigest()
                             for n in ('report.json', 'episodes.jsonl')},
            'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'actor_requests': 0, 'judge_requests': 0, 'tokens': 0, 'gpu_seconds': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('episodes', 'contract_paired_changes', 'seconds')}))
