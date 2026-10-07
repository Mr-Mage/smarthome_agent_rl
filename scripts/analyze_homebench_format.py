"""Read-only output-format diagnosis on completed, already exposed episodes."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmarks.homebench import HomeBenchAdapter, aggregate, qualified_name


def rewrap_candidate(text):
    """Counterfactual only: bracket bare literal instruction lists; never execute."""
    value = text.strip()
    if not value or '{' in value or '}' in value:
        return None
    try:
        nodes = ast.parse('['+value+']', mode='eval').body.elts
    except (SyntaxError, AttributeError):
        return None
    if not nodes:
        return None
    for node in nodes:
        if isinstance(node, ast.Name) and node.id == 'error_input':
            continue
        if not isinstance(node, ast.Call) or not qualified_name(node.func):
            return None
        try:
            for arg in node.args:
                ast.literal_eval(arg)
            for keyword in node.keywords:
                if keyword.arg is None:
                    return None
                ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError):
            return None
    return '{'+value+'}'


def analyze(directory, source):
    lock = json.loads((ROOT/'configs/public-benchmarks.json').read_text(encoding='utf-8'))['HomeBench']
    adapter = HomeBenchAdapter(source, lock)
    rows = [json.loads(line) for line in (directory/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    result = {'input_report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
              'scope': 'read-only counterfactual diagnosis on exposed calibration; no new online experiment or holdout benefit',
              'actor_calls': 0, 'judge_calls': 0, 'gpu_seconds': 0, 'arms': {}}
    for arm in ('B0', 'B1'):
        selected = sorted((r for r in rows if r['arm'] == arm), key=lambda r: r['task_id'])
        formats = Counter()
        syntax = Counter()
        counters = {'raw': [], 'rewrap_only': [], 'rewrap_then_existing_guard': []}
        changes = []
        for row in selected:
            prediction = row['prediction']
            candidate = rewrap_candidate(prediction)
            if prediction.strip() == 'error_input':
                kind = 'bare_error_input'
            elif candidate is not None:
                kind = 'bare_literal_calls'
            elif '{' in prediction:
                kind = 'has_braces'
            elif not prediction.strip():
                kind = 'empty'
            else:
                kind = 'other_uncovered_text'
            formats[kind] += 1
            if '{' in prediction:
                bodies = re.findall(r'\{(.*?)\}', prediction, re.S)
                if any(';' in body for body in bodies):
                    syntax['semicolon_in_braces'] += 1
                if row['arm'] == 'B1' and adapter.guard(row['task_id'], prediction)['uncovered']:
                    syntax['braced_but_unparsed'] += 1
            projected = prediction if candidate is None else candidate
            guarded = adapter.guard(row['task_id'], projected)
            options = {'raw': prediction, 'rewrap_only': projected, 'rewrap_then_existing_guard': guarded['prediction']}
            scores = {key: adapter.score(row['task_id'], value) for key, value in options.items()}
            for key, score in scores.items():
                if row['error']:
                    score['exact_match'] = False
                counters[key].append(score)
            if scores['raw']['exact_match'] != scores['rewrap_only']['exact_match']:
                changes.append({'task_id': row['task_id'], 'category': row['category'],
                    'public_input': adapter._cases[row['task_id']]['input'], 'raw_prediction': prediction,
                    'format_projection': projected, 'expected': adapter._cases[row['task_id']]['output'],
                    'raw_exact_match': scores['raw']['exact_match'], 'projection_exact_match': scores['rewrap_only']['exact_match']})
        result['arms'][arm] = {'episodes': len(selected), 'formats': dict(formats), 'syntax': dict(syntax),
                             'counterfactual_metrics': {k: aggregate(v) for k, v in counters.items()},
                             'format_changed_match_cases': changes[:3], 'format_changed_match_count': len(changes)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({a: {'formats': r['formats'], 'counterfactual_em': {k: v['exact_match'] for k, v in r['counterfactual_metrics'].items()}}
                      for a, r in result['arms'].items()}))
