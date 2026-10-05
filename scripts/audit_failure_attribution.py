"""Offline evidence indicators and complete review cards, without actor-side hidden goals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.benchmark import digest
from smarthome_agent_rl.concurrency import episode_directory


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def public_errors(messages):
    start = max(i for i, m in enumerate(messages) if 'This is your actual task.' in m['content'])
    result = []
    for index, message in enumerate(messages[start + 1:], start + 1):
        if message['role'] != 'user' or not message['content'].startswith('observation:'):
            continue
        try:
            response = json.loads(message['content'].split(':', 1)[1])
        except (ValueError, TypeError):
            continue
        if isinstance(response, dict) and (response.get('error') is not None or response.get('status', {}).get('code', 200) >= 400):
            result.append({'message_index': index, 'response': response})
    return result


def required_actions(value):
    """Evaluator receipts used offline only; not guessed from whether an action sounds needed."""
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'required_actions' and isinstance(child, list):
                result.extend(row for row in child if isinstance(row, dict) and row.get('invoked') is False)
            else:
                result.extend(required_actions(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(required_actions(child))
    return result


def actions(calls):
    result = []
    for index, call in enumerate(calls, 1):
        raw = call['response']['choices'][0]['message'].get('content') or ''
        try:
            body = json.loads(raw)
            action = body['call']
        except (ValueError, KeyError, TypeError):
            action = {'parseable': False, 'raw': raw[:300]}
        result.append({'turn': index, 'action': action})
    return result


def audit_episode(path):
    summary, audit, calls = (read(path / name) for name in ('summary.json', 'harness_audit.json', 'model_calls.json'))
    official = read(path / 'official_result.json') if (path / 'official_result.json').exists() else None
    plan = actions(calls)
    errors = [p for p in audit['proposals'] if p.get('blocked') or p.get('simulator_error')]
    room_errors = [p for p in errors if p['tool'] == 'get_room_devices' and p.get('reached_executor')]
    missing = required_actions(official['evaluation_result']) if official else []
    invalid_format = sum('validation_error' in r for r in audit['structured'])
    counts = {
        'episodes': 1, 'successes': int(summary['success']), 'failed_episodes': int(not summary['success']),
        'proposed_errors': len(errors), 'executor_errors': sum(p.get('simulator_error', False) and p.get('reached_executor', False) for p in audit['proposals']),
        'room_query_errors': len(room_errors), 'episodes_with_room_error': int(bool(room_errors)),
        'missing_required_action_receipts': len(missing), 'episodes_with_missing_required_action': int(bool(missing)),
        'format_rejections': invalid_format, 'unfinished': int(summary['task_failure']),
        'actor_tokens': summary['actor_tokens'], 'actor_model_calls': len(calls),
    }
    counts['room_error_followed_by_get_rooms'] = sum(any(
        p['turn'] > error['turn'] and p['tool'] == 'get_rooms' and p.get('reached_executor') and not p.get('simulator_error')
        for p in audit['proposals']) for error in room_errors)
    counts['failed_episodes_with_any_proposal_error'] = int(not summary['success'] and bool(errors))
    counts['failed_episodes_with_missing_required_action'] = int(not summary['success'] and bool(missing))
    assert sum(c['response'].get('usage', {}).get('total_tokens', 0) for c in calls) == summary['actor_tokens']
    final = next((r['action']['arguments'].get('answer') for r in reversed(plan)
                  if r['action'].get('tool') == 'finish'), None)
    return {'task_id': summary['task_id'], 'variant': summary['variant'], 'actor_seed': summary['actor_seed'],
        'path': str(path.relative_to(ROOT)), 'query': official['query'] if official else None,
        'success': summary['success'], 'task_failure_kind': summary['task_failure_kind'],
        'evaluation_result': official['evaluation_result'] if official else None,
        'actions': plan, 'final_answer': final, 'errors': errors, 'missing_required_actions': missing,
        'counts': counts, 'source_sha256': {name: digest(path / name) for name in
            ('summary.json', 'harness_audit.json', 'model_calls.json')},
        'official_result_sha256': digest(path / 'official_result.json') if official else None}


def audit_data(config):
    data = ROOT / config['data']
    admission = read(ROOT / 'runs/sft-pilot/n23-v1/audit.json')
    assert digest(data) == admission['data_sha256']
    train_ids = {t['id'] for t in read(ROOT / 'configs/sft-pilot/train.json')['tasks']}
    rows = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()]
    sources, examples, counts = {}, [], Counter()
    error_tasks = set()
    for index, row in enumerate(rows):
        assert row['task_id'] in train_ids
        source = row['source']
        if source not in sources:
            path = ROOT / source
            assert digest(path / 'model_calls.json') == row['model_calls_sha256']
            assert digest(path / 'summary.json') == row['summary_sha256']
            sources[source] = read(path / 'model_calls.json')
        assert sources[source][row['turn'] - 1]['request']['messages'] == row['messages']
        target = json.loads(row['target'])['call']
        counts['samples'] += 1
        counts['tool_' + target['tool']] += 1
        errors = public_errors(row['messages'])
        if errors:
            counts['targets_with_prior_public_errors'] += 1
            error_tasks.add(row['task_id'])
            last_error = errors[-1]['message_index']
            if last_error == len(row['messages']) - 1:
                counts['targets_immediately_after_error'] += 1
                counts['immediate_tool_' + target['tool']] += 1
                examples.append({'sample_index': index, 'task_id': row['task_id'], 'source': source,
                    'turn': row['turn'], 'target': target, 'last_error': errors[-1]['response']})
        counts['finish_targets'] += target['tool'] == 'finish'
    assert len(rows) == 361 and len(sources) == 65
    return {'passed_provenance': True, 'data_sha256': digest(data), 'counts': dict(counts),
        'independent_tasks': len({r['task_id'] for r in rows}), 'source_episodes': len(sources),
        'tasks_with_error_context': sorted(error_tasks), 'immediate_error_examples': examples,
        'limitations': 'Context after an error is not proof that the target semantically recovers or covers all user goals; prefix masked and short-thought transformation unchanged'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    started = time.monotonic()
    config = read(ROOT / args.config)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    aggregates, reviews, receipts = {}, {}, {}
    eval_ids = {t['id'] for t in read(ROOT / 'configs/sft-pilot/eval.json')['tasks']}
    for tag, name, arms in [('primary', config['primary_stage'], config['primary_arms']),
                            ('supplemental', config['supplemental_stage'], config['supplemental_arms'])]:
        stage = ROOT / name
        from scripts.verify_benchmark import verify
        verified = verify(stage)
        protocol, official = read(stage / 'protocol.json'), read(stage / 'report.json')
        assert {i['task']['id'] for i in protocol['schedule']} <= eval_ids
        receipts[tag] = {'stage': name, 'report_sha256': digest(stage / 'report.json'),
            'protocol_sha256': digest(stage / 'protocol.json'), 'verified': verified}
        for variant in arms:
            rows = [audit_episode(episode_directory(stage, item, variant)) for item in protocol['schedule']]
            counts = Counter()
            for row in rows:
                counts.update(row['counts'])
            arm = official['arms'][variant]
            assert counts['successes'] == arm['successes'] and counts['actor_tokens'] == arm['all_totals']['actor_tokens']
            assert counts['executor_errors'] == arm['all_totals']['invalid_reached_executor']
            aggregates[tag + '/' + variant] = dict(counts)
            reviews[tag + '/' + variant] = rows
    dataset = audit_data(config)
    manual = [row for row in reviews['primary/G'] if row['actor_seed'] == 42 and not row['success']]
    # Recover query only for frozen exposed cases whose episode exhausted its budget.
    for row in manual:
        if row['query'] is None:
            task_path = ROOT / 'deps/SimuHome/data/benchmark' / (row['task_id'] + '.json')
            task = next(t for t in read(ROOT / 'configs/sft-pilot/eval.json')['tasks'] if t['id'] == row['task_id'])
            assert digest(task_path) == task['sha256']
            row['query'] = read(task_path)['query']
    assert len(manual) == 31
    for name, value in [('automatic-indicators.json', aggregates), ('episode-review.json', reviews),
                         ('manual-review-cards.json', manual), ('dataset-context-audit.json', dataset)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'audit-receipt.json').write_text(json.dumps({'passed': True,
        'config_sha256': digest(ROOT / args.config), 'source_sha256': digest(Path(__file__)),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_stages': receipts, 'manual_review_count': len(manual), 'seconds': time.monotonic() - started,
        'new_actor_judge_calls': 0, 'allocated_h100_gpu_seconds': 0,
        'scope': 'Existing exposed cases only; no unused95 opened; historical generation costs not counted again; overlapping sources separate; automated indicators are not causal labels'}, indent=2) + '\n')
    print(json.dumps({'counts': aggregates, 'data': dataset['counts'], 'manual_failures': len(manual)}, indent=2))


if __name__ == '__main__':
    main()
