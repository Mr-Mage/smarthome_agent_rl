"""Single-turn paired instruction experiments; labels remain in the evaluator."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

from .homebench import aggregate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def select_tasks(adapter, selection):
    ids = adapter.task_ids()
    if selection['kind'] == 'full':
        return ids
    if selection['kind'] != 'category_hash':
        raise ValueError('Unknown frozen selection')
    groups = {}
    for task_id in ids:
        groups.setdefault(adapter.category(task_id), []).append(task_id)
    result = []
    for category in sorted(groups):
        ordered = sorted(groups[category], key=lambda t: digest([selection['salt'], t]))
        result.extend(ordered[:selection['per_category']])
    return result


def messages_for(adapter, task_id, arm):
    messages = adapter.public_input(task_id)
    if arm == 'B0':
        return messages
    if arm != 'B1':
        raise ValueError('B2 must reuse B1 raw output')
    _, _, spec = adapter.contract(task_id)
    # These are capabilities only, never target labels or task-category hints.
    context = '\n<public_device_contract>\n'+json.dumps(spec, ensure_ascii=False, separators=(',', ':'))+'\n</public_device_contract>\n'
    return [{'role': m['role'], 'content': m['content']+context} for m in messages]


def completion(endpoint, body, timeout):
    request = urllib.request.Request(endpoint+'/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    row = {'endpoint': endpoint, 'body': body, 'response': None, 'error': None,
           'text': '', 'usage': None, 'finish_reason': None}
    try:
        with opener.open(request, timeout=timeout) as response:
            row['status'] = response.status
            raw = response.read().decode('utf-8')
        row['raw_response'] = raw
        data = json.loads(raw)
        row['response'] = data
        choice = data['choices'][0]
        text = choice['message']['content']
        if not isinstance(text, str):
            raise ValueError('Missing text content')
        if data['model'] != body['model']:
            raise ValueError('Response model differs from frozen actor')
        row.update(text=text, usage=data.get('usage'), finish_reason=choice.get('finish_reason'))
    except urllib.error.HTTPError as exc:
        row['status'] = exc.code
        row['raw_response'] = exc.read().decode('utf-8', errors='replace')
        row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    except Exception as exc:
        row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    row['request_seconds'] = time.monotonic()-start
    return row


def summarize(records, total_tasks, seconds):
    arms = {}
    for name in ('B0', 'B1', 'B2'):
        rows = [r for r in records if r['arm'] == name]
        scores = [r['score'] for r in rows]
        native = aggregate(scores)
        by_category = {}
        for row in rows:
            by_category.setdefault(row['category'], []).append(row['score'])
        arms[name] = {**native, 'completed': len(rows), 'planned': total_tasks,
                      'categories': {k: aggregate(v) for k, v in sorted(by_category.items())},
                      'failed_requests': sum(r['error'] is not None for r in rows),
                      'actor_tokens_attributed': sum(r['usage']['total_tokens'] for r in rows if valid_usage(r['usage'])),
                      'request_latency_median': statistics.median(r['request_seconds'] for r in rows) if rows else None,
                      'guard_rejected_instructions': sum(len(r.get('guard', {}).get('rejections', [])) for r in rows),
                      'guard_uncovered_episodes': sum(bool(r.get('guard', {}).get('uncovered')) for r in rows),
                      'cost_attribution': 'shared B1 request; no additional actor request' if name == 'B2' else 'own requests'}
    actual = [r for r in records if r['arm'] != 'B2']
    complete_usage = [r['usage'] for r in actual if valid_usage(r['usage'])]
    changes = {}
    indexed = {(r['task_id'], r['arm']): r for r in records}
    for left, right in [('B0', 'B1'), ('B1', 'B2')]:
        pairs = [(indexed[(t, left)], indexed[(t, right)]) for t, a in indexed
                 if a == left and (t, right) in indexed]
        wins = [b['task_id'] for a, b in pairs if not a['score']['exact_match'] and b['score']['exact_match']]
        losses = [b['task_id'] for a, b in pairs if a['score']['exact_match'] and not b['score']['exact_match']]
        changes[left+'_'+right] = {'paired': len(pairs), 'wins': len(wins), 'losses': len(losses),
                                  'win_case_ids': wins[:3], 'loss_case_ids': losses[:3]}
    return {'status': 'complete' if len(records) == 3*total_tasks else 'running',
            'arms': arms, 'paired_changes': changes, 'seconds': seconds,
            'cost': {'actual_actor_requests': len(actual), 'failed_actor_requests': sum(r['error'] is not None for r in actual),
                     'prompt_tokens': sum(u['prompt_tokens'] for u in complete_usage),
                     'completion_tokens': sum(u['completion_tokens'] for u in complete_usage),
                     'total_tokens': sum(u['total_tokens'] for u in complete_usage),
                     'missing_usage_requests': len(actual)-len(complete_usage),
                     'request_latency_median': statistics.median(r['request_seconds'] for r in actual) if actual else None,
                     'judge_requests': 0, 'tool_calls': 0},
            'scope': 'public static instruction generation; no world-state execution or independent holdout benefit'}


def valid_usage(usage):
    return isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')) and \
        usage['total_tokens'] == usage['prompt_tokens']+usage['completion_tokens']


def run(adapter, config, output, selection, request_fn=completion):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    ids = select_tasks(adapter, selection)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate selection')
    actors = config['actors']
    if len({a['endpoint'] for a in actors}) != len(actors) or not actors:
        raise ValueError('Actor endpoints must be nonempty and distinct')
    for task_id in ids:
        adapter.contract(task_id)
    freeze = {'config': config, 'selection': selection, 'task_ids': ids,
              'config_sha256': digest(config), 'task_ids_sha256': digest(ids)}
    save(output/'freeze.json', freeze)
    records = []
    def episode(task_id, arm, actor, submitted):
        queued = time.monotonic()
        messages = messages_for(adapter, task_id, arm)
        generation = config['generation']
        body = {'model': config['model'], 'messages': messages, 'seed': config['model_seed'],
                **{k: v for k, v in generation.items() if k != 'extra_body'}, **generation.get('extra_body', {})}
        call = request_fn(actor['endpoint'], body, config['request_timeout'])
        # Request evidence is saved before any evaluator accesses the answer.
        receipt = output/'requests'/f'{digest(task_id)}-{arm}.json'
        receipt.parent.mkdir(exist_ok=True)
        save(receipt, call)
        row = {'task_id': task_id, 'arm': arm, 'actor_id': actor['id'], 'home_id': adapter.home_id(task_id),
               'category': adapter.category(task_id), 'prediction': call['text'], 'error': call['error'],
               'usage': call['usage'], 'finish_reason': call['finish_reason'],
               'request_seconds': call['request_seconds'], 'queue_seconds': queued-submitted,
               'episode_seconds': time.monotonic()-submitted,
               'request_receipt': str(receipt.relative_to(output)), 'input_sha256': digest(messages),
               'score': adapter.score(task_id, call['text'])}
        if call['error']:
            row['score']['exact_match'] = False
        batch = [row]
        if arm == 'B1':
            guard_started = time.monotonic()
            guarded = adapter.guard(task_id, call['text'])
            b2 = {**row, 'arm': 'B2', 'prediction': guarded['prediction'], 'guard': guarded,
                  'guard_seconds': time.monotonic()-guard_started,
                  'shared_request': 'B1', 'score': adapter.score(task_id, guarded['prediction'])}
            if call['error']:
                b2['score']['exact_match'] = False
            batch.append(b2)
        return batch
    # Separate pools enforce the per-actor budget. Home affinity permits prefix reuse.
    pools = [ThreadPoolExecutor(max_workers=config['slots_per_actor']) for _ in actors]
    try:
        futures = []
        for task_id in ids:
            actor_index = int(digest(adapter.home_id(task_id)), 16) % len(actors)
            for arm in ('B0', 'B1'):
                futures.append(pools[actor_index].submit(episode, task_id, arm, actors[actor_index], time.monotonic()))
        with (output/'episodes.jsonl').open('w', encoding='utf-8') as log:
            for future in as_completed(futures):
                batch = future.result()
                for row in batch:
                    log.write(json.dumps(row, ensure_ascii=False)+'\n')
                log.flush()
                records.extend(batch)
                save(output/'report.json', summarize(records, len(ids), time.monotonic()-start))
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    report = summarize(records, len(ids), time.monotonic()-start)
    failures = report['cost']['failed_actor_requests']
    missing = report['cost']['missing_usage_requests']
    uncovered = report['arms']['B2']['guard_uncovered_episodes'] / len(ids)
    report['engineering_gate'] = {'passed': failures == 0 and missing == 0 and uncovered <= config['max_guard_uncovered_rate'],
                                  'failed_requests': failures, 'missing_usage': missing,
                                  'guard_uncovered_rate': uncovered,
                                  'threshold': config['max_guard_uncovered_rate'],
                                  'benefit_proven': False}
    save(output/'report.json', report)
    return report
