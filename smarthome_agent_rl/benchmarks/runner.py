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
from .instructions import serialize_instructions
from .wire import compile_wire


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


def messages_for(adapter, task_id, arm, transport='native', location='before'):
    messages = adapter.public_input(task_id)
    if arm == 'B0':
        pass
    elif arm != 'B1':
        raise ValueError('B2 must reuse B1 raw output')
    else:
        _, _, spec = adapter.contract(task_id)
        # These are capabilities only, never target labels or task-category hints.
        context = '\n<public_device_contract>\n'+json.dumps(spec, ensure_ascii=False, separators=(',', ':'))+'\n</public_device_contract>\n'
        marker = '-------------------------------\nHere are the user instructions you need to reply to.\n'
        enriched = []
        for message in messages:
            if location == 'after':
                enriched.append({'role': message['role'], 'content': message['content']+context})
                continue
            prefix, separator, suffix = message['content'].partition(marker)
            if not separator:
                raise ValueError('Pinned HomeBench task boundary missing')
            enriched.append({'role': message['role'], 'content': prefix+context+separator+suffix})
        messages = enriched
    if transport == 'append_empty_user':
        messages = messages+[{'role': 'user', 'content': ''}]
    elif transport != 'native':
        raise ValueError('Unknown chat template compatibility policy')
    return messages


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


def summarize(records, total_tasks, seconds, arm_names=('B0', 'B1', 'B2')):
    arms = {}
    for name in arm_names:
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
                      'format_uncovered_episodes': sum(bool(r.get('format', {}).get('uncovered')) for r in rows),
                      'cost_attribution': 'shared B1 request; no additional actor request' if name not in ('B0', 'B1') else 'own requests'}
    actual = [r for r in records if r['arm'] in ('B0', 'B1')]
    complete_usage = [r['usage'] for r in actual if valid_usage(r['usage'])]
    changes = {}
    indexed = {(r['task_id'], r['arm']): r for r in records}
    for left, right in [('B0', 'B1'), ('B1', 'B2'), ('B1', 'F'), ('F', 'FG'), ('F', 'W'), ('W', 'WG')]:
        if left not in arm_names or right not in arm_names:
            continue
        pairs = [(indexed[(t, left)], indexed[(t, right)]) for t, a in indexed
                 if a == left and (t, right) in indexed]
        wins = [b['task_id'] for a, b in pairs if not a['score']['exact_match'] and b['score']['exact_match']]
        losses = [b['task_id'] for a, b in pairs if a['score']['exact_match'] and not b['score']['exact_match']]
        changes[left+'_'+right] = {'paired': len(pairs), 'wins': len(wins), 'losses': len(losses),
                                  'win_case_ids': wins[:3], 'loss_case_ids': losses[:3]}
    return {'status': 'complete' if len(records) == len(arm_names)*total_tasks else 'running',
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
    extra_arms = config.get('extra_arms', [])
    if len(set(extra_arms)) != len(extra_arms) or any(a not in ('F', 'FG', 'W', 'WG') for a in extra_arms):
        raise ValueError('Unknown/duplicate additional arm')
    arm_names = ('B0', 'B1', 'B2')+tuple(extra_arms)
    if config.get('engineering_gate_arm', 'B2') not in arm_names:
        raise ValueError('Engineering gate arm absent from frozen experiment')
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
    last_checkpoint = start
    def episode(task_id, arm, actor, submitted):
        queued = time.monotonic()
        location = 'after' if config.get('schema') == 'homebench-ablation-v1' and 'contract_location' not in config else 'before'
        messages = messages_for(adapter, task_id, arm, config.get('chat_transport', 'native'), location)
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
            if extra_arms:
                format_started = time.monotonic()
                formatted = serialize_instructions(call['text'])
                for arm_name in extra_arms:
                    format_result = formatted
                    if arm_name in ('W', 'WG'):
                        _, shapes, spec = adapter.contract(task_id)
                        format_result = compile_wire(call['text'], spec['functions'], shapes)
                    result = {'prediction': format_result['prediction'], 'rejections': [], 'uncovered': []}
                    if arm_name in ('FG', 'WG'):
                        result = adapter.guard(task_id, format_result['prediction'])
                    extra = {**row, 'arm': arm_name, 'prediction': result['prediction'], 'format': format_result,
                             'format_seconds': time.monotonic()-format_started, 'shared_request': 'B1',
                             'score': adapter.score(task_id, result['prediction'])}
                    if arm_name in ('FG', 'WG'):
                        extra['guard'] = result
                    if call['error']:
                        extra['score']['exact_match'] = False
                    batch.append(extra)
        return batch
    # Separate pools enforce the per-actor budget. Home affinity permits prefix reuse.
    pools = [ThreadPoolExecutor(max_workers=config['slots_per_actor']) for _ in actors]
    try:
        futures = []
        # Keep each home's public context contiguous in the actor queue. This
        # changes neither the frozen selection nor either arm's input.
        for task_id in sorted(ids, key=lambda t: (digest(adapter.home_id(t)), digest(t))):
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
                # A full benchmark has tens of thousands of rows. Re-aggregating
                # after every completion would introduce quadratic CPU work.
                now = time.monotonic()
                if len(records) == len(batch) or now-last_checkpoint >= config.get('checkpoint_seconds', 10):
                    save(output/'report.json', summarize(records, len(ids), now-start, arm_names))
                    last_checkpoint = now
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    report = summarize(records, len(ids), time.monotonic()-start, arm_names)
    failures = report['cost']['failed_actor_requests']
    missing = report['cost']['missing_usage_requests']
    gate_arm = config.get('engineering_gate_arm', 'B2')
    uncovered = sum(bool(r.get('guard', {}).get('uncovered') or r.get('format', {}).get('uncovered'))
                    for r in records if r['arm'] == gate_arm) / len(ids)
    report['engineering_gate'] = {'passed': failures == 0 and missing == 0 and uncovered <= config['max_guard_uncovered_rate'],
                                  'failed_requests': failures, 'missing_usage': missing,
                                  'guard_uncovered_rate': uncovered,
                                  'threshold': config['max_guard_uncovered_rate'],
                                  'arm': gate_arm,
                                  'benefit_proven': False}
    save(output/'report.json', report)
    return report
