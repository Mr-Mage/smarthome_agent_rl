"""Read-only developer review of an existing episode, with public evidence links."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from smarthome_agent_rl.action_state import TRANSITIONS, action_signature


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_episode(folder, manifest=None):
    folder = Path(folder).resolve()
    names = ('harness_audit.json', 'contract.json', 'summary.json', 'model_calls.json')
    paths = {name: folder/name for name in names}
    checksums = {name: digest(path) for name, path in paths.items()}
    integrity = 'self_hash_only'
    if manifest:
        manifest = Path(manifest).resolve()
        data = json.loads(manifest.read_text(encoding='utf-8'))
        entries = data.get('files_sha256', data)
        # Receipt archives preserve paths from the project root, under extracted/.
        for name, path in paths.items():
            matching = [key for key, value in entries.items() if value == checksums[name]
                        and key.replace('\\', '/').endswith('/'+name)
                        and path.as_posix().endswith('/'+key.replace('\\', '/'))]
            if len(matching) != 1:
                raise ValueError(f'Missing, ambiguous or changed receipt: {name}')
        integrity = 'matched_frozen_receipts'
    audit, contract, summary, calls = [json.loads(paths[n].read_text(encoding='utf-8')) for n in names]
    config = contract['config']
    if (contract['task_identity']['id'], config['variant'], config['model_seed']) != (
            summary['task_id'], summary['variant'], summary['actor_seed']):
        raise ValueError('Task, variant or seed differs from original contract')
    if any(c['request']['model'] != config['served_model'] or
           c['request'].get('seed') != config['model_seed'] for c in calls):
        raise ValueError('Actor model or seed differs from original contract')
    actions = audit['action_lifecycle']['actions']
    observations = audit['actual_observations']
    ids = {a['action_id'] for a in actions}
    if len(ids) != len(actions):
        raise ValueError('Duplicate action identity')
    lookup = {a['action_id']: a for a in actions}
    rows = []
    for action in actions:
        signature = action_signature(action['tool'], action['arguments'], extra_query=action['extra_query'])
        if hashlib.sha256(signature.encode()).hexdigest() != action['request_sha256']:
            raise ValueError('Action request identity mismatch')
        if action['parent_action_id'] and action['parent_action_id'] not in ids:
            raise ValueError('Missing parent proposal')
        state = None
        for transition in action['transitions']:
            if transition['from'] != state or (state is None and transition['to'] != 'proposed') or (
                state is not None and transition['to'] not in TRANSITIONS[state]):
                raise ValueError('Disconnected or illegal action transition')
            state = transition['to']
        if action['state'] != state:
            raise ValueError('Snapshot state differs from transition history')
        sources = []
        for evidence in action['evidence']:
            ordinal = evidence.get('observation_index')
            if ordinal is None:
                continue
            if type(ordinal) is not int or not 1 <= ordinal <= len(observations):
                raise ValueError('Invalid public observation reference')
            observation = observations[ordinal-1]
            owner = lookup[evidence.get('source_action_id', action['action_id'])]
            if (owner['tool'], owner['arguments']) != (observation['tool'], observation['arguments']):
                raise ValueError('Receipt belongs to a different action')
            if ordinal not in {s['observation_index'] for s in sources}:
                sources.append({'observation_index': ordinal, 'turn': observation['turn'],
                    'tool': observation['tool'], 'response': observation['response']})
        rows.append({k: action[k] for k in ('action_id', 'parent_action_id', 'turn', 'attempt',
            'tool', 'arguments', 'state', 'extra_query', 'goal_ids', 'workflow_id')} | {
                'transitions': [t['to'] for t in action['transitions']], 'public_evidence': sources})
    costs = {'calls': len(calls), 'known_input_tokens': 0, 'known_output_tokens': 0,
             'known_total_tokens': 0, 'calls_without_usage': 0}
    for call in calls:
        usage = call.get('response', {}).get('usage')
        if not usage:
            costs['calls_without_usage'] += 1
            continue
        for target, source in (('known_input_tokens','prompt_tokens'),
                               ('known_output_tokens','completion_tokens'),('known_total_tokens','total_tokens')):
            costs[target] += usage[source]
    if costs['known_total_tokens'] != summary['actor_tokens'] or costs['calls'] != summary['actor_model_calls']:
        raise ValueError('Summary actor accounting differs from original receipts')
    return {'schema': 'episode-review-v1', 'task_id': summary['task_id'], 'variant': summary['variant'],
        'actor_seed': summary['actor_seed'], 'source_episode': str(folder),
        'integrity': integrity, 'source_sha256': checksums, 'official_success': summary['success'],
        'official_score': summary['official_score'], 'task_failure': summary['task_failure'],
        'action_states': dict(Counter(a['state'] for a in actions)), 'actions': rows, 'actor_cost': costs,
        'public_task_message': next((m['content'] for c in calls for m in c['request']['messages']
            if m['role']=='user' and 'This is your actual task.' in m['content']), None),
        'goal_extraction': (audit.get('task_spec') or {}).get('status', 'disabled'),
        'limits': 'Existing developer trace; no live control or re-evaluation. Self-hashes alone do not '
                  'authenticate a source. Action/registration completion is not user goal satisfaction; '
                  'official outcome is copied from the original summary. Missing usage makes costs a lower bound.'}


def markdown(report):
    lines = [f"# {report['task_id']} / {report['variant']} / seed{report['actor_seed']}", '',
        f"原官方结果：success={report['official_success']}；score={report['official_score']}。",
        f"来源核验：{report['integrity']}；目标提取：{report['goal_extraction']}。", '',
        '```text', report['public_task_message'] or 'No task message in recorded requests.', '```', '',
        '| 动作 | 轮次 | 工具 | 状态 | 尝试 | 父提案 | 公开观测 |',
        '|---|---:|---|---|---:|---|---|']
    for action in report['actions']:
        evidence = ', '.join(str(e['observation_index']) for e in action['public_evidence']) or '—'
        lines.append(f"| {action['action_id']} | {action['turn']} | {action['tool']} | {action['state']} | "
                     f"{action['attempt']} | {action['parent_action_id'] or '—'} | {evidence} |")
    lines += ['', '参数与回执：']
    for action in report['actions']:
        lines += ['', f"<details><summary>{action['action_id']} {action['tool']}</summary>", '',
            '```json', json.dumps(action, ensure_ascii=False, indent=2), '```', '', '</details>']
    cost = report['actor_cost']
    lines += ['', f"Actor已知tokens：{cost['known_total_tokens']}；缺usage请求：{cost['calls_without_usage']}。",
              '', report['limits'], '']
    return '\n'.join(lines)
