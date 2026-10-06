"""Build public-only semantic verification examples and deterministic hard negatives."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smarthome_agent_rl.semantic_verifier import RuleSemanticVerifier, VerificationContext


def _action(record):
    return copy.deepcopy(record.get('action') or record.get('proposed_action') or {})


def _devices(state):
    value = state.get('devices', {}) if isinstance(state, dict) else {}
    return value if isinstance(value, dict) else {row.get('device_id'): row for row in value if row.get('device_id')}


def _decision(goal, state, action, recent=()):
    result = RuleSemanticVerifier().verify(VerificationContext(goal, state, action, recent))
    return {name: getattr(result, name).label for name in
            ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute')}


def build_examples(records):
    examples = []
    for record in records:
        if record.get('infrastructure_error') or not record.get('success', True):
            continue
        goal = record.get('user_goal') or record.get('instruction') or record.get('task', {}).get('instruction')
        state = copy.deepcopy(record.get('environment_state') or record.get('initial_state') or {})
        action = _action(record)
        if not goal or not action:
            continue
        recent = list(copy.deepcopy(record.get('recent_actions', ())))
        base = {'source': record.get('task_id') or record.get('task', {}).get('task_id'),
                'user_goal': goal, 'environment_state': state, 'recent_actions': recent,
                'proposed_action': action, 'kind': 'positive',
                'labels': _decision(goal, state, action, recent)}
        examples.append(base)
        devices = _devices(state)
        alternatives = [key for key in devices if key != action.get('device_id')]
        if alternatives:
            wrong = copy.deepcopy(action)
            wrong['device_id'] = alternatives[0]
            examples.append({**base, 'kind': 'hard_negative_wrong_target', 'proposed_action': wrong,
                             'labels': _decision(goal, state, wrong, recent)})
        command = str(action.get('command_id', action.get('function', ''))).lower()
        if command in {'on', 'off', 'start', 'stop'}:
            opposite = copy.deepcopy(action)
            opposite_key = {'on': 'off', 'off': 'on', 'start': 'stop', 'stop': 'start'}[command]
            if 'command_id' in opposite:
                opposite['command_id'] = opposite_key
            else:
                opposite['function'] = opposite_key
            examples.append({**base, 'kind': 'hard_negative_opposite_action', 'proposed_action': opposite,
                             'labels': _decision(goal, state, opposite, recent)})
        examples.append({**base, 'kind': 'hard_negative_duplicate', 'recent_actions': recent + [action],
                         'labels': _decision(goal, state, action, recent + [action])})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='JSON array or JSONL public rollout records')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.input)
    text = source.read_text(encoding='utf-8')
    try:
        value = json.loads(text)
        records = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    examples = build_examples(records)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as stream:
        for row in examples:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    manifest = target.with_suffix(target.suffix + '.manifest.json')
    manifest.write_text(json.dumps({'schema': 'semantic-verifier-dataset-v1', 'records': len(examples),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'hidden_evaluator_fields': 'excluded'}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'records': len(examples), 'data_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
