"""Reference identical public reads while retaining plans, errors and all versions.

This is a prompt representation, never a cache or a goal-satisfaction checker.
Only public, successful, identical reads in one mutation epoch are eligible.
"""
import copy
import hashlib
import json
import time


READS = frozenset({'get_rooms', 'get_room_devices', 'get_device_structure',
    'get_home_state', 'get_room_states', 'get_all_attributes', 'get_attribute',
    'get_cluster_doc', 'get_environment_control_rules'})
MUTATIONS = frozenset({'execute_command', 'write_attribute', 'schedule_workflow',
    'cancel_workflow', 'add_device', 'remove_device', 'set_tick_interval'})
MARKER = 'This is your actual task.'
REFERENCE = 'observation: {"public_evidence_reference":'


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def action(content):
    try:
        body = json.loads(content)
        if 'call' in body:
            tool, arguments = body['call']['tool'], body['call']['arguments']
        else:
            tool, arguments = body['action'], body['action_input']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        return (tool, arguments) if isinstance(tool, str) and isinstance(arguments, dict) else (None, None)
    except (ValueError, TypeError, KeyError):
        return None, None


def successful(value):
    return (isinstance(value, dict) and isinstance(value.get('status'), dict)
            and value['status'].get('code') == 200 and value.get('error') is None)


def render(messages, observations):
    """Pure transform of role/content dictionaries. Indices never change.

    Retain the first full response and latest two complete pairs. A reference
    binds the exact response string, call arguments, turn and public receipt.
    Any mutation proposal (including a failed/unknown one) breaks the epoch.
    Unrecognised histories pass through without guessing.
    """
    converted = copy.deepcopy(messages)
    starts = [i for i, m in enumerate(messages)
              if m['role'] == 'user' and MARKER in m['content']]
    if len(starts) != 1:
        return converted, []
    start = starts[0]
    assistants = [i for i in range(start + 1, len(messages)) if messages[i]['role'] == 'assistant']
    cutoff = assistants[-2] if len(assistants) >= 2 else start + 1
    public = {}
    for ordinal, obs in enumerate(observations, 1):
        if not obs['extra_query']:
            public.setdefault(obs['turn'], []).append((ordinal, obs))
    seen, refs, turn, epoch = {}, [], 0, 0
    previous = None
    for i in range(start + 1, len(messages)):
        message = messages[i]
        if message['role'] == 'assistant':
            turn += 1
            previous = action(message['content'])
            if previous[0] in MUTATIONS or previous[0] is None:
                epoch += 1
                seen.clear()
            continue
        if previous is None or i == 0 or messages[i - 1]['role'] != 'assistant':
            continue
        tool, arguments = previous
        content = message['content']
        if message['role'] != 'user' or tool not in READS or not content.startswith('observation: '):
            continue
        try:
            response = json.loads(content[len('observation: '):])
        except (ValueError, TypeError):
            continue
        matches = [(ordinal, obs) for ordinal, obs in public.get(turn, [])
                   if (obs['tool'], obs['arguments'], obs['response']) == (tool, arguments, response)]
        if not successful(response) or len(matches) != 1:
            continue
        ordinal = matches[0][0]
        key = json.dumps([tool, arguments, content], ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen[key] = (i, turn, ordinal)
            continue
        source_index, source_turn, source_observation = seen[key]
        reference = {'source_message_index': source_index, 'source_turn': source_turn,
            'source_observation_index': source_observation, 'observation_index': ordinal,
            'observed_turn': turn, 'epoch': epoch, 'tool': tool, 'args': arguments,
            'response_content_sha256': sha(content)}
        replacement = 'observation: ' + json.dumps({'public_evidence_reference': reference,
            'meaning': 'This read returned exactly the full response at source_message_index. '
                       'Evidence is historical at observed_turn, not a current precondition or goal success.'},
            ensure_ascii=False, separators=(',', ':'))
        if i < cutoff and len(replacement) < len(content):
            converted[i]['content'] = replacement
            refs.append({'message_index': i, **reference})
    return converted, refs


def restore(messages):
    """Expand references using only evidence still present in the prompt."""
    restored = copy.deepcopy(messages)
    for i, message in enumerate(messages):
        if not message['content'].startswith(REFERENCE):
            continue
        ref = json.loads(message['content'][len('observation: '):])['public_evidence_reference']
        index = ref['source_message_index']
        if type(index) is not int or not 0 <= index < i:
            raise ValueError('Evidence source must precede the reference')
        source = restored[index]
        if source['role'] != 'user' or sha(source['content']) != ref['response_content_sha256']:
            raise ValueError('Evidence reference hash/role mismatch')
        restored[i]['content'] = source['content']
    return restored


class EvidenceContextProvider:
    def __init__(self, inner, executor, token_count_fn):
        if token_count_fn is None:
            raise ValueError('Evidence context requires the actual chat tokenizer')
        self.inner, self.executor, self.token_count_fn = inner, executor, token_count_fn

    def generate(self, messages, response_format=None):
        from src.agents.types import ChatMessage
        started = time.monotonic()
        raw = [{'role': m.role, 'content': m.content} for m in messages]
        compact, refs = render(raw, self.executor.observations)
        if restore(compact) != raw:
            raise ValueError('Evidence context lost original messages')
        converted = [ChatMessage(**m) for m in compact]
        raw_tokens = compact_tokens = None
        if refs:
            raw_tokens = self.token_count_fn(messages)
            compact_tokens = self.token_count_fn(converted)
        used = bool(refs) and compact_tokens < raw_tokens
        self.executor.context_audit.append({'version': 'evidence-reference-v1',
            'used': used, 'references': refs, 'original_tokens': raw_tokens,
            'managed_tokens': compact_tokens, 'lossless_roundtrip': True,
            'management_seconds': time.monotonic() - started,
            'scope': 'Prompt tokens only; full episode usage/latency determine online adoption.'})
        self.executor.save_audit()
        return self.inner.generate(converted if used else messages, response_format=response_format)
