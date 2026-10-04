"""Lossless query ledger with provenance and explicit stale-state markers; no LLM summary."""
import copy
import json

from src.agents.types import ChatMessage

STATIC_TOOLS = {'get_rooms', 'get_room_devices', 'get_cluster_doc', 'get_environment_control_rules'}
MUTATIONS = {'execute_command', 'write_attribute', 'schedule_workflow', 'cancel_workflow',
             'add_device', 'remove_device', 'set_tick_interval'}


def flatten(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, prefix + '/' + str(key).replace('~', '~0').replace('/', '~1')))
        return result or {prefix: {}}
    return {prefix: copy.deepcopy(value)}


def build_ledger(observations, proposals, structured=()):
    facts, receipts, errors, epoch = {}, [], [], 0
    for ordinal, observation in enumerate(observations, 1):
        tool, arguments, response = observation['tool'], observation['arguments'], observation['response']
        source = {'turn': observation['turn'], 'tool': tool, 'arguments': arguments,
                  'extra_query': observation['extra_query'], 'observation_index': ordinal}
        successful = isinstance(response, dict) and response.get('status', {}).get('code') == 200 and response.get('error') is None
        if not successful:
            errors.append({'source': source, 'response': response})
            continue
        if tool in MUTATIONS:
            epoch += 1
            receipts.append({'source': source, 'response': response,
                             'workflow_registration_is_not_future_success': tool == 'schedule_workflow'})
            continue
        if tool in ('get_current_time', 'get_workflow_status', 'get_workflow_list'):
            # Time and asynchronous workflows may invalidate earlier state without an agent write.
            epoch += 1
        key = json.dumps([tool, arguments], sort_keys=True, separators=(',', ':'))
        value = copy.deepcopy(response)
        previous = facts.get(key)
        observed_at = previous['observed_at'] if previous else []
        versions = previous['previous_versions'] if previous else []
        if previous and previous['response'] != value:
            old, new = flatten(previous['response']), flatten(value)
            # Old leaf values and absence markers reconstruct every previous response exactly.
            changes = {path: {'present': path in old, 'value': old.get(path)}
                       for path in old.keys() | new.keys() if old.get(path) != new.get(path) or (path in old) != (path in new)}
            versions = [*versions, {'source': previous['source'], 'epoch': previous['epoch'],
                                    'previous_leaf_values': changes}]
        facts[key] = {'source': source, 'epoch': epoch, 'response': value,
                      'previous_versions': versions, 'static': tool in STATIC_TOOLS,
                      'observed_at': [*observed_at, {'turn': source['turn'], 'epoch': epoch,
                                                   'observation_index': ordinal}]}
    for fact in facts.values():
        fact['stale'] = not fact['static'] and fact['epoch'] < epoch
    for proposal in proposals:
        if proposal.get('blocked'):
            errors.append({'source': {'turn': proposal['turn'], 'tool': proposal['tool'],
                'arguments': proposal['arguments']}, 'guard_error': proposal.get('response')})
        if 'verification' in proposal:
            receipts.append({'source': {'turn': proposal['turn']}, 'verification': proposal['verification']})
    for proposal in structured:
        if 'validation_error' in proposal:
            errors.append({'source': {'turn': proposal['turn']},
                           'output_schema_error': proposal['validation_error']})
    return {'epoch': epoch, 'facts': facts, 'action_receipts': receipts, 'errors': errors,
            'rule': 'Facts are observations at their recorded turn. Stale facts are history, not current preconditions. Query public state when needed.'}


class LedgerProvider:
    def __init__(self, inner, executor):
        self.inner, self.executor = inner, executor

    def generate(self, messages, response_format=None):
        start = max(i for i, message in enumerate(messages) if 'This is your actual task.' in message.content)
        history = messages[start + 1:]
        # Keep the most recent two complete action/observation pairs verbatim.
        assistant_indices = [i for i, message in enumerate(history) if message.role == 'assistant']
        recent = history[assistant_indices[-2]:] if len(assistant_indices) >= 2 else history
        ledger = build_ledger(self.executor.observations, self.executor.audit, self.executor.structured_audit)
        prompt = 'PUBLIC OBSERVATION LEDGER (no task success goals):\n' + json.dumps(ledger, ensure_ascii=False,
                                                                                  separators=(',', ':'))
        converted = [*messages[:start + 1], ChatMessage(role='user', content=prompt), *recent]
        original_size = sum(len(m.content) for m in messages)
        managed_size = sum(len(m.content) for m in converted)
        # Compression is optional when the complete ledger is larger than raw history.
        used = managed_size < original_size
        self.executor.context_audit.append({'turn': len(assistant_indices) + 1,
            'original_characters': original_size, 'managed_characters': managed_size, 'used': used,
            'facts': len(ledger['facts']), 'errors_retained': len(ledger['errors']), 'epoch': ledger['epoch']})
        return self.inner.generate(converted if used else messages, response_format=response_format)


def compact_ledger(executor, recent_turns=()):
    """Keep source evidence in audit files; render current versions without duplicating recent pairs."""
    ledger = build_ledger(executor.observations, executor.audit, executor.structured_audit)
    catalog_turn = max((o['turn'] for o in executor.observations if o['tool'] in ('add_device', 'remove_device')
        and o['response'].get('status', {}).get('code') == 200), default=-1)
    facts = []
    for fact in ledger['facts'].values():
        source = fact['source']
        if source['turn'] in recent_turns:
            continue
        catalog_stale = source['tool'] in ('get_rooms', 'get_room_devices', 'get_device_structure') and source['turn'] < catalog_turn
        facts.append({'tool': source['tool'], 'args': source['arguments'], 'turn': source['turn'],
            'stale': fact['stale'] or catalog_stale, 'data': fact['response'].get('data'),
            'source_index': source['observation_index']})
    receipts = []
    for receipt in ledger['action_receipts']:
        if receipt['source']['turn'] in recent_turns:
            continue
        if 'verification' in receipt:
            receipts.append({'turn': receipt['source']['turn'], 'verification': receipt['verification']})
        else:
            source = receipt['source']
            receipts.append({'turn': source['turn'], 'tool': source['tool'], 'args': source['arguments'],
                'data': receipt['response'].get('data'),
                'registration_only': receipt['workflow_registration_is_not_future_success']})
    errors = []
    for error in ledger['errors']:
        source = error['source']
        errors.append({'turn': source['turn'], 'tool': source.get('tool'), 'args': source.get('arguments'),
            'error': error.get('guard_error', error.get('output_schema_error', error.get('response')))})
    return {'facts': facts, 'receipts': receipts, 'errors': errors,
        'rule': 'Observations are historical. Stale values require fresh public queries for current preconditions. Workflow registration proves no future success.'}


class CompactLedgerProvider:
    def __init__(self, inner, executor, token_count_fn=None):
        self.inner, self.executor, self.token_count_fn = inner, executor, token_count_fn

    def generate(self, messages, response_format=None):
        start = max(i for i, m in enumerate(messages) if 'This is your actual task.' in m.content)
        history = messages[start + 1:]
        assistant_indices = [i for i, m in enumerate(history) if m.role == 'assistant']
        recent = history[assistant_indices[-2]:] if len(assistant_indices) >= 2 else history
        recent_turns = range(max(1, len(assistant_indices) - 1), len(assistant_indices) + 1)
        ledger = compact_ledger(self.executor, recent_turns)
        prompt = 'COMPACT PUBLIC OBSERVATIONS:\n' + json.dumps(ledger, ensure_ascii=False, separators=(',', ':'))
        converted = [*messages[:start + 1], ChatMessage(role='user', content=prompt), *recent]
        original_size = sum(len(m.content) for m in messages)
        managed_size = sum(len(m.content) for m in converted)
        used = len(assistant_indices) > 2 and managed_size < original_size
        raw_tokens = managed_tokens = None
        if used and self.token_count_fn:
            raw_tokens = self.token_count_fn(messages)
            managed_tokens = self.token_count_fn(converted)
            used = managed_tokens < raw_tokens
        self.executor.context_audit.append({'version': 2, 'turn': len(assistant_indices) + 1,
            'original_characters': original_size, 'managed_characters': managed_size, 'used': used,
            'estimated_original_tokens': raw_tokens, 'estimated_managed_tokens': managed_tokens,
            'token_scope': 'Model chat tokenizer including generation prompt; HTTP usage remains authoritative',
            'facts': len(ledger['facts']), 'errors_retained': len(ledger['errors']),
            'recent_pairs': min(2, len(assistant_indices)), 'historical_versions_in_audit_only': True})
        return self.inner.generate(converted if used else messages, response_format=response_format)
