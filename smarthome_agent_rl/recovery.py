"""Opt-in semantic repair budget, independent of postcondition verification.

No automatic transport retry, mutation replay or inference of successful effects.
"""
import hashlib
import json

from smarthome_agent_rl.action_state import MUTATIONS, action_signature, response_outcome
from smarthome_agent_rl.guard import GuardError


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def public_context(tool, arguments, observations):
    """Relevant public evidence, not observation ordinal (identical reads don't reset budgets)."""
    devices = {arguments['device_id']} if arguments.get('device_id') else set()
    if tool == 'schedule_workflow':
        devices.update(step.get('args', {}).get('device_id') for step in arguments.get('steps', []))
    facts, receipts = {}, []
    for row in observations:
        if response_outcome(row['response'])[0] != 'acknowledged':
            continue
        name, args = row['tool'], row['arguments']
        relevant = (args.get('device_id') in devices and args.get('device_id') is not None)
        relevant |= name in ('get_rooms', 'get_room_devices', 'add_device', 'remove_device')
        relevant |= tool == 'schedule_workflow' and name == 'get_current_time'
        relevant |= (arguments.get('workflow_id') is not None
                     and args.get('workflow_id') == arguments['workflow_id'])
        if not relevant:
            continue
        if name in MUTATIONS:
            # Accepted mutations can change preconditions even before a fresh read.
            receipts.append([name, args, row['response']])
        else:
            facts[action_signature(name, args)] = row['response']
    return fingerprint({'facts': facts, 'receipts': receipts})


class RecoveryPolicy:
    def __init__(self, repair_limit=2, total_limit=6):
        if isinstance(repair_limit, bool) or not isinstance(repair_limit, int) or repair_limit < 0:
            raise ValueError('repair_limit must be a nonnegative integer')
        self.max_attempts = repair_limit + 1
        if isinstance(total_limit, bool) or not isinstance(total_limit, int) or total_limit < self.max_attempts:
            raise ValueError('total_limit must be an integer >= initial call plus repairs')
        self.total_limit = total_limit
        self.total_failures = {}
        self.failures = {}
        self.unknown_mutations = {}
        self.events = []

    def check(self, tool, arguments, context):
        signature = action_signature(tool, arguments)
        if signature in self.unknown_mutations:
            self.events.append({'kind': 'unknown_mutation_blocked', 'request_sha256': fingerprint(signature)})
            raise GuardError('recovery_unknown', 'Prior identical mutation has an unknown outcome; '
                             'do not replay it. Query public receipts/state or finish honestly.',
                             prior_action_id=self.unknown_mutations[signature])
        if self.total_failures.get(signature, 0) >= self.total_limit:
            self.events.append({'kind': 'request_budget_blocked', 'request_sha256': fingerprint(signature)})
            raise GuardError('recovery_total', 'Episode failure budget exhausted for identical '
                             'parameters, including attempts after changed evidence; change plan or finish honestly.',
                             attempts=self.total_failures[signature])
        prior = self.failures.get(signature)
        if prior and prior['context'] == context and prior['count'] >= self.max_attempts:
            self.events.append({'kind': 'repair_budget_blocked', **prior})
            raise GuardError('recovery_budget', 'Repair budget exhausted for identical parameters '
                             'and unchanged public evidence; change plan or finish honestly.',
                             prior_action_id=prior['action_id'], attempts=prior['count'],
                             error_fingerprint=prior['error_fingerprint'])

    def observe(self, tool, arguments, response, context, action_id):
        signature = action_signature(tool, arguments)
        outcome, code = response_outcome(response)
        if outcome == 'unknown':
            if tool in MUTATIONS:
                self.unknown_mutations[signature] = action_id
            self.events.append({'kind': 'unknown_result', 'action_id': action_id, 'mutation': tool in MUTATIONS})
            return
        if outcome == 'acknowledged':
            self.failures.pop(signature, None)
            return
        self.total_failures[signature] = self.total_failures.get(signature, 0) + 1
        error = response.get('error') if isinstance(response, dict) else None
        error_key = fingerprint({'code': code, 'error': error})
        previous = self.failures.get(signature)
        repeated = previous and previous['context'] == context and previous['error_fingerprint'] == error_key
        record = {'request_sha256': fingerprint(signature), 'context': context,
                  'error_fingerprint': error_key, 'action_id': action_id,
                  'count': previous['count'] + 1 if repeated else 1}
        self.failures[signature] = record
        self.events.append({'kind': 'confirmed_failure', **record})

    def snapshot(self):
        return {'schema': 'episode-recovery-v1', 'max_attempts': self.max_attempts,
                'total_failure_limit': self.total_limit,
                'events': self.events}
