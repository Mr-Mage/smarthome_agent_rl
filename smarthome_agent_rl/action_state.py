"""Episode-local action evidence; observation only, never dispatches or retries tools."""
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re


MUTATIONS = frozenset({'execute_command', 'write_attribute', 'schedule_workflow',
                      'cancel_workflow', 'add_device', 'remove_device', 'set_tick_interval'})
TRANSITIONS = {
    'proposed': {'dispatched', 'rejected', 'aborted'},
    'dispatched': {'acknowledged', 'failed', 'unknown'},
    'acknowledged': {'registered', 'completed'},
    'registered': {'running', 'completed', 'failed', 'cancelled'},
    'running': {'completed', 'failed', 'cancelled'},
    'unknown': {'acknowledged', 'completed', 'failed'},
    'rejected': set(), 'aborted': set(), 'failed': set(),
    'completed': set(), 'cancelled': set(),
}


def response_outcome(response):
    """Normalize public envelopes, including the upstream HTTPError wrapper."""
    status = response.get('status') if isinstance(response, dict) else None
    code = status.get('code') if isinstance(status, dict) else None
    error = response.get('error') if isinstance(response, dict) else None
    if code == 200 and error is None:
        return 'acknowledged', code
    error_type = error.get('type') if isinstance(error, dict) else None
    if error_type == 'HTTP_ERROR':
        # tools._make_request wraps HTTPError with envelope 400 and original status in detail.
        detail = error.get('detail')
        match = re.match(r'^(\d{3}):', detail) if isinstance(detail, str) else None
        code = int(match.group(1)) if match else None
    uncertain = (not isinstance(code, int) or code in (408, 429, 499) or code >= 500
                 or error_type in ('TIMEOUT_ERROR', 'CONNECTION_ERROR', 'INTERNAL_ERROR'))
    return ('unknown' if uncertain else 'failed'), code


def action_signature(tool, arguments, *, extra_query=False):
    return json.dumps([tool, arguments, extra_query], sort_keys=True,
                      ensure_ascii=False, separators=(',', ':'))


@dataclass
class ActionRecord:
    action_id: str
    tool: str
    arguments: dict
    turn: int
    attempt: int
    request_sha256: str
    observation_version: int
    parent_action_id: str | None = None
    goal_id: str | None = None
    goal_ids: list = field(default_factory=list)
    extra_query: bool = False
    state: str = 'proposed'
    workflow_id: str | None = None
    transitions: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


class ActionLedger:
    def __init__(self):
        self.records = {}
        self.attempts = {}

    def propose(self, tool, arguments, *, turn, observation_version,
                extra_query=False, parent_action_id=None):
        signature = action_signature(tool, arguments, extra_query=extra_query)
        attempt = self.attempts.get(signature, 0) + 1
        self.attempts[signature] = attempt
        action_id = f'a{len(self.records) + 1:06d}'
        self.records[action_id] = ActionRecord(action_id, tool, copy.deepcopy(arguments),
            turn, attempt, hashlib.sha256(signature.encode()).hexdigest(), observation_version,
            parent_action_id=parent_action_id, extra_query=extra_query,
            transitions=[{'from': None, 'to': 'proposed', 'evidence': {'kind': 'tool_proposal'}}])
        return action_id

    def transition(self, action_id, state, evidence):
        record = self.records[action_id]
        if state not in TRANSITIONS[record.state]:
            raise ValueError(f'Invalid action transition: {record.state} -> {state}')
        evidence = copy.deepcopy(evidence)
        record.transitions.append({'from': record.state, 'to': state, 'evidence': evidence})
        record.evidence.append(evidence)
        record.state = state

    def dispatched(self, action_id):
        self.transition(action_id, 'dispatched', {'kind': 'dispatch_started'})

    def rejected(self, action_id, response):
        evidence = {'kind': 'guard_rejection', 'response': response}
        if self.records[action_id].state == 'proposed':
            self.transition(action_id, 'rejected', evidence)
        else:
            # A later harness failure cannot erase an already dispatched operation.
            self.records[action_id].evidence.append(evidence)

    def interrupted(self, action_id, exc):
        state = self.records[action_id].state
        if state == 'proposed':
            self.transition(action_id, 'aborted', {'kind': 'pre_dispatch_exception',
                                                 'exception_type': type(exc).__name__})
        elif state == 'dispatched':
            self.transition(action_id, 'unknown', {'kind': 'dispatch_exception',
                                                 'exception_type': type(exc).__name__})

    def observed(self, action_id, response, observation_index):
        record = self.records[action_id]
        evidence = {'kind': 'tool_response', 'observation_index': observation_index}
        state, code = response_outcome(response)
        evidence['status_code'] = code
        if state != 'acknowledged':
            # A transport/server failure can arrive as an ordinary tool response.
            # No receipt of successful execution is available for a mutation.
            self.transition(action_id, state, evidence)
            return
        self.transition(action_id, 'acknowledged', evidence)
        data = response.get('data')
        data = data if isinstance(data, dict) else {}
        if record.tool == 'schedule_workflow':
            workflow_id = data.get('workflow_id')
            if isinstance(workflow_id, str) and workflow_id:
                record.workflow_id = workflow_id
                self.transition(action_id, 'registered', {**evidence, 'kind': 'workflow_registration',
                                                         'workflow_id': workflow_id})
        elif record.tool not in MUTATIONS:
            self.transition(action_id, 'completed', {**evidence, 'kind': 'read_receipt',
                                                    'scope': 'query_only'})
        # Only existing public queries/cancellation receipts can advance registrations.
        if record.tool in ('get_workflow_status', 'cancel_workflow'):
            workflow_id = record.arguments.get('workflow_id')
            if data.get('workflow_id') not in (None, workflow_id):
                return
            state = data.get('status')
            if state in ('running', 'completed', 'failed', 'cancelled'):
                for previous in self.records.values():
                    if previous.workflow_id != workflow_id or not workflow_id:
                        continue
                    source = {**evidence, 'kind': 'workflow_status', 'workflow_id': workflow_id,
                              'reported_status': state, 'source_action_id': action_id,
                              'scope': 'workflow_only'}
                    if state in TRANSITIONS[previous.state]:
                        self.transition(previous.action_id, state, source)
                    else:
                        # Keep duplicate/out-of-order evidence without regressing terminal states.
                        previous.evidence.append(source)

    def snapshot(self):
        return {'schema': 'episode-actions-v1', 'scope': 'episode_local',
                'actions': [asdict(record) for record in self.records.values()]}
