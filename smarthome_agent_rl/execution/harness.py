"""Opt-in ReAct tool boundary backed by trace and public state verification.

The original GuardedExecutor remains the frozen G path. This adapter shares its
audit envelope for existing evaluators but does not invent query lifecycles.
It does not infer task goals or silently replace device-native workflows.
"""
import copy

from ..guard import GuardError
from .mutation import MutationExecutor, VerificationStatus
from .trace import ToolTrace


def ok(response):
    return (isinstance(response, dict) and response.get('status', {}).get('code') == 200
            and response.get('error') is None)


class RuntimeExecutor:
    def __init__(self, *, guard, adapter, dispatch, invocation_factory, sources=None,
                 query_limit=40, audit_fn=None, verify_mutations=True):
        self.guard, self.adapter = guard, adapter
        self.invocation_factory = invocation_factory
        self.sources, self.query_limit, self.audit_fn = sources or {}, query_limit, audit_fn
        self.verify_mutations = verify_mutations
        self.trace = ToolTrace(dispatch, sink=self._observe)
        self.mutations = MutationExecutor(self.trace, readback=self._readback)
        self.audit, self.actual, self.observations = [], [], []
        self.structured_audit, self.failures = [], {}
        self.extra_queries, self.turn, self._extra = 0, 0, False
        self.task_spec = self.time_plan = None

    def _observe(self, row):
        response = row['response']
        if row['error'] and response is None:
            response = {'status': {'code': 503, 'state': 'unknown'}, 'data': {},
                        'error': {'type': 'dispatch_unknown', 'evidence': row['error']}}
        self.actual.append(self.invocation_factory(tool=row['tool'], params=row['arguments'],
                                                  observation=copy.deepcopy(response)))
        self.observations.append({'turn': self.turn, 'tool': row['tool'],
            'arguments': row['arguments'], 'response': copy.deepcopy(response),
            'extra_query': self._extra, 'duration_seconds': row['latency'],
            'invocation_id': row['invocation_id']})

    def _readback(self, tool, arguments, **linkage):
        if self.extra_queries >= self.query_limit:
            return None
        self.extra_queries += 1
        self._extra = True
        try:
            return self.trace.call(tool, arguments, **linkage)
        finally:
            self._extra = False

    def record_structured(self, records):
        self.structured_audit = records
        self.save_audit()

    def save_audit(self):
        if self.audit_fn:
            self.audit_fn({'proposals': self.audit, 'actual_observations': self.observations,
                'public_semantics_source_sha256': self.sources, 'extra_queries': self.extra_queries,
                'structured': self.structured_audit, 'tool_trace': [r.as_dict() for r in self.trace.invocations],
                'runtime_verify': self.verify_mutations,
                'execution_runtime': 'public-mutation-v1', 'action_lifecycle': {},
                'context': [], 'binding': [], 'start_semantics': [],
                'recovery_policy': None, 'task_spec': None, 'time_plan': None})

    def execute(self, tool, arguments):
        self.turn = self.structured_audit[-1]['turn'] if self.structured_audit else self.turn + 1
        record = {'turn': self.turn, 'tool': tool, 'arguments': copy.deepcopy(arguments),
                  'blocked': False, 'uncovered': [], 'reached_executor': False}
        self.audit.append(record)
        calls_before, queries_before = len(self.actual), self.extra_queries
        key = repr((tool, arguments))
        try:
            self.guard.schema(tool, arguments)
            if tool in ('execute_command', 'write_attribute'):
                before = self._readback('get_device_structure', {'device_id': arguments['device_id']})
                if before is None:
                    raise GuardError('deterministic', 'No prequery budget remains', reason_code='PREQUERY_BUDGET_EXHAUSTED')
                if before.error or not ok(before.response) or not isinstance(before.response.get('data'), dict):
                    raise GuardError('capability', 'Public device structure unavailable', response=before.response)
                state = before.response['data']
                if state.get('device_id') != arguments['device_id']:
                    raise GuardError('capability', 'Public device identity differs', reason_code='PREQUERY_IDENTITY_MISMATCH')
                contract, name, args = self.adapter.build(tool, arguments, state)
                result = self.mutations.execute(tool, arguments, contract=contract, state=state,
                    function_name=name, contract_args=args, parent_step=str(self.turn), verify=self.verify_mutations)
                record['contract'] = result.guard
                record['uncovered'] = result.guard.get('uncovered', [])
                record['mutation'] = result.as_dict()
                record['verification'] = {**result.verification, 'status': result.status.value,
                    'verified': True if result.status == VerificationStatus.VERIFIED_SUCCESS else
                    False if result.status == VerificationStatus.VERIFIED_FAILURE else None}
                if result.mutation_invocation_id is None:
                    raise GuardError('deterministic', 'Device contract rejected or lacks public preconditions',
                        reason_code=result.guard.get('reason_code', 'CONTRACT_UNCOVERED'), contract=result.guard)
                record['reached_executor'] = True
                mutation = next(r for r in self.trace.invocations if r.invocation_id == result.mutation_invocation_id)
                response = copy.deepcopy(mutation.response)
                if not isinstance(response, dict):
                    response = {'status': {'code': 503, 'state': 'unknown'}, 'data': {},
                                'error': {'type': 'dispatch_unknown', 'evidence': mutation.error},
                                'raw_receipt': response}
                # Preserve the receipt/status/error. The verifier provides a
                # separate business-state conclusion even when the API timed out.
                response['harness_verification'] = copy.deepcopy(record['verification'])
                failed = result.status != VerificationStatus.VERIFIED_SUCCESS
                record['simulator_error'] = not ok(mutation.response)
            else:
                if tool == 'get_attribute':
                    before = self._readback('get_device_structure', {'device_id': arguments['device_id']})
                    if before is None or before.error or not ok(before.response):
                        raise GuardError('capability', 'Public query capability unavailable')
                    self.guard.capability(tool, arguments, before.response['data'], state=False)
                if tool == 'schedule_workflow':
                    self._check_workflow(arguments, record)
                row = self.trace.call(tool, arguments, parent_step=str(self.turn))
                record['reached_executor'] = True
                response = copy.deepcopy(self.observations[-1]['response'])
                if tool in ('schedule_workflow', 'cancel_workflow'):
                    record['verification'] = {'status': 'REGISTRATION_ONLY' if tool == 'schedule_workflow' else 'CANCELLATION_RECEIPT_ONLY',
                                              'verified': None, 'future_success_verified': False}
                    if isinstance(response, dict):
                        response['harness_verification'] = copy.deepcopy(record['verification'])
                failed = bool(row.error) or not ok(response)
                record['simulator_error'] = failed
            if failed:
                self.failures[key] = self.failures.get(key, 0) + 1
            else:
                record['recovered'] = bool(self.failures.pop(key, 0))
            return response
        except GuardError as exc:
            record.update(blocked=True, layer=exc.layer, detail=exc.detail, response=exc.response())
            return record['response']
        finally:
            record['extra_queries'] = self.extra_queries - queries_before
            record['actual_calls'] = len(self.actual) - calls_before
            self.save_audit()

    def _check_workflow(self, arguments, record):
        structures = {}
        queried = False
        for step in arguments['steps']:
            device = step['args']['device_id']
            if not queried:
                queried = True
                row = self._readback('get_device_structure', {'device_id': device})
                if row is not None and not row.error and ok(row.response):
                    structures[device] = row.response['data']
                    if structures[device].get('device_id') != device:
                        raise GuardError('capability', 'Workflow device identity differs')
            # Validate static API capability only. Current state must not be
            # substituted for the state at scheduled execution time.
            if device in structures:
                record['uncovered'].extend(self.guard.capability(step['tool'], step['args'], structures[device], state=False))
            else:
                record['uncovered'].append('workflow_device_not_queried:' + device)
        record['uncovered'].append('future_state_preconditions')
