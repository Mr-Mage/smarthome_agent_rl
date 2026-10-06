"""Opt-in task/scheduler attachment to an actual guarded single-turn episode.

Native workflows execute exactly once through the original dispatcher. Public
reads share the executor's query budget and evidence. Task goals come only from
the user query; action intentions never substitute for validated user goals.
"""
import copy
from datetime import datetime, timezone

from .harness import ok
from .mutation import PostconditionResult, VerificationStatus, check_postconditions, resolve
from .scheduler import Scheduler
from .store import RuntimeStore
from .tasks import TaskManager, TaskStatus
from .trace import ToolTrace
from ..device_contract import FunctionContract, _get_path


def virtual_seconds(value):
    # A coordinate for the simulator's naive calendar, not the machine clock.
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()


class EpisodeRuntime:
    def __init__(self, executor, adapter, path, *, tolerance=2, save=None):
        self.executor, self.adapter = executor, adapter
        self.store = RuntimeStore(path)
        self.manager = TaskManager(self.store)
        self.raw_dispatch = executor.dispatch
        self.trace = ToolTrace(self._dispatch_raw, sink=self._observe)
        self.scheduler = Scheduler(self.manager, self.trace, execute_action=self._unsupported,
            verify_job=self._verify_job, wake_agent=self._unsupported)
        self.tolerance, self.save = tolerance, save
        self.task = None
        self.structures, self.events = {}, []
        self._dispatch_exception = None
        self._supervising = False
        self.supervisor_queries = 0
        executor.dispatch = self.dispatch

    @staticmethod
    def _unsupported(*args):
        return PostconditionResult(VerificationStatus.UNVERIFIED, reason='OUTSIDE_SINGLE_TURN_NATIVE_SCOPE')

    def start(self, query, *, user_location=None, current_time=None):
        if self.task is not None:
            raise ValueError('An episode runtime cannot be reused for a second task')
        self.task = self.manager.create('episode-user', query, 'single-turn', constraints={
            'user_location': user_location, 'initial_public_time': current_time,
            'resource_namespace': 'isolated-episode'})
        self._event('task_created', task_id=self.task.task_id, goal_source='public_query',
                    expected_goal_conditions='uncovered; no evaluator labels or action-goal substitution')

    def _dispatch_raw(self, tool, arguments):
        self._dispatch_exception = None
        try:
            return self.raw_dispatch(tool, arguments)
        except Exception as exc:
            self._dispatch_exception = exc
            raise

    def _observe(self, row):
        row['task_id'] = self.task.task_id if self.task else None
        self.store.trace_sink(row)
        response = row['response']
        if row['tool'] == 'get_device_structure' and ok(response):
            device = row['arguments'].get('device_id')
            if isinstance(response.get('data'), dict) and response['data'].get('device_id') == device:
                self.structures[device] = copy.deepcopy(response['data'])

    def _event(self, kind, **evidence):
        self.events.append({'kind': kind, **copy.deepcopy(evidence)})
        self.flush()

    def flush(self):
        if self.save:
            self.save('task_runtime.json', {'version': 1, 'tasks': self.store.list('task'),
                'workflows': self.store.list('workflow'), 'jobs': self.store.list('job'),
                'events': self.events, 'trace': self.store.list('trace'),
                'supervisor_queries': self.supervisor_queries,
                'scope': 'public single-turn native delegation; action verification is not official task scoring'})

    def _read(self, tool, arguments):
        if self.executor.extra_queries >= self.executor.query_limit:
            return None
        before = len(self.trace.invocations)
        self.supervisor_queries += 1
        try:
            response = self.executor.call(tool, arguments, extra=True)
        except Exception as exc:
            self._event('supervisor_read_error', tool=tool, error=f'{type(exc).__name__}: {exc}')
            return None
        if len(self.trace.invocations) == before or not ok(response):
            return None
        return self.trace.invocations[-1]

    def _clock(self):
        receipt = self._read('get_current_time', {})
        try:
            return virtual_seconds(receipt.response['data']['now']) if receipt else None
        except (ValueError, TypeError, KeyError):
            return None

    def _intent(self, steps):
        conditions, uncovered, pending_paths = {}, [], {}
        for index, step in enumerate(steps):
            args, tool = step['args'], step['tool']
            device = args['device_id']
            structure = self.structures.get(device)
            if structure is None:
                receipt = self._read('get_device_structure', {'device_id': device})
                structure = receipt.response['data'] if receipt else None
            if structure is None:
                uncovered.append({'step': index, 'reason': 'PUBLIC_STRUCTURE_UNAVAILABLE'})
                continue
            try:
                contract, name, action_args = self.adapter.build(tool, args, structure, future=True)
                function = contract.functions.get(name)
                if function is None or not function.postconditions:
                    uncovered.append({'step': index, 'reason': 'PUBLIC_POSTCONDITION_UNCOVERED'})
                    continue
                pending_paths.setdefault(device, []).extend(function.verification.get('pending_paths', []))
                for condition in function.postconditions:
                    row = {**copy.deepcopy(condition), 'device_id': device,
                           'value': resolve(condition['value'], action={'args': action_args}, state=structure)}
                    conditions[(device, row['path'])] = row
            except (ValueError, TypeError, KeyError) as exc:
                uncovered.append({'step': index, 'reason': 'PUBLIC_CONTRACT_UNRESOLVED', 'error': str(exc)})
        return {'conditions': list(conditions.values()), 'uncovered': uncovered, 'pending_paths': pending_paths,
                'goal_source': 'actor-selected action intention, not validated user-task goal'}

    def dispatch(self, tool, arguments):
        if self.task is None:
            raise RuntimeError('Start a public task before dispatch')
        if tool == 'schedule_workflow':
            # Guards already ran before reaching this boundary. Persist intent
            # before the actual native call; never register the workflow twice.
            try:
                target = virtual_seconds(arguments['start_time'])
            except (ValueError, TypeError, KeyError):
                target = None
            if target is None:
                self._event('schedule_uncovered', reason='INVALID_PUBLIC_CALENDAR')
                row = self.trace.call(tool, arguments, task_id=self.task.task_id)
            else:
                payload = self._intent(arguments['steps'])
                job = self.scheduler.schedule(self.task.task_id, self.task.user_id,
                    expected_version=self.task.version, target_time=target, tolerance=self.tolerance,
                    payload=payload, native_call={'tool': tool, 'args': arguments})
                wf = self.store.get('workflow', job['workflow_id'])[0]
                row = next(r for r in self.trace.invocations
                           if r.invocation_id == wf['evidence'][-1]['invocation_id'])
                self._event('native_registration', job_id=job['job_id'], invocation_id=row.invocation_id)
        else:
            row = self.trace.call(tool, arguments, task_id=self.task.task_id)
            if tool in ('cancel_workflow', 'get_workflow_status'):
                self._observe_cancellation(arguments, row)
        if self._dispatch_exception is not None:
            raise self._dispatch_exception
        return copy.deepcopy(row.response)

    def _observe_cancellation(self, arguments, row):
        response = row.response
        external = arguments.get('workflow_id')
        data = response.get('data') if isinstance(response, dict) else None
        if not ok(response) or not isinstance(data, dict) or data.get('workflow_id') != external or \
                data.get('status') != 'cancelled':
            return
        with self.store.transaction() as db:
            for job in self.store.list('job', db=db):
                wf, wr = self.store.get('workflow', job['workflow_id'], db=db)
                if wf['device_workflow_id'] != external or job['status'] != 'SCHEDULED':
                    continue
                _, jr = self.store.get('job', job['job_id'], db=db)
                job['status'], wf['status'] = 'CANCELLED', 'CANCELLED'
                wf['evidence'].append({'kind': 'public_cancellation', 'invocation_id': row.invocation_id})
                self.store.put('job', job['job_id'], job, expected_revision=jr, db=db)
                self.store.put('workflow', wf['workflow_id'], wf, expected_revision=wr, db=db)

    def _verify_job(self, job):
        wf = self.store.get('workflow', job['workflow_id'])[0]
        external = wf['device_workflow_id']
        if not external:
            return PostconditionResult(VerificationStatus.UNVERIFIED, reason='NATIVE_REGISTRATION_UNCERTAIN')
        receipt = self._read('get_workflow_status', {'workflow_id': external})
        data = receipt.response.get('data') if receipt else None
        if not isinstance(data, dict) or data.get('workflow_id') != external:
            return PostconditionResult(VerificationStatus.UNVERIFIED, reason='NATIVE_STATUS_UNAVAILABLE_OR_IDENTITY_MISMATCH')
        state = data.get('status')
        evidence = [{'kind': 'public_native_status', 'invocation_id': receipt.invocation_id, 'status': state}]
        if state != 'completed':
            return PostconditionResult(VerificationStatus.UNVERIFIED, tuple(evidence), 'NATIVE_NOT_COMPLETED')
        payload = job['payload']
        if payload['uncovered'] or not payload['conditions']:
            return PostconditionResult(VerificationStatus.UNVERIFIED, tuple(evidence), 'INTENTION_POSTCONDITIONS_UNCOVERED')
        states = {}
        for device in sorted({c['device_id'] for c in payload['conditions']}):
            row = self._read('get_device_structure', {'device_id': device})
            if row is None or row.response.get('data', {}).get('device_id') != device:
                return PostconditionResult(VerificationStatus.UNVERIFIED, tuple(evidence), 'PUBLIC_READBACK_UNAVAILABLE')
            states[device] = row.response['data']
            evidence.append({'kind': 'readback', 'device_id': device, 'invocation_id': row.invocation_id})
            for path in payload.get('pending_paths', {}).get(device, []):
                found, remaining = _get_path(states[device], path)
                if found and type(remaining) in (int, float) and remaining > 0:
                    return PostconditionResult(VerificationStatus.UNVERIFIED, tuple(evidence),
                                               'PUBLIC_TRANSITION_PENDING')
        observed = self._clock()
        results = []
        for condition in payload['conditions']:
            device = condition['device_id']
            function = FunctionContract('workflow_intention', 'workflow_intention', postconditions=(
                {k: v for k, v in condition.items() if k != 'device_id'},))
            result = check_postconditions(function, {}, {}, states[device])
            results.append(result.status)
            evidence.append({'kind': 'intention_postcondition', 'device_id': device, **result.as_dict()})
        status = (VerificationStatus.VERIFIED_FAILURE if VerificationStatus.VERIFIED_FAILURE in results else
                  VerificationStatus.UNVERIFIED if VerificationStatus.UNVERIFIED in results else
                  VerificationStatus.VERIFIED_SUCCESS)
        return PostconditionResult(status, tuple(evidence), observed_at=observed)

    def supervise(self, *, phase):
        if self.task is None or self._supervising:
            return []
        pending = any(j['status'] == 'SCHEDULED' for j in self.store.list('job'))
        if not pending:
            return []
        self._supervising = True
        try:
            now = self._clock()
            outcomes = self.scheduler.tick(now) if now is not None else []
            self._event('public_clock_supervision', phase=phase, now=now, outcomes=outcomes)
            self.executor.save_audit()
            return outcomes
        finally:
            self._supervising = False

    def finish(self, *, supervise=True):
        if self.task is None:
            self.flush()
            return
        if supervise:
            self.supervise(phase='agent_return')
        task = self.manager.get(self.task.task_id, self.task.user_id)
        if task.status in (TaskStatus.ACTIVE, TaskStatus.EXECUTING, TaskStatus.VERIFYING):
            self.manager.transition(task.task_id, task.user_id, TaskStatus.WAITING,
                expected_version=task.version, evidence={'kind': 'conversation_returned',
                'reason': 'PUBLIC_USER_GOAL_POSTCONDITIONS_UNCOVERED'})
        self.flush()
