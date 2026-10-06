"""Durable scheduler for delegated execution, fixed actions and agent wake-up.

Atomic claims prevent two workers dispatching a job. A claimed job left after a
crash is UNKNOWN and never blindly replays its mutation. All times are supplied
by the adapter (including simulator virtual time); no sleeps or clock changes.
"""
import copy
import math
from uuid import uuid4

from .mutation import PostconditionResult, VerificationStatus
from .store import RevisionConflict
from .tasks import TaskManager, TaskStatus, TERMINAL
from .trace import ToolTrace
from .workflow import Workflow, WorkflowStatus
from .conflicts import resource_claims as normalize_claims


class Scheduler:
    def __init__(self, manager: TaskManager, trace: ToolTrace, *, execute_action,
                 verify_job, wake_agent, read_task_states=None):
        self.manager, self.store, self.trace = manager, manager.store, trace
        self.execute_action, self.verify_job, self.wake_agent = execute_action, verify_job, wake_agent
        self.read_task_states = read_task_states

    def schedule(self, task_id, user_id, *, expected_version, target_time, payload=None,
                 native_call=None, wake_agent=False, tolerance=0, resource_claims=()):
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in (target_time, tolerance)) or tolerance < 0:
            raise ValueError('Schedule times must be finite; tolerance nonnegative')
        if native_call and wake_agent:
            raise ValueError('A native fixed plan cannot also be an agent wake-up')
        if native_call and (not isinstance(native_call.get('tool'), str) or not isinstance(native_call.get('args'), dict)):
            raise ValueError('Native schedule requires a tool and argument object')
        mode = 'agent_wakeup' if wake_agent else 'device_native' if native_call else 'harness_action'
        workflow = Workflow(task_id, expected_version, mode)
        job_id = uuid4().hex
        job = {'job_id': job_id, 'workflow_id': workflow.workflow_id, 'task_id': task_id,
               'user_id': user_id, 'task_version': expected_version, 'target_time': target_time,
               'tolerance': tolerance, 'mode': mode, 'payload': copy.deepcopy(payload or {}),
               'status': 'REGISTERING' if native_call else 'SCHEDULED', 'evidence': []}
        with self.store.transaction() as db:
            task, revision = self.manager._load(task_id, user_id, db)
            if task.version != expected_version or task.status in TERMINAL:
                raise RevisionConflict('Task changed before scheduling')
            # Intents are supplied by the planner. Do not infer tomorrow's
            # command targets from today's state or from the whole task goal.
            job['resource_claims'] = normalize_claims(resource_claims,
                namespace=task.constraints.get('resource_namespace','home'),
                target_time=target_time,tolerance=tolerance)
            conflicts, uncovered = self.manager._plan_conflicts(job,db)
            job['conflict_ids'] = [row['conflict_id'] for row in conflicts]
            job['conflict_uncovered'] = uncovered
            workflow.transition(WorkflowStatus.SCHEDULED, {'target_time': target_time, 'mode': mode})
            if native_call:
                # Native registration starts outside the transaction; durable
                # REGISTERING state retains the ambiguity if the process dies.
                workflow.status = WorkflowStatus.CREATED
            if conflicts or uncovered:
                workflow.evidence.append({'kind':'resource_conflict_detection',
                    'conflict_ids':job['conflict_ids'],'uncovered':uncovered,'policy':'report only'})
            self.store.put('workflow', workflow.workflow_id, workflow.as_dict(), db=db)
            self.store.put('job', job_id, job, db=db)
            task.related_workflows.append(workflow.workflow_id)
            if conflicts or uncovered:
                task.evidence.append({'kind':'resource_conflict_detection','job_id':job_id,
                                      'conflict_ids':job['conflict_ids'],'uncovered':uncovered})
            task.status = TaskStatus.WAITING
            self.store.put('task', task_id, task.as_dict(), expected_revision=revision, db=db)
        if native_call:
            receipt = self.trace.call(native_call['tool'], native_call['args'], task_id=task_id,
                                      workflow_id=workflow.workflow_id, parent_step=job_id)
            response = receipt.response or {}
            registered = not receipt.error and response.get('status', {}).get('code') == 200
            external_id = response.get('data', {}).get('workflow_id')
            with self.store.transaction() as db:
                wf, revision = self.store.get('workflow', workflow.workflow_id, db=db)
                job, jr = self.store.get('job', job_id, db=db)
                wf['device_workflow_id'] = external_id
                wf['evidence'].append({'kind': 'registration', 'invocation_id': receipt.invocation_id,
                                       'acknowledged': registered, 'external_id': external_id})
                # Ambiguous registration may have created a native schedule;
                # monitor at due time, but never issue registration again.
                wf['status'] = 'SCHEDULED' if registered else 'CREATED'
                job['status'] = 'SCHEDULED'
                job['registration_uncertain'] = not registered or external_id is None
                self.store.put('workflow', workflow.workflow_id, wf, expected_revision=revision, db=db)
                self.store.put('job', job_id, job, expected_revision=jr, db=db)
        return copy.deepcopy(job)

    def _claim(self, job_id, now):
        with self.store.transaction() as db:
            job, revision = self.store.get('job', job_id, db=db)
            if job['status'] != 'SCHEDULED' or job['target_time'] > now:
                return None
            task, tr = self.manager._load(job['task_id'], job['user_id'], db)
            if task.version != job['task_version'] or task.status in TERMINAL:
                job['status'] = 'CANCELLED'
                self.store.put('job', job_id, job, expected_revision=revision, db=db)
                return None
            job['status'] = 'CLAIMED'
            job['claimed_at'] = now
            self.store.put('job', job_id, job, expected_revision=revision, db=db)
            wf, wr = self.store.get('workflow', job['workflow_id'], db=db)
            wf['status'] = 'ACTIVE'
            wf['evidence'].append({'kind': 'due_job_claim', 'job_id': job_id, 'time': now})
            self.store.put('workflow', job['workflow_id'], wf, expected_revision=wr, db=db)
            task.status = TaskStatus.EXECUTING
            self.store.put('task', task.task_id, task.as_dict(), expected_revision=tr, db=db)
            return job, task

    def _finish(self, job, result: PostconditionResult):
        with self.store.transaction() as db:
            current, revision = self.store.get('job', job['job_id'], db=db)
            if current['status'] not in ('CLAIMED', 'UNKNOWN'):
                raise RevisionConflict('Job no longer owned by this execution')
            current['status'] = ('DONE' if result.status == VerificationStatus.VERIFIED_SUCCESS else
                                 'FAILED' if result.status == VerificationStatus.VERIFIED_FAILURE else 'UNKNOWN')
            current['evidence'].append(result.as_dict())
            wf, wr = self.store.get('workflow', job['workflow_id'], db=db)
            workflow = Workflow(**wf)
            workflow.status = WorkflowStatus(workflow.status)
            workflow.verify(result)
            self.store.put('workflow', workflow.workflow_id, workflow.as_dict(), expected_revision=wr, db=db)
            self.store.put('job', job['job_id'], current, expected_revision=revision, db=db)
            task, tr = self.manager._load(job['task_id'], job['user_id'], db)
            if task.version == job['task_version'] and task.status not in TERMINAL:
                task.status = TaskStatus.VERIFYING
                self.store.put('task', task.task_id, task.as_dict(), expected_revision=tr, db=db)
        self._verify_task_when_ready(job)

    def _verify_task_when_ready(self, job):
        if self.read_task_states is None:
            return
        task = self.manager.get(job['task_id'], job['user_id'])
        if task.version != job['task_version'] or task.status in TERMINAL:
            return
        # Do not declare timed task success while future jobs remain pending.
        pending = any(row['task_id'] == task.task_id and row['task_version'] == task.version
                      and row['status'] in ('SCHEDULED', 'REGISTERING', 'CLAIMED', 'UNKNOWN')
                      for row in self.store.list('job'))
        if not pending:
            self.manager.verify(task.task_id, task.user_id, expected_version=task.version,
                                public_states=self.read_task_states(task))

    def tick(self, now):
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('Scheduler requires finite adapter time')
        outcomes = []
        for row in sorted(self.store.list('job'), key=lambda row: (row['target_time'], row['job_id'])):
            claimed = self._claim(row['job_id'], now)
            if claimed is None:
                continue
            job, task = claimed
            try:
                if now > job['target_time'] + job['tolerance']:
                    result = PostconditionResult(VerificationStatus.UNVERIFIED,
                        reason='VERIFICATION_WINDOW_MISSED; late state cannot prove the timed goal')
                elif job['mode'] == 'device_native':
                    result = self.verify_job(job)
                elif job['mode'] == 'agent_wakeup':
                    result = self.wake_agent(task, job)
                else:
                    result = self.execute_action(task, job)
                if not isinstance(result, PostconditionResult):
                    raise ValueError('Scheduler callbacks must return postcondition evidence')
            except Exception as exc:
                result = PostconditionResult(VerificationStatus.UNVERIFIED, reason=f'{type(exc).__name__}: {exc}')
            self._finish(job, result)
            outcomes.append({'job_id': job['job_id'], **result.as_dict()})
        return outcomes

    def reconcile(self, job_id):
        """Recover a claimed/unknown job by read-back only, never dispatch again."""
        with self.store.transaction() as db:
            job, revision = self.store.get('job', job_id, db=db)
            if job['status'] not in ('CLAIMED', 'UNKNOWN', 'REGISTERING'):
                raise ValueError('Only uncertain jobs need reconciliation')
            task, _ = self.manager._load(job['task_id'], job['user_id'], db)
            if task.version != job['task_version']:
                raise RevisionConflict('Cannot reconcile an obsolete task version')
            if job['status'] == 'REGISTERING':
                raise RevisionConflict('Native registration is uncertain; inspect registration before due verification')
            job['status'] = 'UNKNOWN'
            self.store.put('job', job_id, job, expected_revision=revision, db=db)
        result = self.verify_job(job)
        self._finish(job, result)
        return result

    def cancel_workflow(self, workflow_id, user_id, *, cancel_native=None):
        with self.store.transaction() as db:
            wf, _ = self.store.get('workflow', workflow_id, db=db)
            task, _ = self.manager._load(wf['task_id'], user_id, db)
            active = [job for job in self.store.list('job', db=db) if job['workflow_id'] == workflow_id
                      and job['status'] not in ('DONE', 'FAILED', 'CANCELLED')]
            if any(job['status'] in ('CLAIMED', 'UNKNOWN', 'REGISTERING') for job in active):
                raise RevisionConflict('Reconcile an in-flight/uncertain job before cancellation')
            if wf['status'] in ('VERIFIED_SUCCESS', 'FAILED', 'CANCELLED'):
                return wf
        receipt = None
        if wf['mode'] == 'device_native':
            if not cancel_native or not wf['device_workflow_id']:
                raise RevisionConflict('Device-native cancellation requires confirmed external id')
            receipt = self.trace.call(cancel_native['tool'],
                {**cancel_native.get('args', {}), 'workflow_id': wf['device_workflow_id']},
                task_id=task.task_id, workflow_id=workflow_id)
            if receipt.error or (receipt.response or {}).get('status', {}).get('code') != 200:
                raise RevisionConflict('Device-native cancellation not acknowledged')
            state = (receipt.response or {}).get('data', {}).get('status')
            if state not in ('CANCELLED', 'cancelled', 'canceled'):
                verification = cancel_native.get('verification')
                if not verification:
                    raise RevisionConflict('Cancellation acknowledgement needs public cancellation-state evidence')
                readback = self.trace.call(verification['tool'],
                    {**verification.get('args', {}), 'workflow_id': wf['device_workflow_id']},
                    task_id=task.task_id, workflow_id=workflow_id)
                state = (readback.response or {}).get('data', {}).get('status')
                if readback.error or state not in ('CANCELLED', 'cancelled', 'canceled'):
                    raise RevisionConflict('Device-native cancellation state is unverified')
        with self.store.transaction() as db:
            wf, revision = self.store.get('workflow', workflow_id, db=db)
            # The atomic claim may have raced with the remote cancellation.
            active = [job for job in self.store.list('job', db=db) if job['workflow_id'] == workflow_id
                      and job['status'] not in ('DONE', 'FAILED', 'CANCELLED')]
            if any(job['status'] != 'SCHEDULED' for job in active):
                raise RevisionConflict('Job claimed during cancellation; reconcile its state')
            wf['status'] = 'CANCELLED'
            wf['evidence'].append({'kind': 'cancellation', 'invocation_id': receipt.invocation_id if receipt else None})
            self.store.put('workflow', workflow_id, wf, expected_revision=revision, db=db)
            for job in active:
                _, jr = self.store.get('job', job['job_id'], db=db)
                job['status'] = 'CANCELLED'
                self.store.put('job', job['job_id'], job, expected_revision=jr, db=db)
            return wf
