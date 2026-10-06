"""Persistent user goals and task versions, independent of chat sessions."""
import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from uuid import uuid4

from ..device_contract import FunctionContract
from .mutation import check_postconditions, VerificationStatus
from .store import RevisionConflict, RuntimeStore


class TaskStatus(str, Enum):
    ACTIVE = 'ACTIVE'
    WAITING = 'WAITING'
    EXECUTING = 'EXECUTING'
    VERIFYING = 'VERIFYING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


TERMINAL = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
TRANSITIONS = {
    TaskStatus.ACTIVE: {TaskStatus.WAITING, TaskStatus.EXECUTING, TaskStatus.VERIFYING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.WAITING: {TaskStatus.ACTIVE, TaskStatus.EXECUTING, TaskStatus.VERIFYING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.EXECUTING: {TaskStatus.WAITING, TaskStatus.VERIFYING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.VERIFYING: {TaskStatus.WAITING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(), TaskStatus.FAILED: set(), TaskStatus.CANCELLED: set(),
}


@dataclass
class TaskSession:
    user_id: str
    goal: str
    source_conversation: str
    task_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.ACTIVE
    constraints: dict = field(default_factory=dict)
    relevant_devices: list[str] = field(default_factory=list)
    related_workflows: list[str] = field(default_factory=list)
    expected_postconditions: list[dict] = field(default_factory=list)
    trigger: dict | None = None
    version: int = 1
    evidence: list[dict] = field(default_factory=list)

    def as_dict(self):
        return copy.deepcopy(asdict(self))


class TaskManager:
    def __init__(self, store: RuntimeStore):
        self.store = store

    def create(self, user_id, goal, source_conversation, **options):
        if not user_id or not goal or not source_conversation:
            raise ValueError('Task requires user, goal and source conversation')
        if set(options) - {'constraints', 'relevant_devices', 'expected_postconditions', 'trigger'}:
            raise ValueError('Unsupported task fields')
        task = TaskSession(user_id, goal, source_conversation, **copy.deepcopy(options))
        self.store.put('task', task.task_id, task.as_dict())
        return task

    def _load(self, task_id, user_id, db):
        data, revision = self.store.get('task', task_id, db=db)
        if data['user_id'] != user_id:
            raise PermissionError('Task belongs to another user')
        data['status'] = TaskStatus(data['status'])
        return TaskSession(**data), revision

    def get(self, task_id, user_id):
        with self.store.transaction() as db:
            return self._load(task_id, user_id, db)[0]

    def active(self, user_id):
        return [TaskSession(**row) for row in self.store.list('task')
                if row['user_id'] == user_id and row['status'] not in TERMINAL]

    def _invalidate_plans(self, task, db):
        # A native schedule survives this process. It must be explicitly
        # cancelled at the device before accepting a contradictory revision.
        for id in task.related_workflows:
            workflow, revision = self.store.get('workflow', id, db=db)
            if workflow['status'] in ('VERIFIED_SUCCESS', 'FAILED', 'CANCELLED'):
                continue
            if workflow['mode'] == 'device_native':
                raise RevisionConflict('Cancel the device-native schedule and confirm its receipt before revision')
            workflow['status'] = 'CANCELLED'
            workflow['evidence'].append({'reason': 'task_revision_or_cancellation', 'task_version': task.version})
            self.store.put('workflow', id, workflow, expected_revision=revision, db=db)
        for job in self.store.list('job', db=db):
            if job['task_id'] != task.task_id or job['status'] not in ('SCHEDULED', 'CLAIMED', 'UNKNOWN'):
                continue
            if job['status'] in ('CLAIMED', 'UNKNOWN'):
                raise RevisionConflict('An in-flight job must be reconciled before task revision')
            _, revision = self.store.get('job', job['job_id'], db=db)
            job['status'] = 'CANCELLED'
            self.store.put('job', job['job_id'], job, expected_revision=revision, db=db)

    def revise(self, task_id, user_id, *, expected_version, source_conversation, **changes):
        if not changes or set(changes) - {'goal', 'constraints', 'relevant_devices', 'expected_postconditions', 'trigger'}:
            raise ValueError('Revision requires supported task fields')
        with self.store.transaction() as db:
            task, revision = self._load(task_id, user_id, db)
            if task.version != expected_version or task.status in TERMINAL:
                raise RevisionConflict('Task is terminal or version changed')
            self._invalidate_plans(task, db)
            task.evidence.append({'kind': 'revision', 'old_version': task.version,
                                  'source_conversation': source_conversation,
                                  'previous_goal': task.goal})
            for name, value in changes.items():
                setattr(task, name, copy.deepcopy(value))
            task.version += 1
            task.status = TaskStatus.ACTIVE
            self.store.put('task', task_id, task.as_dict(), expected_revision=revision, db=db)
            return task

    def transition(self, task_id, user_id, status, *, expected_version, evidence):
        status = TaskStatus(status)
        if status == TaskStatus.COMPLETED:
            raise ValueError('Task completion requires verify()')
        with self.store.transaction() as db:
            task, revision = self._load(task_id, user_id, db)
            if task.version != expected_version:
                raise RevisionConflict('Task version changed')
            if status not in TRANSITIONS[TaskStatus(task.status)]:
                raise ValueError(f'Invalid task transition {task.status} -> {status}')
            if status == TaskStatus.CANCELLED:
                self._invalidate_plans(task, db)
            task.status = status
            task.evidence.append(copy.deepcopy(evidence))
            self.store.put('task', task_id, task.as_dict(), expected_revision=revision, db=db)
            return task

    def verify(self, task_id, user_id, *, expected_version, public_states):
        with self.store.transaction() as db:
            task, revision = self._load(task_id, user_id, db)
            if task.version != expected_version:
                raise RevisionConflict('Task version changed')
            if task.status == TaskStatus.COMPLETED:
                return task
            if task.status in TERMINAL:
                raise ValueError('Cannot verify a terminal task')
            decisions = []
            for condition in task.expected_postconditions:
                device = condition['device_id']
                function = FunctionContract('task_goal', 'task_goal', postconditions=(
                    {key: value for key, value in condition.items() if key != 'device_id'},))
                result = check_postconditions(function, {}, {}, public_states.get(device, {}))
                decisions.append({'device_id': device, **result.as_dict()})
            pending = any(job['task_id'] == task_id and job['task_version'] == task.version
                          and job['status'] in ('REGISTERING', 'SCHEDULED', 'CLAIMED', 'UNKNOWN')
                          for job in self.store.list('job', db=db))
            all_verified = bool(decisions) and not pending and all(
                row['status'] == VerificationStatus.VERIFIED_SUCCESS for row in decisions)
            task.status = TaskStatus.COMPLETED if all_verified else TaskStatus.WAITING
            task.evidence.append({'kind': 'task_postconditions', 'version': task.version, 'decisions': decisions})
            self.store.put('task', task_id, task.as_dict(), expected_revision=revision, db=db)
            return task
