"""Long-lived workflow states are separate from invocation and task states."""
import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from uuid import uuid4

from .mutation import PostconditionResult, VerificationStatus


class WorkflowStatus(str, Enum):
    CREATED = 'CREATED'
    SCHEDULED = 'SCHEDULED'
    ACTIVE = 'ACTIVE'
    VERIFIED_SUCCESS = 'VERIFIED_SUCCESS'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


TRANSITIONS = {
    WorkflowStatus.CREATED: {WorkflowStatus.SCHEDULED, WorkflowStatus.CANCELLED, WorkflowStatus.FAILED},
    WorkflowStatus.SCHEDULED: {WorkflowStatus.ACTIVE, WorkflowStatus.CANCELLED, WorkflowStatus.FAILED},
    WorkflowStatus.ACTIVE: {WorkflowStatus.VERIFIED_SUCCESS, WorkflowStatus.CANCELLED, WorkflowStatus.FAILED},
    WorkflowStatus.VERIFIED_SUCCESS: set(), WorkflowStatus.FAILED: set(), WorkflowStatus.CANCELLED: set(),
}


@dataclass
class Workflow:
    task_id: str
    task_version: int
    mode: str
    workflow_id: str = field(default_factory=lambda: uuid4().hex)
    status: WorkflowStatus = WorkflowStatus.CREATED
    device_workflow_id: str | None = None
    evidence: list[dict] = field(default_factory=list)

    def transition(self, status: WorkflowStatus, evidence: dict):
        status = WorkflowStatus(status)
        if status not in TRANSITIONS[self.status]:
            raise ValueError(f'Invalid workflow transition: {self.status} -> {status}')
        if status == WorkflowStatus.VERIFIED_SUCCESS:
            raise ValueError('Use verify() with public postcondition evidence')
        self.status = status
        self.evidence.append(copy.deepcopy(evidence))

    def verify(self, result: PostconditionResult):
        if self.status != WorkflowStatus.ACTIVE:
            raise ValueError('Only active workflows can be verified')
        if result.status == VerificationStatus.UNVERIFIED:
            self.evidence.append(result.as_dict())
            return
        if not result.evidence:
            raise ValueError('Verification requires public state evidence')
        self.status = (WorkflowStatus.VERIFIED_SUCCESS if result.status == VerificationStatus.VERIFIED_SUCCESS
                       else WorkflowStatus.FAILED)
        self.evidence.append(result.as_dict())

    def as_dict(self):
        return copy.deepcopy(asdict(self))
