"""Every dispatched tool invocation is evidence; queries have no lifecycle."""
import copy
from dataclasses import asdict, dataclass
import time
from typing import Any, Callable
from uuid import uuid4


@dataclass(frozen=True)
class ToolInvocation:
    invocation_id: str
    task_id: str | None
    workflow_id: str | None
    parent_step: str | None
    tool: str
    arguments: dict[str, Any]
    timestamp: float
    response: Any
    latency: float
    error: dict[str, Any] | None

    def as_dict(self):
        return copy.deepcopy(asdict(self))


class ToolTrace:
    def __init__(self, dispatch: Callable, *, sink=None, clock=time.time, timer=time.monotonic):
        self.dispatch, self.sink, self.clock, self.timer = dispatch, sink, clock, timer
        self.invocations: list[ToolInvocation] = []

    def call(self, tool, arguments, *, task_id=None, workflow_id=None, parent_step=None):
        snapshot = copy.deepcopy(arguments)
        timestamp, start = self.clock(), self.timer()
        response, error = None, None
        try:
            response = self.dispatch(tool, copy.deepcopy(snapshot))
            if isinstance(response, dict):
                error = response.get('error')
                status = response.get('status', {})
                if isinstance(status, dict) and status.get('code', 200) != 200 and not error:
                    error = {'type': 'tool_status', 'code': status.get('code')}
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        row = ToolInvocation(uuid4().hex, task_id, workflow_id, parent_step,
                             tool, snapshot, timestamp, copy.deepcopy(response),
                             self.timer() - start, copy.deepcopy(error))
        self.invocations.append(row)
        if self.sink:
            self.sink(row.as_dict())
        return row

    def call_read(self, tool, arguments, *, retry_budget=None, task_id=None,
                  workflow_id=None, parent_step=None):
        """Retry a read only when the failure is classified transient."""
        from .retry import transient_read_failure
        while True:
            row = self.call(tool, arguments, task_id=task_id,
                            workflow_id=workflow_id, parent_step=parent_step)
            if not row.error or retry_budget is None or not transient_read_failure(row):
                return row
            if not retry_budget.acquire("read", reason=row.error.get("type")):
                return row
