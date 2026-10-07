"""Single construction boundary for the durable task runtime.

The benchmark runner remains a separate adapter.  This object owns the
durable store, task manager, trace and scheduler wiring so callers do not
silently construct a second state machine with different policies.
"""
from dataclasses import dataclass
from pathlib import Path

from .scheduler import Scheduler
from .store import RuntimeStore
from .tasks import TaskManager
from .trace import ToolTrace


@dataclass(frozen=True)
class RuntimePolicy:
    schema: str = "task-runtime-boundary-v1"
    query_budget: int = 40
    automatic_mutation_retry: bool = False
    unknown_mutation_replay: bool = False


class RuntimeBoundary:
    """The only supported composition root for persistent execution."""

    def __init__(self, *, store, tasks, trace, scheduler, policy):
        self.store = store
        self.tasks = tasks
        self.trace = trace
        self.scheduler = scheduler
        self.policy = policy

    @classmethod
    def open(cls, path, *, dispatch, execute_action, verify_job, wake_agent,
             read_task_states=None, policy=None):
        policy = policy or RuntimePolicy()
        if policy.automatic_mutation_retry or policy.unknown_mutation_replay:
            raise ValueError("Mutation retries/replays require explicit reconciliation")
        store = RuntimeStore(Path(path))
        store.recover_inflight()
        tasks = TaskManager(store)
        trace = ToolTrace(dispatch, sink=store.trace_sink)
        scheduler = Scheduler(tasks, trace, execute_action=execute_action,
                               verify_job=verify_job, wake_agent=wake_agent,
                               read_task_states=read_task_states)
        return cls(store=store, tasks=tasks, trace=trace,
                   scheduler=scheduler, policy=policy)

    def health(self):
        """Return an operator-facing snapshot without changing runtime state."""
        return {
            "schema": self.policy.schema,
            "store": str(self.store.path),
            "tasks": len(self.store.list("task")),
            "jobs": len(self.store.list("job")),
            "workflows": len(self.store.list("workflow")),
            "trace_invocations": len(self.store.list("trace")),
            "recovery_events": len(self.store.list("recovery")),
            "policy": {
                "query_budget": self.policy.query_budget,
                "automatic_mutation_retry": self.policy.automatic_mutation_retry,
                "unknown_mutation_replay": self.policy.unknown_mutation_replay,
            },
        }
