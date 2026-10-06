from typing import Any, Protocol

PublicInput = dict | list[dict]


class AgentAdapter(Protocol):
    """Static completion and dynamic ReAct share a public-only invocation."""
    def invoke(self, public_input: PublicInput) -> Any: ...


class BenchmarkAdapter(Protocol):
    """Actor consumes public_input only; scoring is a separate invocation."""
    def task_ids(self) -> list[str]: ...
    def public_input(self, task_id: str) -> PublicInput: ...
    def score(self, task_id: str, prediction: Any) -> dict: ...
