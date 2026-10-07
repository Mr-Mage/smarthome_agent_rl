"""Deterministic runtime context selection; no semantic inference or summaries."""
import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPolicy:
    max_characters: int = 12000
    max_receipts: int = 32
    max_errors: int = 32

    def __post_init__(self):
        if type(self.max_characters) is not int or self.max_characters < 256:
            raise ValueError("context budget must be at least 256 characters")
        if type(self.max_receipts) is not int or self.max_receipts < 0:
            raise ValueError("max_receipts must be nonnegative")
        if type(self.max_errors) is not int or self.max_errors < 0:
            raise ValueError("max_errors must be nonnegative")

    def render(self, *, task, facts=(), receipts=(), errors=()):
        """Build a bounded envelope with stable priority and source fields.

        Goal/status/constraints are mandatory. Current facts are preferred over
        stale history; receipts and errors are newest-first but never rewritten.
        If the budget is too small, lower-priority history is omitted with a
        count rather than silently presented as current state.
        """
        if not isinstance(task, dict) or not task.get("task_id"):
            raise ValueError("context requires a persisted task")
        facts = [copy.deepcopy(row) for row in facts]
        current = [row for row in facts if not row.get("stale", False)]
        stale = [row for row in facts if row.get("stale", False)]
        envelope = {
            "schema": "runtime-context-v1",
            "task": {"task_id": task["task_id"], "version": task.get("version", 1),
                     "goal": task.get("goal", ""), "status": task.get("status"),
                     "constraints": copy.deepcopy(task.get("constraints", {})),
                     "source_conversation": task.get("source_conversation")},
            "current_facts": current,
            "receipts": [copy.deepcopy(x) for x in list(receipts)[-self.max_receipts:]],
            "errors": [copy.deepcopy(x) for x in list(errors)[-self.max_errors:]],
            "stale_facts": stale,
            "omitted": {"current_facts": 0, "stale_facts": 0, "receipts": 0, "errors": 0},
            "rule": "Current facts are observations; stale facts are history and require a fresh public read.",
        }
        # Remove lowest-priority rows until the serialized envelope fits.
        def size():
            import json
            return len(json.dumps(envelope, ensure_ascii=False, sort_keys=True))
        for key in ("stale_facts", "receipts", "errors", "current_facts"):
            rows = envelope[key]
            while rows and size() > self.max_characters:
                rows.pop(0)
                envelope["omitted"][key] += 1
        if size() > self.max_characters:
            raise ValueError("context budget cannot retain mandatory task envelope")
        envelope["characters"] = size()
        return envelope
