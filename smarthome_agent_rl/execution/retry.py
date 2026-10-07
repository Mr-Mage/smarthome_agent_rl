"""Bounded retry accounting for the runtime boundary.

Only reads may be retried automatically.  A mutation timeout is an UNKNOWN
outcome and must use reconciliation; changing its arguments is a new plan.
"""
from dataclasses import dataclass, field


TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
TRANSIENT_ERRORS = frozenset({"TimeoutError", "ConnectionError", "ConnectError",
                              "ReadTimeout", "ConnectTimeout", "PoolTimeout"})


def transient_read_failure(invocation):
    error = invocation.error or {}
    if error.get("type") in TRANSIENT_ERRORS:
        return True
    return error.get("type") == "tool_status" and error.get("code") in TRANSIENT_STATUS


@dataclass
class RetryBudget:
    """Separate counters prevent nested transport/replan loops."""
    read_attempts: int = 1
    correction_attempts: int = 2
    reconciliation_attempts: int = 1
    used: dict = field(default_factory=lambda: {
        "read": 0, "correction": 0, "reconciliation": 0})
    reasons: list = field(default_factory=list)

    def __post_init__(self):
        for value in (self.read_attempts, self.correction_attempts,
                      self.reconciliation_attempts):
            if type(value) is not int or value < 0:
                raise ValueError("retry budgets must be nonnegative integers")

    def acquire(self, kind, *, reason):
        limits = {"read": self.read_attempts,
                  "correction": self.correction_attempts,
                  "reconciliation": self.reconciliation_attempts}
        if kind not in limits:
            raise ValueError(f"unknown retry budget: {kind}")
        if self.used[kind] >= limits[kind]:
            self.reasons.append({"kind": kind, "reason": reason, "allowed": False})
            return False
        self.used[kind] += 1
        self.reasons.append({"kind": kind, "reason": reason, "allowed": True,
                             "attempt": self.used[kind]})
        return True

    def snapshot(self):
        return {"limits": {"read": self.read_attempts,
                            "correction": self.correction_attempts,
                            "reconciliation": self.reconciliation_attempts},
                "used": dict(self.used), "reasons": list(self.reasons)}
