"""Observation-only accounting for original ReAct aborts before tool feedback."""


def pending_rejection(error, events):
    """Recognize a traced rejection; never turn transport/provider failure into invalid action."""
    if not error or error.get("type") != "AgentExecutionError" or not events:
        return None
    last = events[-1]
    if last.get("event") != "consecutive_failure" or "consecutive failures" not in error.get("message", ""):
        return None
    return {"status": {"code": 400}, "error": last["payload"],
            "source": "upstream_rejection_before_feedback", "agent_feedback_emitted": False}
