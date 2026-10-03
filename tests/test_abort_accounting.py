"""Regression: pristine ReAct aborts the third malformed turn before feedback."""
import unittest

from src.agents.strategies.react_agent import ReActAgent, ReActConfig
from src.agents.strategies.base import AgentExecutionError
from smarthome_agent_rl.accounting import pending_rejection


class BadOutput:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, response_format=None):
        self.calls += 1
        return "not valid JSON"


class AbortAccountingTests(unittest.TestCase):
    def test_original_third_rejection_has_no_feedback(self):
        provider, events = BadOutput(), []
        agent = ReActAgent(provider, config=ReActConfig(max_steps=20, show_assistant_raw=True,
            trace_fn=lambda event, payload: events.append({"event": event, "payload": payload})))
        with self.assertRaises(AgentExecutionError) as caught:
            agent.run("test")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(sum(e["event"] == "observation" for e in events), 2)
        rejected = pending_rejection({"type": type(caught.exception).__name__, "message": str(caught.exception)}, events)
        self.assertEqual(rejected["status"]["code"], 400)
        self.assertFalse(rejected["agent_feedback_emitted"])

    def test_infrastructure_is_not_an_agent_rejection(self):
        events = [{"event": "consecutive_failure", "payload": "3:invalid structured output"}]
        self.assertIsNone(pending_rejection({"type": "TimeoutError", "message": "network failure"}, events))


if __name__ == "__main__":
    unittest.main()
