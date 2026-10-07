import unittest

from smarthome_agent_rl.execution.retry import RetryBudget
from smarthome_agent_rl.execution.trace import ToolTrace


class RetryBudgetTests(unittest.TestCase):
    def test_transient_read_is_bounded_and_traced(self):
        calls = []
        def dispatch(tool, args):
            calls.append(tool)
            if len(calls) == 1:
                raise TimeoutError("temporary")
            return {"status": {"code": 200}, "data": {"ok": True}}
        budget = RetryBudget(read_attempts=1)
        trace = ToolTrace(dispatch)
        row = trace.call_read("get_state", {}, retry_budget=budget)
        self.assertEqual(row.response["status"]["code"], 200)
        self.assertEqual(len(trace.invocations), 2)
        self.assertEqual(budget.snapshot()["used"]["read"], 1)

    def test_exhausted_read_budget_returns_failure_without_loop(self):
        trace = ToolTrace(lambda *args: (_ for _ in ()).throw(TimeoutError("down")))
        budget = RetryBudget(read_attempts=1)
        row = trace.call_read("get_state", {}, retry_budget=budget)
        self.assertEqual(len(trace.invocations), 2)
        self.assertEqual(row.error["type"], "TimeoutError")
        self.assertFalse(budget.acquire("read", reason="second retry"))

    def test_mutation_call_has_no_retry_path(self):
        calls = []
        trace = ToolTrace(lambda *args: calls.append(args) or (_ for _ in ()).throw(TimeoutError("unknown")))
        budget = RetryBudget(read_attempts=3)
        row = trace.call("execute_command", {})
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.snapshot()["used"]["read"], 0)
        self.assertEqual(row.error["type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
