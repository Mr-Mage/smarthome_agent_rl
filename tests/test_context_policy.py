import unittest

from smarthome_agent_rl.execution.context_policy import ContextPolicy


class ContextPolicyTests(unittest.TestCase):
    def task(self):
        return {"task_id": "t1", "version": 2, "goal": "turn off the fan",
                "status": "WAITING", "constraints": {"room": "bedroom"},
                "source_conversation": "chat1"}

    def test_current_facts_and_errors_have_provenance_and_stale_is_explicit(self):
        result = ContextPolicy(max_characters=2000).render(task=self.task(), facts=[
            {"path": "power", "value": False, "source": "obs-2"},
            {"path": "power", "value": True, "source": "obs-1", "stale": True}],
            receipts=[{"invocation_id": "i1", "status": "UNKNOWN"}],
            errors=[{"invocation_id": "i1", "type": "TimeoutError"}])
        self.assertEqual(result["task"]["goal"], "turn off the fan")
        self.assertEqual(result["current_facts"][0]["source"], "obs-2")
        self.assertEqual(result["stale_facts"][0]["source"], "obs-1")
        self.assertEqual(result["errors"][0]["type"], "TimeoutError")
        self.assertLessEqual(result["characters"], 2000)

    def test_budget_omits_history_before_mandatory_task(self):
        result = ContextPolicy(max_characters=500).render(task=self.task(),
            facts=[{"path": str(i), "value": "x" * 40, "stale": True} for i in range(20)])
        self.assertEqual(result["task"]["task_id"], "t1")
        self.assertGreater(result["omitted"]["stale_facts"], 0)


if __name__ == "__main__":
    unittest.main()
