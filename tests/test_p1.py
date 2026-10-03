import copy
import json
from pathlib import Path
import tempfile
import unittest
from smarthome_agent_rl.p1 import B0, H1, schedule, freeze, digest, paired_summary

class P1Tests(unittest.TestCase):
    def test_schedule_complete_and_counterbalanced(self):
        tasks = [{"task_id": str(i)} for i in range(16)]
        first = schedule(tasks, [B0, H1])
        second = schedule(tasks, [B0, H1], "reverse", 1)
        expected = {(str(i), h) for i in range(16) for h in (B0, H1)}
        for rows in (first, second):
            self.assertEqual(len(rows), 32)
            self.assertEqual({(r["task_id"], r["harness"]) for r in rows}, expected)
        self.assertEqual(first[0]["harness"], B0)
        self.assertEqual(second[0]["harness"], H1)
        self.assertEqual(second[0]["task_id"], "15")
        self.assertEqual(first, schedule(tasks, [B0, H1]))

    def test_freeze_preserves_entire_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = [{"task_id": str(i), "family": str(i)} for i in range(4)]
            request = {"messages": [{"role": "user", "content": "exact"}],
                       "response_format": {"type": "json_schema"}, "seed": 42, "max_tokens": 2048}
            (root / "verification.json").write_text(json.dumps({"artifact_checks_passed": True}))
            (root / "tasks.json").write_text(json.dumps(tasks))
            (root / "results.json").write_text(json.dumps([{"task_id": str(i), "path": str(i)} for i in range(4)]))
            for i in range(4):
                (root / str(i) / "episode").mkdir(parents=True)
                (root / str(i) / "episode/model_calls.json").write_text(json.dumps([{"request": request}]))
            frozen = freeze(root)
            self.assertEqual(frozen["purpose"], "diagnostic_only")
            for entry in frozen["entries"]:
                self.assertEqual(entry["request"], request)
                self.assertEqual(entry["backend_request"], {**request, "return_token_ids": True})
                self.assertEqual(entry["backend_sha256"], digest(entry["backend_request"]))
            changed = copy.deepcopy(frozen["entries"][0]["backend_request"])
            changed["seed"] = 43
            self.assertNotEqual(digest(changed), frozen["entries"][0]["backend_sha256"])

    def test_repeats_and_diagnostic_isolation(self):
        metrics = {m: 1 for m in ("goal_success", "model_calls", "tool_calls", "invalid_actions",
            "prompt_tokens", "completion_tokens", "total_tokens", "duration_seconds", "model_latency_seconds", "retrieval_calls", "retrieval_tokens", "retrieval_latency_seconds", "reward")}
        rounds = [[{"task_id": str(i), "harness": h, "success": int(h == H1 and n != 1), **metrics}
                   for i in range(16) for h in (B0, H1)] for n in range(3)]
        result = paired_summary(rounds)
        self.assertEqual(len(result), 16)
        self.assertEqual(result["0"]["H1_minus_B0"]["success"]["values"], [1, 0, 1])
        self.assertTrue(result["0"]["success_flips"][H1])
        rounds[0].append({"task_id": "diagnostic", "harness": B0, **metrics, "success": 1})
        with self.assertRaises(AssertionError):
            paired_summary(rounds)

if __name__ == "__main__":
    unittest.main()
