"""Reject resource/reset collisions before launching parallel GPU workers."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("parallel_runner", ROOT / "scripts/run_parallel_dev.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"shards": []}
        for i in range(2):
            config = {"simulator_url": f"http://127.0.0.1:{21080 + i}/api",
                "model_endpoint": f"http://127.0.0.1:{22000 + i}/v1",
                "gateway_url": f"http://127.0.0.1:{23000 + i}"}
            (self.root / f"config{i}.json").write_text(json.dumps(config))
            (self.root / f"task{i}.json").write_text(json.dumps([{"task_id": f"task{i}"}]))
            self.plan["shards"].append({"gpu": str(i + 2), "config": f"config{i}.json", "suite": f"task{i}.json"})
        patched = patch.object(runner, "ROOT", self.root)
        patched.start()
        self.addCleanup(patched.stop)

    def test_independent_resources(self):
        result = runner.validate_plan(self.plan)
        self.assertEqual(result["gpus"], ["2", "3"])
        self.assertEqual(len(result["ports"]), 6)
        self.assertEqual(result["tasks"], 2)

    def test_reject_shared_gpu(self):
        self.plan["shards"][1]["gpu"] = "2"
        with self.assertRaisesRegex(ValueError, "GPUs"):
            runner.validate_plan(self.plan)

    def test_reject_shared_simulator_port(self):
        file = self.root / "config1.json"
        config = json.loads(file.read_text())
        config["simulator_url"] = "http://127.0.0.1:21080/api"
        file.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "ports"):
            runner.validate_plan(self.plan)

    def test_reject_cross_shard_duplicate_task(self):
        (self.root / "task1.json").write_text(json.dumps([{"task_id": "task0"}]))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            runner.validate_plan(self.plan)


if __name__ == "__main__":
    unittest.main()
