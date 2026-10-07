"""Offline runtime acceptance; never starts a model or simulator service."""
import argparse
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = ("tests.test_runtime_boundary", "tests.test_restart_recovery",
           "tests.test_retry_budget", "tests.test_context_policy",
           "tests.test_scheduler_runtime", "tests.test_execution_runtime",
           "tests.test_task_runtime")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    suite = unittest.TestSuite()
    for module in MODULES:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {"schema": "runtime-acceptance-v1", "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "passed": result.wasSuccessful(),
              "wall_seconds": time.monotonic() - started, "model_requests": 0,
              "simulator_requests": 0, "gpu_seconds": 0,
              "benchmark_reference": "docs/data/full-benchmark.json",
              "scope": "Runtime fault/state acceptance only; no new score claim",
              "modules": list(MODULES)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
