import tempfile
import unittest
from pathlib import Path

from smarthome_agent_rl.execution.mutation import PostconditionResult, VerificationStatus
from smarthome_agent_rl.execution.runtime import RuntimeBoundary, RuntimePolicy


def verified(*args, **kwargs):
    return PostconditionResult(VerificationStatus.VERIFIED_SUCCESS, ())


class RuntimeBoundaryTests(unittest.TestCase):
    def test_open_wires_one_store_trace_and_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = RuntimeBoundary.open(Path(tmp) / "runtime.sqlite3",
                dispatch=lambda tool, args: {"status": {"code": 200}, "data": {}},
                execute_action=lambda task, job: verified(),
                verify_job=lambda job: verified(), wake_agent=lambda task, job: verified())
            self.assertIs(runtime.tasks.store, runtime.store)
            self.assertIs(runtime.scheduler.store, runtime.store)
            self.assertEqual(runtime.health()["schema"], "task-runtime-boundary-v1")

    def test_unknown_replay_policy_is_rejected_at_composition_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                RuntimeBoundary.open(Path(tmp) / "runtime.sqlite3",
                    dispatch=lambda *args: {}, execute_action=lambda *args: verified(),
                    verify_job=lambda *args: verified(), wake_agent=lambda *args: verified(),
                    policy=RuntimePolicy(unknown_mutation_replay=True))


if __name__ == "__main__":
    unittest.main()
