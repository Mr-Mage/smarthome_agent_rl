import tempfile
import unittest
from pathlib import Path

from smarthome_agent_rl.execution.mutation import PostconditionResult, VerificationStatus
from smarthome_agent_rl.execution.runtime import RuntimeBoundary
from smarthome_agent_rl.execution.store import RuntimeStore


def verified():
    return PostconditionResult(VerificationStatus.VERIFIED_SUCCESS, ())


class RestartRecoveryTests(unittest.TestCase):
    def test_claimed_job_becomes_unknown_on_reopen_without_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.sqlite3"
            first = RuntimeBoundary.open(path, dispatch=lambda *a: {},
                execute_action=lambda *a: verified(), verify_job=lambda *a: verified(),
                wake_agent=lambda *a: verified())
            task = first.tasks.create("u", "turn off", "chat", expected_postconditions=[])
            job = first.scheduler.schedule(task.task_id, "u", expected_version=1, target_time=0)
            self.assertIsNotNone(first.scheduler._claim(job["job_id"], 0))
            second = RuntimeBoundary.open(path, dispatch=lambda *a: {},
                execute_action=lambda *a: self.fail("recovery must not dispatch"),
                verify_job=lambda *a: verified(), wake_agent=lambda *a: verified())
            recovered = second.store.get("job", job["job_id"])[0]
            self.assertEqual(recovered["status"], "UNKNOWN")
            self.assertEqual(len(second.store.list("recovery")), 1)
            self.assertEqual(second.scheduler.tick(1), [])

    def test_native_registration_remains_non_replayable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RuntimeStore(Path(tmp) / "runtime.sqlite3")
            store.put("job", "j", {"job_id": "j", "task_id": "missing",
                                    "status": "REGISTERING", "evidence": []})
            self.assertEqual(store.recover_inflight()[0]["status"], "REGISTERING")
            self.assertEqual(store.get("job", "j")[0]["status"], "REGISTERING")


if __name__ == "__main__":
    unittest.main()
