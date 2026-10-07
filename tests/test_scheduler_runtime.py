from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.execution.mutation import PostconditionResult, VerificationStatus
from smarthome_agent_rl.execution.scheduler import Scheduler
from smarthome_agent_rl.execution.store import RevisionConflict, RuntimeStore
from smarthome_agent_rl.execution.tasks import TaskManager
from smarthome_agent_rl.execution.trace import ToolTrace


def verified(observed_at=10):
    return PostconditionResult(VerificationStatus.VERIFIED_SUCCESS,
                               ({'path': 'power', 'observed': False, 'expected': False, 'matched': True},),
                               observed_at=observed_at)


class SchedulerRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tasks.sqlite3'
        self.store = RuntimeStore(self.path)
        self.manager = TaskManager(self.store)
        self.task = self.manager.create('u1', 'turn off at 10', 'chat1', expected_postconditions=[
            {'device_id': 'tv', 'path': 'power', 'value': False}])
        self.calls = []
        def dispatch(tool, args):
            self.calls.append(tool)
            return {'status': {'code': 200}, 'data': {'workflow_id': 'native-1',
                **({'status': 'cancelled'} if tool == 'cancel_native' else {})}}
        self.trace = ToolTrace(dispatch, sink=self.store.trace_sink)
        self.executions = []
        self.wakes = []
        def execute(task, job):
            self.executions.append(job['job_id'])
            return verified(job['target_time'])
        def wake(task, job):
            self.wakes.append((task.goal, task.version))
            return verified(job['target_time'])
        self.scheduler = Scheduler(self.manager, self.trace, execute_action=execute,
                                   verify_job=lambda job: verified(job['target_time']), wake_agent=wake,
                                   read_task_states=lambda task: {'tv': {'power': False}})

    def schedule(self, **args):
        return self.scheduler.schedule(self.task.task_id, 'u1', expected_version=1, target_time=10, **args)

    def test_native_delegation_is_verified_without_duplicate_execution(self):
        job = self.schedule(native_call={'tool': 'native_schedule', 'args': {'at': 10}})
        self.assertEqual(job['mode'], 'device_native')
        self.assertEqual(self.scheduler.tick(9), [])
        self.assertEqual(len(self.scheduler.tick(10)), 1)
        self.assertEqual(self.calls, ['native_schedule'])
        self.assertEqual(self.executions, [])
        self.assertEqual(self.manager.get(self.task.task_id, 'u1').status, 'COMPLETED')

    def test_native_rejection_with_null_data_finishes_receipt_linkage_without_replay(self):
        self.trace.dispatch = lambda *args: {'status': {'code': 400}, 'data': None,
                                            'error': {'type': 'HTTP_ERROR'}}
        job = self.schedule(native_call={'tool': 'native_schedule', 'args': {'at': 10}})
        self.assertEqual(job['status'], 'SCHEDULED')
        self.assertTrue(job['registration_uncertain'])
        wf = self.store.get('workflow', job['workflow_id'])[0]
        self.assertIsNone(wf['device_workflow_id'])
        self.assertFalse(wf['evidence'][-1]['acknowledged'])
        self.assertEqual(wf['evidence'][-1]['invocation_id'], self.trace.invocations[0].invocation_id)
        self.assertEqual(len(self.trace.invocations), 1)

    def test_two_workers_claim_a_harness_job_only_once(self):
        self.schedule()
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(self.scheduler.tick, [10, 10]))
        self.assertEqual(len(self.executions), 1)

    def test_wake_restores_goal_from_persistent_task_after_restart(self):
        self.schedule(wake_agent=True)
        self.scheduler.manager = TaskManager(RuntimeStore(self.path))
        self.scheduler.tick(10)
        self.assertEqual(self.wakes, [('turn off at 10', 1)])
        self.assertEqual(self.executions, [])

    def test_task_revision_cancels_obsolete_harness_jobs(self):
        job = self.schedule()
        self.manager.revise(self.task.task_id, 'u1', expected_version=1, source_conversation='chat2', goal='turn off at 20')
        self.assertEqual(self.store.get('job', job['job_id'])[0]['status'], 'CANCELLED')
        self.assertEqual(self.scheduler.tick(10), [])

    def test_native_task_revision_requires_acknowledged_device_cancellation(self):
        job = self.schedule(native_call={'tool': 'native_schedule', 'args': {'at': 10}})
        with self.assertRaises(RevisionConflict):
            self.manager.revise(self.task.task_id, 'u1', expected_version=1, source_conversation='chat2', goal='new goal')
        self.scheduler.cancel_workflow(job['workflow_id'], 'u1', cancel_native={'tool': 'cancel_native'})
        revised = self.manager.revise(self.task.task_id, 'u1', expected_version=1, source_conversation='chat2', goal='new goal')
        self.assertEqual(revised.version, 2)
        self.assertEqual(self.scheduler.tick(10), [])

    def test_uncertain_execution_is_reconciled_without_replay(self):
        job = self.schedule()
        def timeout(task, job):
            self.executions.append(job['job_id'])
            raise TimeoutError('response lost')
        self.scheduler.execute_action = timeout
        self.scheduler.tick(10)
        self.assertEqual(self.store.get('job', job['job_id'])[0]['status'], 'UNKNOWN')
        self.scheduler.tick(11)
        self.scheduler.reconcile(job['job_id'], now=11)
        self.assertEqual(self.executions, [job['job_id']])
        self.assertEqual(self.store.get('job', job['job_id'])[0]['status'], 'DONE')

    def test_whole_task_cannot_complete_while_future_jobs_remain(self):
        self.schedule()
        self.scheduler.schedule(self.task.task_id, 'u1', expected_version=1, target_time=20)
        self.scheduler.tick(10)
        self.assertNotEqual(self.manager.get(self.task.task_id, 'u1').status, 'COMPLETED')
        self.scheduler.tick(20)
        self.assertEqual(self.manager.get(self.task.task_id, 'u1').status, 'COMPLETED')

    def test_late_tick_does_not_execute_an_expired_fixed_action(self):
        job = self.schedule()
        result = self.scheduler.tick(11)
        self.assertEqual(self.executions, [])
        self.assertEqual(result[0]['status'], VerificationStatus.UNVERIFIED)
        self.assertIn('WINDOW_MISSED', result[0]['reason'])
        self.assertEqual(self.store.get('job', job['job_id'])[0]['status'], 'UNKNOWN')

    def test_late_current_state_cannot_prove_original_timed_goal(self):
        job = self.schedule()
        self.scheduler.tick(11)
        self.scheduler.verify_job = lambda job: verified(11)
        result = self.scheduler.reconcile(job['job_id'], now=11)
        self.assertEqual(result.status, VerificationStatus.UNVERIFIED)
        self.assertEqual(self.store.get('job', job['job_id'])[0]['status'], 'UNKNOWN')
        self.assertNotEqual(self.manager.get(self.task.task_id, 'u1').status, 'COMPLETED')
        self.assertEqual(self.executions, [])

    def test_timed_callbacks_require_timestamp_and_reject_stale_snapshot(self):
        for observation in (None, 9, 11, float('nan'), True):
            with self.subTest(observation=observation):
                job = self.schedule(native_call={'tool': 'native_schedule', 'args': {'at': 10}})
                self.scheduler.verify_job = lambda job: verified(observation)
                self.scheduler.tick(10)
                self.assertEqual(self.store.get('job', job['job_id'])[0]['status'], 'UNKNOWN')

    def test_retained_in_window_observation_reconciles_but_future_one_does_not(self):
        job = self.schedule(tolerance=2)
        self.scheduler.execute_action = lambda *args: verified(None)
        self.scheduler.tick(10)
        self.scheduler.verify_job = lambda job: verified(12)
        self.assertEqual(self.scheduler.reconcile(job['job_id'], now=11).status,
                         VerificationStatus.UNVERIFIED)
        self.scheduler.verify_job = lambda job: verified(10)
        self.assertEqual(self.scheduler.reconcile(job['job_id'], now=20).status,
                         VerificationStatus.VERIFIED_SUCCESS)
        self.assertEqual(self.manager.get(self.task.task_id, 'u1').status, 'COMPLETED')

    def test_later_matching_task_state_cannot_erase_verified_timed_failure(self):
        job = self.schedule()
        self.scheduler.execute_action = lambda *args: PostconditionResult(
            VerificationStatus.VERIFIED_FAILURE,
            ({'path': 'power', 'observed': True, 'expected': False, 'matched': False},),
            observed_at=10)
        self.scheduler.tick(10)
        # Whole-task callback returns power=False, but the timed failure remains.
        task = self.manager.get(self.task.task_id, 'u1')
        self.assertEqual(task.status, 'FAILED')
        self.assertEqual(task.evidence[-1]['failed_job_ids'], [job['job_id']])

    def test_captured_live_observation_uses_clock_after_read_and_still_rejects_future_evidence(self):
        job = self.schedule(tolerance=2)
        self.scheduler.execute_action = lambda *args: verified(None)
        self.scheduler.tick(10)
        self.assertEqual(self.scheduler.reconcile_observation(job['job_id'], verified(11), now=10).status,
                         VerificationStatus.UNVERIFIED)
        self.assertEqual(self.scheduler.reconcile_observation(job['job_id'], verified(11), now=11).status,
                         VerificationStatus.VERIFIED_SUCCESS)
        self.assertEqual(self.store.get('job', job['job_id'])[0]['status'], 'DONE')

    def test_cancellation_acknowledgement_alone_does_not_confirm_cancelled(self):
        job = self.schedule(native_call={'tool': 'native_schedule', 'args': {'at': 10}})
        self.trace.dispatch = lambda *args: {'status': {'code': 200}, 'data': {}}
        with self.assertRaises(RevisionConflict):
            self.scheduler.cancel_workflow(job['workflow_id'], 'u1', cancel_native={'tool': 'cancel_native'})
        self.assertEqual(self.store.get('workflow', job['workflow_id'])[0]['status'], 'SCHEDULED')


if __name__ == '__main__':
    unittest.main()
