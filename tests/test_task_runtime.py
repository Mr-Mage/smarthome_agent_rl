from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.execution.store import RevisionConflict, RuntimeStore
from smarthome_agent_rl.execution.tasks import TaskManager, TaskStatus


class TaskRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tasks.sqlite3'
        self.store = RuntimeStore(self.path)
        self.manager = TaskManager(self.store)

    def test_conversation_end_and_restart_do_not_remove_task(self):
        task = self.manager.create('u1', 'Sleep at 23:00', 'session1')
        restored = TaskManager(RuntimeStore(self.path))
        task = restored.revise(task.task_id, 'u1', expected_version=1,
                              source_conversation='session2', goal='Sleep at midnight')
        self.assertEqual(task.version, 2)
        self.assertEqual(task.source_conversation, 'session1')
        self.assertEqual(task.evidence[-1]['source_conversation'], 'session2')
        self.assertEqual(len(restored.active('u1')), 1)

    def test_two_concurrent_revisions_have_one_winner(self):
        task = self.manager.create('u1', 'goal', 'session1')
        def revise(number):
            try:
                self.manager.revise(task.task_id, 'u1', expected_version=1,
                                    source_conversation=str(number), goal=f'goal{number}')
                return True
            except RevisionConflict:
                return False
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(revise, [1, 2]))
        self.assertEqual(results.count(True), 1)

    def test_cross_user_lookup_is_forbidden(self):
        task = self.manager.create('u1', 'goal', 'session1')
        self.assertEqual(self.manager.active('u2'), [])
        with self.assertRaises(PermissionError):
            self.manager.get(task.task_id, 'u2')

    def test_whole_task_requires_all_device_postconditions(self):
        task = self.manager.create('u1', 'sleep', 'session1', expected_postconditions=[
            {'device_id': 'ac', 'path': 'target', 'value': 25},
            {'device_id': 'tv', 'path': 'power', 'value': False}])
        with self.assertRaises(ValueError):
            self.manager.transition(task.task_id, 'u1', 'COMPLETED', expected_version=1, evidence={'200': True})
        result = self.manager.verify(task.task_id, 'u1', expected_version=1, public_states={'ac': {'target': 25}})
        self.assertEqual(result.status, TaskStatus.WAITING)
        result = self.manager.verify(task.task_id, 'u1', expected_version=1,
                                     public_states={'ac': {'target': 25}, 'tv': {'power': False}})
        self.assertEqual(result.status, TaskStatus.COMPLETED)

    def test_invocation_evidence_is_append_only(self):
        self.store.trace_sink({'invocation_id': 'i1', 'tool': 'read'})
        with self.assertRaises(ValueError):
            self.store.put('trace', 'i1', {'tool': 'mutate'}, expected_revision=1)


if __name__ == '__main__':
    unittest.main()
