import copy
import unittest

from smarthome_agent_rl.execution.events import EventEpisodeRuntime
from tests import test_episode_runtime as fixtures


class EventEpisodeRuntimeTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.EpisodeRuntimeTests('test_single_registration_durable_before_dispatch_and_receipt_preserved')
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        f.executor.dispatch = f.runtime.raw_dispatch
        f.runtime = EventEpisodeRuntime(f.executor, f.runtime.adapter, f.root / 'event.sqlite3',
            save=lambda *args: f.receipts.append(args))
        f.runtime.start('Turn on lamp at 12:00:10.', current_time=f.now)

    def event(self, value):
        return self.f.runtime.observe_native_response({'status': {'code': 200},
                                                      'data': {'current_time': value}})

    def test_early_and_duplicate_native_events_do_not_poll_or_write_snapshots(self):
        f = self.f
        f.register()
        calls, writes, budget = len(f.calls), len(f.receipts), f.executor.extra_queries
        for _ in range(100):
            self.assertEqual(self.event(f.now), [])
        self.assertEqual((len(f.calls), len(f.receipts), f.executor.extra_queries), (calls, writes, budget))
        f.runtime.flush()
        self.assertEqual(f.receipts[-1][1]['event_supervision']['native_clock_callbacks'], 100)

    def test_pending_native_job_is_reconciled_on_later_in_window_event_without_replay(self):
        f = self.f
        f.register()
        f.now = f.due
        self.assertEqual(self.event(f.now)[0]['status'], 'UNVERIFIED')
        self.assertEqual(f.runtime.store.list('job')[0]['status'], 'UNKNOWN')
        before = len(f.calls)
        self.assertEqual(self.event(f.now), [])
        self.assertEqual(len(f.calls), before)
        f.workflow_status = 'completed'
        f.state['endpoints']['1']['clusters']['OnOff']['attributes']['OnOff']['value'] = True
        f.now = '2030-01-01 12:00:11'
        self.assertEqual(self.event(f.now)[0]['status'], 'VERIFIED_SUCCESS')
        self.assertEqual(f.runtime.store.list('job')[0]['status'], 'DONE')
        self.assertEqual(sum(t == 'schedule_workflow' for t, _ in f.calls), 1)
        f.runtime.finish()
        self.assertEqual(f.runtime.store.list('task')[0]['status'], 'WAITING')

    def test_missing_and_backward_clock_do_not_tick_jobs(self):
        f = self.f
        f.register()
        self.event(f.now)
        self.assertFalse(f.runtime.observe_clock('2029-12-31 12:00:00', source={'kind': 'fixture'}))
        self.assertEqual(f.runtime.observe_native_response({'status': {'code': 200}, 'data': {}}), [])
        self.assertEqual(f.runtime.observe_native_response({'status': {'code': 500},
                         'data': {'current_time': f.due}}), [])
        self.assertEqual(f.runtime.store.list('job')[0]['status'], 'SCHEDULED')

    def test_late_public_event_records_missed_window_even_when_budget_is_exhausted(self):
        f = self.f
        f.register()
        f.executor.query_limit = f.executor.extra_queries
        before = len(f.calls)
        result = self.event('2030-01-01 12:00:13')
        self.assertEqual(result[0]['status'], 'UNVERIFIED')
        self.assertEqual(f.runtime.store.list('job')[0]['status'], 'UNKNOWN')
        self.assertEqual(len(f.calls), before)

    def test_readback_bundle_is_not_partially_consumed_when_budget_is_short(self):
        f = self.f
        f.register()
        f.executor.query_limit = f.executor.extra_queries + 2
        before = len(f.calls)
        self.assertEqual(self.event(f.due)[0]['reason'], 'READBACK_BUNDLE_BUDGET_EXHAUSTED')
        self.assertEqual(len(f.calls), before)
        self.assertEqual(f.runtime.store.list('job')[0]['status'], 'UNKNOWN')

    def test_actor_public_clock_is_reused_without_auxiliary_clock_request(self):
        f = self.f
        f.register()
        f.now = f.due
        f.executor.call('get_current_time', {})
        before = len(f.calls)
        f.runtime.supervise(phase='agent_tool_return')
        self.assertEqual([t for t, _ in f.calls[before:]], ['get_workflow_status'])


if __name__ == '__main__':
    unittest.main()
