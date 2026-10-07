import copy
import unittest

from smarthome_agent_rl.execution.resource_context import ResourceEpisodeRuntime, fixed_claims
from smarthome_agent_rl.execution.store import RuntimeStore
from tests import test_episode_runtime as fixtures


class ResourceEpisodeTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.EpisodeRuntimeTests('test_single_registration_durable_before_dispatch_and_receipt_preserved')
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        f.executor.dispatch = f.runtime.raw_dispatch
        f.runtime = ResourceEpisodeRuntime(f.executor, f.runtime.adapter, f.root / 'resources.sqlite3',
            save=lambda *args: f.receipts.append(args))
        def dispatch(tool, args):
            if tool != 'schedule_workflow':
                return f.dispatch(tool, args)
            f.calls.append((tool, copy.deepcopy(args)))
            self.assertTrue(any(j['status'] == 'REGISTERING' for j in f.runtime.store.list('job')))
            count = sum(t == tool for t, _ in f.calls)
            return {'status': {'code': 200}, 'data': {'workflow_id': f'native-{count}'}}
        f.runtime.raw_dispatch = dispatch
        f.runtime.start('Turn on lamp at 12:00:10.', current_time=f.now)

    def test_claims_conflicts_and_context_are_durable_without_extra_reads_or_arbitration(self):
        f = self.f
        self.assertIsNone(f.runtime.actor_context())
        first = f.register()
        f.arguments['steps'][0]['args']['command_id'] = 'Off'
        second = f.register()
        self.assertEqual(first['data']['workflow_id'], 'native-1')
        self.assertEqual(second['data']['workflow_id'], 'native-2')
        self.assertEqual([t for t, _ in f.calls], ['get_device_structure', 'schedule_workflow', 'schedule_workflow'])
        jobs = f.runtime.store.list('job')
        self.assertEqual({j['resource_claims'][0]['value'] for j in jobs}, {True, False})
        self.assertEqual(len(f.runtime.store.list('conflict')), 1)
        context = f.runtime.actor_context()
        self.assertEqual(context['conflicts_total'], 1)
        self.assertEqual({j['status'] for j in context['jobs']}, {'SCHEDULED'})
        self.assertIn('not current state', context['semantics'])
        self.assertEqual(len(RuntimeStore(f.runtime.store.path).list('conflict')), 1)
        self.assertFalse(any(t == 'cancel_workflow' for t, _ in f.calls))
        f.runtime.finish()
        self.assertEqual(f.runtime.store.list('task')[0]['status'], 'WAITING')

    def test_late_unknown_retains_claim_even_after_legacy_cancellation_path(self):
        f = self.f
        f.register()
        f.arguments['steps'][0]['args']['command_id'] = 'Off'
        f.register()
        f.runtime.observe_native_response({'status': {'code': 200},
                                         'data': {'current_time': '2030-01-01 12:00:13'}})
        self.assertEqual({j['status'] for j in f.runtime.store.list('job')}, {'UNKNOWN'})
        self.assertEqual(f.runtime.actor_context()['conflicts_total'], 1)
        # UNKNOWN cannot be treated as definitively cancelled by this legacy
        # cancellation path. It remains live rather than claiming resolution.
        f.executor.call('cancel_workflow', {'workflow_id': 'native-1'})
        self.assertEqual(f.runtime.actor_context()['conflicts_total'], 1)

    def test_scheduled_cancel_has_no_live_conflict_but_keeps_history(self):
        f = self.f
        f.register()
        f.arguments['steps'][0]['args']['command_id'] = 'Off'
        f.register()
        f.executor.call('cancel_workflow', {'workflow_id': 'native-1'})
        self.assertEqual(f.runtime.actor_context()['conflicts_total'], 0)
        self.assertEqual(len(f.runtime.store.list('conflict')), 1)

    def test_last_write_is_fixed_intention_and_unknown_effect_invalidates_prior_device_claims(self):
        f = self.f
        on = f.arguments['steps'][0]
        off = copy.deepcopy(on)
        off['args']['command_id'] = 'Off'
        result = fixed_claims(f.runtime.adapter, [on, off], {'lamp': f.state})
        self.assertEqual([r['value'] for r in result['claims']], [False])
        self.assertEqual(result['provenance'][0]['step'], 1)
        off['args']['command_id'] = 'Toggle'
        result = fixed_claims(f.runtime.adapter, [on, off], {'lamp': f.state})
        self.assertEqual(result['claims'], [])
        self.assertEqual(len(result['uncovered']), 1)

    def test_future_state_reference_and_wrong_schema_identity_remain_uncovered(self):
        f = self.f
        adapter = copy.deepcopy(f.runtime.adapter)
        for value in ('$state', '$state.endpoints.1.clusters.OnOff.attributes.OnOff.value', '$action.args'):
            adapter.rules['commands']['OnOff.On']['postconditions'][0]['value'] = value
            result = fixed_claims(adapter, f.arguments['steps'], {'lamp': f.state})
            self.assertEqual(result['claims'], [])
            self.assertIn('NON_ARGUMENT_FUTURE_REFERENCE', result['uncovered'][0]['reason'])
        state = {**f.state, 'device_id': 'wrong'}
        self.assertEqual(fixed_claims(f.runtime.adapter, f.arguments['steps'], {'lamp': state})['claims'], [])

    def test_uncovered_actions_are_not_compatibility_and_context_marks_truncation(self):
        f = self.f
        f.register()
        f.arguments['steps'][0]['args']['command_id'] = 'Toggle'
        f.register()
        context = f.runtime.actor_context(max_jobs=1, max_claims=0, max_conflicts=0)
        self.assertEqual(context['jobs_total'], 2)
        self.assertTrue(context['jobs_truncated'])
        self.assertEqual(f.runtime.store.list('conflict'), [])
        context = f.runtime.actor_context()
        self.assertEqual(sum(len(j['uncovered_steps']) for j in context['jobs']), 1)
        self.assertIn('does not prove compatibility', context['semantics'])

    def test_context_large_values_are_removed_as_whole_rows_with_coverage_counts(self):
        import json
        f = self.f
        f.register()
        job = f.runtime.store.list('job')[0]
        row, revision = f.runtime.store.get('job', job['job_id'])
        row['resource_claims'][0]['value'] = 'x' * 20000
        f.runtime.store.put('job', row['job_id'], row, expected_revision=revision)
        context = f.runtime.actor_context()
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False, sort_keys=True)), 12000)
        self.assertEqual(context['jobs_total'], 1)
        self.assertTrue(context['size_truncated'])
        self.assertTrue(context['jobs_truncated'])
        self.assertEqual(context['jobs'], [])


if __name__ == '__main__':
    unittest.main()
