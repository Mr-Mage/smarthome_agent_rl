import json
import unittest
from types import SimpleNamespace
from smarthome_agent_rl.time_plan import TimePlan
from smarthome_agent_rl.guard import GuardError
from smarthome_agent_rl.structured import StructuredProvider
from src.agents.types import ChatMessage


class TimePlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = TimePlan()
        self.plan.initialize('Turn on office light 22 minutes from now. Set refrigerator to 1°C '
                             '30 minutes from now and 5°C 21 minutes after the previous action.',
                             '2025-08-23 17:20:08')
        self.body = {'thought': 'test', 'call': {'tool': 'get_rooms', 'arguments': {}},
            'time_plan': {'t0': 'now', 't1': 'now', 't2': 't1'},
            'time_refs': [], 'time_dispositions': []}
        self.plan.consume(self.body)

    def test_case_separate_22_30_51_minutes_and_no_mutation_on_conflict(self):
        self.assertEqual([r['due_time'] for r in self.plan.table.values()],
                         ['2025-08-23 17:42:08', '2025-08-23 17:50:08', '2025-08-23 18:11:08'])
        self.plan.metadata['refs'] = ['t0', 't1', 't2']
        with self.assertRaises(GuardError):
            self.plan.check('schedule_workflow', {'start_time': '2025-08-23 17:42:08', 'steps': [{}, {}, {}]})
        for ref in self.plan.table:
            self.plan.metadata['refs'] = [ref]
            arguments = {'start_time': self.plan.table[ref]['due_time'], 'steps': [{}]}
            self.plan.check('schedule_workflow', arguments)
            self.plan.observe('schedule_workflow', arguments, {'status': {'code': 200}, 'data': {'workflow_id': ref}})
        self.assertEqual(len(self.plan.receipts), 3)

    def test_finish_receipts_infeasibility_and_cancellation(self):
        self.plan.metadata['dispositions'] = [{'id': ref, 'status': 'registered', 'reason': ''} for ref in self.plan.table]
        with self.assertRaises(GuardError):
            self.plan.check('finish', {})
        self.plan.metadata['dispositions'] = [{'id': ref, 'status': 'infeasible', 'reason': 'Public device does not exist'}
                                              for ref in self.plan.table]
        self.plan.check('finish', {})
        self.plan.receipts['t0'] = {'workflow_id': 'w'}
        self.plan.observe('cancel_workflow', {'workflow_id': 'w'}, {'status': {'code': 200}, 'data': {}})
        self.assertNotIn('t0', self.plan.receipts)

    def test_uncovered_formats_and_invalid_anchor_are_not_invented(self):
        plan = TimePlan()
        plan.initialize('When laundry completes, turn on light', '2025-08-23 17:20:08')
        self.assertEqual(plan.rows, [])
        plan.check('schedule_workflow', {'anything': 'unchanged'})
        plan.initialize('Turn on light 5 minutes from now', None)
        self.assertEqual(plan.rows, [])
        plan.initialize('Turn on light 5 minutes from now', '2025-08-23 17:20:08')
        with self.assertRaises(ValueError):
            plan.consume({'time_plan': {'t0': 't0'}})

    def test_finish_gate_runs_in_provider_even_when_upstream_intercepts_finish(self):
        body = {'thought': 'done', 'call': {'tool': 'finish', 'arguments': {'answer': 'Scheduled'}},
                'time_refs': [], 'time_dispositions': [{'id': ref, 'status': 'registered', 'reason': ''} for ref in self.plan.table]}
        inner = SimpleNamespace(generate=lambda *a, **kw: json.dumps(body))
        provider = StructuredProvider(inner, finish_guard=False, time_plan=self.plan)
        result = provider.generate([ChatMessage(role='system', content='tools'),
                                   ChatMessage(role='user', content='This is your actual task.')])
        self.assertEqual(result, '{}')
        self.assertIn('no actual receipt', provider.audit[-1]['validation_error'])

    def test_finite_anchor_schema_prevents_unbounded_plan_rows_and_unknown_previous_is_uncovered(self):
        fresh = TimePlan()
        fresh.initialize('Turn on light 5 minutes from now and off 10 minutes after the previous action.',
                         '2025-08-23 17:20:08')
        schema = fresh.augment_schema({'json_schema': {'schema': {'properties': {}, 'required': []}}})
        plan = schema['json_schema']['schema']['properties']['time_plan']
        self.assertEqual(plan['type'], 'object')
        self.assertEqual(plan['properties']['t0']['enum'], ['now'])
        self.assertEqual(plan['properties']['t1']['enum'], ['t0'])
        fresh.initialize('Turn on light 5 minutes after the previous action.', '2025-08-23 17:20:08')
        fresh.consume({'time_plan': {'t0': 'uncovered'}, 'time_refs': ['t0'], 'time_dispositions': []})
        fresh.check('schedule_workflow', {'start_time': '2025-08-23 17:42:00', 'steps': [{}]})
        self.assertIsNone(fresh.table['t0']['due_time'])
