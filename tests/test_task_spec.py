import copy
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.action_state import ActionLedger
from smarthome_agent_rl.task_spec import TaskSpec, observed_devices


def goal(goal_id='g1', text='Start the washer now', **changes):
    return {'goal_id': goal_id, 'source_text': text, 'kind': 'change', 'target_text': 'washer',
            'condition_text': 'Start', 'time_text': 'now', 'depends_on': [],
            'interpretation': 'interpreted', **changes}


def body(goals=None, refs=None, bindings=None):
    return {'thought': 'plan', 'call': {'tool': 'get_rooms', 'arguments': {}},
            'task_spec': {'goals': goals} if goals is not None else None,
            'goal_refs': refs or [], 'goal_bindings': bindings or []}


def observation(tool, args, data, code=200):
    return {'tool': tool, 'arguments': args, 'response': {'status': {'code': code}, 'error': None, 'data': data}}


class TaskSpecTests(unittest.TestCase):
    def setUp(self):
        self.executor = SimpleNamespace(observations=[], actions=ActionLedger())
        self.spec = TaskSpec(self.executor)
        self.spec.initialize('Start the washer now and pause it when the dryer finishes.')

    def test_multi_goal_spans_and_dependency_are_exact_and_no_hidden_satisfaction(self):
        goals = [goal(), goal('g2', 'pause it when the dryer finishes', target_text='it',
            condition_text='pause', time_text='when the dryer finishes', kind='schedule', depends_on=['g1'])]
        original = copy.deepcopy(goals)
        base, draft = self.spec.prepare(body(goals, refs=['g1']))
        self.spec.commit(draft, {'action': 'get_rooms'}, 1)
        self.assertEqual(set(base), {'thought', 'call'})
        for row in self.spec.snapshot()['goals']:
            span = row['source_span']
            self.assertEqual(self.spec.query[span['start']:span['end']], row['source_text'])
            self.assertEqual(row['satisfaction'], 'unverified')
        self.assertEqual(goals, original)
        self.assertEqual(self.spec.snapshot()['coverage'], 'model_extraction_unverified')

    def test_paraphrases_ambiguous_quotes_and_invented_subfields_are_rejected(self):
        for changed in ({'source_text': 'Run the washing machine'}, {'target_text': 'kitchen washer'},
                        {'time_text': '2030-01-01 12:00:00'}):
            with self.assertRaises(ValueError):
                self.spec.prepare(body([goal(**changed)]))
        spec = TaskSpec(self.executor)
        spec.initialize('Start Start')
        with self.assertRaises(ValueError):
            spec.validate_goals({'goals': [goal(text='Start', target_text='', time_text='')]})

    def test_cycles_unknown_dependencies_and_duplicated_goal_refs_fail(self):
        goals = [goal(depends_on=['g2']), goal('g2', 'pause it when the dryer finishes',
            target_text='it', condition_text='pause', time_text='', depends_on=['g1'])]
        with self.assertRaises(ValueError):
            self.spec.prepare(body(goals))
        with self.assertRaises(ValueError):
            self.spec.prepare(body([goal(depends_on=['missing'])]))
        for refs in (['g9'], ['g1', 'g1'], [True]):
            with self.assertRaises(ValueError):
                self.spec.prepare(body([goal()], refs=refs))

    def test_bindings_need_successful_matching_public_evidence_and_are_not_semantic_proof(self):
        self.executor.observations = [observation('get_device_structure', {'device_id': 'washer'},
            {'device_id': 'other'}), observation('get_device_structure', {'device_id': 'washer'},
            {'device_id': 'washer'}, code=404)]
        binding = [{'goal_id': 'g1', 'device_ids': ['washer']}]
        with self.assertRaises(ValueError):
            self.spec.prepare(body([goal()], bindings=binding))
        self.executor.observations.append(observation('get_device_structure', {'device_id': 'washer'},
                                                     {'device_id': 'washer'}))
        _, draft = self.spec.prepare(body([goal()], bindings=binding))
        self.spec.commit(draft, {'action': 'get_rooms'}, 1)
        bound = self.spec.snapshot()['goals'][0]['binding']
        self.assertEqual(bound['sources'][0]['observation_index'], 3)
        self.assertEqual(bound['semantic_match'], 'model_proposed_unverified')

    def test_refresh_and_topology_invalidation_do_not_leave_stale_binding(self):
        self.executor.observations.append(observation('get_room_devices', {'room_id': 'r'},
                                                     {'washer': {'device_type': 'washer'}}))
        _, draft = self.spec.prepare(body([goal()], bindings=[{'goal_id': 'g1', 'device_ids': ['washer']}]))
        self.spec.commit(draft, {'action': 'get_rooms'}, 1)
        self.executor.observations.append(observation('get_room_devices', {'room_id': 'r'}, {}))
        self.assertIsNone(self.spec.snapshot()['goals'][0]['binding'])
        self.executor.observations.append(observation('get_device_structure', {'device_id': 'washer'}, {'device_id': 'washer'}))
        self.executor.observations.append(observation('remove_device', {'device_id': 'washer'}, {}))
        self.assertEqual(observed_devices(self.executor.observations), {})

    def test_frozen_spec_and_episode_boundary_are_enforced(self):
        _, draft = self.spec.prepare(body([goal()]))
        self.spec.commit(draft, {'action': 'get_rooms'}, 1)
        with self.assertRaises(ValueError):
            self.spec.prepare(body([goal()]))
        self.spec.prepare(body(refs=['g1']))
        with self.assertRaises(ValueError):
            self.spec.initialize('Another request')

    def test_acknowledgement_registration_and_unknown_action_do_not_certify_goal(self):
        _, draft = self.spec.prepare(body([goal()]))
        self.spec.commit(draft, {'action': 'get_rooms'}, 1)
        for tool, response in [('execute_command', {'status': {'code': 200}, 'error': None, 'data': {}}),
                               ('schedule_workflow', {'status': {'code': 200}, 'error': None, 'data': {'workflow_id': 'w'}}),
                               ('execute_command', {'status': {'code': 408}, 'error': 'timeout'})]:
            aid = self.executor.actions.propose(tool, {}, turn=1, observation_version=0)
            self.executor.actions.records[aid].goal_ids = ['g1']
            self.executor.actions.dispatched(aid)
            self.executor.actions.observed(aid, response, 1)
        row = self.spec.snapshot()['goals'][0]
        self.assertEqual([r['state'] for r in row['action_evidence']], ['acknowledged', 'registered', 'unknown'])
        self.assertEqual(row['satisfaction'], 'unverified')


if __name__ == '__main__':
    unittest.main()
