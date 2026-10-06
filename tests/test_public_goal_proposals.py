import json
import unittest

from smarthome_agent_rl.execution.goals import parse_proposal, parse_review, public_messages, review_messages


def goal(index, text, target, condition, time, dependencies=()):
    return {'goal_id': f'g{index}', 'source_text': text, 'target_text': target,
        'condition_text': condition, 'time_text': time, 'kind': 'schedule',
        'depends_on': list(dependencies), 'interpretation': 'interpreted'}


class PublicGoalProposalTests(unittest.TestCase):
    def setUp(self):
        # Text from exposed official qt4-1_feasible_seed_26; engineering fixture,
        # not a generated benchmark or hidden target.
        self.a = 'switch on dehumidifier 2 in the dining room 9 minutes from now power on and fan 40%'
        self.b = '29 minutes after the previous action keep dehumidifier 2 in the dining room powered on and increase fan to 60%'
        self.c = 'switch on air purifier 1 in the bathroom 24 minutes from now power on and fan 50%'
        self.d = '22 minutes after the previous action keep air purifier 1 in the bathroom powered on and increase fan to 80%'
        self.query = self.a + ' and ' + self.b + '. Also ' + self.c + ' and ' + self.d + '.'
        self.goals = [goal(1, self.a, 'dehumidifier 2 in the dining room', 'power on and fan 40%', '9 minutes from now'),
            goal(2, self.b, 'dehumidifier 2 in the dining room', 'powered on and increase fan to 60%',
                 '29 minutes after the previous action', ['g1']),
            goal(3, self.c, 'air purifier 1 in the bathroom', 'power on and fan 50%', '24 minutes from now'),
            goal(4, self.d, 'air purifier 1 in the bathroom', 'powered on and increase fan to 80%',
                 '22 minutes after the previous action', ['g3'])]
        self.now = '2025-08-23 12:42:14'

    def proposal(self):
        return parse_proposal(self.query, self.now, json.dumps({'goals': self.goals}))

    def test_four_public_phases_resolve_separate_sequences_without_execution_claim(self):
        plan = self.proposal()
        self.assertEqual([r['due_time'] for r in plan['time_graph']],
            ['2025-08-23 12:51:14', '2025-08-23 13:20:14', '2025-08-23 13:06:14', '2025-08-23 13:28:14'])
        self.assertFalse(plan['completes_user_task'])
        self.assertEqual(plan['semantic_validity'], 'model_proposed_unverified')

    def test_missing_anchor_or_public_clock_does_not_invent_previous_phase_time(self):
        self.goals[1]['depends_on'] = []
        plan = self.proposal()
        self.assertEqual(plan['time_graph'][1]['status'], 'UNRESOLVED')
        self.assertIsNone(plan['time_graph'][1]['due_time'])
        plan = parse_proposal(self.query, None, json.dumps({'goals': self.goals}))
        self.assertTrue(all(r['due_time'] is None for r in plan['time_graph']))

    def test_event_condition_stays_unresolved_instead_of_becoming_clock_deadline(self):
        text = 'pause it when the dryer finishes'
        row = goal(1, text, 'it', 'pause', 'when the dryer finishes')
        plan = parse_proposal(text, self.now, json.dumps({'goals': [row]}))
        self.assertEqual(plan['time_graph'][0]['status'], 'UNRESOLVED')

    def test_fabricated_quotes_unknown_cycles_and_forward_dependencies_are_rejected(self):
        self.goals[0]['condition_text'] = 'fan 99%'
        with self.assertRaises(ValueError):
            self.proposal()
        self.goals[0]['condition_text'] = 'power on and fan 40%'
        for deps in (['absent'], ['g1'], ['g4']):
            self.goals[0]['depends_on'] = deps
            with self.assertRaises(ValueError):
                self.proposal()

    def test_public_boundary_and_review_do_not_certify_satisfaction(self):
        plan = self.proposal()
        messages = public_messages(self.query, self.now)
        self.assertEqual(set(json.loads(messages[-1]['content'])), {'user_request', 'initial_public_time'})
        self.assertNotIn('official_score', json.dumps(review_messages(self.query, self.now, plan)))
        with self.assertRaises(ValueError):
            review_messages('another request', self.now, plan)
        review = parse_review(json.dumps({d: 'YES' for d in ('coverage', 'fidelity', 'targets', 'dependencies')} |
                                         {'explanation': 'Public phases preserved.'}))
        self.assertEqual(review['coverage'], 'YES')
        self.assertFalse(plan['completes_user_task'])

    def test_hidden_fields_and_forged_clock_graph_are_not_sent_to_reviewer(self):
        plan = self.proposal()
        plan['official_score'] = 1
        with self.assertRaises(ValueError):
            review_messages(self.query, self.now, plan)
        plan = self.proposal()
        plan['time_graph'][1]['due_time'] = plan['time_graph'][0]['due_time']
        with self.assertRaises(ValueError):
            review_messages(self.query, self.now, plan)


if __name__ == '__main__':
    unittest.main()
