import unittest
from smarthome_agent_rl.node_selection import select_guard


class SelectionTests(unittest.TestCase):
    def test_failed_safety_or_cost_gate_retains_control(self):
        def arm(successes, tokens, illegal):
            return {'successes': successes, 'success_rate': successes / 100,
                'all_totals': {'actor_tokens': tokens, 'invalid_reached_executor': illegal}}
        report = {'verified': True, 'phase': 'dev', 'arms': {'G': arm(50, 1000, 10),
            'GD': arm(51, 1101, 7), 'GW': arm(52, 1000, 8), 'GDW': arm(49, 900, 6)}}
        gates = {'illegal_reduction_min': .3, 'actor_token_ratio_max': 1.1}
        self.assertEqual(select_guard(report, gates)['winner'], 'G')
        report['arms']['GD']['all_totals']['actor_tokens'] = 1100
        self.assertEqual(select_guard(report, gates)['winner'], 'GD')
        report['phase'] = 'final'
        with self.assertRaises(ValueError):
            select_guard(report, gates)
