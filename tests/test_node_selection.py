import unittest
from smarthome_agent_rl.node_selection import select_guard, select_time_plan, assess_sft


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

    def test_temporal_improvement_requires_all_four_gates(self):
        def arm(successes, temporal, tokens, illegal):
            return {'success_rate': successes / 360,
                'categories': {'qt4-1:feasible': {'successes': temporal}},
                'all_totals': {'actor_tokens': tokens, 'invalid_reached_executor': illegal}}
        report = {'verified': True, 'phase': 'dev', 'reference': 'GD',
                  'arms': {'GD': arm(140, 10, 100, 4), 'TimePlan': arm(142, 11, 109, 4)}}
        gates = {'sr_delta_min': 0, 'actor_token_ratio_max': 1.1}
        self.assertEqual(select_time_plan(report, gates)['winner'], 'TimePlan')
        report['arms']['TimePlan']['categories']['qt4-1:feasible']['successes'] = 10
        self.assertEqual(select_time_plan(report, gates)['winner'], 'GD')
        report['arms']['TimePlan']['categories']['qt4-1:feasible']['successes'] = 11
        report['arms']['TimePlan']['all_totals']['invalid_reached_executor'] = 5
        self.assertEqual(select_time_plan(report, gates)['winner'], 'GD')

    def test_sft_shorter_output_cannot_hide_success_or_safety_regression(self):
        def arm(successes, tokens, illegal):
            return {'episodes':144,'success_rate':successes/144,
                'all_totals':{'actor_tokens':tokens,'invalid_reached_executor':illegal}}
        report={'verified':True,'phase':'dev','reference':'G',
            'arms':{'G':arm(50,1000,4),'SFT9B':arm(49,500,0)}}
        gates={'sr_delta_min':0,'actor_token_ratio_max':1.1}
        self.assertEqual(assess_sft(report,gates)['winner'],'G')
        report['arms']['SFT9B']=arm(51,1100,4)
        self.assertEqual(assess_sft(report,gates)['winner'],'SFT9B')
        self.assertFalse(assess_sft(report,gates)['checkpoint_selected_from_evaluation'])
        report['arms']['SFT9B']=arm(51,500,5)
        self.assertEqual(assess_sft(report,gates)['winner'],'G')
        report['arms']['SFT9B']['episodes']=143
        with self.assertRaises(ValueError):assess_sft(report,gates)
