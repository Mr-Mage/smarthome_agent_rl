import unittest
from smarthome_agent_rl.paired_stats import mcnemar, bootstrap_ci, holm


class PairedStatsTests(unittest.TestCase):
    def test_exact_mcnemar_known_discordances_and_no_change(self):
        self.assertEqual(mcnemar([False] * 5, [True] * 5)['p_exact'], 0.0625)
        self.assertEqual(mcnemar([True, False], [True, False])['p_exact'], 1.0)
        self.assertEqual(mcnemar([True, False], [False, True])['success_delta'], 0)

    def test_stratified_bootstrap_and_holm(self):
        self.assertEqual(bootstrap_ci([[0, 0], [0, 0]], repeats=100), [0, 0])
        self.assertEqual(bootstrap_ci([[1, 1], [1, 1]], repeats=100), [1, 1])
        self.assertEqual(holm({'G': .01, 'GC': .03, 'Full': .04}), {'G': .03, 'GC': .06, 'Full': .06})


if __name__ == '__main__':
    unittest.main()
