import unittest
from scripts.analyze_homebench_symmetric import compare
from smarthome_agent_rl.benchmarks.homebench import native_counts


class Adapter:
    def contract(self, task_id):
        return None, {}, {'functions': {'room.light.on': {'args': {}}}}

    def guard(self, task_id, prediction):
        return {'prediction': prediction, 'rejections': [], 'uncovered': []}

    def score(self, task_id, prediction):
        return native_counts(prediction, 'room.light.on()')


class SymmetricTests(unittest.TestCase):
    def test_format_advantage_is_removed_from_contract_attribution(self):
        rows = [{'task_id': 'x', 'arm': 'B0', 'prediction': '{room.light.on();}', 'error': None},
                {'task_id': 'x', 'arm': 'B1', 'prediction': '{room.light.on()}', 'error': None}]
        result = compare(Adapter(), rows)
        self.assertEqual(result['contract_paired_changes']['raw']['wins'], 1)
        self.assertEqual(result['contract_paired_changes']['F']['wins'], 0)
        self.assertEqual(result['arms']['B0_F']['exact_match'], 1)
        rows[0]['error'] = 'transport failure'
        result = compare(Adapter(), rows)
        self.assertEqual(result['arms']['B0_WG']['exact_match'], 0)
        self.assertEqual(result['contract_paired_changes']['WG']['wins'], 1)
        rows[0]['error'], rows[1]['error'] = None, 'transport failure'
        self.assertEqual(compare(Adapter(), rows)['contract_paired_changes']['WG']['losses'], 1)

    def test_incomplete_and_duplicate_pairs_fail_closed(self):
        row = {'task_id': 'x', 'arm': 'B0', 'prediction': ''}
        for rows in ([row], [row, row]):
            with self.assertRaises(ValueError):
                compare(Adapter(), rows)


if __name__ == '__main__':
    unittest.main()
