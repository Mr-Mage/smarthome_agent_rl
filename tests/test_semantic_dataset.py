import unittest
import json

from scripts.build_semantic_verifier_dataset import build_examples


class SemanticDatasetTests(unittest.TestCase):
    def test_successful_public_record_generates_positive_and_hard_negatives(self):
        rows = build_examples([{'task_id': 't1', 'success': True, 'user_goal': 'turn on the bedroom light',
            'environment_state': {'devices': {
                'bedroom_light': {'device_id': 'bedroom_light', 'room_id': 'bedroom'},
                'living_light': {'device_id': 'living_light', 'room_id': 'living_room'}}},
            'action': {'device_id': 'bedroom_light', 'command_id': 'On', 'args': {}}}])
        self.assertEqual({row['kind'] for row in rows}, {
            'positive', 'hard_negative_wrong_target', 'hard_negative_opposite_action', 'hard_negative_duplicate'})
        self.assertTrue(all('judge' not in json.dumps(row).lower() for row in rows))


if __name__ == '__main__':
    unittest.main()
