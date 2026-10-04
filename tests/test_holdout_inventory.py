import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_holdout_inventory import inventory


class InventoryTests(unittest.TestCase):
    def test_failed_round_and_config_exposure_and_exact_query_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / 'deps/SimuHome/data/benchmark'
            cases.mkdir(parents=True)
            (root / 'runs/failed').mkdir(parents=True)
            (root / 'configs').mkdir()
            for seed, query in [(101, 'Set light 5 minutes from now'), (102, 'Set light 8 minutes from now'),
                                (103, 'Set light 5 minutes from now'), (104, 'A unique action')]:
                task = {'meta': {'query_type': 'qt1', 'case': 'feasible', 'seed': seed}, 'query': query}
                (cases / f'qt1_feasible_seed_{seed}.json').write_text(json.dumps(task))
            (root / 'runs/failed/protocol.json').write_text(json.dumps({'task_id': 'qt1_feasible_seed_101'}))
            (root / 'configs/old.json').write_text(json.dumps({'tasks': ['qt1_feasible_seed_104']}))
            result = inventory(root)
            self.assertEqual(result['conservative_exposed_tasks'], 2)
            self.assertEqual(result['remaining_tasks'], 1)
            self.assertEqual(result['remaining_manifest'][0]['id'], 'qt1_feasible_seed_102')
            self.assertEqual(result['numeric_normalized_query_overlap'], 1)
            self.assertNotIn('query', result['remaining_manifest'][0])
