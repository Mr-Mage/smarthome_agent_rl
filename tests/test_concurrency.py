import copy
import json
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.concurrency import execution_slots, episode_directory
from scripts.verify_benchmark import verify


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/harness-mvp.json').read_text())

    def test_slots_isolate_simulators_but_share_actor_and_gateway(self):
        self.config['slots_per_actor'] = 16
        slots = execution_slots(self.config)
        self.assertEqual(len({s['simulator_port'] for s in slots}), 32)
        self.assertEqual(len({s['actor_port'] for s in slots}), 2)
        self.assertEqual(len({s['gateway_port'] for s in slots}), 2)
        self.assertEqual([s['actor_id'] for s in slots], [0] * 16 + [1] * 16)
        self.assertEqual(execution_slots({**self.config, 'slots_per_actor': 1})[0]['simulator_port'], 20080)

    def test_collision_is_rejected_before_starting_services(self):
        self.config.update(slots_per_actor=16, simulator_port_base=20000)
        with self.assertRaises(ValueError):
            execution_slots(self.config)

    def test_verifier_requires_the_actual_variant_slot_and_rejects_a_missing_arm(self):
        from smarthome_agent_rl.benchmark import digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {'workflow': 0, 'task': {'id': 'task'}, 'variants': ['B0', 'G'],
                'variant_workflows': {'B0': 2, 'G': 3}}
            manifest = {}
            for variant in item['variants']:
                p = episode_directory(root, item, variant) / 'summary.json'
                p.parent.mkdir(parents=True)
                p.write_text('{}')
                manifest[p.relative_to(root).as_posix()] = digest(p)
            (root / 'protocol.json').write_text(json.dumps({'schedule': [item], 'expected_episodes': 2}))
            (root / 'completion.json').write_text(json.dumps({'complete': True, 'episodes': 2}))
            (root / 'artifact_manifest.json').write_text(json.dumps(manifest))
            self.assertTrue(verify(root)['verified'])
            episode_directory(root, item, 'G').joinpath('summary.json').unlink()
            with self.assertRaises(ValueError):
                verify(root)


if __name__ == '__main__':
    unittest.main()
