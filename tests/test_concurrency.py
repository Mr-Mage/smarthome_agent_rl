import copy
import json
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.concurrency import execution_slots, episode_directory, dispatch_items
from scripts.verify_benchmark import verify


class ConcurrencyTests(unittest.TestCase):
    def test_workflow_priority_preserves_pairing_and_manifest(self):
        from smarthome_agent_rl.benchmark import schedule
        rows = json.loads(Path('configs/benchmark-mvp/smoke.json').read_text())['tasks']
        original = schedule(rows, ['B0', 'G', 'Full'])
        snapshot = copy.deepcopy(original)
        ordered = dispatch_items(original, 'workflow-first')
        self.assertEqual(original, snapshot)
        self.assertEqual({id(i) for i in original}, {id(i) for i in ordered})
        self.assertEqual(ordered[0]['task']['query_type'], 'qt4-1')
        self.assertEqual(ordered[0]['task']['case'], 'feasible')
        self.assertEqual(dispatch_items(original), original)
        for actor in (0, 1):
            queue = [i for i in ordered if i['workflow'] == actor]
            self.assertTrue(all(i['task']['query_type'].startswith('qt4-') for i in queue[:6]))
        with self.assertRaises(ValueError):
            dispatch_items(original, 'unknown')

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

    def test_cross_actor_assignment_is_rejected_even_with_valid_hashes(self):
        from smarthome_agent_rl.benchmark import digest
        self.config['slots_per_actor'] = 2
        slots = execution_slots(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {'workflow': 0, 'task': {'id': 'paired'}, 'variants': ['B0', 'G'],
                'variant_workflows': {'B0': 0, 'G': 2}}
            manifest = {}
            for variant in item['variants']:
                directory = episode_directory(root, item, variant)
                directory.mkdir(parents=True)
                slot = slots[item['variant_workflows'][variant]]
                (directory / 'summary.json').write_text('{}')
                (directory / 'contract.json').write_text(json.dumps({'config': {
                    'simulator_url': f"http://127.0.0.1:{slot['simulator_port']}/api",
                    'model_endpoint': f"http://127.0.0.1:{slot['actor_port']}/v1"}}))
                for path in directory.iterdir():
                    manifest[path.relative_to(root).as_posix()] = digest(path)
            (root / 'protocol.json').write_text(json.dumps({'schedule': [item], 'expected_episodes': 2,
                'scheduler': 'actor-affine-queue-v2', 'execution_slots': slots}))
            (root / 'completion.json').write_text(json.dumps({'complete': True, 'episodes': 2}))
            (root / 'artifact_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'verification failed'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
