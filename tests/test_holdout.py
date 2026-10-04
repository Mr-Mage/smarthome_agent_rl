import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_harness_dev import freeze_holdout
from smarthome_agent_rl.benchmark import select, STRATA


class HoldoutTests(unittest.TestCase):
    def test_new_holdout_excludes_historical_ids_and_content_preserves_dev_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / 'deps/SimuHome/data/benchmark'
            benchmark.mkdir(parents=True)
            for qt, case in STRATA:
                for seed in range(1, 51):
                    (benchmark / f'{qt}_{case}_seed_{seed}.json').write_text(json.dumps({
                        'meta': {'query_type': qt, 'case': case, 'seed': seed}}))
            old = select(benchmark)
            previous = root / 'configs/benchmark-mvp'
            previous.mkdir(parents=True)
            for split in ('dev', 'final', 'smoke'):
                (previous / (split + '.json')).write_text(json.dumps({'split': split,
                    'simuhome_commit': 'source', 'tasks': old[split]}))
            old_ids = {r['id'] for split in ('dev', 'final') for r in old[split]}
            extra = next(p for p in benchmark.glob('*.json') if p.stem not in old_ids)
            historical = root / 'runs/diagnostic'
            historical.mkdir(parents=True)
            (historical / 'protocol.json').write_text(json.dumps({'schedule': [{'task': {'id': extra.stem}}]}))
            duplicate = next(p for p in benchmark.glob('*.json') if p.stem not in old_ids | {extra.stem})
            duplicate.write_bytes(extra.read_bytes())
            first = root / 'first'
            second = root / 'second'
            freeze_holdout(root, first)
            freeze_holdout(root, second)
            tasks = json.loads((first / 'final.json').read_text())['tasks']
            self.assertEqual(len(tasks), 192)
            self.assertFalse({t['id'] for t in tasks} & (old_ids | {extra.stem, duplicate.stem}))
            self.assertEqual((first / 'dev.json').read_bytes(), (previous / 'dev.json').read_bytes())
            self.assertEqual((first / 'final.json').read_bytes(), (second / 'final.json').read_bytes())
            for qt, case in STRATA:
                self.assertEqual(sum((t['query_type'], t['case']) == (qt, case) for t in tasks), 16)
            with self.assertRaises(FileExistsError):
                freeze_holdout(root, first)
