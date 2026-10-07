from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.parallel_review_services import decoder_actors


class ParallelReviewServicesTests(unittest.TestCase):
    def test_both_groups_start_together_and_cleanup_after_consumer_failure(self):
        barrier = threading.Barrier(2, timeout=3)
        alive, commands = set(), {}
        config = {'actors': [{'id': i, 'gpu': i} for i in range(4)], 'actor_extra_args': ['--existing']}

        @contextmanager
        def fake_actors(subset, directory):
            commands[directory.name] = subset
            ids = {actor['id'] for actor in subset['actors']}
            alive.update(ids)
            barrier.wait()
            try:
                yield
            finally:
                alive.difference_update(ids)

        with tempfile.TemporaryDirectory() as temporary, patch('scripts.parallel_review_services.actors', fake_actors):
            output = Path(temporary)/'services'
            with self.assertRaisesRegex(ValueError, 'consumer'):
                with decoder_actors(config, output):
                    self.assertEqual(alive, {0, 1, 2, 3})
                    raise ValueError('consumer failure')
            self.assertEqual(alive, set())
            self.assertEqual(json.loads((output/'lifecycle.json').read_text())['error']['type'], 'ValueError')
        for policy, expected in [('allow', False), ('compact', True)]:
            args = commands[policy]['actor_extra_args']
            self.assertEqual(args[0], '--existing')
            self.assertEqual(json.loads(args[-1]), {'backend': 'xgrammar', 'disable_any_whitespace': expected})
        self.assertEqual(config['actor_extra_args'], ['--existing'])

    def test_one_startup_failure_still_closes_the_successful_group(self):
        alive = set()
        config = {'actors': [{'id': i, 'gpu': i} for i in range(4)], 'actor_extra_args': []}

        @contextmanager
        def fake_actors(subset, directory):
            if directory.name == 'compact':
                raise RuntimeError('startup failure')
            alive.update(actor['id'] for actor in subset['actors'])
            try:
                yield
            finally:
                alive.clear()

        with tempfile.TemporaryDirectory() as temporary, patch('scripts.parallel_review_services.actors', fake_actors):
            with self.assertRaisesRegex(RuntimeError, 'startup failure'):
                with decoder_actors(config, Path(temporary)/'services'):
                    self.fail('Cannot run inference after a startup failure')
        self.assertEqual(alive, set())


if __name__ == '__main__':
    unittest.main()
