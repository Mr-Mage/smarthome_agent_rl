import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smarthome_agent_rl.concurrency import execution_slots, external_judge, judge_endpoint
from smarthome_agent_rl.experiment_v2 import service_identity
from scripts.harness_services import validate_resources, probe


class RemoteJudgeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/remote-judge-capacity.json').read_text())

    def test_four_actors_have_128_independent_simulators_and_no_judge_gpu(self):
        slots = execution_slots(self.config)
        self.assertEqual(len(slots), 128)
        self.assertEqual(len({s['simulator_port'] for s in slots}), 128)
        self.assertEqual(len({s['actor_port'] for s in slots}), 4)
        with patch('scripts.harness_services.socket.socket') as socket:
            validate_resources(self.config)
        bound = [call.args[0][1] for call in socket.return_value.__enter__.return_value.bind.call_args_list]
        self.assertNotIn(self.config['judge_port'], bound)

    def test_remote_port_may_equal_a_local_port_but_local_overlap_is_rejected(self):
        self.config['judge_port'] = self.config['workflows'][0]['actor_port']
        self.assertEqual(len(execution_slots(self.config)), 128)
        self.config['embedding_port'] = self.config['workflows'][0]['actor_port']
        with self.assertRaises(ValueError):
            execution_slots(self.config)

    def test_external_endpoint_and_gpu_reservation_are_validated(self):
        for endpoint in ('', 'file:///v1', 'http://host', 'http://key:secret@host/v1', 'http://host/v1?x=1'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                external_judge({**self.config, 'judge_endpoint': endpoint})
        with self.assertRaises(ValueError):
            external_judge({**self.config, 'judge_gpus': [3]})
        with self.assertRaises(ValueError):
            external_judge({**self.config, 'judge_deployment': 'typo'})

    def test_legacy_endpoint_and_identity_are_preserved_and_remote_changes_detected(self):
        old = json.loads(Path('configs/harness-v2-modules.json').read_text())
        self.assertFalse(external_judge(old))
        self.assertEqual(judge_endpoint(old), 'http://127.0.0.1:20300/v1')
        changed = copy.deepcopy(self.config)
        changed['judge_endpoint'] = 'http://other-host:20300/v1'
        self.assertNotEqual(service_identity(changed), service_identity(self.config))

    def test_three_vote_probes_go_to_external_endpoint(self):
        class Client:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def post(inner, url, json):
                calls.append((url, json))
                content = 'READY' if json['model'] == self.config['actor_model'] else (
                    'A' if 'answer 4.' in json['messages'][0]['content'] else 'B')
                class Response:
                    def raise_for_status(self):
                        pass
                    def json(self):
                        return {'choices': [{'message': {'content': content}}], 'usage': {'total_tokens': 2}}
                return Response()
        calls = []
        with tempfile.TemporaryDirectory() as temp, patch('scripts.harness_services.httpx.Client', Client):
            probe(self.config, Path(temp))
        remote = [body for url, body in calls if url == judge_endpoint(self.config) + '/chat/completions']
        self.assertEqual(len(remote), 6)
        self.assertEqual({body['seed'] for body in remote}, {42, 43, 44})


if __name__ == '__main__':
    unittest.main()
