import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from smarthome_agent_rl.benchmarks.runner import run, select_tasks
from scripts.verify_public_benchmark import verify
from tests import test_homebench_adapter as fixtures


class PublicRunnerTests(unittest.TestCase):
    def setUp(self):
        fixtures.HomeBenchAdapterTests.setUp(self)
        self.adapter._cases['case1']['output'] = "'''error_input'''"
        self.case['output'] = "'''error_input'''"
        case_file = self.root/'dataset/test_data.jsonl'
        case_file.write_text(json.dumps(self.case)+'\n', encoding='utf-8')
        self.files['dataset/test_data.jsonl']['sha256'] = hashlib.sha256(case_file.read_bytes()).hexdigest()
        self.config = {'model': 'actor', 'model_seed': 42, 'generation': {'temperature': 0.7,
                       'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}},
                       'actors': [{'id': i, 'endpoint': 'http://actor'+str(i)+'/v1'} for i in range(4)],
                       'slots_per_actor': 2, 'request_timeout': 1, 'max_guard_uncovered_rate': 0.25}
        self.calls = []
        self.lock = threading.Lock()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)/'experiment'

    def request(self, endpoint, body, timeout):
        with self.lock:
            self.calls.append((endpoint, body))
        return {'endpoint': endpoint, 'body': body, 'text': '{garage.light.turn_on()}', 'error': None,
                'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
                'finish_reason': 'stop', 'request_seconds': 0.01}

    def test_b2_shares_exact_b1_request_and_cost_is_counted_once(self):
        report = run(self.adapter, self.config, self.output, {'kind': 'full'}, self.request)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(report['cost']['actual_actor_requests'], 2)
        self.assertEqual(report['cost']['total_tokens'], 28)
        rows = [json.loads(line) for line in (self.output/'episodes.jsonl').read_text().splitlines()]
        b1, b2 = [next(r for r in rows if r['arm'] == a) for a in ('B1', 'B2')]
        self.assertEqual(b1['request_receipt'], b2['request_receipt'])
        self.assertEqual(b1['input_sha256'], b2['input_sha256'])
        self.assertEqual(report['paired_changes']['B1_B2']['wins'], 1)
        self.assertTrue(report['engineering_gate']['passed'])
        for endpoint, body in self.calls:
            self.assertNotIn('FORBIDDEN_CATEGORY', json.dumps(body))
            self.assertNotIn('extra_body', body)
            self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
        with self.assertRaises(FileExistsError):
            run(self.adapter, self.config, self.output, {'kind': 'full'}, self.request)

    def test_transport_failure_is_kept_in_denominator_and_blocks_gate(self):
        def failed(*args):
            return {'text': '', 'error': {'type': 'TimeoutError'}, 'usage': None,
                    'finish_reason': None, 'request_seconds': 1}
        self.adapter._cases['case1']['output'] = ''
        report = run(self.adapter, self.config, self.output, {'kind': 'full'}, failed)
        self.assertFalse(report['engineering_gate']['passed'])
        self.assertEqual(report['cost']['missing_usage_requests'], 2)
        for arm in report['arms'].values():
            self.assertEqual(arm['episodes'], 1)
            self.assertEqual(arm['exact_match'], 0)

    def test_frozen_selection_is_order_independent_and_home_affinity_limits_concurrency(self):
        original = self.adapter._cases['case1']
        for i in range(12):
            self.adapter._cases['case'+str(i+2)] = {**copy.deepcopy(original), 'id': 'case'+str(i+2)}
        selection = {'kind': 'category_hash', 'per_category': 8, 'salt': 'fixed'}
        chosen = select_tasks(self.adapter, selection)
        self.adapter._cases = dict(reversed(list(self.adapter._cases.items())))
        self.assertEqual(select_tasks(self.adapter, selection), chosen)
        active = peak = 0
        def measured(endpoint, body, timeout):
            nonlocal active, peak
            with self.lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            result = self.request(endpoint, body, timeout)
            with self.lock:
                active -= 1
            return result
        report = run(self.adapter, self.config, self.output, selection, measured)
        self.assertEqual(peak, 2)
        self.assertEqual(len({endpoint for endpoint, body in self.calls}), 1)
        self.assertEqual(report['cost']['actual_actor_requests'], 16)

    def test_invalid_usage_is_not_silently_counted_as_zero(self):
        def invalid(*args):
            value = self.request(*args)
            value['usage']['total_tokens'] = 999
            return value
        report = run(self.adapter, self.config, self.output, {'kind': 'full'}, invalid)
        self.assertEqual(report['cost']['missing_usage_requests'], 2)
        self.assertFalse(report['engineering_gate']['passed'])

    def test_evidence_verifier_rejects_answer_injected_request(self):
        run(self.adapter, self.config, self.output, {'kind': 'full'}, self.request)
        lock = {'files': self.files, 'commit': 'fixture'}
        self.assertTrue(verify(self.output, self.root, lock)['verified'])
        path = next((self.output/'requests').glob('*.json'))
        row = json.loads(path.read_text())
        row['body']['messages'].append({'role': 'system', 'content': 'gold answer: error_input'})
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(AssertionError, 'leakage'):
            verify(self.output, self.root, lock)

    def test_evidence_verifier_rejects_dropped_failure_or_modified_score(self):
        run(self.adapter, self.config, self.output, {'kind': 'full'}, self.request)
        path = self.output/'episodes.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['score']['expected'] += 1
        path.write_text('\n'.join(json.dumps(r) for r in rows))
        with self.assertRaisesRegex(AssertionError, 'score differs'):
            verify(self.output, self.root, {'files': self.files, 'commit': 'fixture'})


if __name__ == '__main__':
    unittest.main()
