import copy
import hashlib
import json
import unittest

from scripts.audit_native_resource_context import PREFIX, audit_resources
from tests import test_resource_episode_runtime as fixtures


class NativeResourceAuditTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ResourceEpisodeTests('test_claims_conflicts_and_context_are_durable_without_extra_reads_or_arbitration')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.f = fixture.f
        self.f.register()
        self.f.arguments['steps'][0]['args']['command_id'] = 'Off'
        self.f.register()
        context = self.f.runtime.actor_context()
        content = PREFIX + json.dumps(context, ensure_ascii=False, sort_keys=True)
        self.f.runtime._event('runtime_context_exposed', context=context,
                            content_sha256=hashlib.sha256(content.encode()).hexdigest())
        self.data = copy.deepcopy(self.f.receipts[-1][1])
        self.calls = [{'request': {'messages': [{'role': 'user', 'content': content}]}}]

    def audit(self):
        return audit_resources(self.data, self.calls, self.f.runtime.adapter)

    def test_actual_registration_provenance_and_http_context_match(self):
        result = self.audit()
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['fixed_claims'], 2)
        self.assertEqual(result['context_requests'], 1)
        self.assertEqual(result['contexts_with_conflicts'], 1)

    def test_future_schema_cannot_replace_missing_prior_public_observation(self):
        first = min(self.data['trace'], key=lambda r: r['timestamp'])
        first['response']['data']['device_id'] = 'other'
        self.assertTrue(any('preceding public' in p for p in self.audit()['problems']))

    def test_changed_target_or_missing_actor_disclosure_is_detected(self):
        self.data['jobs'][0]['resource_claims'][0]['value'] = 'forged'
        self.calls = []
        problems = self.audit()['problems']
        self.assertTrue(any('Durable claims' in p for p in problems))
        self.assertTrue(any('actual actor HTTP' in p for p in problems))

    def test_extra_context_field_and_false_completion_are_detected_even_with_matching_hash(self):
        event = self.data['events'][-1]
        event['context'].update(task_status='COMPLETED', official_score=1)
        content = PREFIX + json.dumps(event['context'], ensure_ascii=False, sort_keys=True)
        event['content_sha256'] = hashlib.sha256(content.encode()).hexdigest()
        self.calls[0]['request']['messages'][0]['content'] = content
        self.assertTrue(any('extra fields' in p for p in self.audit()['problems']))


if __name__ == '__main__':
    unittest.main()
