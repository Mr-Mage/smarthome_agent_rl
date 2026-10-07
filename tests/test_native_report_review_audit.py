import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_native_report_review import audit_episode
from smarthome_agent_rl.report_semantic_review import attach_report_reviewer
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_context_enriched import build_enriched_context
from tests.test_report_semantic_review import fault_transport


class NativeReportReviewAuditTests(unittest.TestCase):
    def fixture(self, directory, kind='deny'):
        base = {'verify': False, 'verification_version': 1, 'context_version': 0}
        config = {'variant_policies': {'G': base, 'Candidate': {**base, 'report_semantic_review': True,
            'semantic_context_version': 2, 'semantic_context_references': True, 'semantic_context_workflow': True}},
            'semantic_review': {'model': '9b', 'model_seed': 42, 'request_timeout': 240,
                                'generation': {'temperature': 0.0, 'max_tokens': 2048}},
            'served_model': '9b', 'model_endpoint': 'http://fixture/v1'}
        action = {'tool': 'execute_command', 'device_id': 'lamp', 'command_id': 'On'}
        public = {'query': 'Switch lamp on.', 'user_location': 'living', 'current_time': '2030-01-01 12:00:00'}
        context = build_enriched_context(public['query'], action, [], user_location=public['user_location'],
            initial_time=public['current_time'], references=True, workflow_rules={'facts': {}})
        def transport(endpoint, body, timeout):
            row = fault_transport(kind)(endpoint, body, timeout)
            if row['error'] is None:
                row['response']['choices'][0]['message']['content'] = row['text']
                row['response']['choices'][0]['finish_reason'] = row['finish_reason']
                row['raw_response'] = json.dumps(row['response'])
            return row
        _, reviewer = attach_report_reviewer(config, 'Candidate', transport=transport)
        result = reviewer.verify(context)
        proposal = {'tool': 'execute_command', 'arguments': {'device_id': 'lamp', 'command_id': 'On'},
                    'actual_calls_before': 0, 'actual_calls': 1, 'reached_executor': True, 'blocked': False,
                    'semantic_context': asdict(context), 'semantic_context_sha256': digest(asdict(context)),
                    'semantic_verification': result.as_dict()}
        response = {'status': {'code': 200}, 'data': {'applied': True}}
        values = {'contract.json': {'config': config, 'public_context': public},
            'summary.json': {'variant': 'Candidate', 'task_id': 'fixture', 'actor_seed': 42, 'success': True,
                             'semantic_review': reviewer.costs(), 'actor_model_calls': 1, 'actor_tokens': 5,
                             'judge_model_calls': 0, 'judge_tokens': 0, 'duration_seconds': 1.0},
            'harness_audit.json': {'proposals': [proposal], 'actual_observations': [
                {'tool': proposal['tool'], 'arguments': proposal['arguments'], 'response': response}]},
            'model_calls.json': [{'request': {'messages': []}, 'response': {'usage':
                {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}}],
            'semantic_review_calls.json': reviewer.records,
            'phase_profile.json': {'spans': [{'kind': 'semantic_review', 'phase': 'agent', 'outcome': 'returned',
                'start_seconds': 0, 'end_seconds': .2}, {'kind': 'agent', 'start_seconds': 0, 'end_seconds': .5}]}}
        for name, value in values.items():
            (directory / name).write_text(json.dumps(value), encoding='utf-8')
        return values

    def test_original_cutoff_raw_request_results_and_all_failure_costs_recomputed(self):
        for kind in ('deny', 'length', 'quote', 'parse'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp);self.fixture(path, kind)
                result = audit_episode(path, {'facts': {}})
                self.assertEqual(result['guard_admitted_mutations'], 1)
                self.assertEqual(result['review']['requests'], 1)
                self.assertEqual(result['review']['tokens'], 5)
                self.assertEqual(result['review']['fallbacks'], 0 if kind == 'deny' else 1)

    def test_future_receipt_request_identity_and_summary_cost_tampering_rejected(self):
        for kind in ('future_context', 'request', 'summary', 'dispatch', 'raw_usage'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp);values = self.fixture(path)
                if kind == 'future_context':
                    values['semantic_review_calls.json'][0]['context']['environment_state']['observation_count'] = 1
                elif kind == 'request':
                    values['semantic_review_calls.json'][0]['call']['body']['seed'] = 43
                elif kind == 'summary':
                    values['summary.json']['semantic_review']['tokens'] = 0
                elif kind == 'dispatch':
                    values['harness_audit.json']['actual_observations'][0]['arguments']['device_id'] = 'other'
                else:
                    values['semantic_review_calls.json'][0]['call']['usage']['total_tokens'] = 6
                for name, value in values.items():
                    (path / name).write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    audit_episode(path, {'facts': {}})


if __name__ == '__main__':
    unittest.main()
