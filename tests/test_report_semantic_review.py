import copy
from dataclasses import asdict
import json
import unittest

from smarthome_agent_rl.report_semantic_review import ReportSemanticReviewer, attach_report_reviewer
from smarthome_agent_rl.semantic_context import digest
from smarthome_agent_rl.semantic_verifier import VerificationContext
from smarthome_agent_rl.typed_semantic_review import request
from tests import test_prompt_semantic_ablation as prompt_fixtures


def fixture_decision(context, label='NO'):
    from smarthome_agent_rl.semantic_prompt_ablation import steps
    values = steps(context['proposed_action'])
    decision = {name: {'label': label, 'probability': .99, 'evidence': {}} for name in
                ('correct_target', 'goal_consistent', 'trajectory_consistent', 'safe_to_execute')}
    decision['correct_target']['evidence'] = {'steps': [
        {'step_index': s['step_index'], 'device_id': s['device_id'],
         'support_quote': None, 'support': 'unknown'} for s in values]}
    decision['goal_consistent']['evidence'] = {'steps': [
        {'step_index': s['step_index'], 'encoded_execution_time': s['encoded_execution_time'],
         'requested_quote': None, 'relation': 'unknown'} for s in values]}
    return decision


def receipt(endpoint, body, decision):
    text = json.dumps(decision)
    usage = {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}
    response = {'model': body['model'], 'usage': usage,
                'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
    return {'endpoint': endpoint, 'body': copy.deepcopy(body), 'response': response,
            'raw_response': json.dumps(response), 'text': text, 'usage': usage,
            'error': None, 'finish_reason': 'stop', 'request_seconds': .1}


def fault_transport(kind):
    def transport(endpoint, body, timeout):
        if kind == 'timeout':
            raise TimeoutError('fixture timeout')
        context = json.loads(body['messages'][1]['content'])
        decision = fixture_decision(context, 'YES' if kind == 'allow' else 'NO')
        if kind == 'schema':
            decision['correct_target']['evidence']['steps'][0]['step_index'] = 0
        if kind == 'quote':
            decision['correct_target']['evidence']['steps'][0]['support_quote'] = 'not in original goal'
        row = receipt(endpoint, body, decision)
        if kind == 'http':
            row.update(status=503, raw_response='service unavailable',
                       error={'type': 'HTTPError', 'message': '503'})
        elif kind == 'parse':
            row['text'] = '{malformed'
        elif kind == 'length':
            row['finish_reason'] = 'length'
        elif kind == 'usage':
            row['usage'] = None
        elif kind == 'model':
            row['response']['model'] = 'different-model'
        elif kind == 'identity':
            row['body']['seed'] = 43
        return row
    return transport


class ReportSemanticReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = prompt_fixtures.PromptSemanticAblationTests('test_literal_evidence_is_not_claimed_to_prove_entailment')
        fixture.setUp()
        self.config = {**fixture.config, 'request_timeout': 240}
        self.context = VerificationContext(**fixture.item['context'])

    def reviewer(self, kind):
        return ReportSemanticReviewer(self.config, 'http://fixture/v1', transport=fault_transport(kind))

    def test_valid_denial_and_allow_keep_exact_n76_request_and_raw_receipts(self):
        for kind, expected in (('deny', 'DENY'), ('allow', 'ALLOW')):
            reviewer = self.reviewer(kind)
            result = reviewer.verify(self.context)
            row = reviewer.records[0]
            self.assertEqual(result.verdict, expected)
            self.assertTrue(row['accepted_model_output'])
            self.assertEqual(row['call']['body'], request(self.config, {'context': asdict(self.context)}))
            self.assertEqual(row['context_sha256'], digest(asdict(self.context)))
            self.assertEqual(row['call']['text'], row['call']['response']['choices'][0]['message']['content'])
            self.assertEqual(reviewer.costs()['tokens'], 5)

    def test_every_failure_returns_uncertain_without_retry_and_preserves_model_output_and_cost(self):
        reasons = {'timeout': 'TRANSPORT_FAILURE', 'http': 'TRANSPORT_FAILURE',
                   'parse': 'PARSE_FAILURE', 'length': 'INCOMPLETE_OUTPUT',
                   'schema': 'SCHEMA_FAILURE', 'quote': 'EVIDENCE_FAILURE',
                   'usage': 'USAGE_MISSING', 'model': 'MODEL_MISMATCH', 'identity': 'RECEIPT_MISMATCH'}
        for kind, reason in reasons.items():
            with self.subTest(kind=kind):
                reviewer = self.reviewer(kind)
                result = reviewer.verify(self.context)
                row = reviewer.records[0]
                self.assertEqual(result.verdict, 'UNCERTAIN')
                self.assertEqual(result.reason_code, 'REVIEW_' + reason)
                self.assertFalse(row['accepted_model_output'])
                self.assertEqual(len(reviewer.records), 1)
                self.assertEqual(reviewer.costs()['requests'], 1)
                self.assertEqual(reviewer.costs()['tokens'], 0 if kind in ('timeout', 'usage') else 5)
                if kind == 'quote':
                    self.assertEqual(row['model_decision']['verdict'], 'DENY')
                    self.assertIn('target:nonliteral_quote', row['evidence']['issues'])
                if kind == 'length':
                    self.assertIsNotNone(row['model_decision'])

    def test_mutated_hidden_context_raises_before_transport_or_fallback(self):
        reviewer = self.reviewer('deny')
        self.context.environment_state['hidden_goal'] = 'leak'
        with self.assertRaisesRegex(ValueError, 'Hidden evaluator'):
            reviewer.verify(self.context)
        self.assertEqual(reviewer.records, [])

    def test_default_and_serializable_native_policy_isolated_from_runtime_or_blocking(self):
        base = {'verify': False, 'verification_version': 1, 'context_version': 0}
        candidate = {**base, 'report_semantic_review': True, 'semantic_context_version': 2,
                     'semantic_context_references': True, 'semantic_context_workflow': True}
        config = {'variant_policies': {'G': base, 'Candidate': candidate},
                  'semantic_review': self.config, 'served_model': '9b', 'model_endpoint': 'http://fixture/v1'}
        frozen = copy.deepcopy(config)
        self.assertEqual(attach_report_reviewer(config, 'G'), (base, None))
        policy, reviewer = attach_report_reviewer(config, 'Candidate')
        self.assertIs(policy['reflection_verifier'], reviewer)
        self.assertEqual(config, frozen)
        for key, value in (('semantic_blocking', True), ('task_runtime', True),
                           ('execution_runtime', True), ('verify', True), ('semantic_context_version', 1)):
            changed = copy.deepcopy(config);changed['variant_policies']['Candidate'][key] = value
            with self.assertRaises(ValueError):
                attach_report_reviewer(changed, 'Candidate')
        changed = copy.deepcopy(config);changed['semantic_review']['model'] = 'other'
        with self.assertRaises(ValueError):
            attach_report_reviewer(changed, 'Candidate')


if __name__ == '__main__':
    unittest.main()
