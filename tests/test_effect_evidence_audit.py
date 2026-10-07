import copy
import json
import unittest

from scripts.audit_effect_evidence_ablation import verify_record
from smarthome_agent_rl.effect_evidence_ablation import request, evaluate, ARMS
from tests import test_effect_evidence_ablation as fixtures


class EffectEvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EffectEvidenceAblationTests(); self.fixture.setUp()
        self.config = self.fixture.config; self.item = self.fixture.item
        self.actor = {'id': 0, 'endpoint': 'http://127.0.0.1:20400/v1'}
        decision = self.fixture.decision()
        text = json.dumps({k: v for k, v in decision.items() if k not in ('verdict', 'reason_code')})
        raw = {'model': '9b', 'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
               'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}
        self.row = {'id': self.item['id'], 'arm': 'combined', 'actor_id': 0, 'decision': decision, 'parse_error': None,
                    'call': {'endpoint': self.actor['endpoint'], 'body': request(self.config, self.item, 'combined'),
                             'error': None, 'raw_response': json.dumps(raw), 'response': raw, 'text': text,
                             'finish_reason': 'stop', 'usage': raw['usage'], 'request_seconds': .1}}

    def verify(self, row):
        verify_record(row, self.item, 'combined', self.actor, self.config)

    def test_raw_http_usage_finish_and_verdict_cannot_be_rewritten(self):
        self.verify(self.row)
        for location, key, value in [('call', 'usage', {'total_tokens': 0}), ('call', 'finish_reason', 'length'),
                                     ('decision', 'verdict', 'DENY')]:
            changed = copy.deepcopy(self.row); changed[location][key] = value
            with self.assertRaises(ValueError): self.verify(changed)

    def test_hidden_context_actor_and_prompt_changes_rejected(self):
        for modify in (lambda r: r.update(actor_id=1),
                       lambda r: r['call']['body']['messages'][0].update(content='Changed rule'),
                       lambda r: r['call']['body']['messages'][1].update(content='Injected outcome')):
            changed = copy.deepcopy(self.row); modify(changed)
            with self.assertRaises(ValueError): self.verify(changed)

    def test_failure_receipt_is_retained_and_cannot_claim_a_decision(self):
        row = copy.deepcopy(self.row); row['call']['error'] = 'timeout'
        row.update(decision=None, parse_error=None); self.verify(row)
        row['decision'] = self.row['decision']
        with self.assertRaises(ValueError): self.verify(row)

    def test_duplicate_and_missing_requests_cannot_pass_screen(self):
        config = {**self.config, 'records': 1, 'new_requests': 3, 'valid_ratio_min': .95,
                  'evidence_valid_ratio_min': .95, 'developer_counts': {'CONSISTENT_CONTROL': 1},
                  'developer_conflict_groups': {'target': [], 'time': []}}
        inputs = [{**self.item, 'origin': 'n76', 'decision': self.fixture.decision('coherence'), 'call': self.row['call']}]
        records = [{**self.row, 'arm': arm, 'decision': self.fixture.decision(arm)} for arm in ARMS]
        labels = [{'id': self.item['id'], 'category': 'CONSISTENT_CONTROL'}]
        result = evaluate(config, inputs, records+[records[0]], labels)
        self.assertFalse(result['complete_records']); self.assertFalse(any(result['diagnostic_screen_passed'].values()))
        self.assertEqual(result['new_tokens'], 20)
        with self.assertRaises(ValueError): evaluate(config, inputs, records, labels+labels)


if __name__ == '__main__': unittest.main()
