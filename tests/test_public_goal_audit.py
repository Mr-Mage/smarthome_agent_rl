import copy
import json
import unittest

from scripts.audit_public_goal_diagnosis import audit_record
from scripts.run_public_goal_diagnosis import request
from smarthome_agent_rl.execution.goals import extraction_schema, parse_proposal, public_messages


class PublicGoalAuditTests(unittest.TestCase):
    def setUp(self):
        self.config = {'model': 'extractor', 'generation': {'temperature': 0, 'max_tokens': 128}}
        self.actor = {'id': 0, 'endpoint': 'http://local/v1'}
        self.item = {'task': {'id': 'public'}, 'public': {'query': 'Turn on the lamp.', 'current_time': None}}
        # An invalid raw output must remain invalid, even if manually repaired.
        text = '{}'
        body = request(self.config, public_messages('Turn on the lamp.', None), extraction_schema(), seed=42)
        response = {'model': 'extractor', 'choices': [{'message': {'content': text}}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}
        self.row = {'task_id': 'public', 'seed': 42, 'actor_id': 0, 'proposal': None, 'review': None,
            'calls': [{'body': body, 'endpoint': self.actor['endpoint'], 'error': None,
                'text': text, 'response': response, 'raw_response': json.dumps(response), 'usage': response['usage']}]}

    def audit(self, row):
        return audit_record(self.config, self.item, row, seed=42, actor=self.actor)

    def test_invalid_raw_output_is_retained_and_unexpected_repaired_proposal_rejected(self):
        self.assertEqual(self.audit(self.row), [])
        row = copy.deepcopy(self.row)
        row['proposal'] = {'completes_user_task': True}
        self.assertTrue(any('raw model output' in p for p in self.audit(row)))

    def test_forged_usage_or_private_request_fields_are_detected(self):
        row = copy.deepcopy(self.row)
        row['calls'][0]['usage']['total_tokens'] = 100
        self.assertTrue(any('raw HTTP receipt' in p for p in self.audit(row)))
        row = copy.deepcopy(self.row)
        row['calls'][0]['body']['messages'][-1]['content'] = '{"official_score":1}'
        self.assertTrue(any('frozen public input' in p for p in self.audit(row)))


if __name__ == '__main__':
    unittest.main()
