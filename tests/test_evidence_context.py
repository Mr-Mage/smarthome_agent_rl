import copy
import json
import unittest

from smarthome_agent_rl.evidence_context import render, restore


class EvidenceContextTests(unittest.TestCase):
    def history(self):
        messages = [{'role': 'system', 'content': 'Keep all instructions'},
                    {'role': 'user', 'content': 'This is your actual task. Keep ALL targets.'}]
        observations = []
        for turn in range(1, 7):
            args = {'device_id': 'device'}
            response = {'status': {'code': 200}, 'data': {'capabilities': 'x' * 2500}, 'error': None}
            messages += [{'role': 'assistant', 'content': json.dumps({'thought': f'plan {turn}',
                'call': {'tool': 'get_device_structure', 'arguments': args}})},
                {'role': 'user', 'content': 'observation: ' + json.dumps(response)}]
            observations.append({'turn': turn, 'tool': 'get_device_structure', 'arguments': args,
                                 'response': response, 'extra_query': False})
        return messages, observations

    def test_exact_roundtrip_keeps_plans_first_evidence_recent_pairs(self):
        raw, observed = self.history()
        before = copy.deepcopy(raw)
        managed, refs = render(raw, observed)
        self.assertEqual(restore(managed), raw)
        self.assertEqual(raw, before)
        self.assertEqual(managed[:4], raw[:4])
        self.assertEqual(managed[-4:], raw[-4:])
        self.assertEqual(len(refs), 3)
        for i, m in enumerate(raw):
            if m['role'] == 'assistant':
                self.assertEqual(managed[i], m)

    def test_changed_versions_errors_and_mutation_receipts_are_verbatim(self):
        for kind in ('changed', 'failed_read', 'write'):
            raw, observed = self.history()
            if kind == 'changed':
                observed[1]['response']['data']['capabilities'] = 'y' * 2500
            elif kind == 'failed_read':
                observed[1]['response'] = {'status': {'code': 400}, 'error': {'detail': 'fix parameters'}}
            else:
                observed[1]['tool'] = 'write_attribute'
                body = json.loads(raw[4]['content'])
                body['call']['tool'] = 'write_attribute'
                raw[4]['content'] = json.dumps(body)
            raw[5]['content'] = 'observation: ' + json.dumps(observed[1]['response'])
            managed, refs = render(raw, observed)
            self.assertEqual(managed[5], raw[5], kind)
            self.assertEqual(restore(managed), raw)
            if kind == 'write':
                self.assertEqual(managed[7], raw[7])  # first full read after mutation

    def test_missing_auxiliary_or_wrong_argument_provenance_cannot_compress(self):
        raw, observed = self.history()
        self.assertEqual(render(raw, [])[0], raw)
        for o in observed:
            o['extra_query'] = True
        self.assertEqual(render(raw, observed)[0], raw)
        for o in observed:
            o['extra_query'] = False
            o['arguments'] = {'device_id': 'different'}
        self.assertEqual(render(raw, observed)[0], raw)

    def test_failed_or_unknown_mutation_breaks_reference_epoch(self):
        raw, observed = self.history()
        raw[4]['content'] = json.dumps({'action': 'remove_device', 'action_input': '{"device_id":"device"}'})
        observed[1]['tool'] = 'remove_device'
        observed[1]['response'] = {'status': {'code': 500}, 'error': 'unknown result'}
        raw[5]['content'] = 'observation: ' + json.dumps(observed[1]['response'])
        managed, refs = render(raw, observed)
        self.assertEqual(managed[5:8], raw[5:8])
        self.assertEqual(restore(managed), raw)

    def test_reference_hash_corruption_is_detected(self):
        raw, observed = self.history()
        managed, refs = render(raw, observed)
        managed[3]['content'] += 'corrupted'
        with self.assertRaises(ValueError):
            restore(managed)

    def test_small_messages_and_ambiguous_task_boundary_pass_through(self):
        raw, observed = self.history()
        for o in observed:
            o['response']['data'] = {}
        for i in range(3, len(raw), 2):
            raw[i]['content'] = 'observation: ' + json.dumps(observed[(i-3)//2]['response'])
        self.assertEqual(render(raw, observed)[0], raw)
        raw.append({'role': 'user', 'content': 'This is your actual task. Ambiguous boundary'})
        self.assertEqual(render(raw, observed)[0], raw)

    def test_same_response_different_parameters_never_share_source(self):
        raw, observed = self.history()
        observed[1]['arguments'] = {'device_id': 'other-device'}
        body = json.loads(raw[4]['content'])
        body['call']['arguments'] = observed[1]['arguments']
        raw[4]['content'] = json.dumps(body)
        managed, refs = render(raw, observed)
        self.assertEqual(managed[5], raw[5])
        self.assertEqual(restore(managed), raw)

    def test_provider_uses_real_token_gate_and_preserves_schema(self):
        # This integration check runs in the existing SimuHome environment.
        try:
            from src.agents.types import ChatMessage
        except ModuleNotFoundError:
            self.skipTest('Existing SimuHome environment required')
        from types import SimpleNamespace
        from smarthome_agent_rl.evidence_context import EvidenceContextProvider
        from smarthome_agent_rl.harness_agent import HarnessAgent
        class Inner:
            def generate(self, messages, response_format=None):
                self.messages, self.schema = messages, response_format
                return 'unchanged reply'
        raw, observed = self.history()
        messages = [ChatMessage(**m) for m in raw]
        executor = SimpleNamespace(observations=observed, context_audit=[], save_audit=lambda: None)
        inner = Inner()
        for variant in ('GTS', 'GR'):
            with self.assertRaises(ValueError):
                HarnessAgent(inner, variant=variant, max_steps=20,
                    policy={'verify':False, 'verification_version':1, 'context_version':0,
                            'evidence_context':True}, token_count_fn=lambda rows:100)
        with self.assertRaises(ValueError):
            HarnessAgent(inner, variant='GEC', max_steps=20)
        baseline = HarnessAgent(inner, variant='G', max_steps=20)
        self.assertIs(baseline.agent.llm.inner, inner)
        schema = {'json_schema': {'name': 'original-schema'}}
        provider = EvidenceContextProvider(inner, executor, lambda rows: 100)
        self.assertEqual(provider.generate(messages, schema), 'unchanged reply')
        self.assertFalse(executor.context_audit[-1]['used'])
        self.assertEqual(inner.messages, messages)
        self.assertIs(inner.schema, schema)
        provider.token_count_fn = lambda rows: sum(len(m.content) for m in rows)
        provider.generate(messages, schema)
        self.assertTrue(executor.context_audit[-1]['used'])
        self.assertEqual(restore([{'role':m.role,'content':m.content} for m in inner.messages]), raw)
        self.assertIs(inner.schema, schema)


if __name__ == '__main__':
    unittest.main()
