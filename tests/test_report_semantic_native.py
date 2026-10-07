"""Scripted native ReAct failure isolation, not generated benchmark data."""
import copy
import json
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.report_semantic_review import ReportSemanticReviewer
from tests.test_report_semantic_review import fault_transport
from tests import test_semantic_context_native as fixtures


@unittest.skipIf(fixtures.GuardedExecutor is None, 'requires existing SimuHome server environment')
class ReportSemanticNativeTests(unittest.TestCase):
    def test_native_failures_and_deny_preserve_original_actor_inputs_dispatch_and_observations(self):
        fixture = fixtures.NativePublicSemanticContextTests('test_native_react_supplies_original_public_inputs_and_no_extra_tool_calls')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        config = {'model': '9b', 'model_seed': 42, 'request_timeout': 240,
                  'generation': {'temperature': 0.0, 'max_tokens': 2048}}
        baseline = None
        self.native_evidence = []
        for kind in ('baseline', 'allow', 'deny', 'timeout', 'http', 'parse', 'length', 'schema', 'quote'):
            fixture.calls.clear()
            captured = []
            outputs = iter([
                {'thought': 'Schedule.', 'call': {'tool': 'schedule_workflow', 'arguments': fixture.workflow}},
                {'thought': 'Registered.', 'call': {'tool': 'finish', 'arguments': {'answer': 'Registered.'}}}])
            def generate(messages, **kwargs):
                captured.append([{'role': m.role, 'content': m.content} for m in messages])
                return json.dumps(next(outputs))
            policy = {'verify': False, 'verification_version': 1, 'context_version': 0}
            reviewer = None
            if kind != 'baseline':
                reviewer = ReportSemanticReviewer(config, 'http://fixture/v1', transport=fault_transport(kind))
                policy.update(reflection_verifier=reviewer, semantic_context_version=2,
                              semantic_context_references=True, semantic_context_workflow=True)
            agent = fixtures.HarnessAgent(SimpleNamespace(generate=generate), variant='G', max_steps=3, policy=policy)
            agent.executor.dispatch = fixture.dispatch
            result = agent.run('Switch lamps at 12:01.', user_location='living', current_time='2030-01-01 12:00:00')
            self.assertEqual(result.final_answer, 'Registered.')
            self.assertFalse(agent.executor.audit[-1]['blocked'])
            self.assertEqual([name for name, _ in fixture.calls], ['get_device_structure', 'schedule_workflow'])
            observed = {'actor_inputs': captured, 'calls': copy.deepcopy(fixture.calls),
                        'observations': [r['response'] for r in agent.executor.observations]}
            if baseline is None:
                baseline = observed
            else:
                self.assertEqual(observed, baseline)
                self.assertEqual(len(reviewer.records), 1)
                self.assertEqual(reviewer.records[0]['result']['verdict'],
                                 'ALLOW' if kind == 'allow' else 'DENY' if kind == 'deny' else 'UNCERTAIN')
            self.native_evidence.append({'kind': kind, **observed,
                                         'review_records': reviewer.records if reviewer else []})

    def test_report_adapter_rejects_blocking_even_when_installed_as_inactive_verifier(self):
        reviewer = SimpleNamespace(report_only=True)
        for key in ('semantic_verifier', 'reflection_verifier'):
            with self.assertRaisesRegex(ValueError, 'Report-only'):
                fixtures.GuardedExecutor(semantic_blocking=True, **{key: reviewer})


if __name__ == '__main__':
    unittest.main()
