import copy
import json
from types import SimpleNamespace
import unittest

from src.agents.types import ChatMessage
from src.simulator.domain.clusters.fan_control import FanControlCluster
from src.simulator.domain.clusters.operational_state import OperationalStateCluster
from src.simulator.domain.clusters.laundry_washer_mode import LaundryWasherModeCluster
from smarthome_agent_rl.verification import expected_effect_v2, verify_effect
from smarthome_agent_rl.harness_agent import GuardedExecutor
from smarthome_agent_rl.context import compact_ledger, CompactLedgerProvider
from smarthome_agent_rl.concurrency import seeded_schedule, episode_directory


def structure(cluster):
    return {'device_id': 'device', 'endpoints': {'1': {'clusters': {cluster.cluster_id: cluster.get_structure()}}}}


def action(cluster, command, args):
    return {'device_id': 'device', 'endpoint_id': 1, 'cluster_id': cluster, 'command_id': command, 'args': args}


class HarnessV2Tests(unittest.TestCase):
    def test_fan_steps_match_real_public_cluster_at_boundaries_and_leave_snapshot_untouched(self):
        for sequence in range(6):
            for current in (0, 17, 50, 100):
                for direction in (0, 1):
                    for wrap in (False, True):
                        for lowest in (False, True):
                            cluster = FanControlCluster(fan_mode_sequence=sequence)
                            cluster.attributes['PercentSetting'] = current
                            before = structure(cluster)
                            snapshot = copy.deepcopy(before)
                            args = {'Direction': direction, 'Wrap': wrap, 'LowestOff': lowest}
                            expected = expected_effect_v2('execute_command', action('FanControl', 'Step', args), before)
                            self.assertTrue(cluster._step(**args).success)
                            self.assertTrue(verify_effect(expected, structure(cluster), {'data': {}})['verified'])
                            self.assertEqual(before, snapshot)

    def test_operational_start_and_washer_mode_match_real_commands(self):
        operational = OperationalStateCluster()
        before = structure(operational)
        effect = expected_effect_v2('execute_command', action('OperationalState', 'Start', {}), before)
        self.assertTrue(operational._start().success)
        self.assertTrue(verify_effect(effect, structure(operational), {'data': {}})['verified'])
        washer = LaundryWasherModeCluster()
        before = structure(washer)
        effect = expected_effect_v2('execute_command', action('LaundryWasherMode', 'ChangeToMode', {'new_mode': 4}), before)
        self.assertTrue(washer._change_to_mode(4).success)
        self.assertTrue(verify_effect(effect, structure(washer), {'data': {}})['verified'])

    def test_uncovered_command_avoids_postquery_and_workflow_receipt_claims_only_registration(self):
        cluster = OperationalStateCluster()
        calls = []
        def dispatch(tool, arguments):
            calls.append(tool)
            return {'status': {'code': 200}, 'error': None, 'data': structure(cluster) if tool == 'get_device_structure' else {'workflow_id': 'workflow'}}
        executor = GuardedExecutor(verify=True, verification_version=2, dispatch=dispatch)
        reply = executor.execute('execute_command', action('OperationalState', 'Stop', {}))
        self.assertEqual(reply['harness_verification']['status'], 'uncovered')
        self.assertEqual(calls.count('get_device_structure'), 1)
        calls.clear()
        reply = executor.execute('schedule_workflow', {'start_time': '2030-01-01 12:00:00', 'steps': [
            {'tool': 'execute_command', 'args': action('OperationalState', 'Start', {})}]})
        self.assertNotIn('get_workflow_status', calls)
        self.assertFalse(reply['harness_verification']['future_success_verified'])

    def test_mismatch_is_feedback_but_actual_simulator_return_remains_unchanged(self):
        cluster = OperationalStateCluster()
        def dispatch(tool, arguments):
            return {'status': {'code': 200}, 'error': None, 'data': structure(cluster) if tool == 'get_device_structure' else {}}
        executor = GuardedExecutor(verify=True, verification_version=2, dispatch=dispatch)
        reply = executor.execute('execute_command', action('OperationalState', 'Start', {}))
        self.assertEqual(reply['error']['type'], 'harness_postcondition')
        self.assertIsNone(executor.observations[1]['response']['error'])
        self.assertFalse(executor.audit[-1]['simulator_error'])
        self.assertTrue(verify_effect({(1, 'OperationalState', 'OperationalState'): 1}, {}, {'data': {'duration': 1}})['verified'] is None)

    def test_compact_directory_invalidates_after_catalog_mutation_and_preserves_error(self):
        observations = [{'turn': 1, 'tool': 'get_room_devices', 'arguments': {'room_id': 'room'}, 'extra_query': False,
            'response': {'status': {'code': 200}, 'data': {'devices': ['device']}, 'error': None}},
            {'turn': 2, 'tool': 'remove_device', 'arguments': {'device_id': 'device'}, 'extra_query': False,
            'response': {'status': {'code': 200}, 'data': {}, 'error': None}}]
        executor = SimpleNamespace(observations=observations, audit=[{'turn': 3, 'tool': 'execute_command',
            'arguments': {}, 'blocked': True, 'response': {'error': 'repair this'}}], structured_audit=[], context_audit=[])
        ledger = compact_ledger(executor)
        self.assertTrue(ledger['facts'][0]['stale'])
        self.assertEqual(ledger['errors'][0]['error'], {'error': 'repair this'})

    def test_compact_prompt_keeps_instruction_recent_pairs_and_respects_token_counter(self):
        class Inner:
            def generate(self, messages, response_format=None):
                self.messages = messages
                return 'reply'
        inner = Inner()
        executor = SimpleNamespace(observations=[], audit=[], structured_audit=[], context_audit=[])
        task = ChatMessage(role='user', content='This is your actual task. Original task text')
        history = [m for _ in range(6) for m in (ChatMessage(role='assistant', content='thought ' * 300),
            ChatMessage(role='user', content='observation: ' + 'public data ' * 300))]
        messages = [ChatMessage(role='system', content='all tools'), task, *history]
        provider = CompactLedgerProvider(inner, executor, token_count_fn=lambda rows: sum(len(m.content) for m in rows))
        provider.generate(messages)
        self.assertTrue(executor.context_audit[-1]['used'])
        self.assertIn(task, inner.messages)
        self.assertEqual(inner.messages[-4:], history[-4:])
        provider.token_count_fn = lambda rows: 100
        provider.generate(messages)
        self.assertFalse(executor.context_audit[-1]['used'])
        self.assertEqual(inner.messages, messages)

    def test_repeated_seeds_share_pairing_but_never_share_artifact_paths(self):
        items = seeded_schedule([{'id': 'task'}], ['G', 'GV2'], [42, 43, 44])
        paths = [episode_directory('/run', item, 'G') for item in items]
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual([i['workflow'] for i in items], [0, 0, 0])
        self.assertEqual(items[1]['variants'], ['GV2', 'G'])
        with self.assertRaises(ValueError):
            seeded_schedule([], ['G'], [42, 42])
