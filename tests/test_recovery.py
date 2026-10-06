import unittest

from smarthome_agent_rl.recovery import RecoveryPolicy, public_context
from smarthome_agent_rl.guard import GuardError


FAIL = {'status': {'code': 400}, 'error': {'type': 'INVALID_STATE', 'detail': 'Stopped'}, 'data': {}}
OK = {'status': {'code': 200}, 'error': None, 'data': {}}


class RecoveryTests(unittest.TestCase):
    def test_default_budget_allows_initial_call_and_two_repairs(self):
        policy = RecoveryPolicy(2)
        args = {'device_id': 'd', 'args': {'x': 1}}
        for i in range(3):
            policy.check('execute_command', args, 'v1')
            policy.observe('execute_command', args, FAIL, 'v1', str(i))
        with self.assertRaises(GuardError) as blocked:
            policy.check('execute_command', args, 'v1')
        self.assertEqual(blocked.exception.layer, 'recovery_budget')
        policy.check('execute_command', args, 'v2')
        policy.check('execute_command', {**args, 'args': {'x': 2}}, 'v1')
        policy.check('execute_command', {**args, 'device_id': 'other'}, 'v1')

    def test_changed_error_context_or_success_resets_consecutive_failure_budget(self):
        policy = RecoveryPolicy(0)
        policy.observe('get_rooms', {}, FAIL, 'v1', '1')
        policy.observe('get_rooms', {}, {**FAIL, 'error': {'type': 'NEW_ERROR'}}, 'v1', '2')
        self.assertEqual(next(iter(policy.failures.values()))['count'], 1)
        policy.observe('get_rooms', {}, FAIL, 'v2', '3')
        self.assertEqual(next(iter(policy.failures.values()))['count'], 1)
        policy.observe('get_rooms', {}, OK, 'v2', '4')
        policy.check('get_rooms', {}, 'v2')

    def test_unknown_mutation_cannot_be_replayed_even_after_new_observation(self):
        policy = RecoveryPolicy()
        for tool in ('execute_command', 'schedule_workflow', 'write_attribute'):
            policy.observe(tool, {}, None, 'v1', 'a')
            with self.assertRaises(GuardError) as blocked:
                policy.check(tool, {}, 'v2')
            self.assertEqual(blocked.exception.layer, 'recovery_unknown')
        policy.check('get_workflow_status', {'workflow_id': 'w'}, 'v2')
        policy.observe('get_rooms', {}, None, 'v1', 'b')
        policy.check('get_rooms', {}, 'v1')

    def test_identical_reads_do_not_reset_context_and_other_device_is_irrelevant(self):
        def observation(device, value):
            return {'tool': 'get_device_structure', 'arguments': {'device_id': device},
                    'response': {**OK, 'data': {'value': value}}}
        args = {'device_id': 'd'}
        first = [observation('d', 1)]
        version = public_context('execute_command', args, first)
        self.assertEqual(version, public_context('execute_command', args,
                         first + [observation('other', 2), observation('d', 1)]))
        self.assertNotEqual(version, public_context('execute_command', args,
                            first + [observation('d', 2)]))
        mutation = {'tool': 'execute_command', 'arguments': args, 'response': OK}
        self.assertNotEqual(version, public_context('execute_command', args, first + [mutation]))

    def test_invalid_budget_is_rejected(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                RecoveryPolicy(value)
        for value in (2, True, 6.5):
            with self.assertRaises(ValueError):
                RecoveryPolicy(2, total_limit=value)

    def test_changing_evidence_cannot_evade_episode_failure_budget(self):
        policy = RecoveryPolicy(2, total_limit=6)
        for index in range(6):
            context = f'v{index}'
            policy.check('schedule_workflow', {'start_time': 'past'}, context)
            policy.observe('schedule_workflow', {'start_time': 'past'}, FAIL, context, str(index))
        with self.assertRaises(GuardError) as blocked:
            policy.check('schedule_workflow', {'start_time': 'past'}, 'new_evidence')
        self.assertEqual(blocked.exception.layer, 'recovery_total')
        policy.check('schedule_workflow', {'start_time': 'corrected'}, 'new_evidence')


if __name__ == '__main__':
    unittest.main()
