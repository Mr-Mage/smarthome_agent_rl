import copy
import unittest

from smarthome_agent_rl.execution.simuhome_contract import SimuHomeContractAdapter
from smarthome_agent_rl.execution.mutation import MutationExecutor, VerificationStatus
from smarthome_agent_rl.execution.trace import ToolTrace


class RuntimeContractAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = SimuHomeContractAdapter({('LevelControl', 'MoveToLevel'): {
            'properties': {'Level': {'type': None}}, 'required': ['Level']}},
            {'new-lamp': {'commands': [], 'attributes': {}}})
        self.state = {'device_type': 'new-lamp', 'endpoints': {'1': {'clusters': {
            'LevelControl': {'commands': ['MoveToLevel'], 'attributes': {'CurrentLevel': {'value': 20}}}}}}}
        self.args = {'device_id': 'lamp2', 'endpoint_id': 1, 'cluster_id': 'LevelControl',
                     'command_id': 'MoveToLevel', 'args': {'Level': 40}}

    def test_public_contract_verifies_new_device_without_type_branch(self):
        contract, name, args = self.adapter.build('execute_command', self.args, self.state)
        after = copy.deepcopy(self.state)
        after['endpoints']['1']['clusters']['LevelControl']['attributes']['CurrentLevel']['value'] = 40
        trace = ToolTrace(lambda tool, a: {'status': {'code': 200}, 'data': after if tool.startswith('get_') else {}})
        result = MutationExecutor(trace).execute('execute_command', self.args, contract=contract,
            state=self.state, function_name=name, contract_args=args)
        self.assertEqual(result.status, VerificationStatus.VERIFIED_SUCCESS)

    def test_range_schema_and_endpoint_fail_without_dispatch(self):
        for change in ({'args': {'Level': 999}}, {'endpoint_id': 2}, {'args': {'bad_key': 10}}):
            arguments = {**self.args, **change}
            contract, name, args = self.adapter.build('execute_command', arguments, self.state)
            result = contract.validate({'function': name, 'args': args}, self.state)
            self.assertFalse(result.accepted)

    def test_readonly_attribute_cannot_be_written(self):
        self.state['endpoints']['1']['clusters']['LevelControl']['attributes']['CurrentLevel']['readonly'] = True
        arguments = {**self.args, 'attribute_id': 'CurrentLevel', 'value': 40}
        contract, name, args = self.adapter.build('write_attribute', arguments, self.state)
        self.assertEqual(contract.validate({'function': name, 'args': args}, self.state).reason_code, 'READONLY_FUNCTION')

    def test_unknown_signature_is_explicitly_uncovered(self):
        self.state['endpoints']['1']['clusters']['LevelControl']['commands'].append('NewCommand')
        arguments = {**self.args, 'command_id': 'NewCommand', 'args': {}}
        contract, name, args = self.adapter.build('execute_command', arguments, self.state)
        self.assertEqual(contract.validate({'function': name, 'args': args}, self.state).uncovered[0]['path'],
                         '__uncovered.command_signature')


if __name__ == '__main__':
    unittest.main()
