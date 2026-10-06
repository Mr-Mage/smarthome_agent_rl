import unittest

from smarthome_agent_rl.device_contract import ContractRegistry, contract_from_structure


class DeviceContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = ContractRegistry.from_dict({
            'lamp-v2': {'capabilities': ['power', 'dimming'], 'functions': {
                'On': {'capability': 'power'},
                'MoveToLevel': {'capability': 'dimming', 'args': {
                    'Level': {'type': 'integer', 'min': 1, 'max': 254}}},
                'SetTemperature': {'capability': 'temperature', 'assertions': [
                    {'path': 'power', 'op': 'eq', 'value': True}]},
            }},
        })

    def test_new_device_is_validated_without_guard_branch(self):
        result = self.registry.validate({'function': 'MoveToLevel', 'args': {'Level': 100}},
                                        {'device_type': 'lamp-v2', 'power': True})
        self.assertTrue(result.accepted)

    def test_type_range_capability_and_precondition_are_structured(self):
        for action, expected in [
            ({'function': 'MoveToLevel', 'args': {'Level': 0}}, 'ARGUMENT_RANGE'),
            ({'function': 'MoveToLevel', 'args': {'Level': '100'}}, 'ARGUMENT_TYPE'),
            ({'function': 'SetTemperature', 'args': {}}, 'UNSUPPORTED_CAPABILITY'),
        ]:
            self.assertEqual(self.registry.validate(action, {'device_type': 'lamp-v2', 'power': False}).reason_code, expected)
        self.assertEqual(self.registry.validate({'function': 'SetTemperature', 'args': {}},
                         {'device_type': 'lamp-v2', 'power': False}).reason_code, 'UNSUPPORTED_CAPABILITY')

    def test_unknown_contract_is_explicitly_rejected(self):
        result = self.registry.validate({'function': 'On', 'args': {}}, {'device_type': 'new-device'})
        self.assertEqual(result.reason_code, 'UNSUPPORTED_DEVICE_CONTRACT')
        self.assertFalse(result.accepted)

    def test_public_structure_can_produce_conservative_contract(self):
        contract = contract_from_structure({'device_type': 'novel-lamp', 'endpoints': {
            '1': {'clusters': {'OnOff': {'commands': ['On', 'Off']}}}}})
        self.assertEqual(contract.device_type, 'novel-lamp')
        self.assertTrue(contract.validate({'function': 'On', 'args': {}}, {}).accepted)


if __name__ == '__main__':
    unittest.main()
