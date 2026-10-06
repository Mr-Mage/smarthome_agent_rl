import unittest

from smarthome_agent_rl.device_contract import DeviceContract
from tests import test_homebench_adapter as fixtures


class HomeBenchDialectTests(unittest.TestCase):
    def setUp(self):
        fixtures.HomeBenchAdapterTests.setUp(self)
        contract, shapes, spec = self.adapter.contract('case1')
        spec['functions']['bedroom.light.set_mode'] = {'capability':'bedroom.light',
            'args':{'mode':{'type':'string','enum':['low','auto']}}}
        self.adapter._contracts[0] = (DeviceContract.from_dict('fixture', spec), shapes, spec)

    def test_only_declared_public_string_enum_tokens_bind(self):
        text='{bedroom.light.set_mode(auto)}'
        self.assertEqual(self.adapter.guard('case1', text)['prediction'], '{error_input}')
        result=self.adapter.guard('case1', text, symbolic_enums=True)
        self.assertEqual(result['prediction'], text)
        self.assertEqual(result['symbolic_bindings'][0]['public_enum'], ['low','auto'])
        self.assertEqual(result['symbolic_bindings'][0]['value'], 'auto')

    def test_keywords_bind_without_normalizing_output_quotes(self):
        for text in ('{bedroom.light.set_mode(mode=low)}', "{bedroom.light.set_mode('low')}"):
            self.assertEqual(self.adapter.guard('case1',text,symbolic_enums=True)['prediction'],text)

    def test_unknown_enum_wrong_argument_type_and_nested_calls_remain_rejected(self):
        for text in ('{bedroom.light.set_mode(medium)}','{bedroom.light.set_brightness(auto)}',
                     '{bedroom.light.set_mode(True)}',"{bedroom.light.set_mode(__import__('os'))}"):
            self.assertEqual(self.adapter.guard('case1',text,symbolic_enums=True)['prediction'],'{error_input}')


if __name__=='__main__':
    unittest.main()
