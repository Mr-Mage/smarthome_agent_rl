import unittest

from smarthome_agent_rl.benchmarks.wire import compile_wire


class WireCompilerTests(unittest.TestCase):
    def setUp(self):
        self.functions={'room.light.level':{'args':{'level':{'type':'integer'}}},
                        'room.light.color':{'args':{'color':{'type':'array'}}},
                        'room.light.on':{'args':{}}}
        self.shapes={('room.light.color','color'):3}

    def test_reserved_quoted_marker_layout_and_duplicates(self):
        row=compile_wire('{\n    "error_input";\n    room.light.level(40);\n    "error_input"\n}',self.functions,self.shapes)
        self.assertEqual(row['prediction'],'{error_input,room.light.level(40),error_input}')
        self.assertEqual(row['uncovered'],[])

    def test_json_explicit_method_parameters_bind_only_by_public_signature(self):
        row=compile_wire('{"room.light.level":{"level":40},"room.light.color":[1,2,3],"room.light.on":null}',self.functions,self.shapes)
        self.assertEqual(row['prediction'],'{room.light.level(40),room.light.color((1, 2, 3)),room.light.on()}')

    def test_repeated_json_method_keys_preserve_both_calls(self):
        row=compile_wire('{"room.light.level":20,"room.light.level":40}',self.functions,self.shapes)
        self.assertEqual(row['prediction'],'{room.light.level(20),room.light.level(40)}')

    def test_unknown_methods_are_not_renamed_and_unsafe_source_abstains(self):
        self.assertEqual(compile_wire('{"other.light.level":50}',self.functions,self.shapes)['prediction'],'{other.light.level(50)}')
        for text in ('{import os}', '{x=room.light.on()}', '{"light": {"on":true}}',
                     '{"room.light.level": {"level":40,"level":20}}', '{room.light.level(eval("unsafe"))}',
                     '{"room.light.level":1e309}'):
            self.assertTrue(compile_wire(text,self.functions,self.shapes)['uncovered'])


if __name__=='__main__':
    unittest.main()
