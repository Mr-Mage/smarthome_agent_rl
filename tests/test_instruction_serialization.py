import unittest

from smarthome_agent_rl.benchmarks.instructions import serialize_instructions


class InstructionSerializationTests(unittest.TestCase):
    def test_comma_semicolon_and_newline_sequences_keep_argument_text(self):
        for body in ("room.light.on();room.light.color((1,2,3));", "room.light.on()\nroom.light.color((1,2,3))",
                     "room.light.on(),room.light.color((1,2,3)),"):
            result = serialize_instructions('{'+body+'}')
            self.assertEqual(result['prediction'], '{room.light.on(),room.light.color((1,2,3))}')
            self.assertEqual(result['uncovered'], [])

    def test_error_input_duplicates_and_quoted_delimiters_remain_data(self):
        result = serialize_instructions("error_input;room.mode('a;{b}');error_input")
        self.assertEqual(result['prediction'], "{error_input,room.mode('a;{b}'),error_input}")

    def test_symbolic_arguments_are_not_inferred_or_quoted(self):
        self.assertEqual(serialize_instructions('{room.mode(dry);}')['prediction'], '{room.mode(dry)}')

    def test_assignments_imports_nested_calls_and_expansions_abstain(self):
        for text in ('{x=room.on()}', '{import os}', "{room.mode(__import__('os').system('unsafe'))}",
                     '{room.on(**payload)}', '{[room.on() for _ in range(4)]}', 'description {room.on()}', '{}'):
            result = serialize_instructions(text)
            self.assertEqual(result['prediction'], text)
            self.assertTrue(result['uncovered'])


if __name__ == '__main__':
    unittest.main()
