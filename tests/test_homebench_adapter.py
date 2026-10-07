from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.benchmarks.homebench import HomeBenchAdapter, aggregate, native_counts


class HomeBenchAdapterTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        home={'home_id':0,'home_status':{'bedroom':{'room_name':'bedroom','light':{'state':'off','attributes':{
            'brightness':{'value':20,'lowest':'0','highest':'100'}}}}},
            'method':[{'room_name':'bedroom','device_name':'light','operation':'set_brightness',
                       'parameters':[{'name':'brightness','type':'int'}]},
                      {'room_name':'bedroom','device_name':'light','operation':'set_color',
                       'parameters':[{'name':'color','type':'typing.Tuple[int, int, int]'}]}]}
        self.case={'id':'case1','home_id':0,'input':'Turn the bedroom light to 30.',
                   'type':'FORBIDDEN_CATEGORY','output':'FORBIDDEN_GOLD'}
        assets={'dataset/test_data.jsonl':json.dumps(self.case)+'\n',
                'dataset/home_status_method.jsonl':json.dumps(home)+'\n',
                'code/system.txt':'Use only provided methods.\n',
                'code/model_test.py':'def chang_json2str(state,methods):\n    return str(state),str(methods)\n',
                'code/eval.py':'# fixture\n'}
        files={}
        for name,content in assets.items():
            path=root/name
            path.parent.mkdir(exist_ok=True,parents=True)
            path.write_text(content,encoding='utf-8')
            files[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        self.adapter=HomeBenchAdapter(root,{'files':files})
        self.root=root
        self.files=files

    def test_actor_input_excludes_gold_and_label_categories(self):
        messages=self.adapter.public_input('case1')
        self.assertNotIn('FORBIDDEN',json.dumps(messages))
        self.assertIn('Turn the bedroom light to 30.',messages[0]['content'])
        self.assertEqual([m['role'] for m in messages],['system'])

    def test_unknown_device_and_out_of_range_become_error_input(self):
        result=self.adapter.guard('case1','{bedroom.light.set_brightness(30),garage.light.turn_on(),bedroom.light.set_brightness(130)}')
        self.assertEqual(result['prediction'],'{bedroom.light.set_brightness(30),error_input,error_input}')
        self.assertEqual([r['reason'] for r in result['rejections']],['UNSUPPORTED_FUNCTION','ARGUMENT_RANGE'])

    def test_generated_calls_are_data_and_argument_functions_never_execute(self):
        result=self.adapter.guard('case1',"{bedroom.light.set_brightness(__import__('os').system('echo unsafe'))}")
        self.assertEqual(result['prediction'],'{error_input}')
        self.assertEqual(result['rejections'][0]['reason'],'ARGUMENT_LITERAL_OR_SHAPE')

    def test_tuple_shape_and_boolean_integer_mismatch(self):
        result=self.adapter.guard('case1','{bedroom.light.set_color((1,2,3)),bedroom.light.set_color((1,2)),bedroom.light.set_brightness(True)}')
        self.assertEqual(result['prediction'],'{bedroom.light.set_color((1,2,3)),error_input,error_input}')

    def test_parse_failure_is_uncovered_without_invented_rejection(self):
        result=self.adapter.guard('case1','Not machine instructions')
        self.assertEqual(result['prediction'],'Not machine instructions')
        self.assertTrue(result['uncovered'])
        self.assertEqual(result['rejections'],[])

    def test_duplicate_keywords_and_extra_arguments_are_rejected(self):
        for text in ('{bedroom.light.set_brightness(1,brightness=2)}','{bedroom.light.set_brightness(1,2)}'):
            self.assertEqual(self.adapter.guard('case1',text)['prediction'],'{error_input}')

    def test_native_metric_retains_duplicate_error_tokens_and_order_independence(self):
        row=native_counts('{error_input,bedroom.light.set_brightness(30),error_input}',
                          "'''bedroom.light.set_brightness(30),error_input,error_input,'''" )
        self.assertTrue(row['exact_match'])
        self.assertEqual(row['true_positive'],3)
        wrong=native_counts('{error_input}',"'''error_input,error_input'''" )
        self.assertFalse(wrong['exact_match'])
        self.assertEqual(aggregate([wrong])['recall'],.5)

    def test_native_empty_prediction_has_explicit_zero_denominator_rule(self):
        result=aggregate([native_counts('No output',"'''error_input'''" )])
        self.assertEqual(result['precision'],0)
        self.assertEqual(result['f1'],0)
        self.assertEqual(result['exact_match'],0)

    def test_changed_public_source_fails_hash_gate(self):
        (self.root/'code/system.txt').write_text('changed',encoding='utf-8')
        with self.assertRaises(ValueError):
            HomeBenchAdapter(self.root,{'files':self.files})


if __name__=='__main__':
    unittest.main()
