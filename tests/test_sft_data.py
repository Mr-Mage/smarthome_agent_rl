import copy,json,unittest
from pathlib import Path
from smarthome_agent_rl.sft_data import partition, action_target, tokenized_target
from smarthome_agent_rl.variant_runtime import runtime

class SftDataTests(unittest.TestCase):
    def test_split_is_task_isolated_and_balanced(self):
        rows=json.loads(Path('configs/benchmark-v2/dev.json').read_text())['tasks']
        queries={r['id']:r['id']+' unique public query' for r in rows}
        result,audit=partition(rows,queries)
        self.assertEqual({k:len(v) for k,v in result.items()},{'train':60,'calibration':12,'eval':48})
        self.assertEqual(len({r['id'] for v in result.values() for r in v}),120)
        self.assertEqual(result,partition(rows,queries)[0])
        queries[rows[1]['id']]=queries[rows[0]['id']]
        with self.assertRaises(ValueError): partition(rows,queries)

    def test_private_reasoning_and_truncated_actions_not_labels(self):
        call={'status':200,'request':{'messages':[{'role':'user','content':'public input'}]},
              'response':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(
                  {'thought':'observed result','call':{'tool':'finish','arguments':{'answer':'done'}}}),
                  'reasoning':'private hidden chain'}}]}}
        messages,target,action=action_target(call)
        self.assertNotIn('private',target)
        call['response']['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):action_target(call)

    def test_mask_all_input_and_reject_partial_target(self):
        class Tokenizer:
            eos_token_id=9
            def apply_chat_template(self,*a,**kw):return [1,2,3]
            def encode(self,*a,**kw):return [4,5]
        row=tokenized_target(Tokenizer(),[{'role':'user','content':'input'}],'target',6)
        self.assertEqual(row['labels'],[-100,-100,-100,4,5,9])
        with self.assertRaises(ValueError):tokenized_target(Tokenizer(),[],'target',5)
        class MappingTokenizer(Tokenizer):
            def apply_chat_template(self,*a,**kw):return {'input_ids':[[1,2,3]],'attention_mask':[[1,1,1]]}
        self.assertEqual(tokenized_target(MappingTokenizer(),[],'target',6)['input_ids'],[1,2,3,4,5,9])

    def test_variant_runtime_is_frozen_without_mutating_control(self):
        config={'actor_model':'base','generation':{'temperature':.7,'max_tokens':2048},
                'variant_runtime':{'GThinking':{'generation':{'temperature':.7,'max_tokens':2048,
                    'extra_body':{'chat_template_kwargs':{'enable_thinking':True}}}}}}
        original=copy.deepcopy(config)
        result=runtime(config,'GThinking',{'actor_port':20000})
        self.assertTrue(result['generation']['extra_body']['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(runtime(config,'G',{'actor_port':20000})['generation'],config['generation'])
        self.assertEqual(original,config)
        config['variant_runtime']['GThinking']['generation']['max_tokens']=False
        with self.assertRaises(ValueError):runtime(config,'GThinking',{'actor_port':20000})
