import json,tempfile,unittest,importlib.util
from pathlib import Path
from scripts.build_sft_dataset import curate
from smarthome_agent_rl.benchmark import digest

class CurationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'Run curation test in the isolated SFT environment')
    def test_only_successful_training_tasks_and_correct_actions_get_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stage=root/'stage';stage.mkdir()
            rows=[{'id':name,'sha256':name,'query_type':'qt1','case':'feasible'} for name in ('train','eval')]
            schedule=[];manifest={}
            for row in rows:
                schedule.append({'task':row,'workflow':0,'variants':['G']})
                ep=stage/'worker0'/row['id']/'G/lightning';ep.mkdir(parents=True)
                actions=[{'thought':'bad target','call':{'tool':'execute_command','arguments':{}}},
                         {'thought':'report tool evidence','call':{'tool':'finish','arguments':{'answer':'done'}}}]
                calls=[{'status':200,'request':{'messages':[{'role':'user','content':'public input'}],
                    'response_format':{'json_schema':{'schema':{'type':'object'}}}},
                    'response':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(action)}}]}}
                    for action in actions]
                audit={'actual_observations':[{'tool':'get_room_devices','extra_query':False}], 'structured':[{'normalized_action':{'action':a['call']['tool'],'action_input':json.dumps(a['call']['arguments'])}} for a in actions],
                       'proposals':[{'turn':1,'blocked':True,'simulator_error':False}]}
                for name,value in [('model_calls.json',calls),('harness_audit.json',audit),('summary.json',
                    {'success':True,'infrastructure_error':False,'task_failure':False,'actor_tokens':10})]:
                    path=ep/name;path.write_text(json.dumps(value));manifest[path.relative_to(stage).as_posix()]=digest(path)
            (stage/'protocol.json').write_text(json.dumps({'schedule':schedule,'expected_episodes':2}))
            (stage/'completion.json').write_text(json.dumps({'complete':True,'episodes':2}))
            (stage/'artifact_manifest.json').write_text(json.dumps(manifest))
            (stage/'report.json').write_text('{}')
            selected,audit=curate(['stage'],rows[:1],root)
            self.assertEqual(len(selected),1)
            self.assertEqual(selected[0]['task_id'],'train')
            self.assertEqual(json.loads(selected[0]['target'])['call']['tool'],'finish')
            self.assertNotIn('bad target',selected[0]['target'])
            self.assertEqual(audit['exclusions']['incorrect_action_target'],1)
            calls_path=stage/'worker0/train/G/lightning/model_calls.json'
            calls_path.write_text('[]')
            with self.assertRaises(ValueError):curate(['stage'],rows[:1],root)
