import copy
import json
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.action_state import ActionLedger
from smarthome_agent_rl.episode_review import digest, markdown, review_episode


class EpisodeReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'episode'
        self.folder.mkdir()
        ledger = ActionLedger()
        args = {'room_id':'living_room'}
        response = {'status':{'code':200},'data':{'device':'public-device'},'error':None}
        aid = ledger.propose('get_room_devices',args,turn=1,observation_version=0)
        ledger.dispatched(aid)
        ledger.observed(aid,response,1)
        self.audit = {'action_lifecycle':ledger.snapshot(),'actual_observations':[
            {'turn':1,'tool':'get_room_devices','arguments':args,'response':response,'extra_query':False}]}
        self.files = {'harness_audit.json':self.audit,
            'contract.json':{'task_identity':{'id':'task'},'config':{'variant':'G','model_seed':42,'served_model':'actor'}},
            'summary.json':{'task_id':'task','variant':'G','actor_seed':42,'success':False,
                'official_score':0,'task_failure':False,'actor_tokens':12,'actor_model_calls':1},
            'model_calls.json':[{'request':{'model':'actor','seed':42,'messages':[
                {'role':'user','content':'This is your actual task. Check device.'}]},
                'response':{'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}}}]}
        self.save()

    def save(self):
        for name,value in self.files.items():
            (self.folder/name).write_text(json.dumps(value),encoding='utf-8')

    def test_review_keeps_official_failure_and_public_evidence(self):
        before = {p.name:p.read_bytes() for p in self.folder.iterdir()}
        result = review_episode(self.folder)
        self.assertEqual(result['action_states'],{'completed':1})
        self.assertFalse(result['official_success'])  # completed read cannot imply goal success
        self.assertEqual(result['integrity'],'self_hash_only')
        self.assertEqual(result['actions'][0]['public_evidence'][0]['response'],self.audit['actual_observations'][0]['response'])
        self.assertIn('Check device.',markdown(result))
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.folder.iterdir()})

    def test_frozen_receipts_detect_changed_source(self):
        index = self.root/'manifest.json'
        index.write_text(json.dumps({'files_sha256':{'episode/'+p.name:digest(p) for p in self.folder.iterdir()}}))
        self.assertEqual(review_episode(self.folder,index)['integrity'],'matched_frozen_receipts')
        self.files['summary.json']['success'] = True
        self.save()
        with self.assertRaises(ValueError):
            review_episode(self.folder,index)

    def test_rejects_wrong_receipt_owner_and_broken_transition(self):
        self.audit['actual_observations'][0]['arguments']={'room_id':'other'}
        self.save()
        with self.assertRaises(ValueError):
            review_episode(self.folder)
        self.audit['actual_observations'][0]['arguments']={'room_id':'living_room'}
        self.audit['action_lifecycle']['actions'][0]['transitions'][1]['from']='completed'
        self.save()
        with self.assertRaises(ValueError):
            review_episode(self.folder)

    def test_rejects_accounting_model_and_identity_mismatch(self):
        for name, field, replacement in (('summary.json','actor_tokens',99),
                                         ('summary.json','actor_seed',43)):
            original = copy.deepcopy(self.files)
            self.files[name][field] = replacement
            self.save()
            with self.assertRaises(ValueError):
                review_episode(self.folder)
            self.files = original
        self.files['model_calls.json'][0]['request']['model']='different-model'
        self.save()
        with self.assertRaises(ValueError):
            review_episode(self.folder)

    def test_missing_usage_remains_visible_lower_bound(self):
        self.files['model_calls.json'][0]['response']={}
        self.files['summary.json']['actor_tokens']=0
        self.save()
        result = review_episode(self.folder)
        self.assertEqual(result['actor_cost']['calls_without_usage'],1)
        self.assertFalse(result['official_success'])


if __name__ == '__main__':
    unittest.main()
