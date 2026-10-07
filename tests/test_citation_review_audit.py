import copy
import json
import unittest

from scripts.audit_citation_review import verify_record
from smarthome_agent_rl.citation_review import request,resolve
from tests import test_citation_review as fixtures


class CitationReviewAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CitationReviewTests();self.fixture.setUp()
        self.item=self.fixture.item;self.config=self.fixture.config;self.actor=self.fixture.fixture.actor
        model=self.fixture.citation_decision();decision=resolve(self.item['context'],model,'citations')
        text=json.dumps({name:model[name] for name in ('correct_target','goal_consistent','trajectory_consistent','safe_to_execute')})
        raw={'model':self.config['model'],'choices':[{'message':{'content':text},'finish_reason':'stop'}],
             'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}
        self.row={'id':self.item['id'],'arm':'citations','actor_id':self.actor['id'],'model_decision':model,
                  'decision':decision,'parse_error':None,'call':{'endpoint':self.actor['endpoint'],
                  'body':request(self.config,self.item,'citations'),'error':None,'text':text,'response':raw,
                  'raw_response':json.dumps(raw),'finish_reason':'stop','usage':raw['usage'],'request_seconds':.1}}

    def verify(self,row):verify_record(row,self.item,'citations',self.actor,self.config)

    def test_audit_rejects_repaired_quote_changed_labels_and_rewritten_usage(self):
        self.verify(self.row)
        for modify in (lambda r:r['decision']['goal_consistent']['evidence']['steps'][0].update(effect_quote='repaired'),
                       lambda r:r['decision']['correct_target'].update(label='NO'),
                       lambda r:r['call'].update(usage={'total_tokens':0})):
            changed=copy.deepcopy(self.row);modify(changed)
            with self.assertRaises(ValueError):self.verify(changed)

    def test_request_catalog_and_raw_reference_are_source_bound(self):
        changed=copy.deepcopy(self.row);changed['call']['body']['messages'][-1]['content']='Edited source'
        with self.assertRaises(ValueError):self.verify(changed)
        changed=copy.deepcopy(self.row);changed['model_decision']['goal_consistent']['evidence']['steps'][0]['effect_ref']=1
        with self.assertRaises(ValueError):self.verify(changed)


if __name__=='__main__':unittest.main()
