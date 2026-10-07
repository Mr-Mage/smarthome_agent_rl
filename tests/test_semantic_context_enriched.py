import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.semantic_context import build_context, MAX_ENVIRONMENT_CHARS
from smarthome_agent_rl.semantic_context_enriched import build_enriched_context
from smarthome_agent_rl.workflow_semantics import load_workflow_semantics
from tests.test_semantic_context import receipt

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT/'configs/public-workflow-semantics.json').read_text(encoding='utf-8'))


class EnrichedSemanticContextTests(unittest.TestCase):
    def setUp(self):
        self.action = {'tool':'schedule_workflow','start_time':'2030-01-01 12:01:00','steps':[
            {'tool':'execute_command','args':{'device_id':'lamp','command_id':'On'}}]}
        self.rows = [receipt('get_room_devices',{'room_id':'utility'}, {'washer':{'device_type':'laundry_washer'}}),
                     receipt('get_device_structure',{'device_id':'washer'}, {'device_id':'washer','countdown':600}),
                     receipt('get_device_structure',{'device_id':'lamp'}, {'device_id':'lamp','endpoints':{}})]

    def build(self, rows=None, action=None, **options):
        return build_enriched_context('Turn on lamp before the washer finishes.', action or self.action,
                                      rows if rows is not None else self.rows, **options)

    def test_reference_receipt_is_prior_public_evidence_not_a_guessed_finish_time(self):
        original = copy.deepcopy(self.rows)
        context = self.build(references=True)
        reference = context.environment_state['observed_reference_receipts']['devices'][0]
        self.assertEqual(reference['device_id'],'washer')
        self.assertEqual(reference['room_id'],'utility')
        self.assertEqual(reference['facts'][0]['response'],self.rows[1]['response'])
        self.assertEqual(reference['facts'][0]['source']['observation_ordinal'],2)
        self.assertEqual(reference['state_freshness'],'UNKNOWN')
        self.assertNotIn('finish_time',reference)
        self.assertNotIn('workflow_execution_semantics',context.environment_state)
        self.assertEqual(self.rows,original)
        reference['facts'][0]['response']['data']['countdown']=1
        self.assertEqual(self.rows,original)

    def test_workflow_only_is_independent_and_nonworkflow_is_v1_identical(self):
        value = self.build(workflow_rules=RULES).environment_state
        self.assertNotIn('observed_reference_receipts',value)
        facts = value['workflow_execution_semantics']
        self.assertFalse(facts['facts']['per_step_wait_supported'])
        self.assertEqual(facts['declared_steps'],1)
        action={'tool':'execute_command','device_id':'lamp','command_id':'On'}
        self.assertEqual(asdict(self.build(action=action,workflow_rules=RULES)),
                         asdict(build_context('Turn on lamp before the washer finishes.',action,self.rows)))

    def test_failed_mismatched_and_unidentified_reads_do_not_fabricate_a_state(self):
        rows = self.rows + [receipt('get_device_structure',{'device_id':'washer'},{'device_id':'other'}),
                           receipt('get_attribute',{'device_id':'washer','attribute_id':'countdown'},None,code=404),
                           receipt('get_attribute',{}, {'value':'unbound'})]
        reference = self.build(rows,references=True).environment_state['observed_reference_receipts']['devices'][0]
        self.assertEqual(reference['failed_queries_total'],2)
        self.assertEqual(reference['facts_total'],1)
        self.assertEqual(reference['facts'][0]['response']['data']['countdown'],600)
        self.assertEqual(reference['state_freshness'],'UNKNOWN')
        self.assertEqual(reference['latest_failed_query']['observation_ordinal'],5)

    def test_attribute_read_binding_is_explicit_and_latest_same_request_wins(self):
        args={'device_id':'dryer','cluster_id':'OperationalState','attribute_id':'CountdownTime'}
        rows=[receipt('get_attribute',args,{'value':600}),receipt('get_attribute',args,{'value':590})]
        ref=self.build(rows,references=True).environment_state['observed_reference_receipts']['devices'][0]
        self.assertEqual(ref['facts_total'],1)
        self.assertEqual(ref['facts'][0]['response']['data']['value'],590)
        self.assertIn('request',ref['facts'][0]['binding'])
        self.assertEqual(ref['facts'][0]['source']['observation_ordinal'],2)

    def test_count_limits_and_size_removals_keep_all_totals(self):
        rows=[]
        for i in range(12):
            rows.append(receipt('get_room_devices',{'room_id':'utility'}, {str(i):{'device_type':'unknown'}}))
            for j in range(6):
                rows.append(receipt('get_attribute',{'device_id':str(i),'attribute_id':str(j)}, {'value':'x'*3000}))
        state=self.build(rows,references=True,workflow_rules=RULES).environment_state
        refs=state['observed_reference_receipts']
        self.assertEqual(refs['devices_total'],12)
        self.assertEqual(refs['facts_total'],72)
        self.assertEqual(refs['devices_omitted']+len(refs['devices']),12)
        self.assertEqual(refs['facts_omitted']+sum(len(d['facts']) for d in refs['devices']),72)
        self.assertEqual(refs['catalog_entries_omitted']+sum(len(d['catalog']) for d in refs['devices']),12)
        self.assertTrue(refs['size_truncated'])
        self.assertLessEqual(len(json.dumps(state,ensure_ascii=False,separators=(',',':'))),MAX_ENVIRONMENT_CHARS)
        self.assertTrue(all(len(d['facts'])<=4 for d in refs['devices']))

    def test_future_receipts_and_hidden_state_never_enter_the_preaction_context(self):
        future=receipt('get_device_structure',{'device_id':'washer'},{'device_id':'washer','countdown':0})
        state=self.build((self.rows+[future])[:3],references=True).environment_state
        self.assertEqual(state['observed_reference_receipts']['devices'][0]['facts'][0]['response']['data']['countdown'],600)
        with self.assertRaisesRegex(ValueError,'Hidden evaluator'):
            self.build(self.rows+[receipt('ignored',{}, {'hidden_goal':'leak'})],references=True)
        bad=copy.deepcopy(RULES);bad['facts']['judge_output']='YES'
        with self.assertRaisesRegex(ValueError,'Hidden evaluator'):
            self.build(workflow_rules=bad)

    def test_source_drift_and_path_escape_stop_rule_activation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(ValueError):
                load_workflow_semantics(root)
            rules=copy.deepcopy(RULES);rules['source_sha256']={'../outside.py':'0'*64}
            path=root/'rules.json';path.write_text(json.dumps(rules),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'source differs'):
                load_workflow_semantics(root,path)


if __name__=='__main__':unittest.main()
