from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from smarthome_agent_rl.execution.mutation import PostconditionResult, VerificationStatus
from smarthome_agent_rl.execution.scheduler import Scheduler
from smarthome_agent_rl.execution.store import RuntimeStore
from smarthome_agent_rl.execution.tasks import TaskManager
from smarthome_agent_rl.execution.trace import ToolTrace


class TaskConflictTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store=RuntimeStore(Path(temp.name)/'tasks.sqlite3')
        self.manager=TaskManager(self.store)
        self.a=self.manager.create('u1','private first user goal','chat1')
        self.b=self.manager.create('u2','private second user goal','chat2')
        self.calls=[]
        def dispatch(tool,args):
            self.calls.append(tool)
            return {'status':{'code':200},'data':{'workflow_id':str(len(self.calls))}}
        pending=lambda *args:PostconditionResult(VerificationStatus.UNVERIFIED,reason='not due yet')
        self.scheduler=Scheduler(self.manager,ToolTrace(dispatch),execute_action=pending,verify_job=pending,wake_agent=pending)

    def schedule(self,task,value=25,*,at=10,**claim):
        return self.scheduler.schedule(task.task_id,task.user_id,expected_version=task.version,target_time=at,
            resource_claims=[{'device_id':'ac','path':'target','value':value,**claim}])

    def conflicts(self):
        return self.manager.detect_conflicts(self.a.task_id,'u1')

    def test_same_device_time_and_incompatible_targets_report_without_arbitration(self):
        first=self.schedule(self.a)
        second=self.schedule(self.b,21)
        result=self.conflicts()['conflicts']
        self.assertEqual(len(result),1)
        self.assertEqual(second['conflict_ids'],[result[0]['conflict_id']])
        self.assertEqual(self.store.get('job',first['job_id'])[0]['status'],'SCHEDULED')
        self.assertEqual(self.store.get('job',second['job_id'])[0]['status'],'SCHEDULED')
        self.assertEqual(self.calls,[])
        self.assertNotIn('private second',str(result))

    def test_numeric_tolerances_allow_compatible_state_bands(self):
        self.schedule(self.a,25,tolerance=.5)
        self.schedule(self.b,26,tolerance=.5)
        self.assertEqual(self.conflicts()['conflicts'],[])
        self.schedule(self.b,27,tolerance=.1)
        self.assertGreater(len(self.conflicts()['conflicts']),0)

    def test_boolean_is_not_interchangeable_with_numeric_state(self):
        self.schedule(self.a,True)
        self.schedule(self.b,1)
        self.assertEqual(len(self.conflicts()['conflicts']),1)

    def test_separate_time_device_property_or_home_has_no_conflict(self):
        self.schedule(self.a)
        for options in ({'at':11},{'device_id':'other'},{'path':'fan_speed'},{'namespace':'other-home'}):
            self.schedule(self.b,21,**options)
        self.assertEqual(self.conflicts()['conflicts'],[])

    def test_closed_window_boundary_and_contradictory_steps_in_one_task(self):
        self.schedule(self.a,25,start_time=9,end_time=10)
        self.schedule(self.a,21,at=11,start_time=10,end_time=12)
        self.assertEqual(len(self.conflicts()['conflicts']),1)

    def test_concurrent_scheduling_and_restart_preserve_one_conflict_record(self):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda pair:self.schedule(*pair),[(self.a,25),(self.b,21)]))
        restored=TaskManager(RuntimeStore(self.store.path))
        result=restored.detect_conflicts(self.a.task_id,'u1')
        self.assertEqual(len(result['conflicts']),1)
        self.assertEqual(len(self.store.list('conflict')),1)

    def test_cancelled_revision_removes_live_conflict_but_keeps_original_evidence(self):
        self.schedule(self.a)
        self.schedule(self.b,21)
        self.manager.revise(self.b.task_id,'u2',expected_version=1,source_conversation='chat3',goal='new goal')
        self.assertEqual(self.conflicts()['conflicts'],[])
        record,rev=self.store.get('conflict',self.store.list('conflict')[0]['conflict_id'])
        with self.assertRaises(ValueError):
            self.store.put('conflict',record['conflict_id'],{},expected_revision=rev)

    def test_uncertain_job_still_holds_resource_claim(self):
        first=self.schedule(self.a)
        row,rev=self.store.get('job',first['job_id'])
        row['status']='UNKNOWN'
        self.store.put('job',first['job_id'],row,expected_revision=rev)
        self.schedule(self.b,21)
        self.assertEqual(len(self.conflicts()['conflicts']),1)

    def test_unresolved_targets_report_coverage_gap_instead_of_false_compatibility(self):
        self.schedule(self.a,'$state.temperature')
        self.schedule(self.b,21)
        result=self.conflicts()
        self.assertEqual(result['conflicts'],[])
        self.assertEqual(len(result['uncovered']),1)

    def test_ownership_and_malformed_claims_are_rejected_before_registration(self):
        with self.assertRaises(PermissionError):
            self.manager.detect_conflicts(self.a.task_id,'u2')
        for changes in ({'start_time':11,'end_time':10},{'value':float('nan')},{'tolerance':-1},{'path':''}):
            with self.assertRaises(ValueError):
                self.scheduler.schedule(self.a.task_id,'u1',expected_version=1,target_time=10,
                    native_call={'tool':'register_native','args':{}},
                    resource_claims=[{'device_id':'ac','path':'target','value':25,**changes}])
        self.assertEqual(self.calls,[])
        self.assertEqual(self.store.list('job'),[])


if __name__ == '__main__':
    unittest.main()
