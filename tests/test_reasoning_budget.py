import unittest
from smarthome_agent_rl.benchmark import task_failure_kind

class ReasoningBudgetTests(unittest.TestCase):
    def test_empty_output_is_model_failure_only_with_successful_truncated_http(self):
        error={'type':'AgentExecutionError','message':'Episode failed: LLM response content was empty'}
        call={'status':200,'response':{'choices':[{'finish_reason':'length','message':{'content':None,'reasoning':'budget used'}}]}}
        self.assertEqual(task_failure_kind(error,[call]),'actor_completion_limit')
        self.assertIsNone(task_failure_kind(error,[]))
        self.assertIsNone(task_failure_kind(error,[{**call,'status':503}]))
        self.assertIsNone(task_failure_kind(error,[{'status':200,'response':{'choices':[{'finish_reason':'stop'}]}}]))
