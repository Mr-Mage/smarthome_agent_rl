import json
import unittest

from scripts.report_benchmark import baseline_action_metrics


def event(kind, payload=''):
    return {'event': kind, 'payload': payload}


def observation(payload):
    return event('observation', json.dumps(payload))


class BaselineReportingTests(unittest.TestCase):
    def test_parser_failures_never_count_as_executor_calls_even_when_third_observation_is_missing(self):
        events = [event('consecutive_failure', '1:invalid structured output'),
            observation({'error': 'Invalid structured output format'}),
            event('consecutive_failure', '2:invalid structured output'),
            observation({'error': 'Invalid structured output format'}),
            event('consecutive_failure', '3:invalid structured output')]
        self.assertEqual(baseline_action_metrics(events), {'invalid_proposed': 3,
            'invalid_reached_executor': 0, 'executed_tool_calls': 0, 'structured_rejections': 3})

    def test_unknown_tool_and_finish_validation_are_preexecution_rejections(self):
        events = [event('consecutive_failure', '1:unknown tool'),
            observation({'error': "Unknown tool 'schedule_tool_call'"}),
            event('consecutive_failure', '2:invalid finish payload'),
            observation({'error': "finish requires 'answer'"})]
        result = baseline_action_metrics(events)
        self.assertEqual(result['invalid_proposed'], 2)
        self.assertEqual(result['invalid_reached_executor'], 0)

    def test_dispatched_domain_errors_are_separate_from_parser_rejections(self):
        events = [event('action', 'execute_command'), event('waiting', 'execute_command'),
            observation({'status': {'code': 400}, 'error': {'type': 'invalid-command'}}),
            event('consecutive_failure', '1:invalid structured output'),
            observation({'error': 'Invalid structured output format'}),
            event('action', 'get_rooms'), observation({'status': {'code': 200}, 'error': None})]
        result = baseline_action_metrics(events)
        self.assertEqual(result['invalid_proposed'], 2)
        self.assertEqual(result['invalid_reached_executor'], 1)
        self.assertEqual(result['executed_tool_calls'], 2)

    def test_poll_budget_failure_before_dispatch_does_not_create_an_executor_call(self):
        events = [event('action', 'get_workflow_status'), event('waiting', 'get_workflow_status')]
        result = baseline_action_metrics(events)
        self.assertEqual(result['executed_tool_calls'], 0)
        self.assertEqual(result['invalid_proposed'], 0)
