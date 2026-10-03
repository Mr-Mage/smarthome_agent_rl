"""Real upstream ReAct + SDK, fake transport: test complete failure trajectories."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
from src.agents.providers.openai_provider import OpenAIChatProvider
from smarthome_agent_rl.reward import check_goals
import scripts.run_official_episode as episode


class BaselineFailureDataTests(unittest.TestCase):
    def exercise(self, *, accept, fail_reward=False):
        state = {"rooms": {"bedroom": {"devices": [{"device_id": "light", "attributes": {"power": False}}]}}}
        task = {"schema": "smarthome-task-v1", "task_id": "failure-test", "seed": 1,
            "instruction": "turn on light", "initial_home_config": {"base_time": "2026-10-03 00:00:00",
            "rooms": {"bedroom": {}}}, "goals": [{"device_id": "light", "attribute": "power", "op": "eq", "value": True}]}
        config = {"simulator_url": "http://fake", "served_model": "test-model", "model_seed": 42,
            "model_endpoint": "http://fake/v1", "max_steps": 20, "reward": {}, "baseline_id": "B0-ReAct-v1"}
        emitted = []
        class Environment:
            def __init__(self, url):
                self.state = state
            def reset(self, task):
                self.checks = check_goals(state, task["goals"])
            def request(self, method, route):
                return state
            def close(self):
                pass
        class EventClient:
            def __init__(self, **kwargs):
                pass
            def post(self, url, *, json):
                if fail_reward and json["event_type"] == "reward":
                    raise httpx.ConnectError("reward store unavailable")
                emitted.append(json)
                return httpx.Response(200, request=httpx.Request("POST", url), json={})
            def close(self):
                pass
        def model_response(request):
            return httpx.Response(200, json={"id": "fake", "object": "chat.completion", "created": 0,
                "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "not valid JSON"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        provider = OpenAIChatProvider(model="test-model", api_key="dummy", api_base="http://fake/v1", max_retries=0)
        provider._client.close()
        provider._client = OpenAI(api_key="dummy", base_url="http://fake/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(model_response)))
        try:
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "episode"
                variables = {"SMARTHOME_TASK": json.dumps(task), "SMARTHOME_CONFIG": json.dumps(config),
                    "SMARTHOME_OUTPUT": str(output), "AGL_OPENAI_BASE_URL": "http://fake/v1",
                    "AGL_KEY": "dummy", "AGL_EVENT_URL": "http://fake/events"}
                with patch.dict(os.environ, variables), patch.object(episode, "SimuHomeAdapter", Environment), \
                    patch.object(episode, "OpenAIChatProvider", return_value=provider), \
                    patch.object(episode.httpx, "Client", EventClient), redirect_stdout(io.StringIO()):
                    caught = None
                    try:
                        episode.run("lightning", task_failure_is_data=accept)
                    except Exception as exc:
                        caught = exc
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["model_calls"], 3)
                self.assertEqual(summary["steps"], 3)
                self.assertEqual(summary["invalid_actions"], 3)
                self.assertEqual(summary["model_rejections_without_feedback"], 1)
                self.assertFalse(summary["success"])
                self.assertTrue(summary["task_failure"])
                self.assertAlmostEqual(summary["reward"], -0.330006)
                return caught, emitted
        finally:
            provider._client.close()

    def test_complete_task_failure_is_data(self):
        caught, events = self.exercise(accept=True)
        self.assertIsNone(caught)
        self.assertEqual(sum(e["event_type"] == "environment_step" for e in events), 3)
        self.assertEqual(sum(e["event_type"] == "reward" for e in events), 1)

    def test_legacy_worker_still_raises(self):
        caught, _ = self.exercise(accept=False)
        self.assertIsInstance(caught, RuntimeError)

    def test_reward_delivery_failure_still_raises(self):
        caught, _ = self.exercise(accept=True, fail_reward=True)
        self.assertIsInstance(caught, httpx.ConnectError)


if __name__ == "__main__":
    unittest.main()
