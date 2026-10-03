import json
import unittest

from smarthome_agent_rl.structured import StructuredProvider
from src.agents.strategies.react_agent import ReActAgent
from src.agents.tools import TOOL_REGISTRY
from src.agents.types import ChatMessage


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages, response_format=None):
        self.calls.append((messages, response_format))
        return json.dumps(self.response)


def call(tool, arguments):
    return {"thought": "test", "call": {"tool": tool, "arguments": arguments}}


class StructuredTests(unittest.TestCase):
    def setUp(self):
        self.command = {"device_id": "light", "endpoint_id": 1, "cluster_id": "OnOff",
                        "command_id": "On", "args": {}}
        self.provider = StructuredProvider(FakeProvider(call("execute_command", self.command)))

    def test_full_registry_and_mandatory_args(self):
        self.assertEqual(set(self.provider.schemas), set(TOOL_REGISTRY))
        self.assertIn("args", self.provider.schemas["execute_command"]["required"])

    def test_upstream_parser_round_trip(self):
        text = json.dumps(self.provider.validate(call("execute_command", self.command)))
        tool, args, _ = ReActAgent(self.provider)._parse_structured_output(text)
        self.assertEqual(tool, "execute_command")
        self.assertEqual(args, self.command)

    def test_historical_missing_args_rejected(self):
        missing = {k: v for k, v in self.command.items() if k != "args"}
        with self.assertRaises(ValueError):
            self.provider.validate(call("execute_command", missing))

    def test_invalid_types_and_extra_fields(self):
        for change in [{"endpoint_id": True}, {"endpoint_id": "1"}, {"args": "{}"}, {"unknown": 1}]:
            with self.assertRaises(ValueError):
                self.provider.validate(call("execute_command", {**self.command, **change}))
        with self.assertRaises(ValueError):
            self.provider.validate(call("invented_tool", {}))

    def test_demo_does_not_enable_finish(self):
        fake = FakeProvider(call("get_room_devices", {"room_id": "bedroom"}))
        harness = StructuredProvider(fake)
        messages = [ChatMessage("system", "base"),
            ChatMessage("assistant", '{"thought":"demo","action":"get_rooms","action_input":"{}"}'),
            ChatMessage("user", 'observation: {"status":{"code":200}}'),
            ChatMessage("user", "This is your actual task. Turn on the light.")]
        harness.generate(messages)
        self.assertFalse(harness.audit[-1]["finish_enabled"])
        messages += [ChatMessage("assistant", '{"thought":"query","action":"get_room_devices","action_input":"{}"}'),
                     ChatMessage("user", 'observation: {"status":{"code":200},"error":null}')]
        harness.generate(messages)
        self.assertTrue(harness.audit[-1]["finish_enabled"])
        self.assertEqual(len(fake.calls), 2)

    def test_error_recovery_without_extra_inference(self):
        messages = [ChatMessage("system", "base"), ChatMessage("user", "This is your actual task."),
            ChatMessage("assistant", '{"thought":"try","action":"execute_command","action_input":"{}"}'),
            ChatMessage("user", 'observation: {"status":{"code":400},"error":{"detail":"Missing args"}}')]
        self.provider.generate(messages)
        self.assertIn("Missing args", self.provider.audit[-1]["recovery_hint"])
        self.assertEqual(len(self.provider.inner.calls), 1)

    def test_ablation_switches_change_only_declared_rules(self):
        messages = [ChatMessage("system", "base"), ChatMessage("user", "This is your actual task.")]
        no_gate = StructuredProvider(FakeProvider(call("finish", {"answer": "test"})), finish_guard=False)
        self.assertEqual(json.loads(no_gate.generate(messages))["action"], "finish")
        self.assertTrue(no_gate.audit[-1]["finish_enabled"])
        self.assertNotIn("required before finish is available", no_gate.inner.calls[0][0][0].content)
        no_recovery = StructuredProvider(FakeProvider(call("get_rooms", {})), recovery=False)
        error_history = messages + [ChatMessage("assistant", '{"action":"get_rooms"}'),
            ChatMessage("user", 'observation: {"status":{"code":400},"error":"failed"}')]
        no_recovery.generate(error_history)
        self.assertIsNone(no_recovery.audit[-1]["recovery_hint"])
        schema_only = StructuredProvider(FakeProvider(call("get_rooms", {})), finish_guard=False, recovery=False, guidance=False)
        schema_only.generate(messages)
        self.assertNotIn("Matter OnOff", schema_only.inner.calls[0][0][0].content)


if __name__ == "__main__":
    unittest.main()
