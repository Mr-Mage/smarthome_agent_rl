import json
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.generation import generation_options, install_generation_options


OPTIONS = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048,
    "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0,
    "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}}


class GenerationTests(unittest.TestCase):
    def test_legacy_default(self):
        self.assertEqual(generation_options({}), {"temperature": 0.0})

    def test_reject_policy_or_invalid_fields(self):
        for options in ({"model": "other"}, {"response_format": {}}, {"seed": 3},
            {"max_tokens": True}, {"temperature": float("nan")}, {"top_p": 0},
            {"extra_body": {"messages": []}},
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": "false"}}}):
            with self.assertRaises(ValueError):
                generation_options({"generation": options})

    def test_decorator_preserves_policy_and_does_not_mutate_settings(self):
        calls = []
        original = lambda **kw: calls.append(kw) or "unchanged-result"
        provider = SimpleNamespace(_client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=original))))
        config = {"generation": json.loads(json.dumps(OPTIONS))}
        install_generation_options(provider, config)
        config["generation"]["extra_body"]["top_k"] = 1
        schema, messages = {"type": "json_schema"}, [{"role": "user", "content": "task"}]
        result = provider._client.chat.completions.create(model="model", seed=42,
            temperature=0, messages=messages, response_format=schema)
        self.assertEqual(result, "unchanged-result")
        self.assertEqual(calls[0]["messages"], messages)
        self.assertEqual(calls[0]["response_format"], schema)
        self.assertEqual(calls[0]["model"], "model")
        self.assertEqual(calls[0]["seed"], 42)
        self.assertEqual(calls[0]["extra_body"]["top_k"], 20)
        self.assertEqual(calls[0]["temperature"], 0.7)

    def test_real_upstream_provider_and_sdk_put_options_on_wire(self):
        import httpx
        from openai import OpenAI
        from src.agents.providers.openai_provider import OpenAIChatProvider
        from src.agents.types import ChatMessage
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"action":"finish"}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        provider = OpenAIChatProvider(model="model", api_key="dummy", api_base="http://local.test/v1", max_retries=0)
        provider._client.close()
        provider._client = OpenAI(api_key="dummy", base_url="http://local.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        try:
            install_generation_options(provider, {"generation": OPTIONS})
            schema = {"type": "json_object"}
            self.assertEqual(provider.generate([ChatMessage("user", "task")], schema), '{"action":"finish"}')
            body = requests[0]
            self.assertEqual(body["response_format"], schema)
            self.assertEqual(body["seed"], 42)
            self.assertEqual(body["messages"], [{"role": "user", "content": "task"}])
            for key, value in OPTIONS.items():
                if key == "extra_body":
                    for extra_key, extra_value in value.items():
                        self.assertEqual(body[extra_key], extra_value)
                else:
                    self.assertEqual(body[key], value)
        finally:
            provider._client.close()


if __name__ == "__main__":
    unittest.main()
