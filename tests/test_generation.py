import json
from types import SimpleNamespace
import unittest

from smarthome_agent_rl.generation import generation_options, install_generation_options, record_generation_errors, use_direct_service_transport, audit_response_payload


OPTIONS = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 2048,
    "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0,
    "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}}


class GenerationTests(unittest.TestCase):
    def test_non_json_http_error_keeps_status_and_does_not_trigger_transport_retry(self):
        import httpx
        from openai import OpenAI
        from src.agents.providers import OpenAIChatProvider
        from src.agents.providers.base import NonRetryableLLMError
        from src.agents.types import ChatMessage
        requests, responses = [], []
        def handle(request):
            requests.append(request)
            return httpx.Response(403, text='Proxy denied this request')
        def capture(response):
            response.read()
            responses.append({'status': response.status_code, 'payload': audit_response_payload(response)})
        p = OpenAIChatProvider(model='judge', api_key='dummy', api_base='http://local.test/v1')
        p._client.close()
        p._client = OpenAI(api_key='dummy', base_url='http://local.test/v1',
            http_client=httpx.Client(transport=httpx.MockTransport(handle),
                event_hooks={'response': [capture]}))
        try:
            with self.assertRaises(NonRetryableLLMError) as caught:
                p.generate([ChatMessage('user', 'task')])
            self.assertIn('HTTP 403', str(caught.exception))
            self.assertEqual(len(requests), 1)
            self.assertEqual(responses[0]['status'], 403)
            self.assertEqual(responses[0]['payload']['body_excerpt'], 'Proxy denied this request')
        finally:
            p._client.close()

    def test_direct_service_survives_inherited_invalid_proxy(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread
        from unittest.mock import patch
        from src.agents.providers import OpenAIChatProvider
        from src.agents.types import ChatMessage
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body = json.dumps({'id': 'test', 'object': 'chat.completion', 'created': 0,
                    'model': 'judge', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'A'}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict('os.environ', {'HTTP_PROXY': 'http://127.0.0.1:1',
                'http_proxy': 'http://127.0.0.1:1', 'ALL_PROXY': 'http://127.0.0.1:1',
                'NO_PROXY': '', 'no_proxy': ''}):
                p = OpenAIChatProvider(model='judge', api_key='dummy',
                    api_base=f'http://127.0.0.1:{server.server_port}/v1', timeout=5, max_retries=0)
                original = p._client
                retries, timeout = original.max_retries, original.timeout
                try:
                    use_direct_service_transport(p)
                    self.assertTrue(original.is_closed())
                    self.assertEqual(p._client.max_retries, retries)
                    self.assertEqual(p._client.timeout, timeout)
                    self.assertEqual(p.generate([ChatMessage('user', 'task')]), 'A')
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(requests[0]['model'], 'judge')
                finally:
                    p._client.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_error_observer_preserves_request_exception_and_next_attempt(self):
        error = RuntimeError('temporary transport failure')
        calls, errors = [], []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise error
            return 'result'
        provider = SimpleNamespace(_client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create))))
        record_generation_errors(provider, 'judge-42', errors.append)
        with self.assertRaises(RuntimeError) as caught:
            provider._client.chat.completions.create(model='judge', seed=42)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(errors[0]['provider'], 'judge-42')
        self.assertEqual(errors[0]['type'], 'RuntimeError')
        self.assertEqual(provider._client.chat.completions.create(model='judge', seed=42), 'result')
        self.assertEqual(calls, [{'model': 'judge', 'seed': 42}] * 2)
        self.assertEqual(len(errors), 1)

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
