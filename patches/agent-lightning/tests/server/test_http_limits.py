"""Check configurable pool limits while retaining the gateway defaults."""
import httpx
import pytest
from fastapi.testclient import TestClient
from agentlightning.server.app import create_app

@pytest.mark.parametrize("settings,connections,keepalive,pool_timeout", [
    ({}, 100, 20, 300),
    ({"max_connections": 512, "max_keepalive_connections": 256, "keepalive_expiry": 30, "pool_timeout": 60}, 512, 256, 60),
])
def test_gateway_http_pool(server_config, monkeypatch, settings, connections, keepalive, pool_timeout):
    captured = {}
    original = httpx.AsyncClient
    def client_factory(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    server_config["http_client"] = settings
    with TestClient(create_app(server_config)) as client:
        assert client.get("/healthz").status_code == 200
    assert captured["limits"].max_connections == connections
    assert captured["limits"].max_keepalive_connections == keepalive
    assert captured["timeout"].pool == pool_timeout
