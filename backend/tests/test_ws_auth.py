"""
D-3 regression tests: WebSocket auth must be FAIL-CLOSED.

The bug these lock down: `if _ENGINE_API_KEY and token != _ENGINE_API_KEY`
allowed every unauthenticated client whenever ENGINE_API_KEY was unset,
turning the live-engine socket into an open order-entry surface.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _build_app(monkeypatch, *, api_key: str, allowed_ips: str = ""):
    """Reload ws_auth so module-level env capture picks up the patched values."""
    monkeypatch.setenv("ENGINE_API_KEY", api_key)
    monkeypatch.setenv("WS_ALLOWED_IPS", allowed_ips)

    import backend.api.ws_auth as ws_auth
    importlib.reload(ws_auth)

    app = FastAPI()

    @app.websocket("/ws/test")
    async def _ep(ws: WebSocket, token: str = ""):
        if not await ws_auth.authenticate_ws(ws, token):
            return
        await ws.accept()
        await ws.send_text("authorized")
        await ws.close()

    return app, ws_auth


def _expect_rejected(client, url: str):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(url) as ws:
            ws.receive_text()


def test_unset_api_key_refuses_connection(monkeypatch):
    """THE regression: unset key must REFUSE, not bypass."""
    app, _ = _build_app(monkeypatch, api_key="")
    _expect_rejected(TestClient(app), "/ws/test?token=anything")


def test_missing_token_rejected(monkeypatch):
    app, _ = _build_app(monkeypatch, api_key="s3cret")
    _expect_rejected(TestClient(app), "/ws/test")


def test_wrong_token_rejected(monkeypatch):
    app, _ = _build_app(monkeypatch, api_key="s3cret")
    _expect_rejected(TestClient(app), "/ws/test?token=wrong")


def test_empty_token_rejected(monkeypatch):
    app, _ = _build_app(monkeypatch, api_key="s3cret")
    _expect_rejected(TestClient(app), "/ws/test?token=")


def test_correct_token_accepted(monkeypatch):
    app, _ = _build_app(monkeypatch, api_key="s3cret")
    with TestClient(app).websocket_connect("/ws/test?token=s3cret") as ws:
        assert ws.receive_text() == "authorized"


def test_bearer_header_accepted(monkeypatch):
    app, _ = _build_app(monkeypatch, api_key="s3cret")
    client = TestClient(app)
    with client.websocket_connect(
        "/ws/test", headers={"Authorization": "Bearer s3cret"}
    ) as ws:
        assert ws.receive_text() == "authorized"


def test_ip_allowlist_blocks_foreign_peer(monkeypatch):
    """SEBI static-IP connectivity: a valid token from a bad IP is still refused."""
    app, _ = _build_app(monkeypatch, api_key="s3cret", allowed_ips="10.0.0.0/8")
    client = TestClient(app, client=("203.0.113.9", 5555))
    _expect_rejected(client, "/ws/test?token=s3cret")


def test_ip_allowlist_permits_listed_peer(monkeypatch):
    app, _ = _build_app(monkeypatch, api_key="s3cret", allowed_ips="10.0.0.0/8")
    client = TestClient(app, client=("10.1.2.3", 5555))
    with client.websocket_connect("/ws/test?token=s3cret") as ws:
        assert ws.receive_text() == "authorized"


def test_unparseable_peer_is_refused_when_allowlist_active(monkeypatch):
    """Fail-closed: if we cannot resolve the peer to an IP, refuse."""
    app, _ = _build_app(monkeypatch, api_key="s3cret", allowed_ips="10.0.0.0/8")
    # Starlette's default TestClient peer is the non-IP string "testclient"
    _expect_rejected(TestClient(app), "/ws/test?token=s3cret")


def test_malformed_allowlist_entry_is_ignored_not_fatal(monkeypatch):
    app, ws_auth = _build_app(
        monkeypatch, api_key="s3cret", allowed_ips="not-an-ip,10.0.0.0/8"
    )
    assert len(ws_auth.ALLOWED_IPS) == 1
    client = TestClient(app, client=("10.1.2.3", 5555))
    with client.websocket_connect("/ws/test?token=s3cret") as ws:
        assert ws.receive_text() == "authorized"


@pytest.mark.parametrize(
    "key,token,expected",
    [
        ("", "x", False),          # unset server key => never authorised
        ("k", None, False),        # no token
        ("k", "", False),          # empty token
        ("k", "wrong", False),
        ("k", "k", True),
    ],
)
def test_is_authorized_message(monkeypatch, key, token, expected):
    _, ws_auth = _build_app(monkeypatch, api_key=key)
    assert ws_auth.is_authorized_message(token) is expected
