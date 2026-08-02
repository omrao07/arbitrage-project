"""
Fail-closed authentication for WebSocket endpoints.

Every WS endpoint that exposes engine state or an order path MUST call
`authenticate_ws` before `accept()`. On any doubt the connection is dropped.

The rule, stated once: a missing/misconfigured ENGINE_API_KEY is a REFUSAL,
not a bypass. The previous pattern

    if _ENGINE_API_KEY and token != _ENGINE_API_KEY:   # fail-OPEN
        await ws.close(...)

silently allowed every unauthenticated client whenever the env var was unset,
which is the posture SEBI's algo framework exists to prohibit.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from typing import Optional

from fastapi import WebSocket, status

log = logging.getLogger(__name__)

# Server-side shared secret. Empty => the server refuses ALL websocket traffic.
ENGINE_API_KEY = os.getenv("ENGINE_API_KEY", "")

# Optional CIDR/IP allowlist (SEBI: static-IP API connectivity).
# Empty => no IP restriction (token is still mandatory).
_RAW_ALLOWED = os.getenv("WS_ALLOWED_IPS", "")


def _parse_allowed_ips(raw: str) -> list[ipaddress._BaseNetwork]:
    nets: list[ipaddress._BaseNetwork] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            log.error("WS_ALLOWED_IPS: ignoring malformed entry %r", part)
    return nets


ALLOWED_IPS = _parse_allowed_ips(_RAW_ALLOWED)


def _extract_token(ws: WebSocket, query_token: Optional[str]) -> str:
    """Token from explicit arg, then ?token=, then Authorization: Bearer."""
    if query_token:
        return query_token
    qp = ws.query_params.get("token")
    if qp:
        return qp
    auth = ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _ip_allowed(ws: WebSocket) -> bool:
    if not ALLOWED_IPS:
        return True
    client = ws.client
    if client is None:
        return False
    try:
        peer = ipaddress.ip_address(client.host)
    except ValueError:
        return False
    return any(peer in net for net in ALLOWED_IPS)


async def authenticate_ws(ws: WebSocket, token: Optional[str] = None) -> bool:
    """
    Fail-closed WS gate. Returns True only if the caller is authorised.

    On False the socket has already been closed; the caller must return
    immediately without calling accept().
    """
    if not ENGINE_API_KEY:
        log.error("ws auth: ENGINE_API_KEY unset — refusing connection (fail-closed)")
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    if not _ip_allowed(ws):
        peer = ws.client.host if ws.client else "unknown"
        log.warning("ws auth: peer %s not in WS_ALLOWED_IPS — refused", peer)
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    supplied = _extract_token(ws, token)
    # constant-time compare; empty token can never match a non-empty key
    if not supplied or not hmac.compare_digest(supplied, ENGINE_API_KEY):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    return True


def is_authorized_message(token: Optional[str]) -> bool:
    """
    Per-message re-validation. A handshake-only check is a fail-open in
    disguise once tokens rotate mid-session.
    """
    if not ENGINE_API_KEY or not token:
        return False
    return hmac.compare_digest(token, ENGINE_API_KEY)
