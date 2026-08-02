# backend/api/ws_live.py
"""
WebSocket endpoint for live trading dashboard (/ws/live).
Pushes engine status, signals, risk gates, positions, P&L, ticks to connected clients.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.ws_auth import authenticate_ws, is_authorized_message

log = logging.getLogger(__name__)

router = APIRouter()

# ---- Optional Redis async client ------------------------------------------
try:
    from redis.asyncio import Redis as AsyncRedis
    _HAVE_REDIS = True
except ImportError:
    AsyncRedis = None  # type: ignore
    _HAVE_REDIS = False

_REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
_REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
# Prefer explicit REDIS_URL if set; otherwise build from components
REDIS_URL = os.getenv("REDIS_URL") or (
    f"redis://:{_REDIS_PASSWORD}@{_REDIS_HOST}:{_REDIS_PORT}/0"
    if _REDIS_PASSWORD
    else f"redis://{_REDIS_HOST}:{_REDIS_PORT}/0"
)
BROADCAST_INTERVAL_MS = int(os.getenv("WS_BROADCAST_INTERVAL_MS", "500"))


# ---- Connection manager ---------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        log.info("WS client connected; total=%d", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        log.info("WS client disconnected; total=%d", len(self._clients))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead = set()
        payload = json.dumps(message)
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._clients.discard(ws)


_manager = ConnectionManager()


# ---- Redis helpers --------------------------------------------------------

def _mk_redis() -> Optional[Any]:
    if not _HAVE_REDIS:
        return None
    try:
        return AsyncRedis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        return None


async def _hgetall(r: Any, key: str) -> Dict[str, str]:
    if r is None:
        return {}
    try:
        return await r.hgetall(key) or {}
    except Exception:
        return {}


def _safe_json(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


# ---- Data fetchers --------------------------------------------------------

async def _fetch_engine_status(r: Any) -> Dict[str, Any]:
    signals_raw = await _hgetall(r, "strategy:signal")
    enabled_raw = await _hgetall(r, "strategy:enabled")
    await _hgetall(r, "strategy:drawdown")

    n_active = sum(1 for v in enabled_raw.values() if v == "true")
    scores = [
        _safe_json(v).get("score", 0)
        for v in signals_raw.values()
        if _safe_json(v)
    ]
    combined = sum(scores) / len(scores) if scores else 0.0

    # Try to get daily P&L from Redis
    pnl_raw = await _hgetall(r, "engine:pnl")
    daily_pnl = float(pnl_raw.get("daily", 0.0))
    drawdown = float(pnl_raw.get("drawdown", 0.0))

    return {
        "type": "engine_status",
        "payload": {
            "running": n_active > 0,
            "n_strategies": n_active,
            "combined_score": round(combined, 4),
            "daily_pnl": daily_pnl,
            "drawdown": drawdown,
        },
    }


async def _fetch_signals(r: Any) -> list[Dict[str, Any]]:
    signals_raw = await _hgetall(r, "strategy:signal")
    enabled_raw = await _hgetall(r, "strategy:enabled")
    vol_raw = await _hgetall(r, "strategy:vol")
    dd_raw = await _hgetall(r, "strategy:drawdown")
    meta_raw = await _hgetall(r, "strategy:meta")

    msgs = []
    for name, v_str in signals_raw.items():
        v = _safe_json(v_str)
        if not isinstance(v, dict):
            continue
        vol_v = _safe_json(vol_raw.get(name, "{}")) or {}
        dd_v = _safe_json(dd_raw.get(name, "{}")) or {}
        meta_v = _safe_json(meta_raw.get(name, "{}")) or {}
        msgs.append({
            "type": "signal",
            "payload": {
                "name": name,
                "score": v.get("score", 0),
                "vol": vol_v.get("vol", 0.2),
                "drawdown": dd_v.get("dd", 0),
                "ts_ms": int(time.time() * 1000),
                "enabled": enabled_raw.get(name, "true") == "true",
                "region": meta_v.get("region"),
                "tags": meta_v.get("tags", []),
            },
        })
    return msgs


async def _fetch_risk_gates(r: Any) -> Optional[Dict[str, Any]]:
    gates_raw = await _hgetall(r, "risk:gates")
    if not gates_raw:
        return None
    gates = []
    for gate_name, v_str in gates_raw.items():
        v = _safe_json(v_str) or {}
        gates.append({
            "gate": gate_name,
            "ok": v.get("ok", True),
            "reason": v.get("reason"),
        })
    return {"type": "risk_gates", "payload": {"gates": gates}}


async def _fetch_positions(r: Any) -> list[Dict[str, Any]]:
    pos_raw = await _hgetall(r, "positions")
    msgs = []
    for symbol, v_str in pos_raw.items():
        v = _safe_json(v_str)
        if not isinstance(v, dict):
            continue
        msgs.append({
            "type": "position",
            "payload": {
                "symbol": symbol,
                "qty": v.get("qty", 0),
                "avg_px": v.get("avg_px", 0),
                "current_px": v.get("current_px", 0),
                "pnl": v.get("pnl", 0),
                "notional": v.get("notional", 0),
                "strategy": v.get("strategy", ""),
            },
        })
    return msgs


async def _fetch_pnl(r: Any) -> Optional[Dict[str, Any]]:
    pnl_raw = await _hgetall(r, "engine:pnl")
    if not pnl_raw:
        return None
    import datetime
    return {
        "type": "pnl",
        "payload": {
            "date": datetime.date.today().isoformat(),
            "realized": float(pnl_raw.get("realized", 0.0)),
            "unrealized": float(pnl_raw.get("unrealized", 0.0)),
            "fees": float(pnl_raw.get("fees", 0.0)),
            "net": float(pnl_raw.get("daily", 0.0)),
            "cumulative_net": float(pnl_raw.get("cumulative", 0.0)),
            "drawdown": float(pnl_raw.get("drawdown", 0.0)),
            "hwm": float(pnl_raw.get("hwm", 0.0)),
        },
    }


async def _fetch_india_status(r: Any) -> Optional[Dict[str, Any]]:
    india_raw = await _hgetall(r, "india:status")
    if not india_raw:
        return None
    fo_ban_str = india_raw.get("fo_ban_list", "[]")
    circuit_str = india_raw.get("circuit_halted", "[]")
    fo_ban = _safe_json(fo_ban_str) or []
    circuit = _safe_json(circuit_str) or []
    return {
        "type": "india_status",
        "payload": {
            "is_open": india_raw.get("is_open", "false") == "true",
            "next_event": india_raw.get("next_event", ""),
            "circuit_halted": circuit,
            "fo_ban_list": fo_ban,
            "margin_used": float(india_raw.get("margin_used", 0.0)),
            "margin_available": float(india_raw.get("margin_available", 0.0)),
            "vix": float(india_raw.get("vix", 0.0)),
            "pcr": float(india_raw.get("pcr", 0.0)),
            "regime": india_raw.get("regime", "unknown"),
        },
    }


# ---- Broadcast loop -------------------------------------------------------

_last_heartbeat_ts: float = 0.0

async def _broadcast_loop() -> None:
    global _last_heartbeat_ts
    r = _mk_redis()
    interval = BROADCAST_INTERVAL_MS / 1000.0

    while True:
        await asyncio.sleep(interval)
        if not _manager._clients:
            continue
        try:
            engine_msg = await _fetch_engine_status(r)
            await _manager.broadcast(engine_msg)

            for sig_msg in await _fetch_signals(r):
                await _manager.broadcast(sig_msg)

            gates_msg = await _fetch_risk_gates(r)
            if gates_msg:
                await _manager.broadcast(gates_msg)

            for pos_msg in await _fetch_positions(r):
                await _manager.broadcast(pos_msg)

            pnl_msg = await _fetch_pnl(r)
            if pnl_msg:
                await _manager.broadcast(pnl_msg)

            india_msg = await _fetch_india_status(r)
            if india_msg:
                await _manager.broadcast(india_msg)

            # Heartbeat every 30s
            now = time.time()
            if now - _last_heartbeat_ts >= 30.0:
                _last_heartbeat_ts = now
                await _manager.broadcast({"type": "heartbeat", "payload": {"t": now}})

        except Exception as exc:
            log.warning("broadcast error: %s", exc)


_broadcast_task: Optional[asyncio.Task] = None


# ---- WebSocket endpoint ---------------------------------------------------

@router.websocket("/ws/live")
async def ws_live(ws: WebSocket, token: str = ""):
    global _broadcast_task

    # Fail-closed: an unset ENGINE_API_KEY refuses the connection, never bypasses it.
    if not await authenticate_ws(ws, token):
        return

    await _manager.connect(ws)

    # Start broadcast loop on first connection
    if _broadcast_task is None or _broadcast_task.done():
        _broadcast_task = asyncio.create_task(_broadcast_loop())

    try:
        while True:
            # Handle incoming messages (pings, strategy toggles, etc.)
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                continue

            op = msg.get("op")
            if op == "ping":
                await ws.send_text(json.dumps({"type": "pong", "payload": {"t": time.time()}}))
            elif op:
                # Any state-mutating op must re-present a valid token: tokens
                # expire mid-session, so handshake-only auth is fail-open.
                if not is_authorized_message(msg.get("token")):
                    await ws.close(code=1008)
                    _manager.disconnect(ws)
                    return
                log.warning("ws_live: unhandled op %r", op)

    except WebSocketDisconnect:
        _manager.disconnect(ws)
    except Exception:
        _manager.disconnect(ws)
