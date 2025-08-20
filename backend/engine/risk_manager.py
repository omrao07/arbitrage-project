# backend/engine/risk_manager.py
"""
Risk Manager
- Central guardrail for orders before they hit the OMS.
- Two modes:
  1) Library:  check_order(order) -> (ok, reason)
  2) Gateway:  run_gateway(in_stream="orders.incoming", out_stream="orders")

Checks:
- Global kill switch / per-strategy / per-symbol / per-region disables
- Global gross cap (USD)
- Per-strategy cap (uses allocator:notional; falls back to env cap)
- Per-symbol cap
- Max order notional
- Max orders per minute (rate limit)
- Region policy compliance (via region_router.check_compliance if available)
- Optional: daily loss limits (set externally via pnl keys)
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

import redis

from backend.bus.streams import (
    consume_stream,
    publish_stream,
    publish_pubsub,
    STREAM_ORDERS,
    CHAN_ORDERS,
)

# Optional: policy compliance from region router
try:
    from backend.engine.region_router import infer_region, check_compliance
    HAS_POLICY = True
except Exception:
    HAS_POLICY = False

# ---------------- Config (override via .env) ----------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

RISK_MAX_GROSS_USD          = float(os.getenv("RISK_MAX_GROSS_USD", "100000"))
RISK_MAX_POS_PER_STRAT_USD  = float(os.getenv("RISK_MAX_POS_PER_STRAT_USD", "25000"))
RISK_MAX_PER_SYMBOL_USD     = float(os.getenv("RISK_MAX_PER_SYMBOL_USD", "50000"))
RISK_MAX_ORDER_NOTIONAL_USD = float(os.getenv("RISK_MAX_ORDER_NOTIONAL_USD", "25000"))
RISK_MAX_ORDERS_PER_MIN     = int(os.getenv("RISK_MAX_ORDERS_PER_MIN", "60"))

# soft daily loss guardrails (block if breached when values are negative thresholds)
RISK_MAX_DAILY_LOSS_PORTFOLIO = float(os.getenv("RISK_MAX_DAILY_LOSS_PORTFOLIO", "2000"))
RISK_MAX_DAILY_LOSS_PER_STRAT = float(os.getenv("RISK_MAX_DAILY_LOSS_PER_STRAT", "1000"))

INCOMING_STREAM = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")
OUT_STREAM = STREAM_ORDERS

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ---------------- Helpers ----------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _json_get_float(s: str, key: str, default: float = 0.0) -> float:
    try:
        return float(json.loads(s).get(key, default))
    except Exception:
        return default

def _gross_usd() -> float:
    v = r.get("portfolio:gross_usd")
    if not v: return 0.0
    return _json_get_float(v, "usd", 0.0)

def _allocator_cap(strategy: str) -> float:
    v = r.hget("allocator:notional", strategy)
    if not v: return 0.0
    try:
        return float(json.loads(v).get("usd", 0.0))
    except Exception:
        return 0.0

def _used_by_strategy(strategy: str) -> float:
    v = r.hget("risk:used_by_strategy", strategy)
    try:
        return float(v) if v else 0.0
    except Exception:
        return 0.0

def _used_by_symbol(symbol: str) -> float:
    v = r.hget("risk:used_by_symbol", symbol)
    try:
        return float(v) if v else 0.0
    except Exception:
        return 0.0

def _bump_usage(strategy: str, symbol: str, usd: float) -> None:
    r.hincrbyfloat("risk:used_by_strategy", strategy, usd)
    r.hincrbyfloat("risk:used_by_symbol", symbol, usd)

def _orders_per_min(strategy: str) -> int:
    key = f"risk:orders_rate:{strategy}"
    now = _now_ms()
    window_ms = 60_000
    r.zremrangebyscore(key, 0, now - window_ms)
    return r.zcard(key)

def _touch_rate(strategy: str) -> None:
    key = f"risk:orders_rate:{strategy}"
    now = _now_ms()
    r.zadd(key, {str(now): now})

def _daily_loss_total() -> float:
    v = r.get("pnl:day_total")
    return _json_get_float(v, "total", 0.0) if v else 0.0

def _daily_loss_strategy(strategy: str) -> float:
    v = r.get(f"pnl:day_strategy:{strategy}")
    return _json_get_float(v, "total", 0.0) if v else 0.0

def _disabled(reason_key: str) -> bool:
    v = r.get(reason_key)
    return str(v).lower() in ("1", "true", "yes")

def _reject(reason: str, order: Dict) -> Tuple[bool, str]:
    publish_pubsub(CHAN_ORDERS, {"event": "reject", "reason": reason, **order})
    return False, reason

# ---------------- Core check ----------------
def check_order(order: Dict) -> Tuple[bool, str | None]:
    """
    order fields: strategy, symbol, side ('buy'/'sell'), qty, [price|limit_price], [region|venue]
    returns: (ok, reason_if_rejected)
    """
    strat = str(order.get("strategy", "")).strip()
    symbol = str(order.get("symbol", "")).strip().upper()
    side = str(order.get("side", "")).lower()
    qty = float(order.get("qty", 0.0) or 0.0)
    if not strat or not symbol or side not in ("buy", "sell") or qty <= 0:
        return False, "malformed"

    # kill switches
    if _disabled("risk:kill_all"):
        return False, "kill_all"
    if _disabled(f"risk:disable:{strat}"):
        return False, "strategy_disabled"
    if _disabled(f"risk:disable_symbol:{symbol}"):
        return False, "symbol_disabled"

    region = order.get("region")
    venue = order.get("venue")
    if not region and HAS_POLICY:
        region = infer_region(symbol, venue)
        order["region"] = region
    if region and _disabled(f"risk:disable_region:{region}"):
        return False, "region_disabled"

    # region compliance
    if HAS_POLICY:
        ok, reason = check_compliance(order, region or "US")
        if not ok:
            return False, reason or "policy_block"

    # notional calc (price optional here; OMS can reject later if missing)
    price = order.get("price") or order.get("limit_price") or order.get("mark_price")
    try:
        price = float(price) if price is not None else None
    except Exception:
        price = None
    notional_abs = abs(price * qty) if price is not None else 0.0

    # rate limit
    if _orders_per_min(strat) >= RISK_MAX_ORDERS_PER_MIN:
        return False, "rate_limited"

    # per-order notional cap
    if notional_abs > 0 and notional_abs > RISK_MAX_ORDER_NOTIONAL_USD:
        return False, "order_notional_cap"

    # global cap
    if notional_abs > 0 and (_gross_usd() + notional_abs) > RISK_MAX_GROSS_USD:
        return False, "global_cap"

    # per-strategy cap (allocator target preferred)
    cap_alloc = _allocator_cap(strat)
    cap_strat = cap_alloc if cap_alloc > 0 else RISK_MAX_POS_PER_STRAT_USD
    if notional_abs > 0 and (_used_by_strategy(strat) + notional_abs) > cap_strat:
        return False, "strategy_cap"

    # per-symbol cap
    if notional_abs > 0 and (_used_by_symbol(symbol) + notional_abs) > RISK_MAX_PER_SYMBOL_USD:
        return False, "symbol_cap"

    # soft daily loss stops (block if thresholds configured as negatives and breached)
    if RISK_MAX_DAILY_LOSS_PORTFOLIO < 0 and _daily_loss_total() <= RISK_MAX_DAILY_LOSS_PORTFOLIO:
        return False, "daily_loss_portfolio"
    if RISK_MAX_DAILY_LOSS_PER_STRAT < 0 and _daily_loss_strategy(strat) <= RISK_MAX_DAILY_LOSS_PER_STRAT:
        return False, "daily_loss_strategy"

    return True, None

# ---------------- Gateway service ----------------
def run_gateway(in_stream: str = INCOMING_STREAM, out_stream: str = OUT_STREAM) -> None:
    """
    Consume pre-risk orders from `in_stream`, validate, forward to `out_stream` if OK,
    publish rejections to CHAN_ORDERS for UI.
    """
    r.set("risk:alive", json.dumps({"ts": _now_ms()}))

    for _, order in consume_stream(in_stream, start_id="$", block_ms=1000, count=200):
        try:
            if isinstance(order, str):
                order = json.loads(order)

            ok, reason = check_order(order)
            if not ok:
                publish_pubsub(CHAN_ORDERS, {"event": "reject", "reason": reason, **order})
                continue

            # Provisional usage + rate touch (helps cap before OMS updates gross)
            price = order.get("price") or order.get("limit_price") or order.get("mark_price")
            try:
                price = float(price) if price is not None else None
            except Exception:
                price = None
            if price is not None:
                _bump_usage(str(order.get("strategy")), str(order.get("symbol")).upper(), abs(price * float(order.get("qty", 0.0))))

            _touch_rate(str(order.get("strategy")))
            publish_stream(out_stream, order)
            publish_pubsub(CHAN_ORDERS, {"event": "accepted", **order})

        except Exception as e:
            publish_pubsub(CHAN_ORDERS, {"event": "error", "component": "risk_gateway", "error": str(e)})

if __name__ == "__main__":
    run_gateway()