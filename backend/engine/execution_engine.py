# backend/engine/execution_engine.py
"""
Execution Engine (paper mode):
- Consumes orders from Redis Stream STREAM_ORDERS
- Basic risk checks (per-strategy, global)
- Fills at last trade price cached by aggregator
- Updates positions, realized/unrealized PnL
- Publishes fills to STREAM_FILLS and CHAN_ORDERS (for UI)
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, Tuple, Optional, Any

import redis

from backend.bus.streams import (
    consume_stream,
    publish_stream,
    publish_pubsub,
    hgetall,
    hset,
    get as kv_get,
    set as kv_set,
    STREAM_ORDERS,
    STREAM_FILLS,
    CHAN_ORDERS,
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Simple env-based limits (override in .env as needed)
MAX_POS_USD_PER_STRAT = float(os.getenv("RISK_MAX_POS_PER_STRAT_USD", "25000"))
MAX_GROSS_USD         = float(os.getenv("RISK_MAX_GROSS_USD", "100000"))
SLIPPAGE_BPS          = float(os.getenv("EXEC_SLIPPAGE_BPS", "0"))   # e.g., "1.5" for 1.5bps
FEES_BPS              = float(os.getenv("EXEC_FEES_BPS", "0"))       # simple all-in fee model

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ---------- Helpers ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _parse_price_payload(data: Any) -> Optional[float]:
    """Accept str/bytes/json-string/number; return float price or None."""
    if data is None:
        return None
    # decode bytes-like
    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode()
        except Exception:
            return None
    # already a number
    if isinstance(data, (int, float)):
        try:
            return float(data)
        except Exception:
            return None
    if not isinstance(data, str):
        return None

    s = data.strip()
    if not s:
        return None

    # try raw float string first
    try:
        return float(s)
    except Exception:
        pass

    # try JSON string
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "price" in obj:
            return float(obj["price"])
        # allow direct numeric in JSON (e.g., "123.4")
        if isinstance(obj, (int, float)):
            return float(obj)
    except Exception:
        return None
    return None

def _last_price(symbol: str) -> float | None:
    # Using direct redis call; decode_responses=True returns str|None
    data = r.hget("last_price", symbol)
    return _parse_price_payload(data)

def _positions_key() -> str:
    return "positions"  # API reads this

def _pos_strategy_key(strategy: str) -> str:
    return f"positions:by_strategy:{strategy}"

def _pnl_key() -> str:
    return "pnl"  # API reads this (total)

def _gross_exposure_usd() -> float:
    # Store as plain number (string), read as float
    v = kv_get("portfolio:gross_usd", "0")
    try:
        # kv_get may return dict or str depending on your helper; handle both
        if isinstance(v, dict):
            # in case older runs stored {"usd": x}
            return float(v.get("usd", 0.0))
        return float(v)
    except Exception:
        return 0.0

def _set_gross_exposure_usd(x: float) -> None:
    # Keep it consistent: store as plain stringified number
    kv_set("portfolio:gross_usd", f"{float(x)}")

def _strategy_notional_usd(strategy: str) -> float:
    v = r.hget("allocator:notional", strategy)
    if not v:
        return 0.0
    try:
        obj = json.loads(v) if isinstance(v, str) else v
        if isinstance(obj, dict):
            return float(obj.get("usd", 0.0))
        return float(obj)
    except Exception:
        return 0.0

def _apply_slippage_and_fees(exec_price: float, side: str) -> float:
    price = exec_price
    if SLIPPAGE_BPS:
        adj = exec_price * (SLIPPAGE_BPS / 10000.0)
        price = exec_price + (adj if side == "buy" else -adj)
    # fees as bps reduce proceeds or increase cost; we fold into price
    if FEES_BPS:
        fee_adj = exec_price * (FEES_BPS / 10000.0)
        price = price + (fee_adj if side == "buy" else -fee_adj)
    return price

# ---------- Positions/PnL ----------
def _load_pos(symbol: str) -> Dict:
    raw = r.hget(_positions_key(), symbol)
    if not raw:
        return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}

def _save_pos(p: Dict) -> None:
    r.hset(_positions_key(), p["symbol"], json.dumps(p))

def _load_pos_strategy(symbol: str, strategy: str) -> Dict:
    raw = r.hget(_pos_strategy_key(strategy), symbol)
    if not raw:
        return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}

def _save_pos_strategy(strategy: str, p: Dict) -> None:
    r.hset(_pos_strategy_key(strategy), p["symbol"], json.dumps(p))

def _update_position(symbol: str, side: str, qty: float, price: float) -> Tuple[Dict, float]:
    """
    Update aggregate position; return (new_position, realized_pnl_change)
    Long convention: qty > 0 is long; selling reduces qty.
    """
    pos = _load_pos(symbol)
    realized = 0.0

    if side == "buy":
        new_qty = pos["qty"] + qty
        if pos["qty"] <= 0:
            pos["avg_price"] = price if new_qty != 0 else 0.0
        else:
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * qty) / new_qty
        pos["qty"] = new_qty

    elif side == "sell":
        sell_qty = qty
        if pos["qty"] > 0:
            close_qty = min(pos["qty"], sell_qty)
            realized += (price - pos["avg_price"]) * close_qty
            pos["qty"] -= close_qty
            sell_qty -= close_qty
        if sell_qty > 0:
            # enter/increase short
            pos["avg_price"] = price
            pos["qty"] -= sell_qty
    else:
        raise ValueError("side must be 'buy' or 'sell'")

    pos["realized_pnl"] = pos.get("realized_pnl", 0.0) + realized
    _save_pos(pos)
    return pos, realized

def _update_position_strategy(strategy: str, symbol: str, side: str, qty: float, price: float) -> float:
    """Track per-strategy realized PnL similarly to aggregate."""
    pos = _load_pos_strategy(symbol, strategy)
    realized = 0.0

    if side == "buy":
        new_qty = pos["qty"] + qty
        if pos["qty"] <= 0:
            pos["avg_price"] = price if new_qty != 0 else 0.0
        else:
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * qty) / new_qty
        pos["qty"] = new_qty
    else:
        sell_qty = qty
        if pos["qty"] > 0:
            close_qty = min(pos["qty"], sell_qty)
            realized += (price - pos["avg_price"]) * close_qty
            pos["qty"] -= close_qty
            sell_qty -= close_qty
        if sell_qty > 0:
            pos["avg_price"] = price
            pos["qty"] -= sell_qty

    pos["realized_pnl"] = pos.get("realized_pnl", 0.0) + realized
    _save_pos_strategy(strategy, pos)
    return realized

def _unrealized_total() -> float:
    """Compute simple unrealized PnL across all symbols using last_price."""
    total_ur = 0.0
    allpos = hgetall(_positions_key())
    for sym, raw in allpos.items():
        try:
            p = json.loads(raw) if isinstance(raw, str) else raw
            lp = _last_price(sym)
            if lp is None:
                continue
            qty = float(p.get("qty", 0.0))
            ap = float(p.get("avg_price", 0.0))
            total_ur += (lp - ap) * qty
        except Exception:
            continue
    return total_ur

def _update_total_pnl() -> None:
    # realized (sum of positions' realized) + unrealized
    realized = 0.0
    allpos = hgetall(_positions_key())
    for _, raw in allpos.items():
        try:
            p = json.loads(raw) if isinstance(raw, str) else raw
            realized += float(p.get("realized_pnl", 0.0))
        except Exception:
            continue
    unreal = _unrealized_total()
    kv_set(_pnl_key(), {"realized": float(realized), "unrealized": float(unreal), "total": float(realized + unreal)})

# ---------- Risk checks ----------
def _would_breach_global(new_notional_abs: float) -> bool:
    gross = _gross_exposure_usd()
    return (gross + new_notional_abs) > MAX_GROSS_USD

def _effective_strategy_cap(strategy: str) -> float:
    """
    Cap is the lesser of:
      - env cap (MAX_POS_USD_PER_STRAT)
      - allocator-provided cap (if > 0), else env cap
    """
    alloc = _strategy_notional_usd(strategy)
    if alloc and alloc > 0:
        return min(alloc, MAX_POS_USD_PER_STRAT)
    return MAX_POS_USD_PER_STRAT

def _would_breach_strategy_cap(strategy: str, new_notional_abs: float) -> bool:
    cap = _effective_strategy_cap(strategy)
    used = r.hget("risk:used_by_strategy", strategy)
    used = float(used) if used else 0.0
    return (used + new_notional_abs) > cap

def _bump_usage(strategy: str, notional_abs: float) -> None:
    used = r.hget("risk:used_by_strategy", strategy)
    used = float(used) if used else 0.0
    r.hset("risk:used_by_strategy", strategy, used + notional_abs)

# ---------- Main order loop ----------
def _process_order(order: Dict) -> None:
    """
    Order dict expected fields:
      strategy, symbol, side ('buy'|'sell'), qty (float), [limit_price]
    """
    strategy = str(order.get("strategy"))
    symbol   = str(order.get("symbol"))
    side     = str(order.get("side")).lower()
    qty      = float(order.get("qty", 0.0))
    if not strategy or not symbol or side not in ("buy", "sell") or qty <= 0:
        return

    lp = _last_price(symbol)
    if lp is None:
        # Can't fill without a market price
        publish_pubsub(CHAN_ORDERS, {"event": "reject", "reason": "no_market_price", **order})
        return

    exec_price = _apply_slippage_and_fees(lp, side)
    notional_abs = abs(exec_price * qty)

    # Risk checks
    if _would_breach_global(notional_abs):
        publish_pubsub(CHAN_ORDERS, {"event": "reject", "reason": "global_cap", **order})
        return
    if _would_breach_strategy_cap(strategy, notional_abs):
        publish_pubsub(CHAN_ORDERS, {"event": "reject", "reason": "strategy_cap", **order})
        return

    # Fill
    fill_id = f"{symbol}-{_now_ms()}"
    # Update aggregate position
    _, realized_delta = _update_position(symbol, side, qty, exec_price)
    # Update per-strategy position
    _ = _update_position_strategy(strategy, symbol, side, qty, exec_price)

    # Track exposures
    _bump_usage(strategy, notional_abs)
    _set_gross_exposure_usd(_gross_exposure_usd() + notional_abs)

    # Update PnL snapshot
    _update_total_pnl()

    fill = {
        "ts_ms": _now_ms(),
        "fill_id": fill_id,
        "symbol": symbol,
        "qty": qty,
        "price": exec_price,
        "strategy": strategy,
        "side": side,
        "realized_delta": realized_delta,
    }

    # Publish to stream (durable) and pubsub (UI)
    publish_stream(STREAM_FILLS, fill)
    publish_pubsub(CHAN_ORDERS, {"event": "fill", **fill})

def run():
    # heartbeat
    kv_set("execution:alive", {"ts": int(time.time())})
    for _, order in consume_stream(STREAM_ORDERS, start_id="$", block_ms=1000, count=100):
        try:
            if isinstance(order, str):
                order = json.loads(order)
            _process_order(order)
        except Exception as e:
            publish_pubsub(CHAN_ORDERS, {"event": "error", "error": str(e), "order": order})

if __name__ == "__main__":
    run()
