# backend/engine/risk_manager.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict

import math

from backend.bus.streams import consume_stream, publish_stream, hget, hset # type: ignore

# --------- Streams / Keys ---------
ORDERS_IN  = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")
ORDERS_OK  = os.getenv("RISK_APPROVED_STREAM", "orders.approved")
ORDERS_NOK = os.getenv("RISK_REJECTED_STREAM", "orders.rejected")
FILLS_BUS  = os.getenv("RISK_FILLS_STREAM", "fills")  # optional if you tee fills here

REDIS_KILL     = "risk:halt"             # "true" -> halt
REDIS_MARKS     = "px:last"               # HSET symbol -> last price
REDIS_POS       = "pos:live"              # HSET symbol -> qty
REDIS_PNL_DAY   = "pnl:day"               # HSET {realized, unrealized, fees, pnl}
REDIS_COOLDOWNS = "risk:cooldowns"        # HSET symbol|strategy -> until_ts
REDIS_SENTINEL  = "risk:seen_orders"      # HSET client_order_id -> ts

# --------- Config (env) ---------
DEF_CCY = os.getenv("BASE_CCY", "USD")

MAX_NOTIONAL_PER_ORDER = float(os.getenv("RISK_MAX_NOTIONAL", "100000"))  # per order
MAX_GROSS_NOTIONAL     = float(os.getenv("RISK_MAX_GROSS", "1000000"))    # abs(sum(qty*px))
MAX_NET_NOTIONAL       = float(os.getenv("RISK_MAX_NET", "500000"))       # abs(sum signed)
MAX_TURNOVER_DAY       = float(os.getenv("RISK_MAX_TURNOVER_DAY", "2000000"))
MAX_POS_PER_SYMBOL     = float(os.getenv("RISK_MAX_POS_PER_SYMBOL", "1000"))  # abs units
MAX_POS_PER_STRAT      = float(os.getenv("RISK_MAX_POS_PER_STRAT", "5000"))

MAX_DAY_DRAWDOWN_PCT   = float(os.getenv("RISK_MAX_DD_PCT", "0.05"))      # stop if -5%
LOSS_COOLDOWN_SEC      = int(os.getenv("RISK_LOSS_COOLDOWN_SEC", "300"))  # 5 min

EXCH_HOURS_LOCAL = os.getenv("RISK_EXCH_HOURS", "0930-1600")  # simple HHMM-HHMM
ALLOW_AFTER_HOURS = os.getenv("RISK_ALLOW_AH", "false").lower() == "true"

# --------- Helpers ---------
def _now() -> int:
    return int(time.time())

def _in_hours() -> bool:
    if ALLOW_AFTER_HOURS:
        return True
    try:
        hhmm = time.strftime("%H%M")
        start, end = EXCH_HOURS_LOCAL.split("-")
        return start <= hhmm <= end
    except Exception:
        return True

def _ceil_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.ceil(x / step) * step

# --------- State ---------
@dataclass
class Exposure:
    pos_by_symbol: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    pos_by_strategy: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    gross_notional: float = 0.0
    net_notional: float = 0.0
    turnover_today: float = 0.0
    day_start_equity: Optional[float] = None   # if you store account equity elsewhere

# --------- Main Risk Manager ---------
class RiskManager:
    def __init__(self):
        self.exp = Exposure()

    # ---- external updates --------------------------------------------------
    def on_fill(self, fill: Dict[str, Any]) -> None:
        """Update exposures from a fill event."""
        sym = str(fill.get("symbol", "")).upper()
        if not sym:
            return
        side = str(fill.get("side", "")).lower()   # 'buy'|'sell'
        qty  = float(fill.get("qty", 0.0) or 0.0)
        px   = float(fill.get("price", 0.0) or 0.0)
        strat = str(fill.get("strategy", "")).lower()

        signed = qty if side == "buy" else -qty
        self.exp.pos_by_symbol[sym] += signed
        if strat:
            self.exp.pos_by_strategy[strat] += signed

        notional = abs(qty * px)
        self.exp.turnover_today += notional

        # persist lightweight views
        hset(REDIS_POS, sym, self.exp.pos_by_symbol[sym])

    def on_price(self, symbol: str, last: float) -> None:
        """Store last marks for notional checks / unrealized estimation."""
        if last and last > 0:
            hset(REDIS_MARKS, symbol.upper(), last)

    # ---- validation --------------------------------------------------------
    def validate(self, order: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Return (ok, reason, possibly_adjusted_order).
        Expects: symbol, side('buy'|'sell'), qty, strategy, limit_price/mark_price(optional)
        """
        # kill switch
        if (hget(REDIS_KILL, "flag") or "").lower() == "true":
            return False, "HALTED", order

        if not _in_hours():
            return False, "OUT_OF_HOURS", order

        # idempotency (optional client_order_id)
        coid = order.get("client_order_id")
        if coid:
            if hget(REDIS_SENTINEL, str(coid)):
                return False, "DUPLICATE", order
            hset(REDIS_SENTINEL, str(coid), _now())

        sym  = str(order.get("symbol", "")).upper()
        side = str(order.get("side", "")).lower()
        qty  = float(order.get("qty", 0.0) or 0.0)
        typ  = str(order.get("typ", "market")).lower()
        strat= str(order.get("strategy", "")).lower()
        limit= order.get("limit_price")
        mark = order.get("mark_price")

        if not sym or side not in ("buy","sell") or qty <= 0:
            return False, "BAD_ORDER", order

        # price/notional reference
        px_ref = None
        if limit and float(limit) > 0:
            px_ref = float(limit)
        elif mark and float(mark) > 0:
            px_ref = float(mark)
        else:
            # lookup from marks store if available
            last = hget(REDIS_MARKS, sym)
            px_ref = float(last) if last else None

        # per-order notional
        if px_ref:
            notional = qty * px_ref
            if notional > MAX_NOTIONAL_PER_ORDER:
                scale = MAX_NOTIONAL_PER_ORDER / max(px_ref, 1e-9)
                new_qty = max(0.0, min(qty, scale))
                if new_qty <= 0.0:
                    return False, "MAX_NOTIONAL_PER_ORDER", order
                order = dict(order, qty=new_qty)

        # per-symbol position cap (projected)
        current = self.exp.pos_by_symbol.get(sym, float(hget(REDIS_POS, sym) or 0.0))
        signed = qty if side == "buy" else -qty
        projected = abs(current + signed)
        if projected > MAX_POS_PER_SYMBOL:
            return False, "MAX_POS_PER_SYMBOL", order

        # per-strategy cap
        if strat:
            s_cur = self.exp.pos_by_strategy.get(strat, 0.0)
            if abs(s_cur + signed) > MAX_POS_PER_STRAT:
                return False, "MAX_POS_PER_STRATEGY", order

        # gross/net notional caps (approx)
        if px_ref:
            gross = float(self.exp.gross_notional)
            net   = float(self.exp.net_notional)
            gross += abs(qty * px_ref)
            net   += (qty * px_ref) if side == "buy" else -(qty * px_ref)
            if gross > MAX_GROSS_NOTIONAL:
                return False, "MAX_GROSS_NOTIONAL", order
            if abs(net) > MAX_NET_NOTIONAL:
                return False, "MAX_NET_NOTIONAL", order

        # turnover cap
        if (self.exp.turnover_today + (qty * (px_ref or 0.0))) > MAX_TURNOVER_DAY:
            return False, "MAX_TURNOVER_DAY", order

        # daily loss / drawdown guard (rough: use pnl:day if populated)
        pnl_tot = float((hget(REDIS_PNL_DAY, "pnl") or 0.0))
        eq0 = float((hget(REDIS_PNL_DAY, "equity_start") or 0.0) or 0.0)
        if eq0 > 0 and pnl_tot < 0:
            if abs(pnl_tot) / eq0 >= MAX_DAY_DRAWDOWN_PCT:
                # activate cooldown
                until = _now() + LOSS_COOLDOWN_SEC
                hset(REDIS_COOLDOWNS, "global", until)
                return False, "DAILY_DRAWDOWN", order

        # cooldowns
        global_cd = int(hget(REDIS_COOLDOWNS, "global") or 0)
        if global_cd and _now() < global_cd:
            return False, "COOLDOWN", order

        return True, "OK", order

    # ---- loop --------------------------------------------------------------
    def run(self) -> None:
        """
        Consume ORDERS_IN, apply checks, publish to approved/rejected.
        """
        for _id, ord_payload in consume_stream(ORDERS_IN, start_id="$", block_ms=1000, count=200):
            try:
                ok, reason, adj = self.validate(ord_payload)
                if ok:
                    publish_stream(ORDERS_OK, dict(adj, risk="pass", reason=reason, ts_ms=_now()*1000))
                else:
                    publish_stream(ORDERS_NOK, dict(ord_payload, risk="fail", reason=reason, ts_ms=_now()*1000))
            except Exception as e:
                publish_stream(ORDERS_NOK, dict(ord_payload, risk="error", reason=str(e), ts_ms=_now()*1000))

# --------- Entrypoint ---------
def main():
    rm = RiskManager()
    rm.run()

if __name__ == "__main__":
    main()