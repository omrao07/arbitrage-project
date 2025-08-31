# backend/risk/risk_policies.py
from __future__ import annotations

import collections
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Deque, List, Optional, Tuple

# -------- Optional YAML + Redis (safe fallbacks) -------------------------
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import redis  # type: ignore
    _R = redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                     port=int(os.getenv("REDIS_PORT","6379")),
                     decode_responses=True)
except Exception:
    _R = None

# -------- Helpers --------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)
def clamp(x: float, a: float, b: float) -> float: return max(a, min(b, x))

def ewma(prev: Optional[float], x: float, alpha: float) -> float:
    return x if prev is None else (1 - alpha) * prev + alpha * x

def pct(a: float, b: float) -> float:
    if b == 0: return 0.0
    return (a - b) / b

# -------- Data Models ----------------------------------------------------
@dataclass
class RiskLimits:
    # Global
    max_gross_notional: float = 5_000_000.0
    max_single_order_notional: float = 500_000.0
    max_single_order_qty: float = 100_000.0
    max_leverage: float = 5.0
    # Liquidity / venue
    min_adv_coverage: float = 0.005        # order_qty / ADV <= 0.5%
    max_spread_bps: float = 50.0
    # Concentration
    max_symbol_weight: float = 0.20        # 20% of NAV
    # Risk metrics
    max_var_1d_frac: float = 0.05          # 5% of NAV 1d VaR
    max_drawdown_frac: float = 0.10        # trailing peak-to-trough on NAV
    # Throttles
    min_ms_between_orders: int = 150
    max_orders_per_min: int = 120
    # Kill switch
    breach_hard_count: int = 3             # consecutive hard breaches -> halt 1 min
    halt_ms: int = 60_000

@dataclass
class OrderRequest:
    strategy: str
    symbol: str
    side: str             # "buy" | "sell"
    qty: float
    price_hint: Optional[float] = None
    venue: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    def notional(self) -> float: return abs(self.qty) * (self.last_price or self.avg_price or 0.0)

@dataclass
class PortfolioSnapshot:
    cash: float = 1_000_000.0
    nav: float = 1_000_000.0
    leverage: float = 1.0
    positions: Dict[str, Position] = field(default_factory=dict)
    adv: Dict[str, float] = field(default_factory=dict)          # symbol -> ADV shares
    spread_bps: Dict[str, float] = field(default_factory=dict)   # symbol -> current spread in bps
    symbol_weights: Dict[str, float] = field(default_factory=dict)  # symbol -> weight in NAV
    var_1d_frac: float = 0.02
    drawdown_frac: float = 0.0

@dataclass
class RiskDecision:
    ok: bool
    level: str                # "allow" | "warn" | "block" | "halt"
    reasons: List[str] = field(default_factory=list)
    soft_caps: Dict[str, Any] = field(default_factory=dict)  # e.g. {"max_child_qty": 1500}
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -------- Engine ---------------------------------------------------------
class RiskEngine:
    """
    Pre-trade risk checks + real-time policy enforcement.
    • Load limits from policy.yaml (section: risk_limits) or use defaults.
    • Track rolling order rate & breaches (soft/hard).
    • Enforce notional/qty/leverage/liquidity/VAR/drawdown/concentration.
    • Emit events to Redis Stream 'risk.events' and keep in-memory ring buffer.
    """

    def __init__(self, policy_path: Optional[str] = None):
        self.limits = RiskLimits()
        self._load(policy_path)
        self._last_order_ts = 0
        self._orders_window: Deque[int] = collections.deque(maxlen=500)
        self._breach_streak = 0
        self._halt_until: Optional[int] = None
        self._events: Deque[Dict[str, Any]] = collections.deque(maxlen=1000)

        # rolling EWMA for spread & slippage awareness (optional)
        self._ew_spread: Dict[str, float] = {}
        self._alpha_spread = 0.2

    # -------------- Public API --------------
    def check_pre_trade(self, order: OrderRequest, pf: PortfolioSnapshot) -> RiskDecision:
        ts = now_ms()
        reasons: List[str] = []
        soft_caps: Dict[str, Any] = {}

        # Kill switch / halt
        if self._halt_until and ts < self._halt_until:
            return self._block("halt", reasons + [f"trading_halted_until={self._halt_until}"], {"retry_after_ms": self._halt_until - ts})

        # Cooldown throttle
        delta = ts - self._last_order_ts
        if delta < self.limits.min_ms_between_orders:
            reasons.append(f"cooldown {delta}ms<{self.limits.min_ms_between_orders}ms")
            return self._block("block", reasons)

        # Rate limit per minute
        self._gc_orders(ts)
        if self._rate_per_min(ts) >= self.limits.max_orders_per_min:
            reasons.append("rate_limited_per_min")
            return self._block("block", reasons)

        # Per-order qty & notional
        px = float(order.meta.get("mark_price") or order.price_hint or pf.positions.get(order.symbol, Position(order.symbol)).last_price or 0.0)
        notional = abs(order.qty) * px
        if notional > self.limits.max_single_order_notional:
            reasons.append(f"order_notional>{self.limits.max_single_order_notional}")
        if abs(order.qty) > self.limits.max_single_order_qty:
            reasons.append(f"order_qty>{self.limits.max_single_order_qty}")

        # Global gross notional (approx post-trade)
        gross = self._gross_notional_after(order, pf, px)
        if gross > self.limits.max_gross_notional:
            reasons.append(f"gross_after>{self.limits.max_gross_notional}")

        # Leverage cap
        if pf.leverage > self.limits.max_leverage:
            reasons.append(f"leverage>{self.limits.max_leverage}")

        # Liquidity checks: ADV coverage and spread gate
        adv = pf.adv.get(order.symbol, 0.0)
        if adv > 0:
            coverage = abs(order.qty) / adv
            if coverage > self.limits.min_adv_coverage:
                reasons.append(f"adv_coverage>{self.limits.min_adv_coverage:.4f}")
        sp_bps = pf.spread_bps.get(order.symbol)
        if sp_bps is not None and sp_bps > self.limits.max_spread_bps:
            reasons.append(f"spread_bps>{self.limits.max_spread_bps}")

        # Concentration: symbol weight post-trade
        tgt_weight = self._symbol_weight_after(order, pf, px)
        if tgt_weight > self.limits.max_symbol_weight:
            reasons.append(f"symbol_weight>{self.limits.max_symbol_weight*100:.1f}%")

        # Portfolio risk: VaR & drawdown
        if pf.var_1d_frac > self.limits.max_var_1d_frac:
            reasons.append(f"var_1d>{self.limits.max_var_1d_frac*100:.1f}% nav")
        if pf.drawdown_frac > self.limits.max_drawdown_frac:
            reasons.append(f"drawdown>{self.limits.max_drawdown_frac*100:.1f}%")

        # Decision
        level = "allow"
        if reasons:
            # classify soft vs hard
            hard = any(
                r.startswith(("gross_after", "leverage", "drawdown", "var_1d", "order_notional"))
                for r in reasons
            )
            level = "block" if hard else "warn"

        # Soft guidance: reduce child size if spread high
        if sp_bps is not None and sp_bps > (0.5 * self.limits.max_spread_bps):
            # suggest a smaller child and post-only behavior
            soft_caps["max_child_qty"] = max(1.0, abs(order.qty) * 0.25)
            soft_caps["post_only"] = True

        dec = RiskDecision(ok=(level == "allow"), level=level, reasons=reasons, soft_caps=soft_caps,
                           meta={"ts_ms": ts, "symbol": order.symbol, "notional": round(notional, 2)})
        self._after_decision(dec, ts)
        return dec

    def on_fill(self, symbol: str, px: float, qty: float, pf: PortfolioSnapshot) -> None:
        # update EWMA spread tracker if present
        sbps = pf.spread_bps.get(symbol)
        if sbps is not None:
            self._ew_spread[symbol] = ewma(self._ew_spread.get(symbol), float(sbps), self._alpha_spread)

    def record_breach(self, decision: RiskDecision) -> None:
        """
        Call when a 'block' decision stops an order, to advance breach streak & kill-switch.
        """
        if decision.level in ("block", "halt"):
            self._breach_streak += 1
            if self._breach_streak >= self.limits.breach_hard_count:
                self._halt_until = now_ms() + self.limits.halt_ms
                self._emit("halt", {"reason": "breach_streak", "until": self._halt_until})
        else:
            # reset on clean allow
            self._breach_streak = 0

    def reset(self) -> None:
        self._breach_streak = 0
        self._halt_until = None
        self._events.clear()
        self._orders_window.clear()
        self._emit("reset", {})

    # -------------- Internals --------------
    def _gross_notional_after(self, order: OrderRequest, pf: PortfolioSnapshot, px: float) -> float:
        gross_now = sum(p.notional() for p in pf.positions.values())
        delta = abs(order.qty) * px
        return gross_now + delta

    def _symbol_weight_after(self, order: OrderRequest, pf: PortfolioSnapshot, px: float) -> float:
        nav = max(1e-9, pf.nav)
        pos = pf.positions.get(order.symbol, Position(order.symbol))
        after_qty = pos.qty + (order.qty if order.side.lower() == "buy" else -order.qty)
        after_notional = abs(after_qty) * (px or pos.last_price or pos.avg_price or 0.0)
        # naive: other holdings unchanged
        others = sum(p.notional() for s, p in pf.positions.items() if s != order.symbol)
        return (after_notional) / max(1e-9, (others + after_notional + pf.cash))

    def _rate_per_min(self, ts: int) -> int:
        one_min_ago = ts - 60_000
        while self._orders_window and self._orders_window[0] < one_min_ago:
            self._orders_window.popleft()
        return len(self._orders_window)

    def _gc_orders(self, ts: int) -> None:
        self._orders_window.append(ts)

    def _after_decision(self, dec: RiskDecision, ts: int) -> None:
        if dec.level == "allow":
            self._last_order_ts = ts
            self._breach_streak = 0
        elif dec.level in ("block", "halt"):
            self.record_breach(dec)
        self._emit("decision", dec.to_dict())

    def _block(self, level: str, reasons: List[str], soft_caps: Optional[Dict[str,Any]]=None) -> RiskDecision:
        dec = RiskDecision(ok=False, level=level, reasons=reasons, soft_caps=soft_caps or {}, meta={"ts_ms": now_ms()})
        self._after_decision(dec, now_ms())
        return dec

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        ev = {"ts_ms": now_ms(), "kind": kind, "payload": payload}
        self._events.append(ev)
        if _R is not None:
            try:
                _R.xadd("risk.events", {"event": json.dumps(ev)})
            except Exception:
                pass

    # -------------- Config --------------
    def _load(self, path: Optional[str]) -> None:
        if not path or yaml is None:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            section = data.get("risk_limits", {})
            if isinstance(section, dict):
                self.limits = RiskLimits(**({**asdict(self.limits), **section}))
        except Exception:
            # keep defaults on parse errors
            pass

# -------- Quick smoke test ------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    eng = RiskEngine("./policy.yaml") if os.path.exists("./policy.yaml") else RiskEngine()
    pf = PortfolioSnapshot(
        cash=1_000_000.0, nav=1_000_000.0, leverage=1.2,
        positions={"AAPL": Position("AAPL", qty=1_000, avg_price=180.0, last_price=190.0)},
        adv={"AAPL": 50_000_000}, spread_bps={"AAPL": 8.0},
        symbol_weights={"AAPL": 0.12}, var_1d_frac=0.03, drawdown_frac=0.04
    )
    order = OrderRequest(strategy="buy_dip", symbol="AAPL", side="buy", qty=10_000, price_hint=190.0)
    dec = eng.check_pre_trade(order, pf)
    print("DECISION:", dec.to_dict())