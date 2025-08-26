# backend/risk/policies.py
"""
Composable risk & sizing policies (stdlib-only).

What it does
------------
- Evaluate/Gate: check a Proposal (or ExecutionDecision-like) against rules.
- Transform/Size: scale or modify quantities/legs to fit risk budgets.
- Explainable: every policy returns a verdict with details.

Integrates with:
- agents/base.py: Proposal, OrderPlan, MarketContext
- coordinator.ExecutionDecision (shape-compatible dict supported)

Typical usage
-------------
from backend.risk.policies import (
    PolicyChain, MaxGrossNotional, PerSymbolCap, CooldownPolicy,
    VolTargetSizer, KellySizer, TradingHoursPolicy
)

chain = PolicyChain([
    TradingHoursPolicy(tz="America/New_York", open_hhmm=930, close_hhmm=1600),
    MaxGrossNotional(1_000_000),
    PerSymbolCap(max_leg_usd=150_000),
    CooldownPolicy(window_sec=300),
    VolTargetSizer(target_annual_vol=0.12),
])

ok, transformed, report = chain.apply(proposal_or_decision, context)
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Import your primitives
try:
    from agents.base import MarketContext, Proposal, OrderPlan, notional_usd, clamp # type: ignore
except Exception:
    # Minimal shims for type hints if imported standalone
    MarketContext = Any  # type: ignore
    Proposal = Any       # type: ignore
    OrderPlan = Any      # type: ignore
    def notional_usd(symbol, qty, prices, fx): return abs(qty) * float(prices.get(symbol, 1.0))  # type: ignore
    def clamp(v, lo, hi): return max(lo, min(hi, v))  # type: ignore


# ----------------------------- verdict model -----------------------------

@dataclass
class Verdict:
    name: str
    ok: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "details": dict(self.details)}


# ----------------------------- base policy --------------------------------

class Policy:
    """
    Base policy: override `evaluate()` or `transform()`.
    - evaluate(item, ctx) -> Verdict (pass/fail with details)
    - transform(item, ctx) -> (item2, details) (pure function; no side-effects)
    Apply order: evaluate -> transform for each policy in the chain.
    """

    name: str = "policy"

    def evaluate(self, item: Any, ctx: MarketContext) -> Verdict: # type: ignore
        return Verdict(self.name, ok=True, details={})

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        return item, {}

    # convenience
    def apply(self, item: Any, ctx: MarketContext) -> Tuple[Verdict, Any, Dict[str, Any]]: # type: ignore
        v = self.evaluate(item, ctx)
        out, info = self.transform(item, ctx) if v.ok else (item, {})
        return v, out, info


# ----------------------------- helpers --------------------------------------

def _iter_legs(obj: Any):
    """Yield mutable (index, leg_dict) over legs in Proposal or ExecutionDecision-like."""
    # Accept Proposal dataclass, or dict with "orders"/"legs"
    legs = getattr(obj, "orders", None)
    field_name = "orders"
    if legs is None:
        legs = getattr(obj, "legs", None)
        field_name = "legs"
    if legs is None and isinstance(obj, dict):
        legs = obj.get("orders") or obj.get("legs")
    if not legs:
        return field_name, []
    # normalize to dict-like (so policies can mutate safely if it's a dict)
    out = []
    for i, L in enumerate(legs):
        if isinstance(L, dict):
            out.append((i, L))
        else:
            out.append((i, asdict(L)))
    return field_name, out

def _rebuild(obj: Any, field_name: str, new_legs: List[Dict[str, Any]]) -> Any:
    """Return an object of same 'shape' with replaced legs."""
    if isinstance(obj, dict):
        k = "orders" if field_name == "orders" else "legs"
        new = dict(obj)
        new[k] = new_legs
        return new
    # Proposal-like dataclass
    try:
        from dataclasses import replace
        if field_name == "orders":
            return replace(obj, orders=[OrderPlan(**d) for d in new_legs]) # type: ignore
        else:
            # ExecutionDecision-like: keep as dict to avoid importing its class
            new = {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
            new["legs"] = new_legs
            return new
    except Exception:
        return obj


def _gross_notional(legs: Sequence[Dict[str, Any]], ctx: MarketContext) -> float: # type: ignore
    g = 0.0
    for L in legs:
        g += notional_usd(L["symbol"], L.get("qty", 0.0), ctx.prices, ctx.fx_usd_per_base)
    return g

def _per_symbol_notional(legs: Sequence[Dict[str, Any]], ctx: MarketContext) -> Dict[str, float]: # type: ignore
    out: Dict[str, float] = {}
    for L in legs:
        n = notional_usd(L["symbol"], L.get("qty", 0.0), ctx.prices, ctx.fx_usd_per_base)
        out[L["symbol"]] = out.get(L["symbol"], 0.0) + n
    return out


# ----------------------------- concrete policies -----------------------------

@dataclass
class MaxGrossNotional(Policy):
    """Gate & scale to respect a gross USD limit."""
    max_usd: float
    scale_down: bool = True

    name: str = "max_gross_notional"

    def evaluate(self, item: Any, ctx: MarketContext) -> Verdict: # type: ignore
        _, legs = _iter_legs(item)
        gross = _gross_notional([l for _, l in legs], ctx)
        ok = gross <= self.max_usd
        return Verdict(self.name, ok=ok, details={"gross": gross, "max": self.max_usd})

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        field, legs = _iter_legs(item)
        gross = _gross_notional([l for _, l in legs], ctx)
        if gross <= self.max_usd or not self.scale_down or gross <= 0.0:
            return item, {}
        k = self.max_usd / gross
        new_legs = []
        for _, L in legs:
            L2 = dict(L); L2["qty"] = L.get("qty", 0.0) * k
            new_legs.append(L2)
        return _rebuild(item, field, new_legs), {"scaled_by": k, "old_gross": gross, "new_gross": self.max_usd}


@dataclass
class PerSymbolCap(Policy):
    """Cap per-symbol gross notional; scales down offending legs proportionally."""
    max_leg_usd: float
    name: str = "per_symbol_cap"

    def evaluate(self, item: Any, ctx: MarketContext) -> Verdict: # type: ignore
        _, legs = _iter_legs(item)
        per = _per_symbol_notional([l for _, l in legs], ctx)
        viol = {s: n for s, n in per.items() if n > self.max_leg_usd}
        return Verdict(self.name, ok=(len(viol) == 0), details={"violations": viol, "max": self.max_leg_usd})

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        field, legs = _iter_legs(item)
        adj: Dict[str, float] = {}
        new_legs = []
        for _, L in legs:
            n = notional_usd(L["symbol"], L.get("qty", 0.0), ctx.prices, ctx.fx_usd_per_base)
            if n > self.max_leg_usd and n > 0:
                k = self.max_leg_usd / n
                L2 = dict(L); L2["qty"] = L.get("qty", 0.0) * k
                new_legs.append(L2)
                adj[L["symbol"]] = k
            else:
                new_legs.append(dict(L))
        return _rebuild(item, field, new_legs), {"scaled": adj}


@dataclass
class ForbidSymbols(Policy):
    """Reject any plan that includes certain symbols."""
    blocked: Sequence[str]
    name: str = "forbid_symbols"

    def evaluate(self, item: Any, ctx: MarketContext) -> Verdict: # type: ignore
        _, legs = _iter_legs(item)
        bad = [L["symbol"] for _, L in legs if L["symbol"] in set(self.blocked)]
        return Verdict(self.name, ok=(len(bad) == 0), details={"blocked": bad})


@dataclass
class TradingHoursPolicy(Policy):
    """
    Allow only between [open_hhmm, close_hhmm] in a given timezone.
    Example: tz="America/New_York", open_hhmm=930, close_hhmm=1600
    """
    tz: str
    open_hhmm: int
    close_hhmm: int
    name: str = "trading_hours"

    def evaluate(self, item: Any, ctx: MarketContext) -> Verdict: # type: ignore
        try:
            import zoneinfo
            from datetime import datetime
            now = datetime.fromtimestamp(ctx.ts, tz=zoneinfo.ZoneInfo(self.tz))
            hhmm = now.hour * 100 + now.minute
            ok = (self.open_hhmm <= hhmm <= self.close_hhmm)
            return Verdict(self.name, ok=ok, details={"local": now.isoformat(), "hhmm": hhmm})
        except Exception as e:
            return Verdict(self.name, ok=False, details={"error": str(e)})


@dataclass
class CooldownPolicy(Policy):
    """
    Prevents re-trading the same symbol in a short window.
    Keeps an internal LRU map of last trade timestamps (seconds).
    """
    window_sec: float = 120.0
    _last_ts: Dict[str, float] = field(default_factory=dict)

    name: str = "cooldown"

    def evaluate(self, item: Any, ctx: MarketContext) -> Verdict: # type: ignore
        _, legs = _iter_legs(item)
        now = float(getattr(ctx, "ts", time.time()))
        viol: Dict[str, float] = {}
        for _, L in legs:
            sym = L["symbol"]
            last = self._last_ts.get(sym, 0.0)
            if now - last < self.window_sec:
                viol[sym] = self.window_sec - (now - last)
        ok = len(viol) == 0
        return Verdict(self.name, ok=ok, details={"cooldowns": viol})

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        # If passes, update timestamps
        _, legs = _iter_legs(item)
        now = float(getattr(ctx, "ts", time.time()))
        for _, L in legs:
            self._last_ts[L["symbol"]] = now
        return item, {}


@dataclass
class VolTargetSizer(Policy):
    """
    Scales all legs so that the plan's *estimated* annualized vol equals target.
    Requires per-symbol instantaneous vol in ctx.signals as f"vol_ann_{SYM}" (fraction).
    If missing, assumes a default vol (e.g., 20%).
    """
    target_annual_vol: float = 0.12
    default_vol: float = 0.20
    name: str = "vol_target_sizer"

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        field, legs = _iter_legs(item)
        if not legs:
            return item, {}
        # crude portfolio vol estimate: sqrt(sum (w_i^2 * vol_i^2)), assuming 0 corr
        # w_i ∝ notional; we'll scale all qties by 'k' to hit target
        prices = ctx.prices or {}
        sig = ctx.signals or {}
        num = 0.0  # sum (n_i^2 * vol_i^2)
        for _, L in legs:
            n = notional_usd(L["symbol"], L.get("qty", 0.0), prices, ctx.fx_usd_per_base)
            v = float(sig.get(f"vol_ann_{L['symbol']}", self.default_vol))
            num += (n * v) ** 2
        port_vol = math.sqrt(num) if num > 0 else 0.0
        if port_vol <= 0:
            return item, {}
        # Scaling k on notionals to achieve target: target = k * port_vol → k = target/port_vol
        k = self.target_annual_vol / port_vol
        if 0.0 < k < 1.0 or k > 1.0:
            new_legs = []
            for _, L in legs:
                L2 = dict(L); L2["qty"] = L.get("qty", 0.0) * k
                new_legs.append(L2)
            return _rebuild(item, field, new_legs), {"scaled_by": k, "prev_vol": port_vol, "target": self.target_annual_vol}
        return item, {}


@dataclass
class KellySizer(Policy):
    """
    Kelly fraction sizing per leg using expected edge and variance proxy.
    Expects per-leg meta:
      - meta["edge_bps"] or proposal/diagnostics["edge_bps_{SYM}"]
    And volatility in ctx.signals: f"vol_ann_{SYM}" (fraction).
    kelly ≈ edge / var  (simplified; edge in fraction, var ≈ vol^2)
    """
    max_fraction: float = 0.25
    floor_fraction: float = 0.02
    name: str = "kelly_sizer"

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        field, legs = _iter_legs(item)
        sig = ctx.signals or {}
        new_legs = []
        info: Dict[str, Any] = {"fractions": {}}
        cash = float((ctx.balances or {}).get("CASH", 0.0)) or 1.0
        for _, L in legs:
            sym = L["symbol"]
            edge_bps = float((L.get("meta") or {}).get("edge_bps", 0.0))
            vol = float(sig.get(f"vol_ann_{sym}", 0.25))  # 25% default
            var = max(1e-6, vol * vol)
            edge = edge_bps / 10_000.0
            f = clamp(edge / var, -self.max_fraction, self.max_fraction)
            # floor sizing (keep tiny positions meaningful)
            if 0 < abs(f) < self.floor_fraction:
                f = math.copysign(self.floor_fraction, f)
            # convert fraction of cash into qty via price
            px = float(ctx.prices.get(sym, 0.0)) or 1.0
            notional = abs(f) * cash
            qty = notional / px
            L2 = dict(L); L2["qty"] = qty
            new_legs.append(L2)
            info["fractions"][sym] = {"kelly": f, "edge_bps": edge_bps, "vol": vol}
        return _rebuild(item, field, new_legs), info


@dataclass
class TimeStopPolicy(Policy):
    """
    Adds/sets a 'ttl_sec' on each leg so downstream router/executor can cancel/expire.
    """
    ttl_sec: int = 300
    name: str = "time_stop"

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        field, legs = _iter_legs(item)
        new_legs = []
        for _, L in legs:
            meta = dict(L.get("meta", {}))
            meta["ttl_sec"] = int(self.ttl_sec)
            L2 = dict(L); L2["meta"] = meta
            new_legs.append(L2)
        return _rebuild(item, field, new_legs), {"ttl_sec": self.ttl_sec}


@dataclass
class DrawdownGuard(Policy):
    """
    Hard block when account drawdown exceeds a threshold.
    Requires ctx.signals["pnl_peak"] and ["pnl_equity"] (both in USD).
    """
    max_dd_frac: float = 0.1  # 10%
    name: str = "drawdown_guard"

    def evaluate(self, item: Any, ctx: MarketContext) -> Verdict: # type: ignore
        sig = ctx.signals or {}
        peak = float(sig.get("pnl_peak", 0.0))
        eq = float(sig.get("pnl_equity", 0.0))
        dd = (peak - eq) / max(1.0, peak) if peak > 0 else 0.0
        ok = dd <= self.max_dd_frac
        return Verdict(self.name, ok=ok, details={"dd": dd, "max": self.max_dd_frac})


@dataclass
class ThrottlePolicy(Policy):
    """
    Dampens position sizes when market 'heat' is high.
    Expects ctx.signals["heat_z"] (e.g., cross-asset vol z). Scales qty by 1/(1+α*max(0, heat)).
    """
    alpha: float = 0.35
    name: str = "throttle"

    def transform(self, item: Any, ctx: MarketContext) -> Tuple[Any, Dict[str, Any]]: # type: ignore
        hz = float((ctx.signals or {}).get("heat_z", 0.0))
        if hz <= 0:
            return item, {}
        k = 1.0 / (1.0 + self.alpha * hz)
        field, legs = _iter_legs(item)
        new_legs = []
        for _, L in legs:
            L2 = dict(L); L2["qty"] = L.get("qty", 0.0) * k
            new_legs.append(L2)
        return _rebuild(item, field, new_legs), {"scaled_by": k, "heat_z": hz}


# ----------------------------- chain -----------------------------------------

@dataclass
class PolicyChain:
    """
    Runs policies in order. If any evaluate() fails, stops and returns the failure.
    Otherwise, applies all transforms sequentially.

    apply(item, ctx) -> (ok, transformed_item, report)
    """
    policies: List[Policy]

    def apply(self, item: Any, ctx: MarketContext) -> Tuple[bool, Any, Dict[str, Any]]: # type: ignore
        report: Dict[str, Any] = {"verdicts": [], "transforms": []}
        cur = item
        # evaluate gates
        for pol in self.policies:
            v = pol.evaluate(cur, ctx)
            report["verdicts"].append(v.to_dict())
            if not v.ok:
                report["blocked_by"] = pol.name
                return False, cur, report
            # Transform after a pass (allows gate-dependent transforms)
            cur, info = pol.transform(cur, ctx)
            if info:
                report["transforms"].append({pol.name: info})
        return True, cur, report


# ----------------------------- tiny demo -------------------------------------

if __name__ == "__main__":
    # Minimal smoke test with a Proposal-like dict
    ctx = type("CTX", (), {"ts": time.time(),
                           "prices": {"AAPL": 210.0, "EURUSD": 1.09},
                           "fx_usd_per_base": {},
                           "balances": {"CASH": 100_000},
                           "signals": {"vol_ann_AAPL": 0.25, "pnl_peak": 120_000, "pnl_equity": 110_000,
                                       "heat_z": 1.2}})()

    prop = {
        "orders": [
            {"symbol": "AAPL", "side": "BUY", "qty": 100, "meta": {}},
            {"symbol": "EURUSD", "side": "SELL", "qty": 50_000, "meta": {}},
        ],
        "thesis": "demo",
    }

    chain = PolicyChain([
        DrawdownGuard(max_dd_frac=0.15),
        MaxGrossNotional(50_000, scale_down=True),
        PerSymbolCap(30_000),
        ThrottlePolicy(alpha=0.5),
        VolTargetSizer(target_annual_vol=0.10),
        TimeStopPolicy(ttl_sec=120),
    ])

    ok, out, rep = chain.apply(prop, ctx)
    print("OK:", ok)
    print("OUT:", out)
    print("REPORT:", rep)