# backend/ai/agents/concrete/execution_agent.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple

# ============================================================
# Optional framework imports (with safe fallbacks)
# ============================================================

# BaseAgent (optional)
try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name: str = "execution_agent"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# Broker interface (submit/cancel/replace). Expected minimal API:
#   submit_order(symbol, side, qty, order_type="market", limit_price=None, tag=None) -> order_id
try:
    from ..skills.trading.broker_interface import submit_order  # type: ignore
except Exception:
    def submit_order(symbol: str, side: str, qty: float, order_type: str = "market",
                     limit_price: Optional[float] = None, tag: Optional[str] = None) -> str:
        # Fallback: pretend we placed it; log to stdout.
        oid = f"SIM-{int(time.time()*1e6)}"
        print(f"[SIM] submit {side} {qty} {symbol} type={order_type} px={limit_price} tag={tag} -> {oid}")
        return oid

# Live/agg volumes & prices
try:
    from ..skills.market.quotes import get_candles  # type: ignore
except Exception:
    def get_candles(symbol: str, *, interval: str = "1m", lookback: int = 120) -> List[Dict[str, Any]]:
        # Tiny synthetic 1m candles with rising volume
        now = int(time.time() * 1000)
        px = 100.0
        out: List[Dict[str, Any]] = []
        for i in range(lookback):
            vol = 10_000 * (1 + 0.02 * i)
            px *= (1 + (0.0002 if i % 7 else -0.001))
            out.append({"ts": now - (lookback - i) * 60_000, "o": px, "h": px*1.001, "l": px*0.999, "c": px, "v": vol})
        return out

# TCA (optional cost model)
try:
    from ..skills.trading.tca import estimate_cost_bps  # type: ignore
except Exception:
    def estimate_cost_bps(symbol: str, side: str, qty: float, px: float, venue: Optional[str] = None) -> float:
        # naive: 2 bps + sqrt slippage vs ADV proxy
        return 2.0 + min(25.0, 10.0 * math.sqrt(max(0.0, qty) / 1_000_000.0))

# Risk gates (optional)
try:
    from ..policies.rules.risk_policies import check_gates  # type: ignore
except Exception:
    def check_gates(symbol: str, side: str, qty: float, px: float, *, context: Dict[str, Any]) -> Tuple[bool, str]:
        # allow unless explicit flag in context
        if context.get("kill_switch", False):
            return (False, "Kill-switch active")
        return (True, "OK")

# ============================================================
# Data models
# ============================================================

Algo = Literal["TWAP", "POV", "VWAP"]
Side = Literal["buy", "sell"]

@dataclass
class ExecutionTarget:
    symbol: str
    side: Side
    qty: float
    limit_price: Optional[float] = None      # if set & order_type='limit'
    order_type: Literal["market", "limit"] = "market"
    tag: Optional[str] = None

@dataclass
class Schedule:
    algo: Algo = "TWAP"
    duration_min: int = 30                   # total duration
    slice_minutes: int = 1                   # bar size for slicing
    max_participation: float = 0.15          # for POV (0..1)
    venue: Optional[str] = None
    allow_cross_spread_bps: float = 10.0     # guard for limit px offset
    sliding_limit: bool = False              # if True, follow last price within allow range

@dataclass
class ExecRequest:
    target: ExecutionTarget
    schedule: Schedule
    market: str = "CASH"                     # or "FUT", "CRYPTO"
    ref_symbol_for_vwap: Optional[str] = None  # defaults to target.symbol
    dry_run: bool = False
    notes: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)   # risk context, user, book, etc.

@dataclass
class SlicePlan:
    t_index: int
    when_ms: int
    px_ref: float
    qty_slice: float
    reason: str

@dataclass
class ExecReport:
    ok: bool
    algo: Algo
    symbol: str
    side: Side
    requested_qty: float
    filled_qty: float
    est_cost_bps: float
    slices: List[SlicePlan]
    messages: List[str]

# ============================================================
# Helpers
# ============================================================

def _sum(x): return float(sum(x))

def _last_close(candles: List[Dict[str, Any]]) -> float:
    return float(candles[-1]["c"]) if candles else float("nan")

def _cum(lst: List[float]) -> List[float]:
    out, s = [], 0.0
    for v in lst:
        s += v
        out.append(s)
    return out

def _cap_participation(q_target: float, adv_per_bar: float, max_participation: float) -> float:
    cap = max(0.0, max_participation) * max(1e-9, adv_per_bar)
    return min(q_target, cap)

# ============================================================
# Agent
# ============================================================

class ExecutionAgent(BaseAgent): # type: ignore
    """
    Deterministic execution (TWAP / POV / VWAP):
      - Builds a slice plan over a duration.
      - Applies risk gates and participation caps.
      - Emits market/limit orders via broker.
      - Returns an ExecReport with planned slices and cost estimate.
    """

    name = "execution_agent"

    # -------------- public API --------------

    def plan(self, req: ExecRequest | Dict[str, Any]) -> ExecRequest:
        if isinstance(req, ExecRequest):
            return req
        t = req.get("target", {})
        s = req.get("schedule", {})
        return ExecRequest(
            target=ExecutionTarget(
                symbol=t.get("symbol"),
                side=t.get("side", "buy"),
                qty=float(t.get("qty", 0)),
                limit_price=t.get("limit_price"),
                order_type=t.get("order_type", "market"),
                tag=t.get("tag"),
            ),
            schedule=Schedule(
                algo=s.get("algo", "TWAP"),
                duration_min=int(s.get("duration_min", 30)),
                slice_minutes=int(s.get("slice_minutes", 1)),
                max_participation=float(s.get("max_participation", 0.15)),
                venue=s.get("venue"),
                allow_cross_spread_bps=float(s.get("allow_cross_spread_bps", 10.0)),
                sliding_limit=bool(s.get("sliding_limit", False)),
            ),
            market=req.get("market", "CASH"),
            ref_symbol_for_vwap=req.get("ref_symbol_for_vwap"),
            dry_run=bool(req.get("dry_run", False)),
            notes=req.get("notes"),
            context=dict(req.get("context", {})),
        )

    def act(self, request: ExecRequest | Dict[str, Any]) -> ExecReport:
        req = self.plan(request)
        sym = req.target.symbol
        side = req.target.side
        qty_total = float(req.target.qty)
        now_ms = int(time.time() * 1000)

        # Fetch reference candles for planning (1m bars)
        bars_needed = max(5, int(req.schedule.duration_min / max(1, req.schedule.slice_minutes)) * 2)
        ref_sym = req.ref_symbol_for_vwap or sym
        refs = get_candles(ref_sym, interval="1m", lookback=bars_needed)
        last_px = _last_close(refs) or 0.0

        # Build slice plan
        if req.schedule.algo == "POV":
            slices = self._plan_pov(qty_total, refs, req)
        elif req.schedule.algo == "VWAP":
            slices = self._plan_vwap(qty_total, refs, req)
        else:
            slices = self._plan_twap(qty_total, refs, req)

        # Risk gate (pre-flight): check the largest slice at reference price
        max_slice_qty = max((sp.qty_slice for sp in slices), default=0.0)
        ok, msg = check_gates(sym, side, max_slice_qty, last_px, context=req.context)
        messages = [msg]
        if not ok:
            return ExecReport(
                ok=False, algo=req.schedule.algo, symbol=sym, side=side,
                requested_qty=qty_total, filled_qty=0.0, est_cost_bps=0.0,
                slices=slices, messages=["Blocked by risk policy: " + msg]
            )

        # Estimate cost
        est_bps = estimate_cost_bps(sym, side, qty_total, last_px, req.schedule.venue)

        # Execute plan (synchronously for now)
        remaining = qty_total
        filled = 0.0
        for sp in slices:
            if remaining <= 0:
                break
            q = min(sp.qty_slice, remaining)
            if q <= 0:
                continue

            if req.dry_run:
                messages.append(f"DRY-RUN {side} {q} {sym} @~{sp.px_ref:.4f} ({sp.reason})")
            else:
                order_type = req.target.order_type
                px = None
                if order_type == "limit":
                    px = self._compute_limit_price(side, sp.px_ref, req.schedule.allow_cross_spread_bps,
                                                   sliding=req.schedule.sliding_limit, live_last=last_px)
                oid = submit_order(sym, side, q, order_type=order_type, limit_price=px, tag=req.target.tag or req.schedule.algo)
                messages.append(f"sent {oid}: {side} {q} {sym} {order_type} {'' if px is None else f'@{px:.4f}'} ({sp.reason})")
            filled += q
            remaining -= q

        return ExecReport(
            ok=True, algo=req.schedule.algo, symbol=sym, side=side,
            requested_qty=qty_total, filled_qty=filled, est_cost_bps=est_bps,
            slices=slices, messages=messages
        )

    # -------------- planners --------------

    def _plan_twap(self, qty: float, refs: List[Dict[str, Any]], req: ExecRequest) -> List[SlicePlan]:
        k = max(1, int(req.schedule.duration_min / max(1, req.schedule.slice_minutes)))
        per = qty / k
        last_px = _last_close(refs) or 0.0
        now = int(time.time() * 1000)
        out: List[SlicePlan] = []
        for i in range(k):
            t_ms = now + (i * req.schedule.slice_minutes * 60_000)
            px = refs[-1]["c"] if refs else last_px
            out.append(SlicePlan(t_index=i, when_ms=t_ms, px_ref=float(px), qty_slice=max(0.0, per), reason="TWAP"))
        return out

    def _plan_pov(self, qty: float, refs: List[Dict[str, Any]], req: ExecRequest) -> List[SlicePlan]:
        # Use recent volumes to infer per-bar ADV and cap participation
        vols = [float(b["v"]) for b in refs[-60:]] or [1.0]
        adv_bar = sum(vols) / len(vols)
        k = max(1, int(req.schedule.duration_min / max(1, req.schedule.slice_minutes)))
        # naive forecast: assume each future bar has ~adv_bar volume
        cap_per_bar = adv_bar * req.schedule.max_participation
        per = min(qty / k, cap_per_bar)
        px_ref = _last_close(refs) or 0.0
        now = int(time.time() * 1000)
        out: List[SlicePlan] = []
        remaining = qty
        for i in range(k):
            q_i = min(per, remaining)
            out.append(SlicePlan(t_index=i, when_ms=now + i * req.schedule.slice_minutes * 60_000,
                                 px_ref=px_ref, qty_slice=max(0.0, q_i), reason=f"POV<= {req.schedule.max_participation:.0%}"))
            remaining -= q_i
        if remaining > 1e-8:
            # tail slice if qty > sum caps
            out.append(SlicePlan(t_index=k, when_ms=now + k * req.schedule.slice_minutes * 60_000,
                                 px_ref=px_ref, qty_slice=remaining, reason="POV tail"))
        return out

    def _plan_vwap(self, qty: float, refs: List[Dict[str, Any]], req: ExecRequest) -> List[SlicePlan]:
        # Use historical intraday volume profile from the last N bars
        vols = [float(b["v"]) for b in refs[-max(10, min(120, len(refs))):]] or [1.0]
        profile = [v / max(1e-9, _sum(vols)) for v in vols]
        # Re-sample profile to K slices
        k = max(1, int(req.schedule.duration_min / max(1, req.schedule.slice_minutes)))
        if len(profile) != k:
            # linear resample
            prof_res = []
            for i in range(k):
                pos = i * (len(profile) - 1) / max(1, k - 1)
                lo, hi = int(math.floor(pos)), int(math.ceil(pos))
                w = pos - lo
                prof_res.append((1 - w) * profile[lo] + w * profile[hi])
            profile = prof_res
            # re-normalize
            s = _sum(profile) or 1.0
            profile = [p / s for p in profile]
        target_qty = [qty * p for p in profile]
        last_px = _last_close(refs) or 0.0
        now = int(time.time() * 1000)
        out: List[SlicePlan] = []
        remaining = qty
        for i, q in enumerate(target_qty):
            q_i = min(q, remaining)
            out.append(SlicePlan(t_index=i, when_ms=now + i * req.schedule.slice_minutes * 60_000,
                                 px_ref=last_px, qty_slice=max(0.0, q_i), reason="VWAP profile"))
            remaining -= q_i
        if remaining > 1e-8:
            out.append(SlicePlan(t_index=len(target_qty), when_ms=now + k * req.schedule.slice_minutes * 60_000,
                                 px_ref=last_px, qty_slice=remaining, reason="VWAP tail"))
        return out

    # -------------- execution helpers --------------

    def _compute_limit_price(self, side: Side, px_ref: float, cross_bps: float,
                             *, sliding: bool, live_last: Optional[float]) -> float:
        base = live_last if (sliding and live_last) else px_ref
        bump = (cross_bps / 1e4) * base
        if side == "buy":
            return round(base + bump, 4)
        return round(base - bump, 4)

    # -------------- docs & health --------------

    def explain(self) -> str:
        return (
            "ExecutionAgent supports TWAP/POV/VWAP. It fetches recent 1m candles, "
            "builds a slice plan with participation controls, runs risk gates, "
            "estimates simple cost (bps), and submits market/limit orders via broker_interface."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "ts": int(time.time())}


# ------------------------------------------------------------
# quick smoke
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = ExecutionAgent()
    req = ExecRequest(
        target=ExecutionTarget(symbol="AAPL", side="buy", qty=50_000, order_type="limit"),
        schedule=Schedule(algo="POV", duration_min=15, slice_minutes=1, max_participation=0.1, sliding_limit=True),
        dry_run=True,
        context={"book": "demo"}
    )
    rep = agent.act(req)
    print("OK:", rep.ok, "filled:", rep.filled_qty, "cost(bps):", rep.est_cost_bps)
    for s in rep.slices[:5]:
        print(s)