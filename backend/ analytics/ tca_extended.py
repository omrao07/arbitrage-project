# backend/analytics/tca_extended.py
"""
TCA Extended (TCA++)
--------------------
High-resolution execution analytics for each parent order.

Listens (best-effort; all optional):
  - oms.parent (parent orders)        {order_id, symbol, side, qty, ts_ms, mark_px, urgency, asset_class}
  - oms.child  (child orders)         {order_id, parent_id, venue, px, qty, ts_ms, typ}
  - oms.fill   (fills)                {order_id|child_id, parent_id, symbol, side, price, qty, ts_ms, venue}
  - exec.router.decisions             {parent_id, symbol, route, params, ts}
  - marks                                {symbol, price, ts}

Emits:
  - tca.extended                      {
      parent_id, symbol, side, route, venues[], horizon_ms,
      vwap, arrival_px, mark_at_send, spread_bps, impact_bps, timing_bps, latency_bps,
      participation, child_count, notional, start_ts, end_ts,
      per_venue: {VENUE: {fills, notional, vwap}},
      tca: {spread_bps, impact_bps, latency_bps},
      context: {...}   # snapshot features for RL (spread_bps, vol_bps, adv_ratio, urgency, imbalance, latency_tier, ...)
    }

Also writes a compact row to Redis HSET("tca:last", parent_id, ...), if bus is present.

Design notes:
- Works with just stdlib. If numpy/pandas exist, some ops are faster.
- Handles partial/cancelled parents gracefully (computes on inactivity timeout).
- Latency cost estimated from decision→first child→first fill vs mark changes.
- Impact decomposition (rough) into temporary/permanent via +/- 60s marks if available.

Usage:
    python -m backend.analytics.tca_extended --run
"""

from __future__ import annotations

import os
import time
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# Bus helpers (graceful if absent)
try:
    from backend.bus.streams import consume_stream, publish_stream, hset
except Exception:
    consume_stream = publish_stream = hset = None  # type: ignore


# ----------------------------- utils -----------------------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _bps(a: float, b: float) -> float:
    if not a or a <= 0 or b is None:
        return 0.0
    return (b - a) / a * 1e4

def _safe_mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return 0.0
    if np is not None:
        return float(np.mean(np.asarray(xs)))
    return sum(xs) / len(xs)

def _vwap(prices: List[float], qtys: List[float]) -> float:
    if not prices or not qtys:
        return 0.0
    num = sum(p*q for p, q in zip(prices, qtys))
    den = sum(qtys)
    return num / den if den > 0 else 0.0

def _sign(side: str) -> int:
    return -1 if (side or "").lower().startswith("sell") else 1


# ----------------------------- state -----------------------------------------

@dataclass
class ParentOrder:
    parent_id: str
    symbol: str
    side: str
    qty: float
    start_ts: int
    arrival_px: float = 0.0        # mid/mark when parent created
    mark_at_send: float = 0.0      # best-effort mark at router decision
    route: Optional[str] = None
    route_params: Dict[str, Any] = field(default_factory=dict)
    fills_p: List[float] = field(default_factory=list)
    fills_q: List[float] = field(default_factory=list)
    fills_ts: List[int] = field(default_factory=list)
    venues: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {"fills":0,"notional":0.0,"vwap":0.0}))
    child_count: int = 0
    first_child_ts: Optional[int] = None
    first_fill_ts: Optional[int] = None
    last_activity_ts: int = 0
    notional: float = 0.0
    # marks around exec for impact/timing
    marks_around: Dict[str, float] = field(default_factory=dict)  # {"t0":px, "t1":px, "t+60s":px, ...}
    # context features for RL / analytics
    ctx: Dict[str, Any] = field(default_factory=dict)

    def ingest_fill(self, px: float, qty: float, ts: int, venue: Optional[str]):
        self.fills_p.append(float(px))
        self.fills_q.append(float(qty))
        self.fills_ts.append(int(ts))
        self.notional += float(px) * float(qty)
        if venue:
            v = self.venues[venue]
            v["fills"] += 1
            v["notional"] += float(px) * float(qty)
            # vwap per venue recomputed later
        if self.first_fill_ts is None:
            self.first_fill_ts = ts
        self.last_activity_ts = ts

    def ingest_child(self, ts: int):
        self.child_count += 1
        if self.first_child_ts is None:
            self.first_child_ts = ts
        self.last_activity_ts = ts

    def done(self) -> bool:
        return sum(self.fills_q) >= 0.999 * float(self.qty) if self.qty > 0 else False

    def inactivity(self, now_ms: int, idle_ms: int) -> bool:
        return (now_ms - (self.last_activity_ts or self.start_ts)) >= idle_ms


# ----------------------------- TCA engine ------------------------------------

class TCAExtended:
    def __init__(
        self,
        idle_close_ms: int = 10_000,    # close parent if idle for 10s
        holdout_ms: int = 60_000,       # window to measure temporary/permanent impact
        publish_stream_name: str = "tca.extended",
    ):
        self.parents: Dict[str, ParentOrder] = {}
        self.by_symbol_marks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))  # (ts, px)
        self.holdout_ms = int(holdout_ms)
        self.idle_close_ms = int(idle_close_ms)
        self.out = publish_stream_name

    # ----- ingestion -----

    def on_parent(self, m: Dict[str, Any]) -> None:
        pid = str(m.get("order_id") or m.get("parent_id") or "")
        if not pid:
            return
        sym = (m.get("symbol") or "").upper()
        po = ParentOrder(
            parent_id=pid, symbol=sym, side=m.get("side","buy"), qty=float(m.get("qty",0.0)),
            start_ts=int(m.get("ts_ms") or _utc_ms()), arrival_px=float(m.get("mark_px") or 0.0),
        )
        # record initial context
        po.ctx = {
            "urgency": float(m.get("urgency", 0.5)),
            "asset_class": m.get("asset_class","equity"),
            "adv_ratio": float(m.get("adv_ratio", 0.05)),
            "latency_tier": int(m.get("latency_tier", 1)),
        }
        po.last_activity_ts = po.start_ts
        self.parents[pid] = po

    def on_child(self, m: Dict[str, Any]) -> None:
        pid = str(m.get("parent_id") or "")
        if pid and pid in self.parents:
            self.parents[pid].ingest_child(int(m.get("ts_ms") or _utc_ms()))

    def on_fill(self, m: Dict[str, Any]) -> None:
        pid = str(m.get("parent_id") or "")
        if not pid or pid not in self.parents:
            return
        p = self.parents[pid]
        p.ingest_fill(float(m.get("price",0.0)), float(m.get("qty",0.0)),
                      int(m.get("ts_ms") or _utc_ms()), m.get("venue"))
        # capture mark around fills for later impact/timing
        px = self._last_mark(p.symbol)
        if px:
            if "t0" not in p.marks_around:
                p.marks_around["t0"] = px  # first available mark after start
            p.marks_around["t_last"] = px

    def on_router(self, m: Dict[str, Any]) -> None:
        pid = str(m.get("parent_id") or m.get("order_id") or "")
        if pid and pid in self.parents:
            p = self.parents[pid]
            p.route = m.get("route")
            p.route_params = m.get("params") or {}
            p.mark_at_send = float(m.get("mark_px") or m.get("mark_price") or p.arrival_px or 0.0)
            # add more ctx if provided
            for k in ("spread_bps","vol_bps","imbalance","latency_tier","venue"):
                if k in m:
                    p.ctx[k] = m[k]

    def on_mark(self, m: Dict[str, Any]) -> None:
        sym = (m.get("symbol") or m.get("s") or "").upper()
        px  = float(m.get("price") or m.get("p") or 0.0)
        ts  = int(m.get("ts") or m.get("ts_ms") or _utc_ms())
        if not sym or px <= 0:
            return
        self.by_symbol_marks[sym].append((ts, px))

    # ----- processing -----

    def tick(self) -> None:
        """
        Periodic maintenance: close finished/idle parents and emit TCA.
        Call this in your loop (e.g., every second).
        """
        now = _utc_ms()
        to_close: List[str] = []
        for pid, p in list(self.parents.items()):
            if p.done() or p.inactivity(now, self.idle_close_ms):
                self._finalize_and_emit(p, now)
                to_close.append(pid)
        for pid in to_close:
            self.parents.pop(pid, None)

    # ----- metrics -----

    def _finalize_and_emit(self, p: ParentOrder, now_ms: int) -> None:
        if not p.fills_q:
            return  # nothing to analyze

        # compute venue VWAPs
        for v, acc in p.venues.items():
            # recompute vwap from individual fills for this venue
            pr, qt = [], []
            for j, venue in enumerate([m for m in p.venues.keys()]):  # placeholder (if you track per-fill venue)
                pass  # keeping minimal; we aggregate above via notional/fills
            acc["vwap"] = (acc["notional"] / max(1e-12, acc["fills"])) if acc["fills"] else 0.0

        vwap = _vwap(p.fills_p, p.fills_q)
        arrival = p.arrival_px or p.mark_at_send or self._mark_at_time(p.symbol, p.start_ts) or vwap
        side_sgn = _sign(p.side)

        # Spread cost: signed slippage vs arrival/mark
        spread_bps = side_sgn * _bps(arrival, vwap)

        # Latency cost: mark drift between router decision and first child/fill
        lat_ref = p.mark_at_send or arrival
        m_first = self._mark_at_time(p.symbol, p.first_fill_ts) if p.first_fill_ts else None
        latency_bps = side_sgn * _bps(lat_ref, m_first if m_first else lat_ref)

        # Impact: difference between pre-trade mark and mark shortly after completion
        end_ts = p.fills_ts[-1] if p.fills_ts else now_ms
        m_pre  = self._mark_at_time(p.symbol, p.first_fill_ts or p.start_ts)
        m_post = self._mark_at_time(p.symbol, end_ts + self.holdout_ms)
        impact_bps = side_sgn * _bps(m_pre or arrival, m_post or vwap)

        # Timing: adverse drift between parent arrival and first child (if any)
        m_arr  = self._mark_at_time(p.symbol, p.start_ts) or arrival
        m_child = self._mark_at_time(p.symbol, p.first_child_ts) if p.first_child_ts else None
        timing_bps = side_sgn * _bps(m_arr, m_child if m_child else m_arr)

        participation = min(1.0, (sum(p.fills_q) / max(1e-9, p.qty))) if p.qty > 0 else 0.0
        horizon_ms = (end_ts - p.start_ts) if p.fills_ts else 0

        # Build context feature snapshot for RL/SOR learning
        ctx = dict(p.ctx)
        ctx.setdefault("symbol", p.symbol)
        ctx.setdefault("side", p.side)
        ctx.setdefault("qty", float(p.qty))
        ctx.setdefault("notional", float(vwap * sum(p.fills_q)))
        ctx.setdefault("spread_bps", float(ctx.get("spread_bps", abs(spread_bps))))
        ctx.setdefault("vol_bps", float(ctx.get("vol_bps", 15.0)))
        ctx.setdefault("imbalance", float(ctx.get("imbalance", 0.0)))

        out = {
            "parent_id": p.parent_id,
            "symbol": p.symbol,
            "side": p.side,
            "route": p.route,
            "params": p.route_params,
            "venues": list(p.venues.keys()),
            "start_ts": p.start_ts,
            "end_ts": end_ts,
            "horizon_ms": int(horizon_ms),
            "arrival_px": float(arrival),
            "mark_at_send": float(p.mark_at_send or arrival),
            "vwap": float(vwap),
            "spread_bps": float(spread_bps),
            "impact_bps": float(impact_bps),
            "timing_bps": float(timing_bps),
            "latency_bps": float(latency_bps),
            "participation": float(participation),
            "child_count": int(p.child_count),
            "notional": float(p.notional),
            "per_venue": {k: {kk: float(vv) for kk, vv in d.items()} for k, d in p.venues.items()},
            "tca": {
                "spread_bps": float(spread_bps),
                "impact_bps": float(impact_bps),
                "latency_bps": float(latency_bps),
            },
            "context": ctx,
        }

        # publish & store
        if publish_stream:
            publish_stream(self.out, out)
            # small “last result” hset for dashboards
            try:
                hset("tca:last", p.parent_id, {
                    "symbol": p.symbol, "route": p.route, "vwap": out["vwap"],
                    "spread_bps": out["spread_bps"], "impact_bps": out["impact_bps"], "latency_bps": out["latency_bps"]
                }) # type: ignore
            except Exception:
                pass

        # optional: train the RL SOR bandit
        try:
            from backend.ai import rl_execution_agent as RL  # type: ignore # local import to avoid hard dep
            RL.learn_from_tca({
                "symbol": p.symbol,
                "route": p.route,
                "tca": out["tca"],
                "context": out["context"],
            })
        except Exception:
            pass

    # ----- mark helpers -----

    def _last_mark(self, sym: str) -> Optional[float]:
        dq = self.by_symbol_marks.get(sym)
        if dq and len(dq) > 0:
            return float(dq[-1][1])
        return None

    def _mark_at_time(self, sym: str, ts: Optional[int]) -> Optional[float]:
        if not ts:
            return self._last_mark(sym)
        dq = self.by_symbol_marks.get(sym)
        if not dq:
            return None
        # simple nearest search (deque is sorted by time)
        best = None; best_dt = 10**15
        for t, p in dq:
            dt = abs(int(ts) - int(t))
            if dt < best_dt:
                best_dt = dt; best = float(p)
        return best


# ----------------------------- runner ----------------------------------------

def run_loop(idle_close_ms: int = 10_000, holdout_ms: int = 60_000):
    assert consume_stream and publish_stream, "bus streams not available"
    tca = TCAExtended(idle_close_ms=idle_close_ms, holdout_ms=holdout_ms)

    cursors = {
        "parent": "$", "child": "$", "fill": "$", "router": "$", "marks": "$"
    }
    streams = {
        "parent": "oms.parent",
        "child": "oms.child",
        "fill": "oms.fill",
        "router": "exec.router.decisions",
        "marks": "marks",
    }

    while True:
        for key, sname in streams.items():
            for _, msg in consume_stream(sname, start_id=cursors[key], block_ms=200, count=200):
                cursors[key] = "$"
                try:
                    if isinstance(msg, str):
                        msg = json.loads(msg)
                except Exception:
                    continue
                if key == "parent":
                    tca.on_parent(msg)
                elif key == "child":
                    tca.on_child(msg)
                elif key == "fill":
                    tca.on_fill(msg)
                elif key == "router":
                    tca.on_router(msg)
                elif key == "marks":
                    tca.on_mark(msg)
        # housekeeping
        tca.tick()
        time.sleep(0.05)


# ----------------------------- CLI -------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extended TCA engine")
    ap.add_argument("--run", action="store_true", help="Run event-loop (bus required)")
    ap.add_argument("--idle-close-ms", type=int, default=10_000)
    ap.add_argument("--holdout-ms", type=int, default=60_000)
    # quick probe: compute from one synthetic bundle
    ap.add_argument("--probe", action="store_true", help="Run a synthetic one-shot example to stdout")
    args = ap.parse_args()

    if args.probe:
        t = TCAExtended(idle_close_ms=args.idle_close_ms, holdout_ms=args.holdout_ms)
        # synthetic example
        pid = "P1"
        now = _utc_ms()
        t.on_parent({"order_id": pid, "symbol":"AAPL", "side":"buy", "qty":10000, "ts_ms":now, "mark_px":190.00, "urgency":0.7})
        t.on_router({"parent_id": pid, "symbol":"AAPL", "route":"POV", "params":{"participation":0.2}, "mark_px":190.02, "ts":now+50})
        # marks
        for dt, px in [(0,190.00),(200,190.01),(600,190.03),(1200,190.06),(2000,190.05),(70_000,189.98)]:
            t.on_mark({"symbol":"AAPL", "price":px, "ts":now+dt})
        # fills
        t.on_child({"parent_id":pid, "ts_ms":now+300})
        for dt, (px, q) in enumerate([(190.04,2000),(190.05,3000),(190.06,5000)], start=600):
            t.on_fill({"parent_id":pid, "symbol":"AAPL","side":"buy","price":px,"qty":q,"ts_ms":now+dt*10,"venue":"NASDAQ"})
        t._finalize_and_emit(t.parents[pid], now+80_000)
        return

    if args.run:
        try:
            run_loop(args.idle_close_ms, args.holdout_ms)
        except KeyboardInterrupt:
            pass
    else:
        print("Nothing to do. Use --run or --probe.")

if __name__ == "__main__":
    main()