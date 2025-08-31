# backend/compliance/regulator_replay.py
from __future__ import annotations
"""
Regulator Replay
----------------
Deterministically rebuild per-order timelines for audit/regulatory review.

Inputs (dicts or JSON files):
- orders:      list[{ id, ts_ms, parent_id?, symbol, side, qty, order_type, limit_price?, venue? }]
- router:      list[{ ts_ms, order_id, action, venue, reason?, scores? }]   # e.g., "route", "reroute", "cancel"
- risk:        list[{ ts_ms, order_id, check, ok, detail? }]
- quotes:      list[{ ts_ms, venue, symbol, bid, ask, bid_sz?, ask_sz? }]
- fills:       list[{ ts_ms, order_id, venue, price, qty, liquidity? }]     # liquidity: "maker"|"taker"
- policies:    { limits: {...}, best_ex_rules: [...], latency_budget_ms?: int }
- metadata:    { session_id?, book?, region? } (optional)

Outputs:
- replay_json: {
    session, orders: { <order_id>: {
      summary:{...}, checks:{...}, tca:{...}, timeline:[...], nbbo_samples:[...], evidence:{...}
    }}, global:{ invariants:[...], violations:[...] }
  }
- replay_md: Markdown report per session (compact)

CLI:
  python -m backend.compliance.regulator_replay \
    --in inputs.json --out_json replay.json --out_md replay.md \
    --symbol AAPL --order O123

Environment:
  REDIS_HOST/PORT (unused unless you wire a bus), REG_REPLAY_STREAM (optional)

Notes:
- Works without pandas/numpy; uses them when available for speed & stats.
- NBBO is computed cross-venue from `quotes`. If your feed is venue-scoped only,
  provide enough venues to approximate NBBO.
"""

import json, os, math, statistics, argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from bisect import bisect_left

# ---- optional bus (no-op if absent) ----
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:  # type: ignore
        pass

OUT_STREAM = os.getenv("REG_REPLAY_STREAM", "compliance.regulator_replay")

# ---- optional speed-ups ----
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

# ----------------- helpers -----------------

def _to_ms(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        # assume ms if big, else seconds
        return int(v if v > 10_000_000_000 else float(v) * 1000.0)
    s = str(v)
    # ISO-ish: 'YYYY-MM-DDTHH:MM:SS(.sss)Z'
    try:
        from datetime import datetime
        s2 = s.replace("Z","").replace("z","")
        if "." in s2:
            dt = datetime.fromisoformat(s2)
        else:
            dt = datetime.fromisoformat(s2)
        return int(dt.timestamp() * 1000.0)
    except Exception:
        # last resort
        try:
            return int(float(s) * 1000.0)
        except Exception:
            return 0

def _bps(px1: float, px2: float) -> float:
    if not px2:
        return 0.0
    return ( (px1 - px2) / px2 ) * 1e4

def _mid(b: float, a: float) -> float:
    if b and a and a > 0.0:
        return 0.5 * (b + a)
    return 0.0

def _safe_div(a: float, b: float) -> float:
    return a / b if b not in (0, 0.0, None) else 0.0

# ----------------- dataclasses -----------------

@dataclass
class NBBO:
    ts_ms: int
    bid: float
    ask: float
    mid: float
    venues: Dict[str, Dict[str, float]]  # venue -> {bid, ask}

@dataclass
class Event:
    ts_ms: int
    kind: str        # market|order|risk|route|fill|cancel
    data: Dict[str, Any]

# ----------------- core engine -----------------

class RegulatorReplay:
    def __init__(self, *, latency_budget_ms: int = 250, quote_staleness_ms: int = 500):
        self.latency_budget_ms = latency_budget_ms
        self.quote_staleness_ms = quote_staleness_ms

    # ---------- main entry ----------
    def build(
        self,
        *,
        orders: List[Dict[str, Any]],
        router: List[Dict[str, Any]],
        risk: List[Dict[str, Any]],
        quotes: List[Dict[str, Any]],
        fills: List[Dict[str, Any]],
        policies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        only_order_id: Optional[str] = None,
        only_symbol: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Returns (replay_json, markdown_report)
        """
        # Normalize & filter
        O = [self._norm_order(o) for o in orders]
        Rr = [self._norm_router(r) for r in router]
        Rk = [self._norm_risk(rk) for rk in risk]
        Q = [self._norm_quote(q) for q in quotes]
        F = [self._norm_fill(f) for f in fills]

        if only_order_id:
            O = [o for o in O if o["id"] == only_order_id]
            Rr = [r for r in Rr if r["order_id"] == only_order_id]
            Rk = [x for x in Rk if x["order_id"] == only_order_id]
            F  = [f for f in F  if f["order_id"] == only_order_id]
            # quotes we keep all (NBBO is market-wide)

        if only_symbol:
            O = [o for o in O if o["symbol"] == only_symbol]
            ids = {o["id"] for o in O}
            Rr = [r for r in Rr if r["order_id"] in ids]
            Rk = [x for x in Rk if x["order_id"] in ids]
            F  = [f for f in F  if f["order_id"] in ids]

        # Build NBBO index per symbol for fast lookup
        nbbo_index = self._build_nbbo_index(Q)

        # Per-order timelines & checks
        out_orders: Dict[str, Any] = {}
        inv_global: List[str] = []
        viol_global: List[str] = []

        limits = (policies or {}).get("limits", {})
        latency_budget = int((policies or {}).get("latency_budget_ms", self.latency_budget_ms))

        for o in O:
            oid = o["id"]
            sym = o["symbol"]
            tl: List[Event] = []

            # Order created
            tl.append(Event(o["ts_ms"], "order", {"state": "created", **o}))

            # Risk checks for this order
            rks = [x for x in Rk if x["order_id"] == oid]
            for x in rks:
                tl.append(Event(x["ts_ms"], "risk", x))

            # Routing decisions
            rt = [x for x in Rr if x["order_id"] == oid]
            for x in rt:
                tl.append(Event(x["ts_ms"], ("cancel" if x["action"]=="cancel" else "route"), x))

            # Fills
            fs = [f for f in F if f["order_id"] == oid]
            for f in fs:
                tl.append(Event(f["ts_ms"], "fill", f))

            # Sort timeline
            tl.sort(key=lambda e: (e.ts_ms, e.kind))

            # NBBO at arrival and alongside key events
            arrival_nbbo = self._nbbo_at(nbbo_index, sym, o["ts_ms"])
            nbbo_samples = []
            if arrival_nbbo:
                nbbo_samples.append(asdict(arrival_nbbo))
            for e in tl:
                # sample NBBO near each event if symbol matches and quote fresh
                n = self._nbbo_at(nbbo_index, sym, e.ts_ms)
                if n:
                    nbbo_samples.append(asdict(n))

            # TCA & policy checks
            tca, checks, inv, viol = self._evaluate(o, tl, arrival_nbbo, limits, latency_budget)
            inv_global.extend(inv)
            viol_global.extend(viol)

            out_orders[oid] = {
                "summary": {
                    "order_id": oid,
                    "symbol": sym,
                    "side": o["side"],
                    "order_type": o["order_type"],
                    "limit_price": o.get("limit_price"),
                    "submitted_ts_ms": o["ts_ms"],
                    "filled_qty": tca.get("filled_qty"),
                    "avg_fill_px": tca.get("avg_fill_px"),
                    "gross_notional": tca.get("notional"),
                    "arrival_mid_px": tca.get("arrival_mid_px"),
                    "slippage_bps_vs_mid": tca.get("slippage_bps_vs_mid"),
                    "venues_touched": sorted({f["venue"] for f in [e.data for e in tl if e.kind=="fill"]}) if tl else []
                },
                "checks": checks,
                "tca": tca,
                "timeline": [asdict(e) for e in tl],
                "nbbo_samples": nbbo_samples,
                "evidence": {
                    "router_count": len(rt),
                    "risk_count": len(rks),
                    "fills_count": len(fs)
                }
            }

        replay_json = {
            "session": (metadata or {}).get("session_id") or "session-unknown",
            "meta": metadata or {},
            "orders": out_orders,
            "global": {
                "invariants": inv_global,
                "violations": viol_global
            }
        }

        # Markdown summary
        md = self._render_markdown(replay_json)

        # Optionally emit a small event
        try:
            publish_stream(OUT_STREAM, {"kind": "replay_ready", "orders": len(out_orders)})
        except Exception:
            pass

        return replay_json, md

    # ---------- normalization ----------
    def _norm_order(self, o: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(o.get("id") or o.get("order_id")),
            "ts_ms": _to_ms(o.get("ts_ms") or o.get("ts") or o.get("timestamp")),
            "symbol": str(o.get("symbol") or ""),
            "side": str(o.get("side") or "").lower(),
            "qty": float(o.get("qty") or o.get("quantity") or 0.0),
            "order_type": str(o.get("order_type") or "market").lower(),
            "limit_price": (float(o.get("limit_price")) if o.get("limit_price") not in (None,"","NaN") else None), # type: ignore
            "venue": (str(o.get("venue")) if o.get("venue") else None),
            "parent_id": (str(o.get("parent_id")) if o.get("parent_id") else None),
        }

    def _norm_router(self, r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ts_ms": _to_ms(r.get("ts_ms") or r.get("ts") or r.get("timestamp")),
            "order_id": str(r.get("order_id")),
            "action": str(r.get("action") or "route"),
            "venue": str(r.get("venue") or ""),
            "reason": r.get("reason"),
            "scores": r.get("scores") or {}
        }

    def _norm_risk(self, rk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ts_ms": _to_ms(rk.get("ts_ms") or rk.get("ts") or rk.get("timestamp")),
            "order_id": str(rk.get("order_id")),
            "check": str(rk.get("check") or ""),
            "ok": bool(rk.get("ok", True)),
            "detail": rk.get("detail")
        }

    def _norm_quote(self, q: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ts_ms": _to_ms(q.get("ts_ms") or q.get("ts") or q.get("timestamp")),
            "venue": str(q.get("venue") or ""),
            "symbol": str(q.get("symbol") or ""),
            "bid": float(q.get("bid") or 0.0),
            "ask": float(q.get("ask") or 0.0),
            "bid_sz": float(q.get("bid_sz") or 0.0),
            "ask_sz": float(q.get("ask_sz") or 0.0),
        }

    def _norm_fill(self, f: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ts_ms": _to_ms(f.get("ts_ms") or f.get("ts") or f.get("timestamp")),
            "order_id": str(f.get("order_id")),
            "venue": str(f.get("venue") or ""),
            "price": float(f.get("price") or 0.0),
            "qty": float(f.get("qty") or f.get("quantity") or 0.0),
            "liquidity": (str(f.get("liquidity") or ""))  # "maker"|"taker"|"" (optional)
        }

    # ---------- NBBO index ----------
    def _build_nbbo_index(self, quotes: List[Dict[str, Any]]):
        by_sym: Dict[str, Dict[str, List[Tuple[int, float, float]]]] = defaultdict(lambda: defaultdict(list))
        for q in quotes:
            if q["bid"] <= 0.0 and q["ask"] <= 0.0:
                continue
            by_sym[q["symbol"]][q["venue"]].append((q["ts_ms"], q["bid"], q["ask"]))
        # sort each venue stream
        for sym, venues in by_sym.items():
            for v, arr in venues.items():
                arr.sort(key=lambda x: x[0])
        return by_sym

    def _nbbo_at(self, nbbo_index, symbol: str, ts_ms: int) -> Optional[NBBO]:
        venues = nbbo_index.get(symbol)
        if not venues:
            return None
        best_bid = 0.0; best_ask = 0.0
        snap: Dict[str, Dict[str, float]] = {}
        for v, arr in venues.items():
            # binary search last quote <= ts_ms within staleness budget
            i = bisect_left(arr, (ts_ms+1, float("inf"), float("inf"))) - 1
            if i >= 0:
                t, b, a = arr[i]
                if ts_ms - t <= self.quote_staleness_ms:
                    snap[v] = {"bid": b, "ask": a}
                    if b > 0.0 and (b > best_bid): best_bid = b
                    if a > 0.0 and (best_ask == 0.0 or a < best_ask): best_ask = a
        if not snap or best_bid <= 0.0 or best_ask <= 0.0:
            return None
        return NBBO(ts_ms=ts_ms, bid=best_bid, ask=best_ask, mid=_mid(best_bid, best_ask), venues=snap)

    # ---------- evaluation ----------
    def _evaluate(
        self,
        order: Dict[str, Any],
        tl: List[Event],
        arrival_nbbo: Optional[NBBO],
        limits: Dict[str, Any],
        latency_budget_ms: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
        inv: List[str] = []
        viol: List[str] = []

        # filled stats
        fills = [e.data for e in tl if e.kind == "fill"]
        filled_qty = sum(f["qty"] for f in fills) if fills else 0.0
        vwap = _safe_div(sum(f["price"] * f["qty"] for f in fills), filled_qty) if filled_qty else 0.0
        notional = sum(f["price"] * f["qty"] for f in fills) if fills else 0.0

        arrival_mid = arrival_nbbo.mid if arrival_nbbo else 0.0
        slip_bps = _bps(vwap, arrival_mid) if arrival_mid else 0.0
        # side-aware slippage (positive = worse)
        if order["side"] == "buy" and vwap and arrival_mid:
            slip_bps = _bps(vwap, arrival_mid)
        elif order["side"] == "sell" and vwap and arrival_mid:
            slip_bps = _bps(arrival_mid, vwap)

        # latencies: first route after order, first fill after first route
        t_order = order["ts_ms"]
        route_events = [e for e in tl if e.kind in ("route","cancel")]
        fill_events  = [e for e in tl if e.kind == "fill"]
        route_latency = (route_events[0].ts_ms - t_order) if route_events else None
        time_to_fill  = ((fill_events[0].ts_ms - route_events[0].ts_ms) if (route_events and fill_events) else None)

        # invariants
        if route_latency is not None:
            inv.append(f"route_latency_ms={route_latency}")
            if latency_budget_ms and route_latency > latency_budget_ms:
                viol.append(f"route_latency_exceeds_budget({route_latency}ms > {latency_budget_ms}ms)")
        if time_to_fill is not None:
            inv.append(f"time_to_first_fill_ms={time_to_fill}")

        # best-ex rough checks:
        bestex: Dict[str, Any] = {}
        if arrival_nbbo and fills:
            # For market orders: price should not violate NBBO side at arrival (tolerances vary by market; here we note)
            if order["order_type"] == "market":
                for f in fills:
                    if order["side"] == "buy" and arrival_nbbo.ask and f["price"] > arrival_nbbo.ask * 1.001:
                        viol.append(f"market_buy_fill_above_arrival_ask({f['price']} > {arrival_nbbo.ask})")
                    if order["side"] == "sell" and arrival_nbbo.bid and f["price"] < arrival_nbbo.bid * 0.999:
                        viol.append(f"market_sell_fill_below_arrival_bid({f['price']} < {arrival_nbbo.bid})")
            # For limit orders: limit relative to NBBO at arrival
            if order["order_type"] == "limit" and order.get("limit_price"):
                L = order["limit_price"]
                if order["side"] == "buy" and L > arrival_nbbo.ask * 1.02:
                    viol.append(f"buy_limit_far_above_nbbo({L} vs ask {arrival_nbbo.ask})")
                if order["side"] == "sell" and L < arrival_nbbo.bid * 0.98:
                    viol.append(f"sell_limit_far_below_nbbo({L} vs bid {arrival_nbbo.bid})")
        bestex["arrival_nbbo"] = asdict(arrival_nbbo) if arrival_nbbo else None

        # limits checks (simple examples)
        max_notional = limits.get("max_notional_per_trade")
        if max_notional and notional > float(max_notional):
            viol.append(f"notional_exceeds_limit({notional} > {max_notional})")

        max_qty = limits.get("max_qty_per_order")
        if max_qty and order["qty"] > float(max_qty):
            viol.append(f"qty_exceeds_limit({order['qty']} > {max_qty})")

        # risk pre-trade check present?
        risk_pre = any(e.kind == "risk" and str(e.data.get("check")).lower().startswith("pre_trade")
                       for e in tl)
        if not risk_pre:
            viol.append("missing_pre_trade_risk_check")

        checks = {
            "latency": {
                "route_latency_ms": route_latency,
                "time_to_first_fill_ms": time_to_fill,
                "budget_ms": latency_budget_ms
            },
            "limits": {"max_notional": max_notional, "max_qty": max_qty},
            "best_ex": bestex,
            "invariants": inv,
            "violations": viol
        }

        tca = {
            "filled_qty": filled_qty,
            "avg_fill_px": vwap if filled_qty else None,
            "notional": notional if filled_qty else 0.0,
            "arrival_mid_px": arrival_mid if arrival_mid else None,
            "slippage_bps_vs_mid": slip_bps if filled_qty and arrival_mid else None
        }
        return tca, checks, inv, viol

    # ---------- render ----------
    def _render_markdown(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        sess = payload.get("session","unknown")
        lines += [f"# Regulator Replay — Session `{sess}`", ""]
        g = payload.get("global", {})
        lines += [f"- Orders: **{len(payload.get('orders',{}))}**",
                  f"- Invariants noted: {len(g.get('invariants', []))}",
                  f"- Potential violations: **{len(g.get('violations', []))}**",
                  ""]
        for oid, row in payload.get("orders", {}).items():
            s = row["summary"]; c = row["checks"]; tca = row["tca"]
            lines += [
                f"## Order `{oid}` — {s['symbol']} ({s['side']})",
                f"- Submitted: `{s['submitted_ts_ms']}`  Type: **{s['order_type']}**  Limit: {s.get('limit_price')}",
                f"- Filled Qty: **{tca.get('filled_qty')}**  VWAP: **{tca.get('avg_fill_px')}**  Notional: **{tca.get('notional')}**",
                f"- Arrival Mid: {tca.get('arrival_mid_px')}  Slippage (bps vs mid): {tca.get('slippage_bps_vs_mid')}",
                f"- Venues touched: {s.get('venues_touched')}",
                "### Checks",
                f"- Latency: route={c['latency']['route_latency_ms']} ms, t_first_fill={c['latency']['time_to_first_fill_ms']} ms (budget {c['latency']['budget_ms']} ms)",
                f"- Violations: {c.get('violations')}",
                ""
            ]
        return "\n".join(lines)

# ----------------- CLI -----------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _save_md(path: str, txt: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def main():
    ap = argparse.ArgumentParser(description="Regulator Replay — rebuild per-order timelines & checks")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSON with keys: orders, router, risk, quotes, fills, policies, metadata")
    ap.add_argument("--out_json", required=True, help="Output JSON timeline/report")
    ap.add_argument("--out_md", required=True, help="Output Markdown summary")
    ap.add_argument("--order", default=None, help="Filter: specific order_id")
    ap.add_argument("--symbol", default=None, help="Filter: specific symbol")
    args = ap.parse_args()

    data = _load_json(args.inp)
    eng = RegulatorReplay()
    replay_json, md = eng.build(
        orders=data.get("orders", []),
        router=data.get("router", []),
        risk=data.get("risk", []),
        quotes=data.get("quotes", []),
        fills=data.get("fills", []),
        policies=data.get("policies", {}),
        metadata=data.get("metadata", {}),
        only_order_id=args.order,
        only_symbol=args.symbol
    )
    _save_json(args.out_json, replay_json)
    _save_md(args.out_md, md)

if __name__ == "__main__":  # pragma: no cover
    main()