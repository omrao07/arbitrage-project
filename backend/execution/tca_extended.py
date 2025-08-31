# backend/tca/tca_extended.py
from __future__ import annotations

import csv, json, math, os, statistics, time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------- optional deps (graceful) ----------
HAVE_NP = True
try:
    import numpy as np  # type: ignore
except Exception:
    HAVE_NP = False
    np = None  # type: ignore

HAVE_PD = True
try:
    import pandas as pd  # type: ignore
except Exception:
    HAVE_PD = False
    pd = None  # type: ignore

# ---------- helpers ----------
def now_ms() -> int: return int(time.time() * 1000)
def _sgn(side: str) -> int:
    s = (side or "").lower()
    if s in ("buy","b","long"): return +1
    if s in ("sell","s","short"): return -1
    raise ValueError(f"side must be buy/sell, got {side}")

def _safe_float(x, d=None):
    try: return float(x)
    except Exception: return d

def _bps(value: float, base_notional: float) -> float:
    if base_notional in (0, 0.0, None): return 0.0
    return float(value) / float(base_notional) * 1e4

def _round(x: float, n: int = 8) -> float:
    try: return float(round(float(x), n))
    except Exception: return 0.0

# ---------- input shapes ----------
@dataclass
class OrderRow:
    order_id: str
    symbol: str
    side: str
    qty: float
    decision_px: float
    decision_ts_ms: int
    arrival_px: Optional[float] = None
    arrival_ts_ms: Optional[int] = None
    final_px: Optional[float] = None
    # Optional execution window (for TWAP/VWAP style benchmarks)
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    currency: str = "USD"
    strategy: Optional[str] = None
    venue_hint: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FillRow:
    order_id: str
    ts_ms: int
    qty: float
    price: float
    fee: float = 0.0
    venue: Optional[str] = None
    child_id: Optional[str] = None

@dataclass
class BarRow:
    # Optional bars for VWAP/TWAP reference (1m bars recommended)
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

# ---------- results ----------
@dataclass
class PerVenueStats:
    venue: str
    notional: float
    fills: int
    avg_px: float
    fees: float

@dataclass
class Markout:
    # signed from portfolio POV (buy positive is cost)
    mo_30s_bps: float
    mo_1m_bps: float
    mo_5m_bps: float

@dataclass
class TCAOrderResult:
    order_id: str
    symbol: str
    side: str
    qty_target: float
    qty_exec: float
    fill_rate: float
    avg_exec_px: float
    # benchmarks & slippages (bps, + = cost)
    is_cost_bps: float                 # Implementation Shortfall vs decision
    arrival_slip_bps: float            # vs arrival price
    close_slip_bps: float              # vs final/close
    vwap_slip_bps: float               # vs session/order-window VWAP
    twap_slip_bps: float               # vs order-window TWAP (simple time avg)
    effective_spread_bps: float        # 2 * (signed exec - mid_at_arrival)/mid
    quoted_half_spread_bps: float      # half spread at arrival if provided
    realized_spread_bps: float         # signed (exec vs mid @ +5m), + = you *earned* spread
    # cost components in currency
    delay_cost: float
    trade_cost: float
    opportunity_cost: float
    fees: float
    total_cost: float
    # timing & participation
    time_to_first_fill_ms: Optional[int]
    time_to_full_fill_ms: Optional[int]
    participation_rate: Optional[float]     # (executed notional / est market notional during window)
    # per-venue
    venues: List[PerVenueStats]
    # markouts
    markouts: Markout

@dataclass
class TCAAggregate:
    n_orders: int
    gross_notional: float
    vw_cost_bps: float
    median_cost_bps: float
    avg_fill_rate: float
    avg_time_to_full_ms: Optional[float]
    components_bps: Dict[str, float]   # delay/trade/opp/fees
    venue_table: List[Dict[str, Any]]

# ---------- core computation ----------
def _vwap_from_fills(fills: List[FillRow]) -> Tuple[float, float]:
    n = sum(abs(f.qty) * f.price for f in fills)
    q = sum(abs(f.qty) for f in fills)
    if q <= 0: return 0.0, 0.0
    return n / q, q

def _vwap_from_bars(bars: List[BarRow]) -> Tuple[float, float]:
    vol = sum(max(0.0, b.volume) for b in bars)
    if vol <= 0:
        # fallback to close avg
        px = statistics.mean([b.close for b in bars]) if bars else 0.0
        return px, 0.0
    num = sum(b.close * max(0.0, b.volume) for b in bars)
    return num / vol, vol

def _twap_from_bars(bars: List[BarRow]) -> float:
    if not bars: return 0.0
    return statistics.mean([b.close for b in bars])

def _mid_from_bbo(arrival_px: Optional[float], best_bid: Optional[float]=None, best_ask: Optional[float]=None) -> Tuple[float, float]:
    # If explicit bid/ask given in meta, compute mid + quoted half spread
    if best_bid and best_ask and best_bid>0 and best_ask>0:
        mid = 0.5*(best_bid+best_ask)
        half_spread_bps = (0.5*(best_ask-best_bid)/mid)*1e4
        return mid, half_spread_bps
    # fallback: treat arrival_px as mid (unknown spread)
    return arrival_px or 0.0, 0.0

def _first_last_fill_ts(fills: List[FillRow]) -> Tuple[Optional[int], Optional[int]]:
    if not fills: return None, None
    ts = sorted([f.ts_ms for f in fills])
    return ts[0], ts[-1]

def _per_venue(fills: List[FillRow]) -> List[PerVenueStats]:
    venues = {}
    for f in fills:
        v = (f.venue or "UNKNOWN").upper()
        ven = venues.setdefault(v, {"notional":0.0,"qty":0.0,"fills":0,"fees":0.0})
        ven["notional"] += abs(f.qty)*f.price
        ven["qty"] += abs(f.qty)
        ven["fills"] += 1
        ven["fees"] += getattr(f, "fee", 0.0) or 0.0
    out: List[PerVenueStats] = []
    for v, s in venues.items():
        avg_px = (s["notional"]/s["qty"]) if s["qty"]>0 else 0.0
        out.append(PerVenueStats(venue=v, notional=s["notional"], fills=s["fills"], avg_px=avg_px, fees=s["fees"]))
    out.sort(key=lambda x: x.notional, reverse=True)
    return out

def _markout(side: str, exec_px: float, ref30: Optional[float], ref1m: Optional[float], ref5m: Optional[float]) -> Markout:
    d = _sgn(side)
    def one(ref):
        if not ref or ref<=0 or exec_px<=0: return 0.0
        # positive = cost for buy; realized spread (negative cost) shows as negative bps here
        return (d * (exec_px - ref) / ref) * 1e4
    return Markout(mo_30s_bps=_round(one(ref30),4), mo_1m_bps=_round(one(ref1m),4), mo_5m_bps=_round(one(ref5m),4))

# ---------- public API ----------
def analyze_order(
    order: OrderRow,
    fills: List[FillRow],
    *,
    window_bars: Optional[List[BarRow]] = None,  # bars across [start_ms,end_ms] (1m preferred)
    arrival_bid: Optional[float] = None,
    arrival_ask: Optional[float] = None,
    ref_30s_px: Optional[float] = None,
    ref_1m_px: Optional[float] = None,
    ref_5m_px: Optional[float] = None,
) -> TCAOrderResult:
    """
    Compute extended TCA for a single order.
    Provide bars if you want VWAP/TWAP over the order window.
    Provide arrival bid/ask to get quoted half-spread & effective spread.
    Provide markout refs (mid at +30s/+1m/+5m) for realized spread.
    """
    d = _sgn(order.side)
    Q = float(order.qty)
    Pd = float(order.decision_px)
    Pa = float(order.arrival_px or Pd)
    Pf = float(order.final_px or Pd)
    base_notional = abs(Q * Pd)

    # Executions
    exec_qty = sum(abs(f.qty) for f in fills)
    avg_exec_px = (sum(abs(f.qty)*f.price for f in fills)/exec_qty) if exec_qty>0 else 0.0
    fill_rate = (exec_qty / Q) if Q>0 else 0.0
    fees = sum(getattr(f, "fee", 0.0) or 0.0 for f in fills)

    # IS decomposition (decision→arrival→exec + opportunity)
    delay = d * exec_qty * (Pa - Pd)
    trade = d * exec_qty * (avg_exec_px - Pa)
    opp   = d * (Q - exec_qty) * (Pf - Pd)
    total = delay + trade + fees + opp

    # Benchmarks (bps)
    is_cost_bps      = _round(_bps(total, base_notional), 4)
    arrival_slip_bps = _round(_bps(d * exec_qty * (avg_exec_px - Pa), base_notional), 4)
    close_slip_bps   = _round(_bps(d * exec_qty * (avg_exec_px - Pf), base_notional), 4)

    # VWAP/TWAP references
    if window_bars:
        vwap_ref, _ = _vwap_from_bars(window_bars)
        twap_ref    = _twap_from_bars(window_bars)
    else:
        # Fallback: if no bars, use fills as proxy (this makes vwap_slip zero; still compute twap via arrival/final if window exists)
        vwap_ref, _ = _vwap_from_fills(fills) if fills else (Pa, 0.0)
        twap_ref    = statistics.mean([x for x in [order.arrival_px, order.final_px] if x]) if (order.arrival_px and order.final_px) else Pa

    vwap_slip_bps = _round(_bps(d * exec_qty * (avg_exec_px - vwap_ref), base_notional), 4)
    twap_slip_bps = _round(_bps(d * exec_qty * (avg_exec_px - twap_ref),  base_notional), 4)

    # Effective vs Quoted spread (at arrival)
    mid_arrival, half_spread_bps = _mid_from_bbo(order.arrival_px, arrival_bid, arrival_ask)
    eff_spread_bps = 0.0
    if mid_arrival and avg_exec_px:
        eff_spread_bps = _round(2.0 * d * (avg_exec_px - mid_arrival) / mid_arrival * 1e4, 4)  # positive implies paid the spread

    # Realized spread (use +5m markout by convention)
    realized_spread_bps = 0.0
    if ref_5m_px and avg_exec_px:
        # realized spread = 2 * d * (exec - mid@+5m)/mid@+5m
        realized_spread_bps = _round(2.0 * d * (avg_exec_px - ref_5m_px) / ref_5m_px * 1e4, 4)

    # Time to fill
    t_first, t_last = _first_last_fill_ts(fills)
    time_to_first = (t_first - (order.arrival_ts_ms or order.decision_ts_ms)) if (t_first and (order.arrival_ts_ms or order.decision_ts_ms)) else None
    time_to_full  = (t_last - (order.arrival_ts_ms or order.decision_ts_ms)) if (t_last and (order.arrival_ts_ms or order.decision_ts_ms) and abs(exec_qty - Q) < 1e-6) else None

    # Participation: executed notional vs market notional in window (if bars provided)
    part = None
    if window_bars:
        market_notional = sum(b.close * max(0.0, b.volume) for b in window_bars)
        exec_notional = exec_qty * avg_exec_px
        part = (exec_notional / market_notional) if market_notional>0 else None

    # Venues
    venues = _per_venue(fills)

    # Markouts package
    mo = _markout(order.side, avg_exec_px, ref_30s_px, ref_1m_px, ref_5m_px)

    return TCAOrderResult(
        order_id=order.order_id,
        symbol=order.symbol.upper(),
        side=order.side.lower(),
        qty_target=Q,
        qty_exec=_round(exec_qty, 6),
        fill_rate=_round(fill_rate, 6),
        avg_exec_px=_round(avg_exec_px, 8),
        is_cost_bps=is_cost_bps,
        arrival_slip_bps=arrival_slip_bps,
        close_slip_bps=close_slip_bps,
        vwap_slip_bps=_round(vwap_slip_bps, 4),
        twap_slip_bps=_round(twap_slip_bps, 4),
        effective_spread_bps=_round(eff_spread_bps, 4),
        quoted_half_spread_bps=_round(half_spread_bps, 4),
        realized_spread_bps=_round(realized_spread_bps, 4),
        delay_cost=_round(delay, 6),
        trade_cost=_round(trade, 6),
        opportunity_cost=_round(opp, 6),
        fees=_round(fees, 6),
        total_cost=_round(total, 6),
        time_to_first_fill_ms=time_to_first,
        time_to_full_fill_ms=time_to_full,
        participation_rate=(None if part is None else _round(part, 8)),
        venues=venues,
        markouts=mo,
    )

def aggregate(results: Iterable[TCAOrderResult]) -> TCAAggregate:
    res = list(results)
    if not res:
        return TCAAggregate(0, 0.0, 0.0, 0.0, 0.0, None, {"delay":0,"trade":0,"opp":0,"fees":0}, [])

    notionals = [abs(r.qty_target * max(r.avg_exec_px, 1e-9)) for r in res]
    W = sum(notionals) or 1.0
    vw = lambda xs: sum(x*w for x,w in zip(xs, notionals)) / W

    vw_cost_bps = _round(vw([r.is_cost_bps for r in res]), 4)
    med_cost_bps = _round(statistics.median([r.is_cost_bps for r in res]), 4)
    avg_fill_rate = _round(sum(r.fill_rate for r in res)/len(res), 6)
    ttfs = [r.time_to_full_fill_ms for r in res if r.time_to_full_fill_ms is not None]
    avg_ttf = _round(sum(ttfs)/len(ttfs), 2) if ttfs else None

    comps_vw = {
        "delay": _round(_bps(sum(r.delay_cost for r in res), sum(abs(r.qty_target * r.decision_px) for r in res)), 4), # type: ignore
        "trade": _round(_bps(sum(r.trade_cost for r in res), sum(abs(r.qty_target * r.decision_px) for r in res)), 4), # type: ignore
        "opp":   _round(_bps(sum(r.opportunity_cost for r in res), sum(abs(r.qty_target * r.decision_px) for r in res)), 4), # type: ignore
        "fees":  _round(_bps(sum(r.fees for r in res), sum(abs(r.qty_target * r.decision_px) for r in res)), 4), # type: ignore
    }

    # venue rollup
    roll: Dict[str, Dict[str, float]] = {}
    for r in res:
        for v in r.venues:
            row = roll.setdefault(v.venue, {"notional":0.0,"fills":0.0,"fees":0.0})
            row["notional"] += v.notional
            row["fills"] += v.fills
            row["fees"] += v.fees
    venue_table = [{"venue":k, **{kk:_round(vv,4) for kk,vv in v.items()}} for k,v in roll.items()]
    venue_table.sort(key=lambda x: x["notional"], reverse=True)

    return TCAAggregate(
        n_orders=len(res),
        gross_notional=_round(W, 6),
        vw_cost_bps=vw_cost_bps,
        median_cost_bps=med_cost_bps,
        avg_fill_rate=avg_fill_rate,
        avg_time_to_full_ms=avg_ttf,
        components_bps=comps_vw,
        venue_table=venue_table,
    )

# ---------- I/O helpers ----------
def read_orders_csv(path: str) -> List[OrderRow]:
    out: List[OrderRow] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(OrderRow(
                order_id=str(row.get("order_id")),
                symbol=str(row.get("symbol")),
                side=str(row.get("side")),
                qty=float(row.get("qty")), # type: ignore
                decision_px=float(row.get("decision_px")), # type: ignore
                decision_ts_ms=int(float(row.get("decision_ts_ms"))), # type: ignore
                arrival_px=_safe_float(row.get("arrival_px")),
                arrival_ts_ms=int(float(row["arrival_ts_ms"])) if row.get("arrival_ts_ms") else None,
                final_px=_safe_float(row.get("final_px")),
                start_ms=int(float(row["start_ms"])) if row.get("start_ms") else None,
                end_ms=int(float(row["end_ms"])) if row.get("end_ms") else None,
                currency=(row.get("currency") or "USD"),
                strategy=row.get("strategy") or None,
                venue_hint=row.get("venue") or None,
            ))
    return out

def read_fills_csv(path: str) -> List[FillRow]:
    out: List[FillRow] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(FillRow(
                order_id=str(row.get("order_id")),
                ts_ms=int(float(row.get("ts_ms"))), # type: ignore
                qty=float(row.get("qty")), # type: ignore
                price=float(row.get("price")), # type: ignore
                fee=float(row.get("fee") or 0.0),
                venue=row.get("venue") or None,
                child_id=row.get("child_id") or None,
            ))
    return out

def read_bars_csv(path: str) -> List[BarRow]:
    out: List[BarRow] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(BarRow(
                ts_ms=int(float(row.get("ts_ms"))), # type: ignore
                open=float(row.get("open")), # type: ignore
                high=float(row.get("high")), # type: ignore
                low=float(row.get("low")), # type: ignore
                close=float(row.get("close")), # type: ignore
                volume=float(row.get("volume") or 0.0),
            ))
    return out

# ---------- CLI ----------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("tca_extended")
    ap.add_argument("--orders", required=True, help="CSV orders (see schema in read_orders_csv)")
    ap.add_argument("--fills", required=True, help="CSV fills (see schema in read_fills_csv)")
    ap.add_argument("--bars", default=None, help="CSV bars for VWAP/TWAP (optional, order-window subset preferred)")
    ap.add_argument("--out-per-order", default="artifacts/tca/extended_per_order.jsonl")
    ap.add_argument("--out-summary", default="artifacts/tca/extended_summary.json")
    args = ap.parse_args()

    orders = read_orders_csv(args.orders)
    fills = read_fills_csv(args.fills)
    bars = read_bars_csv(args.bars) if args.bars else []

    # index helpers
    fills_by_oid: Dict[str, List[FillRow]] = {}
    for f in fills:
        fills_by_oid.setdefault(f.order_id, []).append(f)
    if bars and HAVE_PD:
        # speed: keep bars as DataFrame for quick slicing
        df = pd.DataFrame([asdict(b) for b in bars]) # type: ignore
        df = df.sort_values("ts_ms")
    else:
        df = None

    os.makedirs(os.path.dirname(args.out_per_order) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)

    results: List[TCAOrderResult] = []
    with open(args.out_per_order, "w", encoding="utf-8") as fout:
        for o in orders:
            my_fills = sorted(fills_by_oid.get(o.order_id, []), key=lambda x: x.ts_ms)
            # choose window bars: if start/end provided, slice; else use all bars
            window: Optional[List[BarRow]] = None
            if bars:
                if HAVE_PD and df is not None and o.start_ms and o.end_ms:
                    sub = df[(df.ts_ms>=o.start_ms) & (df.ts_ms<=o.end_ms)]
                    window = [BarRow(**rec) for rec in sub.to_dict("records")] # type: ignore
                else:
                    window = bars
            res = analyze_order(
                o, my_fills,
                window_bars=window,
                arrival_bid=o.meta.get("arrival_bid") if o.meta else None,
                arrival_ask=o.meta.get("arrival_ask") if o.meta else None,
                ref_30s_px=o.meta.get("markout_30s") if o.meta else None,
                ref_1m_px=o.meta.get("markout_1m") if o.meta else None,
                ref_5m_px=o.meta.get("markout_5m") if o.meta else None,
            )
            results.append(res)
            fout.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")

    agg = aggregate(results)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump({
            "n_orders": agg.n_orders,
            "gross_notional": agg.gross_notional,
            "vw_cost_bps": agg.vw_cost_bps,
            "median_cost_bps": agg.median_cost_bps,
            "avg_fill_rate": agg.avg_fill_rate,
            "avg_time_to_full_ms": agg.avg_time_to_full_ms,
            "components_bps": agg.components_bps,
            "venue_table": agg.venue_table,
        }, f, indent=2)
    print(f"[TCA] wrote {args.out_per_order} and {args.out_summary}")

if __name__ == "__main__":
    _cli()