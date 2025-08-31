# backend/tca/implementation_shortfall.py
from __future__ import annotations

import csv, json, math, os, statistics, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------- helpers -----------------------
def now_ms() -> int: return int(time.time() * 1000)

def _sgn(side: str) -> int:
    s = side.lower()
    if s in ("buy", "b", "long"): return +1
    if s in ("sell", "s", "short"): return -1
    raise ValueError(f"side must be buy/sell, got {side}")

def _round(x: float, n: int = 8) -> float:
    try: return float(round(float(x), n))
    except Exception: return 0.0

def _safe_div(a: float, b: float) -> float:
    return a / b if b not in (0, 0.0, None) else 0.0

# ----------------------- data shapes -------------------
@dataclass
class OrderIntent:
    order_id: str
    symbol: str
    side: str                # "buy" | "sell"
    qty: float               # target absolute quantity
    decision_px: float       # price when investment decision was made
    decision_ts_ms: int      # time of decision (ms)
    arrival_px: Optional[float] = None   # price when order released to market (can = decision_px)
    arrival_ts_ms: Optional[int] = None
    final_px: Optional[float] = None     # price at end of horizon (e.g., close)
    currency: str = "USD"
    strategy: Optional[str] = None
    venue: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class Fill:
    order_id: str
    ts_ms: int
    qty: float               # absolute quantity executed in this fill
    price: float
    fee: float = 0.0         # commission + fees in currency
    venue: Optional[str] = None

@dataclass
class ISResult:
    order_id: str
    symbol: str
    side: str
    qty_target: float
    qty_exec: float
    avg_exec_px: float
    fill_rate: float                     # 0..1
    # cost components (in currency, signed from portfolio POV: positive = cost)
    delay_cost: float
    trade_cost: float
    fees: float
    opportunity_cost: float
    total_cost: float                    # sum of components (currency)
    # normalization (bps)
    cost_bps: float
    delay_bps: float
    trade_bps: float
    opp_bps: float
    # metadata
    decision_px: float
    arrival_px: float
    final_px: float
    currency: str
    strategy: Optional[str] = None
    venue: Optional[str] = None

# ----------------------- core math --------------------
def compute_is(
    order: OrderIntent,
    fills: List[Fill],
    *,
    # If arrival or final price missing on order, you may provide fallback here:
    fallback_arrival_px: Optional[float] = None,
    fallback_final_px: Optional[float] = None,
) -> ISResult:
    """
    Implementation Shortfall decomposition (Perold 1988, Almgren et al.):
      Signed direction d = +1 for buy, -1 for sell (portfolio long-only convention).
      Decision value: d * Q * Pd
      Execution value: d * q_exec * P_exec (volume-weighted)
      Unexecuted opportunity: d * (Q - q_exec) * (Pf - Pd)
      Delay: d * q_exec * (Pa - Pd)          (decision -> arrival)
      Trade: d * q_exec * (P_exec - Pa)      (arrival -> execution)
      Fees:  sum(fees)
      IS(total) = Delay + Trade + Fees + Opportunity

    All costs returned in currency and in bps relative to |Q| * Pd.
    """
    d = _sgn(order.side)
    Q = float(order.qty)
    if Q <= 0:
        raise ValueError("order.qty must be > 0")

    Pd = float(order.decision_px)
    Pa = float(order.arrival_px or fallback_arrival_px or Pd)
    Pf = float(order.final_px or fallback_final_px or Pd)

    # executed stats
    exec_qty = sum(abs(f.qty) for f in fills if f.order_id == order.order_id)
    notional = sum(abs(f.qty) * f.price for f in fills if f.order_id == order.order_id)
    fees = sum(getattr(f, "fee", 0.0) or 0.0 for f in fills if f.order_id == order.order_id)
    P_exec = _round(_safe_div(notional, exec_qty), 10) if exec_qty > 0 else 0.0

    # components
    # delay cost only on the executed part
    delay = d * exec_qty * (Pa - Pd)
    # trade cost only on the executed part
    trade = d * exec_qty * (P_exec - Pa)
    # opportunity cost on the unexecuted remainder measured vs decision → final
    opp = d * (Q - exec_qty) * (Pf - Pd)

    total = float(delay + trade + fees + opp)

    # normalization base: |Q| * Pd
    base = abs(Q * Pd)
    cost_bps = _round(_safe_div(total, base) * 1e4, 4)
    delay_bps = _round(_safe_div(delay, base) * 1e4, 4)
    trade_bps = _round(_safe_div(trade, base) * 1e4, 4)
    opp_bps   = _round(_safe_div(opp,   base) * 1e4, 4)

    return ISResult(
        order_id=order.order_id,
        symbol=order.symbol.upper(),
        side=order.side.lower(),
        qty_target=Q,
        qty_exec=_round(exec_qty, 6),
        avg_exec_px=_round(P_exec, 10),
        fill_rate=_round(_safe_div(exec_qty, Q), 6),
        delay_cost=_round(delay, 6),
        trade_cost=_round(trade, 6),
        fees=_round(fees, 6),
        opportunity_cost=_round(opp, 6),
        total_cost=_round(total, 6),
        cost_bps=cost_bps,
        delay_bps=delay_bps,
        trade_bps=trade_bps,
        opp_bps=opp_bps,
        decision_px=Pd,
        arrival_px=Pa,
        final_px=Pf,
        currency=order.currency,
        strategy=order.strategy,
        venue=order.venue,
    )

# ----------------------- aggregation ------------------
@dataclass
class ISAggregate:
    n_orders: int
    gross_notional: float
    avg_fill_rate: float
    vw_cost_bps: float               # notional-weighted IS in bps
    median_cost_bps: float
    components_bps: Dict[str, float] # delay/trade/opp vw-bps
    totals: Dict[str, float]         # currency totals for components

def aggregate(results: Iterable[ISResult]) -> ISAggregate:
    res = list(results)
    if not res:
        return ISAggregate(0, 0.0, 0.0, 0.0, 0.0, {"delay":0,"trade":0,"opp":0}, {"delay":0,"trade":0,"fees":0,"opp":0,"total":0})

    notionals = [abs(r.qty_target * r.decision_px) for r in res]
    W = sum(notionals) or 1.0
    vw = lambda xs: _safe_div(sum(x*w for x,w in zip(xs, notionals)), W)

    vw_cost_bps = _round(vw([r.cost_bps for r in res]), 4)
    avg_fill_rate = _round(sum(r.fill_rate for r in res) / len(res), 6)
    med_cost_bps = _round(statistics.median([r.cost_bps for r in res]), 4)

    comps_vw = {
        "delay": _round(vw([r.delay_bps for r in res]), 4),
        "trade": _round(vw([r.trade_bps for r in res]), 4),
        "opp":   _round(vw([r.opp_bps   for r in res]), 4),
    }
    totals = {
        "delay": _round(sum(r.delay_cost for r in res), 6),
        "trade": _round(sum(r.trade_cost for r in res), 6),
        "fees":  _round(sum(r.fees for r in res), 6),
        "opp":   _round(sum(r.opportunity_cost for r in res), 6),
        "total": _round(sum(r.total_cost for r in res), 6),
    }
    return ISAggregate(
        n_orders=len(res),
        gross_notional=_round(W, 6),
        avg_fill_rate=avg_fill_rate,
        vw_cost_bps=vw_cost_bps,
        median_cost_bps=med_cost_bps,
        components_bps=comps_vw,
        totals=totals,
    )

# ----------------------- I/O utilities ----------------
def read_orders_csv(path: str) -> List[OrderIntent]:
    """
    CSV columns (header case-insensitive):
      order_id,symbol,side,qty,decision_px,decision_ts_ms,arrival_px,arrival_ts_ms,final_px,currency,strategy,venue
    Missing numeric cells allowed (empty string).
    """
    out: List[OrderIntent] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            def fnum(k, d=None):
                v = row.get(k, "")
                try: return float(v) if v not in ("", None) else d
                except Exception: return d
            def fint(k, d=None):
                v = row.get(k, "")
                try: return int(float(v)) if v not in ("", None) else d
                except Exception: return d
            out.append(OrderIntent(
                order_id=str(row.get("order_id")),
                symbol=str(row.get("symbol")),
                side=str(row.get("side")),
                qty=float(row.get("qty")), # type: ignore
                decision_px=float(row.get("decision_px")), # type: ignore
                decision_ts_ms=fint("decision_ts_ms", now_ms()),
                arrival_px=fnum("arrival_px"),
                arrival_ts_ms=fint("arrival_ts_ms"),
                final_px=fnum("final_px"),
                currency=(row.get("currency") or "USD"),
                strategy=row.get("strategy") or None,
                venue=row.get("venue") or None,
                meta=None,
            ))
    return out

def read_fills_csv(path: str) -> List[Fill]:
    """
    CSV columns:
      order_id,ts_ms,qty,price,fee,venue
    """
    out: List[Fill] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(Fill(
                order_id=str(row.get("order_id")),
                ts_ms=int(float(row.get("ts_ms"))), # type: ignore
                qty=float(row.get("qty")), # type: ignore
                price=float(row.get("price")), # type: ignore
                fee=float(row.get("fee") or 0.0),
                venue=row.get("venue") or None,
            ))
    return out

def write_results_csv(path: str, results: Iterable[ISResult]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [asdict(r) for r in results]
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")  # create empty
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ----------------------- CLI --------------------------
def _cli():
    import argparse, json
    ap = argparse.ArgumentParser("implementation_shortfall")
    ap.add_argument("--orders", type=str, required=True, help="CSV of orders")
    ap.add_argument("--fills", type=str, required=True, help="CSV of fills")
    ap.add_argument("--out", type=str, default="artifacts/tca/is_results.csv")
    ap.add_argument("--summary", type=str, default="artifacts/tca/is_summary.json")
    ap.add_argument("--fallback-arrival", type=float, default=None, help="Use if arrival_px missing")
    ap.add_argument("--fallback-final", type=float, default=None, help="Use if final_px missing")
    args = ap.parse_args()

    orders = read_orders_csv(args.orders)
    fills = read_fills_csv(args.fills)

    results: List[ISResult] = []
    for o in orders:
        r = compute_is(o, fills, fallback_arrival_px=args.fallback_arrival, fallback_final_px=args.fallback_final)
        results.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_results_csv(args.out, results)

    agg = aggregate(results)
    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump({
            "n_orders": agg.n_orders,
            "gross_notional": agg.gross_notional,
            "avg_fill_rate": agg.avg_fill_rate,
            "vw_cost_bps": agg.vw_cost_bps,
            "median_cost_bps": agg.median_cost_bps,
            "components_bps": agg.components_bps,
            "totals_currency": agg.totals,
        }, f, indent=2)
    print(f"[IS] wrote {args.out} and {args.summary}")

if __name__ == "__main__":
    _cli()