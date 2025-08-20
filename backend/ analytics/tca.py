# backend/analytics/tca.py
"""
Transaction Cost Analysis (TCA)

What it does
------------
- Track orders, fills, cancel/replace.
- Compute:
  * Arrival/decision price (Px_decision)
  * Implementation Shortfall (IS) in bps and $:
      IS_bps = side_sign * (VWAP_fill - Px_decision) / Px_decision * 1e4
      IS_$   = side_sign * (VWAP_fill - Px_decision) * filled_qty
  * VWAP of fills
  * Slippage vs mid at each fill (bps)
  * Time to first/last fill (seconds)
  * Fill ratio (filled / ordered)
  * Cancel/replace count, #partials
- Aggregate by symbol / strategy / region.

How to use (minimal)
--------------------
tca = TCA(symbol_region_map={"RELIANCE.NS": "india", "AAPL": "us"}, csv_path="logs/tca_fills.csv")

# When you create an order from a strategy:
tca.record_order(order, strategy=strat.name, decision_px=arrival_mid, decision_ts=now_ts)

# On every fill:
tca.record_fill(fill, market_mid=mid_now, bid=bid_now, ask=ask_now)

# On cancel/replace:
tca.record_cancel(order_id)  # or .record_replace(old_id, new_order)

# Periodically (end of bar):
report = tca.snapshot()   # dict with per-order and aggregates

# Optional: mark an order done:
tca.close_order(order_id)
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Iterable


# ------------- Data models -------------

@dataclass
class OrderInfo:
    order_id: str
    symbol: str
    side: str                 # "buy" | "sell"
    qty: float                # target quantity (abs)
    strategy: str
    ts_created: float
    decision_px: float        # arrival/decision price
    decision_ts: float
    region: str = "unknown"
    attrs: Dict[str, Any] = field(default_factory=dict)  # venue, algo, etc.

    # runtime
    filled_qty: float = 0.0
    vwap_fill: float = 0.0
    n_partials: int = 0
    n_replaces: int = 0
    n_cancels: int = 0
    ts_first_fill: Optional[float] = None
    ts_last_fill: Optional[float] = None
    is_closed: bool = False

    def side_sign(self) -> int:
        # +1 for buys (paying up is bad), -1 for sells
        return +1 if self.side.lower().startswith("b") else -1

    def record_fill(self, qty: float, px: float, ts: float):
        prev = self.filled_qty
        self.filled_qty += qty
        # update VWAP
        if self.filled_qty > 0:
            self.vwap_fill = ((self.vwap_fill * prev) + (px * qty)) / self.filled_qty
        self.n_partials += 1
        if self.ts_first_fill is None:
            self.ts_first_fill = ts
        self.ts_last_fill = ts

    def time_to_first_fill(self) -> float:
        return (self.ts_first_fill - self.ts_created) if self.ts_first_fill else float("nan")

    def time_to_last_fill(self) -> float:
        return (self.ts_last_fill - self.ts_created) if self.ts_last_fill else float("nan")

    def fill_ratio(self) -> float:
        return (self.filled_qty / self.qty) if self.qty > 0 else 0.0


@dataclass
class FillInfo:
    order_id: str
    symbol: str
    side: str
    qty: float             # positive number for convenience here
    price: float
    fee: float
    ts: float
    # market context
    mid: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    def slippage_bps_vs_mid(self, side_sign: int) -> Optional[float]:
        if self.mid is None or self.mid <= 0:
            return None
        return side_sign * (self.price - self.mid) / self.mid * 1e4


@dataclass
class OrderTCA:
    order: OrderInfo
    fills: List[FillInfo] = field(default_factory=list)

    # computed
    is_bps: Optional[float] = None
    is_dollar: Optional[float] = None
    vwap_bps_vs_decision: Optional[float] = None
    fill_slippage_bps: List[float] = field(default_factory=list)

    def compute(self):
        if self.order.filled_qty <= 0 or self.order.decision_px <= 0:
            self.is_bps = None
            self.is_dollar = None
            self.vwap_bps_vs_decision = None
        else:
            s = self.order.side_sign()
            vwap = self.order.vwap_fill
            dec = self.order.decision_px
            self.is_bps = s * (vwap - dec) / dec * 1e4
            self.is_dollar = s * (vwap - dec) * self.order.filled_qty
            self.vwap_bps_vs_decision = self.is_bps

        self.fill_slippage_bps.clear()
        for f in self.fills:
            sl = f.slippage_bps_vs_mid(self.order.side_sign())
            if sl is not None and math.isfinite(sl):
                self.fill_slippage_bps.append(sl)


# ------------- TCA engine -------------

class TCA:
    def __init__(self, symbol_region_map: Optional[Dict[str, str]] = None, csv_path: Optional[str] = None):
        self.symbol_region_map = symbol_region_map or {}
        self.csv_path = csv_path

        # per order
        self.orders: Dict[str, OrderTCA] = {}

        # optional CSV writer for fills
        if self.csv_path:
            self._init_csv(self.csv_path)

    # ---- recording API ----

    def record_order(
        self,
        order: Any,
        strategy: str,
        decision_px: float,
        decision_ts: Optional[float] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ):
        """
        Register a new parent order.
        Required on `order`: id, symbol, side, qty, ts (creation time)
        """
        oid = getattr(order, "id", None) or getattr(order, "order_id", None)
        if not oid:
            raise ValueError("order.id/order_id required")
        sym = getattr(order, "symbol")
        side = getattr(order, "side")
        qty = float(getattr(order, "qty"))
        ts_created = float(getattr(order, "ts", time.time()))
        region = self.symbol_region_map.get(sym, "unknown")

        oi = OrderInfo(
            order_id=str(oid),
            symbol=sym,
            side=side,
            qty=abs(qty),
            strategy=str(strategy),
            ts_created=ts_created,
            decision_px=float(decision_px),
            decision_ts=float(decision_ts or ts_created),
            region=region,
            attrs=dict(attrs or {}),
        )
        self.orders[str(oid)] = OrderTCA(order=oi)

    def record_fill(
        self,
        fill: Any,
        market_mid: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ):
        """
        Apply a fill to an existing order.
        Required on `fill`: order_id, symbol, side, qty (signed or abs), price, fee, ts
        """
        oid = getattr(fill, "order_id", None)
        if oid is None:
            raise ValueError("fill.order_id required")
        if oid not in self.orders:
            # unknown parent (could be child order id) — create a stub from fill
            sym = getattr(fill, "symbol")
            side = getattr(fill, "side")
            qty = abs(float(getattr(fill, "qty")))
            ts = float(getattr(fill, "ts", time.time()))
            region = self.symbol_region_map.get(sym, "unknown")
            # decision price falls back to market mid if provided, else fill price
            dec_px = float(market_mid or getattr(fill, "price"))
            oi = OrderInfo(
                order_id=str(oid), symbol=sym, side=side, qty=qty,
                strategy="unknown", ts_created=ts, decision_px=dec_px, decision_ts=ts,
                region=region
            )
            self.orders[str(oid)] = OrderTCA(order=oi)

        o = self.orders[str(oid)].order
        q = abs(float(getattr(fill, "qty")))
        px = float(getattr(fill, "price"))
        fee = float(getattr(fill, "fee", 0.0))
        ts = float(getattr(fill, "ts", time.time()))
        # Update order accumulators
        o.record_fill(q, px, ts)

        f = FillInfo(
            order_id=str(oid),
            symbol=o.symbol,
            side=o.side,
            qty=q,
            price=px,
            fee=fee,
            ts=ts,
            mid=market_mid,
            bid=bid,
            ask=ask,
        )
        self.orders[str(oid)].fills.append(f)
        # compute fill slippage incrementally
        self.orders[str(oid)].compute()

        # CSV
        if self.csv_path:
            self._append_fill_csv(self.csv_path, o, f)

    def record_cancel(self, order_id: str):
        if order_id in self.orders:
            self.orders[order_id].order.n_cancels += 1
            self.orders[order_id].order.is_closed = True

    def record_replace(self, old_order_id: str, new_order: Any, decision_px: Optional[float] = None):
        """Increment replace count and optionally register the new order."""
        if old_order_id in self.orders:
            self.orders[old_order_id].order.n_replaces += 1
            self.orders[old_order_id].order.is_closed = True
        if new_order is not None:
            self.record_order(
                new_order,
                strategy=getattr(new_order, "strategy", "unknown"),
                decision_px=float(decision_px or getattr(new_order, "decision_px", 0.0)),
                decision_ts=getattr(new_order, "decision_ts", None),
                attrs=getattr(new_order, "attrs", None),
            )

    def close_order(self, order_id: str):
        if order_id in self.orders:
            self.orders[order_id].order.is_closed = True

    # ---- reporting ----

    def per_order(self) -> List[Dict[str, Any]]:
        """Flat list of per-order TCA stats."""
        out: List[Dict[str, Any]] = []
        for ot in self.orders.values():
            o = ot.order
            ot.compute()
            fill_slip = {
                "mean_slip_bps_vs_mid": (sum(ot.fill_slippage_bps) / len(ot.fill_slippage_bps))
                    if ot.fill_slippage_bps else None,
                "n_fills": len(ot.fills),
            }
            out.append({
                "order_id": o.order_id,
                "symbol": o.symbol,
                "side": o.side,
                "strategy": o.strategy,
                "region": o.region,
                "qty": o.qty,
                "filled_qty": o.filled_qty,
                "fill_ratio": o.fill_ratio(),
                "vwap_fill": o.vwap_fill if o.filled_qty > 0 else None,
                "decision_px": o.decision_px,
                "IS_bps": ot.is_bps,
                "IS_$": ot.is_dollar,
                "time_to_first_fill_s": o.time_to_first_fill(),
                "time_to_last_fill_s": o.time_to_last_fill(),
                "partials": o.n_partials,
                "replaces": o.n_replaces,
                "cancels": o.n_cancels,
                **fill_slip,
            })
        return out

    def snapshot(self) -> Dict[str, Any]:
        """
        Aggregates across dimensions:
          totals, by_symbol, by_strategy, by_region
        Metrics: avg IS_bps (qty-weighted), median IS_bps, VWAP vs decision (bps),
                 mean fill slippage vs mid (bps), fill ratio, time-to-fill medians.
        """
        per = self.per_order()

        def _agg(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
            rows = [r for r in rows if r.get("IS_bps") is not None and math.isfinite(r["IS_bps"])]
            if not rows:
                return {}
            # qty-weighted averages
            tot_qty = sum(float(r.get("filled_qty", 0.0)) for r in rows)
            if tot_qty <= 0:
                wavg_is = None
                wavg_fill_ratio = sum(r.get("fill_ratio", 0.0) for r in rows) / max(1, len(rows))
            else:
                wavg_is = sum(r["IS_bps"] * float(r.get("filled_qty", 0.0)) for r in rows) / tot_qty
                wavg_fill_ratio = sum(r.get("fill_ratio", 0.0) * float(r.get("qty", 0.0)) for r in rows) / \
                                  max(1e-9, sum(float(r.get("qty", 0.0)) for r in rows))
            # slippage averages
            slip_vals = [r["mean_slip_bps_vs_mid"] for r in rows if r.get("mean_slip_bps_vs_mid") is not None]
            slip_avg = sum(slip_vals) / len(slip_vals) if slip_vals else None

            # time to fill
            ttf_first = [r["time_to_first_fill_s"] for r in rows if r.get("time_to_first_fill_s") and math.isfinite(r["time_to_first_fill_s"])]
            ttf_last  = [r["time_to_last_fill_s"]  for r in rows if r.get("time_to_last_fill_s")  and math.isfinite(r["time_to_last_fill_s"])]
            med_first = _median(ttf_first) if ttf_first else None
            med_last  = _median(ttf_last)  if ttf_last  else None

            return {
                "orders": len(rows),
                "qty_filled": tot_qty,
                "is_bps_wavg": wavg_is,
                "fill_ratio_wavg": wavg_fill_ratio,
                "slippage_bps_vs_mid_avg": slip_avg,
                "ttf_first_med_s": med_first,
                "ttf_last_med_s": med_last,
            }

        # totals
        totals = _agg(per)

        # by symbol / strategy / region
        by_symbol: Dict[str, Any] = {}
        by_strategy: Dict[str, Any] = {}
        by_region: Dict[str, Any] = {}
        for r in per:
            sym = r["symbol"]; strat = r["strategy"]; reg = r["region"]
            by_symbol.setdefault(sym, []).append(r)
            by_strategy.setdefault(strat, []).append(r)
            by_region.setdefault(reg, []).append(r)

        return {
            "totals": totals,
            "by_symbol": {k: _agg(v) for k, v in by_symbol.items()},
            "by_strategy": {k: _agg(v) for k, v in by_strategy.items()},
            "by_region": {k: _agg(v) for k, v in by_region.items()},
            "per_order": per,  # keep the raw rows for drilldown
        }

    # ---- CSV helpers ----

    @staticmethod
    def _init_csv(path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "ts", "order_id", "symbol", "side", "strategy", "region",
                    "fill_qty", "fill_price", "fee",
                    "mid", "bid", "ask",
                ])

    @staticmethod
    def _append_fill_csv(path: str, o: OrderInfo, f: FillInfo):
        with open(path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                f"{f.ts:.3f}", o.order_id, o.symbol, o.side, o.strategy, o.region,
                f"{f.qty:.6f}", f"{f.price:.6f}", f"{f.fee:.6f}",
                f"{(f.mid if f.mid is not None else float('nan')):.6f}",
                f"{(f.bid if f.bid is not None else float('nan')):.6f}",
                f"{(f.ask if f.ask is not None else float('nan')):.6f}",
            ])


# ------------- small utils -------------

def _median(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    xs = sorted(xs)
    m = n // 2
    if n % 2 == 1:
        return xs[m]
    return 0.5 * (xs[m - 1] + xs[m])