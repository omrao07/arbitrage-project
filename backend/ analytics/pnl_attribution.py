# backend/analytics/pnl_attribution.py
"""
PnL Attribution (realized + unrealized) by Strategy, Region, and Symbol.

Assumptions / conventions:
- Every order/fill is tagged with a `strategy` name (string). If your OMS doesn't
  attach this yet, just pass the strategy name alongside fill events when calling
  `record_fill(...)`.
- Region for a symbol is supplied by a `symbol_region_map` dict (e.g., {"RELIANCE.NS":"india"}).
- Fees are included per-fill.
- Supports:
    - Per-strategy, per-region, per-symbol realized PnL
    - Live unrealized PnL (mark-to-market) per strategy/region/symbol
    - Aggregation snapshots, CSV logging (optional)

Minimal integration points in your loop:
    1) On every executed fill:
        attributor.record_fill(fill, strategy="my_strategy")
    2) On every bar after prices update:
        attributor.mark_to_market(prices)
    3) When you want a breakdown:
        summary = attributor.snapshot()

`fill` required fields:
    - fill.order_id: str
    - fill.symbol: str
    - fill.qty: float     (signed: +buy, -sell)
    - fill.price: float
    - fill.fee: float
    - fill.ts: float
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any


# ---------------------- Data models ----------------------

@dataclass
class LegPosition:
    """Position for (symbol, strategy) leg."""
    qty: float = 0.0
    avg_price: float = 0.0

    def apply_fill(self, qty: float, price: float):
        """
        Update position average cost when quantity increases in same direction,
        or realize PnL when reducing/reversing.
        Returns: realized_pnl from this application (float).
        """
        realized = 0.0

        # If adding in same direction or building from zero:
        if self.qty == 0 or (self.qty > 0 and qty > 0) or (self.qty < 0 and qty < 0):
            new_qty = self.qty + qty
            if new_qty != 0:
                # weighted average price update
                self.avg_price = (
                    (abs(self.qty) * self.avg_price + abs(qty) * price) / abs(new_qty)
                )
            else:
                self.avg_price = 0.0
            self.qty = new_qty
            return realized

        # Else we are reducing or flipping direction:
        # Realize PnL on the portion that offsets existing position.
        closing_qty = qty
        # If flip, part (or all) offsets old exposure.
        if (self.qty > 0 and qty < 0) or (self.qty < 0 and qty > 0):
            # Amount that offsets old qty (limited by existing exposure)
            offset = -min(abs(self.qty), abs(qty)) * (1 if qty < 0 else -1)
            # Realized PnL = (sell_price - avg_price)*closed_qty for longs
            # For shorts, sign naturally handled by qty signs below.
            realized = (self.avg_price - price) * offset  # note: offset is signed opposite
            # Update qty after offset
            self.qty += offset

            # Remaining qty (if any) builds a position in the new direction
            remainder = qty - offset
            if remainder != 0:
                # opening new position in opposite direction; avg resets to fill price
                self.avg_price = price
                self.qty += remainder
            else:
                if self.qty == 0:
                    self.avg_price = 0.0
            return realized

        return realized  # should not reach here


@dataclass
class Node:
    """Holds realized and unrealized PnL buckets."""
    realized: float = 0.0
    unrealized: float = 0.0
    fees: float = 0.0


# ---------------------- Attributor ----------------------

class PnLAttributor:
    """
    Tracks PnL at three levels:
      - per-symbol
      - per-strategy
      - per-region
    And their intersections (strategy×symbol, region×strategy).

    Usage:
        attr = PnLAttributor(symbol_region_map={"RELIANCE.NS":"india"})
        attr.record_fill(fill, strategy="mean_rev")   # on each trade
        attr.mark_to_market(prices)                   # each bar
        snap = attr.snapshot()                        # when needed
    """

    def __init__(
        self,
        symbol_region_map: Optional[Dict[str, str]] = None,
        csv_path: Optional[str] = None,
    ):
        self.symbol_region_map = symbol_region_map or {}
        self.csv_path = csv_path

        # Positions keyed by (symbol, strategy)
        self.legs: Dict[Tuple[str, str], LegPosition] = {}

        # PnL buckets
        self.by_symbol: Dict[str, Node] = {}
        self.by_strategy: Dict[str, Node] = {}
        self.by_region: Dict[str, Node] = {}
        self.by_strategy_symbol: Dict[Tuple[str, str], Node] = {}
        self.by_region_strategy: Dict[Tuple[str, str], Node] = {}

        # Last prices cache for unrealized calc
        self.last_prices: Dict[str, float] = {}

        # Optional CSV writer
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "ts", "symbol", "strategy", "region",
                        "fill_qty", "fill_price", "fee",
                        "realized_increment", "pos_qty", "pos_avg_px",
                        "unrealized_after", "price"
                    ])

    # --------------- Public API ---------------

    def record_fill(self, fill: Any, strategy: str):
        """
        Apply a fill to the correct (symbol, strategy) leg; update realized pnl and fees.
        `fill` fields required: symbol, qty (signed), price, fee, ts.
        """
        sym = fill.symbol
        qty = float(fill.qty)     # signed: +buy, -sell
        px  = float(fill.price)
        fee = float(getattr(fill, "fee", 0.0))
        ts  = float(getattr(fill, "ts", 0.0))
        region = self.symbol_region_map.get(sym, "unknown")

        leg_key = (sym, strategy)
        leg = self.legs.get(leg_key, LegPosition())
        realized_inc = leg.apply_fill(qty, px)
        self.legs[leg_key] = leg

        # Update buckets
        self._node(self.by_symbol, sym).realized += realized_inc - fee
        self._node(self.by_strategy, strategy).realized += realized_inc - fee
        self._node(self.by_region, region).realized += realized_inc - fee
        self._node(self.by_strategy_symbol, (strategy, sym)).realized += realized_inc - fee
        self._node(self.by_region_strategy, (region, strategy)).realized += realized_inc - fee

        # Maintain fees separately if desired
        self._node(self.by_symbol, sym).fees += fee
        self._node(self.by_strategy, strategy).fees += fee
        self._node(self.by_region, region).fees += fee
        self._node(self.by_strategy_symbol, (strategy, sym)).fees += fee
        self._node(self.by_region_strategy, (region, strategy)).fees += fee

        # CSV (optional)
        if self.csv_path:
            self._append_csv(ts, sym, strategy, region, qty, px, fee, realized_inc, leg)

    def mark_to_market(self, prices: Dict[str, float]):
        """
        Recompute unrealized PnL for all legs given a prices dict {symbol: last_price}.
        """
        self.last_prices.update(prices)

        # Reset unrealized buckets
        for node in self.by_symbol.values(): node.unrealized = 0.0
        for node in self.by_strategy.values(): node.unrealized = 0.0
        for node in self.by_region.values(): node.unrealized = 0.0
        for node in self.by_strategy_symbol.values(): node.unrealized = 0.0
        for node in self.by_region_strategy.values(): node.unrealized = 0.0

        # Recompute from positions
        for (sym, strat), leg in self.legs.items():
            if abs(leg.qty) < 1e-12:
                continue
            px = self.last_prices.get(sym)
            if px is None:
                continue
            u = (px - leg.avg_price) * leg.qty  # works for long/short
            region = self.symbol_region_map.get(sym, "unknown")

            self._node(self.by_symbol, sym).unrealized += u
            self._node(self.by_strategy, strat).unrealized += u
            self._node(self.by_region, region).unrealized += u
            self._node(self.by_strategy_symbol, (strat, sym)).unrealized += u
            self._node(self.by_region_strategy, (region, strat)).unrealized += u

    def snapshot(self) -> Dict[str, Dict]:
        """
        Return a nested dict with breakdowns and totals:
        {
          "totals": {"realized": x, "unrealized": y, "fees": z, "pnl": x+y-z},
          "by_strategy": {strategy: {...}},
          "by_region": {region: {...}},
          "by_symbol": {symbol: {...}},
          "by_region_strategy": {(region,strat): {...}},
          "by_strategy_symbol": {(strat,symbol): {...}}
        }
        """
        def summarize(d: Dict) -> Dict:
            out = {}
            for k, node in d.items():
                out[str(k)] = {
                    "realized": node.realized,
                    "unrealized": node.unrealized,
                    "fees": node.fees,
                    "pnl": node.realized + node.unrealized - node.fees,
                }
            return out

        totals = Node()
        for node in self.by_strategy.values():
            totals.realized += node.realized
            totals.unrealized += node.unrealized
            totals.fees += node.fees

        return {
            "totals": {
                "realized": totals.realized,
                "unrealized": totals.unrealized,
                "fees": totals.fees,
                "pnl": totals.realized + totals.unrealized - totals.fees,
            },
            "by_strategy": summarize(self.by_strategy),
            "by_region": summarize(self.by_region),
            "by_symbol": summarize(self.by_symbol),
            "by_region_strategy": summarize(self.by_region_strategy),
            "by_strategy_symbol": summarize(self.by_strategy_symbol),
        }

    # --------------- Internals ---------------

    @staticmethod
    def _node(d: Dict, key) -> Node:
        n = d.get(key)
        if n is None:
            n = Node()
            d[key] = n
        return n

    def _append_csv(
        self,
        ts: float,
        sym: str,
        strategy: str,
        region: str,
        qty: float,
        price: float,
        fee: float,
        realized_inc: float,
        leg: LegPosition,
    ):
        with open(self.csv_path, "a", newline="") as f: # type: ignore
            csv.writer(f).writerow(
                [
                    f"{ts:.3f}",
                    sym,
                    strategy,
                    region,
                    f"{qty:.6f}",
                    f"{price:.6f}",
                    f"{fee:.6f}",
                    f"{realized_inc:.6f}",
                    f"{leg.qty:.6f}",
                    f"{leg.avg_price:.6f}",
                    f"{(self.last_prices.get(sym, leg.avg_price) - leg.avg_price) * leg.qty:.6f}",
                    f"{self.last_prices.get(sym, 0.0):.6f}",
                ]
            )