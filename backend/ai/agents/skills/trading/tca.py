# backend/analytics/tca.py
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Trade:
    order_id: str
    symbol: str
    side: str              # "buy" | "sell"
    qty: float
    px_arrival: float      # mid or ref price at decision
    px_exec: float         # actual execution price
    ts_arrival: float
    ts_exec: float
    venue: Optional[str] = None
    fee_bps: float = 0.0
    notes: Dict[str, float] = field(default_factory=dict)

@dataclass
class TCAResult:
    symbol: str
    total_qty: float
    avg_slippage_bps: float
    avg_cost_bps: float
    implementation_shortfall: float
    trades: int
    details: Dict[str, float] = field(default_factory=dict)

class TCAEngine:
    """
    Transaction Cost Analysis (TCA).
    Measures:
      - Slippage vs. arrival price
      - Implementation shortfall
      - Average cost incl. fees
      - Venue stats
    """

    def __init__(self):
        self._trades: List[Trade] = []

    def add_trade(self, trade: Trade):
        self._trades.append(trade)

    def analyze(self, symbol: Optional[str] = None) -> List[TCAResult]:
        trades = [t for t in self._trades if (symbol is None or t.symbol == symbol)]
        by_sym: Dict[str, List[Trade]] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)

        results: List[TCAResult] = []
        for sym, ts in by_sym.items():
            tot_qty = sum(t.qty for t in ts)
            if tot_qty <= 0:
                continue
            slips, costs, isfs = [], [], []
            for t in ts:
                sign = +1 if t.side == "buy" else -1
                slip_bps = ((t.px_exec - t.px_arrival) / t.px_arrival) * 1e4 * sign
                isf = (t.px_exec - t.px_arrival) * sign * t.qty
                cost_bps = slip_bps + t.fee_bps
                slips.append(slip_bps)
                isfs.append(isf)
                costs.append(cost_bps)

            avg_slip = statistics.mean(slips) if slips else 0.0
            avg_cost = statistics.mean(costs) if costs else 0.0
            impl_shortfall = sum(isfs) / tot_qty if tot_qty > 0 else 0.0

            results.append(TCAResult(
                symbol=sym,
                total_qty=tot_qty,
                avg_slippage_bps=avg_slip,
                avg_cost_bps=avg_cost,
                implementation_shortfall=impl_shortfall,
                trades=len(ts),
                details={
                    "max_slip_bps": max(slips) if slips else 0.0,
                    "min_slip_bps": min(slips) if slips else 0.0,
                    "venues": len(set(t.venue for t in ts if t.venue)),
                }
            ))
        return results

if __name__ == "__main__":
    tca = TCAEngine()
    tca.add_trade(Trade(order_id="1", symbol="AAPL", side="buy",
                        qty=100, px_arrival=190.0, px_exec=191.0,
                        ts_arrival=1, ts_exec=2, venue="NASDAQ", fee_bps=0.2))
    tca.add_trade(Trade(order_id="2", symbol="AAPL", side="sell",
                        qty=50, px_arrival=192.0, px_exec=191.5,
                        ts_arrival=3, ts_exec=4, venue="NASDAQ", fee_bps=0.1))
    res = tca.analyze()
    for r in res:
        print(r)