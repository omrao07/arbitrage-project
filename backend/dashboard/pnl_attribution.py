# backend/analytics/pnl_attribution.py
"""
PnL Attribution (wrapper + Brinson-style effects)

What this provides
------------------
1) A thin, friendly wrapper around your existing PnL engine (PnLXray)
   so strategies/OMS can call one place for:
     - ingest_trade(), update_mark(), snapshot(group_by=...)
     - to_json()/from_json()
2) Optional Brinson-style attribution (allocation / selection / interaction)
   against a benchmark, aggregated by a dimension (e.g., sector, region).

No hard deps; pandas is optional for pretty tables.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ----------------------- Optional pandas -------------------------------------
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# ----------------------- Link to your PnL X-Ray ------------------------------
try:
    from backend.analytics.pnl_xray import PnLXray  # type: ignore
except Exception as e:
    # Soft fallback if pnl_xray isn't present yet
    class PnLXray:  # type: ignore
        def __init__(self, *a, **k): raise ImportError(
            "backend.analytics.pnl_xray not found. Please add pnl_xray.py from your repo."
        ) from e


# =============================================================================
# Simple wrapper around PnLXray
# =============================================================================

class PnLAttribution:
    """
    Facade around PnLXray with a compact API + convenience helpers.
    """

    def __init__(self, *, base_ccy: str = "USD"):
        self.xr = PnLXray(base_ccy=base_ccy)

    # ---------- Event ingestion ----------

    def ingest_trade(self, trade: Mapping[str, Any]) -> None:
        """
        trade keys expected:
          ts (sec), strategy, symbol, side ('buy'/'sell'), qty, price
          optional: venue, region, book, fees
        """
        self.xr.ingest_trade(dict(trade)) # type: ignore

    def update_mark(self, symbol: str, price: float, ts: Optional[float] = None) -> None:
        self.xr.update_mark(symbol, float(price), ts) # type: ignore

    # ---------- Factors (optional) ----------

    def set_benchmark_return(self, r: float) -> None:
        self.xr.set_benchmark_return(float(r)) # type: ignore

    def set_beta(self, symbol: str, beta: float) -> None:
        self.xr.set_beta(symbol.upper(), float(beta)) # type: ignore

    def set_trade_tags_callback(self, cb) -> None:
        self.xr.set_trade_tags_callback(cb) # type: ignore

    # ---------- Reporting ----------

    def snapshot(self, *, group_by: Iterable[str] = ("strategy", "symbol")) -> Dict[str, Any]:
        """
        Valid group_by dims: strategy, symbol, venue, region, sector, book, side
        Returns dict with 'rows' list and 'ts'.
        """
        return self.xr.snapshot(group_by=tuple(group_by)) # type: ignore

    def to_json(self) -> str:
        return self.xr.to_json() # type: ignore

    @staticmethod
    def from_json(s: str) -> "PnLAttribution":
        obj = PnLAttribution()
        obj.xr = PnLXray.from_json(s) # type: ignore
        return obj

    # ---------- Pretty DataFrames ----------

    def to_dataframe(self, *, group_by: Iterable[str] = ("strategy","symbol")):
        if pd is None:
            raise RuntimeError("pandas not installed")
        return self.xr.to_dataframe(group_by=tuple(group_by)) # type: ignore


# =============================================================================
# Brinson-style attribution
# =============================================================================

@dataclass
class BrinsonResult:
    group_by: str                    # the dimension key used (e.g., 'sector')
    rows: List[Dict[str, Any]]       # per-bucket effects
    totals: Dict[str, float]         # grand totals
    asof_ts: int

    def to_json(self) -> str:
        return json.dumps({
            "group_by": self.group_by,
            "rows": self.rows,
            "totals": self.totals,
            "asof_ts": self.asof_ts,
        }, indent=2)

def brinson_attribution(
    *,
    portfolio_weights: Mapping[str, float],      # bucket -> weight (sum ~ 1.0, can be signed)
    benchmark_weights: Mapping[str, float],      # bucket -> weight (sum ~ 1.0)
    portfolio_returns: Mapping[str, float],      # bucket -> period return (e.g., 0.012)
    benchmark_returns: Mapping[str, float],      # bucket -> period return
    dimension: str = "bucket",
    interaction: bool = True,
) -> BrinsonResult:
    """
    Basic Brinson-Fachler effects by bucket:
      Allocation  : (w_p - w_b) * r_b
      Selection   : w_b * (r_p - r_b)
      Interaction : (w_p - w_b) * (r_p - r_b)   (optional)

    Inputs can omit some buckets; missing entries default to 0.
    """
    buckets = sorted(set(portfolio_weights) | set(benchmark_weights) | set(portfolio_returns) | set(benchmark_returns))

    rows: List[Dict[str, Any]] = []
    tot_alloc = tot_sel = tot_int = 0.0

    for b in buckets:
        wp = float(portfolio_weights.get(b, 0.0))
        wb = float(benchmark_weights.get(b, 0.0))
        rp = float(portfolio_returns.get(b, 0.0))
        rb = float(benchmark_returns.get(b, 0.0))

        alloc = (wp - wb) * rb
        sel   = wb * (rp - rb)
        inter = (wp - wb) * (rp - rb) if interaction else 0.0

        rows.append({
            dimension: b,
            "weight_port": wp,
            "weight_bench": wb,
            "ret_port": rp,
            "ret_bench": rb,
            "effect_allocation": alloc,
            "effect_selection": sel,
            "effect_interaction": inter,
            "effect_total": alloc + sel + inter,
        })
        tot_alloc += alloc
        tot_sel   += sel
        tot_int   += inter

    rows.sort(key=lambda r: -r["effect_total"])
    totals = {
        "allocation": tot_alloc,
        "selection": tot_sel,
        "interaction": tot_int if interaction else 0.0,
        "total_excess": tot_alloc + tot_sel + (tot_int if interaction else 0.0),
    }
    return BrinsonResult(group_by=dimension, rows=rows, totals=totals, asof_ts=int(time.time() * 1000))


# =============================================================================
# Tiny demo / CLI
# =============================================================================

if __name__ == "__main__":
    # --- Part A: PnLAttribution wrapper demo (uses PnLXray under the hood)
    pa = PnLAttribution(base_ccy="USD")
    pa.ingest_trade({"ts": time.time(), "strategy": "alpha.meanrev", "symbol": "AAPL", "side": "buy",  "qty": 100, "price": 190.0, "venue": "NASDAQ", "region": "US"})
    pa.update_mark("AAPL", 191.0)
    pa.ingest_trade({"ts": time.time(), "strategy": "alpha.meanrev", "symbol": "AAPL", "side": "sell", "qty":  50, "price": 191.5, "venue": "NASDAQ", "region": "US"})
    pa.update_mark("AAPL", 190.8)

    snap = pa.snapshot(group_by=("strategy","symbol"))
    print(json.dumps(snap, indent=2))

    # --- Part B: Brinson demo by sector
    port_w = {"Tech": 0.45, "Energy": 0.15, "Health": 0.20, "Other": 0.20}
    bench_w= {"Tech": 0.35, "Energy": 0.10, "Health": 0.25, "Other": 0.30}
    port_r = {"Tech": 0.012, "Energy": 0.020, "Health": -0.004, "Other": 0.006}
    bench_r= {"Tech": 0.010, "Energy": 0.018, "Health": -0.002, "Other": 0.005}

    br = brinson_attribution(
        portfolio_weights=port_w,
        benchmark_weights=bench_w,
        portfolio_returns=port_r,
        benchmark_returns=bench_r,
        dimension="sector",
    )
    print(br.to_json())

    if pd:
        df = pd.DataFrame(br.rows)
        print(df.round(6))