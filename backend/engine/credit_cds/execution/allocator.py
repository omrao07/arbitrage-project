# engines/credit_cds/allocator.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class AllocatorConfig:
    leverage_cap: float = 1.0          # gross notional / NAV cap
    max_per_name: float = 0.25         # max |weight| per name before scaling
    allow_short_protection: bool = True
    tc_bps: float = 1.0                # transaction cost bps on traded notional

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _cap_and_normalize(weights: pd.Series, cfg: AllocatorConfig) -> pd.Series:
    """
    Clip to per-name max, then normalize so gross = leverage_cap.
    """
    w = weights.fillna(0.0).astype(float)
    if not cfg.allow_short_protection:
        w = w.clip(lower=0.0)
    if cfg.max_per_name is not None:
        w = w.clip(lower=-cfg.max_per_name, upper=cfg.max_per_name)

    gross = w.abs().sum()
    if gross > 0:
        w = w * (cfg.leverage_cap / gross)
    return w

# ---------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------

def allocate_targets(
    *,
    weights: pd.Series,        # raw strategy signals (arbitrary scale)
    nav: float,                # portfolio NAV
    cfg: AllocatorConfig = AllocatorConfig(),
) -> pd.Series:
    """
    Convert raw strategy weights into **target notionals** (USD).
    +ve → long protection (buy CDS), -ve → short protection (sell CDS).
    """
    w = _cap_and_normalize(weights, cfg)
    return w * nav  # target notionals

def compute_trades(
    *,
    current_pos: pd.Series,
    target_pos: pd.Series,
    cfg: AllocatorConfig = AllocatorConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Compute required trades to move from current to target positions.
    Returns:
      {
        "trades": DataFrame[ticker, trade_notional, side, cost$],
        "total_cost$": float
      }
    """
    tickers = target_pos.index
    trade = (target_pos - current_pos).reindex(tickers).fillna(0.0)
    trades = []

    for n, q in trade.items():
        if abs(q) < 1e-8:
            continue
        side = "BUY_PROTECTION" if q > 0 else "SELL_PROTECTION"
        cost = abs(q) * (cfg.tc_bps * 1e-4)
        trades.append({"ticker": n, "trade_notional": float(q), "side": side, "cost$": cost})

    trades_df = pd.DataFrame(trades, columns=["ticker", "trade_notional", "side", "cost$"])
    return {"trades": trades_df, "total_cost$": trades_df["cost$"].sum() if not trades_df.empty else 0.0} # type: ignore

# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    nav = 1_000_000
    raw_weights = pd.Series({"IG_A": 0.6, "HY_B": -0.2, "EM_C": 0.3})

    cfg = AllocatorConfig(leverage_cap=1.5, max_per_name=0.5, tc_bps=2.0)

    targets = allocate_targets(weights=raw_weights, nav=nav, cfg=cfg)
    print("Target notionals:")
    print(targets)

    current = pd.Series({"IG_A": 0.0, "HY_B": 100_000.0, "EM_C": 0.0})
    trades_out = compute_trades(current_pos=current, target_pos=targets, cfg=cfg)
    print("\nTrades to execute:")
    print(trades_out["trades"])
    print(f"Total cost: {trades_out['total_cost$']:.2f}")