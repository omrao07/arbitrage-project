# engines/core/simulator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class SimConfig:
    nav0: float = 1_000_000.0       # initial NAV
    tc_bps: float = 2.0             # transaction cost (bps of traded notional)
    rebalance: str = "M"            # "D"=daily, "W-FRI"=weekly, "M"=monthly
    min_notional: float = 5_000.0   # drop dust orders < this USD
    allow_short: bool = True
    leverage_cap: float = 5.0       # max gross exposure / NAV
    vol_target: Optional[float] = None  # target annual vol (None = off)
    lookback_vol: int = 60

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D","DAILY"): return pd.Series(True, index=idx)
    if f.startswith("W"):  return pd.Series(1, index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1, index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _inv_vol_scale(weights: pd.Series, rets: pd.DataFrame, lb: int, tgt: float) -> pd.Series:
    if tgt is None or weights.abs().sum() == 0: return weights
    vol = rets.rolling(lb).std().iloc[-1].replace(0,np.nan)
    scaled = weights * (1.0 / vol.reindex(weights.index))
    scaled = scaled.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if scaled.abs().sum() > 0:
        scaled = scaled / scaled.abs().sum()
        scaled *= tgt / (np.sqrt(252) * np.sqrt((scaled @ vol @ scaled)))
    return scaled

# ---------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------

def simulate_backtest(
    *,
    prices: pd.DataFrame,                       # date × asset
    signal_func: Callable[[pd.DataFrame], pd.Series],   # returns weights at date t (per asset)
    cfg: SimConfig = SimConfig(),
) -> Dict[str,pd.DataFrame]:
    """
    Generic portfolio simulator:
      - Rebalance at freq
      - signal_func(prices_to_t) → weights
      - Allocates NAV to assets
      - Applies TC (bps of traded notional)
    Returns:
      {"summary": DataFrame, "weights": DataFrame, "positions": DataFrame}
    """
    idx = prices.index
    rb = _rb_mask(idx, cfg.rebalance) # type: ignore
    rets = prices.pct_change().fillna(0.0)

    nav = cfg.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    pos_usd = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    costs = pd.Series(0.0, index=idx)

    last_w = pd.Series(0.0, index=prices.columns, dtype=float)

    for t in idx:
        # 1) Signals + weights
        if rb.loc[t]:
            sig_w = signal_func(prices.loc[:t])
            sig_w = sig_w.reindex(prices.columns).fillna(0.0)
            # Vol scaling if requested
            if cfg.vol_target is not None:
                sig_w = _inv_vol_scale(sig_w, rets.loc[:t], cfg.lookback_vol, cfg.vol_target)
            # Gross leverage cap
            if sig_w.abs().sum() > cfg.leverage_cap:
                sig_w *= cfg.leverage_cap / sig_w.abs().sum()
            # Normalize
            if sig_w.abs().sum() > 0:
                sig_w /= sig_w.abs().sum()
            w = sig_w
        else:
            w = last_w

        weights.loc[t] = w
        pos_usd.loc[t] = w * nav

        # 2) PnL from returns
        pnl_t = float((last_w * rets.loc[t]).sum() * nav)
        nav = nav + pnl_t

        # 3) Trading costs
        turn = (w - last_w).abs().sum()
        cost_t = turn * nav * cfg.tc_bps * 1e-4 if rb.loc[t] else 0.0
        nav -= cost_t
        costs.loc[t] = cost_t

        nav_path.loc[t] = nav
        last_w = w

    summary = pd.DataFrame({
        "nav": nav_path,
        "costs$": costs,
        "ret_net": nav_path.pct_change().fillna(0.0)
    })
    return {"summary": summary, "weights": weights, "positions": pos_usd}

# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Fake asset prices
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    px = pd.DataFrame({
        "AAPL": 150*np.exp(np.cumsum(0.0003+0.02*rng.standard_normal(len(idx)))),
        "MSFT": 200*np.exp(np.cumsum(0.0002+0.018*rng.standard_normal(len(idx)))),
        "SPY":  300*np.exp(np.cumsum(0.00025+0.015*rng.standard_normal(len(idx)))),
    }, index=idx)

    def sig_func(hist: pd.DataFrame) -> pd.Series:
        # Simple momentum: long top asset
        rets = hist.pct_change(60).iloc[-1]
        best = rets.idxmax()
        w = pd.Series(0.0, index=hist.columns)
        w[best] = 1.0
        return w

    out = simulate_backtest(prices=px, signal_func=sig_func, cfg=SimConfig(vol_target=None))
    print(out["summary"].tail())