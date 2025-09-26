# engines/credit/hy_vs_ig.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

TRADING_DAYS = 252

# =============================================================================
# Config
# =============================================================================

@dataclass
class StrategyConfig:
    z_lookback: int = 126           # days for z-score window
    clip_z: float = 4.0
    rebalance: str = "W-FRI"        # daily, weekly, monthly
    notional: float = 1_000_000.0   # gross notional per leg
    tc_bps: float = 2.0             # round-trip transaction cost (bps)

@dataclass
class BacktestConfig:
    nav0: float = 1_000_000.0

# =============================================================================
# Utilities
# =============================================================================

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D","DAILY"): return pd.Series(True, index=idx)
    if f.startswith("W"):  return pd.Series(1,index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1,index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _zscore(s: pd.Series, lb: int, clip: float) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    z = (s - mu) / (sd + 1e-12)
    return z.clip(-clip, clip)

# =============================================================================
# Signal
# =============================================================================

def build_signal(
    hy_spread: pd.Series,   # HY spread (bps)
    ig_spread: pd.Series,   # IG spread (bps)
    cfg: StrategyConfig = StrategyConfig()
) -> Dict[str, pd.Series | pd.DataFrame]:
    hy = hy_spread.sort_index().astype(float)
    ig = ig_spread.sort_index().astype(float)
    idx = hy.index.intersection(ig.index) # type: ignore

    ratio = (hy/ig).reindex(idx).rename("ratio")
    z = _zscore(ratio, cfg.z_lookback, cfg.clip_z).rename("zscore")

    # Trade score: positive → HY rich (short HY, long IG); negative → HY cheap
    score = z.rename("score")

    features = pd.concat([hy.rename("hy_bps"), ig.rename("ig_bps"), ratio, z, score], axis=1)
    return {"features": features, "score": score, "diag": features.iloc[[-1]]}

# =============================================================================
# Backtest
# =============================================================================

def backtest(
    hy_spread: pd.Series,
    ig_spread: pd.Series,
    cfg: StrategyConfig = StrategyConfig(),
    bt: BacktestConfig = BacktestConfig()
) -> Dict[str, pd.DataFrame]:
    hy = hy_spread.sort_index().astype(float)
    ig = ig_spread.sort_index().astype(float)
    idx = hy.index.intersection(ig.index) # type: ignore

    sig = build_signal(hy, ig, cfg)
    score = sig["score"]

    rb = _rb_mask(idx, cfg.rebalance) # type: ignore

    nav = bt.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    pnl = pd.Series(0.0, index=idx)
    costs = pd.Series(0.0, index=idx)

    last_pos = 0.0
    for t in idx:
        if rb.loc[t]:
            # target = +/- notional depending on zscore
            sc = float(score.loc[t])
            target = np.sign(sc) * cfg.notional
            traded = abs(target - last_pos)
            costs.loc[t] += traded * (cfg.tc_bps * 1e-4)
            last_pos = target

        # PnL proxy: Δ(hy/ig spread ratio) * position
        d_ratio = 0.0 if t==idx[0] else ( (hy/ig).loc[t] - (hy/ig).shift(1).loc[t] )
        pnl_t = -last_pos * d_ratio - costs.loc[t]
        pnl.loc[t] = pnl_t
        nav += pnl_t
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "nav$": nav_path,
        "pnl$": pnl,
        "costs$": costs.cumsum(),
        "score": score
    }, index=idx)

    return {"summary": summary, "signals": sig} # type: ignore

# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-02", periods=500, freq="B")

    # Synthetic spreads: IG ~ 100bps, HY ~ 400bps, with noise
    ig = pd.Series(100 + np.cumsum(rng.normal(0,0.2,len(idx))), index=idx)
    hy = pd.Series(400 + np.cumsum(rng.normal(0,0.8,len(idx))), index=idx)

    out = backtest(hy, ig)
    print(out["summary"].tail())