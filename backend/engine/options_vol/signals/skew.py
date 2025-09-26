# engines/options/signals/skew.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

TRADING_DAYS = 252

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SignalMode = Literal["iv_only", "realized_only", "combo"]
SideMode = Literal["mean_revert", "trend"]

@dataclass
class SkewConfig:
    # IV inputs
    use_butterfly: bool = True                 # include 25Δ butterfly slope in signal
    tenor_days: int = 30                       # intended IV tenor for alignment (informational)
    z_lookback: int = 252                      # z-score window for features
    clip_z: float = 4.0

    # Combine realized skew with IV skew
    mode: SignalMode = "combo"
    w_rr: float = 0.6                           # weight on RR25 z-score
    w_bf: float = 0.2                           # weight on BF25 z-score
    w_realized: float = 0.2                     # weight on realized skew z-score
    realized_lb: int = 63                       # lookback for realized skew (3 months)
    realized_annualize: bool = False            # realized skew is shape param; leave raw by default

    # Trading logic
    side: SideMode = "mean_revert"              # mean-revert skew or trend-follow
    unit_gross: float = 1.0                     # target gross risk budget for RR position
    cap: float = 1.0                            # cap abs(position)
    rebal_freq: str = "W-FRI"                   # 'D', 'W-FRI', 'M'
    tc_bps: float = 5.0                         # bps of traded notional (proxy)
    vega_notional_per_unit: float = 1_000_000.0 # $ vega notional per |1.0| position unit

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D","DAILY"): return pd.Series(True, index=idx)
    if f.startswith("W"):  return pd.Series(1, index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1, index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _zscore(s: pd.Series, lb: int, clip: float) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    z = (s - mu) / (sd + 1e-12)
    return z.clip(-clip, clip)

# ---------------------------------------------------------------------
# Core feature builders
# ---------------------------------------------------------------------

def build_rr_bf_from_iv(
    *,
    iv_put_25d: pd.Series,     # 25Δ put IV (decimal, e.g., 0.24)
    iv_atm: pd.Series,         # ATM IV (e.g., 50Δ call/put midpoint or 0.50 delta vol)
    iv_call_25d: pd.Series,    # 25Δ call IV
) -> pd.DataFrame:
    """
    Compute classic 25Δ Risk Reversal and 25Δ Butterfly:
      RR25 = IV(25Δ call) - IV(25Δ put)
      BF25 = 0.5 * [IV(25Δ call) + IV(25Δ put)] - IV(ATM)
    All inputs must be aligned in time at a common tenor (e.g., 30d).
    """
    idx = iv_atm.index
    ivp = iv_put_25d.reindex(idx).astype(float).ffill()
    ivc = iv_call_25d.reindex(idx).astype(float).ffill()
    iva = iv_atm.astype(float).ffill()

    rr = (ivc - ivp).rename("RR25")
    bf = (0.5 * (ivc + ivp) - iva).rename("BF25")
    return pd.concat([rr, bf], axis=1)

def build_realized_skew(
    *,
    underlying: pd.Series,     # price level (spot/future)
    lookback: int = 63
) -> pd.Series:
    """
    Realized skewness of daily log returns over a rolling window:
      skew = E[(r - μ)^3] / σ^3
    """
    px = underlying.sort_index().astype(float)
    r = np.log(px).diff().dropna() # type: ignore
    def _sk(s):
        if s.count() < 5: return np.nan
        x = s - s.mean()
        sd = s.std()
        if sd == 0: return np.nan
        return float((x**3).mean() / (sd**3 + 1e-12))
    return r.rolling(lookback).apply(_sk, raw=False).rename("realized_skew")

# ---------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------

def build_skew_signal(
    *,
    rr_bf: pd.DataFrame,            # columns: ['RR25','BF25'] (BF25 optional)
    realized_skew: Optional[pd.Series] = None,
    cfg: SkewConfig = SkewConfig(),
) -> Dict[str, object]:
    """
    Returns:
      {
        'features': DataFrame[RR25, BF25, realized_skew],
        'z': DataFrame[z_rr, z_bf, z_realized],
        'score': Series (combined score),
        'position': Series (desired RR exposure, signed),
        'diag': DataFrame (last snapshot row with components),
      }
    """
    df = rr_bf.copy()
    if "BF25" not in df.columns:
        df["BF25"] = np.nan

    if realized_skew is not None:
        df["realized_skew"] = realized_skew.reindex(df.index)
    else:
        df["realized_skew"] = np.nan

    # Z-scores
    z_rr = _zscore(df["RR25"], cfg.z_lookback, cfg.clip_z).rename("z_rr")
    z_bf = _zscore(df["BF25"], cfg.z_lookback, cfg.clip_z).rename("z_bf")
    z_rs = _zscore(df["realized_skew"], cfg.z_lookback, cfg.clip_z).rename("z_realized")

    # Choose mode
    if cfg.mode == "iv_only":
        score = cfg.w_rr * z_rr + (cfg.w_bf * z_bf if cfg.use_butterfly else 0.0)
    elif cfg.mode == "realized_only":
        score = z_rs
    else:
        score = cfg.w_rr * z_rr + (cfg.w_bf * z_bf if cfg.use_butterfly else 0.0) + cfg.w_realized * z_rs
    score = score.rename("score")

    # Position rule
    if cfg.side == "mean_revert":
        # Equity skew is typically negative; extremes tend to mean-revert.
        # Position opposite the score (e.g., very positive score → short RR; very negative → long RR).
        pos = (-score).clip(-cfg.cap, cfg.cap)
    else:
        # Trend-follow skew: go with the sign of the score
        pos = score.clip(-cfg.cap, cfg.cap)
    pos = (cfg.unit_gross * pos / (pos.abs().rolling(20).max().replace(0, np.nan))).fillna(0.0).rename("position")

    z = pd.concat([z_rr, z_bf, z_rs], axis=1)
    features = df[["RR25","BF25","realized_skew"]]
    diag = pd.DataFrame({
        "RR25": [df["RR25"].iloc[-1]],
        "BF25": [df["BF25"].iloc[-1]],
        "realized_skew": [df["realized_skew"].iloc[-1]],
        "z_rr": [z_rr.iloc[-1]],
        "z_bf": [z_bf.iloc[-1]],
        "z_realized": [z_rs.iloc[-1]],
        "score": [score.iloc[-1]],
        "position": [pos.iloc[-1]],
    }, index=[features.index[-1]]) if len(features) else pd.DataFrame()

    return {"features": features, "z": z, "score": score, "position": pos, "diag": diag}

# ---------------------------------------------------------------------
# Proxy backtest: delta-hedged risk-reversal
# ---------------------------------------------------------------------

def backtest_skew_rr(
    *,
    rr_bf: pd.DataFrame,                  # must include 'RR25'
    realized_skew: Optional[pd.Series] = None,
    cfg: SkewConfig = SkewConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Proxy P&L for a delta-hedged risk-reversal book:
      PnL_t ≈ position_{t-1} * (RR25_t - RR25_{t-1}) * VegaNotionalPerUnit
    - Rebalance position on cfg.rebal_freq.
    - Costs: bps * |Δposition| * VegaNotionalPerUnit.

    Note: This is a coarse proxy; a full book would price call/put legs with vega/volga/theta.
    """
    out = build_skew_signal(rr_bf=rr_bf, realized_skew=realized_skew, cfg=cfg)
    if out["position"].empty: # type: ignore
        return {"summary": pd.DataFrame(), "position": pd.Series(dtype=float), "z": pd.DataFrame()} # type: ignore

    rr = rr_bf["RR25"].reindex(out["position"].index).ffill().dropna() # type: ignore
    pos_full = out["position"].reindex(rr.index).fillna(method="ffill").fillna(0.0) # type: ignore

    idx = rr.index
    rb = _rb_mask(idx, cfg.rebal_freq) # type: ignore

    pos = pd.Series(0.0, index=idx)
    last_pos = 0.0
    costs = pd.Series(0.0, index=idx)
    pnl = pd.Series(0.0, index=idx)

    for t in idx:
        if rb.loc[t]:
            target = float(pos_full.loc[t])
            traded = abs(target - last_pos)
            # transaction cost in vega notional
            costs.loc[t] = traded * cfg.vega_notional_per_unit * (cfg.tc_bps * 1e-4)
            last_pos = target
        pos.loc[t] = last_pos

        if t != idx[0]:
            drr = rr.loc[t] - rr.shift(1).loc[t]
            pnl.loc[t] = last_pos * drr * cfg.vega_notional_per_unit

    nav = pnl.cumsum() - costs.cumsum()
    summary = pd.DataFrame({"rr": rr, "position": pos, "pnl$": pnl, "costs$": costs, "nav$": nav})
    return {"summary": summary, "position": pos, "z": out["z"]} # type: ignore

# ---------------------------------------------------------------------
# Example (synthetic)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Build synthetic daily RR/BF & underlying with realistic dynamics
    idx = pd.date_range("2023-01-02", periods=350, freq="B")
    rng = np.random.default_rng(3)

    # Synthetic IV time series
    rr = 0.00 + 0.01*np.sin(np.linspace(0, 6, len(idx))) + 0.005*rng.standard_normal(len(idx))  # RR25 (calls - puts)
    bf = -0.005 + 0.005*np.cos(np.linspace(0, 5, len(idx))) + 0.003*rng.standard_normal(len(idx))  # BF25
    atm = 0.22 + 0.02*np.sin(np.linspace(0, 2, len(idx))) + 0.01*rng.standard_normal(len(idx))
    ivc25 = atm + bf + 0.5*rr
    ivp25 = atm + bf - 0.5*rr
    rr_bf = pd.DataFrame({"RR25": ivc25 - ivp25, "BF25": 0.5*(ivc25 + ivp25) - atm}, index=idx)

    # Underlying for realized skew
    px = pd.Series(4000 * np.exp(np.cumsum(0.0002 + 0.012*rng.standard_normal(len(idx)))), index=idx)
    rskew = build_realized_skew(underlying=px, lookback=63)

    cfg = SkewConfig(mode="combo", side="mean_revert", z_lookback=126, unit_gross=1.0, cap=1.0, rebal_freq="W-FRI")
    out = backtest_skew_rr(rr_bf=rr_bf, realized_skew=rskew, cfg=cfg)
    print(out["summary"].tail())