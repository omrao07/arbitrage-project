# engines/options/signals/var_risk_premium.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Literal

TRADING_DAYS = 252

# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

SignalMode = Literal["vix_only", "iv_var_series", "combo"]
SideMode = Literal["carry_short", "timing_long_short"]

@dataclass
class VRPConfig:
    # Inputs / construction
    realized_window: int = 21           # trading days (~1 month)
    z_lookback: int = 252               # 1y for z-scores
    clip_z: float = 4.0

    # Signal mixing
    mode: SignalMode = "vix_only"       # use VIX 30d proxy or an IV variance series
    w_vix: float = 1.0
    w_iv: float = 1.0

    # Trading logic
    side: SideMode = "carry_short"      # "carry_short": sell var when VRP>0; "timing_long_short": sign(z)
    unit_gross: float = 1.0             # position units (scaled later to vega/var notional)
    cap: float = 1.0                    # max |position|
    rebal_freq: str = "W-FRI"           # 'D','W-FRI','M'
    tc_bps: float = 5.0                 # bps on change in variance-notional
    var_notional_per_unit: float = 1_000_000.0  # $ variance swap notional per |1.0| position

@dataclass
class BacktestConfig:
    nav0: float = 1_000_000.0
    # Mapping daily return to var swap P&L:
    # Daily P&L ≈ pos_{t-1} * (K_var/252 - r_t^2) * var_notional
    # where K_var is the entry variance strike (annualized) held between rebalances.
    hold_days: int = 21                  # hold 1m then refresh on rebalance
    slippage_bps: float = 1.0            # extra bps on notional when rolling
    use_log_returns: bool = True

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D","DAILY"):
        return pd.Series(True, index=idx)
    if f.startswith("W"):
        return pd.Series(1, index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1, index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _zscore(s: pd.Series, lb: int, clip: float) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    z = (s - mu) / (sd + 1e-12)
    return z.clip(-clip, clip)

# ---------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------

def realized_variance(underlying: pd.Series, window: int = 21, use_log: bool = True) -> pd.Series:
    """
    Annualized realized variance over a rolling window (decimal^2).
    r_t are daily returns; RV_t = 252 * mean(r^2) over last 'window' days.
    """
    px = underlying.astype(float).sort_index()
    r = (np.log(px).diff() if use_log else px.pct_change()).dropna() # type: ignore
    rv = (r.pow(2).rolling(window).mean() * TRADING_DAYS).rename(f"rv_{window}d")
    return rv

def implied_variance_from_vix(vix_30d: pd.Series) -> pd.Series:
    """
    VIX is quoted as % annualized volatility for ~30d.
    Convert to annualized variance: IV^2 = (VIX/100)^2.
    """
    v = (vix_30d.astype(float) / 100.0)
    return (v * v).rename("iv_var_vix30d")

def build_vrp_series(
    *,
    underlying: pd.Series,
    vix_30d: Optional[pd.Series] = None,   # optional if you pass iv_var_series
    iv_var_series: Optional[pd.Series] = None,  # own series of annualized var strikes (same tenor)
    cfg: VRPConfig = VRPConfig(),
) -> Dict[str, pd.Series]:
    """
    Returns:
      {
        'iv_var': Series (annualized variance),
        'rv': Series (annualized realized variance),
        'vrp': Series (iv_var - rv),
        'vrp_z': Series (z-score of VRP)
      }
    """
    rv = realized_variance(underlying, window=cfg.realized_window)

    iv_vix = implied_variance_from_vix(vix_30d).reindex(rv.index).ffill() if vix_30d is not None else None
    iv_inp = iv_var_series.reindex(rv.index).ffill() if iv_var_series is not None else None

    if cfg.mode == "vix_only":
        if iv_vix is None:
            raise ValueError("VIX series required for mode='vix_only'")
        iv_var = iv_vix.rename("iv_var")
    elif cfg.mode == "iv_var_series":
        if iv_inp is None:
            raise ValueError("iv_var_series required for mode='iv_var_series'")
        iv_var = iv_inp.rename("iv_var")
    else:
        if iv_vix is None or iv_inp is None:
            raise ValueError("Both vix_30d and iv_var_series required for mode='combo'")
        iv_var = (cfg.w_vix * iv_vix + cfg.w_iv * iv_inp) / (cfg.w_vix + cfg.w_iv)

    vrp = (iv_var - rv).rename("vrp")
    vrp_z = _zscore(vrp, cfg.z_lookback, cfg.clip_z).rename("vrp_z")
    return {"iv_var": iv_var, "rv": rv, "vrp": vrp, "vrp_z": vrp_z}

# ---------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------

def build_vrp_signal(
    *,
    underlying: pd.Series,
    vix_30d: Optional[pd.Series] = None,
    iv_var_series: Optional[pd.Series] = None,
    cfg: VRPConfig = VRPConfig(),
) -> Dict[str, object]:
    """
    Returns:
      {
        'features': DataFrame[iv_var, rv, vrp, vrp_z],
        'position': Series (signed, unitless),
        'diag': DataFrame (latest snapshot)
      }
    """
    parts = build_vrp_series(underlying=underlying, vix_30d=vix_30d, iv_var_series=iv_var_series, cfg=cfg)
    iv_var, rv, vrp, z = parts["iv_var"], parts["rv"], parts["vrp"], parts["vrp_z"]
    idx = vrp.index

    if cfg.side == "carry_short":
        # Default: sell variance in proportion to positive VRP; clamp to [-cap, cap]
        pos = (vrp / (vrp.abs().rolling(60).quantile(0.9) + 1e-12)).clip(-cfg.cap, cfg.cap)
        pos = pos.fillna(0.0).rename("position")
    else:
        # Timing: go with z-score sign
        pos = z.clip(-cfg.cap, cfg.cap).rename("position")

    # Normalize to unit_gross (by max-abs in last 60d to avoid shocks)
    norm = pos.abs().rolling(60).max().replace(0, np.nan)
    pos = (cfg.unit_gross * pos / norm).fillna(0.0)

    feat = pd.concat([iv_var, rv, vrp, z], axis=1)
    diag = pd.DataFrame({
        "iv_var": [iv_var.iloc[-1]],
        "rv": [rv.iloc[-1]],
        "vrp": [vrp.iloc[-1]],
        "vrp_z": [z.iloc[-1]],
        "position": [pos.iloc[-1]],
    }, index=[idx[-1]]) if len(idx) else pd.DataFrame()

    return {"features": feat, "position": pos, "diag": diag}

# ---------------------------------------------------------------------
# Proxy backtest (variance-swap approximation)
# ---------------------------------------------------------------------

def backtest_vrp(
    *,
    underlying: pd.Series,
    vix_30d: Optional[pd.Series] = None,
    iv_var_series: Optional[pd.Series] = None,
    cfg: VRPConfig = VRPConfig(),
    bt: BacktestConfig = BacktestConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    We approximate the P&L of a position in rolling 1-month variance swaps:
      Daily P&L ≈ position_{t-1} * (K_var/252 - r_t^2) * var_notional_per_unit
    where K_var is the variance strike captured at last rebalance, held until roll/hold_days.

    Costs:
      - tc_bps on change in |position| * var_notional_per_unit at rebalances
      - slippage_bps also applied when strike is refreshed (roll)
    """
    px = underlying.astype(float).sort_index()
    r = (np.log(px).diff() if bt.use_log_returns else px.pct_change()).fillna(0.0) # type: ignore
    idx = px.index

    out = build_vrp_signal(underlying=underlying, vix_30d=vix_30d, iv_var_series=iv_var_series, cfg=cfg)
    pos_full = out["position"].reindex(idx).ffill().fillna(0.0) # type: ignore

    parts = build_vrp_series(underlying=underlying, vix_30d=vix_30d, iv_var_series=iv_var_series, cfg=cfg)
    iv_var = parts["iv_var"].reindex(idx).ffill()

    rb = _rb_mask(idx, cfg.rebal_freq) # type: ignore

    pos = pd.Series(0.0, index=idx)
    strike = pd.Series(np.nan, index=idx)  # variance strike K_var to accrue against
    pnl = pd.Series(0.0, index=idx)
    costs = pd.Series(0.0, index=idx)

    last_pos = 0.0
    days_held = 0

    for t in idx:
        # Rebalance position
        if rb.loc[t]:
            target = float(pos_full.loc[t])
            traded = abs(target - last_pos)
            costs.loc[t] += traded * cfg.var_notional_per_unit * (cfg.tc_bps * 1e-4)
            last_pos = target
            days_held = 0  # refresh timer
            strike.loc[t] = float(iv_var.loc[t])
        else:
            strike.loc[t] = strike.loc[t-1] if t != idx[0] else float(iv_var.loc[t])

        # Roll strike after hold_days as a simple refresh (apply slippage once)
        if days_held >= bt.hold_days:
            # refresh to today's strike
            prev = float(strike.loc[t])
            newk = float(iv_var.loc[t])
            if not np.isnan(prev):
                # small roll slippage
                costs.loc[t] += abs(last_pos) * cfg.var_notional_per_unit * (bt.slippage_bps * 1e-4)
            strike.loc[t] = newk
            days_held = 0

        pos.loc[t] = last_pos

        if t != idx[0]:
            # Daily variance swap accrual
            kt = float(strike.loc[t-1])
            pnl.loc[t] = last_pos * (kt / TRADING_DAYS - r.loc[t] ** 2) * cfg.var_notional_per_unit
            days_held += 1

    nav = bt.nav0 + (pnl - costs).cumsum()
    summary = pd.DataFrame({
        "position": pos,
        "strike_var_ann": strike,
        "ret": r,
        "pnl$": pnl,
        "costs$": costs,
        "nav$": nav,
        "ret_net": nav.pct_change().fillna(0.0)
    })
    return {"summary": summary, "features": out["features"], "z": parts["vrp_z"]} # type: ignore

# ---------------------------------------------------------------------
# Optional: build IV variance series from an option strip (Carr–Madan proxy)
# ---------------------------------------------------------------------

def iv_variance_from_option_strip(
    *,
    mid_put: pd.Series, mid_call: pd.Series, strikes: pd.Series,
    forward: pd.Series, expiry_years: pd.Series
) -> pd.Series:
    """
    Very rough discrete approximation to model-free variance:
      Var_T ≈ (2/T) * [ Σ (ΔK/K^2) * OTM_option(K) ]  (ignoring discounting/divs)
    Inputs should be aligned by date; this is a helper if you have per-date
    option grids. In practice, use your proper surface & integration.
    """
    # Align by date; expect each Series to be MultiIndex (date, strike) or similar.
    # For simplicity we assume they’re per-date aligned DataFrames externally
    raise NotImplementedError("Provide your own surface-based variance integration for production.")
    

# ---------------------------------------------------------------------
# Example (synthetic)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    idx = pd.date_range("2022-01-03", periods=600, freq="B")
    rng = np.random.default_rng(3)

    # Synthetic underlying with clustered vol
    vol_state = 0.012 + 0.008*(rng.standard_normal(len(idx))>1.1)
    r = 0.0002 + vol_state * rng.standard_normal(len(idx))
    px = pd.Series(4000*np.exp(np.cumsum(r)), index=idx, name="SPX")

    # Synthetic "VIX" that tracks realized vol + premium
    rv = realized_variance(px, 21)
    vix = (np.sqrt(rv.clip(0.0001)) * 100.0 + 4.0 + 2.0*np.sin(np.linspace(0,6,len(idx))))
    vix = pd.Series(vix, index=idx, name="VIX").clip(8, 80)

    cfg = VRPConfig(mode="vix_only", side="carry_short", rebal_freq="W-FRI",
                    unit_gross=1.0, cap=1.0, tc_bps=5.0, var_notional_per_unit=1_000_000)
    bt = BacktestConfig(nav0=1_000_000, hold_days=21, slippage_bps=1.0, use_log_returns=True)

    out = backtest_vrp(underlying=px, vix_30d=vix, cfg=cfg, bt=bt)
    print(out["summary"].tail())