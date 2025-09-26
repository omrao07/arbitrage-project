# engines/stat_arb/signals/dispersion.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

TRADING_DAYS = 252

# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

SignalMode = Literal["realized_corr", "implied_corr"]

@dataclass
class SignalConfig:
    mode: SignalMode = "realized_corr"
    lookback_days: int = 60                # for realized vol/corr
    iv_horizon_days: int = 30              # for “implied” metrics if IVs are used
    neutralize_beta: bool = True           # beta-neutral vs index on equity legs
    winsor_p: float = 0.01                 # winsorize vols to avoid blowups
    unit_gross: float = 1.0                # target gross for weights
    cap_per_name: float = 0.15             # max |weight| per single-name leg

@dataclass
class BacktestConfig:
    rebalance: str = "W-FRI"               # 'D','W-FRI','M'
    tc_bps_single: float = 5.0             # bps of traded notional for single names
    tc_bps_index: float = 1.0              # cheaper on index leg
    nav0: float = 1_000_000.0

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _winsorize(s: pd.Series, p: float) -> pd.Series:
    if s.dropna().empty: return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D","DAILY"): return pd.Series(True, index=idx)
    if f.startswith("W"):  return pd.Series(1, index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1, index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _beta(y: pd.Series, x: pd.Series) -> float:
    x_ = x - x.mean(); y_ = y - y.mean()
    den = (x_**2).sum()
    return float((x_*y_).sum() / den) if den != 0 else 0.0

# ---------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------

def realized_dispersion(
    *,
    idx_px: pd.Series,                     # index level
    members_px: pd.DataFrame,              # columns = tickers
    lb: int = 60,
) -> Dict[str, pd.Series | float]:
    """
    Build realized vol of index and members and an implied 'avg corr' proxy:
      σ_idx^2  ≈ Σ w_i^2 σ_i^2 + 2 Σ Σ w_i w_j ρ_ij σ_i σ_j
    For equal-weight proxy, avg corr ≈ (σ_idx^2 - Σ σ_i^2 / N^2) / ( (N-1)/N * (Σσ_i/N)^2 )
    We use a simple, stable estimator here.
    """
    idx_rets = idx_px.sort_index().pct_change().dropna()
    mem_rets = members_px.sort_index().pct_change().dropna()
    common = idx_rets.index.intersection(mem_rets.index) # type: ignore
    idx_rets, mem_rets = idx_rets.reindex(common), mem_rets.reindex(common)

    # Rolling vols (annualized) and correlation
    vol_idx = idx_rets.rolling(lb).std() * np.sqrt(TRADING_DAYS)
    vol_i = mem_rets.rolling(lb).std() * np.sqrt(TRADING_DAYS)

    # Average pairwise correlation (realized) using Ledoit–Wolf style shrinkage proxy
    # ρ_bar_t = mean of off-diagonal correlations (rolling)
    corr_t = mem_rets.rolling(lb).corr().dropna()
    # Collapse to time series of average off-diagonal corr:
    # corr_t is MultiIndex (date, name); compute blockwise mean excluding diagonal.
    dates = corr_t.index.get_level_values(0).unique()
    avg_corr = pd.Series(index=dates, dtype=float)
    for d in dates:
        C = corr_t.loc[d].copy()
        if C.shape[0] < 3:
            avg_corr.loc[d] = np.nan
            continue
        np.fill_diagonal(C.values, np.nan)
        avg_corr.loc[d] = np.nanmean(C.values)

    # Dispersion signal: member var sum vs index var scaled by correlation
    # Stable proxy: D_t = mean(σ_i^2) - σ_idx^2
    disp = vol_i.pow(2).mean(axis=1) - vol_idx.pow(2)
    return {"sigma_index": vol_idx, "sigma_members_mean": vol_i.mean(axis=1),
            "avg_corr": avg_corr.reindex(vol_idx.index), "dispersion": disp.reindex(vol_idx.index)}

def implied_dispersion(
    *,
    idx_iv: pd.Series,                     # index IV (decimal, at common tenor)
    members_iv: pd.DataFrame,              # member IVs (same tenor)
) -> Dict[str, pd.Series]:
    """
    Use at-the-money IVs as σ (per √year). Dispersion proxy: mean(σ_i^2) - σ_idx^2.
    """
    idx_iv = idx_iv.sort_index().ffill()
    mem_iv = members_iv.sort_index().ffill()
    common = idx_iv.index.intersection(mem_iv.index) # type: ignore
    idx_iv, mem_iv = idx_iv.reindex(common), mem_iv.reindex(common)

    sigma_idx = idx_iv
    sigma_mem = mem_iv.mean(axis=1)
    disp = (mem_iv.pow(2).mean(axis=1) - sigma_idx.pow(2))
    return {"sigma_index": sigma_idx, "sigma_members_mean": sigma_mem, "dispersion": disp}

# ---------------------------------------------------------------------
# Weights builder (index vs constituents)
# ---------------------------------------------------------------------

def build_weights(
    *,
    idx_px: pd.Series,
    members_px: pd.DataFrame,
    cfg: SignalConfig = SignalConfig(),
    idx_symbol: str = "INDEX",
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Returns:
      {
        'w_members': Series per name (go long/short single-name var proxy),
        'w_index': float (index leg, opposite sign),
        'signal': Series (dispersion_t),
        'diag': DataFrame (sigma_idx, sigma_members_mean, avg_corr if available)
      }
    Strategy intuition (classic dispersion):
      - If single-name variance > index variance (high dispersion), short index variance and long single-name variance:
        Long single-name straddles (or gamma) vs short index straddle.
      - Here we produce equity proxy weights (risk units). Execution layer maps to options or delta-hedged books.
    """
    # 1) Build signal
    if cfg.mode == "realized_corr":
        res = realized_dispersion(idx_px=idx_px, members_px=members_px, lb=cfg.lookback_days)
    else:
        raise NotImplementedError("For implied-corr path, call implied_dispersion() and pass its fields manually.")

    sig = res["dispersion"].ffill().dropna() # type: ignore
    if sig.empty:
        return {"w_members": pd.Series(dtype=float), "w_index": pd.Series(dtype=float),
                "signal": pd.Series(dtype=float), "diag": pd.DataFrame()}

    # 2) Single-name sizing (inverse-vol risk parity on member returns)
    rets = members_px.sort_index().pct_change().reindex(sig.index).dropna()
    vol = (rets.rolling(cfg.lookback_days).std() * np.sqrt(TRADING_DAYS)).iloc[-1].replace(0, np.nan)
    inv_vol = 1.0 / vol
    w_mem = inv_vol / inv_vol.abs().sum()
    w_mem = w_mem.clip(lower=-cfg.cap_per_name, upper=+cfg.cap_per_name)

    # 3) Beta neutralization vs index (if using stocks/futures as proxy)
    if cfg.neutralize_beta:
        betas = pd.Series({_c: _beta(rets[_c].dropna(), idx_px.pct_change().reindex(rets.index).dropna()) for _c in w_mem.index})
        # Orthogonalize weights to beta direction
        bvec = betas.fillna(0.0)
        proj = (w_mem @ bvec) / (bvec @ bvec + 1e-12) # type: ignore
        w_mem = w_mem - proj * bvec

    # Normalize gross to 1.0
    gross = w_mem.abs().sum()
    if gross > 0:
        w_mem = w_mem * (cfg.unit_gross / gross)

    # 4) Index leg weight to offset net beta exposure (or simply set to -unit_gross)
    # Here we choose index weight = - sum(member weights) (keeps net beta close to zero if betas ≈ 1).
    w_idx = -w_mem.sum()

    # Diagnostics snapshot
    diag = pd.DataFrame({
        "sigma_index": res["sigma_index"].reindex(sig.index), # type: ignore
        "sigma_members_mean": res["sigma_members_mean"].reindex(sig.index), # type: ignore
        "avg_corr": res.get("avg_corr", pd.Series(index=sig.index)),
        "dispersion": sig,
    })

    # Latest snapshot weights
    snap_date = sig.index[-1]
    w_mem_snap = w_mem.copy()
    w_idx_snap = pd.Series({idx_symbol: float(w_idx)})

    return {"w_members": w_mem_snap.sort_index(), "w_index": w_idx_snap, "signal": sig, "diag": diag}

# ---------------------------------------------------------------------
# Lightweight backtest (equity proxy)
# ---------------------------------------------------------------------

def backtest_dispersion(
    *,
    idx_px: pd.Series,
    members_px: pd.DataFrame,
    cfg: SignalConfig = SignalConfig(),
    bt: BacktestConfig = BacktestConfig(),
    idx_symbol: str = "INDEX",
) -> Dict[str, pd.DataFrame]:
    """
    Proxy backtest: we trade stocks (or futures) as stand-ins for option variance exposures:
      - Long single-name leg proportional to inv-vol weights
      - Short index leg balancing gross (or beta)
      - PnL from price changes (this is a proxy; real dispersion uses option books / gamma/theta)
    Costs applied in bps of traded notional on rebalance days.
    """
    px_i = idx_px.sort_index()
    px_m = members_px.sort_index().reindex(px_i.index).ffill()
    idx = px_i.index

    rb = _rb_mask(idx, bt.rebalance) # type: ignore
    rets_m = px_m.pct_change().fillna(0.0)
    rets_i = px_i.pct_change().fillna(0.0)

    nav = bt.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    costs = pd.Series(0.0, index=idx, dtype=float)
    w_mem_last = pd.Series(0.0, index=px_m.columns, dtype=float)
    w_idx_last = 0.0

    weights_members = pd.DataFrame(0.0, index=idx, columns=px_m.columns, dtype=float)
    weight_index = pd.Series(0.0, index=idx, dtype=float)

    for t in idx:
        if rb.loc[t]:
            out = build_weights(idx_px=px_i.loc[:t], members_px=px_m.loc[:t], cfg=cfg, idx_symbol=idx_symbol)
            w_mem = out["w_members"].reindex(px_m.columns).fillna(0.0)
            w_idx = float(-w_mem.sum())  # type: ignore # balance

            # trading costs on change
            turn_mem = (w_mem - w_mem_last).abs().sum()
            turn_idx = abs(w_idx - w_idx_last)
            cost_t = (bt.tc_bps_single * 1e-4) * turn_mem * nav + (bt.tc_bps_index * 1e-4) * turn_idx * nav
            costs.loc[t] = cost_t

            w_mem_last, w_idx_last = w_mem, w_idx
        else:
            w_mem = w_mem_last
            w_idx = w_idx_last

        weights_members.loc[t] = w_mem
        weight_index.loc[t] = w_idx

        # daily P&L (proxy)
        pnl_mem = float((w_mem * rets_m.loc[t]).sum() * nav)
        pnl_idx = float((w_idx * rets_i.loc[t]) * nav)
        nav = nav + pnl_mem + pnl_idx - costs.loc[t]
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "nav": nav_path,
        "costs$": costs,
        "ret_net": nav_path.pct_change().fillna(0.0),
    })
    return {"summary": summary, "weights_members": weights_members, "weight_index": weight_index} # type: ignore

# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic SPX vs 20 names demo
    idx = pd.date_range("2023-01-02", periods=400, freq="B")
    rng = np.random.default_rng(5)
    n = 20
    # Create correlated stock panel
    common = rng.standard_normal((len(idx), 1)) * 0.8
    idio   = rng.standard_normal((len(idx), n)) * 0.6
    rets_m = 0.0002 + 0.012*(common @ np.ones((1,n)) + idio)
    rets_m = pd.DataFrame(rets_m, index=idx, columns=[f"N{i:02d}" for i in range(n)])
    px_m = 100 * (1 + rets_m).cumprod()

    # Index is noisy average of members
    px_i = (px_m.mean(axis=1) * (1 + 0.001 * rng.standard_normal(len(idx)))).rename("SPX")

    cfg = SignalConfig(mode="realized_corr", lookback_days=60, neutralize_beta=True, unit_gross=1.0)
    bt = backtest_dispersion(idx_px=px_i, members_px=px_m, cfg=cfg, bt=BacktestConfig(rebalance="W-FRI", tc_bps_single=4.0, tc_bps_index=1.0))
    print(bt["summary"].tail())