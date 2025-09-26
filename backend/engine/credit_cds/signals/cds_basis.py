# engines/cap_struct/cds_basis.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

TRADING_DAYS = 252

# =============================================================================
# Config
# =============================================================================

@dataclass
class BasisConfig:
    tenor_years: float = 5.0
    recovery_rate: float = 0.4
    basis_z_lookback: int = 252
    clip_z: float = 4.0
    rebalance: str = "W-FRI"
    notional: float = 1_000_000.0       # base bond/CDS notional
    tc_bps: float = 2.0                 # transaction cost bps on notional

@dataclass
class BacktestConfig:
    nav0: float = 1_000_000.0

# =============================================================================
# Utilities
# =============================================================================

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

# =============================================================================
# Signal construction
# =============================================================================

def build_basis_signal(
    *,
    cds_spread_bps: pd.Series,       # observed CDS spread (bps)
    bond_yield: pd.Series,           # bond yield (annualized %)
    gov_yield: pd.Series,            # risk-free benchmark yield (%)
    tenor_years: float = 5.0,
    cfg: BasisConfig = BasisConfig(),
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Bond spread ≈ (bond_yield - gov_yield) * 10000 (bps).
    CDS basis = bond spread − CDS spread.
    """
    cds = cds_spread_bps.sort_index().astype(float)
    by = bond_yield.sort_index().astype(float)
    gy = gov_yield.sort_index().astype(float)
    idx = cds.index.intersection(by.index).intersection(gy.index) # type: ignore

    bond_spread = ((by - gy) * 10000).reindex(idx)
    cds = cds.reindex(idx)
    basis = (bond_spread - cds).rename("basis_bps")

    z_basis = _zscore(basis, cfg.basis_z_lookback, cfg.clip_z).rename("z_basis")

    features = pd.concat([bond_spread.rename("bond_spread_bps"), cds.rename("cds_bps"), basis, z_basis], axis=1)
    score = z_basis.rename("score")

    return {"features": features, "score": score, "diag": features.iloc[[-1]]}

# =============================================================================
# Backtest
# =============================================================================

def backtest_basis(
    *,
    cds_spread_bps: pd.Series,
    bond_yield: pd.Series,
    gov_yield: pd.Series,
    cfg: BasisConfig = BasisConfig(),
    bt: BacktestConfig = BacktestConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Trade logic (simplified):
      - If CDS cheap (basis > 0), long CDS protection + long bond (hedged).
      - If CDS rich (basis < 0), short CDS protection + short bond.
    PnL ≈ carry + MTM from spread moves (bond vs CDS).
    """
    cds = cds_spread_bps.sort_index().astype(float)
    by = bond_yield.sort_index().astype(float)
    gy = gov_yield.sort_index().astype(float)
    idx = cds.index.intersection(by.index).intersection(gy.index) # type: ignore

    # Build signal
    sig = build_basis_signal(cds_spread_bps=cds, bond_yield=by, gov_yield=gy, tenor_years=cfg.tenor_years, cfg=cfg)
    basis = sig["features"]["basis_bps"]

    # Returns
    rb = _rb_mask(idx, cfg.rebalance) # type: ignore
    nav = bt.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    pnl = pd.Series(0.0, index=idx)
    costs = pd.Series(0.0, index=idx)

    last_pos = 0.0
    for t in idx:
        if rb.loc[t]:
            sc = float(np.sign(basis.loc[t]))
            target = sc * cfg.notional
            traded = abs(target - last_pos)
            costs.loc[t] = traded * (cfg.tc_bps * 1e-4)
            last_pos = target

        # daily pnl ≈ -pos * Δbasis (bps * notional / 10,000)
        d_basis = 0.0 if t == idx[0] else basis.loc[t] - basis.shift(1).loc[t]
        pnl_t = -last_pos * (d_basis / 10_000.0) - costs.loc[t]
        pnl.loc[t] = pnl_t
        nav += pnl_t
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "nav$": nav_path,
        "pnl$": pnl,
        "costs$": costs.cumsum(),
        "basis_bps": basis,
    }, index=idx)

    return {"summary": summary, "signals": sig} # type: ignore

# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-02", periods=500, freq="B")

    # Synthetic bond yields and gov yields
    gy = pd.Series(0.02 + 0.00001*np.arange(len(idx)), index=idx)  # slowly rising gov yield
    by = gy + 0.015 + 0.002*rng.standard_normal(len(idx))          # bond yield
    cds = pd.Series(120 + np.cumsum(rng.normal(0,1,len(idx))), index=idx)

    out = backtest_basis(cds_spread_bps=cds, bond_yield=by, gov_yield=gy)
    print(out["summary"].tail())