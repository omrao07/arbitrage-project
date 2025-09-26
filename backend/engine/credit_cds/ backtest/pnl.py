# engines/credit_cds/pnl.py
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
class CDSConfig:
    notional: float = 10_000_000.0      # CDS notional
    maturity_years: float = 5.0         # maturity in years
    recovery_rate: float = 0.4          # assumed recovery rate
    accrual_freq: int = 4               # payments per year (quarterly CDS)
    upfront_points: float = 0.0         # upfront % paid at inception
    tc_bps: float = 1.0                 # round-trip transaction cost (bps of notional)

# =============================================================================
# Core PnL Calculation
# =============================================================================

def cds_pnl(
    spreads: pd.Series,            # daily CDS spread quotes in bps
    defaults: Optional[pd.Series] = None,  # 1 if default on day t, else 0
    cfg: CDSConfig = CDSConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Simulates daily CDS PnL with simple spread-DV01 mark-to-market and accrual.
    Inputs:
      spreads: daily CDS spreads (in bps).
      defaults: binary series (1 if default occurs).
    Returns:
      dict with 'summary' DataFrame containing NAV, pnl, accrual, mtm, defaults.
    """
    idx = spreads.index
    spreads = spreads.astype(float)

    if defaults is None:
        defaults = pd.Series(0, index=idx)

    # Approximate risky annuity DV01
    annuity = (1 - cfg.recovery_rate) * cfg.maturity_years
    dv01 = cfg.notional * annuity / 10_000.0  # PV of 1bp spread change

    # Initial NAV
    nav0 = cfg.notional * (cfg.upfront_points * 1e-4)
    nav_path = pd.Series(nav0, index=idx, dtype=float)

    accrual = pd.Series(0.0, index=idx)
    mtm = pd.Series(0.0, index=idx)
    pnl = pd.Series(0.0, index=idx)

    last_spread = spreads.iloc[0]

    for t in idx:
        # Daily accrual of spread premium
        accrual[t] = -(cfg.notional * (last_spread * 1e-4) / cfg.accrual_freq) / (TRADING_DAYS / cfg.accrual_freq)

        # MTM from spread move
        delta_s = spreads.loc[t] - last_spread
        mtm[t] = -dv01 * delta_s
        last_spread = spreads.loc[t]

        # Default event
        if defaults.loc[t] == 1:
            loss = -cfg.notional * (1 - cfg.recovery_rate)
            mtm[t] += loss

        pnl[t] = accrual[t] + mtm[t]
        nav_path[t] = (nav_path.shift(1).loc[t] if t != idx[0] else nav0) + pnl[t]

    summary = pd.DataFrame({
        "nav$": nav_path,
        "pnl$": pnl,
        "accrual$": accrual,
        "mtm$": mtm,
        "spread_bps": spreads,
        "default": defaults,
    }, index=idx)

    return {"summary": summary}

# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    idx = pd.date_range("2023-01-02", periods=250, freq="B")
    rng = np.random.default_rng(42)

    # synthetic CDS spreads (100 → 150bps)
    spreads = pd.Series(100 + np.cumsum(rng.standard_normal(len(idx))), index=idx)

    # no defaults
    defaults = pd.Series(0, index=idx)

    cfg = CDSConfig(notional=10_000_000, maturity_years=5, recovery_rate=0.4)
    out = cds_pnl(spreads=spreads, defaults=defaults, cfg=cfg)

    print(out["summary"].head())
    print(out["summary"].tail())