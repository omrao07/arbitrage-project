# engines/cap_struct/cds_basis/runner.py
from __future__ import annotations
import numpy as np
import pandas as pd

from engines.cap_struct import cds_basis # type: ignore


# ---------------------------------------------------------------------
# Example synthetic data loader
# Replace with Bloomberg/ICE/BAML OAS/Markit CDS index in prod
# ---------------------------------------------------------------------
def load_data(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")

    # Risk-free yield curve (gov yield ~ 2% with slow drift)
    gov_yield = pd.Series(0.02 + 0.00001*np.arange(n), index=idx, name="GovYield")

    # Corporate bond yield = gov + spread (150bps mean, noise)
    bond_yield = gov_yield + 0.015 + 0.001*rng.standard_normal(n)
    bond_yield = pd.Series(bond_yield, index=idx, name="BondYield")

    # CDS spread (bps) ~120bps with noise
    cds_spread = pd.Series(120 + np.cumsum(rng.normal(0, 0.4, n)), index=idx, name="CDS")

    return cds_spread, bond_yield, gov_yield


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run():
    cds, bond, gov = load_data()

    cfg = cds_basis.BasisConfig(
        tenor_years=5.0,
        recovery_rate=0.4,
        basis_z_lookback=126,
        clip_z=4.0,
        rebalance="W-FRI",
        notional=1_000_000.0,
        tc_bps=2.0,
    )
    bt_cfg = cds_basis.BacktestConfig(nav0=1_000_000.0)

    # Build signal
    sig = cds_basis.build_basis_signal(cds_spread_bps=cds, bond_yield=bond, gov_yield=gov, cfg=cfg)
    print("Latest signal diagnostics:")
    print(sig["diag"])

    # Run backtest
    out = cds_basis.backtest(cds_spread_bps=cds, bond_yield=bond, gov_yield=gov, cfg=cfg, bt=bt_cfg)
    summary = out["summary"]

    print("\nBacktest summary (tail):")
    print(summary.tail())

    return {"signal": sig, "backtest": out}


if __name__ == "__main__":
    run()