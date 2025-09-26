# engines/options/var_risk_premium/runner.py
from __future__ import annotations
import pandas as pd
import numpy as np

from engines.options.signals.var_risk_premium import ( # type: ignore
    VRPConfig,
    BacktestConfig,
    backtest_vrp,
    build_vrp_signal,
    realized_variance,
    implied_variance_from_vix,
)

# ---------------------------------------------------------------------
# Example loader (replace with your actual data adapters)
# ---------------------------------------------------------------------

def load_data() -> tuple[pd.Series, pd.Series]:
    """
    Stub loader. Replace with calls to your connectors/data APIs.
    Returns:
      - underlying (e.g., SPX close)
      - vix (VIX 30d close)
    """
    idx = pd.date_range("2022-01-03", periods=500, freq="B")
    rng = np.random.default_rng(11)
    # synthetic underlying
    vol_state = 0.012 + 0.008*(rng.standard_normal(len(idx)) > 1.1)
    r = 0.0002 + vol_state * rng.standard_normal(len(idx))
    px = pd.Series(4000*np.exp(np.cumsum(r)), index=idx, name="SPX")

    # synthetic vix: sqrt(rv) + premium
    rv = realized_variance(px, 21)
    vix = (np.sqrt(rv.clip(0.0001)) * 100.0 + 4.0 + 2.0*np.sin(np.linspace(0,6,len(idx))))
    vix = pd.Series(vix, index=idx, name="VIX").clip(8, 80)
    return px, vix

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run():
    # 1. Load data
    px, vix = load_data()

    # 2. Config
    cfg = VRPConfig(
        mode="vix_only",
        side="carry_short",        # or "timing_long_short"
        rebal_freq="W-FRI",
        unit_gross=1.0,
        cap=1.0,
        tc_bps=5.0,
        var_notional_per_unit=1_000_000,
    )
    bt_cfg = BacktestConfig(
        nav0=1_000_000,
        hold_days=21,
        slippage_bps=1.0,
        use_log_returns=True,
    )

    # 3. Build signal (diag only)
    sig = build_vrp_signal(underlying=px, vix_30d=vix, cfg=cfg)
    print("Latest VRP diag snapshot:")
    print(sig["diag"])

    # 4. Run backtest
    out = backtest_vrp(underlying=px, vix_30d=vix, cfg=cfg, bt=bt_cfg)
    summary = out["summary"]

    print("\nBacktest summary (tail):")
    print(summary.tail())

    return {"signal": sig, "backtest": out}


if __name__ == "__main__":
    run()