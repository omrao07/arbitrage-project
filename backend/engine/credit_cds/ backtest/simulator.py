# engines/credit_cds/simulator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

TRADING_DAYS = 252

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    nav0: float = 1_000_000.0
    leverage_cap: float = 1.0            # target gross notional / NAV
    max_per_name: float = 0.25           # max |weight| per name (pre-norm)
    rebalance: str = "W-FRI"             # 'D', 'W-FRI', or 'M'
    tc_bps: float = 1.0                  # transaction cost (bps of traded notional)
    maturity_years: float = 5.0
    recovery_rate: float = 0.40          # fallback if no per-name map is provided
    accrual_freq: int = 4                # payments per year (for info; accrual uses dailyized premium)
    allow_short_protection: bool = True  # allow selling protection
    # Optional per-name recovery overrides
    recovery_map: Optional[Dict[str, float]] = None

# Strategy function signature:
# def signal_func(history_spreads: pd.DataFrame) -> pd.Series[weights]
StrategyFunc = Callable[[pd.DataFrame], pd.Series]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D", "DAILY"):
        return pd.Series(True, index=idx)
    if f.startswith("W"):
        return pd.Series(1, index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1, index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _annuity(maturity_years: float, recovery: float) -> float:
    # Simple risky annuity proxy; good enough for daily PnL approximation.
    return (1.0 - recovery) * maturity_years

def _cap_and_normalize(weights: pd.Series, max_per_name: float, unit_gross: float) -> pd.Series:
    w = weights.fillna(0.0).astype(float)
    if max_per_name is not None:
        w = w.clip(lower=-max_per_name, upper=max_per_name)
    gross = w.abs().sum()
    if gross > 0:
        w = w * (unit_gross / gross)
    return w

def _recovery_for(name: str, cfg: PortfolioConfig) -> float:
    if cfg.recovery_map and name in cfg.recovery_map:
        return float(cfg.recovery_map[name])
    return float(cfg.recovery_rate)

# ---------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------

def simulate_portfolio(
    *,
    spreads_bps: pd.DataFrame,                  # index: dates; columns: CDS tickers; values: spread in bps
    defaults: Optional[pd.DataFrame] = None,    # same shape as spreads; 1 on default day, else 0
    side: str = "long_protection",              # 'long_protection' or 'short_protection'
    signal_func: StrategyFunc = None,           # maps history → weights per name (any scale) # type: ignore
    cfg: PortfolioConfig = PortfolioConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Simulate a CDS portfolio with scheduled rebalances.

    Positions are held in **protection notional** (USD), signed:
      +notional → long protection (benefits from spread widening & defaults)
      -notional → short protection (opposite).

    Daily P&L per name i:
      - Accrual ≈ -sign * Notional * (Spread_i / 10,000) / 252
      - MTM     ≈ sign * DV01_i * ΔSpread_i, where DV01_i = |Notional| * Annuity / 10,000
      - Default = sign * Notional * (1 - Recovery_i) (position then goes to 0)
    """
    if signal_func is None:
        raise ValueError("signal_func is required: history DataFrame -> weight Series")

    px = spreads_bps.sort_index().astype(float)
    names = list(px.columns)
    idx = px.index

    if defaults is None:
        defaults = pd.DataFrame(0, index=idx, columns=names, dtype=int)
    else:
        defaults = defaults.reindex(index=idx, columns=names).fillna(0).astype(int)

    rb = _rb_mask(idx, cfg.rebalance) # type: ignore

    # State
    nav = float(cfg.nav0)
    nav_path = pd.Series(index=idx, dtype=float)
    pnl_total = pd.Series(0.0, index=idx)
    accrual = pd.DataFrame(0.0, index=idx, columns=names)
    mtm = pd.DataFrame(0.0, index=idx, columns=names)
    default_pnl = pd.DataFrame(0.0, index=idx, columns=names)
    costs = pd.Series(0.0, index=idx)

    # Per-name signed protection notionals (USD); + = long protection, − = short protection
    pos = pd.Series(0.0, index=names, dtype=float)
    last_spread = px.iloc[0]

    # Side sign: +1 for long protection book; -1 for short protection book by default signal direction
    side_sign = 1.0 if side == "long_protection" else -1.0

    for t in idx:
        # 1) Rebalance if due
        if rb.loc[t]:
            # Strategy weights from history up to t (inclusive)
            hist = px.loc[:t]
            w_raw = signal_func(hist).reindex(names).fillna(0.0).astype(float)
            if not cfg.allow_short_protection:
                w_raw = w_raw.clip(lower=0.0)

            # Normalize to target gross = leverage_cap
            w = _cap_and_normalize(w_raw, cfg.max_per_name, cfg.leverage_cap)

            # Convert weights → signed protection notionals
            target_pos = side_sign * w * nav  # gross target = leverage_cap * nav

            # Trade to target & apply costs
            trade = (target_pos - pos)
            traded_notional = trade.abs().sum()
            costs.loc[t] += traded_notional * (cfg.tc_bps * 1e-4)

            # Commit new positions
            pos = target_pos

        # 2) Accrual (dailyized premium)
        #    accrual_i = -sign(pos_i) * |pos_i| * spread_i / 10,000 / TRADING_DAYS
        spread_t = px.loc[t]
        accrual_t = -(np.sign(pos) * pos.abs()) * (spread_t / 10_000.0) / TRADING_DAYS
        accrual.loc[t] = accrual_t

        # 3) MTM from spread move
        #    DV01_i = |pos_i| * Annuity_i / 10,000
        #    mtm_i  = sign(pos_i) * DV01_i * ΔSpread_i
        delta_s = (spread_t - last_spread).fillna(0.0)
        dv01 = (pos.abs() * pd.Series({n: _annuity(cfg.maturity_years, _recovery_for(n, cfg)) for n in names})) / 10_000.0
        mtm_t = np.sign(pos) * dv01 * delta_s
        mtm.loc[t] = mtm_t
        last_spread = spread_t

        # 4) Default events (cash settlement; position to zero next day)
        if (defaults.loc[t] == 1).any():
            for n in names:
                if defaults.at[t, n] == 1 and pos[n] != 0.0:
                    rec = _recovery_for(n, cfg)
                    default_pnl.at[t, n] = np.sign(pos[n]) * abs(pos[n]) * (1.0 - rec)
                    pos[n] = 0.0  # tear up after settlement

        # 5) Aggregate PnL and NAV
        pnl_t = accrual_t.sum() + mtm_t.sum() + default_pnl.loc[t].sum() - costs.loc[t]
        pnl_total.loc[t] = pnl_t
        nav = nav + pnl_t
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "nav$": nav_path,
        "pnl$": pnl_total,
        "costs$": costs.cumsum()
    }, index=idx)

    components = {
        "accrual$": accrual,
        "mtm$": mtm,
        "default$": default_pnl,
        "positions_notional$": pd.DataFrame([pos], index=[idx[-1]], columns=names)
    }

    return {"summary": summary, "components": components} # type: ignore

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Build a small synthetic panel: 6 names, 500 days
    idx = pd.date_range("2023-01-02", periods=500, freq="B")
    names = ["IG_A", "IG_B", "IG_C", "HY_X", "HY_Y", "EM_Z"]
    rng = np.random.default_rng(7)

    # Simulate spreads (bps)
    base = np.array([80, 90, 100, 300, 350, 250], dtype=float)
    shocks = rng.normal(0, 1.5, (len(idx), len(names))).cumsum(axis=0)
    level = base + shocks
    spreads = pd.DataFrame(level, index=idx, columns=names).clip(lower=20)

    # Random defaults (rare)
    defaults = pd.DataFrame(0, index=idx, columns=names, dtype=int)
    for n in ["HY_X", "HY_Y"]:
        if rng.random() < 0.2:
            d = int(rng.integers(200, 480))
            defaults.iloc[d, defaults.columns.get_loc(n)] = 1 # type: ignore

    # Simple momentum strategy: overweight wideners, underweight tighteners
    def strat(history: pd.DataFrame) -> pd.Series:
        # 60-day change
        chg = history.diff(60).iloc[-1].fillna(0.0)
        # rank to [-1, 1]
        r = chg.rank() - (len(chg) + 1) / 2.0
        r = (r / r.abs().max()).fillna(0.0)
        return r

    cfg = PortfolioConfig(
        nav0=2_000_000,
        leverage_cap=1.5,
        max_per_name=0.4,
        rebalance="W-FRI",
        tc_bps=1.0,
        maturity_years=5.0,
        recovery_rate=0.4,
        allow_short_protection=True,
        recovery_map={"EM_Z": 0.35}
    )

    out = simulate_portfolio(
        spreads_bps=spreads,
        defaults=defaults,
        side="long_protection",   # or "short_protection"
        signal_func=strat,
        cfg=cfg,
    )

    print(out["summary"].tail())