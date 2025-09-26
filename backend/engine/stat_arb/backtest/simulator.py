# engines/stat_arb/backtest/simulator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Callable

from .pnl import (
    estimate_spread,
    compute_pair_pnl,
    compute_portfolio_pnl,
    PairSpec,
)

BetaMethod = Literal["rolling_ols", "expanding_ols", "static_ols"]

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class PairConfig:
    y: str                        # dependent leg ticker
    x: str                        # hedge leg ticker
    beta_method: BetaMethod = "rolling_ols"
    beta_lookback: int = 120
    z_lookback: int = 60
    entry_z: float = 1.0          # enter +1u when z < -entry, -1u when z > +entry
    exit_z: float = 0.25          # flat when |z| < exit (hysteresis)
    max_units: float = 1.0        # cap absolute units per pair
    stop_z: Optional[float] = 3.0 # optional hard stop (flat if |z| >= stop)
    risk_parity: bool = True      # scale units by inverse spread vol
    unit_risk_target: float = 1.0 # risk-normalized unit target (arbitrary)
    # costs / carry (bps per annum or per notional)
    fee_bps: float = 0.3
    slippage_bps: float = 1.5
    borrow_bps_y: float = 50.0
    borrow_bps_x: float = 50.0
    div_yield_bps_y: float = 150.0
    div_yield_bps_x: float = 150.0

@dataclass
class SimConfig:
    rebalance: str = "D"          # 'D','W-FRI','M','Q','BMS'
    seed: int = 7                 # for any stochastic tie-breakers (unused now)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> pd.Series:
    if freq.upper() in ("D", "DAY", "DAILY"):
        return pd.Series(True, index=dates)
    resamp = pd.Series(1, index=dates).resample(freq).last().reindex(dates).fillna(0)
    return resamp.astype(bool)

def _zscore(s: pd.Series, lb: int) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    return (s - mu) / (sd + 1e-12)

def _risk_scale(units_raw: float, spread_vol: float, unit_risk_target: float) -> float:
    """
    Scale nominal units so that units * spread_vol ≈ unit_risk_target (simple risk parity).
    """
    if spread_vol <= 0:
        return 0.0
    k = unit_risk_target / spread_vol
    return float(units_raw * k)

def _hysteresis_position(z: float, prev_units: float, cfg: PairConfig) -> float:
    """
    3-state policy with hysteresis:
      - Enter +1 when z <= -entry_z
      - Enter -1 when z >= +entry_z
      - Exit to 0 when |z| < exit_z
      - Otherwise keep previous units
    """
    if cfg.stop_z is not None and abs(z) >= cfg.stop_z:
        return 0.0
    if z <= -cfg.entry_z:
        return +1.0
    if z >= +cfg.entry_z:
        return -1.0
    if abs(z) < cfg.exit_z:
        return 0.0
    return prev_units

def _to_shares_from_units(units: pd.Series, beta: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert spread units to per-leg shares assuming:
      +1 unit = +1 share Y and -β shares X.
    """
    u = units.squeeze().astype(float).reindex(beta.index).fillna(0.0) # type: ignore
    y = u.to_frame("shares_y")
    x = (-u * beta).to_frame("shares_x")
    return y, x

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def simulate_pairs(
    prices: pd.DataFrame,
    pairs: List[PairConfig],
    sim: SimConfig = SimConfig(),
) -> Dict[str, object]:
    """
    Run a multi-pair stat-arb backtest on a price DataFrame (date x ticker).
    Returns:
      {
        'by_pair': Dict[name -> {'units','beta','spread','z','shares_y','shares_x','trades_y','trades_x','summary'}],
        'portfolio': DataFrame (aggregated pnl),
      }
    """
    px = prices.sort_index()
    dates = px.index
    rb_mask = _rebalance_mask(dates, sim.rebalance) # type: ignore

    by_pair: Dict[str, Dict[str, pd.DataFrame | pd.Series]] = {}

    for cfg in pairs:
        y = px[cfg.y].astype(float).copy()
        x = px[cfg.x].astype(float).copy()
        idx = dates.intersection(y.index).intersection(x.index)
        y = y.reindex(idx); x = x.reindex(idx)

        # Spread & rolling beta
        spread, beta = estimate_spread(y, x, beta=None, beta_method=cfg.beta_method, lookback=cfg.beta_lookback)

        # Z-score of spread
        z = _zscore(spread, cfg.z_lookback)

        # Rebalance loop → units series
        units = pd.Series(0.0, index=idx)
        last_u = 0.0
        for t in idx:
            if not rb_mask.loc[t]:
                units.loc[t] = last_u
                continue

            zt = float(z.loc[t]) if not np.isnan(z.loc[t]) else 0.0
            u_nominal = _hysteresis_position(zt, last_u, cfg)

            if cfg.risk_parity:
                # Use rolling spread vol (same window as z) for scaling
                spr_vol = spread.loc[:t].pct_change().rolling(cfg.z_lookback).std().iloc[-1]
                u = _risk_scale(u_nominal, float(spr_vol if spr_vol == spr_vol else 0.0), cfg.unit_risk_target)
            else:
                u = u_nominal

            u = float(np.clip(u, -cfg.max_units, cfg.max_units))
            units.loc[t] = u
            last_u = u

        # Convert units → shares per leg; trades = diffs
        shares_y, shares_x = _to_shares_from_units(units.to_frame("u"), beta) # type: ignore
        trades_y = shares_y.diff().fillna(shares_y.iloc[0])
        trades_x = shares_x.diff().fillna(shares_x.iloc[0])

        # Compute P&L for this pair
        summary = compute_pair_pnl(
            # reuse costs/carry directly from cfg via PairInputs inside compute_pair_pnl
            inputs=None  # type: ignore
        )  # placeholder so mypy quiet; we’ll build a PairSpec below and use compute_portfolio_pnl

        # Store artifacts (we’ll actually aggregate with compute_portfolio_pnl using PairSpec)
        by_pair_key = f"{cfg.y}/{cfg.x}"
        by_pair[by_pair_key] = {
            "units": units.to_frame("units"),
            "beta": beta.to_frame("beta"),
            "spread": spread.to_frame("spread"),
            "z": z.to_frame("z"),
            "shares_y": shares_y,
            "shares_x": shares_x,
            "trades_y": trades_y,
            "trades_x": trades_x,
            # 'summary' will be filled from portfolio aggregation below
        }

    # Build PairSpec dict and aggregate portfolio PnL
    pair_specs: Dict[str, PairSpec] = {}
    for cfg in pairs:
        key = f"{cfg.y}/{cfg.x}"
        art = by_pair[key]
        pair_specs[key] = PairSpec(
            price_y=px[cfg.y],
            price_x=px[cfg.x],
            beta_method=cfg.beta_method,
            beta_lookback=cfg.beta_lookback,
            spread_units=art["units"], # type: ignore
            trades_units=art["units"].diff().fillna(art["units"].iloc[0]), # type: ignore
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
            borrow_bps_y=cfg.borrow_bps_y,
            borrow_bps_x=cfg.borrow_bps_x,
            div_yield_bps_y=cfg.div_yield_bps_y,
            div_yield_bps_x=cfg.div_yield_bps_x,
        )

    res = compute_portfolio_pnl(pair_specs)

    # attach per-pair summaries back into artifacts
    for name, s in res["by_pair"].items():
        by_pair[name]["summary"] = s # type: ignore

    return {
        "by_pair": by_pair,
        "portfolio": res["summary"],
    }

# -----------------------------------------------------------------------------
# Quick example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic two-pair demo
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    rng = np.random.default_rng(42)

    def rw(mu, sig):
        x = np.zeros(len(idx))
        x[0] = 100.0
        for t in range(1, len(idx)):
            x[t] = x[t-1] * (1 + mu + sig * rng.standard_normal())
        return pd.Series(x, index=idx)

    prices = pd.DataFrame({
        "AAA": rw(0.0002, 0.015),
        "BBB": rw(0.00018, 0.016),
        "CCC": rw(0.00025, 0.017),
        "DDD": rw(0.00015, 0.014),
    })

    pairs = [
        PairConfig(y="AAA", x="BBB", entry_z=1.2, exit_z=0.25, stop_z=3.0, max_units=1.0),
        PairConfig(y="CCC", x="DDD", entry_z=1.0, exit_z=0.20, stop_z=2.5, max_units=1.5),
    ]

    out = simulate_pairs(prices, pairs, SimConfig(rebalance="W-FRI"))
    print("Portfolio tail:\n", out["portfolio"].tail()) # type: ignore