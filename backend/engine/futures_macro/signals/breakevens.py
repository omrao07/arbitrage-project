# engines/rates/signals/breakevens.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable

TRADING_DAYS = 252

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _linear_interp_on_maturity(curve: pd.Series, want_tenors: Iterable[float]) -> pd.Series:
    """
    One-date linear interpolation on years-to-maturity.
    `curve` index should be floats (years), values are yields in decimal (e.g., 0.021).
    """
    if curve.dropna().empty:
        return pd.Series(index=list(want_tenors), dtype=float)
    x = np.array(curve.index, dtype=float)
    y = np.array(curve.values, dtype=float)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    out = {}
    for tau in want_tenors:
        if tau <= x[0]:
            out[tau] = float(y[0] + (y[1]-y[0])*(tau-x[0])/(x[1]-x[0])) if len(x) > 1 else float(y[0])
        elif tau >= x[-1]:
            out[tau] = float(y[-2] + (y[-1]-y[-2])*(tau-x[-2])/(x[-1]-x[-2])) if len(x) > 1 else float(y[-1])
        else:
            j = np.searchsorted(x, tau)
            x0, x1 = x[j-1], x[j]
            y0, y1 = y[j-1], y[j]
            out[tau] = float(y0 + (y1 - y0) * (tau - x0) / (x1 - x0))
    return pd.Series(out, dtype=float)

def _zscore(s: pd.Series, lb: int) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    return (s - mu) / (sd + 1e-12)

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)

# ------------------------------------------------------------
# Core computations
# ------------------------------------------------------------

def breakeven_panel(
    *,
    nominal_yields: pd.DataFrame,   # date x maturityYears (float columns; e.g., [2,5,10,30])
    tips_yields: pd.DataFrame,      # date x maturityYears (float columns; e.g., [5,10,30])
    tenors: Iterable[float] = (5.0, 10.0, 30.0),
) -> pd.DataFrame:
    """
    Returns DataFrame[date x tenor] of breakeven rates (nominal - real) in decimal terms.
    If requested tenor not present, linearly interpolates on the curve each day.
    """
    tenors = list(map(float, tenors))
    dates = nominal_yields.index.intersection(tips_yields.index)
    out = pd.DataFrame(index=dates, columns=tenors, dtype=float)

    # Ensure numeric columns (maturities in years)
    nom_cols = [float(c) for c in nominal_yields.columns]
    r_cols = [float(c) for c in tips_yields.columns]
    nom = nominal_yields.copy(); nom.columns = nom_cols
    real = tips_yields.copy(); real.columns = r_cols

    for dt in dates:
        nom_curve = nom.loc[dt].dropna()
        real_curve = real.loc[dt].dropna()
        if nom_curve.empty or real_curve.empty:
            continue
        nom_i = _linear_interp_on_maturity(nom_curve, tenors)
        real_i = _linear_interp_on_maturity(real_curve, tenors)
        out.loc[dt] = (nom_i - real_i).reindex(tenors)
    return out.astype(float)

def breakeven_momentum(
    be: pd.DataFrame,
    *,
    lookback_days: int = 60,
    diff_horizon_days: int = 20,
) -> pd.DataFrame:
    """
    Simple momentum: z-score of the BE change over `diff_horizon_days`, computed with a `lookback_days` window.
    Positive => breakevens rising => long BE (long TIPS vs short nominals).
    """
    d = be.diff(diff_horizon_days)
    z = d.apply(lambda s: _zscore(s, lookback_days))
    return z

def breakeven_carry_rolldown(
    be: pd.DataFrame,
    *,
    tenors: Iterable[float],
    roll_horizon_years: float = 1.0/12.0,  # 1 month
) -> pd.DataFrame:
    """
    Curve carry/rolldown proxy: for tenor τ, estimate how BE changes if the security rolls
    from τ to (τ - Δt) over `roll_horizon_years`. Approximate slope using adjacent knots.
    Positive slope at τ (BE shorter > BE longer) => positive rolldown for long BE at τ.
    """
    ten = np.array(list(map(float, tenors)))
    ten.sort()
    # Precompute slope via centered differences across columns (tenor dimension)
    # Then scale by Δt (roll horizon).
    be_cols = [float(c) for c in be.columns]
    be = be.copy(); be.columns = be_cols
    # Interpolate BEs to dense grid of tenors if needed
    use_cols = sorted(set(be_cols) | set(ten.tolist()))
    be_interp = pd.DataFrame(index=be.index, columns=use_cols, dtype=float)
    for dt, row in be.iterrows():
        series = row.dropna()
        if series.empty:
            continue
        be_interp.loc[dt] = _linear_interp_on_maturity(series, use_cols) # type: ignore

    # slope d(BE)/dτ via centered/forward/backward difference across tenors
    cols = np.array(use_cols, dtype=float)
    slopes = pd.DataFrame(index=be.index, columns=use_cols, dtype=float)
    for j, tau in enumerate(cols):
        if j == 0:
            slopes.iloc[:, j] = (be_interp.iloc[:, j+1] - be_interp.iloc[:, j]) / (cols[j+1] - cols[j])
        elif j == len(cols)-1:
            slopes.iloc[:, j] = (be_interp.iloc[:, j] - be_interp.iloc[:, j-1]) / (cols[j] - cols[j-1])
        else:
            slopes.iloc[:, j] = (be_interp.iloc[:, j+1] - be_interp.iloc[:, j-1]) / (cols[j+1] - cols[j-1])

    # rolldown ≈ - slope * Δt (moving left along the curve as time passes)
    carry = -slopes * float(roll_horizon_years)
    # Return only requested tenors (interpolated)
    out = pd.DataFrame(index=be.index, columns=ten, dtype=float)
    for dt, row in carry.iterrows():
        out.loc[dt] = _linear_interp_on_maturity(row.dropna(), ten) # type: ignore
    return out

# ------------------------------------------------------------
# Signal construction
# ------------------------------------------------------------

@dataclass
class SignalConfig:
    tenors: List[float] = None                       # type: ignore # e.g., [5, 10, 30]
    mom_lb_days: int = 60
    mom_diff_days: int = 20
    carry_roll_horizon_years: float = 1.0/12.0
    winsor_p: float = 0.01
    mom_weight: float = 0.6
    carry_weight: float = 0.4
    cap_per_tenor: float = 0.6                       # cap absolute weight per tenor
    unit_gross: float = 1.0                          # total |weights| normalization

    def __post_init__(self):
        if self.tenors is None:
            self.tenors = [5.0, 10.0, 30.0]

def build_breakeven_signal(
    *,
    nominal_yields: pd.DataFrame,     # date x maturities (years)
    tips_yields: pd.DataFrame,        # date x maturities (years)
    cfg: SignalConfig = SignalConfig(),
) -> Dict[str, object]:
    """
    Returns a dict:
      {
        'breakevens': DataFrame(date x tenor),
        'momentum_z': DataFrame(date x tenor),
        'carry': DataFrame(date x tenor),
        'weights': Series (latest date x tenor),
        'diag': DataFrame row for latest date,
      }
    Weights are *relative* across the chosen tenors, sized to unit gross.
    Positive weight = long BE (long TIPS vs short nominals).
    """
    be = breakeven_panel(nominal_yields=nominal_yields, tips_yields=tips_yields, tenors=cfg.tenors).dropna(how="all")
    if be.empty or len(be) < max(cfg.mom_lb_days, cfg.mom_diff_days) + 5:
        return {"breakevens": be, "momentum_z": pd.DataFrame(), "carry": pd.DataFrame(),
                "weights": pd.Series(dtype=float), "diag": pd.DataFrame()}

    mom = breakeven_momentum(be, lookback_days=cfg.mom_lb_days, diff_horizon_days=cfg.mom_diff_days)
    car = breakeven_carry_rolldown(be, tenors=cfg.tenors, roll_horizon_years=cfg.carry_roll_horizon_years)

    t = be.index[-1]
    mom_t = mom.loc[t].dropna()
    car_t = car.loc[t].dropna()

    # Standardize & winsorize inputs at snapshot
    m = _winsorize(mom_t, cfg.winsor_p)
    c = _winsorize(car_t, cfg.winsor_p)

    # Combine
    score = cfg.mom_weight * m + cfg.carry_weight * c
    # Normalize to unit gross and cap per tenor
    if score.abs().sum() > 0:
        w = score / (score.abs().sum())
    else:
        w = score * 0.0
    w = w.clip(lower=-cfg.cap_per_tenor, upper=cfg.cap_per_tenor)
    if w.abs().sum() > 0:
        w = w * (cfg.unit_gross / w.abs().sum())

    diag = pd.DataFrame({
        "breakeven": be.loc[t].reindex(w.index),
        "momentum_z": mom_t.reindex(w.index),
        "carry_roll": car_t.reindex(w.index),
        "weight": w,
    }, index=w.index)

    return {"breakevens": be, "momentum_z": mom, "carry": car, "weights": w, "diag": diag}

# ------------------------------------------------------------
# Lightweight PnL proxy (optional)
# ------------------------------------------------------------

@dataclass
class PnLConfig:
    rebalance: str = "W-FRI"         # 'D','W-FRI','M'
    slippage_bps: float = 0.2        # on notional per rebalance
    fee_bps: float = 0.1
    nav0: float = 1_000_000.0

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    if freq.upper() in ("D","DAILY"):
        return pd.Series(True, index=idx)
    return pd.Series(1, index=idx).resample(freq).last().reindex(idx).fillna(0).astype(bool)

def backtest_breakeven_signal(
    *,
    nominal_yields: pd.DataFrame,
    tips_yields: pd.DataFrame,
    cfg: SignalConfig = SignalConfig(),
    pnl_cfg: PnLConfig = PnLConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Toy P&L proxy (model-free):
      - Exposure is weights over chosen tenors.
      - Daily P&L ≈ sum_tenors [ weight_tau * ΔBE_tau ]  (breakeven delta proxy)
      - Costs: fee + slippage (bps of gross turnover per rebalance).
    This is a directional proxy, not a full cash/TIPS/UST DV01-exact engine.
    """
    be = breakeven_panel(nominal_yields=nominal_yields, tips_yields=tips_yields, tenors=cfg.tenors).dropna(how="all")
    mom = breakeven_momentum(be, lookback_days=cfg.mom_lb_days, diff_horizon_days=cfg.mom_diff_days)
    car = breakeven_carry_rolldown(be, tenors=cfg.tenors, roll_horizon_years=cfg.carry_roll_horizon_years)

    idx = be.index.intersection(mom.index).intersection(car.index)
    if len(idx) < max(cfg.mom_lb_days, cfg.mom_diff_days) + 20:
        return {"summary": pd.DataFrame(), "weights": pd.DataFrame()}

    be = be.reindex(idx); mom = mom.reindex(idx); car = car.reindex(idx)
    rb = _rb_mask(idx, pnl_cfg.rebalance) # type: ignore

    weights = pd.DataFrame(0.0, index=idx, columns=cfg.tenors)
    last_w = pd.Series(0.0, index=cfg.tenors, dtype=float)
    nav = pnl_cfg.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    price_pnl = pd.Series(0.0, index=idx, dtype=float)
    costs = pd.Series(0.0, index=idx, dtype=float)

    for t in idx:
        if rb.loc[t]:
            # rebuild weights at t using info up to t (contemporaneous in this proxy)
            m_t = _winsorize(mom.loc[t].dropna(), cfg.winsor_p)
            c_t = _winsorize(car.loc[t].dropna(), cfg.winsor_p)
            score = cfg.mom_weight * m_t + cfg.carry_weight * c_t
            if score.abs().sum() > 0:
                w = score / score.abs().sum()
            else:
                w = pd.Series(0.0, index=cfg.tenors)
            w = w.clip(-cfg.cap_per_tenor, cfg.cap_per_tenor)
            if w.abs().sum() > 0:
                w = w * (cfg.unit_gross / w.abs().sum())
            weights.loc[t] = w.reindex(weights.columns).fillna(0.0)

            # cost on turnover
            turn = (w - last_w).abs().sum()
            costs.loc[t] = (pnl_cfg.slippage_bps + pnl_cfg.fee_bps) * 1e-4 * turn * nav
            last_w = w
        else:
            weights.loc[t] = last_w

        # daily BE change pnl
        dbe = be.diff().loc[t].fillna(0.0)
        pnl_t = float((last_w.reindex(dbe.index).fillna(0.0) * dbe).sum() * nav)
        price_pnl.loc[t] = pnl_t

        nav = nav + pnl_t - costs.loc[t]
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "price_pnl$": price_pnl,
        "costs$": costs,
        "pnl$": price_pnl - costs,
        "nav": nav_path,
    })
    equity_base = summary["nav"].shift(1).fillna(pnl_cfg.nav0)
    summary["ret_net"] = summary["pnl$"] / equity_base.replace(0, np.nan)

    return {"summary": summary.fillna(0.0), "weights": weights.fillna(0.0)}

# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":
    # Tiny synthetic demo: generate nominal and real curves for 5/10/30 with slow trends
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    rng = np.random.default_rng(7)
    def randwalk(mu, sig, start):
        x = np.zeros(len(idx)); x[0]=start
        for i in range(1,len(idx)):
            x[i] = x[i-1] + mu + sig*rng.standard_normal()
        return pd.Series(x, index=idx)/100.0  # percent → decimal

    nom = pd.DataFrame({
        5.0:  randwalk(+0.01, 0.05, 2.0),
        10.0: randwalk(+0.00, 0.04, 2.2),
        30.0: randwalk(-0.01, 0.03, 2.5),
    }, index=idx)
    real = pd.DataFrame({
        5.0:  randwalk(+0.00, 0.04, 0.7),
        10.0: randwalk(+0.00, 0.04, 0.9),
        30.0: randwalk(+0.00, 0.04, 1.1),
    }, index=idx)

    sig = build_breakeven_signal(nominal_yields=nom, tips_yields=real)
    print("Snapshot weights:\n", sig["weights"])

    bt = backtest_breakeven_signal(nominal_yields=nom, tips_yields=real)
    print("PnL summary tail:\n", bt["summary"].tail())