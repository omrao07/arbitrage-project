# engines/cap_struct/capital_structure.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from math import log, sqrt, exp, erf

TRADING_DAYS = 252

# =============================================================================
# Config
# =============================================================================

@dataclass
class InputsConfig:
    risk_free_annual: float = 0.03      # risk-free rate (annual, dec.)
    asset_vol_lb_days: int = 63         # lookback for equity vol proxy
    leverage_ratio: float = 0.6         # D/(D+E) if no balance-sheet series provided
    recovery_rate: float = 0.4          # for mapping hazard -> spread
    tenor_years: float = 5.0            # CDS tenor for comparisons
    use_log_returns: bool = True

@dataclass
class SignalConfig:
    z_lookback: int = 252               # z-score window
    clip_z: float = 4.0
    mode: str = "parity"                # "parity" (CDS vs equity-implied), "trend", "combo"
    w_parity: float = 0.7
    w_trend: float = 0.3

@dataclass
class HedgeConfig:
    rebalance: str = "W-FRI"            # 'D', 'W-FRI', 'M'
    regress_lb: int = 60                # window for hedge ratio regression Δspread ~ beta * r_equity
    dv01_notional: float = 1_000_000.0  # $ per 1bp spread PV01 (for CDS leg sizing proxy)
    equity_notional_cap: float = 2_000_000.0

@dataclass
class BacktestConfig:
    nav0: float = 1_000_000.0
    tc_bps_cds: float = 1.0             # bps of traded CDS notional
    tc_bps_equity: float = 5.0          # bps of traded equity notional
    hold_days: int = 21                 # refresh CDS strike every ~1m (for accrual calc)
    accrual_freq: int = 4               # quarterly premiums (informational)
    default_recovery: float = 0.4       # used if a default flag is supplied

# =============================================================================
# Utilities
# =============================================================================

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

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# =============================================================================
# Equity-implied credit (simple Merton-ish mapping)
# =============================================================================

def equity_implied_spread(
    *,
    equity_px: pd.Series,               # price (split-adjusted)
    shares_out: float,                  # share count (to scale to equity value)
    debt_value: Optional[pd.Series] = None, # total debt market value (or book proxied, constant ok)
    cfg: InputsConfig = InputsConfig(),
) -> pd.Series:
    """
    Map equity to a simple default intensity via distance-to-default, then to a CDS-like spread.
    Steps (toy but stable):
      1) Firm value V ≈ E + D  (D from series or leverage ratio)
      2) Asset vol σ_V ≈ σ_E * (E/V)  (equity vol from rolling realized)
      3) d2 = [ln(V/D) + (r - 0.5 σ_V^2) T] / (σ_V √T)
      4) Default prob ~ N(-d2); convert to hazard λ ≈ -ln(1 - PD)/T
      5) Spread ≈ λ * (1 - R)  (annualized, in decimal) → * 10,000 for bps
    This is NOT a full structural model; it’s a robust signal proxy.
    """
    px = equity_px.sort_index().astype(float)
    idx = px.index

    # Equity value E
    E = px * float(shares_out)

    # Debt value D
    if debt_value is not None:
        D = debt_value.reindex(idx).ffill().bfill().astype(float)
    else:
        # Approx via leverage ratio: D = (lev/(1-lev)) * E
        lev = max(1e-6, min(0.95, cfg.leverage_ratio))
        D = (lev / (1.0 - lev)) * E

    V = E + D
    T = float(cfg.tenor_years)
    r = float(cfg.risk_free_annual)

    # Equity vol (realized) and map to asset vol: σ_V ≈ σ_E * (E/V)
    r_e = (np.log(px).diff() if cfg.use_log_returns else px.pct_change()).fillna(0.0) # type: ignore
    sig_e = r_e.rolling(cfg.asset_vol_lb_days).std() * sqrt(TRADING_DAYS)
    # Avoid zero/NaN
    lev_equity = (E / V).replace(0, np.nan)
    sig_v = (sig_e * lev_equity).fillna(method="ffill").fillna(0.0001).clip(0.0001, 3.0) # type: ignore

    # Distance to default (d2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_VD = np.log((V / D).replace(0, np.nan))
        d2 = (ln_VD + (r - 0.5 * sig_v**2) * T) / (sig_v * sqrt(T))
    pd_t = (1.0 - pd.Series([_norm_cdf(x) for x in d2], index=idx)).clip(1e-8, 1-1e-8)
    hazard = -np.log(1.0 - pd_t) / T
    spread_dec = hazard * (1.0 - cfg.recovery_rate)
    spread_bps = (spread_dec * 10_000.0).rename("equity_implied_cds_bps") # pyright: ignore[reportAttributeAccessIssue]
    return spread_bps

# =============================================================================
# Signal construction
# =============================================================================

def build_capstruct_signal(
    *,
    cds_spread_bps: pd.Series,          # observed CDS 5Y spread in bps
    equity_px: pd.Series,
    shares_out: float,
    debt_value: Optional[pd.Series] = None,
    in_cfg: InputsConfig = InputsConfig(),
    sig_cfg: SignalConfig = SignalConfig(),
) -> Dict[str, pd.Series | pd.DataFrame]:
    cds = cds_spread_bps.sort_index().astype(float)
    eq = equity_px.sort_index().astype(float)
    idx = cds.index.intersection(eq.index) # type: ignore
    cds, eq = cds.reindex(idx), eq.reindex(idx)

    eq_imp = equity_implied_spread(equity_px=eq, shares_out=shares_out, debt_value=debt_value, cfg=in_cfg).reindex(idx)
    parity = (cds - eq_imp).rename("parity_basis_bps")  # + = CDS rich vs equity

    # Trend component (spread momentum)
    trend = cds.diff(20).rename("spread_trend_20d")

    z_par = _zscore(parity, sig_cfg.z_lookback, sig_cfg.clip_z).rename("z_parity")
    z_tr  = _zscore(trend,  sig_cfg.z_lookback, sig_cfg.clip_z).rename("z_trend")

    if sig_cfg.mode == "parity":
        score = z_par
    elif sig_cfg.mode == "trend":
        score = z_tr
    else:
        score = (sig_cfg.w_parity*z_par + sig_cfg.w_trend*z_tr) / (sig_cfg.w_parity + sig_cfg.w_trend)
    score = score.rename("score")

    features = pd.concat([cds.rename("cds_bps"), eq_imp, parity, trend, z_par, z_tr, score], axis=1)
    diag = features.iloc[[-1]].copy() if len(features) else pd.DataFrame()
    return {"features": features, "equity_implied_bps": eq_imp, "score": score, "diag": diag}

# =============================================================================
# Hedge ratio & portfolio mapping
# =============================================================================

def estimate_hedge_ratio(
    *,
    cds_spread_bps: pd.Series,
    equity_px: pd.Series,
    lb: int = 60
) -> float:
    """
    Regress Δspread (bps) on equity returns to estimate hedge ratio:
      ΔS ≈ beta * r_equity
    Returns beta_hat in (bps per unit return).
    """
    cds = cds_spread_bps.sort_index().astype(float)
    eq = equity_px.sort_index().astype(float).reindex(cds.index)
    dS = cds.diff().rolling(lb).mean()  # smoother ΔS (avoid noise)
    rE = np.log(eq).diff() # type: ignore
    x = rE.dropna().align(dS.dropna(), join="inner")[0]
    y = dS.dropna().align(rE.dropna(), join="inner")[0]
    X, Y = rE.reindex(x.index), dS.reindex(x.index)
    X = X.dropna(); Y = Y.dropna()
    common = X.index.intersection(Y.index)
    if len(common) < 5: 
        return 0.0
    X = X.reindex(common); Y = Y.reindex(common)
    xv = X.values; yv = Y.values
    num = np.dot(xv - xv.mean(), yv - yv.mean()) # type: ignore
    den = np.dot(xv - xv.mean(), xv - xv.mean())
    beta = float(num / den) if den != 0 else 0.0
    return beta

def build_positions(
    *,
    score: pd.Series,                    # trading score (signed)
    cds_spread_bps: pd.Series,
    equity_px: pd.Series,
    hedge_cfg: HedgeConfig = HedgeConfig(),
    sign_convention: int = +1,           # +1: long CDS (buy protection) when score>0, short equity; -1 flips
) -> Dict[str, pd.DataFrame | pd.Series | float]:
    """
    Translate score into positions at each rebalance:
      - CDS leg: notional via dv01_notional units times |score|
      - Equity leg: hedge ratio converts expected ΔS (bps) to equity returns; target equity notional to offset
    Returns time series of target notionals (piecewise constant between rebalances).
    """
    cds = cds_spread_bps.sort_index().astype(float)
    eq = equity_px.sort_index().astype(float).reindex(cds.index)
    idx = cds.index

    rb = _rb_mask(idx, hedge_cfg.rebalance) # type: ignore
    beta = estimate_hedge_ratio(cds_spread_bps=cds, equity_px=eq, lb=hedge_cfg.regress_lb)
    # equity notional per 1 unit of CDS dv01 exposure (heuristic):
    # If ΔS ≈ beta * r_E, then a 1bp move (0.01%) corresponds to r_E ≈ 0.0001 / beta
    # Scale equity notional so that equity PnL offsets CDS MTM for small moves.
    equity_per_dv01 = 0.0 if beta == 0 else (hedge_cfg.dv01_notional * 0.0001 / beta)

    # Build stepwise positions
    cds_notional = pd.Series(0.0, index=idx)
    equity_notional = pd.Series(0.0, index=idx)

    last_cds = 0.0
    last_eq = 0.0

    for t in idx:
        if rb.loc[t]:
            sc = float(score.reindex(idx).ffill().loc[t])
            # Target CDS notional in DV01 dollars (signed)
            target_cds = sign_convention * np.sign(sc) * min(abs(sc), 1.0) * hedge_cfg.dv01_notional
            # Equity leg
            target_eq = -np.sign(target_cds) * equity_per_dv01
            target_eq = float(np.clip(target_eq, -hedge_cfg.equity_notional_cap, hedge_cfg.equity_notional_cap))
            last_cds, last_eq = target_cds, target_eq
        cds_notional.loc[t] = last_cds
        equity_notional.loc[t] = last_eq

    return {"cds_notional_dv01$": cds_notional, "equity_notional$": equity_notional, "beta_bps_per_ret": beta, "equity_per_dv01$": equity_per_dv01}

# =============================================================================
# Backtest (proxy: CDS DV01 leg + equity cash leg)
# =============================================================================

def backtest_capstruct(
    *,
    cds_spread_bps: pd.Series,
    equity_px: pd.Series,
    score: pd.Series,
    defaults_flag: Optional[pd.Series] = None,   # 1 on default date (optional)
    in_cfg: InputsConfig = InputsConfig(),
    hedge_cfg: HedgeConfig = HedgeConfig(),
    bt_cfg: BacktestConfig = BacktestConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    PnL approximation:
      - CDS leg: MTM ≈ (-ΔS * DV01_notional * sign), accrual premium ignored/optional
      - Equity leg: PnL = notional * r_equity
      - Costs: bps on notional changes for each leg at rebalance
      - Default: if default=1, CDS payoff = + sign * Notional_DV01 * ( (1-R)*10000 )? (We map a jump on spread to very large; here use lump sum)
        For simplicity we book: payoff = sign * nominal_default$, where nominal_default$ = equity_per_dv01 mapping * k. Keep conservative (0 if not used).
    This is intentionally conservative/clean for signal testing; wire to your real CDS pricer for production.
    """
    cds = cds_spread_bps.sort_index().astype(float)
    eq = equity_px.sort_index().astype(float)
    idx = cds.index.intersection(eq.index) # type: ignore
    cds, eq = cds.reindex(idx), eq.reindex(idx)
    if defaults_flag is None:
        defaults_flag = pd.Series(0, index=idx)

    # Positions
    pos = build_positions(score=score.reindex(idx), cds_spread_bps=cds, equity_px=eq, hedge_cfg=hedge_cfg)
    cds_dv01 = pos["cds_notional_dv01$"]
    eq_notional = pos["equity_notional$"]

    # Rebalance mask
    rb = _rb_mask(idx, hedge_cfg.rebalance) # type: ignore

    # Returns / changes
    rE = (np.log(eq).diff() if in_cfg.use_log_returns else eq.pct_change()).fillna(0.0) # type: ignore
    dS = cds.diff().fillna(0.0)  # bps per day

    nav = bt_cfg.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    pnl = pd.Series(0.0, index=idx)
    costs = pd.Series(0.0, index=idx)

    last_cds = 0.0
    last_eq = 0.0

    for t in idx:
        # costs when we change positions
        if rb.loc[t]:
            traded_cds = abs(cds_dv01.loc[t] - last_cds) # type: ignore
            traded_eq  = abs(eq_notional.loc[t] - last_eq) # type: ignore
            costs.loc[t] += traded_cds * (bt_cfg.tc_bps_cds * 1e-4) + traded_eq * (bt_cfg.tc_bps_equity * 1e-4)
            last_cds, last_eq = cds_dv01.loc[t], eq_notional.loc[t] # type: ignore

        # daily PnL
        pnl_cds = -np.sign(last_cds) * abs(last_cds) * dS.loc[t]          # DV01$ * (-ΔS)
        pnl_eq  = last_eq * rE.loc[t]                                     # equity cash leg
        pnl_t = pnl_cds + pnl_eq - costs.loc[t]

        # default jump (very simplified; you can replace with actual CDS payoff)
        if defaults_flag.loc[t] == 1 and last_cds != 0:
            payoff = np.sign(last_cds) * abs(last_cds) * ( (1.0 - bt_cfg.default_recovery) * 100.0 )  # scale factor
            pnl_t += payoff
            # tear-down positions
            last_cds = 0.0
            last_eq = 0.0

        pnl.loc[t] = pnl_t
        nav += pnl_t
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "nav$": nav_path,
        "pnl$": pnl,
        "costs$": costs.cumsum(),
        "cds_dv01$": cds_dv01,
        "equity_notional$": eq_notional,
    }, index=idx)

    return {"summary": summary, "positions": pd.DataFrame({"cds_dv01$": cds_dv01, "equity$": eq_notional}), "returns": pd.DataFrame({"dS_bps": dS, "rE": rE})}

# =============================================================================
# Example (synthetic)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-02", periods=500, freq="B")

    # Synthetic equity
    rE = 0.0003 + 0.02 * rng.standard_normal(len(idx))
    px = pd.Series(50.0 * np.exp(np.cumsum(rE)), index=idx, name="EQ")

    # Synthetic CDS spreads (bps) somewhat anti-correlated with equity returns
    dS = (-80 * rE + 0.5 * rng.standard_normal(len(idx))).cumsum()
    cds = pd.Series(120 + dS, index=idx, name="CDS").clip(30, 1000)

    # Build signal
    in_cfg = InputsConfig(risk_free_annual=0.03, asset_vol_lb_days=63, leverage_ratio=0.6, tenor_years=5.0, recovery_rate=0.4)
    sig_cfg = SignalConfig(mode="combo", z_lookback=126, w_parity=0.7, w_trend=0.3)

    out_sig = build_capstruct_signal(cds_spread_bps=cds, equity_px=px, shares_out=500_000_00, debt_value=None, in_cfg=in_cfg, sig_cfg=sig_cfg)
    score = out_sig["score"]

    # Backtest
    hedge_cfg = HedgeConfig(rebalance="W-FRI", regress_lb=60, dv01_notional=1_000_000, equity_notional_cap=2_000_000)
    bt_cfg = BacktestConfig(nav0=1_000_000, tc_bps_cds=1.0, tc_bps_equity=5.0, hold_days=21)

    bt = backtest_capstruct(cds_spread_bps=cds, equity_px=px, score=score, in_cfg=in_cfg, hedge_cfg=hedge_cfg, bt_cfg=bt_cfg) # type: ignore
    print(bt["summary"].tail())