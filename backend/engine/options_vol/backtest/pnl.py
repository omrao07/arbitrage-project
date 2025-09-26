# engines/options/backtest/pnlk.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class CostSpec:
    commission_per_contract: float = 0.0   # one-way, in base currency
    slippage_bps: float = 0.0              # of option notional (Px * multiplier * |traded|)

# ---------------------------------------------------------------------
# Black–Scholes (european)
# ---------------------------------------------------------------------

SQRT_2PI = np.sqrt(2.0 * np.pi)

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0))) # type: ignore

def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQRT_2PI

def _bs_d1(S, K, r, q, vol, T):
    vol = np.maximum(vol, 1e-9)
    T = np.maximum(T, 1e-9)
    return (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))

def _bs_price(right: np.ndarray, S, K, r, q, vol, T):
    # right: +1 for call, -1 for put
    d1 = _bs_d1(S, K, r, q, vol, T)
    d2 = d1 - vol * np.sqrt(np.maximum(T, 1e-9))
    cp = (right > 0).astype(float) * 2.0 - 1.0  # +1 for C, -1 for P
    disc_q = np.exp(-q * T); disc_r = np.exp(-r * T)
    call = disc_q * S * _norm_cdf(d1) - disc_r * K * _norm_cdf(d2)
    put  = disc_r * K * _norm_cdf(-d2) - disc_q * S * _norm_cdf(-d1)
    return np.where(right > 0, call, put)

def _bs_greeks(right: np.ndarray, S, K, r, q, vol, T):
    """
    Returns dict of arrays: delta, gamma, theta, vega, vanna, vomma (per 1 unit of option, notional-neutral)
    Conventions:
      - vega = dP/dσ (per 1.00 vol, not per 1%)
      - theta = dP/dt (per calendar year; negative is decay)
      - vanna = d2P/(dS dσ)
      - vomma = d2P/dσ^2
    """
    T = np.maximum(T, 1e-9)
    vol = np.maximum(vol, 1e-9)
    d1 = _bs_d1(S, K, r, q, vol, T)
    d2 = d1 - vol * np.sqrt(T)
    disc_q = np.exp(-q * T); disc_r = np.exp(-r * T)
    pdf = _norm_pdf(d1)
    sign = (right > 0).astype(float) * 2.0 - 1.0  # +1 call, -1 put

    delta = disc_q * _norm_cdf(sign * d1) * sign
    gamma = disc_q * pdf / (S * vol * np.sqrt(T))
    vega  = disc_q * S * pdf * np.sqrt(T)          # per 1.00 vol
    theta = (-disc_q * S * pdf * vol / (2*np.sqrt(T))
             - r * disc_r * K * _norm_cdf(sign * d2) * (sign > 0).astype(float)
             + q * disc_q * S * _norm_cdf(sign * d1) * (sign > 0).astype(float))
    # Put/call adjustments for theta closed form:
    # use put-call parity derivative; simpler: finite-diff small dt if needed. This closed form is adequate.

    # Cross-greeks:
    vanna = disc_q * pdf * np.sqrt(T) * (1 - d1 / (vol * np.sqrt(T)))  # d2P/dS dσ (approx form)
    vomma = vega * d1 * d2 / np.maximum(vol, 1e-9)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "vanna": vanna,
        "vomma": vomma,
        "d1": d1,
        "d2": d2,
    }

# ---------------------------------------------------------------------
# Portfolio P&L / attribution
# ---------------------------------------------------------------------

def _ensure_series(x, index, fill=0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index).astype(float).fillna(fill)
    return pd.Series(fill, index=index, dtype=float)

def _yearfrac(t1: pd.Series, t2: pd.Series) -> pd.Series:
    return (t2 - t1).dt.days.astype(float) / 365.0

def compute_options_pnl(
    *,
    # Static option metadata (indexed by option_id)
    meta: pd.DataFrame,  # columns: ['underlying','right','strike','expiry','multiplier'] right in {'C','P'}
    # Time series panels indexed by date, columns = option_id (for IV) / underlying_id (for S)
    S: pd.DataFrame,     # underlying spot (date x underlying)
    IV: pd.DataFrame,    # implied vol (date x option_id), in decimal e.g. 0.24
    positions: pd.DataFrame,    # option contracts (date x option_id), + long / - short
    trades: Optional[pd.DataFrame] = None,  # contracts traded (date x option_id)
    rate: float | pd.Series = 0.02,         # risk-free (annualized, decimal). scalar or date series
    div_yield: float | pd.Series | Dict[str,float] = 0.0,  # dividend yield by underlying (scalar/series/map)
    cost: CostSpec = CostSpec(),
) -> Dict[str, pd.DataFrame]:
    """
    Exact reval + Greeks attribution between t-1 and t:
      dP_exact = Price_t - Price_{t-1}
      dP_attr  ≈ Δ*dS + 0.5*Γ*(dS)^2 + Vega*dσ + Θ*dt + Vanna*(dS*dσ) + 0.5*Vomma*(dσ)^2
    Units: currency (after applying option multiplier and contracts).
    """
    # ---- Setup / alignment
    meta = meta.copy()
    if "option_id" in meta.columns:
        meta = meta.set_index("option_id")
    req_cols = {"underlying","right","strike","expiry","multiplier"}
    missing = req_cols - set(meta.columns)
    if missing:
        raise ValueError(f"meta is missing columns: {sorted(missing)}")

    # Ensure panels align on dates
    idx = positions.index.sort_values()
    IV = IV.reindex(index=idx, columns=positions.columns).ffill()
    if trades is None:
        trades = positions.diff().fillna(positions.iloc[0])
    else:
        trades = trades.reindex(index=idx, columns=positions.columns).fillna(0.0)
    right = meta["right"].map({"C": 1, "P": -1}).astype(int)

    # Rate / div term structures
    if isinstance(rate, (float, int)):
        r = pd.Series(rate, index=idx)
    else:
        r = rate.reindex(idx).ffill().astype(float)
    if isinstance(div_yield, (float, int)):
        q_by_und = {u: float(div_yield) for u in meta["underlying"].unique()}
    elif isinstance(div_yield, dict):
        q_by_und = {u: float(div_yield.get(u, 0.0)) for u in meta["underlying"].unique()}
    else:
        # series by date (global)
        q_by_und = None
        q_series = div_yield.reindex(idx).ffill().astype(float)

    # Helper to get q per option_id at date t
    def q_for_opts(at_date) -> np.ndarray:
        if q_by_und is not None:
            return meta["underlying"].map(q_by_und).values.astype(float)
        return np.repeat(float(q_series.loc[at_date]), len(meta))

    # Underlying spot panel aligned
    und_cols = list(S.columns)
    U = pd.DataFrame(index=idx, columns=positions.columns, dtype=float)
    for oid, und in meta["underlying"].items():
        if und not in S.columns:
            raise KeyError(f"Missing spot series for underlying '{und}' required by option {oid}")
        U[oid] = S[und]
    U = U.ffill()

    # Static arrays
    K = meta["strike"].astype(float).values
    mult = meta["multiplier"].astype(float).values
    right_arr = right.values
    exp = pd.to_datetime(meta["expiry"]).values.astype("datetime64[ns]")

    # Output collectors
    cols = ["px", "delta", "gamma", "vega", "theta", "vanna", "vomma"]
    snap = pd.DataFrame(index=idx, columns=pd.MultiIndex.from_product([positions.columns, cols]), dtype=float)
    pnl_rows = []

    # Loop dates to compute prices/greeks and attribution
    prev_vals = None
    prev_date = None

    for t in idx:
        S_t = U.loc[t].values  # per option id
        iv_t = IV.loc[t].values
        r_t = float(r.loc[t])
        # time to expiry (in years) for each option
        T_t = np.maximum((exp - np.datetime64(t)) / np.timedelta64(1, "D"), 0.0) / 365.0
        q_t = q_for_opts(t)

        # price & greeks per option (unit)
        px_t = _bs_price(right_arr, S_t, K, r_t, q_t, iv_t, T_t) # type: ignore
        greeks = _bs_greeks(right_arr, S_t, K, r_t, q_t, iv_t, T_t) # type: ignore
        row = np.column_stack([px_t, greeks["delta"], greeks["gamma"], greeks["vega"], greeks["theta"], greeks["vanna"], greeks["vomma"]])
        # fill frame
        snap_t = pd.DataFrame(row.T, index=cols, columns=positions.columns).T
        snap.loc[t, :] = snap_t.values

        # Attribution vs previous date
        if prev_vals is not None:
            dt_years = max((t - prev_date).days / 365.0, 1e-9)
            dS = S_t - prev_vals["S"]
            dIV = iv_t - prev_vals["iv"]
            # use previous greeks for incremental attribution (can also mid-point average)
            Δ = prev_vals["delta"] * dS
            Γ = 0.5 * prev_vals["gamma"] * (dS ** 2)
            V = prev_vals["vega"] * dIV
            Θ = prev_vals["theta"] * dt_years
            Va = prev_vals["vanna"] * (dS * dIV)
            Vo = 0.5 * prev_vals["vomma"] * (dIV ** 2)

            # per-option value change (unit):
            dP_attr_unit = Δ + Γ + V + Θ + Va + Vo
            dP_exact_unit = px_t - prev_vals["px"]

            # position sizes (contracts) → notional P&L
            pos_prev = positions.loc[prev_date].values  # attribution from t-1 inventory
            mult_arr = mult

            exact = dP_exact_unit * pos_prev * mult_arr
            attr  = dP_attr_unit * pos_prev * mult_arr
            resid = exact - attr

            # Trading costs on today's trades (at t): commissions + slippage
            trd_t = trades.loc[t].values if t in trades.index else np.zeros_like(pos_prev)
            # execution price proxy = current option price
            traded_notional = np.abs(trd_t) * mult_arr * px_t
            fees = np.abs(trd_t) * float(cost.commission_per_contract)
            slip = traded_notional * (float(cost.slippage_bps) * 1e-4)
            costs = fees + slip

            pnl_rows.append({
                "date": t,
                "price_pnl$": np.nansum(exact),
                "delta$": np.nansum(Δ * pos_prev * mult_arr),
                "gamma$": np.nansum(Γ * pos_prev * mult_arr),
                "vega$":  np.nansum(V * pos_prev * mult_arr),
                "theta$": np.nansum(Θ * pos_prev * mult_arr),
                "vanna$": np.nansum(Va * pos_prev * mult_arr),
                "vomma$": np.nansum(Vo * pos_prev * mult_arr),
                "residual$": np.nansum(resid),
                "fees$": np.nansum(fees),
                "slippage$": np.nansum(slip),
                "pnl_net$": np.nansum(exact) - np.nansum(fees + slip),
                "gross_traded_notional$": np.nansum(traded_notional),
            })

        # store for next step
        prev_vals = {
            "S": S_t, "iv": iv_t, "px": px_t,
            "delta": greeks["delta"], "gamma": greeks["gamma"], "vega": greeks["vega"],
            "theta": greeks["theta"], "vanna": greeks["vanna"], "vomma": greeks["vomma"],
        }
        prev_date = t

    # Build outputs
    detail = snap.sort_index(axis=1)  # date x (option, metric)
    summary = pd.DataFrame(pnl_rows).set_index("date") if pnl_rows else pd.DataFrame()

    # Per-option cumulative contributions (sum across time)
    if not summary.empty:
        # rebuild per-option contributions on last loop? Provide a condensed table now:
        pass

    return {"detail": detail, "summary": summary}

# ---------------------------------------------------------------------
# Example (synthetic)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # One underlying 'SPX' and 4 options (2 calls, 2 puts)
    idx = pd.date_range("2024-01-02", periods=80, freq="B")
    rng = np.random.default_rng(7)
    S = pd.DataFrame({"SPX": 4800 * np.exp(np.cumsum(0.0002 + 0.012 * rng.standard_normal(len(idx))))}, index=idx)

    meta = pd.DataFrame({
        "option_id": ["C4800_M1","P4800_M1","C5000_M3","P4600_M3"],
        "underlying": ["SPX","SPX","SPX","SPX"],
        "right": ["C","P","C","P"],
        "strike": [4800, 4800, 5000, 4600],
        "expiry": [idx[40], idx[40], idx[65], idx[65]],
        "multiplier": [100.0, 100.0, 100.0, 100.0],
    }).set_index("option_id")

    # IV panel per option
    base_iv = pd.Series({"C4800_M1": 0.20, "P4800_M1": 0.22, "C5000_M3": 0.19, "P4600_M3": 0.21})
    IV = pd.DataFrame({c: (base_iv[c] + 0.02*np.sin(np.linspace(0, 6, len(idx))) + 0.01*rng.standard_normal(len(idx)))
                       for c in base_iv.index}, index=idx).abs().clip(0.05, 1.0)

    # Positions (contracts) and trades
    pos = pd.DataFrame(0.0, index=idx, columns=meta.index)
    # Enter a 1x call spread and 1x put hedge at day 5; roll the M1 at day 40 to zero
    pos.loc[idx[5]:, "C4800_M1"] = +10
    pos.loc[idx[5]:, "P4800_M1"] = -8
    pos.loc[idx[5]:, "C5000_M3"] = +4
    pos.loc[idx[5]:, "P4600_M3"] = -3
    pos.loc[idx[40]:, "C4800_M1"] = 0
    pos.loc[idx[40]:, "P4800_M1"] = 0
    pos = pos.ffill()

    trades = pos.diff().fillna(pos.iloc[0])

    out = compute_options_pnl(
        meta=meta, S=S, IV=IV, positions=pos, trades=trades,
        rate=0.045, div_yield=0.015, cost=CostSpec(commission_per_contract=0.5, slippage_bps=5.0)
    )
    print(out["summary"].tail())