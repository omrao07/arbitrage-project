# engines/stat_arb/backtest/pnl.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

BPS = 1e-4
TRADING_DAYS = 252

BetaMethod = Literal["static_ols", "rolling_ols", "expanding_ols"]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_df(x: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(name=name)
    return x.copy()

def _align_two(a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    a = a.squeeze() # type: ignore
    b = b.squeeze() # type: ignore
    idx = a.index.intersection(b.index) # type: ignore
    return a.reindex(idx).astype(float), b.reindex(idx).astype(float) # type: ignore

def _safe_pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _rolling_beta(y: pd.Series, x: pd.Series, method: BetaMethod = "rolling_ols", lookback: int = 252) -> pd.Series:
    """
    y_t ≈ α + β x_t → return β_t series (no intercept used for hedging weight).
    """
    y, x = _align_two(y, x)
    if method == "static_ols":
        X = np.vstack([x.values, np.ones(len(x))]).T # type: ignore
        beta = np.linalg.lstsq(X, y.values, rcond=None)[0][0] # type: ignore
        return pd.Series(beta, index=y.index)
    betas = pd.Series(index=y.index, dtype=float)
    if method == "expanding_ols":
        for i in range(2, len(x)+1):
            yw = y.iloc[:i]
            xw = x.iloc[:i]
            X = np.vstack([xw.values, np.ones(len(xw))]).T # type: ignore
            b = np.linalg.lstsq(X, yw.values, rcond=None)[0][0] # type: ignore
            betas.iloc[i-1] = b
        betas.iloc[:1] = betas.iloc[1]
        return betas.ffill()
    # rolling_ols
    if lookback < 5:
        lookback = 5
    for i in range(lookback, len(x)+1):
        yw = y.iloc[i-lookback:i]
        xw = x.iloc[i-lookback:i]
        X = np.vstack([xw.values, np.ones(len(xw))]).T # type: ignore
        b = np.linalg.lstsq(X, yw.values, rcond=None)[0][0] # type: ignore
        betas.iloc[i-1] = b
    betas.iloc[:lookback] = betas.iloc[lookback]
    return betas.ffill()

def estimate_spread(y: pd.Series, x: pd.Series, *, beta: Optional[pd.Series | float] = None,
                    beta_method: BetaMethod = "rolling_ols", lookback: int = 252) -> Tuple[pd.Series, pd.Series]:
    """
    Return (spread, beta_series). spread_t = y_t - β_t * x_t
    """
    y, x = _align_two(y, x)
    if beta is None:
        beta_series = _rolling_beta(y, x, method=beta_method, lookback=lookback)
    elif np.isscalar(beta):
        beta_series = pd.Series(float(beta), index=y.index) # type: ignore
    else:
        beta_series = pd.Series(beta).reindex(y.index).ffill().bfill()
    spread = y - beta_series * x
    return spread, beta_series

# -----------------------------------------------------------------------------
# P&L for one pair
# -----------------------------------------------------------------------------

@dataclass
class PairInputs:
    """
    Provide either:
      - spread_units: DataFrame[date x 1] of units of the spread (1 unit = +1 share Y, -β shares X)
        and optional trades_units DataFrame for executions; OR
      - shares_y / shares_x DataFrames and optional trades_y / trades_x.
    """
    price_y: pd.Series           # dependent leg price series
    price_x: pd.Series           # hedge leg price series
    beta: Optional[pd.Series | float] = None
    beta_method: BetaMethod = "rolling_ols"
    beta_lookback: int = 252

    # Positioning (choose one style)
    spread_units: Optional[pd.DataFrame] = None
    trades_units: Optional[pd.DataFrame] = None
    shares_y: Optional[pd.DataFrame] = None
    shares_x: Optional[pd.DataFrame] = None
    trades_y: Optional[pd.DataFrame] = None
    trades_x: Optional[pd.DataFrame] = None

    # Costs/carry
    fee_bps: float = 0.3
    slippage_bps: float = 1.5
    borrow_bps_y: float = 50.0
    borrow_bps_x: float = 50.0
    div_yield_bps_y: float = 150.0
    div_yield_bps_x: float = 150.0

def _carry_for_leg(shares: pd.DataFrame, prices: pd.DataFrame, borrow_bps: float, div_bps: float) -> pd.Series:
    notion = (shares.shift(1).fillna(0.0) * prices)  # SOD notional
    long_notional = notion.clip(lower=0.0).sum(axis=1)
    short_notional = (-notion.clip(upper=0.0)).sum(axis=1)
    long_carry = long_notional * (div_bps * BPS / TRADING_DAYS)
    short_carry = - short_notional * (borrow_bps * BPS / TRADING_DAYS)
    return (long_carry + short_carry).fillna(0.0)

def _flat_costs(trades_usd: pd.Series, fee_bps: float, slip_bps: float) -> pd.Series:
    return trades_usd * ((fee_bps + slip_bps) * BPS)

def compute_pair_pnl(inputs: PairInputs) -> Dict[str, pd.DataFrame | pd.Series | float]:
    """
    Compute daily P&L for a single pair.
    Returns dict with: summary (DataFrame), per_leg (DataFrame), spread (Series), beta (Series)
    """
    y = inputs.price_y.astype(float).sort_index()
    x = inputs.price_x.astype(float).sort_index()
    idx = y.index.intersection(x.index) # type: ignore
    y = y.reindex(idx); x = x.reindex(idx)

    spread, beta_series = estimate_spread(y, x, beta=inputs.beta,
                                          beta_method=inputs.beta_method, lookback=inputs.beta_lookback)

    # Build positions
    if inputs.spread_units is not None:
        u = inputs.spread_units.reindex(idx).fillna(0.0).squeeze()
        shares_y = u.to_frame("u").apply(lambda s: s)  # type: ignore # +u on Y 
        shares_x = (-u * beta_series).to_frame("u")    # type: ignore # -β*u on X
        shares_y.columns = ["shares_y"]; shares_x.columns = ["shares_x"]
        shares_y = shares_y.astype(float); shares_x = shares_x.astype(float)

        if inputs.trades_units is not None:
            tu = inputs.trades_units.reindex(idx).fillna(0.0).squeeze()
            trades_y = tu.to_frame("u") # type: ignore
            trades_x = (-tu * beta_series).to_frame("u") # type: ignore
            trades_y.columns = ["trades_y"]; trades_x.columns = ["trades_x"]
        else:
            trades_y = shares_y.diff().fillna(shares_y.iloc[0])
            trades_x = shares_x.diff().fillna(shares_x.iloc[0])
    else:
        if inputs.shares_y is None or inputs.shares_x is None:
            raise ValueError("Provide either spread_units or both shares_y and shares_x.")
        shares_y = inputs.shares_y.reindex(idx).fillna(0.0).astype(float)
        shares_x = inputs.shares_x.reindex(idx).fillna(0.0).astype(float)
        trades_y = (inputs.trades_y.reindex(idx).fillna(0.0) if inputs.trades_y is not None
                    else shares_y.diff().fillna(shares_y.iloc[0]))
        trades_x = (inputs.trades_x.reindex(idx).fillna(0.0) if inputs.trades_x is not None
                    else shares_x.diff().fillna(shares_x.iloc[0]))

    # Convert to aligned DataFrames
    py = _ensure_df(y, "price_y"); px = _ensure_df(x, "price_x")
    pos_y = shares_y.rename(columns={shares_y.columns[0]: "shares_y"})
    pos_x = shares_x.rename(columns={shares_x.columns[0]: "shares_x"})
    trd_y = trades_y.rename(columns={trades_y.columns[0]: "trades_y"})
    trd_x = trades_x.rename(columns={trades_x.columns[0]: "trades_x"})

    # Price returns
    ret_y = _safe_pct_change(py.squeeze()) # type: ignore
    ret_x = _safe_pct_change(px.squeeze()) # type: ignore

    # Price P&L (use SOD positions)
    pos_y_lag = pos_y.shift(1).fillna(0.0).squeeze()
    pos_x_lag = pos_x.shift(1).fillna(0.0).squeeze()

    price_pnl_y = (pos_y_lag * ret_y * py.squeeze()).fillna(0.0) # type: ignore
    price_pnl_x = (pos_x_lag * ret_x * px.squeeze()).fillna(0.0) # type: ignore
    price_pnl_total = price_pnl_y + price_pnl_x

    # Spread attribution (approx): d(y - βx) * u_{t-1}
    # If you passed spread_units, u_{t-1} ≈ pos_y_lag (since shares_y≈u). For generality:
    if inputs.spread_units is not None:
        u_lag = inputs.spread_units.reindex(idx).fillna(0.0).squeeze().shift(1).fillna(0.0) # type: ignore
    else:
        u_lag = pos_y_lag  # proxy
    d_spread = spread.diff().fillna(0.0)
    spread_pnl = (u_lag * d_spread).astype(float) # type: ignore

    # Carry P&L per leg
    carry_y = _carry_for_leg(pos_y, py, inputs.borrow_bps_y, inputs.div_yield_bps_y)
    carry_x = _carry_for_leg(pos_x, px, inputs.borrow_bps_x, inputs.div_yield_bps_x)
    carry_total = carry_y + carry_x

    # Trading costs
    notional_traded = (trd_y.abs().squeeze() * py.squeeze()).fillna(0.0) +  # type: ignore
                      (trd_x.abs().squeeze() * px.squeeze()).fillna(0.0) # type: ignore
    costs = _flat_costs(notional_traded, inputs.fee_bps, inputs.slippage_bps)

    # Exposures
    notion_y = (pos_y_lag * py.squeeze()).abs() # type: ignore
    notion_x = (pos_x_lag * px.squeeze()).abs() # type: ignore
    gross = notion_y + notion_x
    net = (pos_y_lag * py.squeeze()) + (pos_x_lag * px.squeeze()) # type: ignore

    pnl_total = price_pnl_total + carry_total - costs
    equity_base = gross.replace(0, np.nan)
    ret_net = (pnl_total / equity_base).fillna(0.0)

    summary = pd.DataFrame({
        "gross_exposure$": gross,
        "net_exposure$": net,
        "turnover": (notional_traded / gross.replace(0, np.nan)).fillna(0.0),
        "pnl$": pnl_total,
        "price_pnl$": price_pnl_total,
        "spread_pnl$": spread_pnl,
        "carry_pnl$": carry_total,
        "costs$": costs,
        "ret_net": ret_net,
    }).fillna(0.0)

    per_leg = pd.DataFrame({
        "price_pnl_y$": price_pnl_y,
        "price_pnl_x$": price_pnl_x,
        "carry_y$": carry_y,
        "carry_x$": carry_x,
        "traded_notional_y$": (trd_y.abs().squeeze() * py.squeeze()).fillna(0.0), # type: ignore
        "traded_notional_x$": (trd_x.abs().squeeze() * px.squeeze()).fillna(0.0), # type: ignore
    }).fillna(0.0)

    return {
        "summary": summary,
        "per_leg": per_leg,
        "spread": spread,
        "beta": beta_series,
    }

# -----------------------------------------------------------------------------
# Portfolio of many pairs
# -----------------------------------------------------------------------------

@dataclass
class PairSpec:
    price_y: pd.Series
    price_x: pd.Series
    beta: Optional[pd.Series | float] = None
    beta_method: BetaMethod = "rolling_ols"
    beta_lookback: int = 252
    # positions (choose style, same for all pairs ideally)
    spread_units: Optional[pd.DataFrame] = None
    trades_units: Optional[pd.DataFrame] = None
    shares_y: Optional[pd.DataFrame] = None
    shares_x: Optional[pd.DataFrame] = None
    trades_y: Optional[pd.DataFrame] = None
    trades_x: Optional[pd.DataFrame] = None
    # costs/carry
    fee_bps: float = 0.3
    slippage_bps: float = 1.5
    borrow_bps_y: float = 50.0
    borrow_bps_x: float = 50.0
    div_yield_bps_y: float = 150.0
    div_yield_bps_x: float = 150.0

def compute_portfolio_pnl(pairs: Dict[str, PairSpec]) -> Dict[str, pd.DataFrame | Dict]:
    """
    Aggregate P&L across named pairs.
    Returns dict with:
      - summary (DataFrame): portfolio-level aggregates
      - by_pair (Dict[str, DataFrame]): per-pair summary
    """
    by_pair: Dict[str, pd.DataFrame] = {}
    parts = []

    for name, spec in pairs.items():
        res = compute_pair_pnl(PairInputs(
            price_y=spec.price_y, price_x=spec.price_x,
            beta=spec.beta, beta_method=spec.beta_method, beta_lookback=spec.beta_lookback,
            spread_units=spec.spread_units, trades_units=spec.trades_units,
            shares_y=spec.shares_y, shares_x=spec.shares_x,
            trades_y=spec.trades_y, trades_x=spec.trades_x,
            fee_bps=spec.fee_bps, slippage_bps=spec.slippage_bps,
            borrow_bps_y=spec.borrow_bps_y, borrow_bps_x=spec.borrow_bps_x,
            div_yield_bps_y=spec.div_yield_bps_y, div_yield_bps_x=spec.div_yield_bps_x,
        ))
        s = res["summary"].copy() # type: ignore
        s = s.add_prefix(f"{name}::")
        parts.append(s)
        by_pair[name] = res["summary"] # type: ignore

    # align on common index
    if not parts:
        return {"summary": pd.DataFrame(), "by_pair": by_pair}
    all_df = pd.concat(parts, axis=1).sort_index().fillna(0.0)

    # aggregate columns with the same suffixes
    def sum_cols(suffix: str) -> pd.Series:
        cols = [c for c in all_df.columns if c.endswith(suffix)]
        return all_df[cols].sum(axis=1)

    portfolio = pd.DataFrame({
        "gross_exposure$": sum_cols("gross_exposure$"),
        "net_exposure$": sum_cols("net_exposure$"),
        "pnl$": sum_cols("pnl$"),
        "price_pnl$": sum_cols("price_pnl$"),
        "spread_pnl$": sum_cols("spread_pnl$"),
        "carry_pnl$": sum_cols("carry_pnl$"),
        "costs$": sum_cols("costs$"),
    })
    # turnover as notional traded / gross (sum across pairs)
    turnover_num = sum_cols("turnover") * sum_cols("gross_exposure$")
    gross = portfolio["gross_exposure$"].replace(0, np.nan)
    portfolio["turnover"] = (turnover_num / gross).fillna(0.0)
    portfolio["ret_net"] = (portfolio["pnl$"] / gross).fillna(0.0)

    return {"summary": portfolio.fillna(0.0), "by_pair": by_pair}

# -----------------------------------------------------------------------------
# Quick demo (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny synthetic example: co-moving random walks for a pair
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    rng = np.random.default_rng(7)
    y = pd.Series(100.0, index=idx) * np.exp(np.cumsum(0.0002 + 0.015 * rng.standard_normal(len(idx))))
    x = pd.Series(90.0, index=idx)  * np.exp(np.cumsum(0.00015 + 0.016 * rng.standard_normal(len(idx))))

    # Buy 1 spread unit when spread z < -1, sell -1 when z > 1 (toy)
    spread, beta = estimate_spread(y, x, beta_method="rolling_ols", lookback=60)
    z = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std() + 1e-12)
    units = (z < -1).astype(float) - (z > 1).astype(float)
    units = units.replace(0.0, np.nan).ffill().fillna(0.0).to_frame("u")

    res = compute_pair_pnl(PairInputs(
        price_y=y, price_x=x,
        beta=beta,
        spread_units=units,
        fee_bps=0.3, slippage_bps=1.5,
        borrow_bps_y=50.0, borrow_bps_x=50.0,
        div_yield_bps_y=150.0, div_yield_bps_x=150.0
    ))
    print(res["summary"].tail()) # type: ignore