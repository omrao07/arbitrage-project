# engines/equity_ls/backtest/pnl.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional

"""
Equity L/S PnL attribution

Inputs (daily frequency recommended):
- prices:   DataFrame[date x ticker] close prices
- positions:DataFrame[date x ticker] end-of-day positions in SHARES (positive=long, negative=short)
- trades:   DataFrame[date x ticker] signed SHARES traded that day (optional; for costs)
- borrow_bps:   float or Series[date] annualized borrow cost (shorts), in basis points
- div_yield_bps:float or Series[date] annualized dividend yield (long carry), in basis points
- fee_bps:      float broker commission per notional traded (both sides), in bps
- slippage_bps: float impact/slippage per notional traded (both sides), in bps
- adv_usd:      DataFrame[date x ticker] average daily $ volume (optional; if provided and
                slippage_mode='sqrt', we’ll use sqrt( participation ) impact model)
- factor_rets:  DataFrame[date x factor] optional style/market factor returns
- betas:        DataFrame[ticker x factor] optional constant betas; or
                Panel-like dict of DataFrame[date x factor] for time-varying betas

Outputs:
- df: DataFrame with columns:
    ['gross_exposure','net_exposure','turnover',
     'pnl$', 'price_pnl$', 'carry_pnl$', 'fees$', 'slippage$',
     'ret_gross','ret_net']
- per_ticker_pnl: DataFrame[date x ticker] daily total pnl$
- (optional) factor_attrib: DataFrame[date x factor] if factor inputs provided
"""

BPS = 1e-4
DAYS_IN_YEAR = 252


def _to_series(x, index):
    if x is None:
        return pd.Series(0.0, index=index)
    if np.isscalar(x):
        return pd.Series(float(x), index=index) # type: ignore
    s = pd.Series(x)
    return s.reindex(index).fillna(method="ffill").fillna(0.0) # type: ignore


def _dollar_notional(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return positions * prices


def _slippage_costs(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    adv_usd: Optional[pd.DataFrame] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    slippage_mode: str = "flat",  # "flat" or "sqrt"
    k_impact: float = 0.1,        # used if slippage_mode="sqrt"
) -> pd.Series:
    """Return daily $ cost series (fees + slippage)."""
    if trades is None or trades.empty:
        return pd.Series(0.0, index=prices.index)

    notional = (trades.abs() * prices).sum(axis=1)  # $ traded per day
    fees = notional * (fee_bps * BPS)

    if slippage_mode == "flat" or adv_usd is None:
        slip = notional * (slippage_bps * BPS)
    else:
        # sqrt( participation ) model per ticket; approximate at day level
        # participation = $traded / ($ADV across traded names)
        adv_day = adv_usd.reindex_like(prices).where(trades.abs() > 0).sum(axis=1)
        participation = (notional / adv_day).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0, 1)
        slip_bps = k_impact * np.sqrt(participation) * 1e4  # convert to bps
        slip = notional * (slip_bps * BPS)

    return (fees + slip).fillna(0.0)


def _carry_pnl(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    borrow_bps: pd.Series,
    div_yield_bps: pd.Series,
) -> pd.Series:
    """
    Carry PnL approximation:
      Longs: + div_yield * notional / 252
      Shorts: - borrow * |notional| / 252
    """
    notional = _dollar_notional(positions, prices)
    long_notional = notional.clip(lower=0.0).sum(axis=1)
    short_notional = (-notional.clip(upper=0.0)).sum(axis=1)

    long_carry = long_notional * (div_yield_bps * BPS / DAYS_IN_YEAR)
    short_carry = - short_notional * (borrow_bps * BPS / DAYS_IN_YEAR)
    return (long_carry + short_carry).fillna(0.0)


def compute_pnl(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    borrow_bps: float | pd.Series | None = 50.0,     # default 50 bps borrow
    div_yield_bps: float | pd.Series | None = 150.0, # default 1.5% div yield
    fee_bps: float = 0.5,
    slippage_bps: float = 2.0,
    adv_usd: Optional[pd.DataFrame] = None,
    slippage_mode: str = "flat",
    cash_equity: float = 0.0,  # starting cash (for net return denominator)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core P&L calculator (no factor attribution).
    """
    prices = prices.sort_index()
    positions = positions.reindex_like(prices).fillna(0.0)
    trades = trades.reindex_like(prices).fillna(0.0) if trades is not None else None

    # Daily simple returns
    rets = prices.pct_change().fillna(0.0)

    # Price PnL per ticker: start-of-day position * daily return * price (approx using close-to-close)
    # Using lagged positions so trades executed today affect tomorrow’s PnL.
    pos_lag = positions.shift(1).fillna(0.0)
    per_ticker_price_pnl = (pos_lag * rets * prices).fillna(0.0)

    # Carry PnL (borrow/dividends)
    idx = prices.index
    borrow_series = _to_series(borrow_bps, idx)
    div_series = _to_series(div_yield_bps, idx)
    carry = _carry_pnl(pos_lag, prices, borrow_series, div_series)

    # Costs
    costs = _slippage_costs(trades, prices, adv_usd, fee_bps, slippage_bps, slippage_mode) # type: ignore

    # Summaries
    price_pnl$ = per_ticker_price_pnl.sum(axis=1) # type: ignore
    pnl$ = price_pnl$ + carry - costs # type: ignore

    # Exposures & turnover
    notional = _dollar_notional(pos_lag, prices)
    gross = notional.abs().sum(axis=1)
    net = notional.sum(axis=1)
    turnover = 0.0 if trades is None else ( (trades.abs() * prices).sum(axis=1) / gross.replace(0, np.nan) ).fillna(0.0)

    # Returns
    equity_base = cash_equity + gross  # rough proxy if you don’t track NAV
    ret_gross = (price_pnl + carry) / equity_base.replace(0, np.nan) # type: ignore
    ret_net = pnl$ / equity_base.replace(0, np.nan) # type: ignore

    

    return df, (per_ticker_price_pnl.add(carry, axis=0).sub(costs, axis=0)) # type: ignore


def factor_attribution(
    per_ticker_price_pnl: pd.DataFrame,
    prices: pd.DataFrame,
    factor_rets: pd.DataFrame,
    betas: pd.DataFrame | Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Simple daily factor attribution:
      factor_contrib_t = sum_i( notional_{i,t-1} * sum_f( beta_{i,f} * ret_{f,t} ) )
    We map price PnL into factor-contributed vs residual (not returned here).
    """
    dates = per_ticker_price_pnl.index
    factors = factor_rets.columns
    # Notional at t-1 as weights
    notional_lag = prices.shift(1)  # multiply by pos elsewhere; we only need alignment

    out = pd.DataFrame(0.0, index=dates, columns=factors)

    if isinstance(betas, dict):
        # time-varying: betas[date] -> DataFrame[ticker x factor]
        for t in dates:
            b = betas.get(str(t)) or betas.get(t)  # tolerate str keys
            if b is None: 
                continue
            tickers = b.index.intersection(per_ticker_price_pnl.columns)
            pnl_t = per_ticker_price_pnl.loc[t, tickers]
            # convert pnl$ weights to proportion
            w = pnl_t / (abs(pnl_t).sum() + 1e-9)
            fr = factor_rets.loc[t, factors]
            contrib = (b.loc[tickers, :].mul(w, axis=0)).T.dot(fr)
            out.loc[t, :] = contrib.values
    else:
        # static betas[ticker x factor]
        common = betas.index.intersection(per_ticker_price_pnl.columns)
        B = betas.loc[common, factors] # type: ignore
        for t in dates:
            pnl_t = per_ticker_price_pnl.loc[t, common]
            w = pnl_t / (abs(pnl_t).sum() + 1e-9)
            fr = factor_rets.loc[t, factors]
            contrib = (B.mul(w, axis=0)).T.dot(fr)
            out.loc[t, :] = contrib.values

    return out.fillna(0.0)


# ---------- Convenience wrapper ----------

def run_equity_ls_pnl(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    borrow_bps: float | pd.Series | None = 50.0,
    div_yield_bps: float | pd.Series | None = 150.0,
    fee_bps: float = 0.5,
    slippage_bps: float = 2.0,
    adv_usd: Optional[pd.DataFrame] = None,
    slippage_mode: str = "flat",
    cash_equity: float = 0.0,
    factor_rets: Optional[pd.DataFrame] = None,
    betas: Optional[pd.DataFrame | Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end PnL with optional factor attribution.
    Returns dict with keys: summary, per_ticker_pnl, (optional) factor_attrib
    """
    summary, per_ticker = compute_pnl(
        prices=prices,
        positions=positions,
        trades=trades,
        borrow_bps=borrow_bps,
        div_yield_bps=div_yield_bps,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        adv_usd=adv_usd,
        slippage_mode=slippage_mode,
        cash_equity=cash_equity,
    )
    out = {"summary": summary, "per_ticker_pnl": per_ticker}
    if factor_rets is not None and betas is not None:
        out["factor_attrib"] = factor_attribution(
            per_ticker_price_pnl=per_ticker.add(0.0),  # price contribution embedded
            prices=prices,
            factor_rets=factor_rets,
            betas=betas,
        )
    return out