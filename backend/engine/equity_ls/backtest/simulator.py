# engines/equity_ls/backtest/simulator.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Callable, Dict, Optional

from .pnl import compute_pnl  # summary + per_ticker pnl$

TRADING_DAYS = 252

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _align(df: pd.DataFrame, idx: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    out = df.reindex(index=idx, columns=cols)
    return out.sort_index()

def _rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> pd.Series:
    """Return boolean Series marking rebalancing days for a pandas freq ('D','W-FRI','M','Q','BMS', etc.)."""
    if freq.upper() in ("D", "DAY", "DAILY"):
        return pd.Series(True, index=dates)
    # take period end according to frequency then mark those dates
    resamp = pd.Series(1, index=dates).resample(freq).last().reindex(dates).fillna(0)
    return resamp.astype(bool)

def _target_dollar_weights(
    scores: pd.Series,
    nav: float,
    prices: pd.Series,
    gross_target: float,
    per_name_cap: float,
) -> pd.Series:
    """Convert cross-sectional scores → dollar weights meeting gross and per-name caps."""
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or nav <= 0:
        return pd.Series(dtype=float)

    z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    w = z / (z.abs().sum() + 1e-12)           # unit gross
    w = w.clip(-per_name_cap, per_name_cap)   # per-name cap
    gross = w.abs().sum()
    if gross > 0:
        w = w * (gross_target / gross)
        weights = w * nav                         # convert to $
    return weights.reindex(prices.index).fillna(0.0)

def _dollar_to_shares(target_dollar: pd.Series, prices: pd.Series) -> pd.Series:
    sh = target_dollar / prices.replace(0, np.nan)
    return sh.fillna(0.0)

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def simulate_from_scores(
    prices: pd.DataFrame,                             # date x ticker (close)
    scores_func: Callable[[pd.DataFrame, pd.Timestamp], pd.Series],
    *,
    start_nav: float = 1_000_000.0,
    gross_target: float = 1.0,                        # target gross = 1x NAV
    per_name_cap: float = 0.05,                       # <=5% NAV per name
    rebalance: str = "W-FRI",                         # 'D','W-FRI','M','Q','BMS', etc.
    borrow_bps: float = 50.0,
    div_yield_bps: float = 150.0,
    fee_bps: float = 0.5,
    slippage_bps: float = 2.0,
    adv_usd: Optional[pd.DataFrame] = None,
    slippage_mode: str = "flat",
) -> Dict[str, pd.DataFrame]:
    """
    Drive a backtest from a user-supplied `scores_func`.
    The callback receives (prices_up_to_t, t) and returns a cross-sectional score Series at date t.
    """
    dates = prices.index
    tickers = list(prices.columns)
    prices = _align(prices, dates, tickers) # type: ignore

    # State
    nav = start_nav
    positions = pd.DataFrame(0.0, index=dates, columns=tickers)  # in shares
    trades = pd.DataFrame(0.0, index=dates, columns=tickers)     # shares traded per day
    nav_series = pd.Series(np.nan, index=dates)

    rb_mask = _rebalance_mask(dates, rebalance) # type: ignore

    last_pos = pd.Series(0.0, index=tickers)

    for t in dates:
        px_t = prices.loc[t]

        if rb_mask.loc[t]:
            # Build target from current NAV and latest scores
            scores = scores_func(prices.loc[:t], t).reindex(tickers).fillna(0.0)
            w_t = _target_dollar_weights(scores, nav, px_t, gross_target, per_name_cap)
            tgt_shares = _dollar_to_shares(w_t, px_t)

            # Trades = target - last
            day_trades = (tgt_shares - last_pos).fillna(0.0)
            trades.loc[t] = day_trades.values
            cur_pos = tgt_shares.copy()
        else:
            # No rebalance → carry last positions forward
            cur_pos = last_pos.copy()
            trades.loc[t] = 0.0

        positions.loc[t] = cur_pos.values
        last_pos = cur_pos

        # Rough NAV mark (EoD): we’ll recompute precise PnL later via compute_pnl
        nav_series.loc[t] = nav  # placeholder; final NAV can be built from summary returns

    # PnL & returns (uses lagged positions for price PnL and charges costs on trade dates)
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
        cash_equity=start_nav,
    )

    # Build NAV from net returns
    ret_net = summary["ret_net"].fillna(0.0)
    nav_curve = (1.0 + ret_net).cumprod() * start_nav

    out = {
        "positions": positions,
        "trades": trades,
        "summary": summary.assign(nav=nav_curve),
        "per_ticker_pnl": per_ticker,
        "nav": nav_curve.to_frame("nav"),
    }
    return out


def simulate_from_weight_table(
    prices: pd.DataFrame,                # date x ticker
    target_weights: pd.DataFrame,        # date x ticker (gross sum <= 1), rebalanced when row changes
    *,
    start_nav: float = 1_000_000.0,
    borrow_bps: float = 50.0,
    div_yield_bps: float = 150.0,
    fee_bps: float = 0.5,
    slippage_bps: float = 2.0,
    adv_usd: Optional[pd.DataFrame] = None,
    slippage_mode: str = "flat",
) -> Dict[str, pd.DataFrame]:
    """
    Alternative path: you already computed daily target weights (long + short).
    We convert them to shares, compute trades on weight changes, and run P&L.
    """
    dates = prices.index
    tickers = list(prices.columns)
    prices = _align(prices, dates, tickers) # type: ignore
    target_weights = _align(target_weights, dates, tickers).fillna(0.0) # type: ignore

    # Convert weights → $ → shares each day
    nav_curve = pd.Series(start_nav, index=dates)
    alloc = (target_weights * nav_curve.values.reshape(-1, 1)) # type: ignore
    positions = alloc / prices.replace(0, np.nan)
    positions = positions.fillna(0.0)

    # Trades = day-over-day change in shares
    trades = positions.diff().fillna(positions.iloc[0])

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
        cash_equity=start_nav,
    )
    nav = (1.0 + summary["ret_net"].fillna(0.0)).cumprod() * start_nav

    return {
        "positions": positions,
        "trades": trades,
        "summary": summary.assign(nav=nav),
        "per_ticker_pnl": per_ticker,
        "nav": nav.to_frame("nav"),
    }