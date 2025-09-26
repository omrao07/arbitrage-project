# engines/equity_ls/signals/overnight_reversal.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal, Iterable

Method = Literal["overnight", "intraday"]  # which move to reverse

# ----------------------------- utils -----------------------------

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)

def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / (sd + 1e-12) if sd > 0 else s * 0.0

def _sector_demean(weights: pd.Series, sector_map: Optional[Dict[str, str]]) -> pd.Series:
    if not sector_map:
        return weights
    sec = pd.Series({t: sector_map.get(t, "Unknown") for t in weights.index})
    out = weights.copy().astype(float)
    for g, idx in sec.groupby(sec):
        tickers = idx.index
        out.loc[tickers] = out.loc[tickers] - out.loc[tickers].mean()
    return out

def _normalize_unit_gross(weights: pd.Series, unit_gross: float) -> pd.Series:
    gross = weights.abs().sum()
    return weights * (unit_gross / (gross + 1e-12))

# ----------------------------- core -----------------------------

def _overnight_returns(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    r_ov_t = (Open_t / Close_{t-1}) - 1
    Requires aligned date index and same columns (tickers).
    """
    return (open_ / close.shift(1)) - 1.0

def _intraday_returns(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    r_intra_t = (Close_t / Open_t) - 1
    """
    return (close / open_) - 1.0

def build_signal(
    *,
    open_prices: pd.DataFrame,           # date x ticker (open)
    close_prices: pd.DataFrame,          # date x ticker (close)
    method: Method = "overnight",        # reverse which move
    delay: int = 1,                      # use info up to t-delay (usually 1)
    winsor_p: float = 0.01,
    vol_scale_lb: int = 20,              # scale by recent vol of the chosen leg
    sector_map: Optional[Dict[str, str]] = None,
    long_cap: float = 0.05,              # per-name weight cap
    short_cap: float = -0.05,
    unit_gross: float = 1.0,
    combine_with: Optional[Iterable[pd.Series]] = None,  # optional additional z-scored signals to sum (e.g., quality)
) -> pd.Series:
    """
    Cross-sectional OVERNIGHT (or INTRADAY) reversal:
      - Score = - last move (prior-day)  [negative sign for reversal]
      - Optional vol scaling (divide by rolling stdev of that move)
      - Winsorize + z-score; optional sector-neutral demeaning
      - Clip and normalize to unit gross
    Returns weights at the last date available.
    """
    o = open_prices.sort_index()
    c = close_prices.sort_index()
    # align
    common_cols = [t for t in c.columns if t in o.columns]
    o = o.loc[:, common_cols]
    c = c.loc[:, common_cols]

    # pick move series
    if method == "overnight":
        mv = _overnight_returns(o, c)
    elif method == "intraday":
        mv = _intraday_returns(o, c)
    else:
        raise ValueError(f"Unknown method: {method}")

    # need enough rows for vol window + delay
    need = max(3, vol_scale_lb) + delay + 1
    if mv.shape[0] < need:
        return pd.Series(dtype=float)

    # prior-day move (t - delay)
    last_move = mv.shift(delay).iloc[-1]

    # recent volatility of chosen leg for scaling (per-name)
    vol = mv.rolling(vol_scale_lb).std().shift(delay).iloc[-1]
    vol = vol.replace(0, np.nan)

    # reversal score: negative of prior move (losers -> positive score; winners -> negative)
    raw = (-last_move / vol).replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna()
    if raw.empty:
        return pd.Series(dtype=float)

    raw = _winsorize(raw, p=winsor_p)
    z = _zscore(raw)

    # Optionally blend with other (already standardized) signals
    if combine_with:
        for s in combine_with:
            if isinstance(s, pd.Series) and not s.empty:
                z = z.add(s.reindex(z.index).fillna(0.0), fill_value=0.0)

    # Optional sector neutrality
    z = _sector_demean(z, sector_map)

    # Cap and normalize
    w = z.clip(lower=short_cap, upper=long_cap)
    w = _normalize_unit_gross(w, unit_gross=unit_gross)

    return w.sort_values(ascending=False)

# ----------------------------- convenience: simple backtest helper -----------------------------

def build_daily_weights_series(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    *,
    method: Method = "overnight",
    delay: int = 1,
    winsor_p: float = 0.01,
    vol_scale_lb: int = 20,
    sector_map_ts: Optional[Dict[str, str]] = None,
    long_cap: float = 0.05,
    short_cap: float = -0.05,
    unit_gross: float = 1.0,
) -> pd.DataFrame:
    """
    Vectorized helper: returns a panel of daily weights (date x ticker),
    sized as if you rebalance each day using t-delay information.
    Good for quick backtests of the signal alone.

    Note: sector neutrality uses sector_map_ts at each step (static mapping).
    """
    o = open_prices.sort_index()
    c = close_prices.sort_index()
    cols = [t for t in c.columns if t in o.columns]
    o = o.loc[:, cols]; c = c.loc[:, cols]

    if method == "overnight":
        mv = _overnight_returns(o, c)
    else:
        mv = _intraday_returns(o, c)

    need = max(3, vol_scale_lb) + delay + 1
    if mv.shape[0] < need:
        return pd.DataFrame(index=o.index, columns=cols, dtype=float)

    vol = mv.rolling(vol_scale_lb).std().shift(delay)
    last_mv = mv.shift(delay)

    frames = []
    for t in mv.index[need-1:]:
        raw = (-(last_mv.loc[t] / vol.loc[t])).replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            frames.append(pd.Series(0.0, index=cols, name=t))
            continue
        raw = _winsorize(raw, p=winsor_p)
        z = _zscore(raw)
        if sector_map_ts:
            z = _sector_demean(z, sector_map_ts)
        w = z.clip(lower=short_cap, upper=long_cap)
        w = _normalize_unit_gross(w, unit_gross)
        frames.append(w.reindex(cols).fillna(0.0).rename(t))

    return pd.DataFrame(frames).reindex(mv.index).fillna(0.0)