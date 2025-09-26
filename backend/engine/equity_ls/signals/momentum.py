# engines/equity_ls/signals/momentum.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Tuple, Literal

Method = Literal["cross", "timeseries"]

# ----------------------------- utils -----------------------------

def _pct_change(prices: pd.DataFrame, lb: int) -> pd.DataFrame:
    # close-to-close simple returns
    return prices.pct_change(lb)

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

def _normalize_unit_gross(weights: pd.Series) -> pd.Series:
    gross = weights.abs().sum()
    return weights / (gross + 1e-12)

# ----------------------------- core builders -----------------------------

def build_cross_sectional_signal(
    prices: pd.DataFrame,                   # date x ticker (close)
    *,
    lookbacks: Iterable[int] = (63, 126, 252),
    vol_scale_lookback: int = 20,           # divide momentum by recent vol
    winsor_p: float = 0.01,
    sector_map: Optional[Dict[str, str]] = None,
    long_cap: float = 0.05,                 # per-name cap as fraction of gross
    short_cap: float = -0.05,
    delay: int = 1,                         # use info as of t-1 to avoid lookahead
    unit_gross: float = 1.0,                # final gross exposure
) -> pd.Series:
    """
    Cross-sectional momentum signal (ranked long/short across names).
    - Momentum score = average of multi-horizon returns (t-delay)
    - Volatility adjust: divide by recent stdev to favor stable trends
    - Winsorization for robustness
    - Optional sector-neutral demeaning
    Returns: weights at the last available date, clipped and normalized to |w| sum == unit_gross.
    """
    px = prices.copy().sort_index()
    if px.shape[0] < max(max(lookbacks), vol_scale_lookback) + delay + 1:
        return pd.Series(dtype=float)

    # compute horizon returns at t-delay
    scores = []
    for lb in lookbacks:
        r = _pct_change(px, lb).shift(delay)  # use info up to t-delay
        scores.append(r.iloc[-1])
    mom = pd.concat(scores, axis=1).mean(axis=1).dropna()  # index = tickers

    # recent vol (stdev of daily returns) to normalize
    vol = px.pct_change().rolling(vol_scale_lookback).std().shift(delay).iloc[-1]
    vol = vol.replace(0, np.nan)

    raw = (mom / vol).replace([np.inf, -np.inf], np.nan).dropna()
    raw = _winsorize(raw, p=winsor_p)

    # cross-sectional z-score -> weights
    z = _zscore(raw)

    # optional sector-neutral
    z = _sector_demean(z, sector_map)

    # cap per name, then normalize to unit gross
    w = z.clip(lower=short_cap, upper=long_cap)
    w = _normalize_unit_gross(w) * unit_gross
    return w.sort_values(ascending=False)


def build_timeseries_signal(
    prices: pd.DataFrame,             # date x ticker
    *,
    lookback: int = 126,              # e.g., 6 months
    vol_target: float = 0.20,         # annualized per-asset target (trend sizing)
    vol_lb: int = 20,
    delay: int = 1,
    unit_gross: float = 1.0,
    cap_per_name: float = 0.05,
) -> pd.Series:
    """
    Time-series momentum (per-name trend):
      sign( price_{t} / price_{t-LB} - 1 ) * (vol_target / recent_vol)
    - Positive trend → long; negative trend → short
    - Vol-based sizing equalizes risk across names
    Returns: weights at the last date normalized to |w| sum == unit_gross.
    """
    px = prices.copy().sort_index()
    if px.shape[0] < max(lookback, vol_lb) + delay + 1:
        return pd.Series(dtype=float)

    ret_lb = (px / px.shift(lookback) - 1.0).shift(delay).iloc[-1]
    dirn = np.sign(ret_lb).replace(0, 0.0) # type: ignore

    daily_vol = px.pct_change().rolling(vol_lb).std().shift(delay).iloc[-1]
    ann_vol = daily_vol * np.sqrt(252)
    sigma = ann_vol.replace(0, np.nan)

    raw = (dirn * (vol_target / sigma)).replace([np.inf, -np.inf], np.nan).dropna()

    # cap per name
    raw = raw.clip(lower=-cap_per_name, upper=cap_per_name)

    # normalize to unit gross
    w = _normalize_unit_gross(raw) * unit_gross
    return w.sort_values(ascending=False)

# ----------------------------- public API -----------------------------

def build_signal(
    prices: pd.DataFrame,
    *,
    method: Method = "cross",
    lookbacks: Iterable[int] = (63, 126, 252),
    lookback_ts: int = 126,
    vol_scale_lookback: int = 20,
    vol_target_ts: float = 0.20,
    vol_lb_ts: int = 20,
    winsor_p: float = 0.01,
    sector_map: Optional[Dict[str, str]] = None,
    long_cap: float = 0.05,
    short_cap: float = -0.05,
    cap_per_name_ts: float = 0.05,
    delay: int = 1,
    unit_gross: float = 1.0,
) -> pd.Series:
    """
    Unified entrypoint:
      method="cross"      -> cross-sectional momentum (ranked long/short)
      method="timeseries" -> per-name trend following (risk-equalized)
    """
    if method == "cross":
        return build_cross_sectional_signal(
            prices,
            lookbacks=lookbacks,
            vol_scale_lookback=vol_scale_lookback,
            winsor_p=winsor_p,
            sector_map=sector_map,
            long_cap=long_cap,
            short_cap=short_cap,
            delay=delay,
            unit_gross=unit_gross,
        )
    elif method == "timeseries":
        return build_timeseries_signal(
            prices,
            lookback=lookback_ts,
            vol_target=vol_target_ts,
            vol_lb=vol_lb_ts,
            delay=delay,
            unit_gross=unit_gross,
            cap_per_name=cap_per_name_ts,
        )
    else:
        raise ValueError(f"Unknown method: {method}")