#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py
-----------
Feature engineering for strategies and models.

Includes:
- Technical indicators: momentum, moving averages, RSI, volatility, z-score
- Rolling stats: skew, kurtosis, autocorrelation
- Cross-asset spreads and ratios
- Fundamental-style ratios (valuation, leverage, margins)
- Feature pipelines: apply many functions and merge

Usage:
    import features as F

    df = pd.DataFrame({"px": ..., "vol": ..., "ret": ...})
    f = {}
    f["mom_20"] = F.momentum(df["px"], window=20)
    f["vol_20"] = F.volatility(df["ret"], window=20)
    out = F.build_features(df, {"mom": {"func": F.momentum, "col": "px", "window": 20}})
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable, Any, Optional


# =============================================================================
# Technical indicators
# =============================================================================

def momentum(series: pd.Series, window: int = 20) -> pd.Series:
    """Momentum = px / px.shift(window) - 1"""
    return series.pct_change(periods=window)

def moving_average(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window).mean()

def volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling volatility (std)."""
    return returns.rolling(window).std(ddof=0)

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score."""
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std(ddof=0)

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)

def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})

# =============================================================================
# Statistical features
# =============================================================================

def rolling_skew(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).skew()

def rolling_kurt(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).kurt()

def autocorr(series: pd.Series, lag: int = 1, window: int = 20) -> pd.Series:
    """Rolling autocorrelation with lag."""
    return series.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag), raw=False)

# =============================================================================
# Cross-asset spreads / ratios
# =============================================================================

def spread(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Spread between two series."""
    return s1 - s2

def ratio(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Ratio between two series."""
    return s1 / (s2.replace(0, np.nan))

def beta(asset: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    """Rolling beta of asset returns vs benchmark returns."""
    cov = asset.rolling(window).cov(benchmark)
    var = benchmark.rolling(window).var()
    return cov / (var + 1e-12)

# =============================================================================
# Fundamental-style ratios (works if fundamentals merged into df)
# =============================================================================

def pe_ratio(price: pd.Series, eps: pd.Series) -> pd.Series:
    return price / (eps.replace(0, np.nan))

def pb_ratio(price: pd.Series, bvps: pd.Series) -> pd.Series:
    return price / (bvps.replace(0, np.nan))

def ps_ratio(price: pd.Series, sps: pd.Series) -> pd.Series:
    return price / (sps.replace(0, np.nan))

def ev_ebitda(ev: pd.Series, ebitda: pd.Series) -> pd.Series:
    return ev / (ebitda.replace(0, np.nan))

# =============================================================================
# Feature pipeline utilities
# =============================================================================

def build_features(df: pd.DataFrame, spec: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Build multiple features at once.

    spec = {
        "mom20": {"func": momentum, "col": "px", "window": 20},
        "vol20": {"func": volatility, "col": "ret", "window": 20},
    }
    """
    out = pd.DataFrame(index=df.index)
    for name, cfg in spec.items():
        func: Callable = cfg["func"]
        col = cfg["col"]
        params = {k: v for k, v in cfg.items() if k not in {"func", "col"}}
        out[name] = func(df[col], **params)
    return out

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-score normalization."""
    return (df - df.mean()) / df.std(ddof=0)

def winsorize(series: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Clip extremes by quantiles."""
    lo, hi = series.quantile(limits[0]), series.quantile(limits[1])
    return series.clip(lower=lo, upper=hi)

# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    # quick smoke test with random walk
    rng = np.random.default_rng(0)
    px = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
    ret = px.pct_change().fillna(0)

    df = pd.DataFrame({"px": px, "ret": ret})
    f = build_features(df, {
        "mom20": {"func": momentum, "col": "px", "window": 20},
        "vol20": {"func": volatility, "col": "ret", "window": 20},
        "rsi14": {"func": rsi, "col": "px", "window": 14},
    })

    print(f.tail())