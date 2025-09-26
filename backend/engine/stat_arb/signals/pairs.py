# engines/stat_arb/signals/pairs.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal, List
from dataclasses import dataclass
import statsmodels.api as sm # type: ignore

BetaMethod = Literal["rolling_ols", "expanding_ols", "static_ols"]

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def _align(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = y.index.intersection(x.index) # type: ignore
    return y.reindex(idx).astype(float), x.reindex(idx).astype(float)

def _rolling_beta(y: pd.Series, x: pd.Series, method: BetaMethod = "rolling_ols", lookback: int = 120) -> pd.Series:
    y, x = _align(y, x)
    if method == "static_ols":
        X = sm.add_constant(x.values)
        beta = sm.OLS(y.values, X).fit().params[0]
        return pd.Series(beta, index=y.index)
    betas = pd.Series(index=y.index, dtype=float)
    if method == "expanding_ols":
        for i in range(20, len(x)+1):
            yw = y.iloc[:i]; xw = x.iloc[:i]
            X = sm.add_constant(xw)
            beta = sm.OLS(yw, X).fit().params[0]
            betas.iloc[i-1] = beta
        betas.iloc[:20] = betas.iloc[20]
        return betas.ffill()
    # rolling_ols
    lb = max(20, lookback)
    for i in range(lb, len(x)+1):
        yw = y.iloc[i-lb:i]; xw = x.iloc[i-lb:i]
        X = sm.add_constant(xw)
        beta = sm.OLS(yw, X).fit().params[0]
        betas.iloc[i-1] = beta
    betas.iloc[:lb] = betas.iloc[lb]
    return betas.ffill()

def _zscore(s: pd.Series, lb: int) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    return (s - mu) / (sd + 1e-12)

# -------------------------------------------------------------------
# Cointegration / correlation screens
# -------------------------------------------------------------------

def rolling_corr(y: pd.Series, x: pd.Series, lookback: int = 60) -> pd.Series:
    y, x = _align(y, x)
    return y.pct_change().rolling(lookback).corr(x.pct_change())

def engle_granger(y: pd.Series, x: pd.Series, lookback: int = 252, signif: float = 0.05) -> Tuple[float, bool]:
    """
    Engle-Granger two-step cointegration test.
    Returns (adf_pvalue, is_cointegrated).
    """
    y, x = _align(y, x)
    if len(y) < lookback:
        return np.nan, False
    y = y.iloc[-lookback:]; x = x.iloc[-lookback:]
    X = sm.add_constant(x)
    resid = y - sm.OLS(y, X).fit().predict(X)
    adf_res = sm.tsa.adfuller(resid.dropna(), maxlag=1, autolag="AIC")
    pval = adf_res[1]
    return pval, pval < signif

# -------------------------------------------------------------------
# Pair diagnostics
# -------------------------------------------------------------------

@dataclass
class PairDiagnostics:
    beta: float
    spread: float
    z: float
    spread_vol: float
    half_life: float

def compute_pair_diagnostics(
    y: pd.Series,
    x: pd.Series,
    *,
    beta_method: BetaMethod = "rolling_ols",
    beta_lookback: int = 120,
    z_lookback: int = 60,
) -> PairDiagnostics:
    y, x = _align(y, x)
    beta = _rolling_beta(y, x, beta_method, beta_lookback).iloc[-1]
    spread = y - beta * x
    spread_vol = spread.pct_change().std()
    z = _zscore(spread, z_lookback).iloc[-1]

    # half-life via AR(1) fit on spread
    spr = spread.dropna().diff().dropna()
    lag = spread.shift(1).dropna().loc[spr.index]
    phi = np.polyfit(lag.values, spr.values, 1)[0]
    half_life = -np.log(2) / np.log(abs(1 + phi)) if abs(1 + phi) > 0 else np.inf

    return PairDiagnostics(
        beta=float(beta),
        spread=float(spread.iloc[-1]),
        z=float(z if z == z else 0.0),
        spread_vol=float(spread_vol),
        half_life=float(half_life),
    )

# -------------------------------------------------------------------
# Signal generation
# -------------------------------------------------------------------

def generate_pair_signal(
    y: pd.Series,
    x: pd.Series,
    *,
    entry_z: float = 1.0,
    exit_z: float = 0.25,
    beta_method: BetaMethod = "rolling_ols",
    beta_lookback: int = 120,
    z_lookback: int = 60,
    max_units: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Generate units for a single pair at last date.
    Returns (units, beta, z).
      +1 units: long Y / short β*X
      -1 units: short Y / long β*X
       0 units: flat
    """
    y, x = _align(y, x)
    beta_series = _rolling_beta(y, x, beta_method, beta_lookback)
    beta = float(beta_series.iloc[-1])
    spread = y - beta * x
    z = _zscore(spread, z_lookback).iloc[-1]
    z_val = float(z if z == z else 0.0)

    if abs(z_val) < exit_z:
        u = 0.0
    elif z_val <= -entry_z:
        u = +1.0
    elif z_val >= +entry_z:
        u = -1.0
    else:
        u = 0.0

    return float(np.clip(u, -max_units, max_units)), beta, z_val

# -------------------------------------------------------------------
# Portfolio selection
# -------------------------------------------------------------------

def select_pairs(
    prices: pd.DataFrame,
    candidates: List[Tuple[str, str]],
    *,
    corr_lb: int = 60,
    coint_lb: int = 252,
    signif: float = 0.05,
    min_corr: float = 0.5,
) -> List[Tuple[str, str]]:
    """
    From a list of candidate ticker pairs, filter those that
    - have rolling correlation > min_corr
    - are cointegrated at significance level
    """
    out: List[Tuple[str, str]] = []
    for y_name, x_name in candidates:
        y = prices[y_name]; x = prices[x_name]
        corr = rolling_corr(y, x, corr_lb).iloc[-1]
        pval, is_coint = engle_granger(y, x, lookback=coint_lb, signif=signif)
        if corr >= min_corr and is_coint:
            out.append((y_name, x_name))
    return out