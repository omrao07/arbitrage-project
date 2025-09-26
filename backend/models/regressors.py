#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regressors.py
-------------
Quant-friendly regression utilities.

Includes
- OLS with t-stats, p-values, R^2, adj-R^2, AIC/BIC; optional add-const
- HAC (Newey–West) covariance for time-series OLS
- Huber robust regression (statsmodels if available; else IRLS fallback)
- Ridge / Lasso via scikit-learn if available
- Rolling / expanding OLS (vectorized where possible)
- Fama–MacBeth cross-sectional regression (factor premia & t-stats)
- Design helpers: add_constant, zscore, winsorize, one_hot

Dependencies: numpy, pandas. Optional: statsmodels, scikit-learn, scipy (for p-values).
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

# Optional libs
try:
    import statsmodels.api as sm  # type: ignore
    _HAS_SM = True
except Exception:
    _HAS_SM = False

try:
    from sklearn.linear_model import Ridge as _Ridge, Lasso as _Lasso  # type: ignore
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    from scipy import stats as _sstats  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =============================================================================
# Utilities
# =============================================================================

def add_constant(X: pd.DataFrame | np.ndarray, name: str = "const") -> pd.DataFrame:
    Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    return pd.concat([pd.Series(1.0, index=Xdf.index, name=name), Xdf], axis=1)

def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

def winsorize(x: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lower=lo, upper=hi)

def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    d = pd.get_dummies(series.astype("category"), prefix=prefix)
    # Drop one level to avoid dummy trap if you also add a constant
    return d.iloc[:, 1:] if d.shape[1] > 0 else d


# =============================================================================
# Core OLS
# =============================================================================

@dataclass
class OLSResult:
    coef: pd.Series
    se: pd.Series
    t: pd.Series
    p: pd.Series
    r2: float
    r2_adj: float
    aic: float
    bic: float
    resid: pd.Series
    fitted: pd.Series
    df_model: int
    df_resid: int
    cov: pd.DataFrame

def _p_from_t(t_vals: np.ndarray, df: int) -> np.ndarray:
    if _HAS_SCIPY:
        return 2 * (1 - _sstats.t.cdf(np.abs(t_vals), df))
    # Normal approx
    zcdf = 0.5 * (1 + np.erf(np.abs(t_vals) / np.sqrt(2)))#type:ignore
    return 2 * (1 - zcdf)

def ols(y: pd.Series, X: pd.DataFrame, add_const: bool = True) -> OLSResult:
    """
    Closed-form OLS with classic (iid) covariance.
    """
    if add_const:
        X = add_constant(X)
    # Align & drop NaNs
    df = pd.concat([y, X], axis=1).dropna()
    yv = df.iloc[:, 0].values.astype(float)
    Xv = df.iloc[:, 1:].values.astype(float)
    cols = df.columns[1:]

    XtX = Xv.T @ Xv
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xv.T @ yv)
    fitted = Xv @ beta
    resid = yv - fitted

    n, k = Xv.shape
    df_resid = max(1, n - k)
    sigma2 = float(resid.T @ resid) / df_resid
    cov = XtX_inv * sigma2
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = beta / (se + 1e-18)
    pvals = _p_from_t(tvals, df_resid)

    # Goodness-of-fit
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    r2_adj = 1 - (1 - r2) * (n - 1) / max(1, n - k)

    # IC
    ll = -0.5 * n * (math.log(2 * math.pi) + 1) - 0.5 * n * math.log(max(1e-18, sigma2))
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * math.log(max(1, n))

    return OLSResult(
        coef=pd.Series(beta, index=cols),
        se=pd.Series(se, index=cols),
        t=pd.Series(tvals, index=cols),
        p=pd.Series(pvals, index=cols),
        r2=r2,
        r2_adj=r2_adj,
        aic=aic,
        bic=bic,
        resid=pd.Series(resid, index=df.index),
        fitted=pd.Series(fitted, index=df.index),
        df_model=k,
        df_resid=df_resid,
        cov=pd.DataFrame(cov, index=cols, columns=cols),
    )


# =============================================================================
# HAC (Newey–West) covariance for time-series OLS
# =============================================================================

def ols_hac(y: pd.Series, X: pd.DataFrame, lags: int = 5, add_const: bool = True) -> OLSResult:
    """
    OLS but with Newey–West HAC standard errors (Bartlett kernel).
    """
    res = ols(y, X, add_const=add_const)
    df = pd.concat([y, X if not add_const else add_constant(X)], axis=1).dropna()
    yv = df.iloc[:, 0].values.astype(float)
    Xv = df.iloc[:, 1:].values.astype(float)
    cols = df.columns[1:]

    n, k = Xv.shape
    u = (yv - Xv @ res.coef.values)

    # Meat of sandwich
    S = np.zeros((k, k))
    for h in range(-lags, lags + 1):
        w = 1 - abs(h) / (lags + 1)  # Bartlett
        if w <= 0:
            continue
        if h >= 0:
            X1 = Xv[h:, :]
            X2 = Xv[: n - h, :]
            u1 = u[h:]
            u2 = u[: n - h]
        else:
            X1 = Xv[: n + h, :]
            X2 = Xv[-h:, :]
            u1 = u[: n + h]
            u2 = u[-h:]
        S += w * (X1 * u1[:, None]).T @ (X2 * u2[:, None])

    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    cov_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.maximum(np.diag(cov_hac), 0.0))
    tvals = res.coef.values / (se_hac + 1e-18)#type:ignore
    pvals = _p_from_t(tvals, n - k)

    # Replace in result
    res.se = pd.Series(se_hac, index=cols)
    res.t = pd.Series(tvals, index=cols)
    res.p = pd.Series(pvals, index=cols)
    res.cov = pd.DataFrame(cov_hac, index=cols, columns=cols)
    return res


# =============================================================================
# Robust / Regularized regressions
# =============================================================================

def huber(y: pd.Series, X: pd.DataFrame, add_const: bool = True, c: float = 1.345, max_iter: int = 100) -> OLSResult:
    """
    Huber robust regression. Uses statsmodels if available; else IRLS fallback.
    """
    if add_const:
        X = add_constant(X)
    df = pd.concat([y, X], axis=1).dropna()
    yv = df.iloc[:, 0].values.astype(float)
    Xv = df.iloc[:, 1:].values.astype(float)
    cols = df.columns[1:]

    if _HAS_SM:
        model = sm.RLM(yv, Xv, M=sm.robust.norms.HuberT(t=c))
        fit = model.fit(maxiter=max_iter)
        beta = fit.params
        se = fit.bse
        resid = fit.resid
        fitted = yv - resid
        # t/p values:
        tvals = beta / (se + 1e-18)
        pvals = _p_from_t(tvals, max(1, len(yv) - Xv.shape[1]))
        r2 = 1 - float((resid**2).sum()) / max(1e-18, ((yv - yv.mean())**2).sum())
        return OLSResult(
            coef=pd.Series(beta, index=cols),
            se=pd.Series(se, index=cols),
            t=pd.Series(tvals, index=cols),
            p=pd.Series(pvals, index=cols),
            r2=r2, r2_adj=np.nan, aic=np.nan, bic=np.nan,
            resid=pd.Series(resid, index=df.index),
            fitted=pd.Series(fitted, index=df.index),
            df_model=Xv.shape[1], df_resid=max(1, len(yv) - Xv.shape[1]),
            cov=pd.DataFrame(fit.cov_params(), index=cols, columns=cols),
        )

    # IRLS fallback
    beta = np.linalg.pinv(Xv) @ yv
    for _ in range(max_iter):
        r = yv - Xv @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12  # MAD
        w = np.ones_like(r)
        z = np.abs(r) / (c * s)
        w[z > 1] = (c * s) / (np.abs(r[z > 1]) + 1e-12)
        W = np.diag(w)
        beta_new = np.linalg.pinv(Xv.T @ W @ Xv) @ (Xv.T @ W @ yv)
        if np.linalg.norm(beta_new - beta) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    fitted = Xv @ beta
    resid = yv - fitted
    # classic cov on weighted X
    XtWX_inv = np.linalg.pinv(Xv.T @ W @ Xv)
    sigma2 = float((w * resid**2).sum() / max(1, len(yv) - Xv.shape[1]))
    cov = XtWX_inv * sigma2
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = beta / (se + 1e-18)
    pvals = _p_from_t(tvals, max(1, len(yv) - Xv.shape[1]))
    r2 = 1 - float((resid**2).sum()) / max(1e-18, ((yv - yv.mean())**2).sum())

    return OLSResult(
        coef=pd.Series(beta, index=cols),
        se=pd.Series(se, index=cols),
        t=pd.Series(tvals, index=cols),
        p=pd.Series(pvals, index=cols),
        r2=r2, r2_adj=np.nan, aic=np.nan, bic=np.nan,
        resid=pd.Series(resid, index=df.index),
        fitted=pd.Series(fitted, index=df.index),
        df_model=Xv.shape[1], df_resid=max(1, len(yv) - Xv.shape[1]),
        cov=pd.DataFrame(cov, index=cols, columns=cols),
    )


def ridge(y: pd.Series, X: pd.DataFrame, alpha: float = 1.0, add_const: bool = True) -> OLSResult:
    if add_const:
        X = add_constant(X)
    df = pd.concat([y, X], axis=1).dropna()
    yv = df.iloc[:, 0].values.astype(float)
    Xv = df.iloc[:, 1:].values.astype(float)
    cols = df.columns[1:]

    if _HAS_SK:
        model = _Ridge(alpha=alpha, fit_intercept=False)
        model.fit(Xv, yv)
        beta = model.coef_
    else:
        # closed form (X'X + αI)^-1 X'y
        k = Xv.shape[1]
        beta = np.linalg.solve(Xv.T @ Xv + alpha * np.eye(k), Xv.T @ yv)

    fitted = Xv @ beta
    resid = yv - fitted
    n, k = Xv.shape
    sigma2 = float(resid.T @ resid) / max(1, n - k)
    XtX_inv = np.linalg.pinv(Xv.T @ Xv + alpha * np.eye(k))
    cov = XtX_inv * sigma2
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = beta / (se + 1e-18)
    pvals = _p_from_t(tvals, max(1, n - k))
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / max(1e-18, ss_tot)

    return OLSResult(
        coef=pd.Series(beta, index=cols),
        se=pd.Series(se, index=cols),
        t=pd.Series(tvals, index=cols),
        p=pd.Series(pvals, index=cols),
        r2=r2, r2_adj=np.nan, aic=np.nan, bic=np.nan,
        resid=pd.Series(resid, index=df.index),
        fitted=pd.Series(fitted, index=df.index),
        df_model=k, df_resid=max(1, n - k),
        cov=pd.DataFrame(cov, index=cols, columns=cols),
    )


def lasso(y: pd.Series, X: pd.DataFrame, alpha: float = 0.001, add_const: bool = True, max_iter: int = 10000) -> OLSResult:
    if not _HAS_SK:
        raise RuntimeError("Lasso requires scikit-learn. `pip install scikit-learn`")
    if add_const:
        X = add_constant(X)
    df = pd.concat([y, X], axis=1).dropna()
    yv = df.iloc[:, 0].values.astype(float)
    Xv = df.iloc[:, 1:].values.astype(float)
    cols = df.columns[1:]

    model = _Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
    model.fit(Xv, yv)
    beta = model.coef_
    fitted = model.predict(Xv)
    resid = yv - fitted

    # naive SE via OLS on selected set
    sel = np.where(np.abs(beta) > 0)[0]
    if sel.size > 0:
        o = ols(pd.Series(yv, index=df.index), pd.DataFrame(Xv[:, sel], index=df.index), add_const=False)
        se = np.zeros_like(beta); se[sel] = o.se.values
        cov = np.zeros((beta.size, beta.size)); cov[np.ix_(sel, sel)] = o.cov.values
    else:
        se = np.zeros_like(beta); cov = np.zeros((beta.size, beta.size))

    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / max(1e-18, ss_tot)
    tvals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = _p_from_t(tvals, max(1, len(yv) - len(sel)))

    return OLSResult(
        coef=pd.Series(beta, index=cols),
        se=pd.Series(se, index=cols),
        t=pd.Series(tvals, index=cols),
        p=pd.Series(pvals, index=cols),
        r2=r2, r2_adj=np.nan, aic=np.nan, bic=np.nan,
        resid=pd.Series(resid, index=df.index),
        fitted=pd.Series(fitted, index=df.index),
        df_model=len(cols), df_resid=max(1, len(yv) - len(cols)),
        cov=pd.DataFrame(cov, index=cols, columns=cols),
    )


# =============================================================================
# Rolling / expanding OLS
# =============================================================================

def rolling_ols(y: pd.Series, X: pd.DataFrame, window: int, add_const: bool = True, expanding: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame of coefficients aligned to window end.
    """
    if add_const:
        X = add_constant(X)
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    cols = [c for c in df.columns if c != "__y__"]
    out = []

    idx = df.index
    for i in range(window, len(df) + 1):
        i0 = 0 if expanding else i - window
        sl = df.iloc[i0:i]
        res = ols(sl["__y__"], sl[cols], add_const=False)
        out.append(pd.Series(res.coef, name=idx[i - 1]))

    return pd.DataFrame(out).reindex(idx)  # align to original index (leading NaN)


# =============================================================================
# Fama–MacBeth cross-sectional regression
# =============================================================================

@dataclass
class FMResult:
    lambdas: pd.Series      # time-series mean premia (per factor)
    t: pd.Series            # FM t-stats
    n_periods: int
    by_period: pd.DataFrame # raw per-period lambdas
    intercept: Optional[float] = None
    intercept_t: Optional[float] = None

def fama_macbeth(
    df: pd.DataFrame,
    ret_col: str,
    factor_cols: List[str],
    group_col: str = "date",
    add_const: bool = True,
) -> FMResult:
    """
    Expects a long panel: each row is an asset-period observation with realized return and factor exposures.
    Example columns: date, asset, ret, beta_mkt, size, value, quality, ...
    """
    periods = []
    betas = []
    for g, gdf in df.groupby(group_col):
        y = gdf[ret_col]
        X = gdf[factor_cols]
        if add_const:
            X = add_constant(X)
        # require >= factors + const + buffer
        if len(gdf) < X.shape[1] + 2:
            continue
        res = ols(y, X, add_const=False)
        s = res.coef.rename(g)
        betas.append(s)
        periods.append(g)

    if not betas:
        raise ValueError("No valid periods for FM regression.")
    per = pd.DataFrame(betas).sort_index()
    # separate intercept if present
    if add_const and "const" in per.columns:
        intercept_series = per["const"]
        per = per.drop(columns=["const"])
        intercept = float(intercept_series.mean())
        it = float(intercept_series.mean() / (intercept_series.std(ddof=1) / math.sqrt(len(intercept_series))))
    else:
        intercept = None; it = None

    lambdas = per.mean(axis=0)
    se = per.std(ddof=1) / math.sqrt(per.shape[0])
    tvals = lambdas / (se.replace(0, np.nan))

    return FMResult(
        lambdas=lambdas,
        t=tvals,
        n_periods=per.shape[0],
        by_period=per,
        intercept=intercept,
        intercept_t=it,
    )


# =============================================================================
# Demo / self-test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Synthetic time-series OLS
    n = 1000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eps = rng.normal(scale=0.5, size=n)
    y = 1.5 + 2.0 * x1 - 0.7 * x2 + eps

    yS = pd.Series(y)
    XS = pd.DataFrame({"x1": x1, "x2": x2})
    res = ols(yS, XS)
    print("OLS coef:\n", res.coef.round(3))
    res_hac = ols_hac(yS, XS, lags=5)
    print("HAC t-stats:\n", res_hac.t.round(3))

    # Rolling OLS
    roll = rolling_ols(yS, XS, window=120).dropna().tail()
    print("Rolling tail:\n", roll.round(3))

    # Fama–MacBeth (panel)
    T = 24; N = 200
    dates = np.repeat(pd.date_range("2022-01-31", periods=T, freq="M"), N)
    asset = np.tile([f"A{i:03d}" for i in range(N)], T)
    # true lambdas
    lam = np.array([0.4, -0.2, 0.1])
    betas = rng.normal(size=(T*N, 3))
    eps_cs = rng.normal(scale=0.05, size=T*N)
    ret = (betas @ lam) + eps_cs
    panel = pd.DataFrame({"date": dates, "asset": asset, "ret": ret, "b1": betas[:,0], "b2": betas[:,1], "b3": betas[:,2]})
    fm = fama_macbeth(panel, ret_col="ret", factor_cols=["b1", "b2", "b3"])
    print("FM lambdas:\n", fm.lambdas.round(3), "\nFM t:\n", fm.t.round(3))