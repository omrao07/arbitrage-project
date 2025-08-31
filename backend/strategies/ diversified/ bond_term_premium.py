# backend/research/alpha/term_premium/bond_term_premium.py
"""
Bond Term Premium (NS + AR(1) Expectations)

Given a history of zero-coupon yields across maturities (e.g., 0.25y, 1y, 2y, 5y, 10y),
this module:
  1) Fits Nelson–Siegel (NS) factors (level β0, slope β1, curvature β2, decay λ)
  2) Models (β0, β1, β2) as AR(1):  f_t = c + A * f_{t-1} + ε_t
  3) Uses the factor VAR(1) to compute the expected path of the short rate
  4) Computes Term Premium for each maturity:
        TP(n)_t = y(n)_t − (1/n) * Σ_{h=1}^{n*steps_per_year} E_t[ r_{t+h} ] * Δ
     where r is the model-implied short rate (NS at very short maturity)

You can use monthly, weekly, or daily data—set steps_per_year accordingly.

Dependencies: numpy, pandas; (optional) scipy for robust λ search (falls back to coarse grid).

Typical usage:
    import pandas as pd
    from bond_term_premium import TermPremiumNS

    # df_yields: DataFrame indexed by date, columns are maturities in years (floats), values in decimals (0.025 = 2.5%)
    maturities = [0.25, 1, 2, 5, 10]
    tp = TermPremiumNS(steps_per_year=12)  # monthly data
    res = tp.fit(df_yields[maturities])   # fits NS & AR(1)
    tp_today = tp.term_premium_latest()   # dict {maturity: TP}
    tp_hist  = tp.term_premium_history()  # DataFrame time series

This is a pedagogical, production-friendly baseline (fast, stable, no external data).
For research parity with ACM/Kim-Wright, swap in an affine term structure & macro VAR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# --------------------------- Nelson–Siegel ---------------------------

def _ns_loadings(mats: np.ndarray, lam: float) -> np.ndarray:
    """
    Nelson–Siegel loadings for yields:
        y(τ) = β0 + β1 * ((1 - e^{-λτ})/(λτ)) + β2 * [ ((1 - e^{-λτ})/(λτ)) - e^{-λτ} ]
    Returns design matrix X with columns [1, L1, L2]
    """
    tau = np.maximum(mats, 1e-6)
    x1 = (1 - np.exp(-lam * tau)) / (lam * tau)
    x2 = x1 - np.exp(-lam * tau)
    X = np.column_stack([np.ones_like(tau), x1, x2])
    return X


def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # stable OLS via lstsq
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b  # β0, β1, β2


def _fit_ns_once(yields: np.ndarray, mats: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    """Fit β for a given λ; return (β, mse)."""
    X = _ns_loadings(mats, lam)
    beta = _ols_beta(X, yields)
    resid = yields - X @ beta
    mse = float(np.mean(resid ** 2))
    return beta, mse


def _search_lambda(yields: np.ndarray, mats: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Find λ that minimizes MSE. Use SciPy if available; fallback to coarse→refined grid.
    """
    # sensible bounds for λ (per-year). NS literature often finds λ ~ [0.01, 5]
    lb, ub = 0.01, 5.0

    if _HAS_SCIPY:
        def obj(lam_arr):
            lam = float(lam_arr[0])
            _, mse = _fit_ns_once(yields, mats, lam)
            return mse
        res = minimize(obj, x0=np.array([0.8]), bounds=[(lb, ub)], method="L-BFGS-B")
        lam_best = float(np.clip(res.x[0], lb, ub))
        beta, mse = _fit_ns_once(yields, mats, lam_best)
        return lam_best, beta, mse

    # fallback: grid + local refine
    grid = np.logspace(np.log10(lb), np.log10(ub), 30)
    best = (None, None, np.inf)
    for g in grid:
        beta, mse = _fit_ns_once(yields, mats, g)
        if mse < best[2]:
            best = (g, beta, mse)
    lam0, beta0, mse0 = best

    # small neighborhood refine
    local = np.linspace(max(lb, lam0 * 0.5), min(ub, lam0 * 1.5), 25) # type: ignore
    for g in local:
        beta, mse = _fit_ns_once(yields, mats, g)
        if mse < mse0:
            lam0, beta0, mse0 = g, beta, mse
    return float(lam0), beta0, float(mse0) # type: ignore


def fit_ns_cross_section(curve: pd.Series) -> Tuple[np.ndarray, float]:
    """
    Fit NS to one curve snapshot (index: maturities in years; values: yields in decimals).
    Returns (params, mse) where params = [β0, β1, β2, λ]
    """
    y = curve.astype(float).values
    mats = curve.index.to_numpy(dtype=float)
    lam, beta, mse = _search_lambda(y, mats) # type: ignore
    return np.array([beta[0], beta[1], beta[2], lam], dtype=float), mse


# --------------------------- Factor model (AR(1)) ---------------------------

@dataclass
class AR1:
    c: np.ndarray   # intercept (3,)
    A: np.ndarray   # diag of AR(1) coeffs (3,3)
    Sigma: np.ndarray  # residual cov (3,3)

    def step_expectation(self, f: np.ndarray, h: int) -> np.ndarray:
        """
        E[f_{t+h} | f_t] for diagonal AR(1) with intercept.
        f_{t+1} = c + A f_t + eps
        """
        # closed-form (iterative is fine for small h)
        ft = f.copy()
        for _ in range(h):
            ft = self.c + self.A @ ft
        return ft


def fit_ar1(factors: pd.DataFrame) -> AR1:
    """
    Fit diagonal AR(1) on columns ['beta0','beta1','beta2'].
    Simple OLS per factor; residual cov estimated from residuals.
    """
    cols = ["beta0", "beta1", "beta2"]
    X = factors[cols].shift(1).dropna()
    Y = factors[cols].loc[X.index]

    c = []
    a = []
    resid = []
    for col in cols:
        x = np.column_stack([np.ones(len(X)), X[col].values]) # type: ignore
        y = Y[col].values
        b, *_ = np.linalg.lstsq(x, y, rcond=None)  # type: ignore # y = b0 + b1 * x
        c.append(b[0])
        a.append(b[1])
        resid.append(y - x @ b)

    c = np.array(c)
    A = np.diag(np.array(a))
    R = np.vstack(resid).T
    Sigma = np.cov(R.T) if R.shape[0] > 2 else np.eye(3) * 1e-6
    return AR1(c=c, A=A, Sigma=Sigma)


# --------------------------- Term premium engine ---------------------------

def ns_yield_from_factors(mats: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    X = _ns_loadings(mats, lam)
    return X @ np.array([beta0, beta1, beta2], dtype=float)


def implied_short_rate(beta0: float, beta1: float, beta2: float, lam: float, eps: float = 1/365) -> float:
    """
    Approximate short rate as NS yield at an epsilon maturity.
    """
    return float(ns_yield_from_factors(np.array([eps]), beta0, beta1, beta2, lam)[0])


class TermPremiumNS:
    def __init__(self, steps_per_year: int = 12):
        """
        steps_per_year: sampling frequency of your panel (e.g., 12 for monthly, 252 for daily).
        """
        self.steps_per_year = int(steps_per_year)
        self.curve_history: Optional[pd.DataFrame] = None
        self.factors_: Optional[pd.DataFrame] = None
        self.ar1_: Optional[AR1] = None
        self.maturities_: Optional[np.ndarray] = None
        self.ns_fit_mse_: Optional[pd.Series] = None

    # ---- Fit over history ----
    def fit(self, df_yields: pd.DataFrame) -> Dict[str, any]: # type: ignore
        """
        df_yields: index = dates; columns = maturities in years (floats); values in decimals.
        Returns summary dict.
        """
        assert df_yields.shape[1] >= 3, "Need at least 3 maturities for NS"
        self.curve_history = df_yields.sort_index().copy()
        mats = np.array(df_yields.columns, dtype=float)
        self.maturities_ = mats

        rows = []
        mses = []
        for dt, row in df_yields.iterrows():
            params, mse = fit_ns_cross_section(row.dropna())
            rows.append(params)
            mses.append(mse)
        fac = pd.DataFrame(
            rows,
            index=df_yields.index,
            columns=["beta0", "beta1", "beta2", "lambda"],
        )
        self.factors_ = fac
        self.ns_fit_mse_ = pd.Series(mses, index=df_yields.index, name="mse")

        # Fit AR(1) on betas (ignore lambda; hold it at last obs or moving median)
        self.ar1_ = fit_ar1(fac[["beta0", "beta1", "beta2"]])

        return {
            "n_obs": len(df_yields),
            "avg_fit_rmse_bps": float(np.sqrt(np.nanmean(self.ns_fit_mse_.values)) * 1e4), # type: ignore
            "last_lambda": float(fac["lambda"].iloc[-1]),
        }

    # ---- Expectations path ----
    def expected_short_path(self, horizon_years: float, start_at: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Builds E_t[r_{t+h}] for h = 1..H steps (H = horizon_years * steps_per_year)
        using AR(1) expectations on (β0, β1, β2) with λ fixed at last observed.
        """
        assert self.factors_ is not None and self.ar1_ is not None, "Call fit() first."
        H = int(round(horizon_years * self.steps_per_year))
        fac_last = self.factors_.iloc[-1]
        f0 = np.array([fac_last["beta0"], fac_last["beta1"], fac_last["beta2"]], dtype=float)
        lam = float(fac_last["lambda"])

        path = []
        f = f0.copy()
        for h in range(1, H + 1):
            f = self.ar1_.step_expectation(f, 1)
            r = implied_short_rate(f[0], f[1], f[2], lam)
            path.append(r)

        idx0 = (self.factors_.index[-1] if start_at is None else start_at)
        # make a PeriodIndex-like forward dates (keeps simple spacing)
        idx = pd.date_range(idx0, periods=H, freq=pd.infer_freq(self.factors_.index) or "M") # type: ignore
        return pd.Series(path, index=idx, name="E_short")

    # ---- Term premium (latest) ----
    def term_premium_latest(self, horizon_years_list: Optional[Iterable[float]] = None) -> Dict[float, float]:
        """
        Returns {maturity_years: TP} at the last date.
        If horizon_years_list is None, uses the original maturities of the input curve.
        """
        assert self.curve_history is not None, "Call fit() first."
        last_curve = self.curve_history.iloc[-1]
        mats = self.maturities_ if horizon_years_list is None else np.array(list(horizon_years_list), dtype=float)

        # expected average short over horizon n:
        tps = {}
        for n in mats: # type: ignore
            path = self.expected_short_path(horizon_years=float(n))
            exp_avg_short = float(path.mean())
            # observed y(n)
            fac_last = self.factors_.iloc[-1] # type: ignore
            y_n = float(ns_yield_from_factors(np.array([n]), fac_last["beta0"], fac_last["beta1"], fac_last["beta2"], fac_last["lambda"])[0])
            tps[float(n)] = y_n - exp_avg_short
        return tps

    # ---- Term premium (full history) ----
    def term_premium_history(self, maturities_out: Optional[Iterable[float]] = None) -> pd.DataFrame:
        """
        Computes a historical TP series for requested maturities using a rolling fit.
        (Uses expanding AR(1) re-fit each step for simplicity; for speed use fixed AR(1).)
        """
        assert self.curve_history is not None, "Call fit() first."
        mats = self.maturities_ if maturities_out is None else np.array(list(maturities_out), dtype=float)

        records = []
        fac = self.factors_
        for i in range(12, len(fac)):  # type: ignore # start after a small burn-in
            fac_sub = fac.iloc[:i] # type: ignore
            ar = fit_ar1(fac_sub[["beta0", "beta1", "beta2"]])
            last = fac_sub.iloc[-1]
            lam = float(last["lambda"])
            f0 = np.array([last["beta0"], last["beta1"], last["beta2"]], dtype=float)

            row = {"date": fac_sub.index[-1]}
            for n in mats: # type: ignore
                # build expectations path of short rate
                H = int(round(float(n) * self.steps_per_year))
                f = f0.copy()
                shp = []
                for _ in range(H):
                    f = ar.step_expectation(f, 1)
                    shp.append(implied_short_rate(f[0], f[1], f[2], lam))
                exp_avg_short = float(np.mean(shp))
                # observed y(n) from NS factors at that date
                y_n = float(ns_yield_from_factors(np.array([n]), last["beta0"], last["beta1"], last["beta2"], lam)[0])
                row[float(n)] = y_n - exp_avg_short # type: ignore
            records.append(row)

        out = pd.DataFrame(records).set_index("date")
        out.columns = [float(c) if isinstance(c, str) else c for c in out.columns]
        return out


# --------------------------- Quick demo (optional) ---------------------------

if __name__ == "__main__":
    # Minimal synthetic demo so the file can be run directly.
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-31", periods=60, freq="M")
    mats = [0.25, 1, 2, 5, 10]

    # create a toy upward-sloping curve + noise (in decimals)
    base_curve = np.array([0.01, 0.012, 0.014, 0.017, 0.02])
    Y = []
    for t in range(len(dates)):
        shock = rng.normal(0, 0.0005, size=len(mats))
        Y.append(base_curve + 0.0003 * t + shock)
    df = pd.DataFrame(Y, index=dates, columns=mats)

    tp = TermPremiumNS(steps_per_year=12)
    summary = tp.fit(df)
    print("[fit]", summary)
    print("[latest TP]", tp.term_premium_latest())
    print("[history head]\n", tp.term_premium_history([2, 5, 10]).head())