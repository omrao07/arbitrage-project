# engines/options/vol/vol_surface.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Callable

# ============================================================
# SVI (raw) parameterization
#   Total variance w(k) = a + b * ( rho*(k-m) + sqrt((k-m)^2 + sigma^2) )
#   where k = log-moneyness = ln(K / F_T)
#   Implied vol: iv(k) = sqrt( w(k) / T )
# ============================================================

@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_tuple(self) -> Tuple[float,float,float,float,float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)

def svi_total_var(k: np.ndarray, p: SVIParams) -> np.ndarray:
    a, b, rho, m, sigma = p.as_tuple()
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))

def svi_implied_vol(k: np.ndarray, T: float, p: SVIParams) -> np.ndarray:
    w = np.maximum(svi_total_var(k, p), 1e-12)
    return np.sqrt(w / max(T, 1e-12))

# ============================================================
# Robust SVI slice calibration (no SciPy, random multi-start + local jitters)
# Inputs:
#   strikes: array (K)
#   prices: not used; we fit on implied vols passed in 'iv'
#   forward F, time T, iv(K)
# ============================================================

@dataclass
class FitConfig:
    n_starts: int = 64
    n_refine: int = 200
    seed: int = 7
    # simple bounds to avoid arbitragey shapes exploding
    a_bounds: Tuple[float,float] = (0.0, 5.0)       # total variance floor at center
    b_bounds: Tuple[float,float] = (1e-6, 10.0)     # slope scale
    rho_bounds: Tuple[float,float] = (-0.999, 0.999)
    m_bounds: Tuple[float,float] = (-1.0, 1.0)      # center shift in k
    sigma_bounds: Tuple[float,float] = (1e-3, 5.0)

def _rand_uniform(rng, lo, hi, size=None):
    return lo + (hi - lo) * rng.random(size=size)

def _clip(p: SVIParams, cfg: FitConfig) -> SVIParams:
    a = float(np.clip(p.a, *cfg.a_bounds))
    b = float(np.clip(p.b, *cfg.b_bounds))
    rho = float(np.clip(p.rho, *cfg.rho_bounds))
    m = float(np.clip(p.m, *cfg.m_bounds))
    sigma = float(np.clip(p.sigma, *cfg.sigma_bounds))
    return SVIParams(a,b,rho,m,sigma)

def _loss_iv(k: np.ndarray, T: float, iv_obs: np.ndarray, p: SVIParams) -> float:
    iv_fit = svi_implied_vol(k, T, p)
    # L2 with mild wings regularization to keep slope finite
    resid = iv_fit - iv_obs
    reg = 1e-4 * (p.b**2 + p.sigma**2) + 1e-4 * (p.rho**2)
    return float(np.nanmean(resid * resid) + reg)

def fit_svi_slice(
    strikes: np.ndarray, iv: np.ndarray, forward: float, T: float, cfg: FitConfig = FitConfig()
) -> SVIParams:
    """
    Fit a single expiry's SVI parameters to observed implied vols.
    - strikes: K array (same units as F)
    - iv: Black vols (decimal)
    - forward: F_T
    - T: year fraction to expiry
    """
    strikes = np.asarray(strikes, dtype=float)
    iv = np.asarray(iv, dtype=float)
    mask = np.isfinite(iv) & np.isfinite(strikes) & (strikes > 0)
    strikes, iv = strikes[mask], iv[mask]
    if strikes.size < 5:
        # Fallback flat vol
        v = float(np.nanmean(iv)) if iv.size else 0.2
        return SVIParams(a=max(v*v*T, 1e-6), b=0.05, rho=-0.3, m=0.0, sigma=0.2)

    k = np.log(strikes / max(forward, 1e-12))

    rng = np.random.default_rng(cfg.seed)

    # initial population around coarse heuristics
    v_atm = float(np.nanmedian(iv))
    a0 = max((v_atm**2) * T * 0.8, cfg.a_bounds[0])
    best = None
    best_loss = np.inf

    for _ in range(cfg.n_starts):
        p = SVIParams(
            a=_rand_uniform(rng, cfg.a_bounds[0], cfg.a_bounds[1]),
            b=_rand_uniform(rng, cfg.b_bounds[0], cfg.b_bounds[1]),
            rho=_rand_uniform(rng, cfg.rho_bounds[0], cfg.rho_bounds[1]),
            m=_rand_uniform(rng, cfg.m_bounds[0], cfg.m_bounds[1]),
            sigma=_rand_uniform(rng, cfg.sigma_bounds[0], cfg.sigma_bounds[1]),
        )
        # warm tweak towards atm guess
        p.a = 0.5*p.a + 0.5*a0
        loss = _loss_iv(k, T, iv, p)
        if loss < best_loss:
            best, best_loss = p, loss

    # local random refinements ("poor man's Nelder-Mead")
    p = best
    step = np.array([0.2, 0.2, 0.1, 0.1, 0.2])  # relative steps
    for i in range(cfg.n_refine):
        scale = 0.9 ** (i/50)
        cand = SVIParams(
            a=p.a * (1 + step[0] * rng.normal(0, scale)), # type: ignore
            b=p.b * (1 + step[1] * rng.normal(0, scale)), # type: ignore
            rho=p.rho + step[2] * rng.normal(0, scale), # type: ignore
            m=p.m + step[3] * rng.normal(0, scale), # type: ignore
            sigma=p.sigma * (1 + step[4] * rng.normal(0, scale)), # type: ignore
        )
        cand = _clip(cand, cfg)
        loss = _loss_iv(k, T, iv, cand)
        if loss < best_loss:
            p, best, best_loss = cand, cand, loss

    # light convexity guard: ensure total variance is positive and convex enough in far wings
    # (b>0 and |rho|<1 already help). Clip 'a' to keep w_min >= small positive.
    w_min = np.min(svi_total_var(np.array([0.0]), p)) # type: ignore
    if w_min < 1e-8:
        p = SVIParams(a=float(p.a + (1e-8 - w_min) + 1e-8), b=p.b, rho=p.rho, m=p.m, sigma=p.sigma) # type: ignore
    return p # type: ignore

# ============================================================
# Surface container: fit per-expiry, time-interpolate, query IV(K,T)
# ============================================================

@dataclass
class Surface:
    forwards: pd.Series                 # index: expiry (datetime64), values: forward F_T
    params_by_expiry: Dict[pd.Timestamp, SVIParams]  # fitted SVI params per expiry
    expiries: pd.DatetimeIndex          # sorted expiries
    tenors_years: pd.Series             # year fraction from valuation to each expiry
    valuation_date: pd.Timestamp

    def iv(self, K: float, expiry: pd.Timestamp) -> float:
        """Black vol at strike K and expiry date."""
        T = float(self.tenors_years.loc[expiry])
        F = float(self.forwards.loc[expiry])
        p = self.params_by_expiry[expiry]
        k = np.log(K / F)
        return float(svi_implied_vol(np.array([k]), T, p)[0])

    def iv_grid(self, K: Iterable[float], expiry: pd.Timestamp) -> np.ndarray:
        K = np.asarray(list(K), dtype=float)
        T = float(self.tenors_years.loc[expiry])
        F = float(self.forwards.loc[expiry])
        p = self.params_by_expiry[expiry]
        k = np.log(K / F)
        return svi_implied_vol(k, T, p)

    def iv_bilinear(self, K: float, T_query: float) -> float:
        """
        Interpolate across time: linear in *total variance* (w = iv^2 * T).
        We first locate the surrounding expiries (T1<=T<=T2) and linearly blend w.
        """
        # edge cases
        Tseries = self.tenors_years.sort_values()
        if T_query <= Tseries.iloc[0]:
            e = Tseries.index[0]
            return self.iv(K, e) # type: ignore
        if T_query >= Tseries.iloc[-1]:
            e = Tseries.index[-1]
            return self.iv(K, e) # type: ignore

        # bracket
        idx_right = np.searchsorted(Tseries.values, T_query, side="right") # type: ignore
        e1, e2 = Tseries.index[idx_right-1], Tseries.index[idx_right]
        T1, T2 = float(Tseries.loc[e1]), float(Tseries.loc[e2]) # type: ignore

        F1, F2 = float(self.forwards.loc[e1]), float(self.forwards.loc[e2]) # type: ignore
        # forward interpolation in log space (simple)
        Fq = np.exp(np.interp(T_query, [T1, T2], [np.log(F1), np.log(F2)]))

        # map given K to log-moneyness against each forward
        k1, k2 = np.log(K / F1), np.log(K / F2)
        iv1 = self.iv(F1*np.exp(k1), e1)  # type: ignore # equals same as above; explicit for clarity
        iv2 = self.iv(F2*np.exp(k2), e2) # type: ignore

        w1, w2 = iv1*iv1 * T1, iv2*iv2 * T2
        wq = np.interp(T_query, [T1, T2], [w1, w2])
        ivq = np.sqrt(max(wq, 1e-12) / max(T_query, 1e-12))
        return float(ivq)

# ============================================================
# Builder
# ============================================================

def build_surface_from_iv(
    *,
    iv_table: pd.DataFrame,      # rows: strikes (float K), cols: expiries (datetime64), values: iv (decimal)
    forwards: pd.Series,         # index: same expiries, values: F_T
    valuation_date: pd.Timestamp,
    fit_cfg: FitConfig = FitConfig(),
) -> Surface:
    """
    Calibrate SVI per expiry from an IV matrix and forwards.
    Returns a Surface with callable iv(K, expiry)/iv_bilinear(K, T).
    """
    # sanitize / align
    expiries = pd.to_datetime(sorted([c for c in iv_table.columns]))
    iv = iv_table.reindex(columns=expiries).sort_index()
    fwd = forwards.reindex(expiries).astype(float)
    tenors = ((expiries - pd.to_datetime(valuation_date)).days.values / 365.0).astype(float)
    tenors = pd.Series(tenors, index=expiries)

    params: Dict[pd.Timestamp, SVIParams] = {}
    for e in expiries:
        T = max(float(tenors.loc[e]), 1e-6)
        K = iv.index.values.astype(float)
        iv_e = iv[e].values.astype(float)
        p = fit_svi_slice(K, iv_e, forward=float(fwd.loc[e]), T=T, cfg=fit_cfg)
        params[e] = p

    return Surface(forwards=fwd, params_by_expiry=params, expiries=expiries, tenors_years=tenors, valuation_date=pd.to_datetime(valuation_date))

# ============================================================
# No-arbitrage diagnostics (basic)
# ============================================================

def butterfly_arbitrage_metric(K: np.ndarray, iv: np.ndarray, T: float) -> float:
    """
    Rough static-arb check on slice convexity in strike:
    - Convert IV to total variance and check discrete convexity of call price vs strike.
    Returns a nonnegative penalty (0 = looks fine; >0 worse).
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)
    # Black call price convex in strike under no butterfly arb.
    # We'll approximate using finite differences on normalized call prices.
    from math import exp, sqrt, log
    F = 1.0  # normalized
    tau = max(T, 1e-12)
    # Rough Black call with F=1, discount=1:
    def black_call(k_, vol_):
        # K = exp(k)
        K_ = np.exp(k_)
        # d1,d2 in Black-76 with F=1
        v = max(vol_, 1e-9)
        d1 = (-np.log(K_) + 0.5*v*v*tau) / (v*sqrt(tau))
        d2 = d1 - v*sqrt(tau)
        # N(d1)-K*N(d2)
        return 0.5*(1+erf(d1/np.sqrt(2))) - K_*0.5*(1+erf(d2/np.sqrt(2)))

    from math import erf
    k = np.log(K / max(K.mean(), 1e-9))
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    C = np.array([black_call(kk, vv) for kk, vv in zip(k, iv)])
    # second finite difference
    if len(C) < 3:
        return 0.0
    sec = C[:-2] - 2*C[1:-1] + C[2:]
    penalty = float(np.sum(np.clip(-sec, 0, None)))
    return penalty

# ============================================================
# Convenience helpers
# ============================================================

def grid_from_quotes(
    quotes: pd.DataFrame,
    *,
    strike_col: str = "strike",
    expiry_col: str = "expiry",
    iv_col: str = "iv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Turn a long table of quotes into an IV matrix and forward series.
    `quotes` must contain one forward per expiry in a column named 'forward'.
    """
    if "forward" not in quotes.columns:
        raise ValueError("quotes must include a 'forward' column (per expiry)")
    piv = quotes.pivot(index=strike_col, columns=expiry_col, values=iv_col)
    fwd = quotes.drop_duplicates([expiry_col]).set_index(expiry_col)["forward"].sort_index()
    piv.columns = pd.to_datetime(piv.columns)
    fwd.index = pd.to_datetime(fwd.index)
    return piv.sort_index(), fwd.sort_index()

# ============================================================
# Example (synthetic)
# ============================================================

if __name__ == "__main__":
    # Synthetic smile: three expiries
    valuation = pd.Timestamp("2024-06-03")
    expiries = pd.to_datetime(["2024-07-05","2024-09-06","2024-12-13"])
    F = pd.Series([5050.0, 5075.0, 5100.0], index=expiries)

    K = np.linspace(0.6, 1.4, 25) * F.iloc[0]  # strikes around first forward scale (ok for demo)

    rng = np.random.default_rng(2)
    def make_iv(kshift):
        # toy U-shaped smile around ATM with skew
        m = np.log(K / F.iloc[0]) - kshift
        base = 0.20 + 0.04*np.abs(m) - 0.06*m  # skew + smile
        noise = 0.002*rng.standard_normal(len(m))
        return np.clip(base + noise, 0.05, 2.0)

    iv_mat = pd.DataFrame({expiries[0]: make_iv(0.00),
                           expiries[1]: make_iv(-0.05),
                           expiries[2]: make_iv(-0.10)}, index=K)

    surf = build_surface_from_iv(iv_table=iv_mat, forwards=F, valuation_date=valuation)

    # Query: vol for K=5200 at a mid tenor (T=0.25y)
    Tq = 0.25
    Kq = 5200.0
    vq = surf.iv_bilinear(Kq, Tq)
    print("IV(K=5200, T≈0.25y) ≈", round(vq, 4))

    # Quick no-arb metric per slice
    for e in surf.expiries:
        pen = butterfly_arbitrage_metric(iv_mat.index.values, iv_mat[e].values, float(surf.tenors_years.loc[e])) # type: ignore
        print(f"Slice {e.date()} penalty:", round(pen, 6))