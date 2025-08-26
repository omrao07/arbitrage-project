# backend/options/fit_vol_surface.py
from __future__ import annotations

"""
Vol Surface Fitter (SVI + guards)
---------------------------------
- Converts option quotes -> implied vols
- Fits Raw-SVI per expiry slice:  w(k)=a + b*( rho*(k-m) + sqrt((k-m)^2 + sigma^2 ) )
- Builds a T–k surface of *total variance* w = iv^2 * T
- Enforces light static/calendar arbitrage guards (bounds + monotone-in-T adjust)
- Exposes: iv(K,T), price_call/put(K,T), export_grid()

Inputs:
  DataFrame with columns (required):
    - expiry (datetime-like)
    - strike (float)
    - cp (str: 'C' or 'P')
    - mid (float)  # mid-price
  Also need spot S, rate r, dividend q (annualized, continuously compounded).
  If you have forwards per expiry, you can pass F_by_T instead of (S,r,q).

CLI:
  python backend/options/fit_vol_surface.py \
    --csv data/chain.csv --spot 221.35 --rate 0.04 --div 0.00 --out data/vol/tech.json

Dependencies: numpy, pandas, scipy (optimize), (optional) pyarrow for parquet export
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq

# ---------- Black / BS helpers ---------------------------------------------

SQRT_2PI = math.sqrt(2.0 * math.pi)

def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _norm_cdf(x: float) -> float:
    # Abramowitz-Stegun approximation (good enough for pricer/vega)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.319381530*t - 0.356563782*t**2 + 1.781477937*t**3 - 1.821255978*t**4 + 1.330274429*t**5
    cnd = 1.0 - _phi(x) * d
    return cnd if x >= 0 else 1.0 - cnd

def bs_price(S: float, K: float, T: float, r: float, q: float, iv: float, cp: str) -> float:
    """Black-Scholes with continuous carry (q)."""
    if T <= 0 or iv <= 0 or K <= 0:
        # intrinsic approx
        fwd = S * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        if cp.upper() == "C":
            return max(S*math.exp(-q*T) - K*disc, 0.0)
        return max(K*disc - S*math.exp(-q*T), 0.0)

    sqrtT = math.sqrt(max(1e-12, T))
    sig = max(1e-9, iv)
    d1 = (math.log(S/K) + (r - q + 0.5*sig*sig)*T) / (sig*sqrtT)
    d2 = d1 - sig*sqrtT
    df_r = math.exp(-r*T)
    df_q = math.exp(-q*T)
    if cp.upper() == "C":
        return df_q*S*_norm_cdf(d1) - df_r*K*_norm_cdf(d2)
    else:
        return df_r*K*_norm_cdf(-d2) - df_q*S*_norm_cdf(-d1)

def bs_vega(S: float, K: float, T: float, r: float, q: float, iv: float) -> float:
    if T <= 0 or iv <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r - q + 0.5*iv*iv)*T) / (iv*sqrtT)
    return S * math.exp(-q*T) * _phi(d1) * sqrtT

def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, cp: str) -> float:
    """Brent solve for IV. Brackets [1e-6, 6.0]."""
    if T <= 0 or K <= 0 or price <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_price(S, K, T, r, q, sig, cp) - price

    # intrinsic and upper bounds sanity
    try:
        # Quick reject: if price > almost-forward notional, skip
        f0, f1 = f(1e-6), f(6.0)
        if f0 * f1 > 0:
            # Try expand a bit
            f2 = f(10.0)
            if f0 * f2 > 0:
                return float("nan")
            return brentq(f, 1e-6, 10.0, maxiter=100) # type: ignore
        return brentq(f, 1e-6, 6.0, maxiter=100) # type: ignore
    except Exception:
        return float("nan")

# ---------- SVI (raw paramization) -----------------------------------------

@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

def svi_total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """w(k) total variance for given log-moneyness k."""
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x*x + p.sigma*p.sigma))

def _svi_bounds() -> Tuple[np.ndarray, np.ndarray]:
    # a >= 0; b >= 1e-6; |rho| < 1; sigma > 0
    lb = np.array([0.0, 1e-8, -0.999, -5.0, 1e-8])
    ub = np.array([5.0,  5.0,   0.999,  5.0,  5.0])
    return lb, ub

def fit_svi_slice(k: np.ndarray, w: np.ndarray, weights: Optional[np.ndarray] = None) -> SVIParams:
    """Least-squares fit of SVI to points (k, w). Robust loss, bounded."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if weights is None:
        weights = np.ones_like(w)

    # initial guess: symmetric smile around m ~ median(k), slope ~ 0.1
    m0 = np.median(k) if len(k) else 0.0
    a0 = max(1e-4, np.percentile(w, 10)) # type: ignore
    b0 = 0.2
    rho0 = 0.0
    sigma0 = max(0.05, (np.percentile(abs(k - m0), 75) or 0.2)) # type: ignore
    x0 = np.array([a0, b0, rho0, m0, sigma0])

    lb, ub = _svi_bounds()

    def resid(x):
        p = SVIParams(*x)
        model = svi_total_variance(k, p)
        # robust residuals, weight by 1/sqrt(w) (approx inverse variance) and input weights
        return (model - w) * weights / np.sqrt(np.maximum(1e-12, w))

    res = least_squares(resid, x0=x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.01, max_nfev=2000)
    a, b, rho, m, sigma = res.x
    # Light SVI static guard: ensure overall minimum non-negative
    # a + b*sigma*sqrt(1-rho^2) >= 0
    floor = a + b*sigma*math.sqrt(max(0.0, 1.0 - rho*rho))
    if floor < 0:
        a += (-floor + 1e-8)
    return SVIParams(a, b, rho, m, sigma)

# ---------- Surface builder -------------------------------------------------

@dataclass
class SliceFit:
    T: float                # years
    F: float                # forward used for this slice
    params: SVIParams
    rms: float              # RMSE on total variance
    n: int                  # number of quotes used

@dataclass
class Surface:
    spot: float
    r: float
    q: float
    slice_fits: List[SliceFit]
    # Interpolation grids
    Ts: np.ndarray          # sorted unique T (years)
    ks: np.ndarray          # k grid used for interpolation
    W: np.ndarray           # total variance grid (len(Ts) x len(ks))

    def iv(self, K: float, T: float) -> float:
        if T <= 0 or K <= 0: return float("nan")
        w = self._w_interp(T, K)
        return math.sqrt(max(1e-12, w) / max(1e-12, T))

    def price(self, K: float, T: float, cp: str) -> float:
        iv = self.iv(K, T)
        return bs_price(self.spot, K, T, self.r, self.q, iv, cp)

    # --- helpers ---
    def _fwd(self, T: float) -> float:
        return self.spot * math.exp((self.r - self.q) * T)

    def _w_interp(self, T: float, K: float) -> float:
        # bilinear interpolation in (T, k)
        F = self._fwd(T)
        k = math.log(K / F)
        # clamp k inside grid
        k = max(float(self.ks[0]), min(float(self.ks[-1]), k))
        # locate T and k indices
        iT = np.searchsorted(self.Ts, T, side="left")
        jK = np.searchsorted(self.ks, k, side="left")
        i0 = max(0, min(len(self.Ts) - 1, iT - 1)) # type: ignore
        i1 = max(0, min(len(self.Ts) - 1, iT)) # type: ignore
        j0 = max(0, min(len(self.ks) - 1, jK - 1)) # type: ignore
        j1 = max(0, min(len(self.ks) - 1, jK)) # type: ignore

        if i0 == i1 and j0 == j1:
            return float(self.W[i0, j0])

        # weights
        T0, T1 = float(self.Ts[i0]), float(self.Ts[i1])
        K0, K1 = float(self.ks[j0]), float(self.ks[j1])
        # handle degenerate spans
        tden = (T1 - T0) or 1e-12
        kden = (K1 - K0) or 1e-12
        wtT = (T - T0) / tden
        wtK = (k - K0) / kden

        # bilinear
        w00 = self.W[i0, j0]
        w01 = self.W[i0, j1]
        w10 = self.W[i1, j0]
        w11 = self.W[i1, j1]
        w0 = (1 - wtK) * w00 + wtK * w01
        w1 = (1 - wtK) * w10 + wtK * w11
        w = (1 - wtT) * w0 + wtT * w1
        return float(max(0.0, w))

# ---------- Fitting pipeline -----------------------------------------------

def _prepare_chain(
    df: pd.DataFrame,
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    min_pts: int = 6,
    drop_bad: bool = True,
) -> pd.DataFrame:
    """
    Normalize chain; compute T (years), forward F, implied vols, k, total variance.
    """
    d = df.copy()
    d = d.rename(columns={"CallPut": "cp", "option_type": "cp"})
    # parse expiry
    d["expiry"] = pd.to_datetime(d["expiry"], utc=True).dt.tz_localize(None)
    if "asof" in d.columns:
        asof = pd.to_datetime(d["asof"], utc=True).dt.tz_localize(None)
        t0 = asof.iloc[0]
    else:
        t0 = pd.Timestamp.utcnow().tz_localize(None)
    d["T"] = (d["expiry"] - t0).dt.total_seconds() / (365.0 * 24 * 3600.0)
    d = d[d["T"] > 0]

    # sanity filters
    if drop_bad:
        d = d[(d["mid"] > 0) & (d["strike"] > 0)]
        # very deep ITM/OTM sometimes garbage; keep a wide band
        # (we'll also filter NaN IVs later)

    # compute IVs
    ivs = []
    for S_, K, T_, r_, q_, cp_, px in zip(
        [spot]*len(d), d["strike"], d["T"], [r]*len(d), [q]*len(d), d["cp"], d["mid"]
    ):
        ivs.append(implied_vol(px, S_, float(K), float(T_), r_, q_, str(cp_)))
    d["iv"] = ivs
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv"])

    # forward & log-moneyness
    d["F"] = spot * np.exp((r - q) * d["T"])
    d["k"] = np.log(d["strike"] / d["F"])
    d["w"] = d["iv"] ** 2 * d["T"]

    # keep per-expiry min points
    if min_pts:
        d = d.groupby("expiry").filter(lambda x: x["iv"].count() >= min_pts)
    return d

def fit_surface_from_chain(
    df_chain: pd.DataFrame,
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    k_grid: Optional[Iterable[float]] = None,
) -> Surface:
    """
    Fit SVI per expiry, then build a T–k grid with monotone-in-T guard.
    """
    d = _prepare_chain(df_chain, spot=spot, r=r, q=q)

    slice_fits: List[SliceFit] = []
    for exp, sl in d.groupby("expiry"):
        T = float(sl["T"].iloc[0])
        F = float(sl["F"].iloc[0])

        # weights: vega weighting to focus on ATM
        wts = []
        for K, T_, iv in zip(sl["strike"], sl["T"], sl["iv"]):
            wts.append(max(1e-6, bs_vega(spot, float(K), float(T_), r, q, float(iv))))
        wts = np.asarray(wts)
        wts = wts / np.max(wts) if len(wts) else wts

        params = fit_svi_slice(sl["k"].values, sl["w"].values, weights=wts) # type: ignore
        w_fit = svi_total_variance(sl["k"].values, params) # type: ignore
        rms = float(np.sqrt(np.nanmean((w_fit - sl["w"].values) ** 2)))
        slice_fits.append(SliceFit(T=T, F=F, params=params, rms=rms, n=len(sl)))

    if not slice_fits:
        raise ValueError("No valid slices to fit.")

    # sort by T
    slice_fits.sort(key=lambda x: x.T)

    # Build interpolation grid
    Ts = np.array([sf.T for sf in slice_fits], dtype=float)
    # k grid: union of observed k, expanded to symmetric range
    all_k = d["k"].values
    kmin, kmax = float(np.nanpercentile(all_k, 1)), float(np.nanpercentile(all_k, 99)) # type: ignore
    span = max(0.4, (kmax - kmin))
    k_lo, k_hi = -max(abs(kmin), abs(kmax), span / 2), max(abs(kmin), abs(kmax), span / 2)
    ks = np.linspace(k_lo, k_hi, 101) if k_grid is None else np.array(list(k_grid), dtype=float)

    # evaluate each slice on ks -> W[T, k]
    W = np.zeros((len(Ts), len(ks)), dtype=float)
    for i, sf in enumerate(slice_fits):
        W[i, :] = svi_total_variance(ks, sf.params)

    # Calendar no-arb guard: total variance should be non-decreasing in T for each k
    # Make each column cumulative max down T
    for j in range(W.shape[1]):
        W[:, j] = np.maximum.accumulate(W[:, j])

    return Surface(
        spot=float(spot),
        r=float(r),
        q=float(q),
        slice_fits=slice_fits,
        Ts=Ts,
        ks=ks,
        W=W,
    )

# ---------- Export for frontend --------------------------------------------

def export_grid(surface: Surface, strikes: Iterable[float], expiries: Iterable[pd.Timestamp]) -> Dict:
    """
    Build {strikes, expiries, iv} where iv is [len(expiries) x len(strikes)].
    Expiries array must be actual datetimes; we’ll map to year fractions with (S,r,q).
    """
    strikes = list(map(float, strikes))
    expiries = list(pd.to_datetime(list(expiries)))

    # Convert expiries to T using first slice's asof proxy: assume "now"
    now = pd.Timestamp.utcnow()
    Ts = [max(1e-9, (e - now).total_seconds() / (365.0 * 24 * 3600.0)) for e in expiries]

    IV = []
    for T in Ts:
        row = [surface.iv(K, T) for K in strikes]
        IV.append(row)

    return {
        "strikes": strikes,
        "expiries": [e.isoformat() for e in expiries],
        "iv": IV,  # 2D list
    }

# ---------- CLI -------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit SVI vol surface from option chain CSV.")
    p.add_argument("--csv", type=str, required=True, help="CSV with columns: expiry,strike,cp,mid[,asof]")
    p.add_argument("--spot", type=float, required=True, help="Spot price S0")
    p.add_argument("--rate", type=float, default=0.03, help="Risk-free rate (annual, cont. comp)")
    p.add_argument("--div", type=float, default=0.0, help="Dividend/borrow rate q (annual, cont. comp)")
    p.add_argument("--out", type=str, default="data/vol/surface.json", help="Output JSON for frontend")
    p.add_argument("--grid-strikes", type=str, default="", help="Comma strikes for export grid (optional)")
    p.add_argument("--grid-expiries", type=str, default="", help="Comma ISO expiries for export grid (optional)")
    return p.parse_args()

def main():
    args = _parse_args()
    df = pd.read_csv(args.csv)
    surf = fit_surface_from_chain(df, spot=args.spot, r=args.rate, q=args.div)

    # default export grid: use observed unique strikes/expiries
    strikes = sorted(df["strike"].unique().tolist()) if not args.grid_strikes else [float(x) for x in args.grid_strikes.split(",")]
    expiries = sorted(pd.to_datetime(df["expiry"].unique()).tolist()) if not args.grid_expiries else [pd.to_datetime(x) for x in args.grid_expiries.split(",")]

    payload = {
        "meta": {
            "spot": args.spot,
            "r": args.rate,
            "q": args.div,
            "slices": [
                {
                    "T_years": sf.T,
                    "F": sf.F,
                    "params": sf.params.__dict__,
                    "rms": sf.rms,
                    "n": sf.n,
                } for sf in surf.slice_fits
            ],
        },
        "grid": export_grid(surf, strikes=strikes, expiries=expiries),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[fit_vol_surface] wrote {args.out} with {len(expiries)} expiries × {len(strikes)} strikes")

if __name__ == "__main__":
    main()