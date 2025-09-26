#!/usr/bin/env python3
"""
scenarios.py — Scenario generator for risk & backtesting

Generates multi-asset return scenarios using several models:

Modes
-----
- historical         : sample past slices (no replacement) of length H
- block_bootstrap    : circular block bootstrap (overlapping blocks) of length H
- mvn                : multivariate normal with sample (or shrunk) cov
- student_t          : multivariate t (Gaussian copula scaling)
- pca_mvn            : simulate principal components then reconstruct
- gaussian_copula    : Gaussian copula + empirical marginals (rank-preserving)
- mix                : mixture of regimes; pass multiple return files & weights
- shock              : apply deterministic shocks to a base mode (--base-mode)

Common options
--------------
--returns RET.csv            Wide CSV (Date x Assets) of historical returns (in decimals)
--n 10000                    Number of scenarios
--h 10                       Horizon length in steps (e.g., days)
--aggregate sum|geom         Aggregate across horizon (sum vs geometric compounding)
--mean 0                     Drift add-back per step (if >0)
--shrink 0.0                 Ledoit-Wolf-like shrink to identity (0..1)
--df 6                       Degrees of freedom (student_t)
--block 20                   Block size for block bootstrap
--pca-k 0                    # of PCs to keep (0 => all, or fraction 0<k<1)
--seed 123                   RNG seed
--outdir out_scen            Output directory

Shocks (optional, can be combined with any mode)
-----------------------------------------------
--shock "AAPL:-0.05,MSFT:-0.03"    Per-step additive shock to named assets
--scale 1.0                        Scale factor on simulated returns post-gen

Mixture mode
------------
--mix-returns "reg1.csv,reg2.csv"
--mix-weights "0.6,0.4"
--mode mix --base-mode mvn         (the base model applied per regime file)

Outputs
-------
- scenarios.parquet    : Panel (scenario, step, asset) -> return
- agg.csv              : Aggregated per-scenario returns per asset
- stats.json           : Summary (per-asset mean/vol, total corr heatmap flattened)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from math import erf, sqrt, log

# -------- I/O --------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    c0 = df.columns[0].lower()
    if c0 in ("date", "time"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    return df.sort_index()

# -------- Stats helpers --------
def shrink_cov(S: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink covariance to identity-scaled: (1-a)S + a*tr(S)/p * I."""
    if alpha <= 0: return S
    p = S.shape[0]
    tr = np.trace(S) / max(p, 1)
    return (1 - alpha) * S + alpha * tr * np.eye(p)

def cov_from_returns(R: np.ndarray, mean_add: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    mu = R.mean(axis=0) + mean_add
    S = np.cov(R, rowvar=False, ddof=1)
    return mu, S

# Fast approximations for Φ, Φ^{-1} (no scipy)
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_ppf(u: np.ndarray) -> np.ndarray:
    # Acklam's approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    u = np.clip(u, 1e-12, 1 - 1e-12)
    q = u - 0.5
    r = np.zeros_like(u)
    lo = u < 0.02425
    hi = u > 0.97575
    mid = (~lo) & (~hi)
    # lower
    if lo.any():
        x = np.sqrt(-2*np.log(u[lo]))
        r[lo] = (((((c[0]*x + c[1])*x + c[2])*x + c[3])*x + c[4])*x + c[5]) / \
                 ((((d[0]*x + d[1])*x + d[2])*x + d[3])*x + 1)
        r[lo] *= -1
    # upper
    if hi.any():
        x = np.sqrt(-2*np.log(1 - u[hi]))
        r[hi] = (((((c[0]*x + c[1])*x + c[2])*x + c[3])*x + c[4])*x + c[5]) / \
                 ((((d[0]*x + d[1])*x + d[2])*x + d[3])*x + 1)
    # middle
    if mid.any():
        x = q[mid]**2
        r[mid] = (((((a[0]*x + a[1])*x + a[2])*x + a[3])*x + a[4])*x + a[5]) * q[mid] / \
                  (((((b[0]*x + b[1])*x + b[2])*x + b[3])*x + b[4])*x + 1)
    return r

def chol_psd(S: np.ndarray) -> np.ndarray:
    # Robust Cholesky for PSD: add tiny jitter if needed
    jitter = 1e-10
    for _ in range(7):
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S = S + jitter * np.eye(S.shape[0])
            jitter *= 10
    # Fallback: eigen decomposition -> nearest PSD
    w, V = np.linalg.eigh(S)
    w[w < 0] = 0
    Spsd = (V * w) @ V.T
    return np.linalg.cholesky(Spsd + 1e-12 * np.eye(S.shape[0]))

# -------- Scenario generators --------
def gen_historical(R: np.ndarray, n: int, h: int, rng: np.random.Generator) -> np.ndarray:
    T = R.shape[0]
    if T < h:
        raise ValueError("Not enough historical length for requested horizon.")
    starts = rng.integers(0, T - h + 1, size=n)
    paths = np.stack([R[s:s+h, :] for s in starts], axis=0)  # (n,h,p)
    return paths

def gen_block_bootstrap(R: np.ndarray, n: int, h: int, block: int, rng: np.random.Generator) -> np.ndarray:
    T, p = R.shape
    if block <= 0: block = max(1, h)
    # circular indices
    paths = np.empty((n, h, p), dtype=float)
    n_blocks = int(np.ceil(h / block))
    starts = rng.integers(0, T, size=(n, n_blocks))
    for i in range(n):
        segs = []
        for b in range(n_blocks):
            s = starts[i, b]
            idx = (np.arange(block) + s) % T
            segs.append(R[idx])
        path = np.vstack(segs)[:h]
        paths[i] = path
    return paths

def gen_mvn(mu: np.ndarray, S: np.ndarray, n: int, h: int, rng: np.random.Generator) -> np.ndarray:
    L = chol_psd(S)
    p = len(mu)
    Z = rng.standard_normal(size=(n*h, p))
    X = Z @ L.T + mu
    return X.reshape(n, h, p)

def gen_student_t(mu: np.ndarray, S: np.ndarray, df: int, n: int, h: int, rng: np.random.Generator) -> np.ndarray:
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance.")
    L = chol_psd(S * (df-2)/df)  # scale so covariance equals S
    p = len(mu)
    chi = rng.chisquare(df, size=(n*h, 1))
    G = rng.standard_normal(size=(n*h, p))
    X = (G / np.sqrt(chi/df)) @ L.T + mu
    return X.reshape(n, h, p)

def pca_decompose(S: np.ndarray):
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]

def gen_pca_mvn(mu: np.ndarray, S: np.ndarray, k: float | int, n: int, h: int, rng: np.random.Generator) -> np.ndarray:
    w, V = pca_decompose(S)
    p = len(mu)
    if isinstance(k, float) and 0 < k < 1:
        # keep enough comps to reach k fraction of variance
        cum = np.cumsum(w) / np.sum(w)
        keep = np.searchsorted(cum, k) + 1
    else:
        keep = p if (not isinstance(k, int) or k <= 0 or k > p) else k
    Vk = V[:, :keep]
    wk = w[:keep]
    Lk = Vk @ np.diag(np.sqrt(np.maximum(wk, 0)))
    Z = rng.standard_normal(size=(n*h, keep))
    X = Z @ Lk.T + mu
    return X.reshape(n, h, p)

def empirical_inv_cdf(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse ECDF per column: map U(0,1) to sample quantiles."""
    # x: (T,p), u: (m,p)
    T, p = x.shape
    qs = np.linspace(0, 1, T, endpoint=False) + 0.5/T
    x_sorted = np.sort(x, axis=0)
    # interpolate per column
    out = np.empty_like(u)
    for j in range(p):
        out[:, j] = np.interp(u[:, j], qs, x_sorted[:, j], left=x_sorted[0, j], right=x_sorted[-1, j])
    return out

def gen_gaussian_copula(R: np.ndarray, n: int, h: int, rng: np.random.Generator) -> np.ndarray:
    # Estimate correlation, generate Gaussian Z, transform to U via Φ, then to returns via inverse ECDF per asset.
    T, p = R.shape
    Rcs = (R - R.mean(0)) / (R.std(0, ddof=1) + 1e-12)
    C = np.corrcoef(Rcs, rowvar=False)
    L = chol_psd(C)
    Z = rng.standard_normal(size=(n*h, p)) @ L.T
    U = norm_cdf(Z)
    X = empirical_inv_cdf(R, U).reshape(n, h, p)#type:ignore
    return X

# -------- Utilities --------
def aggregate(paths: np.ndarray, method: str) -> np.ndarray:
    """
    paths: (n,h,p) per-step returns.
    Returns aggregated per-scenario return per asset: (n,p)
    """
    if method == "geom":
        return np.prod(1 + paths, axis=1) - 1
    else:
        return np.sum(paths, axis=1)

def parse_shocks(sh: str, assets: List[str]) -> np.ndarray:
    v = np.zeros(len(assets), dtype=float)
    if not sh: return v
    parts = [s.strip() for s in sh.split(",") if s.strip()]
    m = {}
    for it in parts:
        k, val = it.split(":")
        m[k.strip()] = float(val)
    for i, a in enumerate(assets):
        if a in m: v[i] = m[a]
    return v

def mixture_weights(ws: str, k: int) -> np.ndarray:
    vals = np.array([float(x) for x in ws.split(",")], dtype=float)
    if vals.size != k: raise ValueError("mix-weights length must match mix-returns files.")
    vals = np.clip(vals, 0, None)
    s = vals.sum()
    if s <= 0: raise ValueError("mix-weights must sum > 0")
    return vals / s

# -------- Orchestration --------
def run_mode(
    mode: str,
    base_mode: str,
    R_hist: Optional[pd.DataFrame],
    n: int, h: int,
    mean_add: float, shrink: float, df: int, block: int, pca_k: float | int,
    rng: np.random.Generator,
    mix_files: Optional[List[str]] = None,
    mix_w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    if mode == "mix":
        if not mix_files or mix_w is None:
            raise ValueError("Provide --mix-returns and --mix-weights for mode 'mix'.")
        # sample regime per scenario, then generate using base_mode on that regime's data
        Rs = [read_wide_csv(p).dropna(how="all").dropna(axis=1, how="all").values for p in mix_files]
        assets = list(read_wide_csv(mix_files[0]).columns)
        out = np.empty((n, h, len(assets)), dtype=float)
        reg_idx = rng.choice(len(Rs), size=n, p=mix_w)
        for i in range(n):
            R = Rs[reg_idx[i]]
            mu, S = cov_from_returns(R, mean_add)
            S = shrink_cov(S, shrink)
            if base_mode == "mvn":
                out[i:i+1] = gen_mvn(mu, S, 1, h, rng)
            elif base_mode == "student_t":
                out[i:i+1] = gen_student_t(mu, S, df, 1, h, rng)
            elif base_mode == "pca_mvn":
                out[i:i+1] = gen_pca_mvn(mu, S, pca_k, 1, h, rng)
            elif base_mode == "gaussian_copula":
                out[i:i+1] = gen_gaussian_copula(R, 1, h, rng)
            elif base_mode == "historical":
                out[i:i+1] = gen_historical(R, 1, h, rng)
            elif base_mode == "block_bootstrap":
                out[i:i+1] = gen_block_bootstrap(R, 1, h, block, rng)
            else:
                raise ValueError("Unknown --base-mode for mixture.")
        return out, assets

    if R_hist is None and mode not in ("mvn", "student_t", "pca_mvn"):  # safety (though mvn needs stats too)
        raise ValueError("Historical returns required (except in mixture where regimes supply them).")
    R = None if R_hist is None else R_hist.dropna(how="all").dropna(axis=1, how="all").values
    assets = [] if R_hist is None else list(R_hist.columns)

    if mode == "historical":
        out = gen_historical(R, n, h, rng)#type:ignore
    elif mode == "block_bootstrap":
        out = gen_block_bootstrap(R, n, h, block, rng)#type:ignore
    elif mode in ("mvn", "student_t", "pca_mvn"):
        mu, S = cov_from_returns(R, mean_add) if R is not None else (np.zeros(1), np.eye(1))
        S = shrink_cov(S, shrink)
        if mode == "mvn":
            out = gen_mvn(mu, S, n, h, rng)
        elif mode == "student_t":
            out = gen_student_t(mu, S, df, n, h, rng)
        else:
            out = gen_pca_mvn(mu, S, pca_k, n, h, rng)
    elif mode == "gaussian_copula":
        out = gen_gaussian_copula(R, n, h, rng)#type:ignore
    else:
        raise ValueError("Unknown mode.")
    return out, assets

def main():
    ap = argparse.ArgumentParser(description="Scenario generator (historical, bootstrap, MVN, t, PCA, copula, mixture, shocks)")
    ap.add_argument("--mode", default="mvn",
                    choices=["historical","block_bootstrap","mvn","student_t","pca_mvn","gaussian_copula","mix","shock"])
    ap.add_argument("--base-mode", default="mvn", help="For mix/shock modes: base generator to use.")
    ap.add_argument("--returns", default="", help="Wide CSV Date x Assets of returns (decimals).")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--aggregate", default="geom", choices=["geom","sum"])
    ap.add_argument("--mean", type=float, default=0.0, help="Per-step drift add-back.")
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--df", type=int, default=6)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--pca-k", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--shock", default="", help='Comma list "ASSET:ret" additive per-step shock after generation.')
    # mixture
    ap.add_argument("--mix-returns", default="", help="Comma list of regime return CSVs.")
    ap.add_argument("--mix-weights", default="", help="Comma list of weights summing to 1.")
    ap.add_argument("--outdir", default="out_scen")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    R_hist = read_wide_csv(args.returns) if args.returns else None

    # Mixture prep
    mix_files = None; mix_w = None
    if args.mode == "mix":
        mix_files = [s.strip() for s in args.mix_returns.split(",") if s.strip()]
        mix_w = mixture_weights(args.mix_weights, len(mix_files))

    # Shock-only means: generate with base-mode first
    mode_to_run = args.base_mode if args.mode == "shock" else args.mode

    paths, assets = run_mode(
        mode_to_run, args.base_mode, R_hist,
        n=args.n, h=args.h,
        mean_add=args.mean, shrink=args.shrink, df=args.df, block=args.block, pca_k=args.pca_k,
        rng=rng, mix_files=mix_files, mix_w=mix_w
    )

    if not assets and R_hist is not None:
        assets = list(R_hist.columns)
    if not assets:
        assets = [f"A{i}" for i in range(paths.shape[2])]

    # scale + shocks
    if args.scale != 1.0:
        paths = paths * float(args.scale)
    sh = parse_shocks(args.shock, assets)
    if np.any(sh != 0):
        paths = paths + sh[None, None, :]

    # Write scenarios (long panel)
    n, h, p = paths.shape
    idx = pd.MultiIndex.from_product([range(n), range(h)], names=["scenario","step"])#type:ignore
    df = pd.DataFrame(paths.reshape(n*h, p), index=idx, columns=assets)
    df.to_parquet(outdir / "scenarios.parquet")

    # Aggregated per-scenario returns
    agg = aggregate(paths, args.aggregate)
    agg_df = pd.DataFrame(agg, index=pd.Index(range(n), name="scenario"), columns=assets)
    agg_df.to_csv(outdir / "agg.csv")

    # Summary stats from simulated single-step distribution
    flat = paths.reshape(-1, p)
    mu_sim = flat.mean(0)
    vol_sim = flat.std(0, ddof=1)
    corr = np.corrcoef(flat, rowvar=False)
    stats = {
        "assets": assets,
        "mean_per_step": {a: float(m) for a, m in zip(assets, mu_sim)},
        "vol_per_step": {a: float(v) for a, v in zip(assets, vol_sim)},
        "corr_flat_upper": corr[np.triu_indices(p, 1)].round(6).tolist(),
        "n": int(n), "h": int(h), "mode": args.mode, "base_mode": args.base_mode,
    }
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"[OK] Saved {n} scenarios x {h} steps for {len(assets)} assets to {outdir}")

if __name__ == "__main__":
    main()
