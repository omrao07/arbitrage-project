#!/usr/bin/env python3
"""
multiverse.py — Run a “multiverse” of scenario generators and summarize risk

Why
---
Model risk is real. This tool lets you explore MANY plausible scenario-generation
choices (Gaussian, t, bootstraps, EVT tails, copulas, PCA, etc.) across a grid of
hyperparameters and RNG seeds, then aggregates portfolio risk metrics so you can
see how conclusions vary across the modeling universe.

What it does
------------
- Loads a wide returns matrix (Date x Asset, decimals)
- Optionally loads portfolio weights to aggregate asset scenarios to a single P&L
- Generates scenarios for each choice in a configurable grid:
    * historical            : slice sampling
    * block_bootstrap       : circular block bootstrap
    * mvn                   : multivariate normal (with optional covariance shrink)
    * student_t             : multivariate t
    * pca_mvn               : simulate PCs then reconstruct
    * gaussian_copula       : Gaussian copula + empirical marginals
    * evt                   : EVT-style tail resampling (lower-α tails)
    * t_copula              : t-copula + empirical marginals
- Computes risk metrics per run (VaR, ES, mean/vol, skew/kurt, drawdown, tail corr)
- Writes a tidy results table and (optionally) samples of generated scenarios

Inputs
------
--returns RET.csv               Wide CSV Date x Asset (decimals)
--weights W.csv                 Optional CSV with columns {asset, weight} (sums normalized)
--n 10000                       Scenarios per run
--h 10                          Horizon steps
--modes "mvn,student_t,evt"     Generators to run (comma list)
--seeds "1,2,3"                 RNG seeds to sweep (comma list or single int)
--shrink "0,0.25"               For mvn/student_t/pca_mvn
--df "4,6"                      For student_t and t_copula
--block "20,60"                 For block_bootstrap
--alpha 0.05                    Tail prob for ES/EVT
--pca-k "0,0.8"                 0=all PCs; 0<k<1 keeps fraction of variance; int keeps k comps
--sample-out 0                  Save first K runs’ scenarios to parquet (0 disables)
--aggregate geom|sum            Horizon aggregation method (geometric or sum)
--outdir out_multi

Outputs
-------
- results.csv                   One row per (mode × params × seed) with risk metrics
- sample_scenarios_<i>.parquet  Optional samples of (scenario, step, asset) per first K runs
- config.json                   Reproducibility dump
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from math import erf, sqrt


# ---------------- I/O helpers ----------------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    c0 = df.columns[0].lower()
    if c0 in ("date", "time", "t"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    return df.sort_index()


def read_weights(path: Optional[str], assets: List[str]) -> pd.Series:
    if not path:
        return pd.Series(1.0 / len(assets), index=assets)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    a = cols.get("asset", df.columns[0])
    w = cols.get("weight", df.columns[-1])
    s = pd.Series(df[w].values, index=df[a].astype(str))
    s = s.reindex(assets).fillna(0.0)
    if s.sum() != 0:
        s = s / s.sum()
    return s


# --------------- Math utils -----------------
def norm_cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def chol_psd(S: np.ndarray) -> np.ndarray:
    jitter = 1e-12
    for _ in range(7):
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S = S + jitter * np.eye(S.shape[0])
            jitter *= 10
    w, V = np.linalg.eigh(S)
    w[w < 0] = 0
    return np.linalg.cholesky((V * w) @ V.T + 1e-12 * np.eye(S.shape[0]))


def shrink_cov(S: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return S
    p = S.shape[0]
    tr = np.trace(S) / max(1, p)
    return (1 - alpha) * S + alpha * tr * np.eye(p)


def cov_from_returns(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = R.mean(axis=0)
    S = np.cov(R, rowvar=False, ddof=1)
    return mu, S


def empirical_inv_cdf(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    T, p = x.shape
    qs = np.linspace(0, 1, T, endpoint=False) + 0.5 / T
    xs = np.sort(x, axis=0)
    out = np.empty_like(u)
    for j in range(p):
        out[:, j] = np.interp(u[:, j], qs, xs[:, j], left=xs[0, j], right=xs[-1, j])
    return out


# --------------- Generators ------------------
def gen_historical(R: np.ndarray, n: int, h: int, rng) -> np.ndarray:
    T = R.shape[0]
    if T < h:
        raise ValueError("Historical length < horizon.")
    starts = rng.integers(0, T - h + 1, size=n)
    return np.stack([R[s:s+h] for s in starts], axis=0)


def gen_block_bootstrap(R: np.ndarray, n: int, h: int, block: int, rng) -> np.ndarray:
    T, p = R.shape
    n_blocks = int(np.ceil(h / block))
    paths = np.empty((n, h, p))
    starts = rng.integers(0, T, size=(n, n_blocks))
    for i in range(n):
        segs = []
        for b in range(n_blocks):
            s = starts[i, b]
            idx = (np.arange(block) + s) % T
            segs.append(R[idx])
        paths[i] = np.vstack(segs)[:h]
    return paths


def gen_mvn(mu: np.ndarray, S: np.ndarray, n: int, h: int, rng) -> np.ndarray:
    L = chol_psd(S)
    p = len(mu)
    Z = rng.standard_normal((n*h, p))
    return (Z @ L.T + mu).reshape(n, h, p)


def gen_student_t(mu: np.ndarray, S: np.ndarray, df: int, n: int, h: int, rng) -> np.ndarray:
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance.")
    p = len(mu)
    L = chol_psd(S * (df - 2) / df)
    chi = rng.chisquare(df, size=(n*h, 1))
    Z = rng.standard_normal((n*h, p))
    X = (Z / np.sqrt(chi / df)) @ L.T + mu
    return X.reshape(n, h, p)


def pca_decompose(S: np.ndarray):
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def gen_pca_mvn(mu: np.ndarray, S: np.ndarray, k: float | int, n: int, h: int, rng) -> np.ndarray:
    w, V = pca_decompose(S)
    p = len(mu)
    if isinstance(k, float) and 0 < k < 1:
        cum = np.cumsum(w) / np.sum(w)
        keep = np.searchsorted(cum, k) + 1
    elif isinstance(k, int) and k > 0:
        keep = min(k, p)
    else:
        keep = p
    Vk = V[:, :keep]
    wk = np.maximum(w[:keep], 0)
    Lk = Vk @ np.diag(np.sqrt(wk))
    Z = rng.standard_normal((n*h, keep))
    X = Z @ Lk.T + mu
    return X.reshape(n, h, p)


def gen_gaussian_copula(R: np.ndarray, n: int, h: int, rng) -> np.ndarray:
    T, p = R.shape
    Rcs = (R - R.mean(0)) / (R.std(0, ddof=1) + 1e-12)
    C = np.corrcoef(Rcs, rowvar=False)
    L = chol_psd(C)
    Z = rng.standard_normal((n*h, p)) @ L.T
    U = norm_cdf(Z)
    return empirical_inv_cdf(R, U).reshape(n, h, p)


def gen_evt(R: np.ndarray, n: int, h: int, alpha: float, rng) -> np.ndarray:
    """Resample from lower-alpha tails per asset, i.i.d. over steps."""
    T, p = R.shape
    out = np.empty((n, h, p))
    for j in range(p):
        thr = np.quantile(R[:, j], alpha)
        tail = R[R[:, j] <= thr, j]
        if len(tail) == 0:
            tail = R[:, j]
        out[:, :, j] = rng.choice(tail, size=(n, h), replace=True)
    return out


def gen_t_copula(R: np.ndarray, n: int, h: int, df: int, rng) -> np.ndarray:
    T, p = R.shape
    Rcs = (R - R.mean(0)) / (R.std(0, ddof=1) + 1e-12)
    C = np.corrcoef(Rcs, rowvar=False)
    L = chol_psd(C)
    G = rng.standard_normal((n*h, p)) @ L.T
    chi = rng.chisquare(df, size=(n*h, 1))
    Tz = G / np.sqrt(chi / df)
    # Convert to t CDF via numerical approx using scipy fallback? Avoid dependency: use numpy’s erf for normal only,
    # so we’ll use rank mapping: approximate t-CDF by normal cdf on scaled input (ok for heavy-tail sampling needs).
    U = norm_cdf(Tz)  # surrogate; preserves heavy tails in practice for stress exploration
    return empirical_inv_cdf(R, U).reshape(n, h, p)


# --------------- Aggregation & metrics ---------------
def aggregate(paths: np.ndarray, method: str) -> np.ndarray:
    return np.prod(1 + paths, axis=1) - 1 if method == "geom" else np.sum(paths, axis=1)


def portfolio_agg(paths: np.ndarray, W: np.ndarray) -> np.ndarray:
    # paths: (n,h,p), W: (p,)
    return np.einsum("nhp,p->nh", paths, W)


def var_es(x: np.ndarray, alpha: float) -> Tuple[float, float]:
    q = np.quantile(x, alpha)
    mask = x <= q
    es = float(x[mask].mean()) if mask.any() else float(q)
    return float(q), es


def drawdown_stats(series: np.ndarray) -> Tuple[float, float]:
    """Return worst drawdown and avg drawdown across path-aggregated series interpreted as 1-period PnL stream."""
    # Treat nh aggregated returns (per scenario) as one-period—so drawdown equals min cumulative path across horizon.
    # Here we compute on per-scenario *cumulative* path, then average worst across scenarios.
    # Input expected: shape (n, h) of portfolio step returns
    if series.ndim == 1:
        series = series[None, :]
    worst = []
    avg = []
    for path in series:
        eq = np.cumprod(1 + path)
        peak = np.maximum.accumulate(eq)
        dd = 1 - eq / peak
        worst.append(float(np.max(dd)))
        avg.append(float(np.mean(dd)))
    return float(np.mean(worst)), float(np.mean(avg))


def tail_corr(paths_port: np.ndarray, alpha: float) -> float:
    """Crude tail correlation: corr between step t and t+1 inside tail events (averaged)."""
    n, h = paths_port.shape
    if h < 2:
        return np.nan
    vals = []
    for t in range(h - 1):
        x = paths_port[:, t]
        y = paths_port[:, t + 1]
        qx = np.quantile(x, alpha)
        mask = x <= qx
        if mask.any():
            xs = x[mask] - x[mask].mean()
            ys = y[mask] - y[mask].mean()
            denom = (xs.std(ddof=1) * ys.std(ddof=1) + 1e-12)
            vals.append(float(np.mean(xs * ys) / denom))
    return float(np.mean(vals)) if vals else np.nan


# --------------- Multiverse runner ---------------
def parse_list(s: str, typ=float) -> List:
    if s is None or str(s).strip() == "":
        return []
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def run_once(mode: str, R: np.ndarray, n: int, h: int, rng, params: dict) -> np.ndarray:
    mu, S = cov_from_returns(R)
    if mode == "historical":
        return gen_historical(R, n, h, rng)
    if mode == "block_bootstrap":
        return gen_block_bootstrap(R, n, h, int(params.get("block", 20)), rng)
    if mode == "mvn":
        S2 = shrink_cov(S, float(params.get("shrink", 0.0)))
        return gen_mvn(mu, S2, n, h, rng)
    if mode == "student_t":
        S2 = shrink_cov(S, float(params.get("shrink", 0.0)))
        return gen_student_t(mu, S2, int(params.get("df", 6)), n, h, rng)
    if mode == "pca_mvn":
        S2 = shrink_cov(S, float(params.get("shrink", 0.0)))
        return gen_pca_mvn(mu, S2, params.get("pca_k", 0), n, h, rng)
    if mode == "gaussian_copula":
        return gen_gaussian_copula(R, n, h, rng)
    if mode == "evt":
        return gen_evt(R, n, h, float(params.get("alpha", 0.05)), rng)
    if mode == "t_copula":
        return gen_t_copula(R, n, h, int(params.get("df", 4)), rng)
    raise ValueError(f"Unknown mode '{mode}'")


def main():
    ap = argparse.ArgumentParser(description="Multiverse scenario runner")
    ap.add_argument("--returns", required=True, help="Wide CSV Date x Asset (decimals)")
    ap.add_argument("--weights", default="", help="Optional CSV with columns {asset, weight}")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--modes", default="mvn,student_t,gaussian_copula,block_bootstrap,evt,t_copula,pca_mvn,historical")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--shrink", default="0")
    ap.add_argument("--df", default="4,6")
    ap.add_argument("--block", default="20")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--pca-k", default="0,0.8")
    ap.add_argument("--sample-out", type=int, default=0, help="Save first K runs' scenarios (0 disables)")
    ap.add_argument("--aggregate", choices=["geom", "sum"], default="geom")
    ap.add_argument("--outdir", default="out_multi")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    Rdf = read_wide_csv(args.returns).dropna(how="all", axis=1)
    assets = list(Rdf.columns)
    R = Rdf.values
    W = read_weights(args.weights or None, assets).reindex(assets).values

    # Prepare grids
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = parse_list(args.seeds, int) or [None]
    shrinks = parse_list(args.shrink, float) or [0.0]
    dfs = parse_list(args.df, int) or [6]
    blocks = parse_list(args.block, int) or [20]
    pca_ks_raw = [x.strip() for x in str(args.pca_k).split(",") if x.strip()]
    pca_ks: List[float | int] = []
    for x in pca_ks_raw:
        try:
            if "." in x:
                pca_ks.append(float(x))
            else:
                pca_ks.append(int(x))
        except ValueError:
            pass

    # Build param combos per mode
    universe = []
    for mode in modes:
        if mode in ("mvn", "student_t", "pca_mvn"):
            for s in shrinks:
                if mode == "student_t":
                    for dfv in dfs:
                        universe.append((mode, {"shrink": s, "df": dfv}))
                elif mode == "pca_mvn":
                    for k in (pca_ks or [0]):
                        universe.append((mode, {"shrink": s, "pca_k": k}))
                else:
                    universe.append((mode, {"shrink": s}))
        elif mode == "block_bootstrap":
            for b in blocks:
                universe.append((mode, {"block": b}))
        elif mode == "t_copula":
            for dfv in dfs:
                universe.append((mode, {"df": dfv}))
        elif mode == "evt":
            universe.append((mode, {"alpha": args.alpha}))
        else:
            universe.append((mode, {}))

    # Run
    rows = []
    sample_count = 0
    run_id = 0
    for mode, params in universe:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            paths = run_once(mode, R, n=args.n, h=args.h, rng=rng, params=params)

            # Portfolio aggregation
            port_steps = portfolio_agg(paths, W)  # (n,h)
            port_agg = aggregate(paths=np.einsum("nhp->nhp", paths), method=args.aggregate)  # placeholder to keep signature
            # Actually aggregate portfolio per scenario:
            port_agg = aggregate(port_steps[:, :, None], args.aggregate).squeeze()  # (n,)

            # Metrics
            mean = float(port_agg.mean())
            vol = float(port_agg.std(ddof=1))
            skew = float(((port_agg - mean) ** 3).mean() / ((vol + 1e-12) ** 3))
            kurt = float(((port_agg - mean) ** 4).mean() / ((vol + 1e-12) ** 4)) - 3.0
            q, es = var_es(port_agg, args.alpha)

            # Step-level drawdowns (using per-scenario step returns)
            worst_dd, avg_dd = drawdown_stats(port_steps)

            # Tail serial correlation
            tcorr = tail_corr(port_steps, args.alpha)

            rows.append({
                "run_id": run_id,
                "mode": mode,
                "seed": seed,
                **{f"param_{k}": v for k, v in params.items()},
                "n": int(args.n),
                "h": int(args.h),
                "alpha": float(args.alpha),
                "VaR": float(q),
                "ES": float(es),
                "mean": mean,
                "vol": vol,
                "skew": skew,
                "kurt": kurt,
                "worst_drawdown": worst_dd,
                "avg_drawdown": avg_dd,
                "tail_corr": tcorr,
            })

            # Optional sample export
            if args.sample_out > 0 and sample_count < args.sample_out:
                n_s, h_s, p_s = min(200, paths.shape[0]), paths.shape[1], paths.shape[2]
                sample = paths[:n_s]
                idx = pd.MultiIndex.from_product([range(n_s), range(h_s)], names=["scenario", "step"])
                df = pd.DataFrame(sample.reshape(n_s * h_s, p_s), index=idx, columns=assets)
                df.to_parquet(outdir / f"sample_scenarios_{run_id}.parquet")
                sample_count += 1

            run_id += 1

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "results.csv", index=False)
    (outdir / "config.json").write_text(json.dumps({
        "returns": args.returns,
        "weights": args.weights,
        "n": args.n, "h": args.h,
        "modes": modes, "seeds": seeds,
        "shrink": shrinks, "df": dfs, "block": blocks, "alpha": args.alpha, "pca_k": pca_ks,
        "aggregate": args.aggregate,
    }, indent=2))

    # Console summary: show worst-ES runs
    show = res.sort_values("ES").head(10)
    pd.set_option("display.width", 120)
    print("== Multiverse complete ==")
    print(show[["mode", "seed", "param_shrink", "param_df", "param_block", "param_pca_k", "VaR", "ES", "mean", "vol"]])


if __name__ == "__main__":
    main()
