#!/usr/bin/env python3
"""
var_es.py — Parametric & historical Value-at-Risk (VaR) and Expected Shortfall (ES)

Supports:
- Historical simulation (quantile of empirical P&L)
- Parametric Normal VaR/ES
- Parametric Student-t VaR/ES
- Delta-normal VaR (using portfolio weights * cov)
- Per-asset and portfolio aggregation

Inputs
------
--returns returns.csv         Wide CSV Date x Asset (decimal returns)
--weights weights.csv         CSV with columns {asset, weight} (default = equal weight)
--alpha 0.05                  Confidence level (tail probability)
--method hist|normal|t|delta
--horizon 1                   Horizon in steps (scales variance by sqrt(h))
--df 6                        Degrees of freedom for Student-t
--outdir out_var

Outputs
-------
- var_es.json         : Portfolio-level VaR & ES
- var_es_by_asset.csv : Per-asset VaR & ES
- distr.csv           : Distribution of portfolio returns (for hist)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import norm, t


# ---------- I/O ----------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].lower() in ("date","time"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    return df.sort_index()


def read_weights(path: str, assets: list[str]) -> pd.Series:
    if not path:
        return pd.Series(1.0/len(assets), index=assets)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    a = cols.get("asset", df.columns[0])
    w = cols.get("weight", df.columns[-1])
    s = pd.Series(df[w].values, index=df[a].astype(str))
    s = s.reindex(assets).fillna(0.0)
    if s.sum() != 0:
        s = s / s.sum()
    return s


# ---------- VaR/ES ----------
def hist_var_es(port: np.ndarray, alpha: float) -> tuple[float,float]:
    q = np.quantile(port, alpha)
    es = port[port <= q].mean() if (port<=q).any() else q
    return q, es#type:ignore

def norm_var_es(mu: float, sigma: float, alpha: float, horizon: int) -> tuple[float,float]:
    mu_h = mu * horizon
    sigma_h = sigma * sqrt(horizon)
    q = mu_h + sigma_h * norm.ppf(alpha)
    es = mu_h - sigma_h * norm.pdf(norm.ppf(alpha))/alpha
    return q, es#type:ignore

def t_var_es(mu: float, sigma: float, df: int, alpha: float, horizon: int) -> tuple[float,float]:
    mu_h = mu * horizon
    sigma_h = sigma * sqrt(horizon)
    q = mu_h + sigma_h * t.ppf(alpha, df)
    x = t.ppf(alpha, df)
    es = mu_h + sigma_h * ( (t.pdf(x, df))*(df + x**2 -1)/((df-1)*alpha) )
    return q, es#type:ignore


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--method", choices=["hist","normal","t","delta"], default="hist")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--df", type=int, default=6)
    ap.add_argument("--outdir", default="out_var")
    args = ap.parse_args()

    R = read_wide_csv(args.returns)
    assets = list(R.columns)
    W = read_weights(args.weights, assets)

    port_ret = (R[assets] @ W).values

    if args.method == "hist":
        q, es = hist_var_es(port_ret, args.alpha)#type:ignore
    elif args.method == "normal":
        mu, sigma = port_ret.mean(), port_ret.std(ddof=1)#type:ignore
        q, es = norm_var_es(mu, sigma, args.alpha, args.horizon)
    elif args.method == "t":#type:ignore
        mu, sigma = port_ret.mean(), port_ret.std(ddof=1)#type:ignore
        q, es = t_var_es(mu, sigma, args.df, args.alpha, args.horizon)
    else:  # delta-normal
        mu_vec = R.mean().values
        cov = R.cov().values#type:ignore
        mu_p = float(W @ mu_vec)#type:ignore
        var_p = float(W @ cov @ W)
        sigma_p = sqrt(var_p)
        q, es = norm_var_es(mu_p, sigma_p, args.alpha, args.horizon)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Portfolio metrics
    results = {"alpha": args.alpha, "VaR": float(q), "ES": float(es), "method": args.method}
    (outdir/"var_es.json").write_text(json.dumps(results, indent=2))

    # Per-asset hist (optional)
    if args.method == "hist":
        q_a, es_a = {}, {}
        for a in assets:
            qa, esa = hist_var_es(R[a].values, args.alpha)#type:ignore
            q_a[a], es_a[a] = qa, esa
        df = pd.DataFrame({"VaR": q_a, "ES": es_a})
        df.to_csv(outdir/"var_es_by_asset.csv")
        pd.DataFrame({"port_ret": port_ret}).to_csv(outdir/"distr.csv", index=False)

    print("== VaR/ES Results ==")
    print(results)


if __name__ == "__main__":
    main()
