#!/usr/bin/env python3
"""
tail_ai.py — Tail-risk stress-testing with AI-inspired heavy-tail models

Purpose
-------
- Generate and apply *tail-risk* scenarios for portfolio stress tests
- Models supported:
    * EVT (Extreme Value Theory) tail resampling
    * Block bootstrap focusing on worst blocks
    * t-copula with low df (heavy tails)
    * GAN-like mixture: resample from extreme quantile empirical distribution
- Computes expected shortfall (CVaR), tail correlation, and conditional drawdown
- Flexible: works with returns CSV or simulated shocks

Inputs
------
--returns returns.csv         Wide CSV Date x Asset (decimal returns)
--mode evt|t_copula|block|mix
--n 5000                      Number of scenarios
--h 10                        Horizon steps
--alpha 0.05                  Tail probability (default 5%)
--df 4                        Degrees of freedom for t-copula
--block 20                    Block size for block bootstrap
--seed 123                    RNG seed
--outdir out_tail             Output directory

Outputs
-------
- scenarios.parquet           Panel (scenario, step, asset) -> return
- agg.csv                     Aggregated per-scenario returns
- tail_metrics.json           Expected shortfall, tail correlation matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from math import erf, sqrt


# ---------- Helpers ----------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].lower() in ("date","time"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    return df.sort_index()

def norm_cdf(x): return 0.5 * (1 + erf(x/sqrt(2)))
def norm_ppf(u):
    # Approximate inverse normal CDF
    from scipy.stats import norm
    return norm.ppf(u)


# ---------- Models ----------
def gen_evt(R: np.ndarray, n: int, h: int, alpha: float, rng) -> np.ndarray:
    """Resample from lower alpha-quantile tails per asset, iid across time."""
    T, p = R.shape
    out = np.empty((n,h,p))
    for j in range(p):
        thr = np.quantile(R[:,j], alpha)
        tail = R[R[:,j] <= thr, j]
        if len(tail)==0: tail = R[:,j]
        out[:,:,j] = rng.choice(tail, size=(n,h), replace=True)
    return out

def gen_block(R: np.ndarray, n: int, h: int, block: int, alpha: float, rng) -> np.ndarray:
    """Block bootstrap with preference for worst blocks (tail bias)."""
    T, p = R.shape
    paths = np.empty((n,h,p))
    n_blocks = int(np.ceil(h/block))
    # compute block losses
    losses = []
    for s in range(T-block):
        block_ret = R[s:s+block].sum(0).sum()
        losses.append((block_ret, s))
    losses = sorted(losses)[:max(10, int(alpha*len(losses)))]
    starts_pool = [s for _,s in losses]
    for i in range(n):
        segs=[]
        for b in range(n_blocks):
            s = rng.choice(starts_pool)
            segs.append(R[s:s+block])
        path = np.vstack(segs)[:h]
        paths[i]=path
    return paths

def gen_tcopula(R: np.ndarray, n: int, h: int, df: int, rng) -> np.ndarray:
    """Generate via t-copula preserving correlation structure."""
    from scipy.stats import t
    T,p=R.shape
    Rcs=(R-R.mean(0))/R.std(0,ddof=1)
    C=np.corrcoef(Rcs,rowvar=False)
    L=np.linalg.cholesky(C+1e-12*np.eye(p))
    U=rng.standard_normal((n*h,p)) @ L.T
    chi=rng.chisquare(df, size=(n*h,1))
    Tvar=U/np.sqrt(chi/df)
    Ucdf=t.cdf(Tvar,df)
    out=np.empty((n*h,p))
    for j in range(p):
        out[:,j]=np.quantile(R[:,j], Ucdf[:,j])
    return out.reshape(n,h,p)

def gen_mix(R: np.ndarray, n: int, h: int, alpha: float, rng) -> np.ndarray:
    """Mixture: half normal resample, half tail resample."""
    T,p=R.shape
    out=np.empty((n,h,p))
    for i in range(n):
        for t in range(h):
            if rng.random()<0.5:
                out[i,t]=R[rng.integers(0,T)]
            else:
                thr=np.quantile(R,alpha,axis=0)
                mask=(R<=thr).any(1)
                tail=R[mask]
                if len(tail)==0: tail=R
                out[i,t]=tail[rng.integers(0,len(tail))]
    return out

def aggregate(paths: np.ndarray, method="geom")->np.ndarray:
    return np.prod(1+paths,1)-1 if method=="geom" else paths.sum(1)


# ---------- Tail metrics ----------
def tail_metrics(agg: np.ndarray, alpha: float, assets: List[str]):#type:ignore
    # agg: (n,p)
    q=np.quantile(agg,alpha,0)
    es=agg[agg<=q].mean(0)
    corr=np.corrcoef(agg[agg<=q].T)
    return {
        "ES":{a:float(es[i]) for i,a in enumerate(assets)},
        "q":{a:float(q[i]) for i,a in enumerate(assets)},
        "tail_corr":corr.tolist()
    }


# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--returns",required=True)
    ap.add_argument("--mode",choices=["evt","block","t_copula","mix"],default="evt")
    ap.add_argument("--n",type=int,default=5000)
    ap.add_argument("--h",type=int,default=10)
    ap.add_argument("--alpha",type=float,default=0.05)
    ap.add_argument("--df",type=int,default=4)
    ap.add_argument("--block",type=int,default=20)
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--outdir",default="out_tail")
    args=ap.parse_args()

    rng=np.random.default_rng(args.seed)
    R=read_wide_csv(args.returns).dropna(how="all",axis=1).values
    assets=list(read_wide_csv(args.returns).columns)

    if args.mode=="evt":
        paths=gen_evt(R,args.n,args.h,args.alpha,rng)
    elif args.mode=="block":
        paths=gen_block(R,args.n,args.h,args.block,args.alpha,rng)
    elif args.mode=="t_copula":
        paths=gen_tcopula(R,args.n,args.h,args.df,rng)
    else:
        paths=gen_mix(R,args.n,args.h,args.alpha,rng)

    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    n,h,p=paths.shape
    idx=pd.MultiIndex.from_product([range(n),range(h)],names=["scenario","step"])#type:ignore
    df=pd.DataFrame(paths.reshape(n*h,p),index=idx,columns=assets)
    df.to_parquet(outdir/"scenarios.parquet")

    agg=aggregate(paths)
    pd.DataFrame(agg,columns=assets).to_csv(outdir/"agg.csv",index=False)

    metrics=tail_metrics(agg,args.alpha,assets)
    (outdir/"tail_metrics.json").write_text(json.dumps(metrics,indent=2))

    print(f"[OK] Tail scenarios saved: {args.outdir}")

if __name__=="__main__":
    main()
