#!/usr/bin/env python3
"""
counterfactuals.py — Counterfactual analysis for trading & risk

Implements lightweight, dependency-free causal estimators to answer
“what would have happened if…?” using panel (Date x Asset) data.

Modes
-----
1) synth        : Synthetic control for a single treated asset
2) did          : Difference-in-Differences (simple ATT)
3) ipw          : Inverse Propensity Weighting (ATE/ATT)
4) dr           : Doubly-Robust (Augmented IPW)

Inputs (wide CSVs unless specified)
-----------------------------------
Common:
  --returns returns.csv            # (Date x Asset) outcome/return to explain

synth:
  --target AAPL                    # treated asset column name
  --donors "MSFT,GOOGL,AMZN"       # optional; by default uses all others
  --pre "2018-01-01:2020-02-28"    # pre-intervention in returns index
  --post "2020-03-01:2020-12-31"   # post-intervention in returns index
  --ridge 1.0                      # L2 lambda (>=0)
  --fit-intercept                  # include intercept term in synthetic control
  --nonneg                         # nonnegative weights (project to simplex)

did:
  --treated "XLF,XLE"              # comma list of treated assets (columns)
  --controls "SPY,QQQ"             # comma list of controls (or auto = all others)
  --pre "2018-01-01:2019-12-31"
  --post "2020-01-01:2020-12-31"

ipw / dr:
  --outcome returns.csv            # can pass --returns (alias)
  --treat treat.csv                # (Date x Asset) 0/1 indicator of treatment received
  --propensity prop.csv            # (Date x Asset) estimated P(T=1|X) in (0,1)

Outputs (to --outdir)
---------------------
- cf_timeseries.csv         : Actual vs counterfactual (synth) / DiD group means / weighted outcomes
- weights.csv               : Synthetic control donor weights (synth only)
- att_summary.json          : ATT/ATE summaries (and SEs for simple DiD)
- config.json               : Reproducibility

Usage examples
--------------
# Synthetic control for AAPL around 2020-03 shock
python counterfactuals.py --mode synth --returns ret.csv --target AAPL \
  --pre "2017-01-01:2020-02-28" --post "2020-03-01:2021-12-31" --ridge 1.0 --fit-intercept

# Simple DiD across ETF buckets
python counterfactuals.py --mode did --returns etf_ret.csv \
  --treated "XLF,XLE" --controls "SPY,QQQ" \
  --pre "2018-01-01:2019-12-31" --post "2020-01-01:2020-12-31"

# Doubly robust estimate from panel outcomes, treatments and propensities
python counterfactuals.py --mode dr --outcome pnl.csv --treat T.csv --propensity e.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# I/O
# ---------------------------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].lower() in ("date", "time", "t"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    return df.sort_index()


def parse_span(span: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Parse 'YYYY-MM-DD:YYYY-MM-DD' (inclusive bounds)."""
    if not span or ":" not in span:
        raise ValueError("Provide spans as 'start:end' with dates in your CSV index.")
    a, b = span.split(":", 1)
    return pd.to_datetime(a.strip()), pd.to_datetime(b.strip())


def list_from_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# ---------------------------
# Core estimators
# ---------------------------
def ridge_solve(X: np.ndarray, y: np.ndarray, lam: float, fit_intercept: bool) -> Tuple[np.ndarray, float]:
    """Closed-form ridge with optional intercept. Returns (coef, intercept)."""
    if fit_intercept:
        Xc = np.column_stack([np.ones(len(X)), X])
        I = np.eye(Xc.shape[1])
        I[0, 0] = 0.0  # do not penalize intercept
        beta = np.linalg.pinv(Xc.T @ Xc + lam * I) @ (Xc.T @ y)
        b0, w = float(beta[0]), beta[1:]
        return w, b0
    else:
        I = np.eye(X.shape[1])
        w = np.linalg.pinv(X.T @ X + lam * I) @ (X.T @ y)
        return w, 0.0


def project_simplex_nonneg(w: np.ndarray) -> np.ndarray:
    """Project weights to nonnegative and sum to 1 (Euclidean)."""
    # Duchi et al. (2008) projection
    v = np.maximum(w, 0.0)
    if v.sum() <= 0:
        # fallback to uniform
        return np.ones_like(v) / len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def synthetic_control(
    R: pd.DataFrame,
    target: str,
    donors: List[str],
    pre_span: Tuple[pd.Timestamp, pd.Timestamp],
    post_span: Tuple[pd.Timestamp, pd.Timestamp],
    lam: float,
    fit_intercept: bool,
    nonneg: bool,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Fit on PRE, predict counterfactual in POST."""
    if target not in R.columns:
        raise SystemExit(f"--target '{target}' not found in returns columns.")
    donors = [d for d in donors if d != target and d in R.columns]
    if len(donors) == 0:
        raise SystemExit("No valid donors; pass --donors or ensure other columns exist.")
    pre_idx = R.index[(R.index >= pre_span[0]) & (R.index <= pre_span[1])]
    post_idx = R.index[(R.index >= post_span[0]) & (R.index <= post_span[1])]
    if pre_idx.empty or post_idx.empty:
        raise SystemExit("Empty PRE or POST index span.")

    y_pre = R.loc[pre_idx, target].values
    X_pre = R.loc[pre_idx, donors].values
    w, b0 = ridge_solve(X_pre, y_pre, lam=lam, fit_intercept=fit_intercept)
    if nonneg:
        w = project_simplex_nonneg(w)

    # Predict over pre+post for inspection
    idx_all = R.index[(R.index >= pre_span[0]) & (R.index <= post_span[1])]
    X_all = R.loc[idx_all, donors].values
    y_hat = X_all @ w + b0
    y_act = R.loc[idx_all, target].values

    df = pd.DataFrame(
        {"actual": y_act, "counterfactual": y_hat, "effect": y_act - y_hat},
        index=idx_all,
    )

    # Aggregates (ATT over post)
    att_post = float(df.loc[post_idx, "effect"].mean())
    cum_post = float(df.loc[post_idx, "effect"].sum())

    meta = {
        "target": target,
        "donors": donors,
        "ridge_lambda": float(lam),
        "intercept": float(b0),
        "nonneg": bool(nonneg),
        "ATT_post_mean": att_post,
        "ATT_post_cum": cum_post,
        "T_pre": int(len(pre_idx)),
        "T_post": int(len(post_idx)),
    }
    weights = pd.Series(w, index=donors, name="weight")
    return df, weights, meta


def did_simple(
    R: pd.DataFrame, treated: List[str], controls: List[str], pre_span: Tuple[pd.Timestamp, pd.Timestamp], post_span: Tuple[pd.Timestamp, pd.Timestamp]
) -> Tuple[pd.DataFrame, dict]:
    """Classic 2x2 DiD ATT = (E[Y_T,post]-E[Y_T,pre]) - (E[Y_C,post]-E[Y_C,pre])."""
    for g in treated + controls:
        if g not in R.columns:
            raise SystemExit(f"Column '{g}' not found in returns.")

    pre_idx = R.index[(R.index >= pre_span[0]) & (R.index <= pre_span[1])]
    post_idx = R.index[(R.index >= post_span[0]) & (R.index <= post_span[1])]
    if pre_idx.empty or post_idx.empty:
        raise SystemExit("Empty PRE or POST index span.")

    T_pre = R.loc[pre_idx, treated].mean(axis=1)
    C_pre = R.loc[pre_idx, controls].mean(axis=1)
    T_post = R.loc[post_idx, treated].mean(axis=1)
    C_post = R.loc[post_idx, controls].mean(axis=1)

    did_att = (T_post.mean() - T_pre.mean()) - (C_post.mean() - C_pre.mean())

    # Simple (naive) SE using pooled variance of differences across time (placebo standard error)
    diff_T = T_post.values - T_pre.mean()
    diff_C = C_post.values - C_pre.mean()
    diff_gap = diff_T - diff_C
    se = float(np.std(diff_gap, ddof=1) / np.sqrt(len(diff_gap))) if len(diff_gap) > 1 else np.nan

    df_ts = pd.DataFrame(
        {
            "treated_mean": pd.concat([T_pre, T_post]).sort_index(),
            "control_mean": pd.concat([C_pre, C_post]).sort_index(),
        }
    )
    df_ts["gap"] = df_ts["treated_mean"] - df_ts["control_mean"]

    meta = {
        "ATT_DiD": float(did_att),
        "SE_approx": se,
        "pre_span": [str(pre_span[0].date()), str(pre_span[1].date())],
        "post_span": [str(post_span[0].date()), str(post_span[1].date())],
        "N_treated": int(len(treated)),
        "N_controls": int(len(controls)),
        "T_pre": int(len(pre_idx)),
        "T_post": int(len(post_idx)),
    }
    return df_ts, meta


def ipw_estimate(Y: pd.DataFrame, Tm: pd.DataFrame, e: pd.DataFrame) -> dict:
    """Inverse propensity weighting ATE & ATT on stacked panel."""
    # Align and stack
    idx = Y.index.intersection(Tm.index).intersection(e.index)
    cols = list(set(Y.columns) & set(Tm.columns) & set(e.columns))
    if not cols:
        raise SystemExit("No common assets across outcome/treatment/propensity.")
    Yp = Y.loc[idx, cols].stack().rename("Y")
    Tp = Tm.loc[idx, cols].stack().rename("T").astype(float)
    ep = e.loc[idx, cols].stack().rename("e").clip(1e-6, 1 - 1e-6)

    # ATE
    w1 = Tp / ep
    w0 = (1 - Tp) / (1 - ep)
    mu1 = float((w1 * Yp).sum() / (w1.sum() + 1e-12))
    mu0 = float((w0 * Yp).sum() / (w0.sum() + 1e-12))
    ate = mu1 - mu0

    # ATT (weight controls to treated)
    mask_t = Tp == 1
    mu1_att = float(Yp[mask_t].mean()) if mask_t.any() else np.nan
    wc = ((1 - Tp) * ep / (1 - ep)).where(~mask_t, 0.0)
    mu0_att = float((wc * Yp).sum() / (wc.sum() + 1e-12)) if wc.sum() > 0 else np.nan
    att = mu1_att - mu0_att if (not np.isnan(mu1_att) and not np.isnan(mu0_att)) else np.nan

    return {
        "mu1_ipw": mu1,
        "mu0_ipw": mu0,
        "ATE_ipw": ate,
        "ATT_ipw": att,
        "N": int(len(Yp)),
    }


def dr_estimate(Y: pd.DataFrame, Tm: pd.DataFrame, e: pd.DataFrame) -> dict:
    """Doubly-robust ATE via outcome regression + IPW correction (linear OR per asset & date-wise pooled)."""
    idx = Y.index.intersection(Tm.index).intersection(e.index)
    cols = list(set(Y.columns) & set(Tm.columns) & set(e.columns))
    if not cols:
        raise SystemExit("No common assets across outcome/treatment/propensity.")
    Yp = Y.loc[idx, cols].stack().rename("Y").astype(float)
    Tp = Tm.loc[idx, cols].stack().rename("T").astype(float)
    ep = e.loc[idx, cols].stack().rename("e").clip(1e-6, 1 - 1e-6)

    # Simple outcome regression: per asset linear fit of Y ~ T (pooled over time)
    # Then merge predictions back
    df = pd.concat([Yp, Tp, ep], axis=1).dropna()
    # Fit by asset groups (use MultiIndex .index.get_level_values)
    assets = df.index.get_level_values(1).unique()
    m1 = []
    m0 = []
    for a in assets:
        sub = df.xs(a, level=1)
        if sub.empty:
            continue
        # OLS coef for Y = b0 + b1*T
        X = np.column_stack([np.ones(len(sub)), sub["T"].values])
        beta = np.linalg.lstsq(X, sub["Y"].values, rcond=None)[0]
        b0, b1 = float(beta[0]), float(beta[1])
        # Predict potential outcomes
        y1hat = b0 + b1 * 1.0
        y0hat = b0 + b1 * 0.0
        m1.append(pd.Series(y1hat, index=sub.index, name=a))
        m0.append(pd.Series(y0hat, index=sub.index, name=a))

    if not m1:
        # fallback to IPW only
        return ipw_estimate(Y, Tm, e)

    m1 = pd.concat(m1, axis=1)
    m0 = pd.concat(m0, axis=1)
    # Align to df index, stack to vector
    m1v = m1.stack().reindex(df.index, fill_value=m1.values.mean())
    m0v = m0.stack().reindex(df.index, fill_value=m0.values.mean())

    # ATE DR estimator: mean( m1 - m0 + T*(Y-m1)/e - (1-T)*(Y-m0)/(1-e) )
    term = (m1v - m0v) + df["T"] * (df["Y"] - m1v) / df["e"] - (1 - df["T"]) * (df["Y"] - m0v) / (1 - df["e"])
    ate = float(term.mean())

    # ATT variant (weight controls toward treated): use Hajek stabilized
    mask_t = df["T"] == 1
    mu1_att = float(df.loc[mask_t, "Y"].mean()) if mask_t.any() else np.nan
    # Counterfactual for treated under control:
    cf0_t = m0v.loc[mask_t] + (df.loc[mask_t, "T"] - df.loc[mask_t, "e"]) * (df.loc[mask_t, "Y"] - m0v.loc[mask_t]) / (df.loc[mask_t, "e"])
    mu0_att = float(cf0_t.mean()) if mask_t.any() else np.nan
    att = mu1_att - mu0_att if (not np.isnan(mu1_att) and not np.isnan(mu0_att)) else np.nan

    return {
        "ATE_dr": ate,
        "ATT_dr": att,
        "N": int(len(df)),
    }


# ---------------------------
# CLI / Orchestration
# ---------------------------
@dataclass
class Config:
    mode: str
    returns: Optional[str]
    outcome: Optional[str]
    target: Optional[str]
    donors: Optional[str]
    treated: Optional[str]
    controls: Optional[str]
    pre: Optional[str]
    post: Optional[str]
    ridge: float
    fit_intercept: bool
    nonneg: bool
    treat_csv: Optional[str]
    prop_csv: Optional[str]
    outdir: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactual analysis (synthetic control, DiD, IPW, DR)")
    p.add_argument("--mode", choices=["synth", "did", "ipw", "dr"], default="synth")

    # Common / synth / did:
    p.add_argument("--returns", default="", help="Wide CSV Date x Asset of outcome/returns.")
    p.add_argument("--target", default="", help="[synth] Treated asset column.")
    p.add_argument("--donors", default="", help="[synth] Comma list of donor assets.")
    p.add_argument("--pre", default="", help="[synth/did] 'start:end' span in index.")
    p.add_argument("--post", default="", help="[synth/did] 'start:end' span in index.")
    p.add_argument("--ridge", type=float, default=1.0, help="[synth] Ridge lambda (>=0).")
    p.add_argument("--fit-intercept", action="store_true", help="[synth] Include intercept.")
    p.add_argument("--nonneg", action="store_true", help="[synth] Nonnegative donor weights projected to simplex.")

    p.add_argument("--treated", default="", help="[did] Comma list of treated asset columns.")
    p.add_argument("--controls", default="", help="[did] Comma list of control asset columns.")

    # IPW / DR:
    p.add_argument("--outcome", default="", help="[ipw/dr] Wide CSV of outcomes (if omitted uses --returns).")
    p.add_argument("--treat", default="", help="[ipw/dr] Wide CSV of treatment indicators (0/1).")
    p.add_argument("--propensity", default="", help="[ipw/dr] Wide CSV of propensities in (0,1).")

    p.add_argument("--outdir", default="cf_out")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        mode=args.mode,
        returns=args.returns or None,
        outcome=args.outcome or (args.returns or None),
        target=args.target or None,
        donors=args.donors or None,
        treated=args.treated or None,
        controls=args.controls or None,
        pre=args.pre or None,
        post=args.post or None,
        ridge=max(0.0, float(args.ridge)),
        fit_intercept=bool(args.fit_intercept),
        nonneg=bool(args.nonneg),
        treat_csv=args.treat or None,
        prop_csv=args.propensity or None,
        outdir=args.outdir,
    )
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Dispatch
    if cfg.mode == "synth":
        if not cfg.returns or not cfg.target or not cfg.pre or not cfg.post:
            raise SystemExit("For --mode synth: provide --returns, --target, --pre, --post.")
        R = read_wide_csv(cfg.returns)
        donors = list_from_csv_arg(cfg.donors) if cfg.donors else [c for c in R.columns if c != cfg.target]
        pre_span = parse_span(cfg.pre)
        post_span = parse_span(cfg.post)
        df_ts, wts, meta = synthetic_control(
            R, cfg.target, donors, pre_span, post_span, lam=cfg.ridge, fit_intercept=cfg.fit_intercept, nonneg=cfg.nonneg
        )
        df_ts.to_csv(outdir / "cf_timeseries.csv")
        wts.to_csv(outdir / "weights.csv", header=True)
        (outdir / "att_summary.json").write_text(json.dumps(meta, indent=2))

    elif cfg.mode == "did":
        if not cfg.returns or not cfg.pre or not cfg.post:
            raise SystemExit("For --mode did: provide --returns, --pre, --post, and treated/controls.")
        R = read_wide_csv(cfg.returns)
        treated = list_from_csv_arg(cfg.treated) if cfg.treated else []
        controls = list_from_csv_arg(cfg.controls) if cfg.controls else [c for c in R.columns if c not in treated]
        if not treated or not controls:
            raise SystemExit("Provide --treated and at least one control (via --controls or auto-others).")
        pre_span = parse_span(cfg.pre)
        post_span = parse_span(cfg.post)
        df_ts, meta = did_simple(R, treated, controls, pre_span, post_span)
        df_ts.to_csv(outdir / "cf_timeseries.csv")
        (outdir / "att_summary.json").write_text(json.dumps(meta, indent=2))

    elif cfg.mode in ("ipw", "dr"):
        if not cfg.outcome or not cfg.treat_csv or not cfg.prop_csv:
            raise SystemExit(f"For --mode {cfg.mode}: provide --outcome (or --returns), --treat, and --propensity.")
        Y = read_wide_csv(cfg.outcome)
        Tm = read_wide_csv(cfg.treat_csv)
        e = read_wide_csv(cfg.prop_csv)
        # Align on common index/columns for a clean export
        idx = Y.index.intersection(Tm.index).intersection(e.index)
        cols = sorted(list(set(Y.columns) & set(Tm.columns) & set(e.columns)))
        Y = Y.loc[idx, cols]; Tm = Tm.loc[idx, cols]; e = e.loc[idx, cols]

        res = dr_estimate(Y, Tm, e) if cfg.mode == "dr" else ipw_estimate(Y, Tm, e)
        # For visibility, export stabilized weights time series aggregates
        Tp = Tm.stack().rename("T").astype(float)
        ep = e.stack().rename("e").clip(1e-6, 1 - 1e-6)
        w_stab = (Tp / ep).rename("w1") + ((1 - Tp) / (1 - ep)).rename("w0")
        ts = pd.concat([Tp, ep, w_stab], axis=1).groupby(level=0).mean()  # average per date
        ts.to_csv(outdir / "cf_timeseries.csv")
        (outdir / "att_summary.json").write_text(json.dumps(res, indent=2))

    else:
        raise SystemExit("Unknown mode.")

    # Save config for reproducibility
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("== Counterfactual analysis complete ==")
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
