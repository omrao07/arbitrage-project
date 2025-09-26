#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uk_gilts_vs_gbp.py — Rates/FX co-movement toolkit (UK gilts vs GBP)

What this does
--------------
Given UK gilt yields (or futures returns) and GBP FX (GBPUSD/GBPEUR), this script:

1) Cleans & aligns time series (daily or monthly)
2) Builds yield-change series (Δy in bp) per tenor and an optional PCA "rate shock"
3) Computes rolling regressions: FX % change ~ β * Δy (bp) [+ intercept]
4) Estimates a small VAR and:
    - chooses lag order (AIC)
    - computes impulse responses (IRFs)
    - reports Granger-causality F-stats (p-values if SciPy present)
5) Derives rolling hedge ratios to offset FX with rates (min-variance hedge)
6) Optional event studies around BoE MPC dates (or any provided events)
7) Optional scenario mapping: apply shock.bp to yields → expected FX move from latest β/VAR

Outputs (CSV / JSON)
--------------------
- aligned_panel.csv         Aligned daily/monthly panel (Δy_bp by tenor, PCA1_bp, FX returns %)
- rolling_beta.csv          Rolling OLS β (FX% per 10bp) by tenor/series (+ R², stderr)
- hedge_ratio.csv           Rolling hedge ratio h* ( FX + h*Δy ) variance-minimizing
- var_summary.json          VAR order, AIC table, residual stats, (optional) Granger p-values
- irf.csv                   IRFs for shocks to Δy (and ΔFX) over the horizon
- scatter_last_window.csv   Last-window scatter of Δy vs FX% for quick plotting
- event_study.csv           Mean/median around events (if --events provided)
- scenarios_out.csv         Scenario results (expected FX move from β and VAR)
- summary.json              Key KPIs (last β, sign, IRF(Δy→FX) first steps, etc.)
- config.json               Run configuration (for reproducibility)

Inputs (CSV; flexible, case-insensitive)
----------------------------------------
--gilts gilts.csv   REQUIRED
  Either long or wide. We accept:
   A) Long: date, tenor, yield (in %, or bp), [return_pct optional if futures]
   B) Wide: date, y2y, y5y, y10y (or any cols containing '2', '5', '10')
  We compute Δy in basis points per tenor.

--fx fx.csv         REQUIRED
  Either long or wide:
   A) Long: date, series, value (levels). series in {GBPUSD, GBPEUR, ...}
   B) Wide: date, GBPUSD, GBPEUR, ...
  We compute FX return % = 100*Δlog(level).

--events events.csv  OPTIONAL
  Columns: date, label   (BoE MPC dates etc.) — event window controlled by --event_k

--scenarios scenarios.csv OPTIONAL
  Columns: scenario, name, key, value
  Keys:
    shock.bp = -25              (apply to chosen tenor; negative = yields down)
    shock.tenor = 10            (2, 5, 10; default = --tenor)
    method = beta|var           (use rolling β or VAR IRF mapping; default beta)
    horizon = 10                (IRF horizon, days/months depending on frequency)

CLI
---
--tenor 10                      Tenor to highlight for β/VAR (2/5/10)
--freq D                        'D' (daily) or 'M' (monthly, calendar month)
--window 60                     Rolling window length (obs)
--pmax 10                       VAR max lags
--irf_h 20                      IRF horizon
--event_k 3                     Event study window ±k
--start 2015-01-01
--end   2025-12-31
--outdir out_gilts_gbp

Notes
-----
- FX % is 100*Δlog; yields are in basis point changes (Δy_bp).
- β is reported as "% FX move per 10bp yield change" to be interpretable.
- VAR uses simple OLS with deterministic intercept; Cholesky identification with ordering [Δy, Δfx] by default
  (i.e., yields contemporaneously affect FX, not vice versa). You can change by --order fx_first.
- Granger p-values require SciPy; otherwise F-stat & dof are reported.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def ensure_dir(p: str) -> Path:
    pp = Path(p); pp.mkdir(parents=True, exist_ok=True); return pp

def pct_change_log(x: pd.Series) -> pd.Series:
    return np.log(x.astype(float).replace(0, np.nan)).diff() * 100.0

def diff_bp(x: pd.Series) -> pd.Series:
    # If looks like percent (e.g., 3.50), convert Δ% * 100 → bp; if already bp-scale (hundreds), just diff
    s = x.astype(float)
    # Heuristic: median level < 30 → percentage points; else already bp
    if s.dropna().median() < 30:
        return s.diff() * 100.0
    return s.diff()

def rolling_ols(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    """
    Returns β (per 10bp), stderr, r2 for rolling windows: y = a + b*x + e
    x expected in bp; y in %.
    """
    yv = y.values.astype(float); xv = x.values.astype(float)
    n = len(yv)
    rows = []
    for t in range(window, n):
        Y = yv[t-window:t]
        X = xv[t-window:t]
        mask = np.isfinite(Y) & np.isfinite(X)
        if mask.sum() < max(10, int(window*0.6)):
            rows.append((np.nan, np.nan, np.nan))
            continue
        X1 = np.column_stack([np.ones(mask.sum()), X[mask]])
        beta, *_ = np.linalg.lstsq(X1, Y[mask], rcond=None)
        yhat = X1 @ beta
        resid = Y[mask] - yhat
        s2 = (resid @ resid) / max(1, (mask.sum() - 2))
        XtX_inv = np.linalg.inv(X1.T @ X1)
        se_b = np.sqrt(s2 * XtX_inv[1,1])  # slope stderr
        r2 = 1.0 - (resid @ resid) / max(1e-12, ((Y[mask] - Y[mask].mean())**2).sum())
        # report per 10bp
        rows.append((beta[1]*10.0, se_b*10.0, r2))
    out = pd.DataFrame(rows, columns=["beta_per10bp","stderr_per10bp","r2"], index=y.index[window:n])
    return out

def pca_1(comp_df: pd.DataFrame) -> pd.Series:
    # PCA on standardized columns; return PC1 scaled back to bp (weighted combination of bp changes)
    X = comp_df.dropna().values
    if X.shape[0] < 20:  # not enough points
        return pd.Series(index=comp_df.index, dtype=float)
    # standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Z = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pc1 = Z @ Vt.T[:,0]
    pc1_s = pd.Series( (pc1 - pc1.mean()) / (pc1.std(ddof=0)+1e-12), index=comp_df.dropna().index )
    # scale roughly to bp by matching variance to mean variance of inputs
    scale = np.mean(sd)  # heuristic
    out = pd.Series(index=comp_df.index, dtype=float)
    out.loc[pc1_s.index] = pc1_s * scale
    return out

def aic_of(resid: np.ndarray, k_params: int) -> float:
    n = resid.shape[0]
    sse = float(np.sum(resid**2))
    sigma2 = sse / n if n>0 else np.nan
    return 2*k_params + n*(np.log(2*np.pi*sigma2) + 1)

def try_scipy():
    try:
        import scipy.stats as st  # type: ignore
        return st
    except Exception:
        return None


# ----------------------------- loaders -----------------------------

def load_gilts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={date_c:"date"})
    df["date"] = to_dt(df["date"])
    # Detect long vs wide
    if ncol(df,"tenor") or any(c.lower()=="tenor" for c in df.columns):
        # Long
        ten = ncol(df,"tenor") or "tenor"
        val = ncol(df,"yield") or ncol(df,"level") or ncol(df,"y") or "yield"
        df = df.rename(columns={ten:"tenor", val:"yield"})
        df["tenor"] = df["tenor"].astype(str).str.extract(r'(\d+)').astype(float)
        # pivot to wide yields
        Y = df.pivot_table(index="date", columns="tenor", values="yield", aggfunc="last").sort_index()
        Y.columns = [f"y{int(c)}" for c in Y.columns]
    else:
        # Wide; keep columns with 'y' or containing 2/5/10
        Y = df.set_index("date").copy()
        # normalize column names
        ren = {}
        for c in Y.columns:
            lc = str(c).lower()
            if "2" in lc and "y" in lc: ren[c] = "y2"
            elif "5" in lc and "y" in lc: ren[c] = "y5"
            elif "10" in lc and "y" in lc: ren[c] = "y10"
            else:
                ren[c] = c
        Y = Y.rename(columns=ren)[[c for c in ren.values() if c in ["y2","y5","y10"]]]
    # Δy in bp
    Dy = Y.apply(diff_bp)
    Dy = Dy.add_suffix("_bp")
    out = pd.concat([Y, Dy], axis=1).reset_index().rename(columns={"index":"date"})
    return out.sort_values("date")

def load_fx(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={date_c:"date"})
    df["date"] = to_dt(df["date"])
    if ncol(df,"series"):
        ser = ncol(df,"series") or "series"
        val = ncol(df,"value") or ncol(df,"level") or "value"
        df = df.rename(columns={ser:"series", val:"value"})
        wide = df.pivot_table(index="date", columns="series", values="value", aggfunc="last").sort_index()
    else:
        wide = df.set_index("date").copy()
    # compute returns
    out = wide.copy()
    for c in wide.columns:
        out[f"{c}_ret_pct"] = pct_change_log(wide[c])
    out = out.reset_index()
    return out.sort_values("date")

def resample_freq(df: pd.DataFrame, freq: str, how="last") -> pd.DataFrame:
    if freq.upper().startswith("D"):
        return df
    # Monthly
    g = (df.set_index("date").groupby(pd.Grouper(freq="M")))
    if how=="mean":
        out = g.mean(numeric_only=True)
    else:
        out = g.last()
    out = out.reset_index()
    return out


# ----------------------------- VAR (small, numpy) -----------------------------

def build_lagmat(X: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given Y_t (n x k) as X, build VAR(p) lag matrix.
    Returns (Y_trim, Z) where
      Y_trim: (n-p) x k  (current)
      Z:      (n-p) x (k*p + 1)  [intercept + stacked lags]
    """
    n, k = X.shape
    Y = X[p:,:]
    Zlags = []
    for i in range(1, p+1):
        Zlags.append(X[p-i:n-i, :])
    Z = np.concatenate(Zlags, axis=1)
    Z = np.column_stack([np.ones(len(Z)), Z])
    return Y, Z

def var_fit(data: pd.DataFrame, cols: List[str], pmax: int=10) -> dict:
    Xraw = data[cols].dropna().values
    if Xraw.shape[0] < (pmax + 20):
        pmax = max(1, min(5, Xraw.shape[0]//4))
    aic_tbl = []
    fits = {}
    for p in range(1, pmax+1):
        Y, Z = build_lagmat(Xraw, p)
        # coefficient matrix B: (k*p+1) x k
        B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        resid = Y - Z @ B
        aic = aic_of(resid, B.size)
        aic_tbl.append((p, aic))
        fits[p] = {"B": B, "resid": resid, "Z": Z, "Y": Y}
    aic_tbl = pd.DataFrame(aic_tbl, columns=["p","aic"]).sort_values("aic")
    p_star = int(aic_tbl.iloc[0]["p"])
    k = len(cols)
    # Companion form
    B = fits[p_star]["B"]
    # Separate intercept and lag blocks
    c = B[0,:]                      # (k,)
    A_blocks = [B[1+i*k:1+(i+1)*k,:].T for i in range(p_star)]  # list of kxk (A1..Ap)
    # Residual covariance
    Sigma = (fits[p_star]["resid"].T @ fits[p_star]["resid"]) / fits[p_star]["resid"].shape[0]
    return {"order": p_star, "aic_table": aic_tbl, "A": A_blocks, "c": c, "Sigma": Sigma, "cols": cols}

def var_irf(model: dict, horizon: int, shock_var: int=0, ordering: str="rates_first") -> pd.DataFrame:
    """
    Cholesky IRF with ordering:
      - "rates_first": [Δy, Δfx]  (shock_var=0 means Δy shock)
      - "fx_first":    [Δfx, Δy]
    """
    A = model["A"]; k = len(model["cols"]); p = len(A)
    # Companion matrix
    C = np.zeros((k*p, k*p))
    # top block
    top = np.concatenate(A, axis=1) if p>0 else np.zeros((k, k*p))
    C[:k, :k*p] = top
    # sub-identity
    if p>1:
        C[k:, :-k] = np.eye(k*(p-1))
    # Cholesky of Sigma with desired ordering
    Sigma = model["Sigma"].copy()
    idx = list(range(k))
    if ordering == "fx_first":
        idx = [1,0] if k==2 else idx
    Sigma_ord = Sigma[np.ix_(idx, idx)]
    try:
        P = np.linalg.cholesky(Sigma_ord)
    except np.linalg.LinAlgError:
        # fallback: eigen
        eigval, eigvec = np.linalg.eigh(Sigma_ord)
        P = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 1e-12))) @ eigvec.T
    # Select shock: unit shock in chosen var (in ordered space)
    e = np.zeros((k,1)); e[shock_var,0] = 1.0
    J = np.concatenate([np.eye(k), np.zeros((k, k*(p-1)))], axis=1)
    # IRF recursion
    irfs = []
    Cpow = np.eye(k*p)
    for h in range(horizon+1):
        Theta_h = J @ Cpow @ J.T @ P  # k x k structural moving-average at horizon h
        # Map back to original variable order
        # If we permuted, undo permutation
        if ordering == "fx_first" and k==2:
            # Our model cols are original [Δy, Δfx]; but P was computed on [Δfx, Δy]
            # Build permutation matrix to place shocks appropriately
            perm = np.array([[0,1],[1,0]])
            Theta_h = perm.T @ Theta_h @ perm
        resp = (Theta_h @ e).flatten()
        irfs.append(resp)
        Cpow = Cpow @ C
    irf_df = pd.DataFrame(irfs, columns=[f"resp_{c}" for c in model["cols"]])
    irf_df.insert(0, "h", np.arange(horizon+1))
    return irf_df


# ----------------------------- analytics -----------------------------

def choose_tenor_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for t in [2,5,10]:
        c = f"y{t}_bp"
        if c in df.columns: cols.append(c)
    return cols

def last_window_scatter(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    Y = y.dropna().tail(window)
    X = x.reindex(Y.index)
    return pd.DataFrame({"date": Y.index, "fx_ret_pct": Y.values, "dy_bp": X.values})

def event_study(panel: pd.DataFrame, events: pd.DataFrame, fx_col: str, dy_col: str, k: int) -> pd.DataFrame:
    if events.empty: return pd.DataFrame()
    # Ensure index by date
    P = panel.set_index("date").sort_index()
    rows = []
    for _, ev in events.iterrows():
        d = pd.to_datetime(ev["date"])
        # Build window
        idx = P.index
        if d not in idx: 
            # nearest prior
            loc = idx.searchsorted(d) - 1
            if loc < 0: continue
            d0 = idx[loc]
        else:
            d0 = d
        window = P.loc[idx[(idx>=d0 - pd.Timedelta(days=k*7)) & (idx<=d0 + pd.Timedelta(days=k*7))]]
        # Align by relative day count
        rel = (window.index - d0).days
        tmp = pd.DataFrame({"rel_day": rel, "fx": window[fx_col].values, "dy_bp": window[dy_col].values})
        tmp["label"] = ev.get("label", "")
        rows.append(tmp)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Aggregate by rel_day
    agg = out.groupby(["label","rel_day"], as_index=False).agg(fx_mean=("fx","mean"), fx_median=("fx","median"),
                                                              dy_bp_mean=("dy_bp","mean"), dy_bp_median=("dy_bp","median"))
    return agg.sort_values(["label","rel_day"])

def scenarios_map(scen_df: pd.DataFrame, latest_beta: float, latest_var_irf: Optional[pd.DataFrame], tenor: int, default_h: int, method: str) -> pd.DataFrame:
    if scen_df.empty:
        return pd.DataFrame()
    rows = []
    for scen in scen_df["scenario"].unique():
        sub = scen_df[scen_df["scenario"]==scen]
        shock_bp = float(sub[sub["key"].str.lower()=="shock.bp"]["value"].iloc[0]) if not sub[sub["key"].str.lower()=="shock.bp"].empty else 0.0
        shock_ten = int(float(sub[sub["key"].str.lower()=="shock.tenor"]["value"].iloc[0])) if not sub[sub["key"].str.lower()=="shock.tenor"].empty else tenor
        meth = str(sub[sub["key"].str.lower()=="method"]["value"].iloc[0]).lower() if not sub[sub["key"].str.lower()=="method"].empty else method
        horizon = int(float(sub[sub["key"].str.lower()=="horizon"]["value"].iloc[0])) if not sub[sub["key"].str.lower()=="horizon"].empty else default_h
        if meth == "beta":
            fx_move_pct = latest_beta * (shock_bp/10.0)  # β per 10bp
            rows.append({"scenario": scen, "method": "beta", "tenor": shock_ten, "shock_bp": shock_bp,
                         "h": 0, "fx_pct": fx_move_pct})
        else:
            # Use VAR IRF: scale IRF path of FX to a 1-s.d. Δy shock; we want shock in bp, so rescale
            if latest_var_irf is None or latest_var_irf.empty:
                continue
            # Column 'resp_<col>'; we expect model cols order ["dy_bp", "fx_ret_pct"]
            path = latest_var_irf.copy()
            # Scale: if IRF computed for 1 s.d. structural shock in Δy, we approx map 1 bp shock by dividing by sd_bp_struct
            # Without structural sd, we approximate linear scaling using 1.0 unit ≈ 1 bp; user should treat as relative.
            # We'll just scale proportional to shock_bp (model unit assumed bp).
            path = path.copy()
            path = path[path["h"]<=horizon]
            for _, r in path.iterrows():
                rows.append({"scenario": scen, "method": "var", "tenor": shock_ten, "shock_bp": shock_bp,
                             "h": int(r["h"]), "fx_pct": float(r["resp_fx_ret_pct"] * (shock_bp))})
    return pd.DataFrame(rows)


# ----------------------------- CLI / main -----------------------------

@dataclass
class Config:
    gilts: str
    fx: str
    events: Optional[str]
    scenarios: Optional[str]
    tenor: int
    freq: str
    window: int
    pmax: int
    irf_h: int
    event_k: int
    order: str
    start: str
    end: str
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="UK gilts vs GBP — co-movement & VAR/IRF toolkit")
    ap.add_argument("--gilts", required=True)
    ap.add_argument("--fx", required=True)
    ap.add_argument("--events", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--tenor", type=int, default=10, help="2/5/10 (years)")
    ap.add_argument("--freq", default="D", help="D or M")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--pmax", type=int, default=10)
    ap.add_argument("--irf_h", type=int, default=20)
    ap.add_argument("--event_k", type=int, default=3)
    ap.add_argument("--order", default="rates_first", choices=["rates_first","fx_first"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--outdir", default="out_gilts_gbp")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load
    G = load_gilts(args.gilts)
    F = load_fx(args.fx)
    E = pd.read_csv(args.events) if args.events else pd.DataFrame()
    if not E.empty:
        E = E.rename(columns={(ncol(E,"date") or E.columns[0]):"date", (ncol(E,"label") or "label"):"label"})
        E["date"] = to_dt(E["date"])

    # Merge panel
    panel = pd.merge(G, F, on="date", how="inner").sort_values("date")
    # Filter & resample
    start = pd.to_datetime(args.start); end = pd.to_datetime(args.end)
    panel = panel[(panel["date"]>=start) & (panel["date"]<=end)].copy()
    panel = resample_freq(panel, args.freq.upper(), how="last")

    # Make composite PCA (optional)
    tenor_cols = choose_tenor_cols(panel)
    if len(tenor_cols) >= 2:
        panel["pca1_bp"] = pca_1(panel[tenor_cols])

    # FX: pick main series
    fx_cols = [c for c in panel.columns if c.endswith("_ret_pct")]
    if not fx_cols:
        raise ValueError("No FX return columns found (need *_ret_pct after loading fx).")
    # prefer GBPUSD_ret_pct if present
    fx_main = "GBPUSD_ret_pct" if "GBPUSD_ret_pct" in fx_cols else fx_cols[0]

    # Choose yield change column
    dy_col = f"y{args.tenor}_bp" if f"y{args.tenor}_bp" in panel.columns else (tenor_cols[-1] if tenor_cols else None)
    if dy_col is None:
        raise ValueError("No yield change column found (y2_bp, y5_bp, y10_bp).")
    # Align core series
    base = panel[["date", fx_main, dy_col] + ([ "pca1_bp" ] if "pca1_bp" in panel.columns else [])].dropna()

    # Save aligned panel (plus everything, for convenience)
    panel.to_csv(outdir / "aligned_panel.csv", index=False)

    # Rolling β
    roll = rolling_ols(base[fx_main], base[dy_col], window=args.window)
    roll = roll.reset_index().rename(columns={"index":"date"})
    roll["series"] = f"{fx_main}_on_{dy_col}"
    roll.to_csv(outdir / "rolling_beta.csv", index=False)

    # Hedge ratio (min variance of FX + h*Δy)
    # h* = - cov(FX, Δy) / var(Δy)
    def rolling_hedge(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
        Y = y.values.astype(float); X = x.values.astype(float)
        n = len(Y)
        rows = []
        for t in range(window, n):
            yy = Y[t-window:t]; xx = X[t-window:t]
            mask = np.isfinite(yy) & np.isfinite(xx)
            if mask.sum() < max(10, int(window*0.6)):
                rows.append(np.nan); continue
            cov = np.cov(yy[mask], xx[mask], ddof=0)[0,1]
            varx = np.var(xx[mask], ddof=0)
            h = - cov / (varx + 1e-12)
            rows.append(h)
        return pd.DataFrame({"date": y.index[window:n], "hedge_h": rows})
    hed = rolling_hedge(base[fx_main], base[dy_col], args.window)
    hed["series"] = f"{fx_main}_vs_{dy_col}"
    hed.to_csv(outdir / "hedge_ratio.csv", index=False)

    # Scatter for last window
    scatter = last_window_scatter(base.set_index("date")[fx_main], base.set_index("date")[dy_col], args.window)
    scatter.to_csv(outdir / "scatter_last_window.csv", index=False)

    # VAR on [Δy, Δfx]
    var_df = base.rename(columns={fx_main:"fx_ret_pct", dy_col:"dy_bp"}).dropna()
    # Ensure enough obs
    var_model = None
    irf_out = pd.DataFrame()
    var_info = {}
    if len(var_df) >= max(80, args.window+20):
        model = var_fit(var_df, ["dy_bp","fx_ret_pct"], pmax=args.pmax)
        var_model = model
        irf = var_irf(model, horizon=args.irf_h, shock_var=0, ordering=args.order)
        irf.to_csv(outdir / "irf.csv", index=False)
        # Granger (optional p-values)
        st = try_scipy()
        gr = {}
        p = model["order"]; k = 2
        # dy -> fx: regress fx on its lags and lags of dy ; restricted: only fx lags
        Xraw = var_df[["dy_bp","fx_ret_pct"]].dropna().values
        Y, Z = build_lagmat(Xraw, p)
        # Z columns: [const, dy(-1..-p), fx(-1..-p)]
        # Unrestricted for fx equation:
        cols_fx = [0] + list(range(1,1+p*k))  # all lags
        Zu = Z[:, cols_fx]
        y_fx = Y[:,1]
        bu, *_ = np.linalg.lstsq(Zu, y_fx, rcond=None)
        resid_u = y_fx - Zu @ bu
        SSE_u = float(resid_u @ resid_u)
        # Restricted: drop dy lags
        Zr = np.column_stack([Z[:,0], Z[:, (1+p):(1+p + p)]])  # const + fx lags
        br, *_ = np.linalg.lstsq(Zr, y_fx, rcond=None)
        resid_r = y_fx - Zr @ br
        SSE_r = float(resid_r @ resid_r)
        df1 = p   # number of restrictions
        df2 = Zr.shape[0] - Zr.shape[1]
        F = ((SSE_r - SSE_u)/df1) / (SSE_u/df2) if df2>0 else np.nan
        pval_fx = float(st.f.sf(F, df1, df2)) if st else np.nan
        gr["dy_to_fx"] = {"F": F, "df1": df1, "df2": df2, "pval": pval_fx}

        # fx -> dy
        y_dy = Y[:,0]
        # unrestricted dy eq: const + all lags
        bu2, *_ = np.linalg.lstsq(Zu, y_dy, rcond=None)
        resid_u2 = y_dy - Zu @ bu2
        SSE_u2 = float(resid_u2 @ resid_u2)
        # restricted: drop fx lags
        Zr2 = np.column_stack([Z[:,0], Z[:, 1:1+p]])  # const + dy lags only
        br2, *_ = np.linalg.lstsq(Zr2, y_dy, rcond=None)
        resid_r2 = y_dy - Zr2 @ br2
        SSE_r2 = float(resid_r2 @ resid_r2)
        F2 = ((SSE_r2 - SSE_u2)/df1) / (SSE_u2/(Zr2.shape[0]-Zr2.shape[1])) if (Zr2.shape[0]-Zr2.shape[1])>0 else np.nan
        pval_dy = float(st.f.sf(F2, df1, Zr2.shape[0]-Zr2.shape[1])) if st else np.nan
        gr["fx_to_dy"] = {"F": F2, "df1": df1, "df2": (Zr2.shape[0]-Zr2.shape[1]), "pval": pval_dy}

        var_info = {
            "order": model["order"],
            "aic_table": model["aic_table"].to_dict(orient="records"),
            "granger": gr
        }
        (outdir / "var_summary.json").write_text(json.dumps(var_info, indent=2))

    # Events
    ev_out = pd.DataFrame()
    if not E.empty:
        ev_out = event_study(panel, E, fx_col=fx_main, dy_col=dy_col, k=args.event_k)
        if not ev_out.empty:
            ev_out.to_csv(outdir / "event_study.csv", index=False)

    # Scenarios
    scen_out = pd.DataFrame()
    if args.scenarios:
        S = pd.read_csv(args.scenarios)
        S = S.rename(columns={(ncol(S,"scenario") or "scenario"):"scenario",
                              (ncol(S,"name") or "name"):"name",
                              (ncol(S,"key") or "key"):"key",
                              (ncol(S,"value") or "value"):"value"})
        # Latest β
        latest_beta = float(roll["beta_per10bp"].dropna().iloc[-1]) if not roll.empty and roll["beta_per10bp"].notna().any() else np.nan
        latest_irf = None
        if var_model is not None:
            # compute IRF again to ensure columns labeled as expected
            irf_latest = var_irf(var_model, horizon=args.irf_h, shock_var=0, ordering=args.order)
            latest_irf = irf_latest
        scen_out = scenarios_map(S, latest_beta, latest_irf, args.tenor, args.irf_h, method="beta")
        if not scen_out.empty:
            scen_out.to_csv(outdir / "scenarios_out.csv", index=False)

    # Summary
    last_beta = float(roll["beta_per10bp"].dropna().iloc[-1]) if not roll.empty and roll["beta_per10bp"].notna().any() else np.nan
    sign = "neg" if last_beta < 0 else ("pos" if last_beta>0 else "nan")
    irf_snip = {}
    if var_model is not None:
        irf0 = var_irf(var_model, horizon=min(5, args.irf_h), shock_var=0, ordering=args.order)
        irf_snip = {int(r["h"]): float(r["resp_fx_ret_pct"]) for _, r in irf0.iterrows()}
    summary = {
        "fx_series": fx_main,
        "yield_change_series": dy_col,
        "rolling_window": args.window,
        "beta_per10bp_latest": last_beta,
        "beta_sign": sign,
        "var_order": var_info.get("order", None),
        "irf_dy_to_fx_first_steps": irf_snip
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        gilts=args.gilts, fx=args.fx, events=(args.events or None), scenarios=(args.scenarios or None),
        tenor=args.tenor, freq=args.freq, window=args.window, pmax=args.pmax, irf_h=args.irf_h,
        event_k=args.event_k, order=args.order, start=args.start, end=args.end, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== UK Gilts vs GBP ==")
    print(f"Series: {dy_col} → {fx_main} | Window {args.window} | β(10bp) latest = {last_beta if last_beta==last_beta else float('nan'):+.3f}%")
    if var_info:
        print(f"VAR(p={var_info['order']}) AIC best; Granger dy→fx F={var_info['granger']['dy_to_fx']['F']:.2f}{' p~'+str(round(var_info['granger']['dy_to_fx']['pval'],4)) if var_info['granger']['dy_to_fx']['pval']==var_info['granger']['dy_to_fx']['pval'] else ''}")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
