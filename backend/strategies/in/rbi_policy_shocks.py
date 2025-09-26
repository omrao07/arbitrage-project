#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rbi_policy_shocks.py — RBI MPC surprise extraction, asset reactions, yield betas & macro IRFs
---------------------------------------------------------------------------------------------

What this does
==============
This script extracts **high-frequency RBI policy shocks** (target & path) from OIS/futures
around MPC announcements and measures their impact on:
  • G-Sec yields (by tenor), USD/INR, equities (Nifty/BankNifty),
  • Cross-sectional yield betas (Δy_n = b1*target + b2*path),
  • Monthly macro via local projections (Jordà-style IRFs).

It works with minute-level (preferred) or daily data. If minute data are absent, it falls back
to close-to-close changes on event day vs prior trading day.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--events events.csv    REQUIRED
  Columns (any subset):
    datetime  OR  (date [, time])          # MPC announcement timestamp (local)
    decision_bps, repo_delta_bps           # optional actual move
    label/type                              # optional tag (e.g., "Policy", "Interim")

--ois ois.csv         OPTIONAL  (minute or daily)
  Columns:
    datetime OR (date [, time]),
    1m, 3m, 6m, 1y, 2y, 5y                  # any subset; names like: ois_3m, OIS1Y, m3, y1 accepted

--yields gsec.csv     OPTIONAL  (minute or daily; used for reactions & betas)
  Columns:
    datetime OR (date [, time]),
    gsec_1y, gsec_2y, gsec_5y, gsec_10y     # levels in percent

--fx fx.csv           OPTIONAL  (minute or daily)
  Columns:
    datetime OR (date [, time]),
    usdinr                                  # level

--equity eq.csv       OPTIONAL  (minute or daily)
  Columns:
    datetime OR (date [, time]),
    nifty[, banknifty][, price, close]      # if only one price column, it's interpreted as 'nifty'

--macro macro.csv     OPTIONAL  (monthly; IRFs)
  Columns:
    date, (iip, cpi, wpi, pmi, gdp, gva, credit, ... numeric series)

Key CLI
-------
--w "-15,75"          HF surprise window in minutes relative to announcement (default −15 to +75)
--w_alt "0,30|0,60"   Extra reaction windows (comma sep; use | to separate multiple windows)
--daily_buffer 1      If only daily data, use t to t+daily_buffer change (default 1 day)
--lags 12             LP horizon in months
--winsor 0.0          Winsorize asset returns (0..0.05), default off
--start/--end         Optional date filters (YYYY-MM-DD)
--outdir out_rbi      Output directory

Outputs
-------
- shocks.csv                 Target/Path shocks per event (in **bp**)
- event_reactions.csv        Asset reactions per event & window (bp for yields, log-returns for FX/Equity)
- yield_betas.csv            Per-tenor regression: Δy_n ~ b1*target + b2*path  (coef, se, t)
- macro_irf.csv              Local-projection β_h for each macro variable (h=0..H)
- summary.json               Headline counts & last snapshot
- config.json                Run configuration

Notes/assumptions
-----------------
• Target shock = Δ(3M OIS) in window; Path shock = Δ(1Y OIS) − Δ(3M OIS).
  If 1Y absent, path uses 6M; if 3M absent, uses 1M as target.
• Minute data preferred. With daily data, changes are close(t+buffer) − close(t−1).
• Asset reactions measured over the same HF window (or daily fallback) as shocks.
• Robust (Newey–West) SEs for monthly LP; plain OLS SEs for cross-event yield betas.

DISCLAIMER: Research tooling; validate mappings, timestamps, and microstructure filters before use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================ helpers ============================

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        lc = cand.lower()
        if lc in low: return low[lc]
    for cand in cands:
        t = cand.lower()
        for c in df.columns:
            if t in str(c).lower(): return c
    return None

def parse_window(s: str) -> Tuple[int,int]:
    a, b = s.split(",")
    return int(a.strip()), int(b.strip())

def parse_windows_alt(s: str) -> List[Tuple[int,int]]:
    if not s: return []
    parts = [p for p in s.split("|") if p.strip()]
    return [parse_window(p) for p in parts]

def to_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to a 'dt' column of dtype datetime64[ns] (naive)."""
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.tz_localize(None)
        return df
    dcol = ncol(df, "datetime","timestamp","dt")
    if dcol:
        df = df.rename(columns={dcol:"dt"})
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.tz_localize(None)
        return df
    date = ncol(df, "date") or df.columns[0]
    tcol = ncol(df, "time","hhmm","clock")
    df = df.rename(columns={date:"date"})
    if tcol:
        df = df.rename(columns={tcol:"time"})
        # join date & time
        df["dt"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce").dt.tz_localize(None)
    else:
        df["dt"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    return df

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def winsorize(s: pd.Series, p: float) -> pd.Series:
    if p <= 0: return s
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def last_before(df: pd.DataFrame, t: pd.Timestamp) -> Optional[float]:
    sub = df[df["dt"] <= t]
    if sub.empty: return None
    return float(sub.iloc[-1])

def first_after(df: pd.DataFrame, t: pd.Timestamp) -> Optional[float]:
    sub = df[df["dt"] >= t]
    if sub.empty: return None
    return float(sub.iloc[0])

def delta_window(series: pd.DataFrame, t0: pd.Timestamp, m1: int, m2: int, method: str="last-first") -> Optional[float]:
    """
    For minute data: change between [t0+m1, t0+m2].
    method:
      - "last-first": last value in end window MINUS first value in start window
      - "end-start_nearest": nearest available ticks to exact boundaries
    """
    start = t0 + pd.Timedelta(minutes=m1)
    end   = t0 + pd.Timedelta(minutes=m2)
    seg = series[(series["dt"] >= start) & (series["dt"] <= end)]
    if seg.empty:
        # try nearest ticks
        v0 = last_before(series, start)
        v1 = last_before(series, end)
        if v0 is None or v1 is None: return None
        return v1 - v0
    v0 = first_after(series, start)
    v1 = float(seg.iloc[-1, seg.columns.get_loc("val")])
    if v0 is None: v0 = float(seg.iloc[0, seg.columns.get_loc("val")])
    return v1 - v0

# OLS & Newey–West
def ols(X: np.ndarray, y: np.ndarray):
    XTX = X.T @ X
    XTy = X.T @ y
    beta = np.linalg.pinv(XTX) @ XTy
    resid = y - X @ beta
    return beta, resid, np.linalg.pinv(XTX)

def nw_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
    n = X.shape[0]
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1)+1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))


# ============================ loaders ============================

def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = to_dt(df)
    # Normalize label/type
    lab = ncol(df, "label","event","type","kind")
    if lab: df = df.rename(columns={lab:"label"})
    if "label" not in df.columns: df["label"] = "MPC"
    # repo change
    mv = ncol(df, "decision_bps","repo_delta_bps","move_bps")
    if mv: df = df.rename(columns={mv:"decision_bps"})
    if "decision_bps" in df.columns: df["decision_bps"] = safe_num(df["decision_bps"])
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return df[["dt","label"] + ([ "decision_bps"] if "decision_bps" in df.columns else [])]

def load_ois(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = to_dt(df)
    # map columns
    def map_one(df, names, out):
        c = None
        for n in names:
            c = ncol(df, n) or c
        if c: df.rename(columns={c: out}, inplace=True)
    map_one(df, ["ois_1m","m1","1m"], "ois_1m")
    map_one(df, ["ois_3m","m3","3m"], "ois_3m")
    map_one(df, ["ois_6m","m6","6m"], "ois_6m")
    map_one(df, ["ois_1y","y1","1y"], "ois_1y")
    map_one(df, ["ois_2y","y2","2y"], "ois_2y")
    map_one(df, ["ois_5y","y5","5y"], "ois_5y")
    for c in list(df.columns):
        if c not in {"dt","date","time"}:
            df[c] = safe_num(df[c])
    keep = ["dt"] + [c for c in ["ois_1m","ois_3m","ois_6m","ois_1y","ois_2y","ois_5y"] if c in df.columns]
    return df[keep].dropna(subset=["dt"]).sort_values("dt")

def load_yields(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = to_dt(df)
    maps = [
        (["gsec_1y","y1","1y"], "gsec_1y"),
        (["gsec_2y","y2","2y"], "gsec_2y"),
        (["gsec_5y","y5","5y"], "gsec_5y"),
        (["gsec_10y","y10","10y"], "gsec_10y"),
    ]
    for src, out in maps:
        c = None
        for n in src:
            c = ncol(df, n) or c
        if c: df.rename(columns={c:out}, inplace=True)
    for c in df.columns:
        if c not in {"dt","date","time"}:
            df[c] = safe_num(df[c])
    keep = ["dt"] + [c for c in ["gsec_1y","gsec_2y","gsec_5y","gsec_10y"] if c in df.columns]
    return df[keep].dropna(subset=["dt"]).sort_values("dt")

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path); df = to_dt(df)
    c = ncol(df, "usdinr","fx","usd_inr")
    if not c: raise ValueError("fx.csv must include a USDINR column.")
    df = df.rename(columns={c:"usdinr"})
    df["usdinr"] = safe_num(df["usdinr"])
    return df[["dt","usdinr"]].dropna(subset=["dt"]).sort_values("dt")

def load_equity(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path); df = to_dt(df)
    n  = ncol(df, "nifty","close","price","px")
    bn = ncol(df, "banknifty","nifty_bank","bank_nifty")
    if not n: raise ValueError("equity.csv needs a 'nifty' (or close/price) column.")
    df.rename(columns={n:"nifty"}, inplace=True)
    if bn: df.rename(columns={bn:"banknifty"}, inplace=True)
    for c in df.columns:
        if c not in {"dt","date","time"}:
            df[c] = safe_num(df[c])
    keep = ["dt","nifty"] + (["banknifty"] if "banknifty" in df.columns else [])
    return df[keep].dropna(subset=["dt"]).sort_values("dt")

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date") or df.columns[0]
    df.rename(columns={d:"date"}, inplace=True)
    df["date"] = to_date(df["date"])
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])
    # keep only numeric columns with ≥ 24 obs
    keep = ["date"] + [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    return df[keep].dropna(subset=["date"]).sort_values("date")


# ============================ shock extraction ============================

def compute_shocks(events: pd.DataFrame, ois: pd.DataFrame,
                   w: Tuple[int,int], daily_buffer: int) -> pd.DataFrame:
    """
    Returns per-event shocks in **basis points**:
      target = Δ(3M OIS)
      path   = Δ(1Y OIS) − Δ(3M OIS)    (fallbacks applied)
    """
    rows = []
    have_minute = (not ois.empty) and (ois["dt"].dt.normalize() != ois["dt"]).any()
    for _, ev in events.iterrows():
        t0 = ev["dt"]
        rec = {"dt": t0, "label": ev.get("label","MPC")}
        # default
        tgt = None; pth = None
        if not ois.empty:
            # choose tenors
            t_col = "ois_3m" if "ois_3m" in ois.columns else ("ois_1m" if "ois_1m" in ois.columns else None)
            y_col = "ois_1y" if "ois_1y" in ois.columns else ("ois_6m" if "ois_6m" in ois.columns else None)
            if t_col:
                if have_minute:
                    s = ois[["dt", t_col]].rename(columns={t_col:"val"})
                    d_t = delta_window(s, t0, w[0], w[1])
                else:
                    d_t = daily_delta(ois[["dt", t_col]].rename(columns={t_col:"val"}), t0, daily_buffer)
                tgt = d_t
            if y_col:
                if have_minute:
                    s = ois[["dt", y_col]].rename(columns={y_col:"val"})
                    d_y = delta_window(s, t0, w[0], w[1])
                else:
                    d_y = daily_delta(ois[["dt", y_col]].rename(columns={y_col:"val"}), t0, daily_buffer)
                if d_y is not None and tgt is not None:
                    pth = d_y - tgt
                else:
                    pth = d_y
        # convert to bp (OIS levels in %)
        if tgt is not None: rec["shock_target_bp"] = float(tgt) * 100.0
        if pth is not None: rec["shock_path_bp"]   = float(pth) * 100.0
        if "decision_bps" in ev and pd.notna(ev["decision_bps"]):
            rec["decision_bps"] = float(ev["decision_bps"])
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("dt")
    return out

def daily_delta(series: pd.DataFrame, t0: pd.Timestamp, buffer_days: int) -> Optional[float]:
    """Close(t+buffer) − close(t−1) on daily time series."""
    s = series.copy().sort_values("dt")
    s["date"] = s["dt"].dt.normalize()
    g = s.groupby("date")["val"].last().reset_index()
    # find event date (nearest previous trading day)
    d0 = g[g["date"] <= t0.normalize()]
    if d0.empty: return None
    d0 = d0.iloc[-1]["date"]
    prev = g[g["date"] < d0]
    if prev.empty: return None
    v_prev = float(prev.iloc[-1]["val"])
    # forward buffer
    fwd = g[g["date"] >= d0].head(buffer_days+1)
    if fwd.empty: return None
    v_end = float(fwd.iloc[-1]["val"])
    return v_end - v_prev


# ============================ asset reactions ============================

def event_reactions(events: pd.DataFrame,
                    yields: pd.DataFrame, fx: pd.DataFrame, eq: pd.DataFrame,
                    windows: List[Tuple[int,int]], daily_buffer: int, winsor_p: float) -> pd.DataFrame:
    """
    Returns per-event reactions for each window (bp for yields; log-returns for FX & Equity).
    """
    have_minute_y = (not yields.empty) and (yields["dt"].dt.normalize() != yields["dt"]).any()
    have_minute_fx = (not fx.empty) and (fx["dt"].dt.normalize() != fx["dt"]).any()
    have_minute_eq = (not eq.empty) and (eq["dt"].dt.normalize() != eq["dt"]).any()

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["dt"]
        base = {"dt": t0, "label": ev.get("label","MPC")}
        for (a,b) in windows:
            tag = f"[{a},{b}]m"
            rec = dict(base)
            # yields → Δ in bp
            if not yields.empty:
                for col in [c for c in ["gsec_1y","gsec_2y","gsec_5y","gsec_10y"] if c in yields.columns]:
                    s = yields[["dt", col]].rename(columns={col:"val"})
                    if have_minute_y:
                        d = delta_window(s, t0, a, b)
                    else:
                        d = daily_delta(s, t0, daily_buffer)
                    rec[f"dy_{col}_bp@{tag}"] = float(d)*100.0 if d is not None else np.nan
            # fx → log return (usdinr ↑ = INR depreciation)
            if not fx.empty:
                s = fx[["dt","usdinr"]].rename(columns={"usdinr":"val"})
                if have_minute_fx:
                    v = window_vals(s, t0, a, b)
                    if v is None:
                        r = np.nan
                    else:
                        v0, v1 = v
                        r = np.log(v1) - np.log(v0) if (v0 and v1) else np.nan
                else:
                    # daily
                    r = daily_logret(s, t0, daily_buffer)
                rec[f"fx_logret@{tag}"] = r
            # equity → log return
            if not eq.empty:
                for col in [c for c in ["nifty","banknifty"] if c in eq.columns]:
                    s = eq[["dt", col]].rename(columns={col:"val"})
                    if have_minute_eq:
                        v = window_vals(s, t0, a, b)
                        if v is None:
                            r = np.nan
                        else:
                            v0, v1 = v
                            r = np.log(v1) - np.log(v0) if (v0 and v1) else np.nan
                    else:
                        r = daily_logret(s, t0, daily_buffer)
                    rec[f"eq_{col}_logret@{tag}"] = r
            rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["dt"])
    # winsorize returns if requested
    if winsor_p > 0 and not out.empty:
        for c in out.columns:
            if "logret" in c:
                out[c] = winsorize(out[c], winsor_p)
    return out

def window_vals(series: pd.DataFrame, t0: pd.Timestamp, a: int, b: int) -> Optional[Tuple[float,float]]:
    start = t0 + pd.Timedelta(minutes=a)
    end   = t0 + pd.Timedelta(minutes=b)
    pre  = last_before(series, start)
    post = last_before(series, end)
    if pre is None or post is None: return None
    return float(pre), float(post)

def daily_logret(series: pd.DataFrame, t0: pd.Timestamp, buffer_days: int) -> Optional[float]:
    s = series.copy().sort_values("dt")
    s["date"] = s["dt"].dt.normalize()
    g = s.groupby("date")["val"].last().reset_index()
    d0 = g[g["date"] <= t0.normalize()]
    if d0.empty: return None
    d0 = d0.iloc[-1]["date"]
    prev = g[g["date"] < d0]
    if prev.empty: return None
    v_prev = float(prev.iloc[-1]["val"])
    fwd = g[g["date"] >= d0].head(buffer_days+1)
    if fwd.empty: return None
    v_end = float(fwd.iloc[-1]["val"])
    return np.log(v_end) - np.log(v_prev)


# ============================ yield betas ============================

def yield_betas(shocks: pd.DataFrame, reacts: pd.DataFrame) -> pd.DataFrame:
    """
    For each tenor n, run: Δy_n(bp) = b0 + b1*target_bp + b2*path_bp + eps  (plain OLS)
    Returns coef, se, t for each tenor.
    """
    if shocks.empty or reacts.empty: return pd.DataFrame()
    # choose default window (first window in reacts)
    dy_cols = [c for c in reacts.columns if c.startswith("dy_") and "@[" in c]
    if not dy_cols: return pd.DataFrame()
    win = sorted({c.split("@")[1] for c in dy_cols})[0]
    tenors = sorted({c.split("_")[1].replace("gsec","gsec_") for c in dy_cols if win in c})
    out = []
    M = shocks.merge(reacts[["dt"] + [c for c in reacts.columns if c.endswith(win)]], on="dt", how="inner")
    X = M[["shock_target_bp","shock_path_bp"]].copy()
    if X.isna().any().any():  # drop rows with missing shocks
        M = M.dropna(subset=["shock_target_bp","shock_path_bp"])
    X = np.column_stack([np.ones((len(M),1)), M[["shock_target_bp","shock_path_bp"]].values])
    for tn in tenors:
        c = f"dy_{tn}_bp@{win}"
        if c not in M.columns: continue
        y = M[c].values.reshape(-1,1)
        mask = np.isfinite(y).ravel()
        XX = X[mask,:]; yy = y[mask,:]
        if len(yy) < 10: continue
        beta, resid, XtX_inv = ols(XX, yy)
        # simple homoskedastic se
        s2 = float((resid.T @ resid) / (len(yy) - XX.shape[1]))
        se = np.sqrt(np.diag(s2 * XtX_inv))
        for i, nm in enumerate(["const","b_target","b_path"]):
            out.append({"tenor": tn, "var": nm, "coef": float(beta[i,0]), "se": float(se[i]), 
                        "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "n": int(len(yy))})
    return pd.DataFrame(out)


# ============================ monthly local projections ============================

def monthly_irf(shocks: pd.DataFrame, macro: pd.DataFrame, H: int=12) -> pd.DataFrame:
    """
    Build monthly shock series (sum of shocks within month), then run LP for each macro variable:
      y_{t+h} - y_{t-1} = α_h + β_h * shock_target_t + γ_h * shock_path_t + ε_{t,h}
    Implementation uses log transform for strictly positive series; otherwise first diff.
    HAC (NW) SEs with lag L = 6.
    """
    if shocks.empty or macro.empty: return pd.DataFrame()
    S = shocks.copy()
    S["date"] = S["dt"].dt.to_period("M").dt.to_timestamp("M")
    S = S.groupby("date", as_index=False)[["shock_target_bp","shock_path_bp"]].sum()

    df = macro.merge(S, on="date", how="left").sort_values("date")
    df[["shock_target_bp","shock_path_bp"]] = df[["shock_target_bp","shock_path_bp"]].fillna(0.0)
    df = df.dropna(subset=["date"])
    vars_ = [c for c in df.columns if c not in {"date","shock_target_bp","shock_path_bp"} and pd.api.types.is_numeric_dtype(df[c])]
    out = []

    for var in vars_:
        y = df[var].astype(float)
        # transform
        if (y > 0).all():
            y_t = np.log(y.replace(0, np.nan))
        else:
            y_t = y
        # build dependent for each horizon
        for h in range(0, H+1):
            # y_{t+h} - y_{t-1}
            y_lead = y_t.shift(-h)
            y_lag1 = y_t.shift(1)
            dep = (y_lead - y_lag1)
            XY = pd.DataFrame({
                "dep": dep,
                "const": 1.0,
                "shock_target_bp": df["shock_target_bp"],
                "shock_path_bp": df["shock_path_bp"]
            }).dropna()
            if XY.shape[0] < 24:
                continue
            Yv = XY["dep"].values.reshape(-1,1)
            Xv = XY[["const","shock_target_bp","shock_path_bp"]].values
            beta, resid, XtX_inv = ols(Xv, Yv)
            se = nw_se(Xv, resid, XtX_inv, L=6)
            for i, nm in enumerate(["const","beta_target","beta_path"]):
                out.append({"series": var, "h": h, "var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                            "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "n": int(XY.shape[0])})
    return pd.DataFrame(out)


# ============================ CLI / main ============================

@dataclass
class Config:
    events: str
    ois: Optional[str]
    yields: Optional[str]
    fx: Optional[str]
    equity: Optional[str]
    macro: Optional[str]
    w: str
    w_alt: str
    daily_buffer: int
    lags: int
    winsor: float
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RBI policy shocks: extraction, reactions, yield betas & macro IRFs")
    ap.add_argument("--events", required=True)
    ap.add_argument("--ois", default="")
    ap.add_argument("--yields", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--equity", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--w", default="-15,75")
    ap.add_argument("--w_alt", default="0,30|0,60")
    ap.add_argument("--daily_buffer", type=int, default=1)
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--winsor", type=float, default=0.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_rbi")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    EVENTS = load_events(args.events)
    OIS    = load_ois(args.ois) if args.ois else pd.DataFrame()
    YLD    = load_yields(args.yields) if args.yields else pd.DataFrame()
    FX     = load_fx(args.fx) if args.fx else pd.DataFrame()
    EQ     = load_equity(args.equity) if args.equity else pd.DataFrame()
    MACRO  = load_macro(args.macro) if args.macro else pd.DataFrame()

    # Date filters
    if args.start:
        t = pd.to_datetime(args.start)
        EVENTS = EVENTS[EVENTS["dt"] >= t]
    if args.end:
        t = pd.to_datetime(args.end)
        EVENTS = EVENTS[EVENTS["dt"] <= t]

    if EVENTS.empty:
        raise ValueError("No events after filters. Provide --events with valid timestamps.")

    # Shocks
    w = parse_window(args.w)
    SHOCKS = compute_shocks(EVENTS, OIS, w=w, daily_buffer=int(args.daily_buffer))
    if not SHOCKS.empty: SHOCKS.to_csv(outdir / "shocks.csv", index=False)

    # Reactions (use default + alt windows)
    wins = [w] + parse_windows_alt(args.w_alt)
    REACTS = event_reactions(EVENTS, YLD, FX, EQ, windows=wins, daily_buffer=int(args.daily_buffer), winsor_p=float(args.winsor))
    if not REACTS.empty: REACTS.to_csv(outdir / "event_reactions.csv", index=False)

    # Yield betas
    BETAS = yield_betas(SHOCKS, REACTS) if (not SHOCKS.empty and not REACTS.empty) else pd.DataFrame()
    if not BETAS.empty: BETAS.to_csv(outdir / "yield_betas.csv", index=False)

    # Monthly IRFs
    IRF = monthly_irf(SHOCKS, MACRO, H=int(args.lags)) if (not SHOCKS.empty and not MACRO.empty) else pd.DataFrame()
    if not IRF.empty: IRF.to_csv(outdir / "macro_irf.csv", index=False)

    # Summary
    summary = {
        "n_events": int(len(EVENTS)),
        "has_minute_ois": (not OIS.empty) and (OIS["dt"].dt.normalize() != OIS["dt"]).any(),
        "has_minute_yields": (not YLD.empty) and (YLD["dt"].dt.normalize() != YLD["dt"]).any(),
        "windows_used": [args.w] + ([p for p in args.w_alt.split("|") if p] if args.w_alt else []),
        "files": {
            "shocks": ("shocks.csv" if not SHOCKS.empty else None),
            "reactions": ("event_reactions.csv" if not REACTS.empty else None),
            "yield_betas": ("yield_betas.csv" if not BETAS.empty else None),
            "macro_irf": ("macro_irf.csv" if not IRF.empty else None)
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        events=args.events, ois=(args.ois or None), yields=(args.yields or None), fx=(args.fx or None),
        equity=(args.equity or None), macro=(args.macro or None),
        w=args.w, w_alt=args.w_alt, daily_buffer=int(args.daily_buffer), lags=int(args.lags),
        winsor=float(args.winsor), start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== RBI Policy Shocks ==")
    print(f"Events: {summary['n_events']} | Minute OIS: {summary['has_minute_ois']} | Minute Yields: {summary['has_minute_yields']}")
    if not SHOCKS.empty:
        s_last = SHOCKS.tail(1).iloc[0]
        print(f"Last shock → target {s_last.get('shock_target_bp', np.nan):+.1f} bp | path {s_last.get('shock_path_bp', np.nan):+.1f} bp")
    if not BETAS.empty:
        for tn in sorted(BETAS['tenor'].unique()):
            b1 = BETAS[(BETAS.tenor==tn) & (BETAS.var=="b_target")].iloc[0]
            b2 = BETAS[(BETAS.tenor==tn) & (BETAS.var=="b_path")].iloc[0]
            print(f"{tn}: Δy = {b1.coef:+.2f}*target + {b2.coef:+.2f}*path  (bp/bp)")
    print("Artifacts written to:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
