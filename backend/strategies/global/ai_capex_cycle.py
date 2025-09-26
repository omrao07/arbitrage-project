#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_capex_cycle.py — Hyperscaler/AI infrastructure capex cycle tracker

What this does
--------------
Takes company-level capex (hyperscalers, cloud/colo, foundry, memory, equipment) and a panel of
"supply chain & demand" drivers (GPU/accelerator shipments, optics, memory prices, wafer starts,
bookings/backlog/BTB, lead times, data center pipeline), then:

1) Cleans & aligns to QUARTERS
2) Builds aggregate & cohort-level CAPEX (YoY, QoQ, levels)
3) Computes diffusion indices (% of firms accelerating QoQ/YoY)
4) Builds driver transformations (Δlog %, YoY) and a wide driver matrix
5) Searches lead/lag correlations to find best leading indicators
6) Nowcasts next H quarters of capex YoY with a ridge-regularized ARX model
7) Produces supplier early-warning metrics (BTB, backlog cover, lead time stress)
8) Applies scenario shocks to drivers (e.g., "GPU_SHIP.pct=-15@2025Q1") and recomputes paths
9) Writes tidy CSVs + a compact JSON summary

Core outputs
------------
- capex_panel.csv            Per company & cohort: level, QoQ%, YoY%, currency-normalized if provided
- aggregates.csv             Totals by cohort & global (level, YoY, QoQ)
- diffusion.csv              Diffusion indices (#firms up QoQ/YoY / total)
- drivers_wide.csv           Wide quarterly driver matrix (Δlog % or YoY where applicable)
- lead_lag_matrix.csv        Corr(driver, capex YoY) across leads/lags (−8..+8q), with argmax
- nowcast_baseline.csv       Baseline in-sample fit + forward YoY path per cohort
- scenarios_out.csv          Scenario paths (YoY) per cohort and Δ vs baseline
- supplier_early_warning.csv BTB, backlog cover (months/qtrs), lead-time stress by category
- cycle_phase.csv            Simple phase classifier (expansion/slowdown/contraction/repair)
- summary.json               Latest KPIs (YoY, diffusion, best leads, nowcast)
- config.json                Reproducibility dump

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--financials financials.csv    REQUIRED (quarterly)
  Columns: date, company, capex_[usd|chf|eur]_m  [, cohort] [, currency] [, revenue_* optional]
  Notes: If multiple currency columns exist, usd takes precedence. If none labeled, use 'capex' as level units.

--suppliers suppliers.csv      OPTIONAL (monthly or quarterly)
  Columns: date, category, series, value
  Example category/series:
    GPU   : SHIP_UNITS, BOOKINGS_USD_M, BACKLOG_USD_M
    OPTICS: SHIP_UNITS, ASP_IDX
    MEMORY: DRAM_SPOT_IDX, NAND_SPOT_IDX
    FOUNDRY: WAFER_STARTS_K, WAFER_PRICE_IDX
    EQUIP : BOOKINGS_USD_M, B2B_RATIO
    POWER : UPS_SHIP_MVA, SWITCHGEAR_LEADTIME_WKS
  (Names are free-form; transformation is automatic.)

--prices prices.csv            OPTIONAL (monthly or quarterly)
  Columns: date, series, value
  (Used the same way as suppliers; convenience if you keep prices separately.)

--projects projects.csv        OPTIONAL (data center pipeline)
  Columns: date, region, status, mw (or racks)
  We convert to quarterly pipeline metrics (under_construction MW etc.)

--constraints constraints.csv  OPTIONAL
  Columns: date, component, lead_time_wks [, util_pct] [, capacity_units]
  We compute lead-time z-scores and a composite stress index per component.

--scenarios scenarios.csv      OPTIONAL
  Columns: scenario, name, key, value
  Keys:
    <SERIES>.pct = -15@2025Q1     (additive shock to Δlog % from quarter onwards)
    horizon.quarters = 8
    ridge.lambda = 5.0
    arx.y_lag = 1

CLI (examples)
--------------
--start 2017Q1 --end 2025Q4 --lags 8 --ridge_lambda 3.0 --horizon 8 --cohort_key "cohort" --outdir out_ai_capex

Method notes
------------
- Quarterly frequency (calendar quarter end). Monthly inputs are aggregated to Q by mean (levels) or sum for flows
  if series name contains 'SHIP', 'BOOK', 'REVENUE' (configurable via heuristics).
- Driver transforms: Δlog*100 for levels; keep *_YOY series as provided; keep *_PCT already in %.
- ARX ridge for YoY nowcast per cohort: y_t = α + φ*y_{t-1} + Σ_j Σ_{ℓ=0..L} β_{j,ℓ} x_{j,t-ℓ} + ε_t
- Lead/lag scan uses Pearson corr across −8..+8 quarters; report best lag (positive = driver leads capex).
- Cycle phase (simple): z-score of YoY & 4-q slope → {expansion, slowdown, contraction, repair}.
- All outputs in tidy CSVs; JSON summary highlights top-3 leading drivers by absolute corr at positive leads.

DISCLAIMER: Research tool. Not investment advice.
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

def to_quarter(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("Q").dt.to_timestamp(how="end")

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def yoy(series: pd.Series) -> pd.Series:
    return series.pct_change(4) * 100.0

def qoq(series: pd.Series) -> pd.Series:
    return series.pct_change(1) * 100.0

def dlogpct(series: pd.Series) -> pd.Series:
    return np.log(series.astype(float).replace(0, np.nan)).diff() * 100.0

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=0)
    return (s - m) / (sd if sd and sd==sd else 1.0)

def is_flow_name(name: str) -> bool:
    n = name.upper()
    return any(k in n for k in ["SHIP", "BOOK", "REVENUE", "SALES", "BKGS", "BKG", "FLOW"])

def quarterly_agg(df: pd.DataFrame, series_col: str="series", value_col: str="value") -> pd.DataFrame:
    """Aggregate monthly or daily to quarters. For flows-like names, SUM; else MEAN."""
    df = df.copy()
    df["date"] = to_quarter(df["date"])
    if series_col in df.columns:
        rows = []
        for s, g in df.groupby(series_col):
            if is_flow_name(str(s)):
                gg = g.groupby("date", as_index=False)[value_col].sum()
            else:
                gg = g.groupby("date", as_index=False)[value_col].mean()
            gg[series_col] = s
            rows.append(gg)
        return pd.concat(rows, ignore_index=True)
    else:
        # no series dimension, just aggregate
        return df.groupby("date", as_index=False)[value_col].mean()

def listlike(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray, pd.Index))

# ----------------------------- loaders -----------------------------

def load_financials(path: str, cohort_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = ncol(df,"date") or df.columns[0]
    comp_c = ncol(df,"company") or "company"
    df = df.rename(columns={date_c:"date", comp_c:"company"})
    df["date"] = to_quarter(df["date"])
    df["company"] = df["company"].astype(str)
    # Pick capex column
    cap_cols = [c for c in df.columns if c.lower().startswith("capex")]
    pref = ["capex_usd_m", "capexusd", "capex_usd", "capex_musd", "capex_chf_m", "capex_eur_m", "capex"]
    cap = None
    for p in pref:
        c = ncol(df, p)
        if c: cap = c; break
    if not cap:
        # last numeric column fallback
        numcols = [c for c in df.columns if c not in ["date","company"] and pd.api.types.is_numeric_dtype(df[c])]
        if not numcols: raise ValueError("No CAPEX column found in financials.")
        cap = numcols[-1]
    df = df.rename(columns={cap:"capex"})
    # Cohort (if missing derive crude heuristic)
    coh_c = ncol(df, cohort_key) if cohort_key else None
    if coh_c:
        df["cohort"] = df[coh_c].astype(str)
    else:
        # heuristics by company name
        cc = df["company"].str.upper()
        df["cohort"] = np.where(cc.str.contains("TSMC|SAMSUNG|INTEL|GLOBALFOUNDRIES|UMC"), "FOUND/MEM",
                         np.where(cc.str.contains("ASML|AMAT|LAM|TEL|KLA"), "EQUIP",
                         np.where(cc.str.contains("EQUINIX|DIGITAL REALTY|NTT|GDS|NEXTDC|STACK|QTS|OVH|COLT"), "COLO",
                         np.where(cc.str.contains("MICROSOFT|AMAZON|GOOGLE|ALPHABET|META|APPLE|ORACLE|BABA|TENCENT|BAIDU"), "HYPER", "OTHER"))))
    df["capex"] = num(df["capex"])
    return df[["date","company","cohort","capex"]].sort_values(["company","date"])

def load_suppliers(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"category") or "category"):"category",
           (ncol(df,"series") or "series"):"series",
           (ncol(df,"value") or "value"):"value"}
    df = df.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["category"] = df["category"].astype(str).str.upper().str.strip()
    df["series"] = df["series"].astype(str).str.upper().str.strip()
    df["value"] = num(df["value"])
    # aggregate to quarters
    q = quarterly_agg(df, series_col="series", value_col="value")
    return q.sort_values(["series","date"])

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"series") or "series"):"series",
           (ncol(df,"value") or "value"):"value"}
    df = df.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["series"] = df["series"].astype(str).str.upper().str.strip()
    df["value"] = num(df["value"])
    q = quarterly_agg(df, series_col="series", value_col="value")
    return q.sort_values(["series","date"])

def load_projects(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    date_c = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={date_c:"date"})
    df["date"] = to_quarter(df["date"])
    if ncol(df,"mw"):
        df["mw"] = num(df[ncol(df,"mw")])
    elif ncol(df,"racks"):
        df["mw"] = num(df[ncol(df,"racks")]) * 0.01  # crude 100 racks ≈ 1 MW; user can adjust
    else:
        df["mw"] = np.nan
    df["status"] = df[ncol(df,"status") or "status"].astype(str).str.lower()
    df["region"] = df[ncol(df,"region") or "region"].astype(str)
    # pipeline by status
    pipe = (df.groupby(["date","status"], as_index=False)["mw"].sum().rename(columns={"mw":"pipeline_mw"}))
    return pipe

def load_constraints(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"component") or "component"):"component",
           (ncol(df,"lead_time_wks") or ncol(df,"leadtime_wks") or "lead_time_wks"):"lead_time_wks"}
    df = df.rename(columns=ren)
    df["date"] = to_quarter(df["date"])
    df["component"] = df["component"].astype(str).str.upper().str.strip()
    df["lead_time_wks"] = num(df["lead_time_wks"])
    if ncol(df,"util_pct"): df["util_pct"] = num(df[ncol(df,"util_pct")])
    return df

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"):"scenario",
           (ncol(df,"name") or "name"):"name",
           (ncol(df,"key") or "key"):"key",
           (ncol(df,"value") or "value"):"value"}
    return df.rename(columns=ren)


# ----------------------------- core transforms -----------------------------

def capex_panel(fin_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = fin_df.copy().sort_values(["company","date"])
    # Company-level changes
    out = []
    for comp, g in df.groupby("company"):
        gg = g.set_index("date")["capex"].sort_index()
        panel = pd.DataFrame({
            "date": gg.index,
            "company": comp,
            "capex": gg.values,
            "capex_qoq_pct": qoq(gg).values,
            "capex_yoy_pct": yoy(gg).values
        })
        # add cohort
        cohort = df[df["company"]==comp]["cohort"].iloc[0]
        panel["cohort"] = cohort
        out.append(panel)
    comp_panel = pd.concat(out, ignore_index=True)

    # Aggregates by cohort & total (sum of levels; YoY/QoQ recomputed on aggregate)
    agg_rows = []
    for coh, g in comp_panel.groupby("cohort"):
        s = g.groupby("date")["capex"].sum()
        agg_rows.append(pd.DataFrame({"date": s.index, "cohort": coh,
                                      "capex": s.values,
                                      "capex_qoq_pct": qoq(s).values,
                                      "capex_yoy_pct": yoy(s).values}))
    # Global
    s = comp_panel.groupby("date")["capex"].sum()
    agg_rows.append(pd.DataFrame({"date": s.index, "cohort": "GLOBAL",
                                  "capex": s.values,
                                  "capex_qoq_pct": qoq(s).values,
                                  "capex_yoy_pct": yoy(s).values}))
    agg = pd.concat(agg_rows, ignore_index=True).sort_values(["cohort","date"])
    return comp_panel, agg

def diffusion_indices(comp_panel: pd.DataFrame) -> pd.DataFrame:
    # by date & cohort (including GLOBAL)
    rows = []
    for (d, coh), g in comp_panel.groupby(["date","cohort"]):
        n = g.shape[0]
        up_qoq = (g["capex_qoq_pct"]>0).sum()
        up_yoy = (g["capex_yoy_pct"]>0).sum()
        rows.append({"date": d, "cohort": coh, "n_firms": n,
                     "diff_qoq": up_qoq / n if n else np.nan,
                     "diff_yoy": up_yoy / n if n else np.nan})
    # Add GLOBAL (all firms)
    for d, g in comp_panel.groupby("date"):
        n = g.shape[0]
        rows.append({"date": d, "cohort": "GLOBAL", "n_firms": n,
                     "diff_qoq": (g["capex_qoq_pct"]>0).mean() if n else np.nan,
                     "diff_yoy": (g["capex_yoy_pct"]>0).mean() if n else np.nan})
    return pd.DataFrame(rows).sort_values(["cohort","date"])

def make_driver_wide(suppliers_q: pd.DataFrame, prices_q: pd.DataFrame, projects_q: pd.DataFrame, constraints_q: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for src in [suppliers_q, prices_q]:
        if not src.empty:
            pivot = src.pivot_table(index="date", columns="series", values="value", aggfunc="last")
            parts.append(pivot)
    # Projects: convert statuses to series
    if not projects_q.empty:
        piv = projects_q.pivot_table(index="date", columns="status", values="pipeline_mw", aggfunc="sum")
        piv.columns = [f"PIPE_{c.upper()}" for c in piv.columns]
        parts.append(piv)
    # Constraints: lead_time (we take z-score to make unitless)
    if not constraints_q.empty:
        C = constraints_q.pivot_table(index="date", columns="component", values="lead_time_wks", aggfunc="mean")
        C = C.apply(zscore)  # per component z
        C.columns = [f"LTZ_{c.upper()}" for c in C.columns]
        parts.append(C)
    if not parts:
        return pd.DataFrame()
    wide = pd.concat(parts, axis=1).sort_index()
    # Transform to Δlog% (or keep *_YOY and *_PCT as-is)
    out = {}
    for c in wide.columns:
        cu = str(c).upper()
        if cu.endswith("_YOY") or cu.endswith("PCT") or cu.endswith("%"):
            out[c] = wide[c]
        else:
            out[c] = dlogpct(wide[c])
    W = pd.DataFrame(out, index=wide.index)
    W.index.name = "date"
    return W.reset_index()

def lead_lag_scan(driver_wide: pd.DataFrame, target: pd.Series, max_lag: int = 8) -> pd.DataFrame:
    if driver_wide.empty or target.dropna().empty:
        return pd.DataFrame()
    D = driver_wide.set_index("date").sort_index()
    T = target.sort_index()
    rng = range(-max_lag, max_lag+1)
    rows = []
    for c in D.columns:
        x = D[c]
        for L in rng:
            if L >= 0:
                # driver leads target by L: corr(x_{t-L}, y_t)
                corr = x.shift(L).corr(T)
            else:
                # driver lags target by |L|
                corr = x.corr(T.shift(-L))
            rows.append({"series": c, "lag_q": L, "corr": float(corr) if corr==corr else np.nan})
    LL = pd.DataFrame(rows)
    # best positive lead (L>0), and best overall
    best_rows = []
    for s, g in LL.groupby("series"):
        gpos = g[g["lag_q"]>0].copy()
        if not gpos.empty and gpos["corr"].abs().max()==gpos["corr"].abs().max():  # no-op; placeholder
            idx = gpos["corr"].abs().idxmax()
            best_pos = gpos.loc[idx]
        else:
            best_pos = pd.Series({"lag_q": np.nan, "corr": np.nan})
        idx2 = g["corr"].abs().idxmax()
        best_all = g.loc[idx2] if idx2==idx2 else pd.Series({"lag_q": np.nan, "corr": np.nan})
        best_rows.append({"series": s,
                          "best_pos_lead_q": int(best_pos["lag_q"]) if best_pos["lag_q"]==best_pos["lag_q"] else np.nan,
                          "best_pos_corr": float(best_pos["corr"]) if best_pos["corr"]==best_pos["corr"] else np.nan,
                          "best_any_lag_q": int(best_all["lag_q"]) if best_all["lag_q"]==best_all["lag_q"] else np.nan,
                          "best_any_corr": float(best_all["corr"]) if best_all["corr"]==best_all["corr"] else np.nan})
    best = pd.DataFrame(best_rows)
    return LL.sort_values(["series","lag_q"]), best.sort_values(by="best_pos_corr", key=lambda s: s.abs(), ascending=False)

def lstsq_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X
    I = np.eye(X.shape[1]); I[0,0] = 0.0  # don't penalize intercept
    return np.linalg.solve(XtX + lam*I, X.T @ y)

def build_lagged_matrix(df: pd.DataFrame, reg_cols: List[str], lags: int) -> pd.DataFrame:
    X = df.copy().set_index("date")
    for c in reg_cols:
        for L in range(0, lags+1):
            X[f"{c}_L{L}"] = X[c].shift(L)
    keep = [k for k in X.columns if any(k.startswith(f"{c}_L") for c in reg_cols)]
    return X[keep].reset_index()

def arx_nowcast(y: pd.Series, drivers: pd.DataFrame, lags: int, ridge_lambda: float, y_lag: int=1) -> Tuple[pd.DataFrame, Dict]:
    # drivers: wide Δlog% with 'date' index or column
    D = drivers.set_index("date").sort_index()
    Y = y.sort_index()
    # Align
    P = pd.DataFrame({"y": Y}).join(D, how="left")
    # Lagged X matrix
    reg_cols = [c for c in D.columns]
    Xlag = build_lagged_matrix(P.reset_index()[["date"] + reg_cols], reg_cols, lags)
    Z = P.reset_index().merge(Xlag, on="date", how="left").set_index("date").sort_index()
    # AR term(s)
    Xcols = [c for c in Z.columns if c.startswith(tuple(f"{r}_L" for r in reg_cols))]
    for k in range(1, y_lag+1):
        Z[f"y_L{k}"] = Z["y"].shift(k)
        Xcols.append(f"y_L{k}")
    # Drop early rows
    M = Z[Xcols + ["y"]].dropna()
    if M.shape[0] < max(24, 4+lags*len(reg_cols)//3):
        return pd.DataFrame(), {"note":"insufficient_sample","n_obs":int(M.shape[0])}
    X = M[Xcols].values
    yv = M["y"].values
    # add intercept
    X = np.column_stack([np.ones(len(X)), X])
    beta = lstsq_ridge(X, yv, ridge_lambda) if ridge_lambda and ridge_lambda>0 else np.linalg.lstsq(X, yv, rcond=None)[0]
    yhat = X @ beta
    r2 = 1 - ((yv - yhat)**2).sum() / max(1e-12, ((yv - yv.mean())**2).sum())
    # In-sample fit
    fit = pd.DataFrame({"date": M.index, "y_hat": yhat, "y": yv}).sort_values("date")
    # Forward path (hold last known drivers at last values)
    last_date = Z.index.max()
    horizon = 8
    fdates = pd.period_range(last_date.to_period("Q")+1, periods=horizon, freq="Q").to_timestamp(how="end")
    # Build last feature row
    last_row = Z[Xcols].dropna().iloc[-1].values
    fpath = []
    y_prev = M["y"].iloc[-y_lag:] if y_lag>0 else pd.Series(dtype=float)
    for h in range(horizon):
        # replace AR lags with most recent y's
        xrow = last_row.copy()
        # overwrite last y_lag positions (at the end of Xcols)
        for k in range(1, y_lag+1):
            xrow[-k] = y_prev.iloc[-k] if len(y_prev)>=k else np.nan
        xr = np.concatenate([[1.0], np.nan_to_num(xrow, nan=0.0)])
        yh = float(np.dot(xr, beta))
        fpath.append(yh)
        if y_lag>0:
            y_prev = pd.concat([y_prev, pd.Series([yh])], ignore_index=True)
    fc = pd.DataFrame({"date": fdates, "y_hat": fpath})
    out = pd.concat([fit, fc], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out, {"r2": float(r2), "beta_len": int(len(beta)), "y_lag": int(y_lag), "lags": int(lags), "ridge_lambda": float(ridge_lambda)}

def apply_scenarios(drivers_base: pd.DataFrame, scenarios: pd.DataFrame, horizon_q: int, last_hist_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    if scenarios.empty:
        return {}
    # Build forward driver path: hold last values; then apply shocks from effective dates
    D = drivers_base.set_index("date").sort_index().copy()
    fdates = pd.period_range(last_hist_date.to_period("Q")+1, periods=horizon_q, freq="Q").to_timestamp(how="end")
    Df = pd.DataFrame(index=fdates, columns=D.columns, dtype=float)
    # Hold last row forward
    last = D.dropna().iloc[-1]
    for c in D.columns:
        Df[c] = last[c]
    res = {}
    for scen in scenarios["scenario"].unique():
        sub = scenarios[scenarios["scenario"]==scen]
        Z = pd.concat([D, Df], axis=0)
        for _, r in sub.iterrows():
            k = str(r["key"]).strip()
            if k.lower() in ["horizon.quarters","ridge.lambda","arx.y_lag"]:
                continue
            v = float(r["value"]) if pd.notna(r["value"]) else 0.0
            eff_q = None
            if ".pct@" in k.lower():
                series_key, date_key = k.split("@", 1)
                series = series_key.replace(".pct","").upper()
                eff_q = pd.Period(date_key.strip(), freq="Q").to_timestamp(how="end")
            else:
                series = k.replace(".pct","").upper()
                eff_q = fdates[0]
            if series in Z.columns:
                Z.loc[Z.index>=eff_q, series] = Z.loc[Z.index>=eff_q, series].fillna(0.0) + v
        res[scen] = Z.reset_index().rename(columns={"index":"date"})
    return res

def supplier_ewm(suppliers_q: pd.DataFrame) -> pd.DataFrame:
    if suppliers_q.empty: return pd.DataFrame()
    # Expect series including BOOKINGS_USD_M, SHIP_UNITS/SHIP_USD_M, BACKLOG_USD_M, B2B_RATIO
    df = suppliers_q.copy()
    df["series"] = df["series"].astype(str).str.upper()
    # Pivot by category x series
    P = df.pivot_table(index=["date","category"], columns="series", values="value", aggfunc="last")
    # Compute metrics
    out = P.copy()
    if "B2B_RATIO" not in out.columns:
        if "BOOKINGS_USD_M" in out.columns and ("SHIP_USD_M" in out.columns or "BILLINGS_USD_M" in out.columns):
            denom = out["SHIP_USD_M"] if "SHIP_USD_M" in out.columns else out["BILLINGS_USD_M"]
            out["B2B_RATIO"] = out["BOOKINGS_USD_M"] / denom.replace(0, np.nan)
    if "BACKLOG_USD_M" in out.columns:
        ship = out["SHIP_USD_M"] if "SHIP_USD_M" in out.columns else (out["BOOKINGS_USD_M"]*0 + np.nan)
        out["BACKLOG_COVER_Q"] = out["BACKLOG_USD_M"] / (ship.replace(0, np.nan))
    # Lead time stress proxy (if any series resembling LEADTIME_WKS/LEAD_TIME_WKS exists in original)
    # Already handled in constraints to LTZ_*; skip here
    out = out.reset_index()
    # Z-score by category for B2B and Backlog cover
    rows = []
    for cat, g in out.groupby("category"):
        gg = g.sort_values("date").copy()
        for k in ["B2B_RATIO","BACKLOG_COVER_Q"]:
            if k in gg.columns:
                gg[f"{k}_Z"] = zscore(gg[k])
        rows.append(gg)
    return pd.concat(rows, ignore_index=True)

def cycle_phase(agg: pd.DataFrame) -> pd.DataFrame:
    # Simple state from YoY z + slope of 4q MA of YoY
    rows = []
    for coh, g in agg.groupby("cohort"):
        s = g.set_index("date")["capex_yoy_pct"].copy()
        z = zscore(s)
        ma = s.rolling(4, min_periods=2).mean()
        slope = ma.diff()
        state = []
        for i in range(len(s)):
            if z.iloc[i] > 0.5 and slope.iloc[i] > 0: state.append("expansion")
            elif z.iloc[i] > 0.5 and slope.iloc[i] <= 0: state.append("slowdown")
            elif z.iloc[i] <= 0.5 and slope.iloc[i] < 0: state.append("contraction")
            else: state.append("repair")
        df = pd.DataFrame({"date": s.index, "cohort": coh, "phase": state, "yoy_pct": s.values})
        rows.append(df)
    return pd.concat(rows, ignore_index=True).sort_values(["cohort","date"])


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    financials: str
    suppliers: Optional[str]
    prices: Optional[str]
    projects: Optional[str]
    constraints: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    lags: int
    ridge_lambda: float
    horizon: int
    cohort_key: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AI CAPEX cycle tracker — aggregates, diffusion, leads & nowcasts")
    ap.add_argument("--financials", required=True)
    ap.add_argument("--suppliers", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--projects", default="")
    ap.add_argument("--constraints", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2017Q1")
    ap.add_argument("--end", default="2026Q4")
    ap.add_argument("--lags", type=int, default=8)
    ap.add_argument("--ridge_lambda", type=float, default=3.0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--cohort_key", default="cohort", help="column in financials for cohort; leave as 'cohort' or blank")
    ap.add_argument("--outdir", default="out_ai_capex")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load inputs
    FIN = load_financials(args.financials, cohort_key=(args.cohort_key or ""))
    SUP = load_suppliers(args.suppliers) if args.suppliers else pd.DataFrame()
    PRI = load_prices(args.prices) if args.prices else pd.DataFrame()
    PRO = load_projects(args.projects) if args.projects else pd.DataFrame()
    CON = load_constraints(args.constraints) if args.constraints else pd.DataFrame()
    SCN = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Filter window
    start = pd.Period(args.start, freq="Q").to_timestamp(how="end")
    end   = pd.Period(args.end,   freq="Q").to_timestamp(how="end")
    FIN = FIN[(FIN["date"]>=start) & (FIN["date"]<=end)]
    if not SUP.empty: SUP  = SUP[(SUP["date"]>=start) & (SUP["date"]<=end)]
    if not PRI.empty: PRI  = PRI[(PRI["date"]>=start) & (PRI["date"]<=end)]
    if not PRO.empty: PRO  = PRO[(PRO["date"]>=start) & (PRO["date"]<=end)]
    if not CON.empty: CON  = CON[(CON["date"]>=start) & (CON["date"]<=end)]

    # Build panels
    comp_panel, agg = capex_panel(FIN)
    comp_panel.to_csv(outdir / "capex_panel.csv", index=False)
    agg.to_csv(outdir / "aggregates.csv", index=False)

    # Diffusion
    diff = diffusion_indices(comp_panel)
    diff.to_csv(outdir / "diffusion.csv", index=False)

    # Drivers
    drivers = make_driver_wide(SUP, PRI, PRO, CON)
    if not drivers.empty:
        drivers.to_csv(outdir / "drivers_wide.csv", index=False)

    # Lead/lag scan vs GLOBAL YoY
    LL, best = pd.DataFrame(), pd.DataFrame()
    if not drivers.empty:
        target = agg[agg["cohort"]=="GLOBAL"].set_index("date")["capex_yoy_pct"]
        LL, best = lead_lag_scan(drivers, target, max_lag=8)
        LL.to_csv(outdir / "lead_lag_matrix.csv", index=False)
        best.to_csv(outdir / "lead_lag_best.csv", index=False)

    # Nowcast per cohort
    now_rows = []
    kpi = {}
    for coh in agg["cohort"].unique():
        y = agg[agg["cohort"]==coh].set_index("date")["capex_yoy_pct"]
        if y.dropna().shape[0] < (16 + args.lags):
            continue
        out, info = arx_nowcast(y, drivers if not drivers.empty else pd.DataFrame({"date": y.index}), lags=args.lags, ridge_lambda=args.ridge_lambda, y_lag=1)
        if not out.empty:
            out["cohort"] = coh
            now_rows.append(out)
            kpi[coh] = info
    if now_rows:
        nowcast = pd.concat(now_rows, ignore_index=True)
        nowcast.to_csv(outdir / "nowcast_baseline.csv", index=False)
    else:
        nowcast = pd.DataFrame()

    # Supplier early-warning
    ewm = supplier_ewm(SUP) if not SUP.empty else pd.DataFrame()
    if not ewm.empty:
        ewm.to_csv(outdir / "supplier_early_warning.csv", index=False)

    # Cycle phase
    phase = cycle_phase(agg)
    phase.to_csv(outdir / "cycle_phase.csv", index=False)

    # Scenarios
    scen_out = pd.DataFrame()
    if not drivers.empty and not SCN.empty and not nowcast.empty:
        # Determine horizon override if present
        hz = args.horizon
        if not SCN[SCN["key"].str.lower()=="horizon.quarters"].empty:
            try: hz = int(float(SCN[SCN["key"].str.lower()=="horizon.quarters"]["value"].iloc[0]))
            except Exception: pass
        last_hist_date = drivers["date"].max()
        drv_paths = apply_scenarios(drivers, SCN, horizon_q=hz, last_hist_date=last_hist_date)
        scen_rows = []
        for scen, drv_df in drv_paths.items():
            for coh in agg["cohort"].unique():
                y = agg[agg["cohort"]==coh].set_index("date")["capex_yoy_pct"]
                # ARX with same config on shocked drivers
                out, info = arx_nowcast(y, drv_df, lags=args.lags,
                                        ridge_lambda=float(SCN[SCN["key"].str.lower()=="ridge.lambda"]["value"].iloc[0]) if not SCN[SCN["key"].str.lower()=="ridge.lambda"].empty else args.ridge_lambda,
                                        y_lag=int(float(SCN[SCN["key"].str.lower()=="arx.y_lag"]["value"].iloc[0])) if not SCN[SCN["key"].str.lower()=="arx.y_lag"].empty else 1)
                if out.empty: continue
                fut = out[out["date"]>y.index.max()].copy()
                fut["scenario"] = scen
                fut["cohort"] = coh
                # baseline align
                base = nowcast[(nowcast["cohort"]==coh) & (nowcast["date"].isin(fut["date"]))]
                fut = fut.merge(base[["date","y_hat"]].rename(columns={"y_hat":"y_hat_base"}), on="date", how="left")
                fut["delta_vs_base_pp"] = fut["y_hat"] - fut["y_hat_base"]
                scen_rows.append(fut)
        if scen_rows:
            scen_out = pd.concat(scen_rows, ignore_index=True)
            scen_out.to_csv(outdir / "scenarios_out.csv", index=False)

    # Summary
    latest_date = agg["date"].max()
    latest_row = agg[agg["date"]==latest_date]
    latest_global = latest_row[latest_row["cohort"]=="GLOBAL"]["capex_yoy_pct"]
    latest_global_val = float(latest_global.iloc[0]) if not latest_global.empty else np.nan
    # Top-3 positive-lead drivers
    top_leads = []
    if not best.empty:
        bpos = best.dropna(subset=["best_pos_lead_q"]).sort_values(by="best_pos_corr", key=lambda s: s.abs(), ascending=False).head(3)
        for _, r in bpos.iterrows():
            top_leads.append({"series": str(r["series"]), "lead_q": int(r["best_pos_lead_q"]), "corr": float(r["best_pos_corr"])})
    # Diffusion latest
    diff_latest = diff[diff["date"]==latest_date]
    d_global = diff_latest[diff_latest["cohort"]=="GLOBAL"]
    diff_qoq = float(d_global["diff_qoq"].iloc[0]) if not d_global.empty else np.nan
    diff_yoy = float(d_global["diff_yoy"].iloc[0]) if not d_global.empty else np.nan

    summary = {
        "latest_quarter": str(latest_date.date()) if pd.notna(latest_date) else None,
        "global_capex_yoy_pct": latest_global_val,
        "diffusion_global": {"qoq_up_share": diff_qoq, "yoy_up_share": diff_yoy},
        "top_leading_drivers": top_leads,
        "nowcast_info": kpi
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        financials=args.financials, suppliers=(args.suppliers or None), prices=(args.prices or None),
        projects=(args.projects or None), constraints=(args.constraints or None), scenarios=(args.scenarios or None),
        start=args.start, end=args.end, lags=args.lags, ridge_lambda=args.ridge_lambda,
        horizon=args.horizon, cohort_key=(args.cohort_key or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== AI CAPEX Cycle Tracker ==")
    print(f"Latest {summary['latest_quarter']} | Global YoY {summary['global_capex_yoy_pct'] if summary['global_capex_yoy_pct']==summary['global_capex_yoy_pct'] else float('nan'):.2f}% | Diff YoY {diff_yoy if diff_yoy==diff_yoy else float('nan'):.2%}")
    if top_leads:
        tops = ", ".join([f"{t['series']} (lead {t['lead_q']}q, r={t['corr']:+.2f})" for t in top_leads])
        print("Top leads:", tops)
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
