#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uk_food_inflation.py — Decompose, nowcast & scenario-test UK Food CPI inflation

What this does
--------------
Takes monthly UK CPI (food headline or detailed) and a panel of drivers (commodities, FX, energy,
wages, freight, retail margins, etc.). It:

1) Builds a clean monthly panel
2) Computes Food CPI YoY and MoM (if an index is provided)
3) Estimates pass-through elasticities with distributed lags
4) Produces a decomposition (contributions by driver)
5) Nowcasts / forecasts Food CPI YoY (baseline)
6) Applies scenario shocks (e.g., wheat -20%, GBP +5%, energy -10%) to show deltas
7) Writes tidy CSVs + a compact JSON summary

Core outputs (CSV)
------------------
- food_panel.csv         Cleaned time series: Food YoY/MoM (+ drivers, wide)
- decomposition.csv      Per month contributions by driver group (+ fitted YoY, residual)
- elasticities.csv       Sum of coefficients across lags per driver & "peak lag"
- forecast_baseline.csv  Baseline fitted & forward path (YoY) for horizon months
- scenarios_out.csv      For each scenario: YoY path and Δ vs baseline
- summary.json           KPIs (latest YoY, 3m ann., in-sample R², top drivers)
- config.json            Run configuration dump

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--food_cpi food.csv    REQUIRED
  Columns (either path):
    A) date, yoy_pct
    B) date, index  (price index level; we compute MoM & YoY)
  Optional: series/category to filter (we auto-choose "Food" if multiple present via --food_series)

--drivers drivers.csv   OPTIONAL (long panel)
  Columns: date, series, value
  Examples for `series`: WHEAT_USD, DAIRY_IDX, VEG_OILS_IDX, MEAT_IDX, GAS_UK_EUR_MWH,
                         ELEC_UK_IDX, OIL_BRENT_USD, GBP_EUR, GBP_USD, WAGES_AWE_YOY,
                         FREIGHT_FBXD, PACKAGING_PULP_IDX, RETAIL_MARGIN_IDX, etc.

--scenarios scenarios.csv OPTIONAL
  Columns: scenario, name, key, value
  Supported keys (apply from T+1 onward unless suffixed with @YYYY-MM):
    <SERIES>.pct = -20            (level shock in %, applied multiplicatively to the driver value)
    horizon.months = 12           (forecast horizon)
    ridge.lambda = 5.0            (override regularization)
    arx.y_lag = 1                 (include 1 lag of YoY as AR term)

CLI options
-----------
--start 2015-01
--end   2025-12
--food_series "FOOD"        (if food_cpi file contains multiple series/categories)
--lags 12                   (distributed lags for each driver)
--ridge_lambda 3.0          (L2 in pct-units; 0 = OLS)
--horizon 12
--outdir out_uk_food

Notes
-----
- Dependent variable is Food CPI YoY (%). Regressors are (optionally standardized) driver changes.
- By default we use Δlog (pct change) for drivers whose units are levels (FX, prices, indices).
- For already-percentage drivers (e.g., WAGES_AWE_YOY), we use them as-is (you can suffix name with _YOY to force this).
- Base effects are implicitly captured through lag structure and the AR term (optional).
- This is a research tool; coefficients are not policy advice.
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

def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def ensure_dir(path: str) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def pct_change(series: pd.Series) -> pd.Series:
    return series.astype(float).replace(0, np.nan).pipe(np.log).diff() * 100.0  # Δlog * 100 ≈ % change

def yoy_pct(series: pd.Series) -> pd.Series:
    return (series.astype(float).pct_change(12)) * 100.0

def mom_pct(series: pd.Series) -> pd.Series:
    return (series.astype(float).pct_change(1)) * 100.0

def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return z-scored df, with means & stds (ddof=0)."""
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, np.nan)
    z = (df - mu) / sd
    return z, mu, sd

def lstsq_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    # Solve (X'X + λI)β = X'y  ; do not penalize intercept (assumed first column)
    XtX = X.T @ X
    I = np.eye(X.shape[1]); I[0,0] = 0.0
    beta = np.linalg.solve(XtX + lam * I, X.T @ y)
    return beta

def var_explained(y: np.ndarray, yhat: np.ndarray) -> float:
    ssr = np.sum((y - yhat)**2)
    sst = np.sum((y - y.mean())**2)
    return float(1.0 - ssr / max(sst, 1e-12))


# ----------------------------- loaders -----------------------------

def load_food(path: str, food_series: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Flexible columns
    date_c = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={date_c: "date"})
    df["date"] = to_month(df["date"])
    # If multiple series/categories, pick the requested one
    if ncol(df, "series") or ncol(df, "category"):
        sc = ncol(df, "series") or ncol(df, "category")
        val_c = ncol(df, "yoy_pct") or ncol(df, "index") or ncol(df, "value")
        df = df.rename(columns={sc: "series", val_c: "value"})
        if food_series:
            sub = df[df["series"].astype(str).str.upper().str.contains(food_series.upper())].copy()
            if not sub.empty:
                df = sub
        # If multiple still, collapse by mean (or choose first)
        if df["series"].nunique() > 1:
            df = df.groupby("date", as_index=False)["value"].mean()
        else:
            df = df[["date","value"]]
    else:
        val_c = ncol(df, "yoy_pct") or ncol(df, "index") or ncol(df, "value")
        df = df.rename(columns={val_c: "value"})[["date","value"]]

    # If value looks like YoY (range typical ±0..30), accept; else treat as index and derive YoY & MoM
    s = df.set_index("date")["value"].sort_index()
    looks_like_yoy = (s.dropna().abs().median() < 40.0) and (s.dropna().abs().max() < 200.0)
    out = pd.DataFrame({"date": s.index})
    if looks_like_yoy and s.notna().sum() > 24:
        out["food_yoy"] = s.values
        out["food_mom"] = np.nan  # unknown
        out["food_index"] = np.nan
    else:
        out["food_index"] = s.values
        out["food_mom"] = mom_pct(s)
        out["food_yoy"] = yoy_pct(s)
    return out.sort_values("date")

def load_drivers(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"series") or "series"): "series",
           (ncol(df,"value") or "value"): "value"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["series"] = df["series"].astype(str).str.upper().str.strip()
    df["value"] = num(df["value"])
    return df.sort_values(["series","date"])

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"): "scenario",
           (ncol(df,"name") or "name"): "name",
           (ncol(df,"key") or "key"): "key",
           (ncol(df,"value") or "value"): "value"}
    return df.rename(columns=ren)


# ----------------------------- feature engineering -----------------------------

def make_driver_wide(drivers_long: pd.DataFrame) -> pd.DataFrame:
    if drivers_long.empty: return pd.DataFrame()
    # Decide transformation per series:
    # - If series name ends with _YOY, use value as-is (%).
    # - Else compute Δlog * 100 (approx % change).
    out = []
    for s, g in drivers_long.groupby("series"):
        gg = g.sort_values("date").set_index("date")["value"]
        if s.endswith("_YOY"):
            x = gg.copy()  # already %
        else:
            x = pct_change(gg)
        out.append(x.rename(s))
    W = pd.concat(out, axis=1)
    W.index.name = "date"
    return W.reset_index()

def create_lagged(df: pd.DataFrame, cols: List[str], lags: int) -> pd.DataFrame:
    X = df.copy().set_index("date")
    for c in cols:
        for L in range(0, lags+1):
            X[f"{c}_L{L}"] = X[c].shift(L)
    keep = [k for k in X.columns if any(k.startswith(c+"_L") for c in cols)]
    X = X[keep].reset_index()
    return X

def align_panel(food: pd.DataFrame, drivers_wide: pd.DataFrame) -> pd.DataFrame:
    if drivers_wide.empty:
        return food.copy()
    d = food.merge(drivers_wide, on="date", how="left")
    return d.sort_values("date")


# ----------------------------- model & decomposition -----------------------------

def build_design(df: pd.DataFrame, y_col: str, reg_cols: List[str], lags: int, include_y_lag: int, ridge_lambda: float):
    # Create lagged X
    base = df[["date"] + reg_cols].copy()
    Xlags = create_lagged(base, reg_cols, lags)
    Z = df.merge(Xlags, on="date", how="left").sort_values("date")
    # Optional AR term(s) for YoY
    y = Z[y_col].astype(float).values
    Xcols = [c for c in Z.columns if any(c.startswith(f"{rc}_L") for rc in reg_cols)]
    if include_y_lag > 0:
        for k in range(1, include_y_lag+1):
            col = f"yoy_L{k}"
            Z[col] = Z[y_col].shift(k)
            Xcols.append(col)
    # Drop rows with NaNs in y or too-early rows without lags
    good = Z[Xcols + [y_col]].dropna()
    if good.empty or good.shape[0] < 24:
        raise ValueError("Not enough data after lags to fit model.")
    y_fit = good[y_col].values
    X_fit = good[Xcols].values
    # Add intercept
    X_fit = np.column_stack([np.ones(len(X_fit)), X_fit])
    # Fit
    beta = lstsq_ridge(X_fit, y_fit, ridge_lambda) if ridge_lambda and ridge_lambda>0 else np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
    y_hat = X_fit @ beta
    r2 = var_explained(y_fit, y_hat)
    return {
        "Xcols": ["intercept"] + Xcols,
        "beta": beta,
        "r2": r2,
        "train_index": good.index,  # original df indices for alignment
        "Z": Z,                     # full panel with features
    }

def contributions(model: dict, y_col: str) -> pd.DataFrame:
    Z = model["Z"].copy()
    Xcols = model["Xcols"]
    beta = model["beta"]
    # Align to training rows only (no NaNs)
    idx = model["train_index"]
    Zt = Z.loc[idx, :].copy()
    # Build contributions per column
    Xmat = np.column_stack([np.ones(len(idx))] + [Zt[c].values for c in Xcols[1:]])
    yhat = Xmat @ beta
    contrib = {Xcols[i]: Xmat[:, i] * beta[i] for i in range(len(Xcols))}
    dfc = pd.DataFrame(contrib, index=Zt.index)
    dfc["yhat"] = yhat
    dfc[y_col] = Zt[y_col].values
    dfc["residual"] = dfc[y_col] - dfc["yhat"]
    dfc["date"] = Zt["date"].values
    # Group contributions by driver ROOT (before _L#)
    cols = Xcols[1:]  # exclude intercept
    root = pd.Series({c: c.split("_L")[0] for c in cols})
    groups = sorted(root.unique())
    by_group = pd.DataFrame({"date": dfc["date"]})
    for g in groups:
        cols_g = [c for c in cols if root[c]==g]
        by_group[g] = dfc[cols_g].sum(axis=1).values
    by_group["intercept"] = dfc["intercept"].values
    by_group["yhat"] = dfc["yhat"].values
    by_group[y_col] = dfc[y_col].values
    by_group["residual"] = dfc["residual"].values
    return by_group.sort_values("date")

def elasticities_table(model: dict) -> pd.DataFrame:
    cols = model["Xcols"][1:]  # drop intercept
    beta = model["beta"][1:]
    rows = []
    # Sum over lags for each driver root; track peak lag
    roots = {}
    for c, b in zip(cols, beta):
        r = c.split("_L")[0]
        L = int(c.split("_L")[1]) if "_L" in c else 0
        roots.setdefault(r, {"sum_beta":0.0, "lags":{}})
        roots[r]["sum_beta"] += float(b)
        roots[r]["lags"][L] = float(b)
    for r, d in roots.items():
        peak_lag = max(d["lags"], key=lambda L: abs(d["lags"][L])) if d["lags"] else 0
        rows.append({"driver": r, "beta_sum": d["sum_beta"], "peak_lag": int(peak_lag), "beta_peak": float(d["lags"].get(peak_lag, np.nan))})
    out = pd.DataFrame(rows).sort_values(by="beta_sum", key=lambda s: s.abs(), ascending=False)
    return out

def forecast_path(model: dict, y_col: str, horizon: int) -> pd.DataFrame:
    Z = model["Z"].copy().sort_values("date").reset_index(drop=True)
    Xcols = model["Xcols"]
    beta = model["beta"]
    # We'll roll forward deterministically using last known features (no driver extrapolation beyond persistence)
    dates = list(Z["date"].dropna().unique())
    last = pd.to_datetime(dates[-1])
    fdates = pd.period_range(last.to_period("M")+1, periods=horizon, freq="M").to_timestamp()
    # Construct future frame with NaNs for future features (we'll carry-forward last available values)
    Zf = pd.DataFrame({"date": fdates})
    cols_needed = [c for c in Xcols[1:]]  # exclude intercept
    # Build each lagged feature forward by shifting the known columns
    Zfull = pd.concat([Z, Zf], ignore_index=True)
    for c in cols_needed:
        if c in Zfull.columns:
            # Already exists (y lags); will be filled as we generate forecasts
            continue
        root, lag = c.split("_L")[0], int(c.split("_L")[1])
        # If root base exists in Z, recreate lagged from that base (requires historical base)
        base_cols = [k for k in Z.columns if k == root]  # original driver change column
        if base_cols:
            base = Z[["date", root]].copy()
            s = base.set_index("date")[root].shift(lag)
            Zfull[c] = s.reindex(Zfull["date"]).values
        else:
            # y lags handled below
            Zfull[c] = np.nan

    # Fill forward driver lags with last observed values
    for c in cols_needed:
        if c.startswith("yoy_L"):  # AR term handled dynamically
            continue
        Zfull[c] = Zfull[c].ffill()

    # Recursive AR term if present
    include_y_lags = [c for c in cols_needed if c.startswith("yoy_L")]
    if include_y_lags:
        # Initialize with historical y
        y_hist = Z.set_index("date")[y_col].copy()
        for d in fdates:
            # Build design row for date d
            row = [1.0]  # intercept
            for c in Xcols[1:]:
                if c.startswith("yoy_L"):
                    L = int(c.split("_L")[1])
                    val = y_hist.shift(L).reindex([d]).iloc[0] if d in y_hist.shift(L).index else np.nan
                else:
                    val = Zfull.set_index("date").loc[d, c]
                row.append(val if pd.notna(val) else 0.0)
            y_next = float(np.dot(row, beta))
            y_hist.loc[d] = y_next
        y_path = y_hist.reindex(fdates)
    else:
        # No AR term: static projection using last feature vector
        last_row = Zfull.set_index("date").loc[dates[-1], Xcols[1:]].fillna(0.0).values
        y_next = []
        for _ in range(horizon):
            row = np.concatenate([[1.0], last_row])
            y_next.append(float(np.dot(row, beta)))
        y_path = pd.Series(y_next, index=fdates)

    out_hist = pd.DataFrame({"date": Z.loc[model["train_index"], "date"], "y_hat": (np.column_stack([np.ones(len(model["train_index"]))] + [Z.loc[model["train_index"], c].values for c in Xcols[1:]]) @ beta)})
    out_forw = pd.DataFrame({"date": fdates, "y_hat": y_path.values})
    out = pd.concat([out_hist, out_forw], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out

def apply_scenarios(model: dict, base_df: pd.DataFrame, scenarios_df: pd.DataFrame, y_col: str, default_horizon: int, default_lambda: float) -> pd.DataFrame:
    if scenarios_df.empty:
        return pd.DataFrame()
    Z = model["Z"].copy().sort_values("date").reset_index(drop=True)
    Xcols = model["Xcols"]; beta = model["beta"]
    last_date = pd.to_datetime(Z["date"].dropna().max())
    rows_all = []
    for scen in scenarios_df["scenario"].unique():
        sub = scenarios_df[scenarios_df["scenario"]==scen]
        # Horizon / lambda / AR lags overrides if any
        horizon = int(float(sub[sub["key"].str.lower()=="horizon.months"]["value"].iloc[0])) if not sub[sub["key"].str.lower()=="horizon.months"].empty else default_horizon
        # Build copy of driver base (not lagged) to which we'll apply shocks
        # Start with original drivers (non-lagged) that were used to build lagged features
        driver_roots = sorted({c.split("_L")[0] for c in Xcols[1:] if not c.startswith("yoy_L")})
        base_cols = [c for c in base_df.columns if c in driver_roots]
        # If no explicit base_df passed, reconstruct from Z by "unlagging" L0
        base_series = pd.DataFrame({"date": Z["date"]})
        for r in driver_roots:
            colL0 = f"{r}_L0"
            if colL0 in Z.columns:
                base_series[r] = Z[colL0]
        # Apply shocks
        Zproj = Z.copy()
        for _, r in sub.iterrows():
            k = str(r["key"]).strip()
            v = float(r["value"]) if pd.notna(r["value"]) else 0.0
            if k.lower() in ["horizon.months","ridge.lambda","arx.y_lag"]:
                continue
            # Allow time-scoped shocks: SERIES.pct@YYYY-MM
            if ".pct@" in k.lower():
                series_key, date_key = k.split("@", 1)
                series = series_key.replace(".pct","").upper()
                eff_date = pd.to_datetime(date_key.strip()).to_period("M").to_timestamp()
            else:
                series = k.replace(".pct","").upper()
                eff_date = last_date + pd.offsets.MonthBegin(1)
            # We apply pct *additive* to Δlog(%) series (i.e., shift the change in that month)
            # For example, WHEAT_USD.pct = -20 means: in that month, Δlog (%) for WHEAT is lowered by 20pp.
            # (You can tune to your preference.)
            colL0 = f"{series}_L0"
            if colL0 in Zproj.columns:
                Zproj.loc[Zproj["date"]>=eff_date, colL0] = (Zproj.loc[Zproj["date"]>=eff_date, colL0].fillna(0.0) + v)

        # Forward to horizon (hold features at last available + shocks from eff_date onward)
        fdates = pd.period_range(last_date.to_period("M")+1, periods=horizon, freq="M").to_timestamp()
        Zf = pd.DataFrame({"date": fdates})
        Zfull = pd.concat([Zproj, Zf], ignore_index=True)

        # Fill all lag columns forward
        for c in Xcols[1:]:
            if c.startswith("yoy_L"):
                continue
            Zfull[c] = Zfull[c].ffill()

        # Recursive AR term if present
        include_y_lags = [c for c in Xcols[1:] if c.startswith("yoy_L")]
        if include_y_lags:
            y_hist = Zproj.set_index("date")[y_col].copy()
            for d in fdates:
                row = [1.0]
                for c in Xcols[1:]:
                    if c.startswith("yoy_L"):
                        L = int(c.split("_L")[1])
                        val = y_hist.shift(L).reindex([d]).iloc[0] if d in y_hist.shift(L).index else np.nan
                    else:
                        val = Zfull.set_index("date").loc[d, c]
                    row.append(val if pd.notna(val) else 0.0)
                y_next = float(np.dot(row, beta))
                y_hist.loc[d] = y_next
            y_path = y_hist.reindex(fdates).values
        else:
            last_row = Zfull.set_index("date").loc[last_date, Xcols[1:]].fillna(0.0).values
            y_path = [float(np.dot(np.concatenate([[1.0], last_row]), beta)) for _ in range(horizon)]

        rows_all.append(pd.DataFrame({"scenario": scen, "date": fdates, "y_hat": y_path}))
    return pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    food_cpi: str
    drivers: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    food_series: Optional[str]
    lags: int
    ridge_lambda: float
    horizon: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="UK Food CPI inflation — decomposition, nowcast & scenarios")
    ap.add_argument("--food_cpi", required=True)
    ap.add_argument("--drivers", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--food_series", default="")
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--ridge_lambda", type=float, default=3.0)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--outdir", default="out_uk_food")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load inputs
    food = load_food(args.food_cpi, args.food_series or None)
    drivers_long = load_drivers(args.drivers) if args.drivers else pd.DataFrame()
    scenarios_df = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Filter window
    start = pd.to_datetime(args.start).to_period("M").to_timestamp()
    end   = pd.to_datetime(args.end).to_period("M").to_timestamp()
    food = food[(food["date"]>=start) & (food["date"]<=end)].copy()
    if not drivers_long.empty:
        drivers_long = drivers_long[(drivers_long["date"]>=start - pd.offsets.DateOffset(months=args.lags+2)) & (drivers_long["date"]<=end)].copy()

    # Drivers → wide % changes (Δlog * 100) or _YOY kept as provided
    drivers_wide = make_driver_wide(drivers_long) if not drivers_long.empty else pd.DataFrame()

    # Panel for export
    panel = align_panel(food, drivers_wide)
    panel.to_csv(outdir / "food_panel.csv", index=False)

    # If we don't have enough data, stop after writing panel
    if panel["food_yoy"].dropna().shape[0] < (24 + args.lags):
        (outdir / "summary.json").write_text(json.dumps({
            "note": "Not enough data to estimate model after applying lags.",
            "obs_food_yoy": int(panel["food_yoy"].dropna().shape[0])
        }, indent=2))
        print("Not enough data to estimate model. Wrote food_panel.csv and summary.json.")
        return

    # Choose regressors
    reg_cols = [c for c in panel.columns if c not in ["date","food_yoy","food_mom","food_index"]]
    include_y_lag = 1  # simple ARX(1) term; can be overridden via scenarios if desired (handled in scenario-only)
    # Fit baseline model
    model = build_design(panel, y_col="food_yoy", reg_cols=reg_cols, lags=args.lags, include_y_lag=include_y_lag, ridge_lambda=args.ridge_lambda)

    # Decomposition
    decomp = contributions(model, y_col="food_yoy")
    decomp.to_csv(outdir / "decomposition.csv", index=False)

    # Elasticities (sum of betas across lags)
    elast = elasticities_table(model)
    elast.to_csv(outdir / "elasticities.csv", index=False)

    # Baseline forecast
    fc = forecast_path(model, y_col="food_yoy", horizon=args.horizon)
    fc.to_csv(outdir / "forecast_baseline.csv", index=False)

    # Scenarios
    scen_out = apply_scenarios(model, panel, scenarios_df, y_col="food_yoy", default_horizon=args.horizon, default_lambda=args.ridge_lambda) if not scenarios_df.empty else pd.DataFrame()
    if not scen_out.empty:
        scen_base = fc.rename(columns={"y_hat":"y_hat_baseline"})[["date","y_hat_baseline"]]
        scen_merged = scen_out.merge(scen_base, on="date", how="left")
        scen_merged["delta_vs_baseline"] = scen_merged["y_hat"] - scen_merged["y_hat_baseline"]
        scen_merged.to_csv(outdir / "scenarios_out.csv", index=False)

    # KPIs
    latest = panel["date"].max()
    latest_yoy = float(panel.loc[panel["date"]==latest, "food_yoy"].iloc[0]) if not panel.loc[panel["date"]==latest, "food_yoy"].empty else np.nan
    # 3m annualized (approx): sum of last 3 MoM * 4
    if "food_mom" in panel.columns and panel["food_mom"].notna().tail(3).shape[0]==3:
        m3_ann = float(panel["food_mom"].tail(3).sum() * 4.0)
    else:
        m3_ann = np.nan
    # Top contributors (last available month)
    last_decomp = decomp[decomp["date"]==decomp["date"].max()].drop(columns=["yhat","food_yoy","residual","intercept"]) if not decomp.empty else pd.DataFrame()
    top_contrib = []
    if not last_decomp.empty:
        s = last_decomp.drop(columns=["date"]).T[0].sort_values(key=lambda x: x.abs(), ascending=False)
        for k in s.index[:5]:
            top_contrib.append({"driver": k, "pp_contribution": float(s[k])})

    summary = {
        "latest": str(latest.date()) if pd.notna(latest) else None,
        "food_yoy_latest_pct": latest_yoy,
        "food_3m_ann_pct": m3_ann,
        "in_sample_R2": model["r2"],
        "top_contributors_last": top_contrib
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        food_cpi=args.food_cpi, drivers=(args.drivers or None), scenarios=(args.scenarios or None),
        start=args.start, end=args.end, food_series=(args.food_series or None),
        lags=args.lags, ridge_lambda=args.ridge_lambda, horizon=args.horizon, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== UK Food Inflation Model ==")
    print(f"Latest: {summary['latest']} | YoY {summary['food_yoy_latest_pct']:.2f}% | 3m ann {summary['food_3m_ann_pct'] if summary['food_3m_ann_pct']==summary['food_3m_ann_pct'] else float('nan'):.2f}% | R² {summary['in_sample_R2']:.2f}")
    if top_contrib:
        tops = ", ".join([f"{t['driver']} {t['pp_contribution']:+.2f}pp" for t in top_contrib[:3]])
        print("Top drivers (last month):", tops)
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
