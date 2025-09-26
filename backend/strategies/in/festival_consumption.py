#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
festival_consumption.py — Festival uplift, cannibalization & forecast engine
----------------------------------------------------------------------------

What this does
==============
Given daily sales and a festival calendar, this script:

1) Cleans & aligns data (optionally: promos, macro, weather, prices).
2) Builds a transparent "baseline" using a ridge OLS with calendar controls:
      Sales ~ Intercept + Trend + DOW + Month + Macro + Promo + Weather (+Region/Category FE)
   (Festival indicators are EXCLUDED when computing the baseline.)
3) Estimates *festival uplift* around each event via counterfactual:
      Uplift = Actual − Baseline   within a window [−pre, +post] days.
   Also computes pre/post dips to estimate *demand shifting* (cannibalization).
4) Aggregates to festival, region, and category; computes effectiveness KPIs:
      lift %, ROI proxy (if promo cost), share of uplift vs cannibalization.
5) Produces a simple *forecast* for future festivals by reusing historical
   per-festival uplift per region/category and scaling by macro/promo intensity.

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--sales sales.csv           REQUIRED (long)
  Columns (min): date, revenue
  Optional: region, category, units, price
  If revenue missing but units+price available, revenue=units*price.

--calendar calendar.csv     REQUIRED
  Columns: date, festival [, region] [, category] [, weight] [, type]

--promos promos.csv         OPTIONAL (promo intensity/cost)
  Columns: date [, region] [, category] [, promo_intensity] [, promo_cost]

--macro macro.csv           OPTIONAL (macro scaler)
  Columns: date [, macro_index]  (e.g., consumer confidence or income proxy)

--weather weather.csv       OPTIONAL
  Columns: date [, region] [, tmean] [, rain] (any numeric will be used)

CLI
---
Example:
  python festival_consumption.py \
    --sales sales.csv --calendar calendar.csv --promos promos.csv \
    --macro macro.csv --weather weather.csv \
    --pre 7 --post 7 --ridge 5.0 --min_days 180 \
    --scope REGION --outdir out_festival

Key parameters
--------------
--pre 7             Days before event included in the window (inclusive)
--post 7            Days after event included in the window (inclusive)
--ridge 5.0         Ridge penalty λ (>=0)
--min_days 180      Minimum history per (region,category) to fit model
--scope AUTO        Aggregation key for modeling: AUTO/REGION/CATEGORY/BOTH
--forecast_days 90  Forecast future festivals within N days from last sales date
--outdir out_festival

Outputs
-------
- baseline_vs_actual.csv     Per date×group: actual, baseline, delta
- uplift_by_event.csv        Per event×group: uplift_abs, lift_pct, cannibalization
- uplift_by_festival.csv     Aggregated by festival name
- uplift_by_group.csv        Aggregated by region/category
- forecast.csv               Simple forward view for upcoming festivals
- summary.json               Headline KPIs
- config.json                Run configuration

Notes
-----
• Baseline excludes festival effects by fitting on non-window dates.
• Ridge OLS is closed-form: β = (X'X + λI)^−1 X'Y
• If macro/promos/weather provided, each numeric column becomes a regressor.
• Cannibalization is the sum of negative deltas in pre/post windows; uplift is
  the positive delta on the event window. Reported "net_uplift = uplift + cann".
• Monetary fields are assumed in the same currency.

DISCLAIMER: Research tool; validate before production use.
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

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if str(c).lower() == t: return c
    for c in df.columns:
        if t in str(c).lower(): return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    d = pd.get_dummies(series.astype("category"), prefix=prefix, drop_first=True)
    d.columns = d.columns.astype(str)
    return d

def ridge_ols(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form ridge: beta = (X'X + lam*I)^-1 X'y
    """
    XtX = X.T @ X
    k = XtX.shape[0]
    beta = np.linalg.pinv(XtX + lam * np.eye(k)) @ (X.T @ y)
    return beta

def design_matrix(df: pd.DataFrame, use_cols: List[str], add_dow=True, add_month=True, add_trend=True) -> Tuple[np.ndarray, List[str]]:
    """
    Build X with intercept + optional DOW/MONTH + trend + numeric covariates from use_cols.
    Returns (X, names). Assumes df sorted by date.
    """
    pieces = []
    names = []

    # Intercept
    pieces.append(np.ones((len(df), 1)))
    names.append("intercept")

    # Trend (linear)
    if add_trend:
        t = np.arange(len(df)).reshape(-1, 1)
        pieces.append(t)
        names.append("trend")

    # Day-of-week & Month
    if add_dow:
        d = one_hot(df["date"].dt.dayofweek, "dow")
        pieces.append(d.values)
        names.extend(list(d.columns))
    if add_month:
        m = one_hot(df["date"].dt.month, "mon")
        pieces.append(m.values)
        names.extend(list(m.columns))

    # Numeric regressors
    for col in use_cols:
        if col in df.columns:
            v = safe_num(df[col]).values.reshape(-1, 1)
            pieces.append(v)
            names.append(col)

    X = np.concatenate(pieces, axis=1) if pieces else np.empty((len(df), 0))
    return X, names

def daterange_mask(dates: pd.DatetimeIndex, center: pd.Timestamp, pre: int, post: int) -> np.ndarray:
    lo = center - pd.Timedelta(days=pre)
    hi = center + pd.Timedelta(days=post)
    return (dates >= lo) & (dates <= hi)

def infer_group_scope(df: pd.DataFrame, scope: str) -> List[Tuple[str, Optional[str]]]:
    """
    Returns list of (region, category) keys to iterate over based on scope.
    """
    regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else [None]
    cats    = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else [None]
    out = []
    if scope == "REGION":
        for r in regions:
            out.append((r, None))
    elif scope == "CATEGORY":
        for c in cats:
            out.append((None, c))
    elif scope == "BOTH":
        for r in regions:
            for c in cats:
                out.append((r, c))
    else:  # AUTO
        if len(regions) > 1 and len(cats) > 1:
            for r in regions:
                for c in cats:
                    out.append((r, c))
        elif len(regions) > 1:
            for r in regions:
                out.append((r, None))
        elif len(cats) > 1:
            for c in cats:
                out.append((None, c))
        else:
            out = [(None, None)]
    return out


# ----------------------------- loaders -----------------------------

def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt: "date"})
    df["date"] = to_date(df["date"])
    if ncol(df, "revenue"):
        df = df.rename(columns={ncol(df,"revenue"): "revenue"})
    else:
        # try to build from units*price
        if ncol(df,"units") and ncol(df,"price"):
            df["revenue"] = safe_num(df[ncol(df,"units")]) * safe_num(df[ncol(df,"price")])
        else:
            raise ValueError("sales.csv must have 'revenue' or both 'units' and 'price'.")
    if ncol(df, "region"):
        df = df.rename(columns={ncol(df,"region"): "region"})
    if ncol(df, "category"):
        df = df.rename(columns={ncol(df,"category"): "category"})
    # Normalize dtypes
    for c in ["region","category"]:
        if c in df.columns: df[c] = df[c].astype(str)
    df = df.sort_values("date")
    return df

def load_calendar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={dt:"date", (ncol(df,"festival") or "festival"):"festival"})
    df["date"] = to_date(df["date"])
    df["festival"] = df["festival"].astype(str)
    if ncol(df,"region"):
        df = df.rename(columns={ncol(df,"region"):"region"})
        df["region"] = df["region"].astype(str)
    if ncol(df,"category"):
        df = df.rename(columns={ncol(df,"category"):"category"})
        df["category"] = df["category"].astype(str)
    if ncol(df,"weight"):
        df = df.rename(columns={ncol(df,"weight"):"weight"})
    else:
        df["weight"] = 1.0
    if ncol(df,"type"):
        df = df.rename(columns={ncol(df,"type"):"type"})
    return df.sort_values("date")

def load_promos(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    if ncol(df,"promo_intensity") is None:
        # pick any numeric columns as proxies
        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            df = df.rename(columns={num_cols[0]:"promo_intensity"})
        else:
            df["promo_intensity"] = 0.0
    else:
        df = df.rename(columns={ncol(df,"promo_intensity"):"promo_intensity"})
    if ncol(df,"promo_cost"):
        df = df.rename(columns={ncol(df,"promo_cost"):"promo_cost"})
    if ncol(df,"region"):
        df = df.rename(columns={ncol(df,"region"):"region"}); df["region"]=df["region"].astype(str)
    if ncol(df,"category"):
        df = df.rename(columns={ncol(df,"category"):"category"}); df["category"]=df["category"].astype(str)
    return df

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    # pick numeric columns
    num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("macro.csv must include at least one numeric column.")
    # Rename first numeric to 'macro_index' for convenience but keep all
    if "macro_index" not in df.columns:
        df = df.rename(columns={num_cols[0]:"macro_index"})
    return df

def load_weather(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    if ncol(df,"region"):
        df = df.rename(columns={ncol(df,"region"):"region"}); df["region"]=df["region"].astype(str)
    # keep numeric weather vars
    keep = ["date"] + ([ "region" ] if "region" in df.columns else [])
    keep += [c for c in df.columns if c not in keep and pd.api.types.is_numeric_dtype(df[c])]
    return df[keep]


# ----------------------------- core computation -----------------------------

def fit_baseline_for_group(
    G: pd.DataFrame,
    pre: int, post: int,
    ridge_lambda: float,
    extra_regs: List[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fit ridge baseline on NON-festival-window days, then predict for all dates.
    Returns (df with actual/baseline/delta, diagnostics).
    """
    df = G.sort_values("date").copy()
    # Mark festival windows (for exclusion)
    df["is_festival_window"] = 0
    for d in df.loc[df["is_festival_day"]==1, "date"].unique():
        mask = daterange_mask(df["date"], d, pre, post)
        df.loc[mask, "is_festival_window"] = 1

    train = df[df["is_festival_window"]==0].copy()
    if train.empty or train["revenue"].count() < 30:
        # Nothing to fit
        df["baseline"] = df["revenue"].median()
        df["delta"] = df["revenue"] - df["baseline"]
        return df, {"used_median": True, "n_train": int(train.shape[0])}

    # Build design matrix
    X_train, names = design_matrix(train, use_cols=extra_regs, add_dow=True, add_month=True, add_trend=True)
    y_train = safe_num(train["revenue"]).values

    beta = ridge_ols(X_train, y_train, ridge_lambda)

    # Predict on full df (festival dummies are not in X by design)
    X_full, _ = design_matrix(df, use_cols=extra_regs, add_dow=True, add_month=True, add_trend=True)
    y_hat = X_full @ beta
    # Guard: baseline cannot be negative
    y_hat = np.maximum(y_hat, 0.0)

    df["baseline"] = y_hat
    df["delta"] = df["revenue"] - df["baseline"]

    diag = {
        "used_median": False,
        "n_train": int(train.shape[0]),
        "ridge_lambda": float(ridge_lambda),
        "features": names
    }
    return df, diag

def compute_uplift_for_events(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    pre: int, post: int
) -> pd.DataFrame:
    """
    Per event (festival instance on a specific date) compute uplift and cannibalization.
    """
    rows = []
    # Build quick lookup of group keys
    group_cols = [c for c in ["region","category"] if c in df.columns]

    # Iterate events in the group's calendar
    for _, ev in calendar.iterrows():
        center = ev["date"]
        win_mask = daterange_mask(df["date"], center, pre, post)
        pre_mask = daterange_mask(df["date"], center, pre, 0)
        post_mask = daterange_mask(df["date"], center, 0, post)

        sub = df[win_mask].copy()
        if sub.empty: 
            continue
        uplift_abs = float((sub["delta"]).clip(lower=0).sum())   # positive deltas
        cannib = float((sub["delta"]).clip(upper=0).sum())       # negative deltas (<=0)

        # Same-day lift pct vs baseline
        same_day = df[df["date"]==center]
        lift_pct = float((same_day["delta"].sum()) / (same_day["baseline"].sum() + 1e-9)) if not same_day.empty else np.nan

        # Split pre/post contributions (for diagnostics)
        pre_cann = float(df[pre_mask]["delta"].clip(upper=0).sum())
        post_cann = float(df[post_mask]["delta"].clip(upper=0).sum())

        row = {
            "festival": ev["festival"],
            "event_date": center,
            "uplift_abs": uplift_abs,
            "lift_pct_same_day": lift_pct,
            "cannibalization_abs": cannib,
            "cannibalization_pre": pre_cann,
            "cannibalization_post": post_cann,
            "net_uplift": uplift_abs + cannib,
            "weight": float(ev.get("weight", 1.0)),
            "type": ev.get("type", None)
        }
        for c in group_cols:
            row[c] = same_day[c].iloc[0] if (c in same_day.columns and not same_day.empty) else None
        rows.append(row)
    return pd.DataFrame(rows)


def simple_forecast(
    baseline_last: pd.DataFrame,
    hist_uplift: pd.DataFrame,
    future_events: pd.DataFrame,
    macro_scaler: Optional[pd.Series]=None,
    promo_intensity: Optional[pd.Series]=None
) -> pd.DataFrame:
    """
    Forecast = baseline (carry-forward mean of same DOW in month) + expected uplift
    Expected uplift for a festival = median historical net_uplift for same
    festival×region×category (fallbacks progressively).
    Scaled by latest macro_index / median(macro) and promo intensity ratio.
    """
    if future_events.empty:
        return pd.DataFrame(columns=["date","festival","region","category","forecast_revenue"])

    # Build uplift lookup with fallbacks
    keys = ["festival","region","category"]
    # primary: by all three
    G = hist_uplift.groupby(keys)["net_uplift"].median().reset_index()
    # fallbacks
    G_fest_reg = hist_uplift.groupby(["festival","region"])["net_uplift"].median().reset_index()
    G_fest_cat = hist_uplift.groupby(["festival","category"])["net_uplift"].median().reset_index()
    G_fest     = hist_uplift.groupby(["festival"])["net_uplift"].median().reset_index()

    # Baseline carry-forward: for each future date, take mean baseline for same DOW in the last 8 weeks
    base_idx = baseline_last[["date","baseline","region","category"]].copy()
    rows = []
    for _, ev in future_events.iterrows():
        d = ev["date"]; fest = ev["festival"]
        region = ev.get("region", None)
        category = ev.get("category", None)
        # baseline proxy
        dow = d.dayofweek
        recent = base_idx[base_idx["date"] <= base_idx["date"].max()].copy()
        recent = recent[recent["date"] >= (base_idx["date"].max() - pd.Timedelta(days=56))]
        if region is not None: recent = recent[(recent["region"]==region) | recent["region"].isna()]
        if category is not None: recent = recent[(recent["category"]==category) | recent["category"].isna()]
        recent = recent[recent["date"].dt.dayofweek == dow]
        base_cf = float(recent["baseline"].mean()) if not recent.empty else float(baseline_last["baseline"].tail(28).mean())

        # uplift estimate with fallbacks
        def lookup(df, cond):
            m = df
            for k,v in cond.items():
                m = m[m[k]==v]
            return float(m["net_uplift"].median()) if not m.empty else None

        est = lookup(G, {"festival":fest, "region":region, "category":category})
        if est is None:
            est = lookup(G_fest_reg, {"festival":fest, "region":region})
        if est is None:
            est = lookup(G_fest_cat, {"festival":fest, "category":category})
        if est is None:
            est = lookup(G_fest, {"festival":fest})
        if est is None:
            est = 0.0

        # scale by macro/promo
        scale = 1.0
        if macro_scaler is not None and len(macro_scaler)>0:
            recent_macro = float(macro_scaler.dropna().tail(30).mean())
            ref_macro = float(macro_scaler.dropna().median())
            if ref_macro and np.isfinite(ref_macro):
                scale *= (recent_macro / ref_macro)
        if promo_intensity is not None and len(promo_intensity)>0:
            recent_promo = float(promo_intensity.dropna().tail(30).mean())
            ref_promo = float(promo_intensity.dropna().median())
            if ref_promo and np.isfinite(ref_promo):
                scale *= (recent_promo / ref_promo)

        rows.append({
            "date": d, "festival": fest, "region": region, "category": category,
            "baseline_cf": base_cf, "uplift_expected": est*scale,
            "forecast_revenue": max(0.0, base_cf + est*scale)
        })
    return pd.DataFrame(rows)


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    sales: str
    calendar: str
    promos: Optional[str]
    macro: Optional[str]
    weather: Optional[str]
    pre: int
    post: int
    ridge: float
    min_days: int
    scope: str
    forecast_days: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Festival consumption uplift engine")
    ap.add_argument("--sales", required=True)
    ap.add_argument("--calendar", required=True)
    ap.add_argument("--promos", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--weather", default="")
    ap.add_argument("--pre", type=int, default=7)
    ap.add_argument("--post", type=int, default=7)
    ap.add_argument("--ridge", type=float, default=5.0)
    ap.add_argument("--min_days", type=int, default=180, help="Minimum observations per group")
    ap.add_argument("--scope", default="AUTO", choices=["AUTO","REGION","CATEGORY","BOTH"])
    ap.add_argument("--forecast_days", type=int, default=90)
    ap.add_argument("--outdir", default="out_festival")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    SALES = load_sales(args.sales)
    CAL   = load_calendar(args.calendar)
    PROMO = load_promos(args.promos) if args.promos else pd.DataFrame()
    MACRO = load_macro(args.macro) if args.macro else pd.DataFrame()
    WEAT  = load_weather(args.weather) if args.weather else pd.DataFrame()

    # Merge auxiliaries onto sales (left join by date and, if present, region/category)
    df = SALES.copy()
    # promos
    if not PROMO.empty:
        on_cols = ["date"] + [c for c in ["region","category"] if c in df.columns and c in PROMO.columns]
        df = df.merge(PROMO, on=on_cols, how="left")
    # macro
    if not MACRO.empty:
        df = df.merge(MACRO, on=["date"], how="left")
    # weather
    if not WEAT.empty:
        on_cols = ["date"] + ([ "region" ] if "region" in df.columns and "region" in WEAT.columns else [])
        df = df.merge(WEAT, on=on_cols, how="left")

    # mark festival day per row (by region/category match if provided)
    df["is_festival_day"] = 0
    # calendar matching rules: if calendar has region/category specified, enforce equality; else apply to all
    cal_rows = []
    for _, r in df.iterrows():
        d = r["date"]
        cond = (CAL["date"]==d)
        if "region" in CAL.columns and "region" in df.columns and isinstance(r["region"], str):
            cond = cond & ((CAL["region"]==r["region"]) | CAL["region"].isna())
        if "category" in CAL.columns and "category" in df.columns and isinstance(r["category"], str):
            cond = cond & ((CAL["category"]==r["category"]) | CAL["category"].isna())
        if cond.any():
            df.at[_, "is_festival_day"] = 1
            # keep possible multiple festivals same day (rare); collect for event table later
        # For efficiency, we won't expand here; we'll reuse CAL directly for events
        cal_rows.append(None)

    # Decide modeling groups
    groups = infer_group_scope(df, args.scope)
    group_cols = []
    if any(r is not None for r,_ in groups): group_cols.append("region")
    if any(c is not None for _,c in groups): group_cols.append("category")

    baseline_frames = []
    event_frames = []

    # Extra numeric regressors available
    numeric_cols = [c for c in df.columns if c not in ["date","region","category","revenue","is_festival_day"] and pd.api.types.is_numeric_dtype(df[c])]
    # Drop promo_cost from regressors (kept for ROI proxies)
    extra_regs = [c for c in numeric_cols if c != "promo_cost"]

    for region, category in groups:
        sub = df.copy()
        if region is not None and "region" in df.columns:
            sub = sub[sub["region"]==region]
        if category is not None and "category" in df.columns:
            sub = sub[sub["category"]==category]
        sub = sub.dropna(subset=["date","revenue"]).sort_values("date")
        if sub.shape[0] < max(30, args.min_days):
            continue

        # Event calendar restricted to this group (if calendar has region/category)
        cal_g = CAL.copy()
        if "region" in CAL.columns and region is not None:
            cal_g = cal_g[(cal_g["region"]==region) | (cal_g["region"].isna())]
        if "category" in CAL.columns and category is not None:
            cal_g = cal_g[(cal_g["category"]==category) | (cal_g["category"].isna())]
        cal_g = cal_g.sort_values("date")
        if cal_g.empty:
            continue

        # Flag in-sub festivals (for exclusion only)
        sub["is_festival_day"] = sub["date"].isin(cal_g["date"]).astype(int)

        # Fit baseline
        base_df, diag = fit_baseline_for_group(
            G=sub,
            pre=int(args.pre), post=int(args.post),
            ridge_lambda=float(args.ridge),
            extra_regs=extra_regs
        )
        base_df["region"] = region if region is not None else (base_df["region"] if "region" in base_df.columns else None)
        base_df["category"] = category if category is not None else (base_df["category"] if "category" in base_df.columns else None)
        baseline_frames.append(base_df[["date","region","category","revenue","baseline","delta"]])

        # Per-event uplift
        ev_u = compute_uplift_for_events(base_df, cal_g, pre=int(args.pre), post=int(args.post))
        if not ev_u.empty:
            event_frames.append(ev_u)

    if not baseline_frames:
        raise ValueError("No groups met the minimum data requirement; relax --min_days or check data.")
    BASELINE = pd.concat(baseline_frames, ignore_index=True).sort_values(["region","category","date"])
    BASELINE.to_csv(outdir / "baseline_vs_actual.csv", index=False)

    UPLIFT_EVENTS = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(
        columns=["festival","event_date","uplift_abs","lift_pct_same_day","cannibalization_abs","net_uplift","region","category","weight","type"])
    if not UPLIFT_EVENTS.empty:
        UPLIFT_EVENTS.to_csv(outdir / "uplift_by_event.csv", index=False)

    # Aggregations
    if not UPLIFT_EVENTS.empty:
        by_fest = (UPLIFT_EVENTS.groupby("festival")
                   .agg(uplift_abs=("uplift_abs","sum"),
                        cannibalization_abs=("cannibalization_abs","sum"),
                        net_uplift=("net_uplift","sum"),
                        events=("festival","count"))
                   .reset_index()
                   .sort_values("net_uplift", ascending=False))
        by_fest.to_csv(outdir / "uplift_by_festival.csv", index=False)

        group_keys = [c for c in ["region","category"] if c in UPLIFT_EVENTS.columns]
        if group_keys:
            by_group = (UPLIFT_EVENTS.groupby(group_keys)
                        .agg(uplift_abs=("uplift_abs","sum"),
                             cannibalization_abs=("cannibalization_abs","sum"),
                             net_uplift=("net_uplift","sum"),
                             events=("festival","count"))
                        .reset_index()
                        .sort_values("net_uplift", ascending=False))
            by_group.to_csv(outdir / "uplift_by_group.csv", index=False)
        else:
            by_group = pd.DataFrame()

    else:
        by_fest, by_group = pd.DataFrame(), pd.DataFrame()

    # Forecast within N days after last sales date
    forecast_df = pd.DataFrame()
    last_date = BASELINE["date"].max()
    future_events = CAL[CAL["date"].between(last_date + pd.Timedelta(days=1),
                                            last_date + pd.Timedelta(days=int(args.forecast_days)))].copy()
    if not future_events.empty and not UPLIFT_EVENTS.empty:
        # Prepare macro & promo scalers
        macro_s = None
        promo_s = None
        if not MACRO.empty:
            macro_s = MACRO.set_index("date").sort_index()["macro_index"]
        if not PROMO.empty and "promo_intensity" in PROMO.columns:
            promo_s = PROMO.set_index("date").sort_index()["promo_intensity"]
        forecast_df = simple_forecast(BASELINE, UPLIFT_EVENTS, future_events, macro_s, promo_s)
        if not forecast_df.empty:
            forecast_df.to_csv(outdir / "forecast.csv", index=False)

    # Summary
    head = {
        "groups": len(groups),
        "n_events": int(UPLIFT_EVENTS.shape[0]) if not UPLIFT_EVENTS.empty else 0,
        "top_festival": (by_fest.iloc[0]["festival"] if not by_fest.empty else None),
        "top_festival_net_uplift": (float(by_fest.iloc[0]["net_uplift"]) if not by_fest.empty else None),
        "total_net_uplift": (float(by_fest["net_uplift"].sum()) if not by_fest.empty else 0.0),
        "total_cannibalization": (float(by_fest["cannibalization_abs"].sum()) if not by_fest.empty else 0.0),
        "baseline_last_date": str(last_date.date())
    }
    (outdir / "summary.json").write_text(json.dumps(head, indent=2))

    # Config dump
    cfg = asdict(Config(
        sales=args.sales, calendar=args.calendar, promos=(args.promos or None),
        macro=(args.macro or None), weather=(args.weather or None),
        pre=int(args.pre), post=int(args.post), ridge=float(args.ridge),
        min_days=int(args.min_days), scope=args.scope,
        forecast_days=int(args.forecast_days), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Festival Consumption ==")
    print(f"Groups modeled: {head['groups']} | Events analyzed: {head['n_events']}")
    if head["top_festival"]:
        print(f"Top festival by net uplift: {head['top_festival']} ({head['top_festival_net_uplift']:.0f})")
    if not forecast_df.empty:
        print(f"Forecasted future events: {forecast_df.shape[0]} (through {args.forecast_days} days)")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
