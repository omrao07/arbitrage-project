#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
german_wage_negotiations.py — Tarifrunde analytics for Germany

What this does
--------------
Given a table of collective bargaining agreements (CBAs) and optional macro/market inputs,
this script builds nominal/real wage paths by sector/union, converts one-off payments into
annualized equivalents, computes effective settlement rates, aggregates by coverage, spots
upcoming expiries, and (optionally) runs a simple event study around settlement dates.

Core outputs
------------
- agreements_enriched.csv   Per-agreement metrics (effective annualized rate, gap vs demand, expiry flags)
- wage_paths.csv            Monthly wage indices by sector/union (nominal, incl. lump sum, real if CPI provided)
- economy_aggregate.csv     Weighted (by coverage) aggregate wage indices & YoY growth
- ulc.csv                   Unit labour cost proxy (wage/prod) per sector & aggregate
- calendar.csv              Expiry/renegotiation calendar (with horizon flag)
- strikes_agg.csv           Strike days & participants by sector/date (if provided)
- event_study.csv           Asset returns around settlement dates (if markets provided)
- summary.json              Headline KPIs (latest YoY nominal/real, weighted settlement rate, expiries)
- config.json               Run configuration (for reproducibility)

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--agreements agreements.csv   REQUIRED
  Suggested columns (use what you have; script is robust to missing):
    union, sector, region, agreement_name, settlement_date, start_date, end_date, duration_months,
    base_monthly_eur, coverage_employees, coverage_weight, demand_pct_first12,
    steps (free text, e.g., "2024-01:+5.5%; 2025-01:+3.0%"),
    step1_date, step1_pct, step2_date, step2_pct, ... (up to 6 supported),
    lumpsum_total_eur, lumpsum_months, lumpsum_start_date, iap_taxfree(0/1), working_time_change, notes

--cpi cpi.csv                 OPTIONAL (monthly CPI index; used for real wages)
  Columns: date, cpi_index  (index level, any base; non-seasonally adjusted is fine)

--productivity productivity.csv OPTIONAL (monthly or quarterly productivity index)
  Columns: date, productivity_index

--strikes strikes.csv         OPTIONAL
  Columns: date, sector, union(optional), strike_days, participants

--markets markets.csv         OPTIONAL (for event study)
  Columns (daily):
    date, asset, close (or return)
  If 'close' provided, log-returns are computed.

Key options
-----------
--start 2018-01
--end   2026-12
--horizon_months 6           Look-ahead for upcoming expiries in calendar.csv
--employer_contrib 0.20      On-cost factor to approximate employer labour cost (for ULC context)
--annualize_window 12        Window (months) for effective annualized settlement measures
--event_win 3                Event-study window (days) around settlement_date
--outdir out_tarifrunden

Notes & caveats
---------------
- One-off (tax-free) inflation compensation payments (IAP) are apportioned evenly across
  'lumpsum_months' from 'lumpsum_start_date'. If base_monthly_eur is missing, the “incl. lump sum”
  path is left as NaN and only the indexed base is produced.
- Effective settlement % in first 12 months combines additive step increases (compound)
  and the lump-sum as a % of annual base (if base_monthly_eur known).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------- Helpers ----------------------------

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None

def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def yesno(x) -> int:
    try:
        return 1 if float(x) > 0 else 0
    except Exception:
        s = str(x).strip().lower()
        return 1 if s in {"y", "yes", "true", "t", "1"} else 0

def pct_to_float(x) -> float:
    """Accept '5.5', '5,5', '5.5%', return 0.055"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).replace("%", "").replace(",", ".").strip()
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan

def parse_steps_text(s: str) -> List[Tuple[pd.Timestamp, float]]:
    """
    Parse patterns like:
      "2024-01:+5.5%; 2025-01:+3.0%" or "5.5% from 2024-01, 3% 2025-01"
    Returns list of (timestamp_month_start, fraction).
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    txt = str(s).lower().replace("from", "").replace("ab", "")
    import re
    # capture (date)(% or +% anywhere near)
    pat = re.compile(r"(\d{4}-\d{1,2})(?:-\d{1,2})?\s*[: ]*\+?(\d{1,2}(?:[\.,]\d+)?)\s*%")
    out = []
    for date_s, pct_s in pat.findall(txt):
        try:
            d = pd.to_datetime(date_s).to_period("M").to_timestamp()
            p = float(pct_s.replace(",", ".")) / 100.0
            out.append((d, p))
        except Exception:
            continue
    # also support "% date"
    if not out:
        pat2 = re.compile(r"\+?(\d{1,2}(?:[\.,]\d+)?)\s*%\s*(\d{4}-\d{1,2})")
        for pct_s, date_s in pat2.findall(txt):
            try:
                d = pd.to_datetime(date_s).to_period("M").to_timestamp()
                p = float(pct_s.replace(",", ".")) / 100.0
                out.append((d, p))
            except Exception:
                continue
    # de-duplicate and sort
    out = list({(d, round(p, 6)) for d, p in out})
    return sorted(out, key=lambda x: x[0])

def collect_structured_steps(row: pd.Series, max_n: int = 6) -> List[Tuple[pd.Timestamp, float]]:
    steps: List[Tuple[pd.Timestamp, float]] = []
    for i in range(1, max_n + 1):
        dcol = f"step{i}_date"; pcol = f"step{i}_pct"
        if dcol in row.index or pcol in row.index:
            d = pd.to_datetime(row.get(dcol), errors="coerce")
            p = pct_to_float(row.get(pcol))
            if pd.notna(d) and pd.notna(p):
                steps.append((d.to_period("M").to_timestamp(), float(p)))
    return sorted(steps, key=lambda x: x[0])

def build_steps(row: pd.Series) -> List[Tuple[pd.Timestamp, float]]:
    steps = collect_structured_steps(row)
    if (not steps) and ("steps" in row.index):
        steps = parse_steps_text(row.get("steps"))
    # If still empty but duration and demand exist, leave empty; we won't infer
    return steps

def month_range(start_m: pd.Timestamp, end_m: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.period_range(start_m.to_period("M"), end_m.to_period("M"), freq="M").to_timestamp()

def compound_from_steps(steps: List[Tuple[pd.Timestamp, float]], start_m: pd.Timestamp, window_m: int) -> float:
    """
    Compound percentage gain over [start_m, start_m + window_m - 1] inclusive,
    counting steps whose effective date falls within window.
    """
    if not steps or window_m <= 0:
        return 0.0
    end_m = (start_m + pd.offsets.MonthBegin(window_m)) - pd.offsets.MonthBegin(1)
    cum = 1.0
    for d, p in steps:
        if start_m <= d <= end_m:
            cum *= (1.0 + float(p))
    return cum - 1.0

def equivalent_pct_from_lumpsum(base_monthly: Optional[float], lumpsum_total: Optional[float], months_in_window: int) -> float:
    if not base_monthly or not np.isfinite(base_monthly) or base_monthly <= 0:
        return np.nan
    if not lumpsum_total or not np.isfinite(lumpsum_total) or lumpsum_total <= 0:
        return 0.0
    # Lump-sum equivalent as % of annual base during the window
    denom = base_monthly * months_in_window
    return float(lumpsum_total / denom)

def safe_weight(w: float) -> float:
    if w is None or not np.isfinite(w) or w < 0:
        return 0.0
    return float(w)

# ---------------------------- Loaders ----------------------------

def load_agreements(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df, "union") or "union"): "union",
        (ncol(df, "sector") or "sector"): "sector",
        (ncol(df, "region") or "region"): "region",
        (ncol(df, "agreement_name") or ncol(df, "agreement") or "agreement_name"): "agreement_name",
        (ncol(df, "settlement_date") or "settlement_date"): "settlement_date",
        (ncol(df, "start_date") or "start_date"): "start_date",
        (ncol(df, "end_date") or "end_date"): "end_date",
        (ncol(df, "duration_months") or "duration_months"): "duration_months",
        (ncol(df, "base_monthly_eur") or ncol(df, "base_wage_monthly_eur") or "base_monthly_eur"): "base_monthly_eur",
        (ncol(df, "coverage_employees") or ncol(df, "coverage") or "coverage_employees"): "coverage_employees",
        (ncol(df, "coverage_weight") or "coverage_weight"): "coverage_weight",
        (ncol(df, "demand_pct_first12") or ncol(df, "demand_pct") or "demand_pct_first12"): "demand_pct_first12",
        (ncol(df, "steps") or "steps"): "steps",
        (ncol(df, "lumpsum_total_eur") or ncol(df, "lump_sum_total_eur") or "lumpsum_total_eur"): "lumpsum_total_eur",
        (ncol(df, "lumpsum_months") or "lumpsum_months"): "lumpsum_months",
        (ncol(df, "lumpsum_start_date") or "lumpsum_start_date"): "lumpsum_start_date",
        (ncol(df, "iap_taxfree") or "iap_taxfree"): "iap_taxfree",
        (ncol(df, "working_time_change") or "working_time_change"): "working_time_change",
        (ncol(df, "notes") or "notes"): "notes",
    }
    df = df.rename(columns=ren)
    df["union"] = df.get("union", "").astype(str)
    df["sector"] = df.get("sector", "").astype(str)
    df["region"] = df.get("region", "").astype(str)
    df["agreement_name"] = df.get("agreement_name", "").astype(str)
    df["settlement_date"] = pd.to_datetime(df.get("settlement_date", pd.NaT), errors="coerce")
    df["start_date"] = pd.to_datetime(df.get("start_date", pd.NaT), errors="coerce")
    df["end_date"] = pd.to_datetime(df.get("end_date", pd.NaT), errors="coerce")
    for c in ["duration_months","lumpsum_months"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["base_monthly_eur","coverage_employees","coverage_weight","lumpsum_total_eur","demand_pct_first12"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "iap_taxfree" in df.columns:
        df["iap_taxfree"] = df["iap_taxfree"].apply(yesno)
    # carry steps structured columns as-is
    return df

def load_cpi(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={(ncol(df,"date") or df.columns[0]):"date", (ncol(df,"cpi_index") or "cpi_index"):"cpi_index"})
    df["date"] = to_month(df["date"])
    df["cpi_index"] = num_series(df["cpi_index"])
    return df.dropna(subset=["date","cpi_index"])

def load_prod(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={(ncol(df,"date") or df.columns[0]):"date", (ncol(df,"productivity_index") or "productivity_index"):"productivity_index"})
    df["date"] = to_month(df["date"])
    df["productivity_index"] = num_series(df["productivity_index"])
    return df.dropna(subset=["date","productivity_index"])

def load_strikes(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df, "date") or df.columns[0]): "date",
        (ncol(df, "sector") or "sector"): "sector",
        (ncol(df, "union") or "union"): "union",
        (ncol(df, "strike_days") or ncol(df,"days") or "strike_days"): "strike_days",
        (ncol(df, "participants") or "participants"): "participants",
    }
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    for c in ["strike_days","participants"]:
        if c in df.columns:
            df[c] = num_series(df[c])
    return df

def load_markets(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"asset") or "asset"): "asset",
        (ncol(df,"close") or ncol(df,"price") or "close"): "close",
        (ncol(df,"return") or ncol(df,"ret") or "return"): "return",
    }
    df = df.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "return" not in df.columns or df["return"].isna().all():
        df = df.sort_values(["asset","date"])
        df["return"] = df.groupby("asset")["close"].apply(lambda s: np.log(s) - np.log(s.shift(1)))
    return df.dropna(subset=["date","asset","return"])

# ---------------------------- Core analytics ----------------------------

def enrich_agreements(agreements: pd.DataFrame, annualize_window: int) -> pd.DataFrame:
    rows = []
    for _, r in agreements.iterrows():
        steps = build_steps(r)
        start_m = r.get("start_date")
        if pd.isna(start_m):
            # if missing, try settlement_date
            start_m = r.get("settlement_date")
        if pd.isna(start_m):
            continue
        start_m = pd.to_datetime(start_m).to_period("M").to_timestamp()
        comp_first12 = compound_from_steps(steps, start_m, annualize_window)
        demand12 = pct_to_float(r.get("demand_pct_first12"))
        base_m = r.get("base_monthly_eur", np.nan)
        # Lump-sum window overlap: if defined months & start, use min(months, window)
        lump_total = r.get("lumpsum_total_eur", np.nan)
        lump_months = int(r.get("lumpsum_months", 0) or 0)
        lump_equiv = np.nan
        if lump_months > 0 and pd.notna(lump_total) and lump_total > 0:
            months_in_window = min(lump_months, annualize_window)
            lump_equiv = equivalent_pct_from_lumpsum(base_m, lump_total, months_in_window)
        eff12 = comp_first12 + (lump_equiv if np.isfinite(lump_equiv) else 0.0)
        # coverage weight fallback
        cov_emp = r.get("coverage_employees", np.nan)
        cov_w = r.get("coverage_weight", np.nan)
        if (not np.isfinite(cov_w)) and np.isfinite(cov_emp):
            cov_w = float(cov_emp)  # use employees as weight for aggregation
        rows.append({
            "union": r.get("union",""),
            "sector": r.get("sector",""),
            "region": r.get("region",""),
            "agreement_name": r.get("agreement_name",""),
            "settlement_date": pd.to_datetime(r.get("settlement_date")),
            "start_date": start_m,
            "end_date": pd.to_datetime(r.get("end_date")),
            "duration_months": r.get("duration_months"),
            "steps_count": len(steps),
            "sum_steps_first12_pct": float(comp_first12*100.0) if np.isfinite(comp_first12) else np.nan,
            "lumpsum_equiv_first12_pct": float(lump_equiv*100.0) if np.isfinite(lump_equiv) else np.nan,
            "effective_first12_pct": float(eff12*100.0) if np.isfinite(eff12) else float(comp_first12*100.0),
            "demand_first12_pct": float(demand12*100.0) if np.isfinite(demand12) else np.nan,
            "gap_vs_demand_pp": float((eff12 - demand12)*100.0) if (np.isfinite(eff12) and np.isfinite(demand12)) else np.nan,
            "base_monthly_eur": float(base_m) if np.isfinite(base_m) else np.nan,
            "coverage_weight": float(cov_w) if np.isfinite(cov_w) else np.nan,
            "has_lumpsum": int((lump_months > 0) and (pd.notna(lump_total) and lump_total > 0)),
        })
    return pd.DataFrame(rows)

def expand_wage_path(r: pd.Series, steps: List[Tuple[pd.Timestamp, float]],
                     start: pd.Timestamp, end: pd.Timestamp,
                     base_monthly_eur: Optional[float],
                     lumpsum_total: Optional[float],
                     lumpsum_months: int,
                     lumpsum_start: Optional[pd.Timestamp]) -> pd.DataFrame:
    months = month_range(start, end)
    idx = 100.0
    path = []
    # Convert lumpsum to monthly flow over schedule
    lumps_per_m = 0.0
    lump_sched_months = set()
    if base_monthly_eur and np.isfinite(base_monthly_eur) and lumpsum_total and np.isfinite(lumpsum_total) and lumpsum_total > 0 and lumpsum_months > 0:
        lumps_per_m = float(lumpsum_total) / int(lumpsum_months)
        if pd.notna(lumpsum_start):
            ls = pd.to_datetime(lumpsum_start).to_period("M").to_timestamp()
        else:
            # default to agreement start
            ls = start
        lump_sched_months = set(month_range(ls, ls + pd.offsets.MonthBegin(lumpsum_months) - pd.offsets.MonthBegin(1)))
    # Pre-build a dictionary of step factors per month
    step_map = {d: (1.0 + p) for d, p in steps}
    for m in months:
        if m in step_map:
            idx *= step_map[m]
        eff_idx = idx
        # Add lump-sum equivalence into an "effective pay index" if base known
        if base_monthly_eur and lumps_per_m > 0 and m in lump_sched_months:
            # translate monthly lump sum into index points relative to base
            eff_idx = idx + (lumps_per_m / float(base_monthly_eur)) * 100.0
        path.append({"date": m, "nominal_index": idx, "eff_index": eff_idx})
    out = pd.DataFrame(path)
    out["union"] = r.get("union","")
    out["sector"] = r.get("sector","")
    out["region"] = r.get("region","")
    out["agreement_name"] = r.get("agreement_name","")
    return out

def build_all_wage_paths(agreements: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for _, r in agreements.iterrows():
        steps = build_steps(r)
        s = r.get("start_date")
        e = r.get("end_date")
        # Default window: intersect requested window with agreement window (if provided)
        s_m = pd.to_datetime(s).to_period("M").to_timestamp() if pd.notna(s) else start
        e_m = pd.to_datetime(e).to_period("M").to_timestamp() if pd.notna(e) else end
        s_m = max(s_m, start); e_m = min(e_m, end)
        if s_m > e_m:
            continue
        base_m_eur = r.get("base_monthly_eur", np.nan)
        lump_total = r.get("lumpsum_total_eur", np.nan)
        lump_months = int(r.get("lumpsum_months", 0) or 0)
        lump_start = r.get("lumpsum_start_date", np.nan)
        rows.append(expand_wage_path(r, steps, s_m, e_m, base_m_eur, lump_total, lump_months, lump_start))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def join_cpi_and_real(paths: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return paths
    if cpi.empty:
        paths["real_index"] = np.nan
        paths["real_eff_index"] = np.nan
        return paths
    p = paths.merge(cpi.rename(columns={"date":"date_cpi"}), left_on="date", right_on="date_cpi", how="left")
    p.drop(columns=["date_cpi"], inplace=True)
    # Normalize CPI to 100 at first available
    if "cpi_index" in p.columns:
        cpi0 = p["cpi_index"].dropna().iloc[0] if p["cpi_index"].dropna().shape[0] else np.nan
        if cpi0 and np.isfinite(cpi0) and cpi0 != 0:
            p["cpi_norm"] = p["cpi_index"] / cpi0 * 100.0
            p["real_index"] = p["nominal_index"] / (p["cpi_norm"] / 100.0)
            p["real_eff_index"] = p["eff_index"] / (p["cpi_norm"] / 100.0)
        else:
            p["real_index"] = np.nan; p["real_eff_index"] = np.nan
    else:
        p["real_index"] = np.nan; p["real_eff_index"] = np.nan
    return p

def aggregate_coverage(paths: pd.DataFrame, agreements: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    # Build weight per (union, sector, region, agreement_name)
    key_cols = ["union","sector","region","agreement_name"]
    w = agreements.copy()
    w["coverage_weight"] = w.get("coverage_weight", np.nan)
    w["coverage_employees"] = w.get("coverage_employees", np.nan)
    if w["coverage_weight"].isna().all() and not w["coverage_employees"].isna().all():
        # use employees as weights
        w["coverage_weight"] = w["coverage_employees"]
    w["coverage_weight"] = w["coverage_weight"].fillna(0.0)
    # Normalize weights within sector if you prefer sector aggregates; for economy aggregate just sum
    p = paths.merge(w[key_cols + ["coverage_weight"]], on=key_cols, how="left")
    p["coverage_weight"] = p["coverage_weight"].fillna(0.0)
    # Economy aggregate
    g = (p.groupby("date", as_index=False)
           .apply(lambda df: pd.Series({
               "nominal_index_w": np.average(df["nominal_index"], weights=df["coverage_weight"]) if df["coverage_weight"].sum()>0 else df["nominal_index"].mean(),
               "eff_index_w": np.average(df["eff_index"], weights=df["coverage_weight"]) if df["coverage_weight"].sum()>0 else df["eff_index"].mean(),
               "real_index_w": np.average(df["real_index"].dropna(), weights=df.loc[df["real_index"].notna(),"coverage_weight"]) if (("real_index" in df) and df["real_index"].notna().sum()>0 and df["coverage_weight"].sum()>0) else (df["real_index"].mean() if "real_index" in df else np.nan),
               "real_eff_index_w": np.average(df["real_eff_index"].dropna(), weights=df.loc[df["real_eff_index"].notna(),"coverage_weight"]) if (("real_eff_index" in df) and df["real_eff_index"].notna().sum()>0 and df["coverage_weight"].sum()>0) else (df["real_eff_index"].mean() if "real_eff_index" in df else np.nan),
               "coverage_sum": df["coverage_weight"].sum()
           })))
    g["nominal_yoy"] = g["nominal_index_w"].pct_change(12)
    g["real_yoy"] = g["real_index_w"].pct_change(12) if "real_index_w" in g.columns else np.nan
    return g

def compute_ulc(paths: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    if paths.empty or prod.empty:
        return pd.DataFrame()
    # Sector-level ULC proxy
    keep_cols = ["date","union","sector","region","nominal_index"]
    if "eff_index" in paths.columns:
        keep_cols += ["eff_index"]
    px = paths[keep_cols].copy()
    pr = prod.rename(columns={"date":"date_prod"})
    out = px.merge(pr, left_on="date", right_on="date_prod", how="left").drop(columns=["date_prod"])
    out["ulc_index"] = out["nominal_index"] / (out["productivity_index"] / out["productivity_index"].dropna().iloc[0]) * 100.0
    if "eff_index" in out.columns:
        out["ulc_eff_index"] = out["eff_index"] / (out["productivity_index"] / out["productivity_index"].dropna().iloc[0]) * 100.0
    out["ulc_yoy"] = out["ulc_index"].pct_change(12)
    return out

def build_calendar(enriched: pd.DataFrame, horizon_months: int, ref_month: pd.Timestamp) -> pd.DataFrame:
    cal = enriched[["union","sector","region","agreement_name","start_date","end_date","duration_months","effective_first12_pct"]].copy()
    cal["expires_soon"] = 0
    if "end_date" in cal.columns:
        cal["end_m"] = pd.to_datetime(cal["end_date"]).dt.to_period("M").dt.to_timestamp()
        future_cut = ref_month + pd.offsets.MonthBegin(horizon_months)
        cal["expires_soon"] = ((cal["end_m"] >= ref_month) & (cal["end_m"] <= future_cut)).astype(int)
    return cal.drop(columns=["end_m"], errors="ignore")

def aggregate_strikes(strikes: pd.DataFrame) -> pd.DataFrame:
    if strikes.empty:
        return pd.DataFrame()
    g = (strikes.groupby(["date","sector"], as_index=False)
         .agg(strike_days=("strike_days","sum"), participants=("participants","sum")))
    return g.sort_values(["date","sector"])

def event_study(markets: pd.DataFrame, enriched: pd.DataFrame, win: int) -> pd.DataFrame:
    if markets.empty or enriched.empty:
        return pd.DataFrame()
    # Use settlement_date as event day; align to trading days (by date match)
    ev = enriched.dropna(subset=["settlement_date"]).copy()
    ev["d"] = pd.to_datetime(ev["settlement_date"]).dt.normalize()
    rows = []
    for asset, g in markets.groupby("asset"):
        g = g.sort_values("date").reset_index(drop=True)
        g["d0"] = g["date"].dt.normalize()
        idx_map = {d: i for i, d in enumerate(g["d0"])}
        for d in ev["d"].unique():
            if d not in idx_map:
                continue
            i0 = idx_map[d]
            lo = max(0, i0 - win); hi = min(len(g)-1, i0 + win)
            sub = g.iloc[lo:hi+1].copy()
            sub["t"] = range(lo - i0, hi - i0 + 1)
            sub["asset"] = asset
            sub["event_date"] = d
            rows.append(sub[["asset","event_date","t","return"]])
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    avg = (df.groupby(["asset","t"], as_index=False)
             .agg(mean_ret=("return","mean"), med_ret=("return","median"), n=("return","count")))
    return avg.sort_values(["asset","t"])

# ---------------------------- CLI / Orchestration ----------------------------

@dataclass
class Config:
    agreements: str
    cpi: Optional[str]
    productivity: Optional[str]
    strikes: Optional[str]
    markets: Optional[str]
    start: str
    end: str
    horizon_months: int
    employer_contrib: float
    annualize_window: int
    event_win: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="German wage negotiations (Tarifrunden) analytics")
    ap.add_argument("--agreements", required=True)
    ap.add_argument("--cpi", default="")
    ap.add_argument("--productivity", default="")
    ap.add_argument("--strikes", default="")
    ap.add_argument("--markets", default="")
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2026-12")
    ap.add_argument("--horizon_months", type=int, default=6)
    ap.add_argument("--employer_contrib", type=float, default=0.20)
    ap.add_argument("--annualize_window", type=int, default=12)
    ap.add_argument("--event_win", type=int, default=3)
    ap.add_argument("--outdir", default="out_tarifrunden")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    agreements = load_agreements(args.agreements)
    if agreements.empty:
        raise SystemExit("No agreements loaded. Provide --agreements CSV.")

    cpi = load_cpi(args.cpi) if args.cpi else pd.DataFrame()
    prod = load_prod(args.productivity) if args.productivity else pd.DataFrame()
    strikes = load_strikes(args.strikes) if args.strikes else pd.DataFrame()
    markets = load_markets(args.markets) if args.markets else pd.DataFrame()

    # Enrich agreements with settlement metrics
    enriched = enrich_agreements(agreements, args.annualize_window)
    enriched.to_csv(outdir / "agreements_enriched.csv", index=False)

    # Wage paths
    start = pd.to_datetime(args.start).to_period("M").to_timestamp()
    end = pd.to_datetime(args.end).to_period("M").to_timestamp()
    paths = build_all_wage_paths(agreements, start, end)
    if paths.empty:
        # still write empty files for consistency
        pd.DataFrame().to_csv(outdir / "wage_paths.csv", index=False)
    else:
        # attach real indices if CPI available
        paths = join_cpi_and_real(paths, cpi)
        # compute YoY per sector
        paths = paths.sort_values(["sector","union","date"])
        paths["nominal_yoy"] = paths.groupby(["sector","union"])["nominal_index"].pct_change(12)
        if "real_index" in paths.columns:
            paths["real_yoy"] = paths.groupby(["sector","union"])["real_index"].pct_change(12)
        paths.to_csv(outdir / "wage_paths.csv", index=False)

    # Aggregates
    econ = aggregate_coverage(paths, agreements) if not paths.empty else pd.DataFrame()
    if not econ.empty:
        econ.to_csv(outdir / "economy_aggregate.csv", index=False)

    # ULC proxy
    ulc = compute_ulc(paths, prod) if (not paths.empty and not prod.empty) else pd.DataFrame()
    if not ulc.empty:
        ulc.to_csv(outdir / "ulc.csv", index=False)

    # Calendar / expiries
    ref_m = end  # could use current month; here we use end of window to flag within horizon
    cal = build_calendar(enriched, args.horizon_months, ref_m)
    if not cal.empty:
        cal.to_csv(outdir / "calendar.csv", index=False)

    # Strikes
    strikes_agg = aggregate_strikes(strikes)
    if not strikes_agg.empty:
        strikes_agg.to_csv(outdir / "strikes_agg.csv", index=False)

    # Event study
    evt = event_study(markets, enriched, args.event_win) if not markets.empty else pd.DataFrame()
    if not evt.empty:
        evt.to_csv(outdir / "event_study.csv", index=False)

    # KPIs / summary
    latest = econ["date"].max().date() if not econ.empty else (paths["date"].max().date() if not paths.empty else None)
    top_sectors = (agreements.groupby("sector")["coverage_employees"].sum().sort_values(ascending=False).head(5).to_dict()
                   if ("coverage_employees" in agreements.columns and agreements["coverage_employees"].notna().any()) else {})
    kpi = {
        "window": {"start": args.start, "end": args.end},
        "agreements": int(len(agreements)),
        "with_lumpsum": int(enriched["has_lumpsum"].sum()) if not enriched.empty else 0,
        "weighted_effective_first12_pct": float(
            np.average(enriched["effective_first12_pct"].dropna(),
                       weights=enriched.loc[enriched["effective_first12_pct"].notna(), "coverage_weight"].fillna(1.0))
        ) if (not enriched.empty and enriched["effective_first12_pct"].notna().any()) else None,
        "econ_nominal_yoy_latest": float(econ["nominal_yoy"].dropna().iloc[-1]) if (not econ.empty and econ["nominal_yoy"].notna().any()) else None,
        "econ_real_yoy_latest": float(econ["real_yoy"].dropna().iloc[-1]) if ("real_yoy" in econ.columns and econ["real_yoy"].notna().any()) else None,
        "upcoming_expiries": int(cal["expires_soon"].sum()) if not cal.empty else 0,
        "latest_month": str(latest) if latest else None,
        "top_sectors_by_coverage_employees": top_sectors
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))

    # Config dump
    cfg = asdict(Config(
        agreements=args.agreements, cpi=args.cpi or None, productivity=args.productivity or None,
        strikes=args.strikes or None, markets=args.markets or None, start=args.start, end=args.end,
        horizon_months=args.horizon_months, employer_contrib=args.employer_contrib,
        annualize_window=args.annualize_window, event_win=args.event_win, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== German Wage Negotiations (Tarifrunden) ==")
    print(f"Agreements: {len(agreements)} | Weighted effective first-12m: {kpi['weighted_effective_first12_pct'] if kpi['weighted_effective_first12_pct'] is not None else 'n/a'} %")
    if kpi["econ_nominal_yoy_latest"] is not None:
        print(f"Econ nominal YoY (latest): {kpi['econ_nominal_yoy_latest']*100:.2f}% | Real YoY: {(kpi['econ_real_yoy_latest']*100 if kpi['econ_real_yoy_latest'] is not None else float('nan')):.2f}%")
    if not cal.empty:
        soon = cal[cal["expires_soon"]==1][["union","sector","end_date"]].sort_values("end_date").head(8)
        if not soon.empty:
            print("Expiries within horizon (sample):")
            for _, rr in soon.iterrows():
                print(f"  {rr['end_date']:%Y-%m}  {rr['union']} — {rr['sector']}")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
