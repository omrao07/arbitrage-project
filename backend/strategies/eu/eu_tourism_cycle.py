#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eu_tourism_cycle.py — Seasonality, cycle, and recovery analytics for EU tourism

What it does
------------
Given monthly tourism data (arrivals / nights / occupancy) and optional macro/air capacity inputs,
this script builds country-level and EU-aggregate seasonality/cycle diagnostics:

Core outputs
- Seasonality: monthly seasonal indices, peak/shoulder/off-peak classification, seasonality strength
- Cycle: YoY, 12m moving averages, turning points, simple phase tagging (expansion / slowdown / contraction / recovery)
- Baseline recovery: % vs baseline-year (e.g., 2019) and time-to-full-recovery
- Momentum: 3m/6m annualized momentum for arrivals & nights
- EU rollups (member list provided) and country snapshots
- Optional overlays:
  * Air capacity (monthly seats) → demand-capacity gap
  * CPI (price level) and FX (competitiveness) → simple elasticities via OLS
  * Events (shocks) table joined for context (e.g., COVID, conflicts, visa changes)

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--tourism tourism.csv    REQUIRED
  Columns (suggested; case-insensitive, extras ignored):
    date (YYYY-MM), country (ISO-2/3 or name),
    arrivals, nights, occupancy (0..1 or 0..100), revenue_eur (optional)

--cpi cpi.csv            OPTIONAL (national CPI indices, 2015=100 or similar)
  Columns: date, country, cpi

--fx fx.csv              OPTIONAL (effective exchange rate or bilateral EUR rate; higher → stronger domestic currency)
  Columns: date, country, fx

--air air.csv            OPTIONAL (air capacity; scheduled seats or arrivals)
  Columns: date, country, seats, flights (optional)

--events events.csv      OPTIONAL (known shocks)
  Columns: date, country(optional or 'ALL'), tag, note

Key options
-----------
--eu-members "AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IE,IT,LV,LT,LU,MT,NL,PL,PT,RO,SK,SI,ES,SE"
--baseline-year 2019
--start 2010-01
--end   2025-12
--min-months 36             Minimum monthly observations per country for full seasonality metrics
--outdir out_tourism
--country-filter ""         Comma-separated allowlist; default all in file

Outputs
-------
- clean_panel.csv            Standardized monthly panel (country, date, arrivals, nights, occupancy, etc.)
- seasonality.csv            Monthly seasonal indices (arrivals/nights) and seasonality strength per country
- cycle.csv                  YoY, 12m MA, momentum, phase flags
- recovery.csv               % vs baseline-year level, months-to-recovery per country/metric
- eu_aggregate.csv           EU rollups by month and simple EU seasonality/cycle
- capacity_gap.csv           (if air) demand vs seats (utilization proxy) and lead/lag correlation
- elasticities.csv           (if cpi/fx) OLS elasticities of arrivals on CPI and FX (country-by-country)
- events_joined.csv          (if events) tourism metrics joined with event tags
- summary.json               Headline KPIs (latest)
- config.json                Run configuration for reproducibility

Notes
-----
- This is a generic framework; bring your own official stats (Eurostat, NTOs).
- Occupancy can be in 0..1 or percentage; we normalize to 0..1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict

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

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pct_to_01(x: pd.Series) -> pd.Series:
    x = num(x)
    if x.max(skipna=True) is not None and x.max(skipna=True) > 1.5:
        return x / 100.0
    return x

def ma(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(w, min_periods=max(2, w//2)).mean()

def ann_growth(last: float, prev: float, months: int) -> float:
    if not np.isfinite(last) or not np.isfinite(prev) or prev <= 0 or months <= 0:
        return np.nan
    return float((last / prev) ** (12.0 / months) - 1.0)

def first_valid_year(series: pd.Series) -> Optional[int]:
    idx = series.dropna().index
    if len(idx) == 0:
        return None
    return int(idx.min().year)


# ---------------------------- Loaders ----------------------------

def load_tourism(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df, "date") or df.columns[0]): "date",
        (ncol(df, "country") or "country"): "country",
        (ncol(df, "arrivals") or "arrivals"): "arrivals",
        (ncol(df, "nights") or ncol(df, "overnights") or "nights"): "nights",
        (ncol(df, "occupancy") or "occupancy"): "occupancy",
        (ncol(df, "revenue_eur") or ncol(df, "revenue") or "revenue_eur"): "revenue_eur",
    }
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["country"] = df["country"].astype(str)
    for c in ["arrivals", "nights", "revenue_eur"]:
        if c in df.columns:
            df[c] = num(df[c])
    if "occupancy" in df.columns:
        df["occupancy"] = pct_to_01(df["occupancy"]).clip(0, 1)
    return df

def load_optional(path: Optional[str], cols_map: Dict[str, str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Minimal renamer
    out = {}
    for want, fallback in cols_map.items():
        col = ncol(df, want) or ncol(df, fallback) or fallback
        if col in df.columns:
            out[col] = want
    df = df.rename(columns=out)
    if "date" in df.columns:
        df["date"] = to_month(df["date"])
    for c in ["cpi", "fx", "seats", "flights"]:
        if c in df.columns:
            df[c] = num(df[c])
    if "country" in df.columns:
        df["country"] = df["country"].astype(str)
    return df


# ---------------------------- Seasonality & Cycle ----------------------------

def seasonal_indices(g: pd.DataFrame, col: str, min_months: int) -> pd.DataFrame:
    """
    Classical multiplicative seasonal index:
      SI_m = avg(value_m / 12m-MA) normalized to mean=1.
    Returns per-month indices and strength metric.
    """
    s = g.set_index("date").sort_index()[col].copy()
    if s.dropna().shape[0] < min_months:
        months = pd.DataFrame({"month": list(range(1,13)), "si": np.nan})
        months["strength"] = np.nan
        months["country"] = g["country"].iloc[0]
        months["metric"] = col
        return months
    trend = ma(s, 12)
    ratio = s / trend.replace(0, np.nan)
    by_m = ratio.groupby(ratio.index.month).mean()
    si = by_m / by_m.mean()
    # Seasonality strength (Hyndman style): 1 - Var(remainder) / Var(seasonal + remainder)
    rem = ratio / si.reindex(ratio.index.month).values
    var_r = np.nanvar(rem)
    var_s_plus_r = np.nanvar(ratio)
    strength = float(max(0.0, 1.0 - var_r / (var_s_plus_r + 1e-12)))
    months = pd.DataFrame({"month": si.index.values, "si": si.values})
    months["strength"] = strength
    months["country"] = g["country"].iloc[0]
    months["metric"] = col
    return months

def phase_flags(yoy: pd.Series, mom3_ann: pd.Series) -> pd.Series:
    """
    Simple phase logic per month:
      - Expansion: YoY>0 and 3m ann > 0
      - Slowdown: YoY>0 and 3m ann < 0
      - Recovery: YoY<0 and 3m ann > 0
      - Contraction: YoY<0 and 3m ann < 0
    """
    out = []
    for yy, mm in zip(yoy.values, mom3_ann.values):
        if not np.isfinite(yy) or not np.isfinite(mm):
            out.append("NA")
        elif yy >= 0 and mm >= 0:
            out.append("Expansion")
        elif yy >= 0 and mm < 0:
            out.append("Slowdown")
        elif yy < 0 and mm >= 0:
            out.append("Recovery")
        else:
            out.append("Contraction")
    return pd.Series(out, index=yoy.index)

def compute_cycle(g: pd.DataFrame, col: str) -> pd.DataFrame:
    s = g.set_index("date").sort_index()[col].copy()
    yoy = s.pct_change(12)
    m3 = ma(s, 3)
    m6 = ma(s, 6)
    mom3_ann = (m3 / m3.shift(3)) - 1.0
    mom6_ann = (m6 / m6.shift(6)) - 1.0
    phase = phase_flags(yoy, mom3_ann)
    out = pd.DataFrame({
        "date": s.index, "country": g["country"].iloc[0],
        f"{col}_yoy": yoy.values,
        f"{col}_ma12": ma(s, 12).values,
        f"{col}_mom3_ann": mom3_ann.values,
        f"{col}_mom6_ann": mom6_ann.values,
        f"{col}_phase": phase.values
    })
    return out

def recovery_vs_baseline(g: pd.DataFrame, col: str, baseline_year: int) -> pd.DataFrame:
    s = g.set_index("date").sort_index()[col].copy()
    base = s[s.index.year == baseline_year].mean()
    pct = s / (base if np.isfinite(base) and base > 0 else np.nan) - 1.0
    # months-to-recovery: first month ≥0 since 2020-01 (common shock baseline)
    start = pd.Timestamp(year=min(2020, int(s.index.min().year)), month=1, day=1)
    post = pct[pct.index >= start]
    mtr = None
    idx = post[post >= 0].index
    if len(idx) > 0:
        mtr = int((idx.min().to_period('M') - start.to_period('M')).n)
    return pd.DataFrame({
        "date": s.index, "country": g["country"].iloc[0],
        f"{col}_vs_{baseline_year}": pct.values
    }), mtr


# ---------------------------- Overlays ----------------------------

def capacity_gap(tour: pd.DataFrame, air: pd.DataFrame) -> pd.DataFrame:
    if air.empty:
        return pd.DataFrame()
    t = tour[["date","country","arrivals"]].copy()
    a = air[["date","country","seats"]].copy()
    m = t.merge(a, on=["date","country"], how="inner")
    m["seats"] = num(m["seats"])
    m["arrivals"] = num(m["arrivals"])
    m["demand_capacity_ratio"] = m["arrivals"] / m["seats"].replace(0, np.nan)
    # rough lead/lag correlation (arrivals vs seats lead 1-3m)
    rows = []
    for ctry, g in m.sort_values("date").groupby("country"):
        corr0 = np.corrcoef(g["arrivals"].fillna(method="ffill"), g["seats"].fillna(method="ffill"))[0,1] if len(g)>6 else np.nan
        lead1 = g["seats"].shift(1)
        lead2 = g["seats"].shift(2)
        corr1 = np.corrcoef(g["arrivals"].dropna(), lead1.dropna()) if len(g)>8 else [[np.nan,np.nan],[np.nan,np.nan]]
        corr2 = np.corrcoef(g["arrivals"].dropna(), lead2.dropna()) if len(g)>10 else [[np.nan,np.nan],[np.nan,np.nan]]
        rows.append({"country": ctry, "corr_same": float(corr0) if np.isfinite(corr0) else np.nan,
                     "corr_lead1": float(corr1[0][1]) if np.isfinite(corr1[0][1]) else np.nan,
                     "corr_lead2": float(corr2[0][1]) if np.isfinite(corr2[0][1]) else np.nan})
    corr_tbl = pd.DataFrame(rows)
    return m.merge(corr_tbl, on="country", how="left")

def elasticities(tour: pd.DataFrame, cpi: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    if cpi.empty and fx.empty:
        return pd.DataFrame()
    # Build log-arrivals and regress on Δlog CPI, Δlog FX with 3m lag (simple)
    df = tour[["date","country","arrivals"]].copy()
    df["log_arrivals"] = np.log(df["arrivals"].replace(0, np.nan))
    if not cpi.empty:
        cpi2 = cpi[["date","country","cpi"]].copy()
        cpi2["dlcpi"] = np.log(cpi2["cpi"]).diff()
        df = df.merge(cpi2[["date","country","dlcpi"]], on=["date","country"], how="left")
    if not fx.empty:
        fx2 = fx[["date","country","fx"]].copy()
        fx2["dlfx"] = np.log(fx2["fx"]).diff()
        df = df.merge(fx2[["date","country","dlfx"]], on=["date","country"], how="left")
    rows = []
    for ctry, g in df.sort_values("date").groupby("country"):
        y = g["log_arrivals"].diff()  # Δlog arrivals (approx %)
        Xcols = [c for c in ["dlcpi","dlfx"] if c in g.columns]
        if not Xcols or y.dropna().shape[0] < 24:
            rows.append({"country": ctry, "beta_cpi": np.nan, "beta_fx": np.nan, "r2": np.nan, "n": int(len(g))})
            continue
        X = g[Xcols].shift(3)  # 3-month lag
        Z = pd.concat([y, X], axis=1).dropna()
        if Z.empty or Z.shape[0] < 18:
            rows.append({"country": ctry, "beta_cpi": np.nan, "beta_fx": np.nan, "r2": np.nan, "n": int(len(g))})
            continue
        Y = Z.iloc[:,0].values
        A = np.column_stack([np.ones(len(Z)), *(Z[c].values for c in Xcols)])
        # OLS: beta = (A'A)^(-1) A'Y
        try:
            coef = np.linalg.lstsq(A, Y, rcond=None)[0]
            yhat = A @ coef
            r2 = 1.0 - np.sum((Y - yhat)**2) / max(1e-12, np.sum((Y - Y.mean())**2))
            beta_cpi = coef[Xcols.index("dlcpi")+1] if "dlcpi" in Xcols else np.nan
            beta_fx = coef[Xcols.index("dlfx")+1] if "dlfx" in Xcols else np.nan
        except Exception:
            beta_cpi = np.nan; beta_fx = np.nan; r2 = np.nan
        rows.append({"country": ctry, "beta_cpi": float(beta_cpi), "beta_fx": float(beta_fx), "r2": float(r2), "n": int(len(Z))})
    return pd.DataFrame(rows)


# ---------------------------- Aggregation ----------------------------

def eu_rollup(panel: pd.DataFrame, eu_members: List[str]) -> pd.DataFrame:
    df = panel.copy()
    df["is_eu"] = df["country"].str.upper().isin(eu_members).astype(int)
    agg = (df[df["is_eu"]==1]
           .groupby("date", as_index=False)
           .agg(arrivals=("arrivals","sum"),
                nights=("nights","sum"),
                occupancy=("occupancy","mean"),
                revenue_eur=("revenue_eur","sum")))
    # Add same cycle metrics as countries
    g = agg.copy(); g["country"] = "EU"
    cyc = compute_cycle(g, "arrivals").merge(compute_cycle(g, "nights"), on=["date","country"], how="outer")
    return agg.merge(cyc, on=["date"], how="left")

def country_snapshot_latest(panel: pd.DataFrame) -> pd.DataFrame:
    latest = panel["date"].max()
    snap = (panel[panel["date"]==latest]
            .groupby("country", as_index=False)
            .agg(arrivals=("arrivals","sum"),
                 nights=("nights","sum"),
                 occupancy=("occupancy","mean"),
                 revenue_eur=("revenue_eur","sum")))
    return snap.sort_values("arrivals", ascending=False)


# ---------------------------- CLI ----------------------------

@dataclass
class Config:
    tourism: str
    cpi: Optional[str]
    fx: Optional[str]
    air: Optional[str]
    events: Optional[str]
    eu_members: List[str]
    baseline_year: int
    start: str
    end: str
    min_months: int
    outdir: str
    country_filter: List[str]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EU tourism seasonality & cycle analytics")
    ap.add_argument("--tourism", required=True)
    ap.add_argument("--cpi", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--air", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--eu-members", default="AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IE,IT,LV,LT,LU,MT,NL,PL,PT,RO,SK,SI,ES,SE")
    ap.add_argument("--baseline-year", type=int, default=2019)
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--min-months", type=int, default=36)
    ap.add_argument("--country-filter", default="")
    ap.add_argument("--outdir", default="out_tourism")
    return ap.parse_args()


# ---------------------------- Main ----------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    eu_members = [x.strip().upper() for x in args.eu_members.split(",") if x.strip()]
    country_filter = [x.strip().upper() for x in args.country_filter.split(",") if x.strip()]

    tourism = load_tourism(args.tourism)

    # Optional inputs
    cpi = load_optional(args.cpi, {"date":"date","country":"country","cpi":"cpi"})
    fx  = load_optional(args.fx, {"date":"date","country":"country","fx":"fx"})
    air = load_optional(args.air, {"date":"date","country":"country","seats":"seats","flights":"flights"})
    events = load_optional(args.events, {"date":"date","country":"country","tag":"tag","note":"note"})

    # Filter date range & countries
    start = pd.to_datetime(args.start).to_period("M").to_timestamp()
    end = pd.to_datetime(args.end).to_period("M").to_timestamp()
    panel = tourism[(tourism["date"]>=start) & (tourism["date"]<=end)].copy()
    if country_filter:
        panel = panel[panel["country"].str.upper().isin(country_filter)].copy()

    # Normalize missing columns
    for col in ["arrivals","nights","occupancy","revenue_eur"]:
        if col not in panel.columns:
            panel[col] = np.nan

    # Write clean panel
    panel.sort_values(["country","date"]).to_csv(outdir / "clean_panel.csv", index=False)

    # Seasonality per country & metric
    seas_rows = []
    cyc_rows = []
    rec_rows = []
    mtr_map = {}
    for ctry, g in panel.groupby("country"):
        for metric in ["arrivals","nights"]:
            if metric not in g.columns: continue
            seas = seasonal_indices(g, metric, args.min_months)
            seas_rows.append(seas)
            cyc = compute_cycle(g, metric)
            cyc_rows.append(cyc)
            rec_tbl, mtr = recovery_vs_baseline(g, metric, args.baseline_year)
            rec_rows.append(rec_tbl)
            mtr_map[(ctry, metric)] = mtr
    seasonality = pd.concat(seas_rows, ignore_index=True) if seas_rows else pd.DataFrame()
    cycle_tbl = pd.concat(cyc_rows, ignore_index=True) if cyc_rows else pd.DataFrame()
    recovery_tbl = pd.concat(rec_rows, ignore_index=True) if rec_rows else pd.DataFrame()

    # EU aggregate
    eu_tbl = eu_rollup(panel, eu_members)

    # Capacity gap
    cap_tbl = capacity_gap(panel, air) if not air.empty else pd.DataFrame()

    # Elasticities
    elast_tbl = elasticities(panel, cpi, fx) if (not cpi.empty or not fx.empty) else pd.DataFrame()

    # Events join (optional)
    events_joined = pd.DataFrame()
    if not events.empty:
        ev = events.copy()
        # allow country='ALL'
        ev_all = ev[ev.get("country","").astype(str).str.upper().isin(["ALL","EU"])]
        ev_ctry = ev[~ev.index.isin(ev_all.index)]
        ej = panel.merge(ev_ctry, on=["date","country"], how="left")
        if not ev_all.empty:
            ej = ej.merge(ev_all.drop(columns=["country"]), on=["date"], how="left", suffixes=("","_all"))
            # combine tags/notes
            def combine(a, b):
                a = str(a) if pd.notna(a) else ""
                b = str(b) if pd.notna(b) else ""
                return "|".join([x for x in [a,b] if x])
            ej["tag"] = [combine(x,y) for x,y in zip(ej.get("tag",""), ej.get("tag_all",""))]
            ej["note"] = [combine(x,y) for x,y in zip(ej.get("note",""), ej.get("note_all",""))]
            ej = ej.drop(columns=[c for c in ej.columns if c.endswith("_all")], errors="ignore")
        events_joined = ej

    # Snapshots
    snapshot = country_snapshot_latest(panel)

    # Persist outputs
    if not seasonality.empty: seasonality.to_csv(outdir / "seasonality.csv", index=False)
    if not cycle_tbl.empty: cycle_tbl.to_csv(outdir / "cycle.csv", index=False)
    if not recovery_tbl.empty: recovery_tbl.to_csv(outdir / "recovery.csv", index=False)
    if not eu_tbl.empty: eu_tbl.to_csv(outdir / "eu_aggregate.csv", index=False)
    if not cap_tbl.empty: cap_tbl.to_csv(outdir / "capacity_gap.csv", index=False)
    if not elast_tbl.empty: elast_tbl.to_csv(outdir / "elasticities.csv", index=False)
    if not events_joined.empty: events_joined.to_csv(outdir / "events_joined.csv", index=False)
    snapshot.to_csv(outdir / "country_snapshot_latest.csv", index=False)

    # Summary KPIs
    latest = panel["date"].max() if not panel.empty else None
    latest_str = str(latest.date()) if latest is not None else None
    kpi = {
        "latest_month": latest_str,
        "countries": int(panel["country"].nunique()),
        "eu_members_in_data": int(panel["country"].str.upper().isin(eu_members).sum()),
        "min_months_required": args.min_months,
        "baseline_year": int(args.baseline_year),
        "months_to_recovery_estimates": {f"{k[0]}:{k[1]}": v for k,v in mtr_map.items() if v is not None},
        "top_countries_by_arrivals_latest": snapshot.head(10).set_index("country")["arrivals"].round(0).to_dict() if not snapshot.empty else {},
        "seasonality_strength_avg": float(seasonality.groupby("country")["strength"].mean().mean()) if not seasonality.empty else None
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        tourism=args.tourism, cpi=args.cpi or None, fx=args.fx or None, air=args.air or None, events=args.events or None,
        eu_members=eu_members, baseline_year=args.baseline_year, start=args.start, end=args.end,
        min_months=args.min_months, outdir=args.outdir, country_filter=country_filter
    )), indent=2))

    # Console output
    print("== EU Tourism Cycle ==")
    print(f"Latest month: {latest_str}  |  Countries: {kpi['countries']}")
    if kpi["months_to_recovery_estimates"]:
        ex = list(kpi["months_to_recovery_estimates"].items())[:5]
        print("Sample months-to-recovery:", dict(ex))
    if kpi["seasonality_strength_avg"] is not None:
        print(f"Avg seasonality strength: {kpi['seasonality_strength_avg']:.2f}")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
