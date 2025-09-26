#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
india_space_launch.py — ISRO launch analytics: cadence, reliability, mass-to-orbit & simple forecasts
-----------------------------------------------------------------------------------------------------

What this does
==============
Given a CSV of launches (ideally India-origin / ISRO/NSIL missions), this script:

1) Cleans & standardizes fields with flexible headers (date, vehicle, outcome, mass_kg, orbit, customer, site, mission).
2) Builds a per-month time series of launch cadence, success rate, and payload mass to orbit.
3) Computes per-vehicle reliability with Bayesian (Beta) credible intervals and rolling reliability.
4) Breaks down domestic vs foreign customer share (by count & mass), orbit mix (LEO/SSO/GTO/Deep Space).
5) Forecasts launch cadence for the next 12 months via a simple seasonal Poisson baseline,
   optionally integrating a planned **manifest.csv** (planned launches with tentative dates).

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--launches launches.csv     REQUIRED
  Columns (any subset; extras ignored):
    date, mission, vehicle, outcome, payload_mass_kg, orbit, customer, site, country
  • outcome accepted examples → Success / Failure / Partial (many variants auto-mapped)
  • orbit examples → LEO, SSO/PSO, GTO, MEO, GEO, Lunar/Interplanetary/DeepSpace, Suborbital
  • payload_mass_kg: numeric; if missing we still compute cadence/reliability

--manifest manifest.csv     OPTIONAL (planned launches)
  Columns: date_planned (or date), vehicle (optional), payload_mass_kg (optional)

CLI (examples)
--------------
python india_space_launch.py \
  --launches launches.csv --manifest manifest.csv \
  --start 2010-01-01 --windows 6,12,24 --horizon 12 --outdir out_isro

Key parameters
--------------
--windows 6,12,24      Rolling windows (months) for cadence and success rate
--horizon 12           Forecast horizon in months
--start/--end          Date filters (inclusive)
--outdir out_isro      Output directory

Outputs
-------
- panel_launches.csv        Per-launch cleaned panel (standardized fields & derived flags)
- monthly_timeseries.csv    Monthly counts, successes, success_rate, payload mass; with rolling windows
- vehicle_reliability.csv   Per-vehicle successes/failures, posterior mean & credible intervals
- orbit_mix_by_year.csv     Yearly orbit mix (share by count & mass)
- customer_mix_by_year.csv  Yearly domestic/foreign share (count & mass)
- forecast.csv              Monthly cadence forecast (baseline + manifest overlay)
- summary.json              Headline KPIs (date range, totals, top vehicle, latest rolling success)
- config.json               Run configuration

DISCLAIMER
----------
Research tool; depends on input completeness/quality. No guarantee of accuracy.
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

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return the first matching column (case-insensitive, fuzzy contains)."""
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

def to_month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def month_id(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("M").dt.to_timestamp("M")

def normalize_outcome(x: str) -> str:
    if not isinstance(x, str):
        return "Unknown"
    t = x.strip().lower()
    # Success variants
    succ = ["success", "successful", "fully successful", "nominal", "operational", "ok"]
    part = ["partial", "partial success", "marginal", "anomaly", "injection anomaly"]
    fail = ["fail", "failure", "unsuccessful", "lost", "abort", "explosion", "scrubbed"]
    for w in succ:
        if w in t: return "Success"
    for w in part:
        if w in t: return "Partial"
    for w in fail:
        if w in t: return "Failure"
    return x.title()

def normalize_orbit(x: str) -> str:
    if not isinstance(x, str): return "Unknown"
    t = x.upper()
    if any(k in t for k in ["LUNAR", "MOON", "INTERPLANET", "MARS", "VENUS", "DEEP"]): return "DeepSpace"
    if "SUB" in t: return "Suborbital"
    if "GTO" in t: return "GTO"
    if "GEO" in t or "IGSO" in t: return "GEO"
    if "MEO" in t: return "MEO"
    if any(k in t for k in ["SSO", "PSO", "POLAR"]): return "SSO"
    if "LEO" in t or "LOW" in t or "LDO" in t: return "LEO"
    return "Other"

def is_domestic(customer: Optional[str], country: Optional[str]) -> Optional[bool]:
    """Heuristic: Domestic if country is India/IN, or customer string contains India/ISRO/NSIL/DoS."""
    if isinstance(country, str):
        if country.strip().lower() in ["india", "in", "ind", "bharat"]: return True
        if country.strip().lower() in ["usa","us","fr","ru","eu","uk","jp","de","it","kr","ae","sg","au","es","il","sa","br","ca","cn"]: return False
    if isinstance(customer, str):
        t = customer.lower()
        if any(k in t for k in ["isro","nsil","doe","dos","govt of india","government of india","in-space", "in–space", "in-space"]):
            return True
        if any(k in t for k in ["nasa","esa","cnsa","roscosmos","jaxa","uae","korea","oneweb","amazon","starlink","uk","fra","usa","europe","german","italy"]):
            return False
    return None

def beta_credible_interval(succ: int, fail: int, a: float=1.0, b: float=1.0, q: Tuple[float,float]=(0.05,0.95)) -> Tuple[float,float,float]:
    """
    Return (posterior_mean, q_low, q_hi) for Beta(a+succ, b+fail).
    We approximate quantiles via normal if scipy isn't available.
    """
    A = a + succ
    B = b + fail
    mean = A / (A + B) if (A+B)>0 else np.nan
    # Normal approximation to Beta quantiles
    var = (A*B) / (((A+B)**2) * (A+B+1)) if (A+B)>2 else np.nan
    if var == var and var > 0:
        sd = np.sqrt(var)
        from math import erf, sqrt
        # inverse CDF approx via Beasley-Springer/Moro would be overkill; use ±1.645 for ~90%
        zlo, zhi = -1.6449, +1.6449
        if q != (0.05, 0.95):
            # fallback to symmetrical around mean
            alpha = (q[1]-q[0])/2.0
            z = 1.6449 if abs(alpha-0.45) < 1e-6 else 1.2816  # crude switch 90% vs 80%
            zlo, zhi = -z, +z
        return mean, max(0.0, mean + zlo*sd), min(1.0, mean + zhi*sd)
    # if too few samples, just return mean and NaNs
    return mean, np.nan, np.nan


# ----------------------------- loaders -----------------------------

def load_launches(path: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map columns
    dtc  = ncol(df, "date", "launch_date", "datetime", "utc")
    vehc = ncol(df, "vehicle", "launcher", "rocket", "lv")
    outc = ncol(df, "outcome", "result", "status")
    mass = ncol(df, "payload_mass_kg", "mass_kg", "payload_kg", "liftoff_mass_kg", "payload_mass")
    orb  = ncol(df, "orbit", "target_orbit", "mission_orbit")
    cust = ncol(df, "customer", "operator", "agency", "owner")
    site = ncol(df, "site", "launch_site", "pad", "location")
    mission = ncol(df, "mission", "name", "payload", "flight")

    if not dtc or not vehc:
        raise ValueError("launches.csv must contain at least date and vehicle columns.")

    df = df.rename(columns={
        dtc: "date_raw",
        vehc: "vehicle",
        outc or "outcome": "outcome",
        mass or "payload_mass_kg": "payload_mass_kg",
        orb or "orbit": "orbit",
        cust or "customer": "customer",
        site or "site": "site",
        mission or "mission": "mission"
    })

    # Parse dates (keep both datetime and month)
    df["date"] = to_date(df["date_raw"])
    # Drop rows with invalid dates
    df = df.dropna(subset=["date"])

    # Apply date range
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    # Normalize fields
    if "outcome" in df.columns:
        df["outcome_std"] = df["outcome"].astype(str).apply(normalize_outcome)
    else:
        df["outcome_std"] = "Unknown"

    df["orbit_std"] = df["orbit"].astype(str).apply(normalize_orbit) if "orbit" in df.columns else "Unknown"
    df["vehicle"] = df["vehicle"].astype(str).str.strip().str.upper()

    # Vehicle family (PSLV/GSLV/SSLV/LVM3 & variants)
    def veh_family(v: str) -> str:
        t = v.upper()
        if "PSLV" in t: return "PSLV"
        if "GSLV-MK" in t or "GSLV MK" in t or "LVM3" in t or "MK3" in t or "MK III" in t: return "LVM3"
        if "GSLV" in t: return "GSLV"
        if "SSLV" in t: return "SSLV"
        if "RH" in t or "ROHINI" in t or "SOUNDING" in t: return "Sounding"
        return v.split()[0] if v else "Other"
    df["vehicle_family"] = df["vehicle"].apply(veh_family)

    # Success flag
    df["success_flag"] = df["outcome_std"].map({"Success": 1, "Partial": 0, "Failure": 0}).fillna(np.nan)

    # Payload mass
    if "payload_mass_kg" in df.columns:
        df["payload_mass_kg"] = safe_num(df["payload_mass_kg"])
    else:
        df["payload_mass_kg"] = np.nan

    # Domestic vs foreign
    country_c = ncol(df, "country", "customer_country", "nation")
    dom = []
    for i, r in df.iterrows():
        dom.append(is_domestic(str(r.get("customer", "")), str(r.get(country_c, "")) if country_c else None))
    df["is_domestic_customer"] = dom

    # Derived date buckets
    df["month"] = month_id(df["date"])
    df["year"] = df["date"].dt.year

    # Sort & index
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date","month","year","mission","vehicle","vehicle_family","outcome_std","success_flag",
               "payload_mass_kg","orbit_std","customer","site","is_domestic_customer"]]

def load_manifest(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date_planned", "date", "no_earlier_than", "target_date")
    if not dt:
        raise ValueError("manifest.csv must include a planned date column (date_planned/date).")
    df = df.rename(columns={dt: "date_planned"})
    df["date_planned"] = to_date(df["date_planned"])
    df["month"] = month_id(df["date_planned"])
    veh = ncol(df, "vehicle", "launcher", "lv")
    mass = ncol(df, "payload_mass_kg", "mass_kg", "payload_kg")
    if veh: df = df.rename(columns={veh: "vehicle"})
    if mass: df = df.rename(columns={mass: "payload_mass_kg"})
    # Keep only future/planned
    df = df.dropna(subset=["month"]).sort_values("month")
    return df[["month"] + ([ "vehicle" ] if "vehicle" in df.columns else []) + ([ "payload_mass_kg" ] if "payload_mass_kg" in df.columns else [])]

# ----------------------------- analytics -----------------------------

def monthly_series(launches: pd.DataFrame) -> pd.DataFrame:
    m = launches.groupby("month").agg(
        launches=("date","count"),
        successes=("success_flag", lambda s: int(np.nansum(s.values)))  # treat NaN as 0 in sum? we nansum
    ).reset_index()
    # Mass totals by month
    mass = launches.dropna(subset=["payload_mass_kg"]).groupby("month")["payload_mass_kg"].sum().rename("payload_mass_kg").reset_index()
    out = m.merge(mass, on="month", how="left")
    out["success_rate"] = out["successes"] / out["launches"].replace(0, np.nan)
    return out.sort_values("month")

def rolling_windows(ts: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = ts.set_index("month").copy()
    for w in windows:
        df[f"launches_roll_{w}m"] = df["launches"].rolling(w, min_periods=max(3, w//2)).sum()
        df[f"successes_roll_{w}m"] = df["successes"].rolling(w, min_periods=max(3, w//2)).sum()
        df[f"success_rate_roll_{w}m"] = df[f"successes_roll_{w}m"] / df[f"launches_roll_{w}m"].replace(0, np.nan)
        df[f"mass_roll_{w}m"] = df["payload_mass_kg"].rolling(w, min_periods=max(3, w//2)).sum()
    return df.reset_index()

def per_vehicle_reliability(launches: pd.DataFrame) -> pd.DataFrame:
    G = (launches.groupby("vehicle_family")
         .agg(launches=("date","count"),
              successes=("success_flag", lambda s: int(np.nansum(s.values))),
              failures=("success_flag", lambda s: int(np.sum((s==0).astype(int)))))
         .reset_index())
    rows = []
    for _, r in G.iterrows():
        mean, lo, hi = beta_credible_interval(int(r["successes"]), int(r["failures"]), a=1.0, b=1.0, q=(0.05,0.95))
        rows.append({
            "vehicle_family": r["vehicle_family"],
            "launches": int(r["launches"]),
            "successes": int(r["successes"]),
            "failures": int(r["failures"]),
            "reliability_post_mean": mean,
            "reliability_ci90_low": lo,
            "reliability_ci90_high": hi
        })
    return pd.DataFrame(rows).sort_values(["launches","reliability_post_mean"], ascending=[False, False])

def mix_by_year(launches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Orbit mix
    by_orbit = (launches.groupby(["year","orbit_std"])
                .agg(count=("date","count"),
                     mass=("payload_mass_kg","sum"))
                .reset_index())
    tot = by_orbit.groupby("year")["count","mass"].sum().rename(columns={"count":"count_tot","mass":"mass_tot"}).reset_index()
    by_orbit = by_orbit.merge(tot, on="year", how="left")
    by_orbit["share_count"] = by_orbit["count"] / by_orbit["count_tot"].replace(0, np.nan)
    by_orbit["share_mass"]  = by_orbit["mass"]  / by_orbit["mass_tot"].replace(0, np.nan)
    # Customer mix (Domestic/Foreign)
    launches["cust_bucket"] = launches["is_domestic_customer"].map({True:"Domestic", False:"Foreign"}).fillna("Unknown")
    by_cust = (launches.groupby(["year","cust_bucket"])
               .agg(count=("date","count"),
                    mass=("payload_mass_kg","sum"))
               .reset_index())
    tot2 = by_cust.groupby("year")["count","mass"].sum().rename(columns={"count":"count_tot","mass":"mass_tot"}).reset_index()
    by_cust = by_cust.merge(tot2, on="year", how="left")
    by_cust["share_count"] = by_cust["count"] / by_cust["count_tot"].replace(0, np.nan)
    by_cust["share_mass"]  = by_cust["mass"]  / by_cust["mass_tot"].replace(0, np.nan)
    return by_orbit.sort_values(["year","orbit_std"]), by_cust.sort_values(["year","cust_bucket"])

def seasonal_poisson_forecast(monthly: pd.DataFrame, horizon: int=12, manifest: pd.DataFrame=pd.DataFrame()) -> pd.DataFrame:
    """
    Simple cadence forecast:
      - Compute monthly counts over last 24–36 months (if available).
      - Seasonal baseline = average count by calendar month-of-year from the last 3Y (fallback overall mean).
      - Expected count = seasonal baseline; 90% interval via normal approx for Poisson.
      - If manifest provided, enforce forecast >= planned count for that month.
    """
    if monthly.empty:
        return pd.DataFrame(columns=["month","forecast_mean","p90_low","p90_high","manifest_count","final_forecast"])
    hist = monthly.copy().sort_values("month")
    if hist.shape[0] < 6:
        # very short history — use overall mean
        lam_overall = float(hist["launches"].mean()) if not hist.empty else 0.0
        last = hist["month"].max() if not hist.empty else pd.Timestamp.today().to_period("M").to_timestamp("M")
        fut = pd.date_range(start=(last + pd.offsets.MonthEnd(1)), periods=horizon, freq="M")
        df = pd.DataFrame({"month": fut})
        df["forecast_mean"] = lam_overall
        sd = np.sqrt(lam_overall)
        df["p90_low"] = np.maximum(0.0, lam_overall - 1.6449*sd)
        df["p90_high"] = lam_overall + 1.6449*sd
    else:
        # last 36 months seasonal profile
        cutoff = hist["month"].max() - pd.DateOffset(months=36)
        recent = hist[hist["month"] > cutoff] if (hist["month"] > cutoff).any() else hist
        recent["moy"] = recent["month"].dt.month
        prof = recent.groupby("moy")["launches"].mean()
        lam_overall = float(recent["launches"].mean())
        last = hist["month"].max()
        fut = pd.date_range(start=(last + pd.offsets.MonthEnd(1)), periods=horizon, freq="M")
        df = pd.DataFrame({"month": fut})
        df["moy"] = df["month"].dt.month
        df["forecast_mean"] = df["moy"].map(prof).fillna(lam_overall)
        sd = np.sqrt(df["forecast_mean"])
        df["p90_low"] = np.maximum(0.0, df["forecast_mean"] - 1.6449*sd)
        df["p90_high"] = df["forecast_mean"] + 1.6449*sd
        df = df.drop(columns=["moy"])
    # Manifest overlay
    if not manifest.empty and "month" in manifest.columns:
        mc = manifest.groupby("month").size().rename("manifest_count").reset_index()
        df = df.merge(mc, on="month", how="left")
    else:
        df["manifest_count"] = np.nan
    df["final_forecast"] = np.maximum(df["forecast_mean"], df["manifest_count"].fillna(0.0))
    return df

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    launches: str
    manifest: Optional[str]
    start: Optional[str]
    end: Optional[str]
    windows: str
    horizon: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ISRO launch analytics: cadence, reliability & simple forecasts")
    ap.add_argument("--launches", required=True, help="CSV of launches (date, vehicle, outcome, payload_mass_kg, orbit, customer, site)")
    ap.add_argument("--manifest", default="", help="CSV of planned launches (date_planned/date[, vehicle, payload_mass_kg])")
    ap.add_argument("--start", default="", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="", help="End date (YYYY-MM-DD)")
    ap.add_argument("--windows", default="6,12,24", help="Rolling windows in months (comma-separated)")
    ap.add_argument("--horizon", type=int, default=12, help="Forecast horizon in months")
    ap.add_argument("--outdir", default="out_isro")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    LAUNCHES = load_launches(args.launches, start=(args.start or None), end=(args.end or None))
    if LAUNCHES.empty:
        raise ValueError("No launches after filtering; check your inputs or date range.")

    MANIFEST = load_manifest(args.manifest) if args.manifest else pd.DataFrame()

    # Monthly time series + rolling
    monthly = monthly_series(LAUNCHES)
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    monthly_roll = rolling_windows(monthly, windows)

    # Per-vehicle reliability
    veh_rel = per_vehicle_reliability(LAUNCHES)

    # Mixes
    orbit_mix, cust_mix = mix_by_year(LAUNCHES)

    # Forecast
    fc = seasonal_poisson_forecast(monthly, horizon=int(args.horizon), manifest=MANIFEST)

    # Save outputs
    LAUNCHES.to_csv(outdir / "panel_launches.csv", index=False)
    monthly_roll.to_csv(outdir / "monthly_timeseries.csv", index=False)
    veh_rel.to_csv(outdir / "vehicle_reliability.csv", index=False)
    orbit_mix.to_csv(outdir / "orbit_mix_by_year.csv", index=False)
    cust_mix.to_csv(outdir / "customer_mix_by_year.csv", index=False)
    if not fc.empty:
        fc.to_csv(outdir / "forecast.csv", index=False)

    # Summary
    last_row = monthly_roll.tail(1).iloc[0]
    top_vehicle = veh_rel.iloc[0]["vehicle_family"] if not veh_rel.empty else None
    summary = {
        "date_range": {
            "start": str(LAUNCHES["date"].min().date()),
            "end": str(LAUNCHES["date"].max().date())
        },
        "total_launches": int(LAUNCHES.shape[0]),
        "total_successes": int(np.nansum(LAUNCHES["success_flag"].values)),
        "overall_success_rate": float(np.nansum(LAUNCHES["success_flag"].values) / LAUNCHES.shape[0]),
        "latest_month": str(last_row["month"].date()),
        "latest_launches": int(last_row["launches"]),
        "latest_success_rate": float(last_row["success_rate"]) if pd.notna(last_row["success_rate"]) else None,
        "top_vehicle_family_by_volume": top_vehicle,
        "vehicle_reliability_top3": veh_rel.head(3).to_dict(orient="records"),
        "forecast_horizon_m": int(args.horizon),
        "manifest_entries": int(MANIFEST.shape[0]) if not MANIFEST.empty else 0
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config
    cfg = asdict(Config(
        launches=args.launches, manifest=(args.manifest or None),
        start=(args.start or None), end=(args.end or None),
        windows=args.windows, horizon=int(args.horizon), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== ISRO Launch Analytics ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Total launches: {summary['total_launches']}")
    print(f"Overall success rate: {summary['overall_success_rate']*100:.1f}% | Top vehicle family: {summary['top_vehicle_family_by_volume']}")
    if not fc.empty:
        m0 = fc.iloc[0]["month"].strftime("%Y-%m")
        print(f"Forecast ({m0} +{args.horizon}m): mean={fc['final_forecast'].mean():.2f} /mo; "
              f"manifest overlay months={int(fc['manifest_count'].notna().sum())}")

if __name__ == "__main__":
    main()
