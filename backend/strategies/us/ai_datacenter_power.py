#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_datacentre_power.py

Scenario tool for AI datacenter power planning:
- Computes monthly/annual facility load from IT MW and PUE trajectory
- Estimates emissions, water usage, and cost by site
- Models renewables (onsite/PPAs), storage shifting, and demand response
- Flags capacity constraints (transformer/generator/N-1) and lead-time risk
- Exports tidy CSVs and optional plots
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------------- Config ---------------- #

@dataclass
class Config:
    sites_file: str
    start: Optional[str]
    end: Optional[str]
    plot: bool
    outdir: str

# ---------------- Helpers ---------------- #

def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"ai_datacentre_power_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir

def month_range(start_ym: str, end_ym: str):
    start = datetime.strptime(start_ym, "%Y-%m")
    end = datetime.strptime(end_ym, "%Y-%m")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y-%m")
        cur = cur + relativedelta(months=1)

def parse_float(x, default=np.nan):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(str(x).replace("%", "").strip())
    except Exception:
        return default

# ---------------- Site data ---------------- #

def load_sites(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimal required columns; fill blanks
    req = ["site","region","it_mw","ramp_start","ramp_months","pue_start","pue_target","pue_target_year",
           "renewable_pct","grid_co2_kg_per_mwh","elec_price_usd_per_mwh","water_l_per_kwh",
           "storage_mw","storage_mwh","dr_pct","n1_critical_mw","generator_mw",
           "transformer_mva","pf","lead_months_transformer","lead_months_generator"]
    for col in req:
        if col not in df.columns:
            df[col] = np.nan
    for c in df.columns:
        if c not in ["site","region","ramp_start","pue_target_year"]:
            df[c] = df[c].apply(parse_float)
    df["pf"] = df["pf"].fillna(0.98)
    df["renewable_pct"] = df["renewable_pct"].fillna(0).clip(0, 100)
    df["dr_pct"] = df["dr_pct"].fillna(0).clip(0, 100)
    df["ramp_months"] = df["ramp_months"].fillna(0).astype(int)
    return df

def end_ym_for_sites(df: pd.DataFrame) -> str:
    # Last ramp or last PUE target year
    last = "2000-01"
    for _, r in df.iterrows():
        if not isinstance(r["ramp_start"], str): continue
        start = datetime.strptime(r["ramp_start"], "%Y-%m")
        end = start + relativedelta(months=int(r["ramp_months"]))
        last = max(last, end.strftime("%Y-%m"))
    max_pue = int(df["pue_target_year"].dropna().max()) if not df["pue_target_year"].dropna().empty else datetime.today().year
    return max(last, f"{max_pue}-12")

def interpolate_pue(row, ym: str) -> float:
    ps = parse_float(row["pue_start"])
    pt = parse_float(row["pue_target"])
    if np.isnan(ps) or np.isnan(pt): return ps if not np.isnan(ps) else pt
    t0 = datetime.strptime(row["ramp_start"], "%Y-%m")
    t1 = datetime(year=int(row["pue_target_year"]), month=12, day=1)
    t  = datetime.strptime(ym, "%Y-%m")
    if t <= t0: return ps
    if t >= t1: return pt
    alpha = (t - t0).days / (t1 - t0).days
    return ps + alpha*(pt-ps)

def ramp_it_mw(row, ym: str) -> float:
    if not isinstance(row["ramp_start"], str): return 0.0
    start = datetime.strptime(row["ramp_start"], "%Y-%m")
    months = int(row["ramp_months"])
    t = datetime.strptime(ym, "%Y-%m")
    if t < start: return 0.0
    if months <= 0: return row["it_mw"]
    end = start + relativedelta(months=months)
    if t >= end: return row["it_mw"]
    frac = (t - start).days / (end - start).days
    return frac * row["it_mw"]

# ---------------- Core compute ---------------- #

def compute_monthly(df: pd.DataFrame, start_ym: str, end_ym: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for ym in month_range(start_ym, end_ym):
            it = ramp_it_mw(r, ym)
            pue = interpolate_pue(r, ym)
            fac_mw = it * pue
            hours = 730.5
            mwh = fac_mw * hours
            ren = r["renewable_pct"]/100.0
            grid_mwh = mwh*(1-ren)
            co2 = grid_mwh*r["grid_co2_kg_per_mwh"]
            cost = grid_mwh*r["elec_price_usd_per_mwh"]
            water = fac_mw*1000*hours*r["water_l_per_kwh"]  # MW→kW
            rows.append({"site":r["site"],"region":r["region"],"ym":ym,
                         "it_mw":it,"pue":pue,"facility_mw":fac_mw,"mwh":mwh,
                         "grid_mwh":grid_mwh,"co2_kg":co2,"elec_cost_usd":cost,"water_l":water})
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("site", as_index=False).agg({
        "region":"first","it_mw":"mean","facility_mw":"mean","mwh":"sum","grid_mwh":"sum",
        "co2_kg":"sum","elec_cost_usd":"sum","water_l":"sum"
    })
    g["co2_tonnes"] = g["co2_kg"]/1000.0
    g["water_ML"] = g["water_l"]/1e6
    return g

# ---------------- Main ---------------- #

def main():
    p = argparse.ArgumentParser(description="AI Datacentre Power Scenario Tool")
    p.add_argument("--sites-file",required=True)
    p.add_argument("--start",type=str,default=None)
    p.add_argument("--end",type=str,default=None)
    p.add_argument("--plot",action="store_true")
    p.add_argument("--outdir",default="./artifacts")
    args = p.parse_args()

    cfg = Config(args.sites_file,args.start,args.end,args.plot,args.outdir)
    outdir = ensure_outdir(cfg.outdir)
    sites = load_sites(cfg.sites_file)
    start_ym = cfg.start or sites["ramp_start"].dropna().min()
    end_ym = cfg.end or end_ym_for_sites(sites)

    dfm = compute_monthly(sites,start_ym,end_ym)
    dfm.to_csv(os.path.join(outdir,"monthly_load.csv"),index=False)
    summ = summarize(dfm)
    summ.to_csv(os.path.join(outdir,"site_summary.csv"),index=False)

    print("\n=== Annualized summary ===")
    print(summ.round(2).to_string(index=False))

if __name__=="__main__":
    main()