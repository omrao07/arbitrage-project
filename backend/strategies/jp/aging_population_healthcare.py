#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aging_population_healthcare.py — Demographics → care demand, capacity gaps & scenarios
-------------------------------------------------------------------------------------

What this does
==============
A research toolkit to connect **population aging** to **healthcare demand, capacity & costs**.
It ingests population by age-cohort, age-specific utilization, and healthcare supply/cost data, then:

1) Baselines & diagnostics
   • Age structure, median age, old-age dependency ratio (OADR = 65+ / 15–64)
   • Potential support ratio (PSR = 15–64 / 65+), aging velocity
   • Per-capita utilization by age → current demand (OP visits, admissions, bed-days, LTC)

2) Projections (annual)
   • Cohort-aging engine (simple survival & migration) → population by age-band to horizon
   • Demand projections: multiply projected cohorts × age-specific utilization
   • Capacity paths: beds, doctors, nurses (observed series or parametric growth)
   • Gaps: bed-days vs capacity @ target occupancy; physician visit capacity vs demand

3) Costs & inflation
   • Service unit-costs (optional) + health CPI → spending projections by service

4) Stress & early warning
   • Healthcare Stress Index (HSI): z-blend of occupancy, physician shortfall, OADR
   • Alerts when HSI > threshold / rises rapidly

5) Scenarios (plug-and-play)
   • Life expectancy ↑ (via survival uplift), fertility / crude birth rate change
   • Net migration by age (young vs elderly inflows)
   • Tech/telemedicine relief (↓ OP utilization by x%)
   • Capacity expansion plan (beds/doctors CAGR or explicit CSV)

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--population population.csv     REQUIRED
  Columns (any subset):
    year|date, age_band|age|age_group (e.g., "0-4","5-9","85+"), sex (optional),
    population|pop

--survival survival.csv         OPTIONAL (annual survival prob to next year)
  Columns:
    age|age_band, survival_prob (0..1)
  Notes: If banded, map by band; else if single-year, averaged to band mid-age.

--fertility fertility.csv       OPTIONAL (to model births; else crude birth rate used)
  Columns:
    age|age_band (female), asfr (age-specific fertility rate per woman per year)

--utilization utilization.csv   RECOMMENDED (per-capita rates by age band)
  Columns:
    age|age_band,
    op_visits_pc, admissions_per_1k, bed_days_pc[, ltc_days_pc]

--supply supply.csv             OPTIONAL (time series of capacity; else parametric)
  Columns (any subset):
    year|date, beds_total[, icu_beds], doctors_fte[, nurses_fte]

--unitcosts unit_costs.csv      OPTIONAL (service unit costs, base-year currency)
  Columns:
    service, unit_cost
  Services recognized: op_visit, admission, bed_day, ltc_day

--prices prices.csv             OPTIONAL (health CPI or deflators)
  Columns:
    year|date, cpi_health_idx[, cpi_all_idx]

--migration migration.csv       OPTIONAL (net migration by age band, persons/yr; +inflow)
  Columns:
    year|date, age|age_band, net_migration

CLI (key)
---------
--horizon 20                    Years to project (forward from last population year)
--target_occ 0.85               Target safe occupancy cap for beds (0–1)
--visits_per_doctor 4500        Annual OP visits a doctor can handle (FTE)
--doctor_cagr 2.0               %/yr if supply.csv missing doctors (else ignored)
--beds_cagr 2.0                 %/yr if supply.csv missing beds (else ignored)
--cbr_per_1000 16               Crude birth rate used when fertility.csv absent
--life_uplift_pp 0.0            Survival uplift (pp) applied each year (scenario)
--telemed_relief_pct 0.0        % reduction in OP utilization (all ages)
--stress_thresh 1.5             HSI z-score threshold for alerts
--start / --end                 Trim history by year (YYYY or YYYY-MM-DD)
--outdir out_health             Output directory

Outputs
-------
- population_clean.csv          Tidy population by year & age band
- population_projection.csv     Projected cohorts to horizon
- demand_projection.csv         OP visits, admissions, bed-days, LTC by year
- capacity_projection.csv       Beds/doctors paths, occupancy, shortfalls
- spending_projection.csv       Service spend by year (if unitcosts/prices provided)
- stress_index.csv              HSI & triggers
- summary.json                  Headline diagnostics and pointers
- config.json                   Echo of run configuration

Assumptions
-----------
• Annual frequency; age bands can be "0-4","5-9",..."85+" (flexible).
• Cohort aging is approximate for 5y bands (uses annual survival @ band mid-age).
• Births: uses fertility.csv (ASFR) if provided; else crude birth rate (CBR) applied on total pop.
• Physician capacity proxy = doctors_fte × visits_per_doctor; nurse constraint not binding.
• Bed capacity proxy = beds × 365 × target_occ (safe cap).

DISCLAIMER
----------
This is research tooling with simplifying assumptions. Validate mappings, survival, and utilization
before operational/policy decisions.
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

def to_year(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    return dt.dt.year

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def parse_age_band(s: str) -> Tuple[int, int]:
    """Returns (age_min, age_max) inclusive. '85+' -> (85, 120). '0-4' -> (0,4). '65–69' okay."""
    t = str(s).strip().replace(" ", "").replace("–","-").replace("—","-")
    if t.endswith("+"):
        a = int(t[:-1])
        return a, 120
    if "-" in t:
        a,b = t.split("-",1)
        return int(a), int(b)
    # single age
    try:
        a = int(t); return a, a
    except:
        raise ValueError(f"Unrecognized age band: {s}")

def age_mid(amin: int, amax: int) -> float:
    return (amin + amax) / 2.0

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

# ----------------------------- loaders -----------------------------

def load_population(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    yr  = ncol(df, "year","date")
    age = ncol(df, "age_band","age","age_group")
    pop = ncol(df, "population","pop")
    sex = ncol(df, "sex","gender")
    if not (yr and age and pop):
        raise ValueError("population.csv must include year/date, age(age_band), and population.")
    df = df.rename(columns={yr:"year", age:"age_band", pop:"population"})
    if not np.issubdtype(df["year"].dtype, np.number):
        df["year"] = to_year(df["year"])
    df["population"] = safe_num(df["population"])
    if sex: df = df.rename(columns={sex:"sex"})
    # parse ages
    ages = df["age_band"].apply(parse_age_band)
    df["age_min"] = [a[0] for a in ages]
    df["age_max"] = [a[1] for a in ages]
    df["age_mid"] = df.apply(lambda r: age_mid(r["age_min"], r["age_max"]), axis=1)
    # tidy
    df = df[["year","age_band","age_min","age_max","age_mid","population"] + (["sex"] if "sex" in df.columns else [])]
    return df.sort_values(["year","age_min"])

def load_survival(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    age = ncol(df, "age_band","age","age_group")
    sv  = ncol(df, "survival_prob","survival","s")
    if not (age and sv): raise ValueError("survival.csv needs age(age_band) and survival_prob.")
    df = df.rename(columns={age:"age_band", sv:"survival_prob"})
    ages = df["age_band"].apply(parse_age_band)
    df["age_min"] = [a[0] for a in ages]; df["age_max"] = [a[1] for a in ages]
    df["age_mid"] = df.apply(lambda r: age_mid(r["age_min"], r["age_max"]), axis=1)
    df["survival_prob"] = df["survival_prob"].clip(lower=0.0, upper=1.0)
    return df[["age_band","age_min","age_max","age_mid","survival_prob"]].sort_values("age_min")

def load_fertility(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    age = ncol(df, "age_band","age","age_group")
    asfr= ncol(df, "asfr","fertility_rate","rate")
    if not (age and asfr): raise ValueError("fertility.csv needs age(age_band) and asfr.")
    df = df.rename(columns={age:"age_band", asfr:"asfr"})
    ages = df["age_band"].apply(parse_age_band)
    df["age_min"] = [a[0] for a in ages]; df["age_max"] = [a[1] for a in ages]
    df["age_mid"] = df.apply(lambda r: age_mid(r["age_min"], r["age_max"]), axis=1)
    df["asfr"] = safe_num(df["asfr"]).clip(lower=0)
    return df[["age_band","age_min","age_max","age_mid","asfr"]].sort_values("age_min")

def load_utilization(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    age = ncol(df, "age_band","age","age_group")
    if not age: raise ValueError("utilization.csv needs age(age_band).")
    df = df.rename(columns={age:"age_band"})
    ages = df["age_band"].apply(parse_age_band)
    df["age_min"] = [a[0] for a in ages]; df["age_max"] = [a[1] for a in ages]
    df["age_mid"] = df.apply(lambda r: age_mid(r["age_min"], r["age_max"]), axis=1)
    # normalize columns
    ren = {}
    for k, std in [
        ("op_visits_pc","op_visits_pc"),
        ("admissions_per_1k","admissions_per_1k"),
        ("bed_days_pc","bed_days_pc"),
        ("ltc_days_pc","ltc_days_pc")
    ]:
        c = ncol(df, k)
        if c: ren[c] = std
    df = df.rename(columns=ren)
    for c in ["op_visits_pc","admissions_per_1k","bed_days_pc","ltc_days_pc"]:
        if c in df.columns: df[c] = safe_num(df[c]).clip(lower=0)
    return df[["age_band","age_min","age_max","age_mid"] + [c for c in ["op_visits_pc","admissions_per_1k","bed_days_pc","ltc_days_pc"] if c in df.columns]].sort_values("age_min")

def load_supply(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    yr = ncol(df, "year","date")
    if not yr: raise ValueError("supply.csv needs year/date.")
    df = df.rename(columns={yr:"year"})
    if not np.issubdtype(df["year"].dtype, np.number):
        df["year"] = to_year(df["year"])
    ren = {}
    for k,std in [("beds_total","beds_total"),("icu_beds","icu_beds"),("doctors_fte","doctors_fte"),("nurses_fte","nurses_fte")]:
        c = ncol(df, k)
        if c: ren[c]=std
    df = df.rename(columns=ren)
    for c in [v for v in ren.values()]:
        df[c] = safe_num(df[c])
    return df.sort_values("year")

def load_unit_costs(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    svc = ncol(df, "service")
    uc  = ncol(df, "unit_cost","unitcost")
    if not (svc and uc): raise ValueError("unit_costs.csv needs service, unit_cost.")
    df = df.rename(columns={svc:"service", uc:"unit_cost"})
    df["service"] = df["service"].str.strip().str.lower()
    df["unit_cost"] = safe_num(df["unit_cost"]).clip(lower=0)
    return df

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    yr = ncol(df, "year","date")
    if not yr: raise ValueError("prices.csv needs year/date.")
    df = df.rename(columns={yr:"year"})
    if not np.issubdtype(df["year"].dtype, np.number):
        df["year"] = to_year(df["year"])
    ren = {}
    for k,std in [("cpi_health_idx","cpi_health_idx"),("cpi_all_idx","cpi_all_idx")]:
        c = ncol(df, k)
        if c: ren[c]=std
    df = df.rename(columns=ren)
    for c in ren.values():
        df[c] = safe_num(df[c])
    return df.sort_values("year")

def load_migration(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    yr = ncol(df, "year","date")
    age= ncol(df, "age_band","age","age_group")
    mig= ncol(df, "net_migration","migration","net")
    if not (yr and age and mig): raise ValueError("migration.csv needs year/date, age(age_band), net_migration.")
    df = df.rename(columns={yr:"year", age:"age_band", mig:"net_migration"})
    if not np.issubdtype(df["year"].dtype, np.number):
        df["year"] = to_year(df["year"])
    ages = df["age_band"].apply(parse_age_band)
    df["age_min"] = [a[0] for a in ages]; df["age_max"] = [a[1] for a in ages]
    df["age_mid"] = df.apply(lambda r: age_mid(r["age_min"], r["age_max"]), axis=1)
    df["net_migration"] = safe_num(df["net_migration"])
    return df.sort_values(["year","age_min"])

# ----------------------------- metrics -----------------------------

def oadr_psr(pop_tidy: pd.DataFrame) -> pd.DataFrame:
    df = pop_tidy.groupby(["year"], as_index=False).apply(lambda g: pd.Series({
        "pop_total": g["population"].sum(),
        "pop_0_14": g[(g["age_min"]<=14)].assign(cut=np.minimum(g["age_max"],14)-g["age_min"]+1).pipe(lambda x: x["population"].sum()),
        "pop_15_64": g[(g["age_min"]<=64) & (g["age_max"]>=15)].assign(cut=1).pipe(lambda x: x["population"].sum()),
        "pop_65p": g[(g["age_min"]>=65) | ((g["age_min"]<=65) & (g["age_max"]>=65))].assign(cut=1).pipe(lambda x: x["population"].sum()),
    })).reset_index(drop=True)
    df["oadr"] = df["pop_65p"] / df["pop_15_64"].replace(0,np.nan)
    df["psr"]  = df["pop_15_64"] / df["pop_65p"].replace(0,np.nan)
    return df[["year","pop_total","pop_0_14","pop_15_64","pop_65p","oadr","psr"]]

def current_demand(pop_tidy: pd.DataFrame, util: pd.DataFrame) -> pd.DataFrame:
    if util.empty: return pd.DataFrame()
    g = pop_tidy.merge(util, on=["age_band","age_min","age_max","age_mid"], how="left")
    out = g.groupby("year", as_index=False).apply(lambda x: pd.Series({
        "op_visits": (x["population"] * x.get("op_visits_pc",0)).sum(),
        "admissions": (x["population"] * x.get("admissions_per_1k",0) / 1000.0).sum(),
        "bed_days": (x["population"] * x.get("bed_days_pc",0)).sum(),
        "ltc_days": (x["population"] * x.get("ltc_days_pc",0)).sum() if "ltc_days_pc" in x else 0.0,
    })).reset_index(drop=True)
    return out

# ----------------------------- projections -----------------------------

def build_age_bands(pop: pd.DataFrame) -> List[Tuple[int,int]]:
    bands = sorted(set((int(a), int(b)) for a,b in zip(pop["age_min"], pop["age_max"])))
    return bands

def align_survival_to_bands(bands: List[Tuple[int,int]], survival: pd.DataFrame, uplift_pp: float) -> Dict[Tuple[int,int], float]:
    out = {}
    if survival.empty:
        # default survival curve: crude stylized (very rough)
        for (a0,a1) in bands:
            mid = (a0+a1)/2
            if mid < 1: s = 0.995
            elif mid < 15: s = 0.999
            elif mid < 45: s = 0.998
            elif mid < 65: s = 0.995
            elif mid < 80: s = 0.985
            else: s = 0.960
            out[(a0,a1)] = np.clip(s + uplift_pp/100.0, 0.0, 1.0)
        return out
    # map by nearest mid-age
    for (a0,a1) in bands:
        mid = (a0+a1)/2
        i = (survival["age_mid"]-mid).abs().argmin()
        s = float(survival.iloc[i]["survival_prob"])
        out[(a0,a1)] = np.clip(s + uplift_pp/100.0, 0.0, 1.0)
    return out

def births_from_fertility(pop_year: pd.DataFrame, fert: pd.DataFrame) -> float:
    """Compute births using ASFR (per woman). If fert empty, return nan to signal fallback."""
    if fert.empty or "sex" not in pop_year.columns:
        return np.nan
    # female pop in 15–49 bands
    females = pop_year[pop_year.get("sex","").astype(str).str.lower().isin(["f","female"])]
    if females.empty: return np.nan
    females = females.merge(fert[["age_band","asfr"]], on="age_band", how="left")
    females["asfr"] = females["asfr"].fillna(0.0)
    births = float((females["population"] * females["asfr"]).sum())
    return max(0.0, births)

def crude_births(pop_total: float, cbr_per_1000: float) -> float:
    return max(0.0, pop_total * (cbr_per_1000 / 1000.0))

def project_population(pop: pd.DataFrame,
                       survival: pd.DataFrame,
                       fert: pd.DataFrame,
                       migration: pd.DataFrame,
                       life_uplift_pp: float,
                       cbr_per_1000: float,
                       horizon: int) -> pd.DataFrame:
    """
    Annual step projection on age bands. Approximates 5y-band transitions by annual survival applied to band mass,
    then "age shift" by moving a fraction 1/w to next band (w=band width).
    """
    base_year = int(pop["year"].max())
    bands = build_age_bands(pop)
    # band widths
    widths = {b: (b[1]-b[0]+1) for b in bands}
    # survival per band (annual)
    svmap = align_survival_to_bands(bands, survival, uplift_pp=life_uplift_pp)
    # organize current distribution as dict
    def band_key(row): return (int(row["age_min"]), int(row["age_max"]))
    hist = pop.copy()
    out_rows = []
    # seed: last year distribution (aggregate sexes)
    y = base_year
    cur = hist[hist["year"]==y].groupby(["age_min","age_max"], as_index=False)["population"].sum()
    cur["age_band"] = cur.apply(lambda r: f"{int(r['age_min'])}-{int(r['age_max'])}" if r["age_max"]<120 else f"{int(r['age_min'])}+", axis=1)
    # loop years
    for t in range(1, horizon+1):
        y_next = y + t
        next_mass = {b: 0.0 for b in bands}
        # age shift
        for _, r in cur.iterrows():
            b = (int(r["age_min"]), int(r["age_max"]))
            mass = float(r["population"])
            w = widths[b]
            survive = svmap[b]
            # fraction aging out in one year ~ 1/w
            out_frac = 1.0 / w
            stay_frac = 1.0 - out_frac
            stay_mass = mass * survive * stay_frac
            move_mass = mass * survive * out_frac
            # add to same band and next band (if exists)
            next_mass[b] += stay_mass
            # find next band (by min age)
            bands_sorted = sorted(bands)
            try:
                idx = bands_sorted.index(b)
            except ValueError:
                idx = None
            if idx is not None and idx+1 < len(bands_sorted):
                bnext = bands_sorted[idx+1]
                next_mass[bnext] += move_mass
            else:
                # last band (e.g., 85+): accumulate there
                next_mass[b] += move_mass
        # births
        # try ASFR; else crude birth rate on total
        cur_full = cur.copy()
        cur_full["year"] = y_next-1
        # merge sex back if available from historical last year (approximation)
        # If no sex data, fall back to CBR.
        births = births_from_fertility(hist[hist["year"]==y], fert) if not fert.empty else np.nan
        if not (births==births):  # nan
            births = crude_births(sum(next_mass.values()), cbr_per_1000)
        # place births into first band
        b0 = min(bands, key=lambda x: x[0])
        next_mass[b0] += births
        # migration (if available for y_next)
        if not migration.empty and y_next in migration["year"].unique():
            migy = migration[migration["year"]==y_next]
            for _, m in migy.iterrows():
                b = (int(m["age_min"]), int(m["age_max"]))
                if b in next_mass:
                    next_mass[b] += float(m["net_migration"])
        # record
        for b in bands:
            out_rows.append({"year": y_next, "age_min": b[0], "age_max": b[1],
                             "age_band": (f"{b[0]}-{b[1]}" if b[1]<120 else f"{b[0]}+"),
                             "population": max(0.0, next_mass[b])})
        # update cur dataframe for next iteration
        cur = pd.DataFrame([{"age_min":b[0], "age_max":b[1], "population": next_mass[b]} for b in bands])
    proj = pd.DataFrame(out_rows).sort_values(["year","age_min"])
    proj["age_mid"] = proj.apply(lambda r: age_mid(r["age_min"], r["age_max"]), axis=1)
    return proj

# ----------------------------- demand/capacity/costs -----------------------------

def demand_from_projection(pop_proj: pd.DataFrame, util: pd.DataFrame, telemed_relief_pct: float) -> pd.DataFrame:
    if util.empty:
        return pd.DataFrame()
    g = pop_proj.merge(util, on=["age_band","age_min","age_max","age_mid"], how="left")
    # apply telemedicine relief on OP utilization
    if "op_visits_pc" in g.columns and telemed_relief_pct:
        g["op_visits_pc"] = g["op_visits_pc"] * (1.0 - telemed_relief_pct/100.0)
    out = g.groupby("year", as_index=False).apply(lambda x: pd.Series({
        "op_visits": (x["population"] * x.get("op_visits_pc",0)).sum(),
        "admissions": (x["population"] * x.get("admissions_per_1k",0) / 1000.0).sum(),
        "bed_days": (x["population"] * x.get("bed_days_pc",0)).sum(),
        "ltc_days": (x["population"] * x.get("ltc_days_pc",0)).sum() if "ltc_days_pc" in x else 0.0,
        "pop_total": x["population"].sum()
    })).reset_index(drop=True)
    return out

def capacity_paths(supply: pd.DataFrame,
                   base_year: int,
                   horizon: int,
                   doctor_cagr: float,
                   beds_cagr: float,
                   visits_per_doc: float,
                   target_occ: float) -> pd.DataFrame:
    years = list(range(base_year+1, base_year+horizon+1))
    rows = []
    # observed last values (or zeros)
    last = supply.sort_values("year").tail(1) if not supply.empty else pd.DataFrame()
    beds0 = float(last["beds_total"].iloc[0]) if ("beds_total" in last.columns and not last.empty) else np.nan
    docs0 = float(last["doctors_fte"].iloc[0]) if ("doctors_fte" in last.columns and not last.empty) else np.nan
    for i,y in enumerate(years, start=1):
        if not supply.empty and y in supply["year"].unique():
            r = supply[supply["year"]==y].iloc[0]
            beds = float(r.get("beds_total", np.nan))
            docs = float(r.get("doctors_fte", np.nan))
        else:
            beds = beds0 * ((1.0 + beds_cagr/100.0) ** i) if beds0==beds0 else np.nan
            docs = docs0 * ((1.0 + doctor_cagr/100.0) ** i) if docs0==docs0 else np.nan
        bed_days_cap = beds * 365.0 * float(target_occ) if beds==beds else np.nan
        doc_visit_cap = docs * float(visits_per_doc) if docs==docs else np.nan
        rows.append({"year": y, "beds_total": beds, "doctors_fte": docs,
                     "bed_days_capacity": bed_days_cap, "visit_capacity": doc_visit_cap})
    return pd.DataFrame(rows)

def spending_projection(demand: pd.DataFrame, unit_costs: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if demand.empty or unit_costs.empty:
        return pd.DataFrame()
    uc = unit_costs.set_index("service")["unit_cost"].to_dict()
    base_year = int(demand["year"].min())
    # health CPI index (base normalize to 1 at base_year if given)
    infl = pd.Series(1.0, index=demand["year"].unique())
    if not prices.empty and "cpi_health_idx" in prices.columns:
        p = prices.set_index("year")["cpi_health_idx"]
        if base_year in p.index:
            infl = p / p.loc[base_year]
        else:
            infl = p / float(p.iloc[0])
        infl = infl.reindex(demand["year"].unique()).fillna(method="ffill").fillna(method="bfill")
    rows = []
    for _, r in demand.iterrows():
        y = int(r["year"]); infl_y = float(infl.loc[y]) if y in infl.index else 1.0
        op_cost = float(r["op_visits"]) * uc.get("op_visit", 0.0) * infl_y
        adm_cost = float(r["admissions"]) * uc.get("admission", 0.0) * infl_y
        bed_cost = float(r["bed_days"]) * uc.get("bed_day", 0.0) * infl_y
        ltc_cost = float(r.get("ltc_days",0.0)) * uc.get("ltc_day", 0.0) * infl_y
        rows.append({"year": y, "spend_op": op_cost, "spend_admissions": adm_cost,
                     "spend_bed_days": bed_cost, "spend_ltc": ltc_cost,
                     "spend_total": op_cost + adm_cost + bed_cost + ltc_cost})
    return pd.DataFrame(rows).sort_values("year")

# ----------------------------- stress & alerts -----------------------------

def zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float); sd = x.std(ddof=0)
    return (x - x.mean()) / (sd if (sd and sd==sd) else np.nan)

def stress_index(demand: pd.DataFrame, capacity: pd.DataFrame, oaps: pd.DataFrame, thresh: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if demand.empty or capacity.empty or oaps.empty: return pd.DataFrame(), pd.DataFrame()
    df = demand.merge(capacity, on="year", how="left").merge(oaps[["year","oadr"]], on="year", how="left")
    df["occ_ratio"] = df["bed_days"] / df["bed_days_capacity"].replace(0,np.nan)
    df["phys_shortfall"] = (df["op_visits"] - df["visit_capacity"]).clip(lower=0.0)
    # z-blend
    Z_occ = zscore(df["occ_ratio"])
    Z_phys= zscore(df["phys_shortfall"])
    Z_oadr= zscore(df["oadr"])
    df["HSI"] = pd.concat([Z_occ, Z_phys, Z_oadr], axis=1).mean(axis=1, skipna=True)
    # alerts
    df = df.sort_values("year")
    df["d_HSI"] = df["HSI"].diff()
    P90 = np.nanpercentile(df["HSI"], 90) if df["HSI"].notna().sum()>=10 else np.nan
    rows = []
    for _, r in df.iterrows():
        cond1 = pd.notna(r["HSI"]) and r["HSI"] >= thresh
        cond2 = pd.notna(P90) and pd.notna(r["HSI"]) and r["HSI"] >= P90
        cond3 = pd.notna(r["d_HSI"]) and r["d_HSI"] > 0.5
        if cond1 or cond2 or cond3:
            rows.append({"year": int(r["year"]), "HSI": float(r["HSI"]),
                         "trigger_thresh": bool(cond1), "trigger_p90": bool(cond2),
                         "trigger_momentum": bool(cond3)})
    return df[["year","occ_ratio","phys_shortfall","oadr","HSI","d_HSI"]], pd.DataFrame(rows)

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    population: str
    survival: Optional[str]
    fertility: Optional[str]
    utilization: Optional[str]
    supply: Optional[str]
    unitcosts: Optional[str]
    prices: Optional[str]
    migration: Optional[str]
    horizon: int
    target_occ: float
    visits_per_doctor: float
    doctor_cagr: float
    beds_cagr: float
    cbr_per_1000: float
    life_uplift_pp: float
    telemed_relief_pct: float
    stress_thresh: float
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aging population → healthcare demand, capacity & scenarios")
    ap.add_argument("--population", required=True)
    ap.add_argument("--survival", default="")
    ap.add_argument("--fertility", default="")
    ap.add_argument("--utilization", default="")
    ap.add_argument("--supply", default="")
    ap.add_argument("--unitcosts", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--migration", default="")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--target_occ", type=float, default=0.85)
    ap.add_argument("--visits_per_doctor", type=float, default=4500.0)
    ap.add_argument("--doctor_cagr", type=float, default=2.0)
    ap.add_argument("--beds_cagr", type=float, default=2.0)
    ap.add_argument("--cbr_per_1000", type=float, default=16.0)
    ap.add_argument("--life_uplift_pp", type=float, default=0.0)
    ap.add_argument("--telemed_relief_pct", type=float, default=0.0)
    ap.add_argument("--stress_thresh", type=float, default=1.5)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_health")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    POP = load_population(args.population)
    SURV= load_survival(args.survival) if args.survival else pd.DataFrame()
    FERT= load_fertility(args.fertility) if args.fertility else pd.DataFrame()
    UTIL= load_utilization(args.utilization) if args.utilization else pd.DataFrame()
    SUP = load_supply(args.supply) if args.supply else pd.DataFrame()
    UC  = load_unit_costs(args.unitcosts) if args.unitcosts else pd.DataFrame()
    PRC = load_prices(args.prices) if args.prices else pd.DataFrame()
    MIG = load_migration(args.migration) if args.migration else pd.DataFrame()

    # Time filters
    if args.start:
        s = int(str(args.start)[:4])
        POP = POP[POP["year"] >= s]
        if not SUP.empty: SUP = SUP[SUP["year"] >= s]
        if not PRC.empty: PRC = PRC[PRC["year"] >= s]
        if not MIG.empty: MIG = MIG[MIG["year"] >= s]
    if args.end:
        e = int(str(args.end)[:4])
        POP = POP[POP["year"] <= e]
        if not SUP.empty: SUP = SUP[SUP["year"] <= e]
        if not PRC.empty: PRC = PRC[PRC["year"] <= e]
        if not MIG.empty: MIG = MIG[MIG["year"] <= e]

    POP.to_csv(outdir / "population_clean.csv", index=False)

    # Baselines
    OAPS = oadr_psr(POP)
    BASE_DEM = current_demand(POP, UTIL) if not UTIL.empty else pd.DataFrame()

    # Projections
    base_year = int(POP["year"].max())
    PROJ = project_population(POP, SURV, FERT, MIG, life_uplift_pp=float(args.life_uplift_pp),
                              cbr_per_1000=float(args.cbr_per_1000), horizon=int(args.horizon))
    PROJ.to_csv(outdir / "population_projection.csv", index=False)

    DEMAND = demand_from_projection(PROJ, UTIL, telemed_relief_pct=float(args.telemed_relief_pct)) if not UTIL.empty else pd.DataFrame()
    if not DEMAND.empty: DEMAND.to_csv(outdir / "demand_projection.csv", index=False)

    CAP = capacity_paths(SUP, base_year=base_year, horizon=int(args.horizon),
                         doctor_cagr=float(args.doctor_cagr), beds_cagr=float(args.beds_cagr),
                         visits_per_doc=float(args.visits_per_doctor), target_occ=float(args.target_occ))
    if not CAP.empty: CAP.to_csv(outdir / "capacity_projection.csv", index=False)

    SPEND = spending_projection(DEMAND, UC, PRC) if (not DEMAND.empty and not UC.empty) else pd.DataFrame()
    if not SPEND.empty: SPEND.to_csv(outdir / "spending_projection.csv", index=False)

    # Stress & alerts (only on projection years)
    HSI, ALERTS = stress_index(DEMAND, CAP, OAPS.append(oadr_psr(PROJ), ignore_index=True), thresh=float(args.stress_thresh)) if (not DEMAND.empty and not CAP.empty) else (pd.DataFrame(), pd.DataFrame())
    if not HSI.empty: HSI.to_csv(outdir / "stress_index.csv", index=False)
    if not ALERTS.empty: ALERTS.to_csv(outdir / "alerts.csv", index=False)

    # Summary
    summary = {
        "base_year": base_year,
        "horizon_year": base_year + int(args.horizon),
        "latest_oadr": float(OAPS[OAPS["year"]==base_year]["oadr"].iloc[0]) if base_year in OAPS["year"].values else None,
        "population_65p_base": float(OAPS[OAPS["year"]==base_year]["pop_65p"].iloc[0]) if base_year in OAPS["year"].values else None,
        "has_utilization": bool(not UTIL.empty),
        "has_supply": bool(not SUP.empty),
        "has_prices": bool(not PRC.empty and not UC.empty),
        "files": {
            "population_clean": "population_clean.csv",
            "population_projection": "population_projection.csv",
            "demand_projection": "demand_projection.csv" if not DEMAND.empty else None,
            "capacity_projection": "capacity_projection.csv" if not CAP.empty else None,
            "spending_projection": "spending_projection.csv" if not SPEND.empty else None,
            "stress_index": "stress_index.csv" if not HSI.empty else None,
            "alerts": "alerts.csv" if not ALERTS.empty else None
        },
        "scenarios": {
            "life_uplift_pp": float(args.life_uplift_pp),
            "telemed_relief_pct": float(args.telemed_relief_pct),
            "doctor_cagr_pct": float(args.doctor_cagr),
            "beds_cagr_pct": float(args.beds_cagr),
            "target_occ": float(args.target_occ)
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        population=args.population, survival=(args.survival or None), fertility=(args.fertility or None),
        utilization=(args.utilization or None), supply=(args.supply or None),
        unitcosts=(args.unitcosts or None), prices=(args.prices or None), migration=(args.migration or None),
        horizon=int(args.horizon), target_occ=float(args.target_occ), visits_per_doctor=float(args.visits_per_doctor),
        doctor_cagr=float(args.doctor_cagr), beds_cagr=float(args.beds_cagr), cbr_per_1000=float(args.cbr_per_1000),
        life_uplift_pp=float(args.life_uplift_pp), telemed_relief_pct=float(args.telemed_relief_pct),
        stress_thresh=float(args.stress_thresh), start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Aging Population → Healthcare ==")
    print(f"Base year: {base_year} → Horizon: {base_year+int(args.horizon)}")
    if summary["latest_oadr"] is not None:
        print(f"OADR (65+/15–64) in base year: {summary['latest_oadr']:.2f}")
    if not DEMAND.empty and not CAP.empty:
        last = DEMAND.tail(1).merge(CAP.tail(1), on="year", suffixes=("",""))
        r = last.iloc[0]
        if pd.notna(r.get("bed_days_capacity", np.nan)):
            occ = r["bed_days"]/r["bed_days_capacity"] if r["bed_days_capacity"] else np.nan
            print(f"End-year occupancy ratio (demand/cap): {occ:.2f}")
        if pd.notna(r.get("visit_capacity", np.nan)):
            short = r["op_visits"] - r["visit_capacity"]
            print(f"End-year physician visit shortfall: {short:,.0f}")
    if not ALERTS.empty:
        print(f"Alerts flagged for years: {', '.join(map(str, ALERTS['year'].tolist()))}")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
