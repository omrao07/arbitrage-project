#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
polish_coal_transition.py — Poland power & heat transition model (coal → RES/gas/others)

What this does
--------------
Given capacity/fleet, demand, prices (fuels & ETS), RES CFs, interconnectors, capacity-market data,
and optional mines & district-heat inputs, this script builds a compact system model for Poland:

Core outputs
------------
- fleet_enriched.csv           Unit/fuel-level enrichment (age, inferred retirement, costs, CO2 intensities)
- retire_schedule.csv          Annual retirements by fuel/tech under the chosen scenario
- monthly_dispatch.csv         Monthly stack: demand, RES, imports, thermal by fuel, marginal price proxy
- emissions.csv                Monthly & annual emissions by fuel + totals (MtCO2)
- cost_stack.csv               Variable cost (fuel+ETS+VOM) by fuel over time (€/MWh el)
- capacity_margin.csv          Monthly capacity margin & LOLE proxy (very stylized)
- capacity_market_revenues.csv Annual € revenues for eligible units (if capacity_market provided)
- jobs_transition.csv          Mine employment path vs coal output proxy (if mines provided)
- heat_system.csv              District-heat mix & CO2 with simple fuel-switch scenarios (if heat provided)
- price_regression.csv         OLS: day-ahead price ~ ETS + gas + coal + net import + demand (if price provided)
- scenarios_out.csv            Impact summary under provided scenarios.csv
- summary.json                 Key KPIs (coal share↓, CO2↓, ETS bill, earliest shortfall, etc.)
- config.json                  Reproducibility dump

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--capacity capacity.csv   REQUIRED
  Columns (any subset; extras ignored)
    unit, plant, fuel, tech, capacity_mw, efficiency (0..1), heat_rate_mwhth_permwh_el,
    commission_year, retirement_year, chp(0/1), co2_intensity_t_per_mwh_el, avail_factor
    Notes: If both efficiency and heat_rate provided, efficiency wins. If neither, defaults by fuel are used.

--demand demand.csv       REQUIRED (monthly or daily; monthly preferred)
  Columns: date, demand_gwh (if load_mw provided, we convert daily→GWh)

--fuel_prices fuel_prices.csv   OPTIONAL (monthly; flexible)
  Columns: date, series, value
  Recognized series (case-insensitive): ETS_EUR_T, COAL_USD_T, COAL_EUR_T, GAS_TTF_EUR_MWH, OIL_EUR_MWHth, BIOMASS_EUR_MWHth,
                                       PLNUSD, USDEUR, EURPLN, USDEUR, EURUSD
  Notes: If only USD coal is provided, we use USDEUR/EURUSD to convert. Coal energy content is configurable.

--res_cf res_cf.csv       OPTIONAL (monthly)
  Columns: date, fuel (wind/solar/hydro/biomass), capacity_factor (0..1) or generation_gwh

--gen_actual gen.csv      OPTIONAL (monthly actuals to anchor/compare)
  Columns: date, fuel, generation_gwh

--interconnectors interconnectors.csv OPTIONAL
  Columns: date, net_import_gwh, cap_mw

--capacity_market cm.csv  OPTIONAL (annual)
  Columns: year, eligible_fuel, capacity_mw, price_eur_kw_yr

--mines mines.csv         OPTIONAL
  Columns: date(optional), mine, output_mt, employees, region(optional)

--heat heat.csv           OPTIONAL (district heating)
  Columns: date, heat_demand_gwh_th, coal_share, gas_share, biomass_share, others_share

--pipeline pipeline.csv   OPTIONAL (capacity additions/retirements ex-ante)
  Columns: date, fuel, add_mw (positive for additions, negative for permanent closures)

--prices prices.csv       OPTIONAL (daily or monthly spot price)
  Columns: date, price_eur_mwh

--scenarios scenarios.csv OPTIONAL
  Columns: scenario, name, key, value
  Supported keys (examples):
    ets.multiplier = 1.25
    coal.energy_content_mwh_per_t = 23.5
    retire.delta_year = -3
    retire.fast.hard_coal_year = 2032
    retire.fast.lignite_year   = 2030
    avail.coal = 0.85
    avail.gas = 0.90
    res.mult.wind = 1.15
    res.mult.solar = 1.25
    demand.electrify_pct_per_year = 2.0
    gas.price_mult = 1.3
    imports.cap_mult = 0.9
    nuclear.start_year = 2033
    nuclear.cap_mw = 3600
    cm.coal_exit_year = 2025
    heat.switch.coal_to_gas_pct = 15
    heat.switch.coal_to_hp_pct  = 10

Key options
-----------
--start 2018-01
--end   2035-12
--coal_energy_content_mwh_per_t 24.0
--coal_vom_eur_mwh 3.0
--gas_vom_eur_mwh  2.0
--oil_vom_eur_mwh  3.0
--biomass_vom_eur_mwh 4.0
--defaults_json '{"eff":{"lignite":0.35,"hard_coal":0.38,"ccgt":0.54,"ocgt":0.35,"biomass":0.28,"oil":0.38,"waste":0.25}, "co2":{"lignite":1.10,"hard_coal":0.90,"ccgt":0.36,"ocgt":0.56,"oil":0.74,"biomass":0.00,"waste":0.40}, "avail":{"lignite":0.80,"hard_coal":0.80,"ccgt":0.90,"ocgt":0.85,"biomass":0.85,"oil":0.80,"nuclear":0.90}}'
--scenario baseline
--outdir out_pl_coal_transition

Notes & caveats
---------------
- Merit-order is computed at *fuel/tech bucket* level (not per-unit), using variable cost = fuel/eff + ETS*CO2 + VOM.
- If res_cf has generation_gwh we use it; else capacity_factor × capacity drives RES output.
- Imports are exogenous via net_import_gwh (clipped by capacity if provided).
- Capacity margin & LOLE proxy are stylized (energy margin vs estimated monthly peak).
- Biomass accounted with 0 tCO2/MWh_el by default (policy-dependent; adjust via --defaults_json).
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

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def to_date_any(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def year(d: pd.Timestamp) -> int:
    return int(pd.to_datetime(d).year)

def month_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.period_range(pd.to_datetime(start).to_period("M"), pd.to_datetime(end).to_period("M"), freq="M").to_timestamp()

def lower(s: str) -> str:
    return str(s).strip().lower() if isinstance(s, str) else str(s)

def fuel_norm(s: str) -> str:
    if s is None: return "other"
    x = lower(s)
    if any(k in x for k in ["lignite", "brown"]): return "lignite"
    if any(k in x for k in ["hard coal", "coal", "stone"]): return "hard_coal"
    if "ccgt" in x or ("gas" in x and "ccgt" in x): return "ccgt"
    if "ocgt" in x or ("gas" in x and "ccgt" not in x): return "ocgt"
    if "gas" in x: return "ccgt"
    if "oil" in x or "diesel" in x: return "oil"
    if "bio" in x: return "biomass"
    if "waste" in x: return "waste"
    if any(k in x for k in ["wind"]): return "wind"
    if any(k in x for k in ["solar","pv","photovoltaic"]): return "solar"
    if "hydro" in x: return "hydro"
    if "nuclear" in x: return "nuclear"
    return "other"

def safe_div(a, b):
    try:
        return float(a) / float(b) if float(b) != 0 else np.nan
    except Exception:
        return np.nan

def load_csv(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    return pd.read_csv(path)

def merge_on_date(df_left: pd.DataFrame, df_right: pd.DataFrame, how="left") -> pd.DataFrame:
    if df_left.empty: return df_left
    if df_right.empty: return df_left
    if "date" not in df_left.columns or "date" not in df_right.columns: return df_left
    return df_left.merge(df_right, on="date", how=how)


# ----------------------------- loaders -----------------------------

def load_capacity(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"unit") or ncol(df,"id") or df.columns[0]): "unit",
        (ncol(df,"plant") or "plant"): "plant",
        (ncol(df,"fuel") or "fuel"): "fuel",
        (ncol(df,"tech") or "tech"): "tech",
        (ncol(df,"capacity_mw") or ncol(df,"net_mw") or ncol(df,"gross_mw") or "capacity_mw"): "capacity_mw",
        (ncol(df,"efficiency") or "efficiency"): "efficiency",
        (ncol(df,"heat_rate_mwhth_permwh_el") or ncol(df,"heat_rate") or "heat_rate_mwhth_permwh_el"): "heat_rate",
        (ncol(df,"commission_year") or ncol(df,"year") or "commission_year"): "commission_year",
        (ncol(df,"retirement_year") or ncol(df,"closure_year") or "retirement_year"): "retirement_year",
        (ncol(df,"chp") or "chp"): "chp",
        (ncol(df,"co2_intensity_t_per_mwh_el") or ncol(df,"co2_intensity") or "co2_intensity_t_per_mwh_el"): "co2_intensity_t_per_mwh_el",
        (ncol(df,"avail_factor") or "avail_factor"): "avail_factor",
    }
    df = df.rename(columns=ren)
    for c in ["capacity_mw","efficiency","heat_rate","commission_year","retirement_year","co2_intensity_t_per_mwh_el","avail_factor","chp"]:
        if c in df.columns: df[c] = num(df[c])
    df["fuel"] = df.get("fuel","").apply(fuel_norm)
    df["tech"] = df.get("tech","").astype(str)
    df["unit"] = df.get("unit","").astype(str)
    return df

def load_demand(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = (ncol(df,"date") or df.columns[0])
    df = df.rename(columns={date_c:"date"})
    df["date"] = to_date(df["date"])
    if ncol(df,"demand_gwh"):
        df["demand_gwh"] = num(df[ncol(df,"demand_gwh")])
    elif ncol(df,"load_mw"):
        # if daily, convert to monthly by sum(load*24)/1000
        df = df.sort_values("date")
        df["demand_gwh"] = num(df[ncol(df,"load_mw")]) * 24.0 / 1000.0
        df = df.groupby(df["date"].dt.to_period("M")).agg(demand_gwh=("demand_gwh","sum")).reset_index()
        df["date"] = df["date"].dt.to_timestamp()
    else:
        raise ValueError("demand.csv must include demand_gwh or load_mw")
    return df[["date","demand_gwh"]].sort_values("date")

def load_fuel_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={(ncol(df,"date") or df.columns[0]):"date", (ncol(df,"series") or "series"):"series", (ncol(df,"value") or "value"):"value"})
    df["date"] = to_date(df["date"])
    df["series"] = df["series"].astype(str)
    df["value"] = num(df["value"])
    wide = df.pivot_table(index="date", columns=df["series"].str.upper(), values="value", aggfunc="mean").reset_index()
    return wide

def load_res_cf(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"fuel") or "fuel"):"fuel",
           (ncol(df,"capacity_factor") or "capacity_factor"):"capacity_factor",
           (ncol(df,"generation_gwh") or "generation_gwh"):"generation_gwh"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["fuel"] = df["fuel"].apply(fuel_norm)
    for c in ["capacity_factor","generation_gwh"]:
        if c in df.columns: df[c] = num(df[c])
    return df.sort_values(["fuel","date"])

def load_gen_actual(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"fuel") or "fuel"):"fuel",
           (ncol(df,"generation_gwh") or "generation_gwh"):"generation_gwh"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["fuel"] = df["fuel"].apply(fuel_norm)
    df["generation_gwh"] = num(df["generation_gwh"])
    return df

def load_intercon(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"net_import_gwh") or "net_import_gwh"):"net_import_gwh",
           (ncol(df,"cap_mw") or "cap_mw"):"cap_mw"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    for c in ["net_import_gwh","cap_mw"]:
        if c in df.columns: df[c] = num(df[c])
    return df.sort_values("date")

def load_capacity_market(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"year") or df.columns[0]):"year",
           (ncol(df,"eligible_fuel") or "eligible_fuel"):"eligible_fuel",
           (ncol(df,"capacity_mw") or "capacity_mw"):"capacity_mw",
           (ncol(df,"price_eur_kw_yr") or "price_eur_kw_yr"):"price_eur_kw_yr"}
    df = df.rename(columns=ren)
    for c in ["year","capacity_mw","price_eur_kw_yr"]:
        if c in df.columns: df[c] = num(df[c])
    df["eligible_fuel"] = df.get("eligible_fuel","").apply(fuel_norm)
    return df

def load_mines(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"mine") or "mine"):"mine",
           (ncol(df,"output_mt") or "output_mt"):"output_mt",
           (ncol(df,"employees") or "employees"):"employees",
           (ncol(df,"region") or "region"):"region"}
    df = df.rename(columns=ren)
    if "date" in df.columns:
        df["date"] = to_date(df["date"])
    for c in ["output_mt","employees"]:
        if c in df.columns: df[c] = num(df[c])
    return df

def load_heat(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"heat_demand_gwh_th") or "heat_demand_gwh_th"):"heat_demand_gwh_th",
           (ncol(df,"coal_share") or "coal_share"):"coal_share",
           (ncol(df,"gas_share") or "gas_share"):"gas_share",
           (ncol(df,"biomass_share") or "biomass_share"):"biomass_share",
           (ncol(df,"others_share") or "others_share"):"others_share"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    for c in ["heat_demand_gwh_th","coal_share","gas_share","biomass_share","others_share"]:
        if c in df.columns: df[c] = num(df[c])
    return df

def load_pipeline(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"fuel") or "fuel"):"fuel",
           (ncol(df,"add_mw") or "add_mw"):"add_mw"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["fuel"] = df["fuel"].apply(fuel_norm)
    df["add_mw"] = num(df["add_mw"])
    return df

def load_prices_spot(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    date_c = (ncol(df,"date") or df.columns[0])
    df = df.rename(columns={date_c:"date", (ncol(df,"price_eur_mwh") or ncol(df,"price") or "price"):"price_eur_mwh"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # Map to month
    m = df.copy()
    m["date"] = m["date"].dt.to_period("M").dt.to_timestamp()
    m = m.groupby("date", as_index=False).agg(price_eur_mwh=("price_eur_mwh","mean"))
    return m

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"):"scenario",
           (ncol(df,"name") or "name"):"name",
           (ncol(df,"key") or "key"):"key",
           (ncol(df,"value") or "value"):"value"}
    return df.rename(columns=ren)


# ----------------------------- enrichment -----------------------------

def apply_defaults_to_fleet(fleet: pd.DataFrame, defaults: Dict[str,Dict[str,float]]) -> pd.DataFrame:
    df = fleet.copy()
    effd = defaults.get("eff", {})
    co2d = defaults.get("co2", {})
    avld = defaults.get("avail", {})
    # Efficiency from heat_rate if missing
    if "efficiency" not in df.columns: df["efficiency"] = np.nan
    if "heat_rate" not in df.columns: df["heat_rate"] = np.nan
    df["efficiency"] = df["efficiency"].where(df["efficiency"].notna(), 1.0 / (df["heat_rate"].replace(0, np.nan)))
    # Fuel defaults
    def eff_f(f):
        return effd.get(f, np.nan)
    def co2_f(f):
        return co2d.get(f, np.nan)
    def avl_f(f):
        return avld.get(f, np.nan)
    df["efficiency"] = df.apply(lambda r: r["efficiency"] if pd.notna(r["efficiency"]) and r["efficiency"]>0 else eff_f(r["fuel"]), axis=1)
    df["co2_intensity_t_per_mwh_el"] = df.apply(lambda r: r["co2_intensity_t_per_mwh_el"] if pd.notna(r.get("co2_intensity_t_per_mwh_el", np.nan)) else co2_f(r["fuel"]), axis=1)
    df["avail_factor"] = df.apply(lambda r: r["avail_factor"] if pd.notna(r.get("avail_factor", np.nan)) else avl_f(r["fuel"]), axis=1)
    # Age
    df["commission_year"] = df.get("commission_year", np.nan)
    df["retirement_year"] = df.get("retirement_year", np.nan)
    return df

def infer_retirements(fleet: pd.DataFrame, scenario: str, scenarios_df: pd.DataFrame, start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build retirement year per unit with scenario overrides. Returns (fleet2, retire_schedule).
    """
    df = fleet.copy()
    # Defaults: heuristic max ages
    max_age = {"lignite": 45, "hard_coal": 45, "ccgt": 35, "ocgt": 35, "biomass": 35, "oil": 35, "waste": 30, "nuclear": 60, "other": 30, "wind": 25, "solar": 30, "hydro": 60}
    # Scenario overrides
    def scen_get(key, default=None):
        sub = scenarios_df[scenarios_df["scenario"]==scenario]
        if sub.empty: return default
        row = sub[sub["key"].str.strip().str.lower()==key.lower()]
        if row.empty: return default
        try:
            return float(row["value"].iloc[0])
        except Exception:
            try:
                return str(row["value"].iloc[0])
            except Exception:
                return default
    delta_year = scen_get("retire.delta_year", 0)
    fast_hc = scen_get("retire.fast.hard_coal_year", None)
    fast_lig = scen_get("retire.fast.lignite_year", None)
    # Compute retirement year per unit
    def retire_year(r):
        ry = r.get("retirement_year", np.nan)
        if pd.notna(ry) and ry>0:
            base = int(ry)
        else:
            cy = int(r.get("commission_year", np.nan)) if pd.notna(r.get("commission_year", np.nan)) else 1990
            base = cy + max_age.get(r["fuel"], 35)
        # fast tracks
        if r["fuel"]=="hard_coal" and fast_hc: base = min(base, int(fast_hc))
        if r["fuel"]=="lignite" and fast_lig: base = min(base, int(fast_lig))
        base = int(base + (delta_year or 0))
        return base
    df["retire_year_scn"] = df.apply(retire_year, axis=1)
    # Retirement schedule (annual MW closing)
    sched = (df.groupby(["retire_year_scn","fuel"], as_index=False)
               .agg(retire_mw=("capacity_mw","sum"))
               .rename(columns={"retire_year_scn":"year"}))
    sched = sched[(sched["year"]>=year(pd.to_datetime(start))) & (sched["year"]<=year(pd.to_datetime(end)))]
    return df, sched.sort_values(["year","fuel"])

def capacity_time_series(fleet: pd.DataFrame, pipeline: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    months = month_range(start, end)
    fuels = sorted(fleet["fuel"].unique().tolist() + (pipeline["fuel"].unique().tolist() if not pipeline.empty else []))
    # Base installed capacity at start from fleet with commission_year<=start.year
    base = (fleet[fleet["commission_year"].fillna(1900) <= year(months[0])]
                .groupby("fuel", as_index=False).agg(capacity_mw=("capacity_mw","sum")))
    df = pd.DataFrame({"date": np.repeat(months, len(fuels)), "fuel": fuels * len(months)})
    df = df.merge(base, on="fuel", how="left").fillna({"capacity_mw": 0.0})
    # Apply retirements monthly
    for _, r in fleet.iterrows():
        ry = int(r["retire_year_scn"])
        if ry <= year(end):
            retire_month = pd.to_datetime(f"{ry}-12-01").to_period("M").to_timestamp()
            df.loc[(df["fuel"]==r["fuel"]) & (df["date"]>=retire_month), "capacity_mw"] -= r["capacity_mw"]
    # Apply pipeline additions
    if not pipeline.empty:
        for _, r in pipeline.iterrows():
            d0 = r["date"]
            if pd.isna(d0): continue
            df.loc[(df["fuel"]==r["fuel"]) & (df["date"]>=d0), "capacity_mw"] += float(r["add_mw"])
    # Clip negatives
    df["capacity_mw"] = df["capacity_mw"].clip(lower=0.0)
    return df

def res_generation(res_cf: pd.DataFrame, cap_ts: pd.DataFrame) -> pd.DataFrame:
    if res_cf.empty:
        return pd.DataFrame()
    # If generation given, keep; else use CF × capacity
    gen = res_cf.copy()
    if "generation_gwh" not in gen.columns or gen["generation_gwh"].isna().all():
        # Map CF to capacity
        cap = cap_ts.copy()
        cap = cap[cap["fuel"].isin(["wind","solar","hydro","biomass"])]
        gen = gen.merge(cap, on=["date","fuel"], how="left")
        gen["generation_gwh"] = gen["capacity_factor"].fillna(0) * gen["capacity_mw"].fillna(0) * (24*gen["date"].dt.days_in_month/1000.0)
    return gen.groupby(["date","fuel"], as_index=False)["generation_gwh"].sum()

def thermal_costs_table(fuel_prices: pd.DataFrame, defaults: Dict[str,Dict[str,float]],
                        coal_energy_content_mwh_per_t: float,
                        vom: Dict[str,float],
                        scen_mult: Dict[str,float]) -> pd.DataFrame:
    """
    Build monthly variable cost per fuel (€/MWh el): fuel/eff + ETS*CO2 + VOM
    """
    if fuel_prices.empty:
        # Create empty shell with date index to be merged later
        return pd.DataFrame()
    p = fuel_prices.copy()
    # Extract prices with flexible names
    def pick(*names):
        for n in names:
            if n.upper() in p.columns: return n.upper()
        return None
    ETS = pick("ETS_EUR_T","CO2_EUR_T","EUA_EUR_T")
    COAL_EUR = pick("COAL_EUR_T")
    COAL_USD = pick("COAL_USD_T","API2_USD_T")
    EURUSD = pick("EURUSD","USDEUR")
    GAS = pick("GAS_TTF_EUR_MWH","TTF_EUR_MWH","GAS_EUR_MWH")
    OIL = pick("OIL_EUR_MWHTH","OIL_EUR_MWH","FO_EUR_MWH")
    BIOM = pick("BIOMASS_EUR_MWHTH","BIOMASS_EUR_MWH")
    out = pd.DataFrame({"date": p["date"]})
    # Build price series in €/MWhth for each fuel where needed
    # Coal: €/t → €/MWhth using energy content
    coal_eur_t = None
    if COAL_EUR:
        coal_eur_t = p[COAL_EUR]
    elif COAL_USD and EURUSD:
        # If EURUSD provided (EUR per USD), or USDEUR (USD per EUR)
        if "EURUSD" in p.columns:
            coal_eur_t = p[COAL_USD] * p["EURUSD"]
        elif "USDEUR" in p.columns:
            coal_eur_t = p[COAL_USD] / p["USDEUR"]
    if coal_eur_t is not None:
        coal_eur_mwhth = coal_eur_t / float(coal_energy_content_mwh_per_t)
    else:
        coal_eur_mwhth = pd.Series(np.nan, index=p.index)
    gas_eur_mwhth = p[GAS] if GAS else pd.Series(np.nan, index=p.index)
    oil_eur_mwhth = p[OIL] if OIL else pd.Series(np.nan, index=p.index)
    biom_eur_mwhth = p[BIOM] if BIOM else pd.Series(np.nan, index=p.index)
    ets_eur_t = p[ETS] * scen_mult.get("ets.multiplier", 1.0) if ETS else pd.Series(80.0, index=p.index)  # default fallback

    # Compose variable cost per el-MWh by fuel
    def vcost(fuel: str, eff: float, co2: float, fuel_price_mwhth: pd.Series, vom_add: float, fuel_mult_key: Optional[str]) -> pd.Series:
        effv = eff if eff and eff>0 else defaults["eff"].get(fuel, 0.38)
        co2v = co2 if co2 is not None else defaults["co2"].get(fuel, 0.9)
        pmult = scen_mult.get(fuel_mult_key, 1.0) if fuel_mult_key else 1.0
        fp = (fuel_price_mwhth * pmult) if fuel_price_mwhth is not None else pd.Series(np.nan, index=p.index)
        fuel_term = fp / max(effv, 1e-6)
        ets_term = ets_eur_t * float(co2v)
        return fuel_term + ets_term + float(vom_add)

    out["VC_hard_coal"] = vcost("hard_coal", defaults["eff"]["hard_coal"], defaults["co2"]["hard_coal"], coal_eur_mwhth, vom["coal"], "coal.price_mult")
    out["VC_lignite"]   = vcost("lignite",   defaults["eff"]["lignite"],   defaults["co2"]["lignite"],   coal_eur_mwhth*0.8, vom["coal"], "coal.price_mult")  # lignite cheaper energy; rough 0.8 factor
    out["VC_ccgt"]      = vcost("ccgt",      defaults["eff"]["ccgt"],      defaults["co2"]["ccgt"],      gas_eur_mwhth*scen_mult.get("gas.price_mult",1.0), vom["gas"], "gas.price_mult")
    out["VC_ocgt"]      = vcost("ocgt",      defaults["eff"]["ocgt"],      defaults["co2"]["ocgt"],      gas_eur_mwhth*scen_mult.get("gas.price_mult",1.0), vom["gas"], "gas.price_mult")
    out["VC_oil"]       = vcost("oil",       defaults["eff"]["oil"],       defaults["co2"]["oil"],       oil_eur_mwhth, vom["oil"], None)
    out["VC_biomass"]   = vcost("biomass",   defaults["eff"]["biomass"],   defaults["co2"]["biomass"],   biom_eur_mwhth, vom["biomass"], None)
    out["ETS_EUR_T"] = ets_eur_t
    return out

def merit_order_dispatch(demand_m: pd.DataFrame,
                         cap_ts: pd.DataFrame,
                         res_gen: pd.DataFrame,
                         intercon: pd.DataFrame,
                         costs: pd.DataFrame,
                         defaults: Dict[str,Dict[str,float]],
                         scenarios_df: pd.DataFrame,
                         scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly dispatch at fuel-bucket granularity.
    Returns (dispatch_table, cost_stack table).
    """
    # Availability overrides from scenario
    sub = scenarios_df[scenarios_df["scenario"]==scenario]
    avl_over = {}
    for fuel in ["coal","hard_coal","lignite","gas","ccgt","ocgt","biomass"]:
        key = f"avail.{fuel}"
        row = sub[sub["key"].str.lower()==key]
        if not row.empty:
            avl_over[fuel] = float(row["value"].iloc[0])

    # Compose monthly table
    base = demand_m.copy()
    base = merge_on_date(base, intercon.rename(columns={"net_import_gwh":"imports_gwh"}) if not intercon.empty else pd.DataFrame())
    base["imports_gwh"] = base.get("imports_gwh", 0.0).fillna(0.0)
    # Cap imports by capacity if provided
    if not intercon.empty and "cap_mw" in intercon.columns:
        cap_m = intercon.copy()
        cap_m["max_imp_gwh"] = intercon["cap_mw"] * 0.9 * (24*intercon["date"].dt.days_in_month/1000.0)
        base = base.merge(cap_m[["date","max_imp_gwh"]], on="date", how="left")
        base["imports_gwh"] = base.apply(lambda r: min(r["imports_gwh"], r["max_imp_gwh"]) if pd.notna(r.get("max_imp_gwh", np.nan)) else r["imports_gwh"], axis=1)

    # RES
    res = res_gen.copy()
    res_pivot = res.pivot_table(index="date", columns="fuel", values="generation_gwh", aggfunc="sum").fillna(0.0)
    res_pivot = res_pivot.reindex(base["date"]).fillna(0.0)
    # Capacity by fuel & availability → available energy headroom per month
    cap = cap_ts.copy()
    cap["avail_factor"] = cap["fuel"].map(lambda f: avl_over.get(f, defaults["avail"].get(f, 0.85)))
    cap["energy_cap_gwh"] = cap["capacity_mw"] * cap["avail_factor"] * (24*cap["date"].dt.days_in_month/1000.0)
    cap_month = cap.pivot_table(index="date", columns="fuel", values="energy_cap_gwh", aggfunc="sum").fillna(0.0)
    # Merge costs (€/MWh) by fuel
    C = costs.copy()
    cost_cols = [c for c in C.columns if c.startswith("VC_")]
    C = C.set_index("date")[cost_cols].rename(columns={
        "VC_hard_coal":"hard_coal","VC_lignite":"lignite","VC_ccgt":"ccgt","VC_ocgt":"ocgt","VC_oil":"oil","VC_biomass":"biomass"
    })
    C = C.reindex(base["date"]).fillna(method="ffill").fillna(method="bfill")

    rows = []
    price_rows = []
    fuels_order = ["nuclear","hydro","biomass","wind","solar","lignite","hard_coal","ccgt","ocgt","oil","waste","other"]
    for i, row in base.iterrows():
        d = row["date"]
        demand = float(row["demand_gwh"])
        imports = float(row.get("imports_gwh", 0.0))
        # non-dispatchable RES & must-runs
        res_this = res_pivot.loc[d] if d in res_pivot.index else pd.Series()
        must = 0.0
        part = {}
        for f in ["wind","solar","hydro","biomass","nuclear","waste","other"]:
            g = float(res_this.get(f, 0.0))
            part[f] = g
            must += g
        # net native demand to serve
        net = demand - imports - must
        if net < 0:
            # curtail RES (uniformly) if over-supplied; simple approach
            scale = clamp(demand / max(imports + must, 1e-6), 0.0, 1.0)
            for f in ["wind","solar","hydro","biomass","nuclear","waste","other"]:
                part[f] *= scale
            net = 0.0

        # Available thermal energy caps this month
        cap_row = cap_month.loc[d] if d in cap_month.index else pd.Series()
        # Build stack among thermal fuels by variable cost ascending
        thermals = ["lignite","hard_coal","ccgt","ocgt","oil"]
        stack = []
        for f in thermals:
            cap_gwh = float(cap_row.get(f, 0.0))
            vc = float(C.loc[d, f]) if (d in C.index and f in C.columns) else (50.0 if f=="hard_coal" else 80.0)
            stack.append((f, vc, cap_gwh))
        stack.sort(key=lambda x: x[1])

        marginal_price = 0.0
        for f, vc, cap_gwh in stack:
            take = min(net, cap_gwh)
            if take > 0:
                part[f] = part.get(f, 0.0) + take
                net -= take
                marginal_price = vc  # last used cost
            if net <= 1e-6:
                break
        # If still unmet load → blackout proxy; count unserved energy
        unserved = max(net, 0.0)
        if unserved > 0:
            part["unserved"] = unserved
            marginal_price = max(marginal_price, 500.0)

        # Record row
        r = {"date": d, "demand_gwh": demand, "imports_gwh": imports, "unserved_gwh": unserved, "marginal_price_eur_mwh": marginal_price}
        for f in fuels_order + ["unserved"]:
            r[f"{f}_gwh"] = part.get(f, 0.0)
        rows.append(r)

        # Price stack snapshot
        price_rows.append({"date": d, **{f"VC_{k}": v for k, v in zip(C.columns, C.loc[d].values)}})

    disp = pd.DataFrame(rows).sort_values("date")
    cost_stack = pd.DataFrame(price_rows).sort_values("date")
    return disp, cost_stack

def emissions_from_dispatch(disp: pd.DataFrame, defaults: Dict[str,Dict[str,float]]) -> pd.DataFrame:
    if disp.empty: return pd.DataFrame()
    # Emission factors tCO2/MWh_el
    ef = defaults["co2"]
    fuels = ["lignite","hard_coal","ccgt","ocgt","oil","biomass","waste"]
    rows = []
    for _, r in disp.iterrows():
        d = r["date"]
        tot = 0.0
        rec = {"date": d}
        for f in fuels:
            g = float(r.get(f"{f}_gwh", 0.0))
            e = g * float(ef.get(f, 0.0))
            rec[f"{f}_mtco2"] = e / 1000.0
            tot += e / 1000.0
        rec["total_mtco2"] = tot
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("date")

def capacity_margin_table(cap_ts: pd.DataFrame, demand_m: pd.DataFrame, defaults: Dict[str,Dict[str,float]], intercon: pd.DataFrame) -> pd.DataFrame:
    if cap_ts.empty or demand_m.empty:
        return pd.DataFrame()
    cap = cap_ts.copy()
    cap["avail_factor"] = cap["fuel"].map(lambda f: defaults["avail"].get(f, 0.85))
    cap["cap_available_mw"] = cap["capacity_mw"] * cap["avail_factor"]
    sys = cap.groupby("date", as_index=False).agg(cap_available_mw=("cap_available_mw","sum"))
    # Add imports (capacity) if provided
    if not intercon.empty and "cap_mw" in intercon.columns:
        sys = sys.merge(intercon[["date","cap_mw"]].rename(columns={"cap_mw":"import_cap_mw"}), on="date", how="left")
        sys["cap_available_mw"] += sys["import_cap_mw"].fillna(0.0)
    # Peak demand ≈ monthly average load × 1.30 (stylized)
    d = demand_m.copy()
    d["avg_load_mw"] = d["demand_gwh"] * 1000.0 / (24*d["date"].dt.days_in_month)
    d["peak_mw"] = d["avg_load_mw"] * 1.30
    out = sys.merge(d[["date","peak_mw"]], on="date", how="left")
    out["capacity_margin_mw"] = out["cap_available_mw"] - out["peak_mw"]
    # LOLE proxy: hours at risk if margin<0 scaled by |margin|/peak
    out["lole_hours_proxy"] = np.where(out["capacity_margin_mw"]<0, 50.0 * (-out["capacity_margin_mw"]/out["peak_mw"]).clip(lower=0), 0.0)
    return out.sort_values("date")

def capacity_market_revenue(cm: pd.DataFrame, fleet: pd.DataFrame, scenario: str, scenarios_df: pd.DataFrame) -> pd.DataFrame:
    if cm.empty or fleet.empty:
        return pd.DataFrame()
    out = cm.copy()
    # Apply exit year for coal if specified
    sub = scenarios_df[scenarios_df["scenario"]==scenario]
    coal_exit = sub[sub["key"].str.lower()=="cm.coal_exit_year"]["value"].iloc[0] if not sub[sub["key"].str.lower()=="cm.coal_exit_year"].empty else None
    if coal_exit:
        out = out[~((out["eligible_fuel"].isin(["lignite","hard_coal"])) & (out["year"]>=float(coal_exit)))]
    out["revenue_eur_m"] = out["capacity_mw"] * out["price_eur_kw_yr"] * 1000.0 / 1e6
    return out.groupby("year", as_index=False).agg(revenue_eur_m=("revenue_eur_m","sum"))

def jobs_transition(mines: pd.DataFrame, disp: pd.DataFrame) -> pd.DataFrame:
    if mines.empty or disp.empty:
        return pd.DataFrame()
    # Proxy coal output ~ coal generation (hard + lignite) scaled
    coal_g = disp[["date","lignite_gwh","hard_coal_gwh"]].copy()
    coal_g["coal_gwh"] = coal_g[["lignite_gwh","hard_coal_gwh"]].sum(axis=1)
    # Normalize mines output & employees annually
    if "date" in mines.columns and mines["date"].notna().any():
        m = mines.copy()
        m["year"] = m["date"].dt.year
        g = coal_g.copy(); g["year"] = g["date"].dt.year
        g = g.groupby("year", as_index=False).agg(coal_gwh=("coal_gwh","sum"))
        out = (m.groupby("year", as_index=False).agg(output_mt=("output_mt","sum"), employees=("employees","sum"))
                 .merge(g, on="year", how="outer").sort_values("year"))
        # Employees path ~ base * coal_gwh_index (very crude)
        base_emp = out["employees"].dropna().iloc[0] if out["employees"].notna().any() else np.nan
        base_g = out["coal_gwh"].dropna().iloc[0] if out["coal_gwh"].notna().any() else np.nan
        if pd.notna(base_emp) and pd.notna(base_g):
            out["employees_proxy"] = base_emp * (out["coal_gwh"]/base_g)
        return out
    return pd.DataFrame()

def heat_switch(heat: pd.DataFrame, defaults: Dict[str,Dict[str,float]], scenarios_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if heat.empty:
        return pd.DataFrame()
    sub = scenarios_df[scenarios_df["scenario"]==scenario]
    coal_to_gas = float(sub[sub["key"].str.lower()=="heat.switch.coal_to_gas_pct"]["value"].iloc[0]) if not sub[sub["key"].str.lower()=="heat.switch.coal_to_gas_pct"].empty else 0.0
    coal_to_hp  = float(sub[sub["key"].str.lower()=="heat.switch.coal_to_hp_pct"]["value"].iloc[0]) if not sub[sub["key"].str.lower()=="heat.switch.coal_to_hp_pct"].empty else 0.0
    ef = {"coal": defaults["co2"]["hard_coal"], "gas": defaults["co2"]["ccgt"], "biomass": defaults["co2"]["biomass"], "hp": 0.0}
    out = heat.copy()
    for c in ["coal_share","gas_share","biomass_share","others_share"]:
        if c not in out.columns: out[c]=0.0
    # Apply switches (percentage points of total heat)
    sw_g = coal_to_gas/100.0; sw_hp = coal_to_hp/100.0
    out["coal_share_new"] = (out["coal_share"]/100.0 - sw_g - sw_hp).clip(lower=0)
    out["gas_share_new"] = (out["gas_share"]/100.0 + sw_g).clip(upper=1.0)
    out["hp_share_new"]  = sw_hp
    out["biomass_share_new"] = out["biomass_share"]/100.0
    # Normalize if sums != 1 (leave "others" absorbing)
    s = out[["coal_share_new","gas_share_new","hp_share_new","biomass_share_new"]].sum(axis=1)
    other_new = (1.0 - s).clip(lower=0)
    # Emissions (tCO2/MWh_th) approximated using electric EF for simplicity; scale factor 0.9 for heat boiler -> el proxy
    scale = 0.9
    out["co2_mt"] = (out["heat_demand_gwh_th"] * (
        out["coal_share"]/100.0 * ef["coal"] +
        out["gas_share"]/100.0 * ef["gas"] +
        out["biomass_share"]/100.0 * ef["biomass"]
    ) * scale)/1000.0
    out["co2_mt_new"] = (out["heat_demand_gwh_th"] * (
        out["coal_share_new"] * ef["coal"] +
        out["gas_share_new"] * ef["gas"] +
        out["hp_share_new"]  * ef["hp"] +
        out["biomass_share_new"] * ef["biomass"] +
        other_new * 0.0
    ) * scale)/1000.0
    out["co2_mt_delta"] = out["co2_mt_new"] - out["co2_mt"]
    return out[["date","heat_demand_gwh_th","coal_share","gas_share","biomass_share","others_share",
                "coal_share_new","gas_share_new","hp_share_new","biomass_share_new","co2_mt","co2_mt_new","co2_mt_delta"]]

def price_regression(prices_m: pd.DataFrame, fuel_prices_m: pd.DataFrame, disp: pd.DataFrame) -> pd.DataFrame:
    if prices_m.empty or disp.empty:
        return pd.DataFrame()
    df = prices_m.copy().merge(fuel_prices_m, on="date", how="left").merge(disp[["date","imports_gwh","demand_gwh"]], on="date", how="left")
    # Choose predictors
    cols = []
    for c in ["ETS_EUR_T","GAS_TTF_EUR_MWH","COAL_EUR_T","API2_USD_T"]:
        if c in df.columns: cols.append(c)
    cols += [c for c in ["imports_gwh","demand_gwh"] if c in df.columns]
    X = df[cols].copy().apply(lambda s: (s - s.mean())/ (s.std(ddof=0)+1e-9))
    X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    y = df["price_eur_mwh"].astype(float)
    try:
        beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        yhat = X.values @ beta
        r2 = 1.0 - np.sum((y.values - yhat)**2) / max(1e-12, np.sum((y.values - y.values.mean())**2))
        out = {"n": int(len(y)), "r2": float(r2)}
        for i, c in enumerate(X.columns):
            out[f"beta_{c}"] = float(beta[i])
        return pd.DataFrame([out])
    except Exception:
        return pd.DataFrame()


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    capacity: str
    demand: str
    fuel_prices: Optional[str]
    res_cf: Optional[str]
    gen_actual: Optional[str]
    interconnectors: Optional[str]
    capacity_market: Optional[str]
    mines: Optional[str]
    heat: Optional[str]
    pipeline: Optional[str]
    prices: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    coal_energy_content_mwh_per_t: float
    coal_vom_eur_mwh: float
    gas_vom_eur_mwh: float
    oil_vom_eur_mwh: float
    biomass_vom_eur_mwh: float
    defaults_json: str
    scenario: str
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Poland coal transition system model")
    ap.add_argument("--capacity", required=True)
    ap.add_argument("--demand", required=True)
    ap.add_argument("--fuel_prices", default="")
    ap.add_argument("--res_cf", default="")
    ap.add_argument("--gen_actual", default="")
    ap.add_argument("--interconnectors", default="")
    ap.add_argument("--capacity_market", default="")
    ap.add_argument("--mines", default="")
    ap.add_argument("--heat", default="")
    ap.add_argument("--pipeline", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2035-12")
    ap.add_argument("--coal_energy_content_mwh_per_t", type=float, default=24.0)
    ap.add_argument("--coal_vom_eur_mwh", type=float, default=3.0)
    ap.add_argument("--gas_vom_eur_mwh", type=float, default=2.0)
    ap.add_argument("--oil_vom_eur_mwh", type=float, default=3.0)
    ap.add_argument("--biomass_vom_eur_mwh", type=float, default=4.0)
    ap.add_argument("--defaults_json", default='{"eff":{"lignite":0.35,"hard_coal":0.38,"ccgt":0.54,"ocgt":0.35,"biomass":0.28,"oil":0.38,"waste":0.25,"nuclear":0.90,"wind":1.0,"solar":1.0,"hydro":1.0,"other":0.35},"co2":{"lignite":1.10,"hard_coal":0.90,"ccgt":0.36,"ocgt":0.56,"oil":0.74,"biomass":0.00,"waste":0.40,"nuclear":0.00,"wind":0.00,"solar":0.00,"hydro":0.00,"other":0.50},"avail":{"lignite":0.80,"hard_coal":0.80,"ccgt":0.90,"ocgt":0.85,"biomass":0.85,"oil":0.80,"waste":0.85,"nuclear":0.90,"wind":0.30,"solar":0.13,"hydro":0.50,"other":0.60}}')
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--outdir", default="out_pl_coal_transition")
    return ap.parse_args()


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    fleet = load_capacity(args.capacity)
    demand_m = load_demand(args.demand)
    fuel_prices_m = load_fuel_prices(args.fuel_prices) if args.fuel_prices else pd.DataFrame()
    res_cf_m = load_res_cf(args.res_cf) if args.res_cf else pd.DataFrame()
    gen_act_m = load_gen_actual(args.gen_actual) if args.gen_actual else pd.DataFrame()
    intercon_m = load_intercon(args.interconnectors) if args.interconnectors else pd.DataFrame()
    cm = load_capacity_market(args.capacity_market) if args.capacity_market else pd.DataFrame()
    mines = load_mines(args.mines) if args.mines else pd.DataFrame()
    heat = load_heat(args.heat) if args.heat else pd.DataFrame()
    pipeline = load_pipeline(args.pipeline) if args.pipeline else pd.DataFrame()
    prices_m = load_prices_spot(args.prices) if args.prices else pd.DataFrame()
    scenarios_df = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Filter demand window
    window = (pd.to_datetime(args.start), pd.to_datetime(args.end))
    demand_m = demand_m[(demand_m["date"]>=window[0]) & (demand_m["date"]<=window[1])].copy()

    # Defaults & VOM & scenario multipliers
    defaults = json.loads(args.defaults_json)
    vom = {"coal": args.coal_vom_eur_mwh, "gas": args.gas_vom_eur_mwh, "oil": args.oil_vom_eur_mwh, "biomass": args.biomass_vom_eur_mwh}
    # Scenario multipliers (basic)
    scen_mult = {"ets.multiplier": 1.0, "gas.price_mult": 1.0, "coal.price_mult": 1.0}
    if not scenarios_df.empty and args.scenario in scenarios_df["scenario"].unique():
        sub = scenarios_df[scenarios_df["scenario"]==args.scenario]
        for k in scen_mult.keys():
            row = sub[sub["key"].str.lower()==k]
            if not row.empty:
                try: scen_mult[k] = float(row["value"].iloc[0])
                except Exception: pass
        # RES multipliers on CF (for quick sensitivity)
        res_mult = {
            "wind": float(sub[sub["key"].str.lower()=="res.mult.wind"]["value"].iloc[0]) if not sub[sub["key"].str.lower()=="res.mult.wind"].empty else 1.0,
            "solar": float(sub[sub["key"].str.lower()=="res.mult.solar"]["value"].iloc[0]) if not sub[sub["key"].str.lower()=="res.mult.solar"].empty else 1.0
        }
    else:
        res_mult = {"wind": 1.0, "solar": 1.0}

    # Enrich fleet (defaults)
    fleet = apply_defaults_to_fleet(fleet, defaults)
    # Retirement years under scenario
    fleet, retire_sched = infer_retirements(fleet, args.scenario, scenarios_df, args.start, args.end)

    # Capacity TS (apply retirements + pipeline)
    cap_ts = capacity_time_series(fleet, pipeline, args.start, args.end)

    # RES CF adjustments
    if not res_cf_m.empty:
        res_cf_m = res_cf_m.copy()
        res_cf_m["capacity_factor"] = res_cf_m["capacity_factor"] * res_mult.get("wind",1.0) if "wind" in res_cf_m["fuel"].unique() else res_cf_m.get("capacity_factor")
        res_cf_m.loc[res_cf_m["fuel"]=="wind", "capacity_factor"] = res_cf_m.loc[res_cf_m["fuel"]=="wind","capacity_factor"] * res_mult["wind"] if "capacity_factor" in res_cf_m.columns else res_cf_m.get("capacity_factor")
        res_cf_m.loc[res_cf_m["fuel"]=="solar","capacity_factor"] = res_cf_m.loc[res_cf_m["fuel"]=="solar","capacity_factor"] * res_mult["solar"] if "capacity_factor" in res_cf_m.columns else res_cf_m.get("capacity_factor")
        res_cf_m = res_cf_m[(res_cf_m["date"]>=window[0]) & (res_cf_m["date"]<=window[1])]

    # Thermal costs (€/MWh el)
    if not fuel_prices_m.empty:
        fuel_prices_m = fuel_prices_m[(fuel_prices_m["date"]>=window[0]) & (fuel_prices_m["date"]<=window[1])]
    costs = thermal_costs_table(fuel_prices_m, defaults, args.coal_energy_content_mwh_per_t, vom, scen_mult)
    costs.to_csv(outdir / "cost_stack.csv", index=False)

    # RES generation
    res_gen = res_generation(res_cf_m, cap_ts) if not res_cf_m.empty else pd.DataFrame(columns=["date","fuel","generation_gwh"])

    # Dispatch
    disp, cost_stack = merit_order_dispatch(demand_m, cap_ts, res_gen, intercon_m, costs, defaults, scenarios_df, args.scenario)
    disp.to_csv(outdir / "monthly_dispatch.csv", index=False)
    cost_stack.to_csv(outdir / "cost_stack.csv", index=False)

    # Emissions
    emis = emissions_from_dispatch(disp, defaults)
    emis.to_csv(outdir / "emissions.csv", index=False)

    # Capacity margin
    cap_margin = capacity_margin_table(cap_ts, demand_m, defaults, intercon_m)
    cap_margin.to_csv(outdir / "capacity_margin.csv", index=False)

    # Capacity market revenues
    cm_rev = capacity_market_revenue(cm, fleet, args.scenario, scenarios_df)
    if not cm_rev.empty:
        cm_rev.to_csv(outdir / "capacity_market_revenues.csv", index=False)

    # Jobs & mines proxy
    jobs = jobs_transition(mines, disp)
    if not jobs.empty:
        jobs.to_csv(outdir / "jobs_transition.csv", index=False)

    # Heat system switching
    heat_out = heat_switch(heat, defaults, scenarios_df, args.scenario)
    if not heat_out.empty:
        heat_out.to_csv(outdir / "heat_system.csv", index=False)

    # Price regression (if spot prices provided)
    pr_reg = price_regression(prices_m, fuel_prices_m, disp)
    if not pr_reg.empty:
        pr_reg.to_csv(outdir / "price_regression.csv", index=False)

    # Scenario impact summary row (coal share, CO2, ETS bill)
    # Coal share
    if not disp.empty:
        tmp = disp.copy()
        tmp["coal_gwh"] = tmp[["lignite_gwh","hard_coal_gwh"]].sum(axis=1)
        tmp["thermal_gwh"] = tmp[[c for c in tmp.columns if c.endswith("_gwh") and ("wind" not in c and "solar" not in c and "hydro" not in c and "imports" not in c and "demand" not in c)]].sum(axis=1)
        coal_share = float((tmp["coal_gwh"].sum() / tmp["demand_gwh"].sum()) if tmp["demand_gwh"].sum()>0 else np.nan)
    else:
        coal_share = np.nan
    # ETS bill
    if not emis.empty and "ETS_EUR_T" in costs.columns:
        # Join ETS price monthly
        e = emis.merge(costs[["date","ETS_EUR_T"]], on="date", how="left")
        e["bill_eur_m"] = e["total_mtco2"] * e["ETS_EUR_T"]
        ets_bill_eur_b = float(e["bill_eur_m"].sum()) / 1e9  # € bn
    else:
        ets_bill_eur_b = np.nan

    # fleet_enriched & retire_schedule
    fleet_out = fleet.copy()
    fleet_out.to_csv(outdir / "fleet_enriched.csv", index=False)
    retire_sched.to_csv(outdir / "retire_schedule.csv", index=False)

    # Scenarios out (if provided) — simple table with high-level stats
    scen_out = pd.DataFrame([{
        "scenario": args.scenario,
        "period": f"{args.start}..{args.end}",
        "coal_share_pct": coal_share*100.0 if coal_share==coal_share else np.nan,
        "total_co2_mt": float(emis["total_mtco2"].sum()) if not emis.empty else np.nan,
        "ets_bill_eur_bn": ets_bill_eur_b,
        "first_capacity_deficit": str(cap_margin.loc[cap_margin["capacity_margin_mw"]<0, "date"].min().date()) if (not cap_margin.empty and (cap_margin["capacity_margin_mw"]<0).any()) else None
    }])
    scen_out.to_csv(outdir / "scenarios_out.csv", index=False)

    # Summary KPIs
    kpi = {
        "window": {"start": args.start, "end": args.end},
        "scenario": args.scenario,
        "demand_twh": float(demand_m["demand_gwh"].sum()/1000.0) if not demand_m.empty else None,
        "coal_share_pct": float(coal_share*100.0) if coal_share==coal_share else None,
        "total_emissions_mt": float(emis["total_mtco2"].sum()) if not emis.empty else None,
        "ets_bill_eur_bn": ets_bill_eur_b,
        "min_monthly_margin_mw": float(cap_margin["capacity_margin_mw"].min()) if not cap_margin.empty else None,
        "max_lole_proxy_hours": float(cap_margin["lole_hours_proxy"].max()) if not cap_margin.empty else None,
        "cm_revenues_eur_m_total": float(cm_rev["revenue_eur_m"].sum()) if not cm_rev.empty else None,
        "heat_co2_delta_mt": float(heat_out["co2_mt_delta"].sum()) if not heat_out.empty else None,
        "price_reg_r2": float(pr_reg["r2"].iloc[0]) if not pr_reg.empty else None
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))

    # Config dump
    cfg = asdict(Config(
        capacity=args.capacity, demand=args.demand, fuel_prices=(args.fuel_prices or None),
        res_cf=(args.res_cf or None), gen_actual=(args.gen_actual or None), interconnectors=(args.interconnectors or None),
        capacity_market=(args.capacity_market or None), mines=(args.mines or None), heat=(args.heat or None),
        pipeline=(args.pipeline or None), prices=(args.prices or None), scenarios=(args.scenarios or None),
        start=args.start, end=args.end, coal_energy_content_mwh_per_t=args.coal_energy_content_mwh_per_t,
        coal_vom_eur_mwh=args.coal_vom_eur_mwh, gas_vom_eur_mwh=args.gas_vom_eur_mwh,
        oil_vom_eur_mwh=args.oil_vom_eur_mwh, biomass_vom_eur_mwh=args.biomass_vom_eur_mwh,
        defaults_json=args.defaults_json, scenario=args.scenario, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Polish Coal Transition Model ==")
    print(f"Scenario: {args.scenario} | Window: {args.start} → {args.end}")
    if kpi["coal_share_pct"] is not None:
        print(f"Coal share of demand: {kpi['coal_share_pct']:.1f}% | Total CO₂: {kpi['total_emissions_mt']:.1f} Mt | ETS bill: {kpi['ets_bill_eur_bn']:.2f} €bn")
    if kpi["min_monthly_margin_mw"] is not None:
        print(f"Min capacity margin: {kpi['min_monthly_margin_mw']:.0f} MW | Max LOLE proxy: {kpi['max_lole_proxy_hours']:.1f} h")
    if not retire_sched.empty:
        print("Retirements (next 5 years):")
        y0 = year(pd.to_datetime(args.start))
        print(retire_sched[(retire_sched['year']>=y0) & (retire_sched['year']<y0+5)].to_string(index=False))
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
def thermal_costs_table(fuel_prices_m: pd.DataFrame,
                        defaults: Dict[str,Dict[str,float]],
                        coal_energy_content_mwh_per_t: float,
                        vom: Dict[str,float],
                        scen_mult: Dict[str,float]) -> pd.DataFrame:
    """