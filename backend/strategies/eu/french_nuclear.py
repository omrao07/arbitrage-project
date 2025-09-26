#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
french_nuclear.py — Fleet availability, generation, river-temperature derates, and price/residual-load analytics

What this does
--------------
Given reactor metadata, outage schedules, hydrology/temperature, load, renewables and (optionally) market prices,
this script builds a daily view of the French nuclear fleet:

Core outputs
- Daily reactor availability & derates (planned, unplanned, thermal/env)
- Fleet-level capacity factor and generation (MWh/GWh/TWh)
- River temperature constraint model (derate above limit; site-specific rivers)
- Residual load = demand − (wind + solar + hydro + expected_nuclear); price sensitivity
- Simple OLS: price ~ residual load (€/MWh per GW) with goodness-of-fit
- Scenario engine to tweak thresholds, extend/retire units, delay outages, apply multipliers
- Monte Carlo stress of unplanned derates around user-defined mean/sd
- CO₂ displacement estimate vs gas/coal counterfactuals

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--reactors reactors.csv   REQUIRED
  Columns (suggested):
    reactor_id, site, river, coolant (River/Sea), net_mw, status(Operating/Retired/Mothballed),
    commission_date, retire_date (optional), refuel_cycle_months (optional)

--outages outages.csv     OPTIONAL (planned/unplanned outages & derates)
  Columns:
    reactor_id, start_date, end_date, type(Planned/Unplanned/Derate), derate_pct (0..100, optional), note

--hydrology hydro.csv     OPTIONAL (river temps/flows; site will join via 'river')
  Columns:
    date, river, temp_c, flow_m3s

--demand demand.csv       OPTIONAL (system load)
  Columns:
    date or datetime, load_mw

--renewables ren.csv      OPTIONAL (supply to compute residual)
  Columns:
    date or datetime, wind_mw, solar_mw, hydro_mw

--market market.csv       OPTIONAL (power prices; hourly or daily)
  Columns:
    timestamp or date, price_eur_mwh

--scenarios scenarios.csv OPTIONAL (overrides and multiple scenarios)
  Columns:
    scenario, name, key, value
  Example keys (dot-notation):
    thermal.river_temp_limit_c = 27
    thermal.derate_per_deg = 0.12
    outages.multiplier_planned = 1.05        # stretch planned outage durations
    outages.multiplier_unplanned = 1.10
    outages.delay_return.reactor_<ID> = +14d # push end_date by +14d for a specific unit
    status.reactor_<ID> = Retired            # force status
    status.extend_to.reactor_<ID> = 2040-12-31
    mc.unplanned_derate_mean = 0.02          # daily random derate mean (fraction)
    mc.unplanned_derate_sd = 0.03            # standard deviation (fraction)
    co2.gas_intensity_t_per_mwh = 0.35
    co2.coal_intensity_t_per_mwh = 0.75

Key options
-----------
--start 2023-01-01
--end   2025-12-31
--scenario baseline
--seed 42
--outdir out_fr_nuclear

Outputs
-------
- availability_daily.csv     Per-reactor daily MW available, derate components, capacity factor
- fleet_daily.csv            Fleet totals (available MW, CF, generation, residual load, price)
- price_regression.csv       OLS results of price ~ residual_load
- scenario_outcomes.csv      Aggregates per scenario (TWh, CF, CO₂ avoided vs gas/coal)
- summary.json               Headline KPIs
- config.json                Run configuration + effective parameters

Notes
-----
- This is a modeling tool with stylized constraints; bring official data (EDF, RTE, Météo-France, etc.) for fidelity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd


# ------------------------ Defaults & helpers ------------------------

DEFAULT_PARAMS = {
    "thermal": {
        "river_temp_limit_c": 27.0,     # °C threshold for environmental derating (default)
        "derate_per_deg": 0.10,         # 10% net derate per °C above limit (river-cooled units)
        "max_derate_fraction": 0.50     # cap thermal derate at 50% of net MW
    },
    "outages": {
        "multiplier_planned": 1.00,
        "multiplier_unplanned": 1.00
    },
    "status": {
        # Optional per-reactor status overrides or life extensions via scenarios
        # "reactor_<ID>": "Operating" | "Retired"
        # "extend_to": { "reactor_<ID>": "YYYY-MM-DD" }
    },
    "mc": {
        "use": 1,                        # apply small random daily unplanned derate
        "unplanned_derate_mean": 0.02,   # 2% mean
        "unplanned_derate_sd": 0.03      # 3% sd
    },
    "co2": {
        "gas_intensity_t_per_mwh": 0.35,
        "coal_intensity_t_per_mwh": 0.75
    }
}

def ncol(df: pd.DataFrame, t: str) -> Optional[str]:
    target = t.lower()
    for c in df.columns:
        if c.lower() == target: return c
    for c in df.columns:
        if target in c.lower(): return c
    return None

def to_day(s: pd.Series) -> pd.Series:
    # Supports date or datetime → date
    x = pd.to_datetime(s, errors="coerce")
    return x.dt.tz_convert("UTC").dt.floor("D").dt.date if hasattr(x, "dt") and x.dt.tz is not None else x.dt.floor("D").dt.date

def parse_date(s) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(s)
    except Exception:
        return None

def daterange(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def safe_num(s) -> float:
    try:
        return float(s)
    except Exception:
        return np.nan

def seed_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(int(seed) if seed is not None else None)

# ------------------------ Loaders ------------------------

def load_reactors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"reactor_id") or df.columns[0]): "reactor_id",
        (ncol(df,"site") or "site"): "site",
        (ncol(df,"river") or "river"): "river",
        (ncol(df,"coolant") or "coolant"): "coolant",
        (ncol(df,"net_mw") or "net_mw"): "net_mw",
        (ncol(df,"status") or "status"): "status",
        (ncol(df,"commission_date") or "commission_date"): "commission_date",
        (ncol(df,"retire_date") or "retire_date"): "retire_date",
        (ncol(df,"refuel_cycle_months") or "refuel_cycle_months"): "refuel_cycle_months",
    }
    df = df.rename(columns=ren)
    df["reactor_id"] = df["reactor_id"].astype(str)
    df["site"] = df.get("site","").astype(str)
    df["river"] = df.get("river","").astype(str)
    df["coolant"] = df.get("coolant","River")
    df["status"] = df.get("status","Operating")
    df["net_mw"] = pd.to_numeric(df.get("net_mw", np.nan), errors="coerce")
    df["commission_date"] = pd.to_datetime(df.get("commission_date", pd.NaT), errors="coerce")
    df["retire_date"] = pd.to_datetime(df.get("retire_date", pd.NaT), errors="coerce")
    return df

def load_outages(path: Optional[str]) -> pd.DataFrame:
    if not path: 
        return pd.DataFrame(columns=["reactor_id","start_date","end_date","type","derate_pct","note"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"reactor_id") or df.columns[0]): "reactor_id",
        (ncol(df,"start_date") or "start_date"): "start_date",
        (ncol(df,"end_date") or "end_date"): "end_date",
        (ncol(df,"type") or "type"): "type",
        (ncol(df,"derate_pct") or "derate_pct"): "derate_pct",
        (ncol(df,"note") or "note"): "note",
    }
    df = df.rename(columns=ren)
    df["reactor_id"] = df["reactor_id"].astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["type"] = df.get("type","Planned")
    df["derate_pct"] = pd.to_numeric(df.get("derate_pct", np.nan), errors="coerce")  # 0..100
    return df

def load_hydrology(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["date","river","temp_c","flow_m3s"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"river") or "river"): "river",
        (ncol(df,"temp_c") or "temp_c"): "temp_c",
        (ncol(df,"flow_m3s") or ncol(df,"flow") or "flow_m3s"): "flow_m3s",
    }
    df = df.rename(columns=ren)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df["river"] = df.get("river","").astype(str)
    for c in ["temp_c","flow_m3s"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_series(path: Optional[str], cols_map: Dict[str,str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    # Map known columns
    out_map = {}
    for want, fallback in cols_map.items():
        cand = ncol(df, want) or ncol(df, fallback) or fallback
        if cand in df.columns: out_map[cand] = want
    df = df.rename(columns=out_map)
    # normalize datetime/date
    if "date" in df.columns and df["date"].dtype != "<M8[ns]":
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
        except Exception:
            pass
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.floor("D")
    # numeric
    for c in ["load_mw","wind_mw","solar_mw","hydro_mw","price_eur_mwh"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: 
        return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"scenario") or "scenario"): "scenario",
        (ncol(df,"name") or "name"): "name",
        (ncol(df,"key") or "key"): "key",
        (ncol(df,"value") or "value"): "value",
    }
    return df.rename(columns=ren)

# ------------------------ Parameter plumbing ------------------------

def apply_overrides(base: Dict, overrides: Dict[str, str]) -> Dict:
    import copy
    P = copy.deepcopy(base)
    def set_by_dot(d: Dict, dotted: str, val):
        parts = str(dotted).split(".")
        cur = d
        for k in parts[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        # attempt numeric cast
        v = val
        try:
            vnum = float(val)
            if vnum.is_integer():
                vnum = int(vnum)
            v = vnum
        except Exception:
            v = str(val)
        cur[parts[-1]] = v
    for k, v in overrides.items():
        set_by_dot(P, k, v)
    return P

def collect_overrides(scen_df: pd.DataFrame, scen_id: str) -> Dict[str,str]:
    if scen_df.empty: return {}
    sub = scen_df[scen_df["scenario"] == scen_id]
    return {str(k): v for k, v in zip(sub["key"], sub["value"])}

# ------------------------ Core modeling ------------------------

def expand_outages_per_day(outages: pd.DataFrame, P: Dict) -> pd.DataFrame:
    if outages.empty:
        return pd.DataFrame(columns=["date","reactor_id","derate_frac","reason"])
    rows = []
    for _, r in outages.iterrows():
        if pd.isna(r["start_date"]) or pd.isna(r["end_date"]): 
            continue
        mult = P["outages"]["multiplier_planned"] if str(r.get("type","")).lower().startswith("plan") else P["outages"]["multiplier_unplanned"]
        sd = r["start_date"]
        ed = r["end_date"]
        # multiplier stretches end date
        if float(mult) != 1.0:
            dur = (ed - sd).days + 1
            ed = sd + pd.Timedelta(days=int(np.ceil(dur * float(mult))) - 1)
        for d in pd.date_range(sd, ed, freq="D"):
            derate_frac = np.nan
            # If explicit derate_pct, use as fractional unavailability (e.g., 100 → full outage)
            if not pd.isna(r.get("derate_pct", np.nan)):
                derate_frac = clip01(float(r["derate_pct"]) / 100.0)
            else:
                # If no derate specified, assume full unavailability for 'Outage', partial for 'Derate'
                derate_frac = 1.0 if str(r.get("type","")).lower() in {"planned","unplanned","outage"} else 0.2
            rows.append({"date": d, "reactor_id": r["reactor_id"], "derate_frac": derate_frac, "reason": str(r.get("type",""))})
    out = pd.DataFrame(rows)
    return out.groupby(["date","reactor_id","reason"], as_index=False)["derate_frac"].max()  # max derate per reason/day

def thermal_derate_for_site(temp_c: float, coolant: str, P: Dict) -> float:
    if str(coolant).lower().startswith("sea"): 
        return 0.0
    if not np.isfinite(temp_c):
        return 0.0
    limit = float(P["thermal"]["river_temp_limit_c"])
    if temp_c <= limit:
        return 0.0
    per_deg = float(P["thermal"]["derate_per_deg"])
    frac = (temp_c - limit) * per_deg
    return clip01(min(frac, float(P["thermal"]["max_derate_fraction"])))

def availability_daily(reactors: pd.DataFrame,
                       outages_daily: pd.DataFrame,
                       hydro: pd.DataFrame,
                       days: pd.DatetimeIndex,
                       P: Dict,
                       rng: np.random.Generator) -> pd.DataFrame:
    # Prepare hydrology per river/day
    hydro_use = hydro.copy()
    if not hydro_use.empty:
        hydro_use = hydro_use.rename(columns={"date":"d"}).dropna(subset=["d"])
        hydro_use["d"] = pd.to_datetime(hydro_use["d"]).dt.floor("D")

    rows = []
    for _, r in reactors.iterrows():
        rid = r["reactor_id"]
        status = str(r.get("status","Operating"))
        net = float(r.get("net_mw", np.nan))
        coolant = str(r.get("coolant","River"))
        river = str(r.get("river",""))
        com = r.get("commission_date", pd.NaT)
        ret = r.get("retire_date", pd.NaT)
        # status override from P
        st_key = f"reactor_{rid}"
        if st_key in P.get("status", {}):
            status = str(P["status"][st_key])
        if "extend_to" in P.get("status", {}) and st_key in P["status"]["extend_to"]:
            try:
                ret = pd.to_datetime(P["status"]["extend_to"][st_key])
            except Exception:
                pass

        if not np.isfinite(net) or status.lower() == "retired":
            continue

        # Join outages for this reactor
        od = outages_daily[outages_daily["reactor_id"] == rid].copy() if not outages_daily.empty else pd.DataFrame(columns=["date","derate_frac","reason"])
        od = od.rename(columns={"date":"d"})
        # By day iterate
        for d in days:
            if (pd.notna(com) and d < com) or (pd.notna(ret) and d > ret):
                continue
            # Base availability = 1.0
            base_av = 1.0
            # Outage/derate
            o_today = od[od["d"] == d]
            derate_outage = 0.0
            if not o_today.empty:
                # If multiple rows (planned/unplanned), take max derate for the day, but keep both reasons
                derate_outage = float(o_today["derate_frac"].max())

            # Thermal derate
            therm = 0.0
            if river and not hydro_use.empty:
                riv = hydro_use[(hydro_use["river"].astype(str) == river) & (hydro_use["d"] == d)]
                if not riv.empty:
                    temp_c = float(riv["temp_c"].iloc[0]) if "temp_c" in riv.columns else np.nan
                    therm = thermal_derate_for_site(temp_c, coolant, P)

            # Monte Carlo small unplanned derate
            mc_frac = 0.0
            if int(P["mc"].get("use", 1)) == 1:
                mu = float(P["mc"]["unplanned_derate_mean"])
                sd = float(P["mc"]["unplanned_derate_sd"])
                mc_frac = max(0.0, rng.normal(mu, sd))
                if mc_frac > 0.25:
                    mc_frac = 0.25  # guardrail

            # Combine derates multiplicatively on available fraction: avail = net * (1 - outage) * (1 - thermal) * (1 - mc)
            avail_frac = (1.0 - derate_outage) * (1.0 - therm) * (1.0 - mc_frac)
            avail_frac = clip01(avail_frac)
            avail_mw = net * avail_frac
            rows.append({
                "date": d, "reactor_id": rid, "site": r.get("site",""), "river": river, "coolant": coolant,
                "net_mw": net, "derate_outage": derate_outage, "derate_thermal": therm, "derate_mc": mc_frac,
                "avail_frac": avail_frac, "avail_mw": avail_mw
            })
    return pd.DataFrame(rows)

def aggregate_fleet(av: pd.DataFrame) -> pd.DataFrame:
    if av.empty:
        return pd.DataFrame(columns=["date","fleet_net_mw","fleet_avail_mw","fleet_cf","gen_gwh"])
    g = (av.groupby("date", as_index=False)
           .agg(fleet_net_mw=("net_mw","sum"), fleet_avail_mw=("avail_mw","sum")))
    g["fleet_cf"] = g["fleet_avail_mw"] / g["fleet_net_mw"].replace(0, np.nan)
    # Generation per day GWh = avail_mw * 24 / 1000
    g["gen_gwh"] = g["fleet_avail_mw"] * 24.0 / 1000.0
    return g

def build_residual_and_prices(fleet: pd.DataFrame, demand: pd.DataFrame, ren: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if fleet.empty:
        return pd.DataFrame()
    df = fleet.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Demand daily
    if not demand.empty:
        d = demand.copy()
        if "timestamp" in d.columns:
            d["date"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_convert("UTC").dt.floor("D")
        if "date" in d.columns and "load_mw" in d.columns:
            d = d.groupby("date", as_index=False)["load_mw"].mean()
            df = df.merge(d, on="date", how="left")
    else:
        df["load_mw"] = np.nan
    # Renewables daily
    if not ren.empty:
        r = ren.copy()
        if "timestamp" in r.columns:
            r["date"] = pd.to_datetime(r["timestamp"], errors="coerce").dt.tz_convert("UTC").dt.floor("D")
        keep = [c for c in ["wind_mw","solar_mw","hydro_mw"] if c in r.columns]
        if "date" in r.columns and keep:
            r = r.groupby("date", as_index=False)[keep].mean()
            df = df.merge(r, on="date", how="left")
    for c in ["wind_mw","solar_mw","hydro_mw"]:
        if c not in df.columns: df[c] = np.nan
    # Market price daily avg
    if not market.empty:
        m = market.copy()
        if "timestamp" in m.columns:
            m["date"] = pd.to_datetime(m["timestamp"], errors="coerce").dt.tz_convert("UTC").dt.floor("D")
        m = m.dropna(subset=["date"])
        if "price_eur_mwh" in m.columns:
            price = m.groupby("date", as_index=False)["price_eur_mwh"].mean().rename(columns={"price_eur_mwh":"price_eur_mwh"})
            df = df.merge(price, on="date", how="left")
    else:
        df["price_eur_mwh"] = np.nan
    # Residual load (simple)
    df["nuclear_mw"] = df["fleet_avail_mw"]
    df["renew_mw"] = df[["wind_mw","solar_mw","hydro_mw"]].sum(axis=1, skipna=True)
    df["residual_load_mw"] = df["load_mw"] - df["renew_mw"] - df["nuclear_mw"]
    return df

def price_regression(df: pd.DataFrame) -> pd.DataFrame:
    """OLS price ~ residual_load (€/MWh per GW)."""
    d = df.dropna(subset=["price_eur_mwh","residual_load_mw"]).copy()
    if d.empty or len(d) < 10:
        return pd.DataFrame(columns=["n","beta_eur_per_mwh_per_mw","beta_eur_per_mwh_per_gw","intercept","r2"])
    X = d["residual_load_mw"].values.astype(float)
    Y = d["price_eur_mwh"].values.astype(float)
    A = np.column_stack([np.ones_like(X), X])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    yhat = A @ beta
    r2 = 1.0 - np.sum((Y - yhat)**2) / max(1e-12, np.sum((Y - Y.mean())**2))
    per_mw = float(beta[1])
    return pd.DataFrame([{
        "n": int(len(d)),
        "beta_eur_per_mwh_per_mw": per_mw,
        "beta_eur_per_mwh_per_gw": per_mw * 1000.0,
        "intercept": float(beta[0]),
        "r2": float(r2)
    }])

def co2_avoided(fleet: pd.DataFrame, P: Dict) -> Tuple[float,float]:
    """Return (gas_tonnes, coal_tonnes) avoided for the modeled period."""
    if fleet.empty: return 0.0, 0.0
    mwh = float(fleet["gen_gwh"].sum() * 1000.0)
    gas_t = mwh * float(P["co2"]["gas_intensity_t_per_mwh"])
    coal_t = mwh * float(P["co2"]["coal_intensity_t_per_mwh"])
    return gas_t, coal_t

# ------------------------ CLI ------------------------

@dataclass
class Config:
    reactors: str
    outages: Optional[str]
    hydrology: Optional[str]
    demand: Optional[str]
    renewables: Optional[str]
    market: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    scenario: str
    seed: Optional[int]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="French nuclear fleet analytics")
    ap.add_argument("--reactors", required=True)
    ap.add_argument("--outages", default="")
    ap.add_argument("--hydrology", default="")
    ap.add_argument("--demand", default="")
    ap.add_argument("--renewables", default="")
    ap.add_argument("--market", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="out_fr_nuclear")
    return ap.parse_args()

# ------------------------ Main ------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    P_base = DEFAULT_PARAMS
    scen_df = load_scenarios(args.scenarios)
    overrides = collect_overrides(scen_df, args.scenario) if args.scenario != "baseline" else {}
    P = apply_overrides(P_base, overrides) if overrides else P_base

    rng = seed_rng(args.seed)

    reactors = load_reactors(args.reactors)
    outages = load_outages(args.outages) if args.outages else pd.DataFrame()
    hydro = load_hydrology(args.hydrology) if args.hydrology else pd.DataFrame()
    demand = load_series(args.demand, {"date":"date","timestamp":"timestamp","load_mw":"load_mw"})
    ren = load_series(args.renewables, {"date":"date","timestamp":"timestamp","wind_mw":"wind_mw","solar_mw":"solar_mw","hydro_mw":"hydro_mw"})
    market = load_series(args.market, {"date":"date","timestamp":"timestamp","price_eur_mwh":"price_eur_mwh"})

    # Apply specific outage-return delays from scenarios: outages.delay_return.reactor_<ID> = +Xd
    if not outages.empty:
        # Clone to avoid mutating input
        outages = outages.copy()
        for col in ["start_date","end_date"]:
            outages[col] = pd.to_datetime(outages[col], errors="coerce")
        for k, v in overrides.items():
            if k.startswith("outages.delay_return.reactor_"):
                rid = k.split("outages.delay_return.reactor_")[-1]
                try:
                    if str(v).startswith("+") and str(v).endswith("d"):
                        extra = int(str(v)[1:-1])
                        mask = outages["reactor_id"].astype(str) == rid
                        outages.loc[mask, "end_date"] = outages.loc[mask, "end_date"] + pd.to_datetime(f"1970-01-01") + pd.to_timedelta(extra, unit="D") - pd.to_datetime("1970-01-02")
                except Exception:
                    pass
        # Status overrides already handled in availability

    # Expand outages into daily fractions
    outages_daily = expand_outages_per_day(outages, P) if not outages.empty else pd.DataFrame(columns=["date","reactor_id","derate_frac","reason"])

    # Time domain
    days = daterange(args.start, args.end)

    # Availability
    av = availability_daily(reactors, outages_daily, hydro, days, P, rng)
    av_out = av.copy()
    av_out["date"] = pd.to_datetime(av_out["date"]).dt.date
    av_out.to_csv(outdir / "availability_daily.csv", index=False)

    # Fleet aggregate and residuals
    fleet = aggregate_fleet(av)
    res = build_residual_and_prices(fleet, demand, ren, market)
    res_out = res.copy()
    res_out["date"] = pd.to_datetime(res_out["date"]).dt.date
    res_out.to_csv(outdir / "fleet_daily.csv", index=False)

    # Price regression
    preg = price_regression(res)
    if not preg.empty:
        preg.to_csv(outdir / "price_regression.csv", index=False)

    # CO2 avoided
    gas_t, coal_t = co2_avoided(fleet, P)

    # Scenario outcomes
    scen_rows = [{
        "scenario": args.scenario,
        "start": args.start, "end": args.end,
        "days": int(len(days)),
        "reactors_modeled": int(av["reactor_id"].nunique()),
        "fleet_net_avg_mw": float(fleet["fleet_net_mw"].mean()) if not fleet.empty else np.nan,
        "fleet_avail_avg_mw": float(fleet["fleet_avail_mw"].mean()) if not fleet.empty else np.nan,
        "fleet_cf_avg": float(fleet["fleet_cf"].mean()) if not fleet.empty else np.nan,
        "gen_twh": float(fleet["gen_gwh"].sum() / 1000.0) if not fleet.empty else 0.0,
        "co2_avoided_gas_tonnes": float(gas_t),
        "co2_avoided_coal_tonnes": float(coal_t),
        "price_slope_eur_per_mwh_per_gw": float(preg["beta_eur_per_mwh_per_gw"].iloc[0]) if not preg.empty else np.nan,
        "price_r2": float(preg["r2"].iloc[0]) if not preg.empty else np.nan
    }]
    scen_df_out = pd.DataFrame(scen_rows)
    scen_df_out.to_csv(outdir / "scenario_outcomes.csv", index=False)

    # Summary
    latest = res_out["date"].max() if not res_out.empty else None
    kpi = {
        "scenario": args.scenario,
        "period": {"start": args.start, "end": args.end},
        "reactors": int(reactors.shape[0]),
        "modeled_reactors": int(av["reactor_id"].nunique()),
        "gen_twh": scen_rows[0]["gen_twh"],
        "fleet_cf_avg": scen_rows[0]["fleet_cf_avg"],
        "co2_avoided_gas_tonnes": scen_rows[0]["co2_avoided_gas_tonnes"],
        "co2_avoided_coal_tonnes": scen_rows[0]["co2_avoided_coal_tonnes"],
        "price_beta_eur_per_mwh_per_gw": scen_rows[0]["price_slope_eur_per_mwh_per_gw"],
        "price_r2": scen_rows[0]["price_r2"],
        "latest_day": str(latest) if latest else None
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        reactors=args.reactors, outages=args.outages or None, hydrology=args.hydrology or None,
        demand=args.demand or None, renewables=args.renewables or None, market=args.market or None,
        scenarios=args.scenarios or None, start=args.start, end=args.end, scenario=args.scenario,
        seed=args.seed, outdir=args.outdir
    )), indent=2))

    # Console
    print("== French Nuclear Fleet Analytics ==")
    print(f"Scenario: {args.scenario} | Window: {args.start} → {args.end}")
    if not fleet.empty:
        print(f"Average CF: {kpi['fleet_cf_avg']:.2%} | Generation: {kpi['gen_twh']:.2f} TWh")
    if kpi["price_beta_eur_per_mwh_per_gw"] == kpi["price_beta_eur_per_mwh_per_gw"]:
        print(f"Price slope: {kpi['price_beta_eur_per_mwh_per_gw']:.2f} €/MWh per GW residual | R²={kpi['price_r2']:.2f}")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
