#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
greek_shipping.py — Greek-controlled fleet, ports, network, sanctions & market linkages

What this does
--------------
Given fleet registries, AIS port calls, fixtures/rates, prices and optional activity/ETS inputs,
this script builds an analytics pack focused on Greece-linked shipping:

Core outputs
------------
- fleet_enriched.csv        Vessel-level enrichment (ownership flags, type buckets, age, size class, CII/EEXI proxies)
- owners.csv                Owner/manager aggregates (count, DWT/GT, avg age, scrubber %, sanctions exposure)
- fleet_by_type.csv         Aggregates by type/class (DWT share, age buckets, orderbook coverage)
- age_distribution.csv      Age histogram per type (0–5, 6–10, …, 30+)
- orderbook_coverage.csv    Orderbook vs in-service by type (renewal ratio, avg age retiring if demolition provided)
- sanctions_exposure.csv    Sanctioned vessels/owners overlap (counts, DWT)
- port_calls.csv            Cleaned AIS port-calls (arr, dep, dwell hours, greek_port flag)
- network_edges.csv         Port-to-port edges from consecutive calls (flows, avg transit hours)
- congestion.csv            Port congestion proxies (median dwell, 90th pct dwell, calls)
- freight_regressions.csv   OLS betas: route rates ~ Brent + congestion + sanctions (if provided)
- ets_costs.csv             (optional) ETS/FuelEU proxy costs for activity entries (if provided)
- summary.json              Headline KPIs
- config.json               Reproducibility dump

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--fleet fleet.csv      REQUIRED
  Suggested cols (use what you have; extras ignored):
    imo, mmsi, name, type, sub_type, class, dwt, gt, kW_main, kW_aux, speed_kn, built_year,
    owner, owner_group, owner_country, manager, manager_country, beneficial_owner,
    beneficial_owner_country, flag, scrubber(0/1), eexi, cii_rating (A..E), is_greek_owner(0/1)

--ais ais.csv          OPTIONAL (port calls or arrivals/departures)
  Columns (any of the below are accepted and auto-detected):
    imo(or mmsi), port, port_unlocode, country, region, arrival(or ata), departure(or atd), lat, lon, status
  Notes: If only timestamps exist per change-of-state, provide both 'arrival' and 'departure' rows; otherwise
         a single row with both 'arrival' and 'departure' times is fine.

--ports ports.csv      OPTIONAL (port metadata)
  Columns: port (or name), port_unlocode, country, region, is_greek(0/1)

--fixtures fixtures.csv OPTIONAL (voyage/time charter fixtures or route indexes)
  Columns:
    date, route (e.g., TD3C, C5, C10, SCFI_Europe), rate (USD/day or WS or index), unit(optional),
    vessel_class(optional), vessel_type(optional)

--prices prices.csv     OPTIONAL (macro & commodity series)
  Columns (daily or weekly):
    date, series, value
  Example series: Brent, BDI, EUA, EURUSD, TTF, C5, TD3C (place route rates here if not in fixtures)

--orderbook orderbook.csv OPTIONAL
  Columns: type, class(optional), dwt, gt, delivery_year

--demolition demolition.csv OPTIONAL
  Columns: imo(optional), type(optional), class(optional), dwt, gt, scrap_date, price_usd_ltd(optional)

--sanctions sanctions.csv OPTIONAL
  Columns: imo, owner, owner_group, list(origin), date_listed

--activity activity.csv OPTIONAL (for ETS/FuelEU proxy)
  Columns:
    imo, date, hours_at_sea, hours_at_berth, avg_speed_kn, main_engine_load_pct, aux_engine_load_pct,
    distance_nm(optional), fuel_type(HFO/VLSFO/MDO/LNG), voyages(optional),
    cargo_tonnes(optional)

Key options
-----------
--asof 2025-09-06
--greek_countries "Greece,GR"       Owner-country tags considered 'Greek' (beneficial or direct)
--size_bins "Handy:0-40000,Supramax:40000-60000,Panamax:60000-80000,PostPanamax:80000-120000,Capesize:120000-220000,VLCC:220000-600000"
--age_bins "0-5,6-10,11-15,16-20,21-25,26-30,30+"
--ets_price 80                      EUA €/tCO2 (for ETS proxy)
--fuel_co2 "HFO:3.114,VLSFO:3.114,MDO:3.206,LNG:2.750"  tCO2 per tonne fuel (approx)
--sfoc "ME:180,AE:215"              g/kWh (defaults used if kW columns present)
--outdir out_greek_shipping

Notes & caveats
---------------
- This is a generic analytics framework. Data quality varies; sanity checks and guards are applied.
- ETS/FuelEU proxies are coarse and depend on activity & power inputs you provide.
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
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.floor("D")

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def yesno_series(s: pd.Series) -> pd.Series:
    def f(x):
        try:
            return 1 if float(x) > 0 else 0
        except Exception:
            return 1 if str(x).strip().lower() in {"y","yes","true","t","e","a"} else 0
    return s.apply(f)

def parse_bins(spec: str, numeric: bool = True) -> List[Tuple[str, float, float]]:
    """
    "Handy:0-40000,Supramax:40000-60000,..." -> [(label, lo, hi)]
    '30+' handled as (30, inf)
    """
    out = []
    if not spec: return out
    for part in spec.split(","):
        if ":" in part:
            label, rng = part.split(":", 1)
        else:
            label, rng = part, part
        rng = rng.strip()
        if rng.endswith("+"):
            lo = float(rng[:-1]); hi = float("inf")
        else:
            lo, hi = rng.split("-")
            lo = float(lo); hi = float(hi)
        out.append((label.strip(), lo, hi))
    return out

def bucketize(x: float, bins: List[Tuple[str, float, float]]) -> Optional[str]:
    if x is None or not np.isfinite(x):
        return None
    for label, lo, hi in bins:
        if lo <= x < hi:
            return label
    # include upper inclusive for last bin
    for label, lo, hi in bins[::-1]:
        if x >= lo and hi == float("inf"):
            return label
    return None

def vessel_type_group(t: str) -> str:
    if t is None: return "Other"
    x = str(t).lower()
    if any(k in x for k in ["tanker","crude","product","vlcc","suezmax","aframax","lr","mr"]): return "Tanker"
    if any(k in x for k in ["bulk","bulker","handy","cape","panamax","kamsarmax","supramax"]): return "DryBulk"
    if any(k in x for k in ["container","feeder","teu"]): return "Container"
    if any(k in x for k in ["lng","lpg","gas"]): return "Gas"
    if any(k in x for k in ["ro-ro","roro","vehicle","car carrier","pcc","pctc"]): return "RoRo/Car"
    if any(k in x for k in ["chemical","chem"]): return "Chemical"
    if any(k in x for k in ["offshore","supply","psv","osv","fpso","drill"]): return "Offshore"
    return "Other"

def greek_owner_flag(row, greek_countries: List[str]) -> int:
    tags = greek_countries
    for col in ["is_greek_owner","owner_country","beneficial_owner_country","manager_country"]:
        v = row.get(col, None)
        if col == "is_greek_owner" and pd.notna(v):
            try: 
                return 1 if float(v) > 0 else 0
            except Exception:
                pass
        if isinstance(v, str):
            v_up = v.strip().upper()
            if v_up in {t.upper() for t in tags}:
                return 1
    return 0

def safe_share(num, den) -> float:
    return float(num) / float(den) if (np.isfinite(num) and np.isfinite(den) and den > 0) else np.nan


# ----------------------------- loaders -----------------------------

def load_fleet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"imo") or df.columns[0]): "imo",
        (ncol(df,"mmsi") or "mmsi"): "mmsi",
        (ncol(df,"name") or "name"): "name",
        (ncol(df,"type") or "type"): "type",
        (ncol(df,"sub_type") or "sub_type"): "sub_type",
        (ncol(df,"class") or "class"): "class",
        (ncol(df,"dwt") or "dwt"): "dwt",
        (ncol(df,"gt") or "gt"): "gt",
        (ncol(df,"kw_main") or ncol(df,"kW_main") or "kw_main"): "kw_main",
        (ncol(df,"kw_aux") or ncol(df,"kW_aux") or "kw_aux"): "kw_aux",
        (ncol(df,"speed_kn") or "speed_kn"): "speed_kn",
        (ncol(df,"built_year") or ncol(df,"year_built") or "built_year"): "built_year",
        (ncol(df,"owner") or "owner"): "owner",
        (ncol(df,"owner_group") or "owner_group"): "owner_group",
        (ncol(df,"owner_country") or "owner_country"): "owner_country",
        (ncol(df,"manager") or "manager"): "manager",
        (ncol(df,"manager_country") or "manager_country"): "manager_country",
        (ncol(df,"beneficial_owner") or "beneficial_owner"): "beneficial_owner",
        (ncol(df,"beneficial_owner_country") or "beneficial_owner_country"): "beneficial_owner_country",
        (ncol(df,"flag") or "flag"): "flag",
        (ncol(df,"scrubber") or "scrubber"): "scrubber",
        (ncol(df,"eexi") or "eexi"): "eexi",
        (ncol(df,"cii_rating") or "cii_rating"): "cii_rating",
        (ncol(df,"is_greek_owner") or "is_greek_owner"): "is_greek_owner",
    }
    df = df.rename(columns=ren)
    for c in ["imo","mmsi","name","type","sub_type","class","owner","owner_group","owner_country",
              "manager","manager_country","beneficial_owner","beneficial_owner_country","flag","cii_rating"]:
        if c in df.columns: df[c] = df[c].astype(str)
    for c in ["dwt","gt","kw_main","kw_aux","speed_kn","built_year","eexi"]:
        if c in df.columns: df[c] = num_series(df[c])
    if "scrubber" in df.columns:
        df["scrubber"] = yesno_series(df["scrubber"])
    return df

def load_ports(path: Optional[str]) -> pd.DataFrame:
    if not path: 
        return pd.DataFrame(columns=["port","port_unlocode","country","region","is_greek"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"port") or ncol(df,"name") or df.columns[0]): "port",
        (ncol(df,"port_unlocode") or ncol(df,"unlocode") or "port_unlocode"): "port_unlocode",
        (ncol(df,"country") or "country"): "country",
        (ncol(df,"region") or "region"): "region",
        (ncol(df,"is_greek") or "is_greek"): "is_greek",
    }
    df = df.rename(columns=ren)
    if "is_greek" in df.columns:
        df["is_greek"] = yesno_series(df["is_greek"])
    return df

def load_ais(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"imo") or ncol(df,"mmsi") or df.columns[0]): "imo_or_mmsi",
        (ncol(df,"port") or df.columns[1] if len(df.columns)>1 else "port"): "port",
        (ncol(df,"port_unlocode") or ncol(df,"unlocode") or "port_unlocode"): "port_unlocode",
        (ncol(df,"country") or "country"): "country",
        (ncol(df,"region") or "region"): "region",
        (ncol(df,"arrival") or ncol(df,"ata") or "arrival"): "arrival",
        (ncol(df,"departure") or ncol(df,"atd") or "departure"): "departure",
        (ncol(df,"lat") or "lat"): "lat",
        (ncol(df,"lon") or ncol(df,"lng") or "lon"): "lon",
        (ncol(df,"status") or "status"): "status",
    }
    df = df.rename(columns=ren)
    df["arrival"] = to_dt(df["arrival"]) if "arrival" in df.columns else pd.NaT
    df["departure"] = to_dt(df["departure"]) if "departure" in df.columns else pd.NaT
    for c in ["lat","lon"]:
        if c in df.columns: df[c] = num_series(df[c])
    # best effort IMO
    if "imo_or_mmsi" in df.columns:
        df["imo"] = df["imo_or_mmsi"]
        df.drop(columns=["imo_or_mmsi"], inplace=True)
    return df

def load_table(path: Optional[str], renmap: Dict[str,str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    # flexible rename
    out = {}
    for want, fallback in renmap.items():
        cand = ncol(df, want) or ncol(df, fallback) or fallback
        if cand in df.columns: out[cand] = want
    return df.rename(columns=out)

def load_fixtures(path: Optional[str]) -> pd.DataFrame:
    return load_table(path, {"date":"date","route":"route","rate":"rate","unit":"unit","vessel_class":"vessel_class","vessel_type":"vessel_type"}).assign(date=lambda d: to_date(d["date"]) if "date" in d else pd.NaT)

def load_prices(path: Optional[str]) -> pd.DataFrame:
    return load_table(path, {"date":"date","series":"series","value":"value"}).assign(date=lambda d: to_date(d["date"]) if "date" in d else pd.NaT)

def load_orderbook(path: Optional[str]) -> pd.DataFrame:
    df = load_table(path, {"type":"type","class":"class","dwt":"dwt","gt":"gt","delivery_year":"delivery_year"})
    for c in ["dwt","gt","delivery_year"]:
        if c in df.columns: df[c] = num_series(df[c])
    return df

def load_demolition(path: Optional[str]) -> pd.DataFrame:
    df = load_table(path, {"imo":"imo","type":"type","class":"class","dwt":"dwt","gt":"gt","scrap_date":"scrap_date","price_usd_ltd":"price_usd_ltd"})
    for c in ["dwt","gt","price_usd_ltd"]:
        if c in df.columns: df[c] = num_series(df[c])
    if "scrap_date" in df.columns:
        df["scrap_date"] = to_date(df["scrap_date"])
    return df

def load_sanctions(path: Optional[str]) -> pd.DataFrame:
    df = load_table(path, {"imo":"imo","owner":"owner","owner_group":"owner_group","list":"list","date_listed":"date_listed"})
    if "date_listed" in df.columns:
        df["date_listed"] = to_date(df["date_listed"])
    return df

def load_activity(path: Optional[str]) -> pd.DataFrame:
    df = load_table(path, {
        "imo":"imo","date":"date","hours_at_sea":"hours_at_sea","hours_at_berth":"hours_at_berth",
        "avg_speed_kn":"avg_speed_kn","main_engine_load_pct":"main_engine_load_pct","aux_engine_load_pct":"aux_engine_load_pct",
        "distance_nm":"distance_nm","fuel_type":"fuel_type","voyages":"voyages","cargo_tonnes":"cargo_tonnes"
    })
    if "date" in df.columns: df["date"] = to_date(df["date"])
    for c in ["hours_at_sea","hours_at_berth","avg_speed_kn","main_engine_load_pct","aux_engine_load_pct","distance_nm","voyages","cargo_tonnes"]:
        if c in df.columns: df[c] = num_series(df[c])
    return df


# ----------------------------- core enrichment -----------------------------

def enrich_fleet(fleet: pd.DataFrame, greek_tags: List[str], size_bins: List[Tuple[str,float,float]], age_bins: List[Tuple[str,float,float]], asof: pd.Timestamp) -> pd.DataFrame:
    df = fleet.copy()
    df["type_group"] = df.get("type","").apply(vessel_type_group)
    df["dwt"] = num_series(df.get("dwt", np.nan))
    df["gt"] = num_series(df.get("gt", np.nan))
    df["built_year"] = num_series(df.get("built_year", np.nan))
    df["age"] = asof.year - df["built_year"]
    df["age_bin"] = [bucketize(a, age_bins) if np.isfinite(a) else None for a in df["age"]]
    df["size_class"] = [bucketize(float(d), size_bins) if np.isfinite(d) else None for d in df["dwt"]]
    df["is_greek"] = [greek_owner_flag(r, greek_tags) for _, r in df.iterrows()]
    # Scrubber, CII
    if "scrubber" in df.columns:
        df["scrubber"] = yesno_series(df["scrubber"])
    # EEXI compliance proxy (if present, treat <=1 as compliant)
    if "eexi" in df.columns:
        df["eexi_compliant"] = (num_series(df["eexi"]) <= 1.0).astype(int)
    # Clean IMO
    if "imo" in df.columns:
        df["imo"] = df["imo"].astype(str).str.extract(r"(\d+)", expand=False)
    return df

def owners_aggregate(fleet_en: pd.DataFrame, sanctions: pd.DataFrame) -> pd.DataFrame:
    df = fleet_en.copy()
    # Join sanctions by IMO or owner_group
    sanc = sanctions.copy()
    sanc["is_sanctioned"] = 1
    own = df.copy()
    if not sanc.empty:
        own = own.merge(sanc[["imo","is_sanctioned"]], on="imo", how="left")
        own["vessel_sanctioned"] = own["is_sanctioned"].fillna(0).astype(int)
        del sanc["is_sanctioned"]
        if "owner_group" in df.columns and "owner_group" in sanctions.columns:
            g2 = sanctions.groupby("owner_group", as_index=False).agg(owner_sanctioned=("owner_group","size"))
            own = own.merge(g2, on="owner_group", how="left")
            own["owner_sanctioned"] = (own["owner_sanctioned"].fillna(0) > 0).astype(int)
        else:
            own["owner_sanctioned"] = 0
    else:
        own["vessel_sanctioned"] = 0
        own["owner_sanctioned"] = 0
    g = (own.groupby(["owner_group","owner_country"], as_index=False)
            .agg(vessels=("imo","nunique"),
                 dwt=("dwt","sum"),
                 gt=("gt","sum"),
                 greek_owned=("is_greek","sum"),
                 avg_age=("age","mean"),
                 scrubber_share=("scrubber","mean") if "scrubber" in own.columns else ("is_greek","size"),  # placeholder if missing
                 cii_E_share=(own.get("cii_rating", pd.Series(index=own.index)).astype(str).str.upper()=="E").mean() if "cii_rating" in own.columns else 0.0,
                 vessel_sanctioned=("vessel_sanctioned","sum"),
                 owner_sanctioned=("owner_sanctioned","max")))
    if "scrubber" not in own.columns:
        g = g.rename(columns={"scrubber_share":"_tmp"}); g["_tmp"] = np.nan; g = g.drop(columns=["_tmp"])
    return g.sort_values("dwt", ascending=False)

def type_aggregate(fleet_en: pd.DataFrame, orderbook: pd.DataFrame) -> pd.DataFrame:
    f = fleet_en.copy()
    g = (f.groupby(["type_group","size_class"], as_index=False)
           .agg(vessels=("imo","nunique"),
                dwt=("dwt","sum"),
                gt=("gt","sum"),
                avg_age=("age","mean"),
                greek_dwt=("is_greek","sum")))
    g["greek_dwt_share"] = g["greek_dwt"] / g["dwt"].replace(0, np.nan)
    # Orderbook coverage by type
    if not orderbook.empty:
        ob = orderbook.copy()
        ob["type_group"] = ob["type"].apply(vessel_type_group) if "type" in ob.columns else "Other"
        ob_g = ob.groupby("type_group", as_index=False).agg(ob_dwt=("dwt","sum"), ob_gt=("gt","sum"))
        type_tot = f.groupby("type_group", as_index=False).agg(in_dwt=("dwt","sum"))
        cov = type_tot.merge(ob_g, on="type_group", how="left")
        cov["orderbook_coverage"] = cov["ob_dwt"] / cov["in_dwt"].replace(0, np.nan)
        g = g.merge(cov[["type_group","orderbook_coverage"]], on="type_group", how="left")
    return g.sort_values(["type_group","size_class"])

def age_distribution(fleet_en: pd.DataFrame, age_bins: List[Tuple[str,float,float]]) -> pd.DataFrame:
    f = fleet_en.copy()
    f["age_bin"] = [bucketize(a, age_bins) if np.isfinite(a) else None for a in f["age"]]
    g = (f.groupby(["type_group","age_bin"], as_index=False)
           .agg(vessels=("imo","nunique"),
                dwt=("dwt","sum")))
    # fill missing labels order
    return g.sort_values(["type_group","age_bin"])

def sanctions_exposure(fleet_en: pd.DataFrame, sanctions: pd.DataFrame) -> pd.DataFrame:
    if sanctions.empty:
        return pd.DataFrame(columns=["scope","count","dwt"])
    sanc = sanctions.copy()
    sanc["is_sanctioned"] = 1
    f = fleet_en.copy().merge(sanc[["imo","is_sanctioned"]], on="imo", how="left")
    f["is_sanctioned"] = f["is_sanctioned"].fillna(0).astype(int)
    rows = []
    for scope, mask in [("All", np.ones(len(f), dtype=bool)), ("Greek-owned", f["is_greek"]==1)]:
        sub = f[mask]
        rows.append({"scope": scope, "count": int(sub["is_sanctioned"].sum()),
                     "dwt": float(sub.loc[sub["is_sanctioned"]==1, "dwt"].sum())})
    return pd.DataFrame(rows)

# ----------------------------- AIS & ports -----------------------------

def clean_port_calls(ais: pd.DataFrame, ports: pd.DataFrame) -> pd.DataFrame:
    if ais.empty:
        return pd.DataFrame()
    df = ais.copy()
    # Arrival/Departure pairing
    df["arrival"] = to_dt(df["arrival"]) if "arrival" in df.columns else pd.NaT
    df["departure"] = to_dt(df["departure"]) if "departure" in df.columns else pd.NaT
    # If departure missing but next arrival exists for same vessel/port, try to infer minimal dwell of 6h
    df["imo"] = df.get("imo", "").astype(str).str.extract(r"(\d+)", expand=False)
    # Join ports metadata
    if not ports.empty:
        df = df.merge(ports[["port","port_unlocode","country","region","is_greek"]].drop_duplicates(subset=["port","port_unlocode"]),
                      on=["port","port_unlocode","country","region"], how="left")
    else:
        df["is_greek"] = (df.get("country","").astype(str).str.upper().isin({"GREECE","GR"})).astype(int)
    df["arr_date"] = df["arrival"].dt.date
    df["dep_date"] = df["departure"].dt.date
    df["dwell_hours"] = (df["departure"] - df["arrival"]).dt.total_seconds() / 3600.0
    # guardrails
    df.loc[df["dwell_hours"] < 0, "dwell_hours"] = np.nan
    df["greek_port"] = df["is_greek"].fillna(0).astype(int)
    return df.sort_values(["imo","arrival"]).reset_index(drop=True)

def build_network_edges(calls: pd.DataFrame) -> pd.DataFrame:
    if calls.empty:
        return pd.DataFrame()
    df = calls.sort_values(["imo","arrival"]).copy()
    rows = []
    for imo, g in df.groupby("imo"):
        g = g.dropna(subset=["port","arrival"]).sort_values("arrival")
        prev_port, prev_dep = None, None
        for _, r in g.iterrows():
            this_port, this_arr = r.get("port"), r.get("arrival")
            if prev_port and this_port and prev_port != this_port and pd.notna(prev_dep) and pd.notna(this_arr):
                transit_h = (this_arr - prev_dep).total_seconds()/3600.0
                rows.append({"imo": imo, "src_port": prev_port, "dst_port": this_port,
                             "departed": prev_dep, "arrived": this_arr, "transit_hours": transit_h})
            prev_port, prev_dep = this_port, r.get("departure")
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    agg = (edges.groupby(["src_port","dst_port"], as_index=False)
             .agg(flows=("imo","nunique"),
                  voyages=("imo","size"),
                  avg_transit_h=("transit_hours","mean"),
                  p90_transit_h=("transit_hours", lambda s: float(np.nanpercentile(s.dropna(), 90)) if s.notna().any() else np.nan)))
    return agg.sort_values("voyages", ascending=False)

def port_congestion(calls: pd.DataFrame) -> pd.DataFrame:
    if calls.empty: return pd.DataFrame()
    g = (calls.groupby(["port","country","region"], as_index=False)
           .agg(calls=("imo","size"),
                median_dwell_h=("dwell_hours","median"),
                p90_dwell_h=("dwell_hours", lambda s: float(np.nanpercentile(s.dropna(), 90)) if s.notna().any() else np.nan),
                greek_port=("greek_port","max")))
    return g.sort_values("calls", ascending=False)

# ----------------------------- Markets & regressions -----------------------------

def daily_series_from_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty: return pd.DataFrame()
    fx = fixtures.copy()
    fx = fx.dropna(subset=["date","route","rate"])
    fx["series"] = fx["route"].astype(str)
    return fx[["date","series","rate"]].rename(columns={"rate":"value"})

def wide_pivot(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty: return pd.DataFrame()
    p = prices.copy().dropna(subset=["date","series","value"])
    wide = p.pivot_table(index="date", columns="series", values="value", aggfunc="mean").sort_index()
    wide.columns.name = None
    return wide.reset_index()

def build_congestion_index(cong: pd.DataFrame) -> pd.DataFrame:
    if cong.empty: return pd.DataFrame()
    c = cong.copy()
    # Greek congestion index: median of (p90 dwell) across Greek ports weighted by calls
    c["weight"] = c["calls"] / c["calls"].sum()
    gr = c[c["greek_port"]==1]
    idx = float(np.nansum(gr["p90_dwell_h"] * gr["weight"])) if not gr.empty else np.nan
    return pd.DataFrame([{"date": None, "GreekCongestionIndex": idx}])  # date-less snapshot if ports not time-stamped

def regress_freight(prices_wide: pd.DataFrame, congestion_idx: Optional[pd.Series], sanctions_daily: Optional[pd.Series]) -> pd.DataFrame:
    """
    For each freight series present (e.g., TD3C, C5, BDI) regress Δlog(rate) on ΔBrent + Δcongestion + Δsanctions.
    """
    if prices_wide.empty: return pd.DataFrame()
    df = prices_wide.copy().sort_values("date")
    # Identify candidate freight series (exclude macro)
    macro = {"Brent","EUA","EURUSD","TTF"}
    candidates = [c for c in df.columns if c not in {"date"} | macro]
    rows = []
    for s in candidates:
        y = np.log(df[s]).diff()
        X_list = []
        names = []
        if "Brent" in df.columns:
            X_list.append(np.log(df["Brent"]).diff()); names.append("dlog_Brent")
        if congestion_idx is not None and congestion_idx.notna().any():
            # align by date if time series; if static, ignore
            if "GreekCongestion" in df.columns:
                X_list.append(df["GreekCongestion"].diff()); names.append("dCongestion")
        if sanctions_daily is not None and sanctions_daily.notna().any():
            if "SanctionsCount" in df.columns:
                X_list.append(df["SanctionsCount"].diff()); names.append("dSanctions")
        if not X_list: 
            continue
        X = pd.concat(X_list, axis=1).dropna()
        Y = y.loc[X.index]
        X_ = np.column_stack([np.ones(len(X))] + [X.iloc[:,i].values for i in range(X.shape[1])])
        try:
            beta, *_ = np.linalg.lstsq(X_, Y.values, rcond=None)
            yhat = X_ @ beta
            r2 = 1.0 - np.sum((Y.values - yhat)**2) / max(1e-12, np.sum((Y.values - Y.values.mean())**2))
            out = {"series": s, "n": int(len(Y)), "r2": float(r2)}
            for i, nm in enumerate(["const"] + names):
                out[f"beta_{nm}"] = float(beta[i])
            rows.append(out)
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("r2", ascending=False)

# ----------------------------- ETS / FuelEU proxies -----------------------------

def ets_proxy(activity: pd.DataFrame, fleet_en: pd.DataFrame, ets_price: float, fuel_co2: Dict[str,float], sfoc_g_per_kwh: Dict[str,float]) -> pd.DataFrame:
    """
    Very coarse ETS proxy:
      fuel_main_t = ME_kW * load * hours * SFOC / 1e6
      fuel_aux_t  = AE_kW * load * hours * SFOC / 1e6
      CO2_t = fuel_t * factor(fuel_type)
      ETS_cost = CO2_t * ets_price
    """
    if activity.empty:
        return pd.DataFrame()
    act = activity.copy()
    f = fleet_en[["imo","kw_main","kw_aux","type_group","dwt","gt"]].copy()
    merged = act.merge(f, on="imo", how="left")
    merged["kw_main"] = num_series(merged["kw_main"])
    merged["kw_aux"] = num_series(merged["kw_aux"])
    merged["main_engine_load_pct"] = merged.get("main_engine_load_pct", 60.0).fillna(60.0)
    merged["aux_engine_load_pct"] = merged.get("aux_engine_load_pct", 40.0).fillna(40.0)
    merged["hours_at_sea"] = merged.get("hours_at_sea", 0.0).fillna(0.0)
    merged["hours_at_berth"] = merged.get("hours_at_berth", 0.0).fillna(0.0)
    merged["fuel_type"] = merged.get("fuel_type","HFO").fillna("HFO").astype(str).str.upper()
    # compute fuel
    me_sfoc = sfoc_g_per_kwh.get("ME", 180.0)
    ae_sfoc = sfoc_g_per_kwh.get("AE", 215.0)
    merged["fuel_main_t"] = (merged["kw_main"] * (merged["main_engine_load_pct"]/100.0) * merged["hours_at_sea"] * me_sfoc) / 1e6
    merged["fuel_aux_t"]  = (merged["kw_aux"] * (merged["aux_engine_load_pct"]/100.0) * (merged["hours_at_sea"] + merged["hours_at_berth"]) * ae_sfoc) / 1e6
    merged["fuel_t"] = merged[["fuel_main_t","fuel_aux_t"]].sum(axis=1, skipna=True)
    def co2_factor(ft: str) -> float:
        key = "HFO"
        if isinstance(ft, str):
            s = ft.upper()
            if "LNG" in s: key = "LNG"
            elif "MDO" in s or "MGO" in s or "DMA" in s: key = "MDO"
            elif "VLSFO" in s or "LSFO" in s: key = "VLSFO"
            else: key = "HFO"
        return float(fuel_co2.get(key, fuel_co2["HFO"]))
    merged["co2_t"] = merged.apply(lambda r: r["fuel_t"] * co2_factor(r["fuel_type"]), axis=1)
    merged["ets_cost_eur"] = merged["co2_t"] * float(ets_price)
    keep = ["imo","date","type_group","dwt","gt","hours_at_sea","hours_at_berth","fuel_type","fuel_t","co2_t","ets_cost_eur"]
    return merged[keep].sort_values(["imo","date"])

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    fleet: str
    ais: Optional[str]
    ports: Optional[str]
    fixtures: Optional[str]
    prices: Optional[str]
    orderbook: Optional[str]
    demolition: Optional[str]
    sanctions: Optional[str]
    activity: Optional[str]
    asof: str
    greek_countries: List[str]
    size_bins: List[Tuple[str,float,float]]
    age_bins: List[Tuple[str,float,float]]
    ets_price: float
    fuel_co2: Dict[str,float]
    sfoc: Dict[str,float]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Greek shipping analytics")
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--ais", default="")
    ap.add_argument("--ports", default="")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--orderbook", default="")
    ap.add_argument("--demolition", default="")
    ap.add_argument("--sanctions", default="")
    ap.add_argument("--activity", default="")
    ap.add_argument("--asof", default="2025-09-06")
    ap.add_argument("--greek_countries", default="Greece,GR")
    ap.add_argument("--size_bins", default="Handy:0-40000,Supramax:40000-60000,Panamax:60000-80000,PostPanamax:80000-120000,Capesize:120000-220000,VLCC:220000-600000")
    ap.add_argument("--age_bins", default="0-5,6-10,11-15,16-20,21-25,26-30,30+")
    ap.add_argument("--ets_price", type=float, default=80.0)
    ap.add_argument("--fuel_co2", default="HFO:3.114,VLSFO:3.114,MDO:3.206,LNG:2.750")
    ap.add_argument("--sfoc", default="ME:180,AE:215")
    ap.add_argument("--outdir", default="out_greek_shipping")
    return ap.parse_args()

def parse_kv_floats(spec: str) -> Dict[str,float]:
    out: Dict[str,float] = {}
    if not spec: return out
    for part in spec.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            try:
                out[k.strip()] = float(v)
            except Exception:
                continue
    return out

def parse_greek_tags(spec: str) -> List[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    asof = pd.to_datetime(args.asof)

    size_bins = parse_bins(args.size_bins)
    # Age bins: e.g., "0-5,6-10,...,30+" -> label = spec
    age_bins = []
    for part in args.age_bins.split(","):
        part = part.strip()
        if not part: continue
        if part.endswith("+"):
            lo = float(part[:-1]); hi = float("inf")
        else:
            lo, hi = part.split("-"); lo = float(lo); hi = float(hi)
        age_bins.append((part, lo, hi))

    greek_tags = parse_greek_tags(args.greek_countries)
    fuel_co2 = parse_kv_floats(args.fuel_co2)
    sfoc = parse_kv_floats(args.sfoc)
    if "HFO" not in fuel_co2: fuel_co2["HFO"] = 3.114

    # Load
    fleet = load_fleet(args.fleet)
    ports = load_ports(args.ports)
    ais = load_ais(args.ais) if args.ais else pd.DataFrame()
    fixtures = load_fixtures(args.fixtures) if args.fixtures else pd.DataFrame()
    prices = load_prices(args.prices) if args.prices else pd.DataFrame()
    orderbook = load_orderbook(args.orderbook) if args.orderbook else pd.DataFrame()
    demolition = load_demolition(args.demolition) if args.demolition else pd.DataFrame()
    sanctions = load_sanctions(args.sanctions) if args.sanctions else pd.DataFrame()
    activity = load_activity(args.activity) if args.activity else pd.DataFrame()

    # Enrich fleet
    fleet_en = enrich_fleet(fleet, greek_tags, size_bins, age_bins, asof)
    fleet_en.to_csv(outdir / "fleet_enriched.csv", index=False)

    # Owners
    owners = owners_aggregate(fleet_en, sanctions)
    owners.to_csv(outdir / "owners.csv", index=False)

    # Type aggregates
    by_type = type_aggregate(fleet_en, orderbook)
    by_type.to_csv(outdir / "fleet_by_type.csv", index=False)

    # Age distribution
    ages = age_distribution(fleet_en, age_bins)
    ages.to_csv(outdir / "age_distribution.csv", index=False)

    # Orderbook coverage details
    if not orderbook.empty:
        ob = orderbook.copy()
        ob["type_group"] = ob["type"].apply(vessel_type_group) if "type" in ob.columns else "Other"
        ob_cov = (ob.groupby(["type_group","delivery_year"], as_index=False)
                    .agg(ob_dwt=("dwt","sum"), ob_gt=("gt","sum")))
        ob_cov.to_csv(outdir / "orderbook_coverage.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "orderbook_coverage.csv", index=False)

    # Sanctions exposure
    sanc_expo = sanctions_exposure(fleet_en, sanctions)
    sanc_expo.to_csv(outdir / "sanctions_exposure.csv", index=False)

    # AIS port calls, congestion, network
    if not ais.empty:
        calls = clean_port_calls(ais, ports)
        calls.to_csv(outdir / "port_calls.csv", index=False)
        cong = port_congestion(calls)
        cong.to_csv(outdir / "congestion.csv", index=False)
        edges = build_network_edges(calls)
        edges.to_csv(outdir / "network_edges.csv", index=False)
        greek_calls = int(calls["greek_port"].sum())
    else:
        calls = pd.DataFrame(); cong = pd.DataFrame(); edges = pd.DataFrame()
        greek_calls = 0

    # Markets: merge fixtures into prices if needed
    prices_all = prices.copy()
    if not fixtures.empty:
        fx = daily_series_from_fixtures(fixtures)
        prices_all = pd.concat([prices_all, fx], ignore_index=True) if not prices_all.empty else fx
    # build wide
    prices_wide = wide_pivot(prices_all) if not prices_all.empty else pd.DataFrame()

    # Optional congestion index time series (static snapshot here; if you have time-stamped port_congestion by date, extend logic)
    if not cong.empty:
        # Use a static latest Greek congestion level; add as a column for regression diff=0 (won't help much).
        if not prices_wide.empty:
            prices_wide["GreekCongestion"] = float(np.nanmean(cong.loc[cong["greek_port"]==1, "p90_dwell_h"]))
    # Sanctions daily count — if sanctions include date_listed
    if not sanctions.empty and "date_listed" in sanctions.columns:
        sanc_daily = sanctions.groupby("date_listed").size().rename("SanctionsCount").reset_index()
        if not prices_wide.empty:
            prices_wide = prices_wide.merge(sanc_daily.rename(columns={"date_listed":"date"}), on="date", how="left").fillna({"SanctionsCount":0})
    else:
        sanc_daily = pd.DataFrame()

    # Freight regressions
    regs = regress_freight(prices_wide, prices_wide.get("GreekCongestion") if "GreekCongestion" in prices_wide.columns else None,
                           prices_wide.get("SanctionsCount") if "SanctionsCount" in prices_wide.columns else None) if not prices_wide.empty else pd.DataFrame()
    regs.to_csv(outdir / "freight_regressions.csv", index=False)

    # ETS proxy
    if not activity.empty:
        ets = ets_proxy(activity, fleet_en, float(args.ets_price), fuel_co2, sfoc)
        ets.to_csv(outdir / "ets_costs.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "ets_costs.csv", index=False)

    # Summary KPIs
    world_dwt = float(fleet_en["dwt"].sum()) if "dwt" in fleet_en.columns else np.nan
    gr_dwt = float(fleet_en.loc[fleet_en["is_greek"]==1, "dwt"].sum()) if "dwt" in fleet_en.columns else np.nan
    kpi = {
        "asof": args.asof,
        "vessels_total": int(fleet_en["imo"].nunique()) if "imo" in fleet_en.columns else int(len(fleet_en)),
        "vessels_greek_owned": int(fleet_en.loc[fleet_en["is_greek"]==1, "imo"].nunique()) if "imo" in fleet_en.columns else int(fleet_en["is_greek"].sum()),
        "dwt_total": world_dwt,
        "dwt_greek": gr_dwt,
        "dwt_greek_share": safe_share(gr_dwt, world_dwt),
        "avg_age_total": float(fleet_en["age"].mean()) if "age" in fleet_en.columns else None,
        "avg_age_greek": float(fleet_en.loc[fleet_en["is_greek"]==1, "age"].mean()) if "age" in fleet_en.columns else None,
        "scrubber_share_total": float(fleet_en["scrubber"].mean()) if "scrubber" in fleet_en.columns else None,
        "orderbook_coverage_tanker": float(by_type.loc[by_type["type_group"]=="Tanker","orderbook_coverage"].mean()) if "orderbook_coverage" in by_type.columns else None,
        "orderbook_coverage_drybulk": float(by_type.loc[by_type["type_group"]=="DryBulk","orderbook_coverage"].mean()) if "orderbook_coverage" in by_type.columns else None,
        "greek_port_calls": greek_calls,
        "top_owner_groups_by_dwt": (owners.sort_values("dwt", ascending=False).head(10)[["owner_group","dwt"]].to_dict(orient="records") if not owners.empty else []),
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))

    # Config dump
    cfg = asdict(Config(
        fleet=args.fleet, ais=(args.ais or None), ports=(args.ports or None),
        fixtures=(args.fixtures or None), prices=(args.prices or None),
        orderbook=(args.orderbook or None), demolition=(args.demolition or None),
        sanctions=(args.sanctions or None), activity=(args.activity or None),
        asof=args.asof, greek_countries=greek_tags, size_bins=size_bins, age_bins=age_bins,
        ets_price=float(args.ets_price), fuel_co2=fuel_co2, sfoc=sfoc, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Greek Shipping Analytics ==")
    print(f"As of {args.asof}: {kpi['vessels_greek_owned']}/{kpi['vessels_total']} vessels greek-owned; DWT share ≈ {kpi['dwt_greek_share']*100 if kpi['dwt_greek_share']==kpi['dwt_greek_share'] else float('nan'):.2f}%")
    if regs is not None and not regs.empty:
        top = regs.head(3).to_dict(orient="records")
        print("Sample freight regressions:", top)
    if greek_calls:
        print(f"Greek port calls in AIS file: {greek_calls:,}")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
