#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
french_agri_subsidies.py — France CAP (2023–2027) style payment emulator & distribution analytics

What this does
--------------
Given a farm registry and optional inputs, this script estimates *indicative* CAP-style payments for
French holdings and produces distribution/region/crop summaries and scenario comparisons.

It is a **policy emulation framework** (not legal advice, not an official calculator). Defaults are
generic and configurable—bring your own parameter file to reflect the rules you care about.

Supported payment “buckets” (simplified names)
- BISS: Basic Income Support for Sustainability (flat €/ha after internal convergence)
- EcoScheme: Eco-schemes (practice-based points → €/ha)
- CRISS: Complementary Redistributive Income Support (first-N hectares top-up)
- YF: Young farmer top-up (CIS-YF; age/start-year conditions, first-N ha)
- Coupled_LU: Coupled livestock (dairy/beef/sheep/goat LU × €/LU)
- Coupled_Crops: Coupled crops (e.g., protein crops €/ha)
- AECM: Agri-environment & climate measures (proxy via flags like organic/HNV/Natura)
- ANC/ICHN: Areas with Natural Constraints (mountain/handicap proxy)
- Degressivity/Capping: Reductions above thresholds (after labour proxy if provided)
- Penalties: Conditionality/GAEC penalty factor (if provided per farm)

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--farms farms.csv  (required)
    Suggested columns (use what you have; script is robust to missing):
      farm_id, region, departement, area_eligible_ha, area_grass_ha, area_arable_ha,
      area_permanent_crops_ha, organic(0/1), hnv(0/1), natura2000(0/1), mountain_zone(0/1),
      livestock_lu, lu_dairy, lu_beef, lu_sheep, lu_goat,
      area_protein_ha, irrigation(0/1), slope_share, hedgerow_km,
      crop_mix   e.g. "wheat:50,barley:20,oilseed:10,maize:10,other:10" (percent of arable area)
      age, start_year, hired_labour_awu (annual work units), conditionality_factor (0..1)

--payments payments.csv (optional; historical payments for benchmarking)
    Columns: farm_id, year, scheme, amount_eur

--params params.yaml|json|csv (optional)
    Key/value overrides for the defaults below; see "DEFAULT_PARAMS" in code for structure.

--scenarios scenarios.csv (optional; multiple scenario overrides)
    Columns: scenario,name,key,value
    Example rows:
      s1,CRISS more generous,criss.rate_eur_per_ha,60
      s1,CRISS more generous,criss.first_ha,60
      s2,Eco stricter,ecoscheme.points.organic,40

--prices prices.csv (optional; not required, used for context KPIs)
    Columns: year, fert_index, feed_index, diesel_index

Key CLI options
---------------
--year 2025                 Year context (for age/start-year checks)
--outdir out_fr_cap
--scenario "baseline"       Scenario id to run (must exist in scenarios.csv if not 'baseline')

Outputs
-------
- payments_baseline.csv     Per-farm breakdown by scheme + totals (for chosen scenario)
- distribution.csv          Deciles, Gini, top 10% share, per scheme and total
- regions.csv               Regional aggregates (sum, €/ha avg), per scheme and total
- crops_livestock.csv       Support by crop/coupled area & livestock class
- scenario_comparison.csv   If scenarios.csv provided: Δ vs baseline per farm and aggregates
- summary.json              KPIs (coverage, averages, inequality)
- config.json               Run configuration and parameters used

Notes & disclaimers
-------------------
- Numbers are *illustrative*. Real CAP payments depend on full eligibility, historical rights,
  controls, and department-level details not modeled here.
- Eco-scheme scoring in this tool is a simple points-to-€/ha heuristic you can tailor via params.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Defaults -----------------------------

DEFAULT_PARAMS = {
    "biss": {
        "rate_eur_per_ha": 120.0,    # flat after convergence; override with your table if needed
        "use_entitlements_column": False,  # if True, expect column 'entitlement_rate_eur_per_ha'
        "cap_eligible_ha_to_area": True
    },
    "ecoscheme": {
        "max_rate_eur_per_ha": 75.0,
        "points": {
            "organic": 100,           # if organic == 1
            "grass_share_ge_60": 40,  # grass share ≥ 60% of UAA
            "low_stocking_density": 30,  # LU per eligible ha ≤ 1.0
            "diversification": 20,    # ≥ 3 crops & HHI <= 0.5
            "hedgerows_per_ha_ge_0_02": 10,  # ≥ 0.02 km hedgerow per ha
        },
        "point_to_rate": {            # piecewise mapping (points >= key) → €/ha fraction of max
            "100": 1.00, "70": 0.70, "50": 0.50, "30": 0.30
        }
    },
    "criss": {
        "rate_eur_per_ha": 50.0,
        "first_ha": 52.0              # top-up for first N ha
    },
    "young_farmer": {
        "eligible_age_max": 40,
        "years_since_start_max": 5,
        "rate_eur_per_ha": 70.0,
        "first_ha": 40.0
    },
    "coupled": {
        "livestock": {                # €/LU
            "dairy": 80.0, "beef": 150.0, "sheep": 170.0, "goat": 120.0
        },
        "crops": {                    # €/ha by coupled crop area proxies
            "protein": 100.0
        },
        "caps": {
            "protein_ha_max": None,   # e.g., 50 to cap supported hectares per farm
            "lu_max": None            # cap total eligible LU for coupled support
        }
    },
    "aecm": {
        "organic_bonus_eur_per_ha": 40.0,
        "hnv_bonus_eur_per_ha": 25.0,
        "natura_bonus_eur_per_ha": 20.0
    },
    "anc": {
        "mountain_rate_eur_per_ha": 35.0
    },
    "reductions": {
        "degressivity_start": 60000.0,   # soft reduction above this
        "degressivity_rate": 0.20,       # 20% on slice between start & cap
        "capping_threshold": 100000.0,   # hard cap above this
        "allow_labour_deduction": True,  # subtract labour proxy before applying degressivity/cap
        "labour_value_per_awu": 25000.0  # proxy € per AWU to deduct if available
    },
    "penalties": {
        "use_conditionality_factor": True,   # multiply all direct payments by 'conditionality_factor' (0..1) if provided
        "default_factor": 1.0
    },
    "aggregates": {
        "region_column": "region"
    }
}

# ----------------------------- Utils -----------------------------

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None

def yesno(x) -> int:
    try:
        return 1 if float(x) > 0 else 0
    except Exception:
        s = str(x).strip().lower()
        return 1 if s in {"y","yes","true","t","1"} else 0

def num(x, default=np.nan) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default

def series_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def to_crop_mix_map(s: str) -> Dict[str, float]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {}
    out = {}
    for part in str(s).split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip().lower()] = float(v)
            except Exception:
                continue
    # normalize if looks like percentages
    tot = sum(out.values())
    if tot and tot > 1.5:  # likely in percents
        out = {k: v / 100.0 for k, v in out.items()}
    return out

def crop_diversity_metrics(crop_mix_map: Dict[str, float]) -> Tuple[int, float]:
    """Return (# crops, Herfindahl index) based on shares (0..1)."""
    if not crop_mix_map:
        return 0, np.nan
    shares = np.array(list(crop_mix_map.values()), dtype=float)
    hhi = float(np.sum(np.square(shares)))
    return int(len(shares)), hhi

def gini(x: np.ndarray) -> float:
    """Gini coefficient (0..1)."""
    arr = np.array(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

# ----------------------------- Loaders -----------------------------

def load_params(path: Optional[str]) -> Dict:
    if not path:
        return DEFAULT_PARAMS
    p = DEFAULT_PARAMS.copy()
    # Try yaml, json, csv
    try:
        if path.lower().endswith((".yaml",".yml")):
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
        elif path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                user = json.load(f)
        else:
            # CSV: key,value with dot-keys (e.g., criss.first_ha,60)
            user = {}
            df = pd.read_csv(path)
            keyc = ncol(df, "key") or df.columns[0]
            valc = ncol(df, "value") or df.columns[1]
            for _, r in df.iterrows():
                user[str(r[keyc])] = r[valc]
        # Merge
        def set_by_dot(d, dotkey, value):
            parts = str(dotkey).split(".")
            cur = d
            for k in parts[:-1]:
                if k not in cur or not isinstance(cur[k], dict):
                    cur[k] = {}
                cur = cur[k]
            # cast numeric if possible
            v = value
            try:
                v = float(value)
                if v.is_integer():
                    v = int(v)
            except Exception:
                v = value
            cur[parts[-1]] = v
        if isinstance(user, dict):
            for k, v in user.items():
                if isinstance(v, dict):
                    p.setdefault(k, {}).update(v)
                else:
                    set_by_dot(p, k, v)
        return p
    except Exception:
        return DEFAULT_PARAMS

def load_farms(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"farm_id") or df.columns[0]): "farm_id",
        (ncol(df,"region") or "region"): "region",
        (ncol(df,"departement") or ncol(df,"department") or "departement"): "departement",
        (ncol(df,"area_eligible_ha") or ncol(df,"eligible_ha") or "area_eligible_ha"): "area_eligible_ha",
        (ncol(df,"area_grass_ha") or "area_grass_ha"): "area_grass_ha",
        (ncol(df,"area_arable_ha") or "area_arable_ha"): "area_arable_ha",
        (ncol(df,"area_permanent_crops_ha") or "area_permanent_crops_ha"): "area_permanent_crops_ha",
        (ncol(df,"organic") or "organic"): "organic",
        (ncol(df,"hnv") or "hnv"): "hnv",
        (ncol(df,"natura2000") or ncol(df,"natura") or "natura2000"): "natura2000",
        (ncol(df,"mountain_zone") or ncol(df,"anc") or "mountain_zone"): "mountain_zone",
        (ncol(df,"livestock_lu") or ncol(df,"lu_total") or "livestock_lu"): "livestock_lu",
        (ncol(df,"lu_dairy") or "lu_dairy"): "lu_dairy",
        (ncol(df,"lu_beef") or "lu_beef"): "lu_beef",
        (ncol(df,"lu_sheep") or "lu_sheep"): "lu_sheep",
        (ncol(df,"lu_goat") or "lu_goat"): "lu_goat",
        (ncol(df,"area_protein_ha") or "area_protein_ha"): "area_protein_ha",
        (ncol(df,"irrigation") or "irrigation"): "irrigation",
        (ncol(df,"slope_share") or "slope_share"): "slope_share",
        (ncol(df,"hedgerow_km") or ncol(df,"hedgerows_km") or "hedgerow_km"): "hedgerow_km",
        (ncol(df,"crop_mix") or "crop_mix"): "crop_mix",
        (ncol(df,"age") or "age"): "age",
        (ncol(df,"start_year") or "start_year"): "start_year",
        (ncol(df,"hired_labour_awu") or "hired_labour_awu"): "hired_labour_awu",
        (ncol(df,"conditionality_factor") or "conditionality_factor"): "conditionality_factor",
        (ncol(df,"entitlement_rate_eur_per_ha") or "entitlement_rate_eur_per_ha"): "entitlement_rate_eur_per_ha",
    }
    df = df.rename(columns=ren)
    # Numeric conversions
    for c in ["area_eligible_ha","area_grass_ha","area_arable_ha","area_permanent_crops_ha",
              "livestock_lu","lu_dairy","lu_beef","lu_sheep","lu_goat",
              "area_protein_ha","slope_share","hedgerow_km","age","start_year",
              "hired_labour_awu","conditionality_factor","entitlement_rate_eur_per_ha"]:
        if c in df.columns: df[c] = series_num(df[c])
    for b in ["organic","hnv","natura2000","mountain_zone","irrigation"]:
        if b in df.columns: df[b] = df[b].apply(yesno)
    df["area_eligible_ha"] = df.get("area_eligible_ha", 0).fillna(0.0).clip(lower=0.0)
    # Derived
    df["uaa_ha"] = df.get("area_grass_ha", 0).fillna(0) + df.get("area_arable_ha", 0).fillna(0) + df.get("area_permanent_crops_ha", 0).fillna(0)
    df["crop_mix_map"] = df.get("crop_mix", "").apply(to_crop_mix_map)
    # Diversity
    div = df["crop_mix_map"].apply(crop_diversity_metrics)
    df["n_crops"] = [t[0] for t in div]
    df["hhi"] = [t[1] for t in div]
    # Safe LU split
    for k in ["lu_dairy","lu_beef","lu_sheep","lu_goat"]:
        df[k] = df.get(k, 0).fillna(0)
    if "livestock_lu" not in df.columns or df["livestock_lu"].isna().all():
        df["livestock_lu"] = df[["lu_dairy","lu_beef","lu_sheep","lu_goat"]].sum(axis=1)
    return df

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df,"scenario") or "scenario"): "scenario",
        (ncol(df,"name") or "name"): "name",
        (ncol(df,"key") or "key"): "key",
        (ncol(df,"value") or "value"): "value",
    })
    return df

# ----------------------------- Core calculations -----------------------------

def calc_biss(row: pd.Series, P: Dict) -> float:
    area = float(row.get("area_eligible_ha", 0.0))
    if P["biss"].get("use_entitlements_column", False) and not np.isnan(row.get("entitlement_rate_eur_per_ha", np.nan)):
        rate = float(row["entitlement_rate_eur_per_ha"])
    else:
        rate = float(P["biss"]["rate_eur_per_ha"])
    if P["biss"].get("cap_eligible_ha_to_area", True):
        area = min(area, float(row.get("uaa_ha", area)))
    return float(area * rate)

def ecoscheme_points(row: pd.Series, P: Dict) -> int:
    pts = 0
    uaa = float(row.get("uaa_ha", 0.0)) or 1.0
    grass_share = float(row.get("area_grass_ha", 0.0)) / uaa
    lu_density = float(row.get("livestock_lu", 0.0)) / max(1e-9, float(row.get("area_eligible_ha", uaa) or uaa))
    hedgerows_per_ha = float(row.get("hedgerow_km", 0.0)) / max(1e-9, uaa)
    n_crops = int(row.get("n_crops", 0))
    hhi = float(row.get("hhi", np.nan))
    pts_cfg = P["ecoscheme"]["points"]
    if int(row.get("organic", 0)) == 1:
        pts += pts_cfg.get("organic", 0)
    if grass_share >= 0.60:
        pts += pts_cfg.get("grass_share_ge_60", 0)
    if lu_density <= 1.0:
        pts += pts_cfg.get("low_stocking_density", 0)
    if n_crops >= 3 and (np.isnan(hhi) or hhi <= 0.5):
        pts += pts_cfg.get("diversification", 0)
    if hedgerows_per_ha >= 0.02:
        pts += pts_cfg.get("hedgerows_per_ha_ge_0_02", 0)
    return int(pts)

def ecoscheme_rate_from_points(points: int, P: Dict) -> float:
    table = {int(k): float(v) for k, v in P["ecoscheme"]["point_to_rate"].items()}
    if not table:
        return 0.0
    cutoff = max([k for k in table.keys() if points >= k] or [0])
    frac = table.get(cutoff, 0.0)
    return float(frac * float(P["ecoscheme"]["max_rate_eur_per_ha"]))

def calc_ecoscheme(row: pd.Series, P: Dict) -> float:
    area = float(row.get("area_eligible_ha", 0.0))
    pts = ecoscheme_points(row, P)
    rate = ecoscheme_rate_from_points(pts, P)
    return float(area * rate)

def calc_criss(row: pd.Series, P: Dict) -> float:
    area = float(row.get("area_eligible_ha", 0.0))
    first = float(P["criss"]["first_ha"])
    rate = float(P["criss"]["rate_eur_per_ha"])
    return float(min(area, first) * rate)

def calc_young_farmer(row: pd.Series, year: int, P: Dict) -> float:
    age = float(row.get("age", np.nan))
    start_year = float(row.get("start_year", np.nan))
    if np.isnan(age) or np.isnan(start_year):
        return 0.0
    eligible = (age <= float(P["young_farmer"]["eligible_age_max"])) and ((year - start_year) <= float(P["young_farmer"]["years_since_start_max"]))
    if not eligible:
        return 0.0
    area = float(row.get("area_eligible_ha", 0.0))
    first = float(P["young_farmer"]["first_ha"])
    rate = float(P["young_farmer"]["rate_eur_per_ha"])
    return float(min(area, first) * rate)

def calc_coupled(row: pd.Series, P: Dict) -> Tuple[float, float]:
    """Return (coupled_lu_eur, coupled_crop_eur)."""
    # Livestock
    lu_caps = P["coupled"]["caps"].get("lu_max", None)
    lu_parts = {
        "dairy": float(row.get("lu_dairy", 0.0)),
        "beef": float(row.get("lu_beef", 0.0)),
        "sheep": float(row.get("lu_sheep", 0.0)),
        "goat": float(row.get("lu_goat", 0.0)),
    }
    if lu_caps is not None:
        # proportionally scale down if over cap
        total_lu = sum(lu_parts.values())
        if total_lu > lu_caps and total_lu > 0:
            scale = float(lu_caps) / total_lu
            lu_parts = {k: v * scale for k, v in lu_parts.items()}
    lu_rates = P["coupled"]["livestock"]
    lu_eur = sum(lu_parts[k] * float(lu_rates.get(k, 0.0)) for k in lu_parts)

    # Crops
    protein_ha = float(row.get("area_protein_ha", 0.0))
    if P["coupled"]["caps"].get("protein_ha_max", None) is not None:
        protein_ha = min(protein_ha, float(P["coupled"]["caps"]["protein_ha_max"]))
    crop_rates = P["coupled"]["crops"]
    crop_eur = protein_ha * float(crop_rates.get("protein", 0.0))
    return float(lu_eur), float(crop_eur)

def calc_aecm(row: pd.Series, P: Dict) -> float:
    area = float(row.get("uaa_ha", 0.0))
    v = 0.0
    if int(row.get("organic", 0)) == 1:
        v += float(P["aecm"]["organic_bonus_eur_per_ha"])
    if int(row.get("hnv", 0)) == 1:
        v += float(P["aecm"]["hnv_bonus_eur_per_ha"])
    if int(row.get("natura2000", 0)) == 1:
        v += float(P["aecm"]["natura_bonus_eur_per_ha"])
    return float(v * area)

def calc_anc(row: pd.Series, P: Dict) -> float:
    if int(row.get("mountain_zone", 0)) != 1:
        return 0.0
    area = float(row.get("uaa_ha", 0.0))
    return float(area * float(P["anc"]["mountain_rate_eur_per_ha"]))

def apply_reductions(total_direct: float, row: pd.Series, P: Dict) -> Tuple[float, float]:
    """
    Return (reduction_amount, total_after_reduction)
    Applies labour deduction (proxy) then degressivity and cap.
    """
    base = float(total_direct)
    if P["reductions"].get("allow_labour_deduction", True):
        awu = float(row.get("hired_labour_awu", 0.0) or 0.0)
        base = max(0.0, base - awu * float(P["reductions"]["labour_value_per_awu"]))
    deg_start = float(P["reductions"]["degressivity_start"])
    deg_rate = float(P["reductions"]["degressivity_rate"])
    cap_thr = float(P["reductions"]["capping_threshold"])
    reduction = 0.0
    if base > deg_start:
        slice_amt = min(base, cap_thr) - deg_start
        reduction += slice_amt * deg_rate
    if base > cap_thr:
        reduction += (base - cap_thr)
    return float(reduction), float(total_direct - reduction)

def apply_penalties(amount: float, row: pd.Series, P: Dict) -> float:
    if not P["penalties"].get("use_conditionality_factor", True):
        return float(amount)
    fac = float(row.get("conditionality_factor", np.nan))
    if np.isnan(fac):
        fac = float(P["penalties"]["default_factor"])
    return float(amount * max(0.0, min(1.0, fac)))

# ----------------------------- Orchestration -----------------------------

@dataclass
class Config:
    farms: str
    payments: Optional[str]
    params: Optional[str]
    scenarios: Optional[str]
    prices: Optional[str]
    year: int
    outdir: str
    scenario_id: str

def compute_for_params(farms: pd.DataFrame, P: Dict, year: int) -> pd.DataFrame:
    rows = []
    for _, r in farms.iterrows():
        biss = calc_biss(r, P)
        eco = calc_ecoscheme(r, P)
        criss = calc_criss(r, P)
        yf = calc_young_farmer(r, year, P)
        coup_lu, coup_crops = calc_coupled(r, P)
        aecm = calc_aecm(r, P)
        anc = calc_anc(r, P)
        # Direct payments subject to reductions (simplified set)
        direct_subtotal = biss + eco + criss + yf + coup_lu + coup_crops
        red, direct_after = apply_reductions(direct_subtotal, r, P)
        direct_after = apply_penalties(direct_after, r, P)
        # Pillar 2 proxies (not reduced by capping typically)
        p2 = aecm + anc
        total = direct_after + p2
        rows.append({
            "farm_id": r.get("farm_id"),
            "region": r.get("region"),
            "departement": r.get("departement"),
            "uaa_ha": float(r.get("uaa_ha", 0.0)),
            "eligible_ha": float(r.get("area_eligible_ha", 0.0)),
            "BISS": round(biss, 2),
            "EcoScheme": round(eco, 2),
            "CRISS": round(criss, 2),
            "YF": round(yf, 2),
            "Coupled_LU": round(coup_lu, 2),
            "Coupled_Crops": round(coup_crops, 2),
            "Reductions": round(red, 2) * -1.0,
            "AECM": round(aecm, 2),
            "ANC": round(anc, 2),
            "Direct_After": round(direct_after, 2),
            "Pillar2": round(p2, 2),
            "Total": round(total, 2)
        })
    return pd.DataFrame(rows)

def distribution_tables(pay: pd.DataFrame) -> pd.DataFrame:
    df = pay.copy()
    df["eur_per_ha"] = df["Total"] / df["uaa_ha"].replace(0, np.nan)
    # deciles by eur/ha (fallback to Total if area missing)
    metric = df["eur_per_ha"].fillna(df["Total"])
    df["_decile"] = pd.qcut(metric.rank(method="first"), 10, labels=False) + 1 if metric.nunique() >= 10 else 1
    agg_rows = []
    for col in ["BISS","EcoScheme","CRISS","YF","Coupled_LU","Coupled_Crops","AECM","ANC","Direct_After","Pillar2","Total"]:
        for d in sorted(df["_decile"].unique()):
            sub = df[df["_decile"] == d]
            agg_rows.append({"scheme": col, "decile": int(d),
                             "sum_eur": float(sub[col].sum()),
                             "avg_eur_per_farm": float(sub[col].mean()),
                             "avg_eur_per_ha": float((sub[col] / sub["uaa_ha"].replace(0, np.nan)).mean())})
        # inequality stats
        g = gini(df[col].values)
        top10 = (df.nlargest(max(1,int(len(df)*0.1)), col)[col].sum() / max(1e-9, df[col].sum())) if df[col].sum() != 0 else np.nan
        agg_rows.append({"scheme": col, "decile": 0, "sum_eur": float(df[col].sum()),
                         "avg_eur_per_farm": float(df[col].mean()),
                         "avg_eur_per_ha": float((df[col] / df["uaa_ha"].replace(0, np.nan)).mean()),
                         "gini": float(g), "top10_share": float(top10)})
    out = pd.DataFrame(agg_rows)
    return out.sort_values(["scheme","decile"])

def region_tables(pay: pd.DataFrame, region_col: str) -> pd.DataFrame:
    cols = ["BISS","EcoScheme","CRISS","YF","Coupled_LU","Coupled_Crops","AECM","ANC","Direct_After","Pillar2","Total"]
    g = (pay.groupby(region_col, as_index=False)
           .agg({**{c:"sum" for c in cols}, **{"uaa_ha":"sum","eligible_ha":"sum","farm_id":"count"}})
           .rename(columns={"farm_id":"farms"}))
    for c in cols:
        g[c+"_per_ha"] = g[c] / g["uaa_ha"].replace(0, np.nan)
    g["Total_per_ha"] = g["Total"] / g["uaa_ha"].replace(0, np.nan)
    return g.sort_values("Total", ascending=False)

def crops_livestock_tables(farms: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    df = farms[["farm_id","area_protein_ha","lu_dairy","lu_beef","lu_sheep","lu_goat"]].copy()
    df = df.merge(pay[["farm_id","Coupled_LU","Coupled_Crops","Total"]], on="farm_id", how="left")
    return df

def apply_scenario_overrides(P: Dict, overrides: Dict[str, str|float|int]) -> Dict:
    import copy
    Q = copy.deepcopy(P)
    def set_by_dot(d, dotkey, value):
        parts = str(dotkey).split(".")
        cur = d
        for k in parts[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        try:
            v = float(value)
            if v.is_integer(): v = int(v)
        except Exception:
            v = value
        cur[parts[-1]] = v
    for k, v in overrides.items():
        set_by_dot(Q, k, v)
    return Q

def collect_scenario_overrides(scen_df: pd.DataFrame, scenario_id: str) -> Dict[str, str]:
    if scen_df.empty:
        return {}
    sub = scen_df[scen_df["scenario"] == scenario_id]
    return {str(k): v for k, v in zip(sub["key"], sub["value"])}

# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="French agri subsidies (CAP-style) emulator & analytics")
    ap.add_argument("--farms", required=True)
    ap.add_argument("--payments", default="")
    ap.add_argument("--params", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--outdir", default="out_fr_cap")
    return ap.parse_args()

# ----------------------------- Main -----------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    farms = load_farms(args.farms)
    params = load_params(args.params)
    scen_df = load_scenarios(args.scenarios)
    overrides = collect_scenario_overrides(scen_df, args.scenario) if args.scenario != "baseline" else {}
    P = apply_scenario_overrides(params, overrides) if overrides else params

    # Compute baseline (or selected scenario) payments
    pay = compute_for_params(farms, P, args.year)
    pay.to_csv(outdir / "payments_baseline.csv", index=False)

    # Distribution & regions
    dist = distribution_tables(pay)
    dist.to_csv(outdir / "distribution.csv", index=False)

    region_col = P["aggregates"].get("region_column", "region")
    regions = region_tables(pay, region_col if region_col in pay.columns or region_col in farms.columns else "region")
    regions.to_csv(outdir / "regions.csv", index=False)

    # Crop/Livestock view
    crops_lu = crops_livestock_tables(farms, pay)
    crops_lu.to_csv(outdir / "crops_livestock.csv", index=False)

    # Scenario comparison (if scenarios.csv has >1 id, compare vs baseline too)
    scen_out = []
    if not scen_df.empty:
        scen_ids = sorted(scen_df["scenario"].unique())
        # Ensure baseline computed (P already applied)
        base = pay.copy()
        base["scenario"] = "baseline"
        scen_out.append(base)
        for sid in scen_ids:
            if sid == "baseline":
                continue
            O = collect_scenario_overrides(scen_df, sid)
            Q = apply_scenario_overrides(params, O)
            pay_s = compute_for_params(farms, Q, args.year)
            pay_s["scenario"] = sid
            scen_out.append(pay_s)
        comp = pd.concat(scen_out, ignore_index=True)
        comp.to_csv(outdir / "scenario_comparison.csv", index=False)

    # KPIs / summary
    kpi = {
        "year": int(args.year),
        "farms": int(len(farms)),
        "avg_total_per_farm": float(pay["Total"].mean()),
        "median_total_per_farm": float(pay["Total"].median()),
        "avg_total_per_ha": float((pay["Total"] / pay["uaa_ha"].replace(0, np.nan)).mean()),
        "gini_total": float(gini(pay["Total"].values)),
        "top10_share_total": float((pay.nlargest(max(1,int(len(pay)*0.1)), "Total")["Total"].sum() / max(1e-9, pay["Total"].sum()))) if pay["Total"].sum() != 0 else np.nan,
        "by_scheme_sum": {c: float(pay[c].sum()) for c in ["BISS","EcoScheme","CRISS","YF","Coupled_LU","Coupled_Crops","AECM","ANC","Direct_After","Pillar2","Total"]}
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))

    # Config dump
    cfg = asdict(Config(
        farms=args.farms, payments=args.payments or None, params=args.params or None, scenarios=args.scenarios or None,
        prices=args.prices or None, year=args.year, outdir=args.outdir, scenario_id=args.scenario))
    (outdir / "config.json").write_text(json.dumps({"config": cfg, "params": P}, indent=2))

    # Console
    print("== French Agri Subsidies (emulator) ==")
    print(f"Year: {args.year}  Farms: {kpi['farms']}  Avg €/farm: {kpi['avg_total_per_farm']:.0f}  Gini: {kpi['gini_total']:.2f}")
    print("Top scheme sums (€m):", {k: round(v/1e6,2) for k,v in kpi["by_scheme_sum"].items() if k != "Total"})
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
