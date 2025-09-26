#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
climate_transition.py — Portfolio transition-risk engine (carbon price, demand shifts, tech adoption)

What this does
--------------
Given portfolio holdings and issuer/sector fundamentals (emissions, financials, abatement curves,
pass-through, elasticities), this script builds *deterministic* and *stochastic* (MC) transition
scenarios and estimates:

1) Carbon cost mechanics with endogenous abatement (MACC) and ramp constraints
2) Pass-through to prices, demand loss from elasticities, subsidies/mandates impacts
3) EBITDA & margin deltas, plus DCF-style value deltas (ΔV ≈ ΔEBITDA/(WACC - g))
4) Credit spread deltas via a simple margin-to-spread mapping
5) Portfolio aggregation & attribution by sector/country/issuer and factor (carbon, abatement capex/opex,
   pass-through, demand loss, subsidies)
6) Sensitivity (“Greeks”) to carbon price, pass-through, elasticity
7) Monte Carlo (optional) for uncertainty around carbon price, pass-through, elasticity

Core outputs (CSV/JSON)
-----------------------
- issuer_impacts.csv         Per issuer, per horizon year: baseline & shocked KPIs, ΔEBITDA, ΔValue, ΔSpread
- factor_attribution.csv     Per issuer-year breakdown: carbon_cost, abatement_spend, residual_cost_absorbed,
                             price_uplift_revenue, demand_loss_revenue, subsidies, tech_shift
- portfolio_aggregates.csv   Aggregation by {total, sector, country} across horizons
- sensitivities.csv          d(ΔValue)/d(carbon), d(ΔValue)/d(pass_through), d(ΔValue)/d(elasticity)
- mc_distribution.csv        If MC enabled: portfolio ΔValue distribution per horizon (all paths)
- scenario_path.csv          Canonical scenario inputs resolved by year (carbon price, demand multipliers, etc.)
- summary.json               Headline metrics (ΔValue, ΔEBITDA, ΔSpread) total & by top-5 contributors
- config.json                Run configuration for reproducibility

Inputs (CSV; headers are case-insensitive & flexible)
-----------------------------------------------------
--portfolio portfolio.csv   REQUIRED
  Columns: issuer_id, name, sector, country, position_value_usd (or weight), [ticker]
           (If weights missing, we compute from position_value_usd; cash excluded unless given.)

--emissions emissions.csv   OPTIONAL (issuer annual)
  Columns: year, issuer_id, scope1_t, scope2_t, scope3_t, revenue_usd
           (We build intensity = (S1+S2 + alpha*S3) / revenue; alpha via --s3_alpha)

--financials financials.csv OPTIONAL (issuer annual)
  Columns: issuer_id, revenue_usd, ebitda_usd, ebitda_margin_pct, wacc_pct, growth_pct
           (If missing, we infer ebitda_margin from industry medians = 15% and WACC=8%, g=2%.)

--rev_breakdown rev.csv     OPTIONAL (issuer-segment)
  Columns: issuer_id, segment, revenue_share_pct [, intensity_override_t_per_usd]
           (Used to apply sector/segment-specific elasticities or policy multipliers.)

--macc macc.csv             OPTIONAL (issuer/sector abatement options)
  Columns: level: issuer_id or sector,
           option, cost_usd_per_t, max_abatable_pct, ramp_pct_per_year
           (We treat options as piecewise steps; abatement possible if carbon_price ≥ cost.)

--pass_through pass.csv     OPTIONAL (sector-level unless issuer provided)
  Columns: level: issuer_id or sector, pass_through_pct   (0..1)

--elasticity elas.csv       OPTIONAL (sector/segment)
  Columns: level: sector or segment, price_elasticity     (negative)

--subsidies subs.csv        OPTIONAL (tech/sector)
  Columns: year, level (issuer_id/sector/segment/tech), subsidy_usd_per_unit or pct_of_revenue
           (Applied as EBITDA uplift in shocked state.)

--policy policies.csv       OPTIONAL (scenario key-value table)
  Columns: scenario, key, value
  Keys examples (by year suffix):
     carbon_price.2025 = 100
     demand_multiplier.STEEL.2030 = 0.95        (exogenous demand shock)
     mandate_share.EV.2030 = 0.60               (share target; affects AUTO ICE vs EV mix if segments exist)
     pass_through.override.CEMENT.2030 = 0.70   (override pass-through)
     elasticity.override.CHEMICALS.2030 = -0.8
     elec_emission_factor.2030 = 0.20 (t/MWh)  (reduces Scope2 intensity proxy)
  Keys without year apply to all horizons unless overridden.

--horizons "2025,2030,2035,2040" (or any comma list of years)

CLI options
-----------
--s3_alpha 0.5               Weight on Scope 3 in intensity (0..1)
--spread_beta 20.0           ΔSpread (bps) per 1pp EBITDA margin decline
--absorption_cap 0.35        Max share of revenue that can be absorbed in-year (rest assumed to force demand/price)
--mc_paths 0                 Monte Carlo paths (0 disables)
--mc_carbon_sd 0.15          Rel. SD on carbon price in MC (e.g., 0.15 = 15%)
--mc_pass_sd 0.10            Abs SD on pass-through (bounded 0..1)
--mc_elas_sd 0.15            Rel SD on elasticity (multiplier on absolute value)
--outdir out_transition
--scenario NZ2050            Scenario name to pick from policies.csv (if provided)

Method (high level)
-------------------
- Carbon cost (pre-abatement) = carbon_price × emissions_effective
- Abatement: for each option with cost ≤ price and ramp capacity, reduce emissions; abatement cost = cost × abated_t
- Residual emissions priced → cost. A share is passed through to price; remainder hits EBITDA.
- Demand loss = elasticity × %price_increase (capped so revenue >= 0); revenue ↓ and EBITDA ↓ (via margin)
- Exogenous demand multipliers & mandates alter segment mix (optional) → change intensities/demand.
- Valuation ΔV ≈ ΔEBITDA / (WACC - g) using baseline WACC & growth. Credit ΔSpread = spread_beta × Δmargin_pp.

DISCLAIMER: Research tooling; many simplifications. Not investment advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- utilities -----------------------------

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def parse_years(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if str(x).strip()]

def wide_lookup(df: pd.DataFrame, level_col: str, key_col: str, val_col: str) -> Dict[Tuple[str, str], float]:
    out = {}
    for _, r in df.iterrows():
        out[(str(r[level_col]).upper(), str(r[key_col]).upper())] = float(r[val_col]) if pd.notna(r[val_col]) else np.nan
    return out


# ----------------------------- loaders -----------------------------

def load_portfolio(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {(ncol(df,"issuer_id") or "issuer_id"): "issuer_id",
           (ncol(df,"name") or "name"): "name",
           (ncol(df,"sector") or "sector"): "sector",
           (ncol(df,"country") or "country"): "country",
           (ncol(df,"position_value_usd") or ncol(df,"market_value_usd") or ncol(df,"value_usd") or "position_value_usd"): "mv_usd",
           (ncol(df,"weight") or "weight"): "weight"}
    df = df.rename(columns=ren)
    for c in ["issuer_id","name","sector","country"]:
        if c in df.columns: df[c] = df[c].astype(str)
    if "mv_usd" in df.columns:
        df["mv_usd"] = num(df["mv_usd"])
    if "weight" not in df.columns or df["weight"].isna().all():
        total = df["mv_usd"].sum()
        df["weight"] = df["mv_usd"] / total if total>0 else np.nan
    return df[["issuer_id","name","sector","country","mv_usd","weight"]].copy()

def load_emissions(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"year") or "year"): "year",
           (ncol(df,"issuer_id") or "issuer_id"): "issuer_id",
           (ncol(df,"scope1_t") or "scope1_t"): "s1",
           (ncol(df,"scope2_t") or "scope2_t"): "s2",
           (ncol(df,"scope3_t") or "scope3_t"): "s3",
           (ncol(df,"revenue_usd") or "revenue_usd"): "revenue"}
    df = df.rename(columns=ren)
    df["year"] = num(df["year"]).astype("Int64")
    df["issuer_id"] = df["issuer_id"].astype(str)
    for c in ["s1","s2","s3","revenue"]:
        if c in df.columns: df[c] = num(df[c])
    return df.sort_values(["issuer_id","year"])

def load_financials(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"issuer_id") or "issuer_id"): "issuer_id",
           (ncol(df,"revenue_usd") or "revenue_usd"): "revenue",
           (ncol(df,"ebitda_usd") or "ebitda_usd"): "ebitda",
           (ncol(df,"ebitda_margin_pct") or "ebitda_margin_pct"): "margin",
           (ncol(df,"wacc_pct") or "wacc_pct"): "wacc",
           (ncol(df,"growth_pct") or "growth_pct"): "g"}
    df = df.rename(columns=ren)
    df["issuer_id"] = df["issuer_id"].astype(str)
    for c in ["revenue","ebitda","margin","wacc","g"]:
        if c in df.columns: df[c] = num(df[c])
    return df

def load_rev_breakdown(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"issuer_id") or "issuer_id"): "issuer_id",
           (ncol(df,"segment") or "segment"): "segment",
           (ncol(df,"revenue_share_pct") or "revenue_share_pct"): "rev_share",
           (ncol(df,"intensity_override_t_per_usd") or "intensity_override_t_per_usd"): "int_override"}
    df = df.rename(columns=ren)
    df["issuer_id"] = df["issuer_id"].astype(str)
    df["segment"] = df["segment"].astype(str)
    for c in ["rev_share","int_override"]:
        if c in df.columns: df[c] = num(df[c])
    df["rev_share"] = clip01(df["rev_share"] / (100.0 if df["rev_share"].max()>1.0 else 1.0))
    return df

def load_macc(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    # level = issuer_id or sector
    ren = {(ncol(df,"issuer_id") or ncol(df,"sector") or "level"): "level",
           (ncol(df,"option") or "option"): "option",
           (ncol(df,"cost_usd_per_t") or "cost_usd_per_t"): "cost",
           (ncol(df,"max_abatable_pct") or "max_abatable_pct"): "max_pct",
           (ncol(df,"ramp_pct_per_year") or "ramp_pct_per_year"): "ramp_pct"}
    df = df.rename(columns=ren)
    df["level"] = df["level"].astype(str)
    for c in ["cost","max_pct","ramp_pct"]:
        if c in df.columns: df[c] = num(df[c])
    df["max_pct"] = clip01(df["max_pct"] / (100.0 if df["max_pct"].max()>1.0 else 1.0))
    df["ramp_pct"] = df["ramp_pct"].fillna(100.0)
    df["ramp_pct"] = df["ramp_pct"] / (100.0 if df["ramp_pct"].max()>1.0 else 1.0)
    return df.sort_values(["level","cost"])

def load_pass_through(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    lv = ncol(df,"issuer_id") or ncol(df,"sector") or "level"
    df = df.rename(columns={lv:"level", (ncol(df,"pass_through_pct") or "pass_through_pct"):"pt"})
    df["level"] = df["level"].astype(str)
    df["pt"] = clip01(num(df["pt"]) / (100.0 if df["pt"].max()>1.0 else 1.0))
    return df

def load_elasticity(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    lv = ncol(df,"sector") or ncol(df,"segment") or "level"
    df = df.rename(columns={lv:"level", (ncol(df,"price_elasticity") or "price_elasticity"):"elas"})
    df["level"] = df["level"].astype(str)
    df["elas"] = num(df["elas"]).fillna(-0.5)  # default modest elasticity
    return df

def load_subsidies(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"year") or "year"):"year",
           (ncol(df,"level") or ncol(df,"sector") or ncol(df,"issuer_id") or "level"):"level",
           (ncol(df,"subsidy_usd_per_unit") or "subsidy_usd_per_unit"):"subsidy_per_unit",
           (ncol(df,"pct_of_revenue") or "pct_of_revenue"):"subsidy_pct_rev"}
    df = df.rename(columns=ren)
    df["year"] = num(df["year"]).astype("Int64")
    df["level"] = df["level"].astype(str)
    for c in ["subsidy_per_unit","subsidy_pct_rev"]:
        if c in df.columns: df[c] = num(df[c])
    if "subsidy_pct_rev" in df.columns:
        df["subsidy_pct_rev"] = df["subsidy_pct_rev"] / (100.0 if df["subsidy_pct_rev"].max()>1.0 else 1.0)
    return df

def load_policies(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"):"scenario",
           (ncol(df,"key") or "key"):"key",
           (ncol(df,"value") or "value"):"value"}
    df = df.rename(columns=ren)
    return df


# ----------------------------- scenario resolution -----------------------------

def resolve_policy_table(policies: pd.DataFrame, scenario: Optional[str], horizons: List[int]) -> pd.DataFrame:
    """Create a tidy table of scenario inputs by year for the chosen scenario (or defaults)."""
    if policies.empty:
        # default: simple linear carbon path rising 15%/yr from $60 in first horizon year
        if len(horizons)==0: return pd.DataFrame()
        base = horizons[0]
        path = []
        cp = 60.0
        for y in horizons:
            if y>base: cp = cp * (1.15 ** ((y-base)/5.0))  # rough
            path.append({"year": y, "key": "carbon_price", "value": cp})
        return pd.DataFrame(path)
    S = policies.copy()
    if scenario:
        S = S[S["scenario"].astype(str)==scenario]
        if S.empty:
            S = policies.copy()
    rows = []
    # Keys may have .YYYY suffix; otherwise apply to all horizons
    for _, r in S.iterrows():
        key = str(r["key"]).strip()
        v = float(r["value"]) if pd.notna(r["value"]) else np.nan
        # Detect year suffix
        year_spec = None
        if "." in key:
            try:
                year_spec = int(key.split(".")[-1])
                base_key = ".".join(key.split(".")[:-1])
            except Exception:
                base_key = key
        else:
            base_key = key
        if year_spec:
            if year_spec in horizons:
                rows.append({"year": year_spec, "key": base_key, "value": v})
        else:
            for y in horizons:
                rows.append({"year": y, "key": base_key, "value": v})
    return pd.DataFrame(rows)

def scenario_value(table: pd.DataFrame, year: int, key: str, default: float, level: Optional[str]=None) -> float:
    """Fetch value by year & key; allow overrides with .override.LEVEL where applicable."""
    # First look for override for a specific level (e.g., pass_through.override.CEMENT)
    if level is not None:
        mask = (table["year"]==year) & (table["key"].str.lower()==f"{key}.override.{str(level).lower()}")
        if not table[mask].empty:
            try: return float(table[mask]["value"].iloc[0])
            except Exception: pass
    mask2 = (table["year"]==year) & (table["key"].str.lower()==key.lower())
    if not table[mask2].empty:
        try: return float(table[mask2]["value"].iloc[0])
        except Exception:
            return default
    return default


# ----------------------------- core mechanics -----------------------------

def baseline_intensity(emi_row: pd.Series, s3_alpha: float) -> float:
    """tons CO2 per USD revenue (t/$). If revenue 0, return 0."""
    rev = float(emi_row.get("revenue", np.nan))
    if not pd.notna(rev) or rev<=0: return 0.0
    s1 = float(emi_row.get("s1", 0.0)); s2 = float(emi_row.get("s2", 0.0)); s3 = float(emi_row.get("s3", 0.0))
    eff_t = s1 + s2 + s3_alpha * s3
    return eff_t / rev

def macc_abatement_fraction(macc_df: pd.DataFrame, level_key: str, carbon_price: float, years_since_start: int) -> Tuple[float, float]:
    """
    Return (abatement_pct, avg_abatement_cost) for given level at this carbon price,
    subject to ramp limits. If level not present, return (0, 0).
    avg_abatement_cost is the *weighted average cost per t* of the abated portion.
    """
    if macc_df.empty: return 0.0, 0.0
    M = macc_df[macc_df["level"].str.upper()==str(level_key).upper()]
    if M.empty: return 0.0, 0.0
    cap_ramp = float(min(1.0, M["ramp_pct"].mean() * max(1, years_since_start)))  # crude: accumulate ramp
    abate = 0.0; cost_weighted = 0.0
    remaining_cap = cap_ramp
    for _, r in M.sort_values("cost").iterrows():
        if r["cost"] <= carbon_price and remaining_cap > 1e-9:
            take = min(float(r["max_pct"]), remaining_cap)
            abate += take
            cost_weighted += take * float(r["cost"])
            remaining_cap -= take
    if abate <= 1e-9: return 0.0, 0.0
    avg_cost = cost_weighted / abate
    return float(min(1.0, abate)), float(avg_cost)

def effective_pass_through(issuer_id: str, sector: str, pass_df: pd.DataFrame, scen_tbl: pd.DataFrame, year: int, default_pt: float=0.5) -> float:
    if not pass_df.empty:
        # issuer precedence
        m1 = pass_df[pass_df["level"].str.upper()==issuer_id.upper()]
        if not m1.empty: default_pt = float(m1["pt"].iloc[0])
        else:
            m2 = pass_df[pass_df["level"].str.upper()==sector.upper()]
            if not m2.empty: default_pt = float(m2["pt"].iloc[0])
    # scenario override
    return float(clip01(pd.Series([scenario_value(scen_tbl, year, "pass_through", default_pt, level=sector)])).iloc[0])

def effective_elasticity(sector: str, segment: Optional[str], elas_df: pd.DataFrame, scen_tbl: pd.DataFrame, year: int, default_elas: float=-0.5) -> float:
    if not elas_df.empty:
        if segment:
            m = elas_df[elas_df["level"].str.upper()==segment.upper()]
            if not m.empty: default_elas = float(m["elas"].iloc[0])
        if default_elas == -0.5:
            m2 = elas_df[elas_df["level"].str.upper()==sector.upper()]
            if not m2.empty: default_elas = float(m2["elas"].iloc[0])
    # override
    o = scenario_value(scen_tbl, year, "elasticity", default_elas, level=segment or sector)
    return float(o)

def cap_absorption_limit(revenue: float, absorption_cap: float) -> float:
    return float(absorption_cap * revenue)

def value_delta_dcf(delta_ebitda: float, wacc: float, g: float) -> float:
    denom = max(1e-6, (wacc - g))
    return float(delta_ebitda / denom)

def spread_delta_bps(delta_margin_pp: float, beta: float) -> float:
    return float(beta * delta_margin_pp)

def sanitize_financials(fin: pd.DataFrame) -> pd.DataFrame:
    f = fin.copy()
    if f.empty: return f
    f["margin"] = f["margin"].fillna(f["ebitda"] / f["revenue"] * 100.0).replace([np.inf,-np.inf], np.nan)
    f["margin"] = f["margin"].fillna(15.0)  # %
    f["wacc"] = f["wacc"].fillna(8.0)       # %
    f["g"] = f["g"].fillna(2.0)             # %
    return f


# ----------------------------- engine -----------------------------

def run_engine(
    portfolio: pd.DataFrame,
    emissions: pd.DataFrame,
    financials: pd.DataFrame,
    rev_breakdown: pd.DataFrame,
    macc_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    elas_df: pd.DataFrame,
    subs_df: pd.DataFrame,
    scen_tbl: pd.DataFrame,
    horizons: List[int],
    s3_alpha: float,
    absorption_cap: float,
    spread_beta: float,
    mc_paths: int,
    mc_carbon_sd: float,
    mc_pass_sd: float,
    mc_elas_sd: float,
) -> Dict[str, pd.DataFrame]:

    # latest emissions row per issuer (use last reported year)
    if emissions.empty:
        # fabricate zeros with a tiny intensity for non-zero math
        E = portfolio[["issuer_id"]].copy()
        E["year"] = max(horizons) if horizons else 2030
        E["s1"] = 0.0; E["s2"] = 0.0; E["s3"] = 0.0; E["revenue"] = 1.0
    else:
        E = emissions.sort_values(["issuer_id","year"]).groupby("issuer_id").tail(1)

    FIN = sanitize_financials(financials)
    # merge a simple baseline financial snapshot
    base_fin = portfolio.merge(FIN, on="issuer_id", how="left")
    for c, v in {"revenue": 1e6, "ebitda": np.nan, "margin": 15.0, "wacc": 8.0, "g": 2.0}.items():
        if c not in base_fin.columns or base_fin[c].isna().all():
            base_fin[c] = v
    base_fin["revenue"] = base_fin["revenue"].fillna(1e6)  # USD
    base_fin["margin"] = base_fin["margin"].fillna(15.0)    # %
    base_fin["wacc"] = base_fin["wacc"].fillna(8.0) / 100.0
    base_fin["g"] = base_fin["g"].fillna(2.0) / 100.0

    # pre-compute intensities
    Eint = E.copy()
    Eint["intensity_t_per_usd"] = Eint.apply(lambda r: baseline_intensity(r, s3_alpha), axis=1)

    # resolve scenario canonical path table (key, year, value)
    scen_tbl = scen_tbl.copy()

    # Containers
    issuer_rows = []
    attr_rows = []
    sens_rows = []

    # Helper: years since "start" for MACC ramp (assume start = min(horizons))
    t0 = min(horizons) if horizons else 2025

    # Main deterministic loop
    for _, p in portfolio.iterrows():
        iid = str(p["issuer_id"]); sec = str(p.get("sector","UNKNOWN"))
        fin = base_fin[base_fin["issuer_id"]==iid].iloc[0]
        emi = Eint[Eint["issuer_id"]==iid]
        # If no emissions row, assume sector-average intensity 0.3 t/$m → 3e-7 t/$; choose conservative tiny
        base_int = float(emi["intensity_t_per_usd"].iloc[0]) if not emi.empty else 3e-7
        base_rev = float(fin["revenue"])
        base_mgn = float(fin["margin"]) / 100.0
        base_ebitda = base_rev * base_mgn
        wacc = float(fin["wacc"]); g = float(fin["g"])

        for y in horizons:
            carbon = scenario_value(scen_tbl, y, "carbon_price", default=60.0)
            years_since = max(0, y - t0)
            # Abatement potential & cost
            abate_pct, avg_abate_cost = macc_abatement_fraction(macc_df, iid, carbon, years_since)
            if abate_pct == 0.0:
                # try sector MACC if issuer-level not present
                abate_pct, avg_abate_cost = macc_abatement_fraction(macc_df, sec, carbon, years_since)

            # Emissions & abatement (apply to S1+S2+(αS3))
            eff_int = base_int * (1.0 - abate_pct)
            emissions_priced_t = eff_int * base_rev  # tons
            pre_abate_t = base_int * base_rev
            abated_t = pre_abate_t - emissions_priced_t
            abatement_spend = abated_t * max(0.0, avg_abate_cost)

            # Carbon cost on residual emissions
            carbon_cost = emissions_priced_t * max(0.0, carbon)

            # Pass-through & demand
            pt = effective_pass_through(iid, sec, pass_df, scen_tbl, y, default_pt=0.5)
            price_uplift_revenue = pt * carbon_cost  # recovered via higher prices
            absorbed_cost = carbon_cost - price_uplift_revenue
            # Cap absorbed share at absorption_cap × revenue
            absorbed_cap = cap_absorption_limit(base_rev, absorption_cap)
            absorbed_cost_eff = float(min(absorbed_cost, absorbed_cap))

            # Demand loss from price increase (% of revenue)
            price_increase_pct = price_uplift_revenue / max(1e-6, base_rev)
            elas = effective_elasticity(sec, None, elas_df, scen_tbl, y, default_elas=-0.5)
            demand_loss_pct = max(-0.9, float(elas) * float(price_increase_pct))
            exog_mult = scenario_value(scen_tbl, y, f"demand_multiplier.{sec}", default=1.0)
            # Net demand shock on revenue (keep ≥ 0)
            revenue_after = base_rev * max(0.0, (1.0 + demand_loss_pct)) * float(exog_mult)
            demand_loss_revenue = base_rev - revenue_after

            # Subsidies (as % of revenue or absolute — we apply % if present)
            subsidy_pct = scenario_value(scen_tbl, y, f"subsidy_pct_rev.{sec}", default=0.0)
            subsidy_usd = subsidy_pct * revenue_after

            # EBITDA delta components
            # Baseline EBITDA on base_rev; shocked EBITDA on revenue_after and after costs & spend
            shocked_ebitda = revenue_after * base_mgn \
                             - absorbed_cost_eff \
                             - abatement_spend \
                             + subsidy_usd
            delta_ebitda = shocked_ebitda - base_ebitda
            delta_margin_pp = (shocked_ebitda / max(1e-6, revenue_after) - base_mgn) * 100.0

            # Valuation & credit
            dV = value_delta_dcf(delta_ebitda, wacc, g)
            dSpr = spread_delta_bps(delta_margin_pp, spread_beta)

            issuer_rows.append({
                "issuer_id": iid, "name": p["name"], "sector": sec, "country": p["country"],
                "year": y,
                "weight": float(p["weight"]), "mv_usd": float(p["mv_usd"]),
                "baseline_revenue_usd": base_rev, "baseline_ebitda_usd": base_ebitda, "baseline_margin_pct": base_mgn*100.0,
                "carbon_price_usd_t": carbon,
                "abatement_pct": abate_pct, "avg_abate_cost": avg_abate_cost,
                "emissions_intensity_t_per_usd": base_int, "effective_intensity_t_per_usd": eff_int,
                "carbon_cost_usd": carbon_cost, "abatement_spend_usd": abatement_spend,
                "price_uplift_revenue_usd": price_uplift_revenue,
                "absorbed_cost_usd": absorbed_cost_eff,
                "price_increase_pct": price_increase_pct,
                "demand_loss_revenue_usd": demand_loss_revenue,
                "subsidies_usd": subsidy_usd,
                "shocked_revenue_usd": revenue_after,
                "shocked_ebitda_usd": shocked_ebitda,
                "delta_ebitda_usd": delta_ebitda,
                "delta_value_usd": dV,
                "delta_spread_bps": dSpr,
            })

            attr_rows.append({
                "issuer_id": iid, "year": y,
                "carbon_cost_usd": carbon_cost,
                "abatement_spend_usd": abatement_spend,
                "residual_cost_absorbed_usd": absorbed_cost_eff,
                "price_uplift_revenue_usd": price_uplift_revenue,
                "demand_loss_revenue_usd": demand_loss_revenue,
                "subsidies_usd": subsidy_usd,
                "tech_shift_usd": 0.0  # placeholder; hook for segment/mandate modeling if rev_breakdown provided
            })

        # Sensitivities (finite differences at first horizon)
        if horizons:
            y0 = horizons[0]
            base = [r for r in issuer_rows if r["issuer_id"]==iid and r["year"]==y0][0]
            eps_c = 10.0  # $/t
            eps_p = 0.05  # +5pp
            eps_e = 0.10  # +10% elasticity magnitude
            # carbon +eps
            c_up = base.copy(); c_up["carbon_price_usd_t"] += eps_c
            # recompute delta_value for carbon up (holding other elements linearized)
            # We approximate by linear term: ΔV ≈ (emissions_priced_t * Δcarbon - pt* that share - cap) / (wacc - g)
            # For simplicity: use emissions_priced at y0:
            eff_int0 = base["effective_intensity_t_per_usd"]; rev0 = base["baseline_revenue_usd"]
            em_t0 = eff_int0 * rev0
            pt0 = effective_pass_through(iid, sec, pass_df, scen_tbl, y0, default_pt=0.5)
            absorbed_cap0 = cap_absorption_limit(rev0, absorption_cap)
            incr_cost = em_t0 * eps_c
            absorbed_incr = min(incr_cost * (1-pt0), absorbed_cap0 - base["absorbed_cost_usd"])
            dEbitda_c = - absorbed_incr  # pass-through portion assumed neutral to EBITDA (price up offsets cost)
            dV_c = value_delta_dcf(dEbitda_c, float(fin["wacc"]), float(fin["g"]))
            # pass-through +5pp
            dpt = eps_p
            # less absorption, more price uplift; EBITDA improves by (dpt * carbon_cost)
            dEbitda_p = dpt * base["carbon_cost_usd"]
            dV_p = value_delta_dcf(dEbitda_p, float(fin["wacc"]), float(fin["g"]))
            # elasticity +10% magnitude (more negative)
            elas0 = effective_elasticity(sec, None, elas_df, scen_tbl, y0, default_elas=-0.5)
            elas1 = elas0 * (1.0 + np.sign(elas0)*eps_e)
            dm0 = elas0 * base["price_increase_pct"]; dm1 = elas1 * base["price_increase_pct"]
            dRev = - (dm1 - dm0) * rev0
            dEbitda_e = - dRev * base_mgn
            dV_e = value_delta_dcf(dEbitda_e, float(fin["wacc"]), float(fin["g"]))
            sens_rows.append({"issuer_id": iid, "year": y0, "dV_dCarbon_usd_per_$10": dV_c, "dV_dPassThrough_usd_per_5pp": dV_p, "dV_dElasticity_usd_per_10pct": dV_e})

    issuer_df = pd.DataFrame(issuer_rows)
    attr_df = pd.DataFrame(attr_rows)
    sens_df = pd.DataFrame(sens_rows)

    # Aggregation
    def agg(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
        keep = ["delta_value_usd","delta_ebitda_usd","delta_spread_bps","mv_usd","weight"]
        numeric = [c for c in keep if c in df.columns]
        out = df.groupby(by, as_index=False)[numeric].sum()
        # portfolio Δ% of MV (rough): ΔV / MV
        out["delta_value_pct_mv"] = out["delta_value_usd"] / out["mv_usd"].replace(0, np.nan)
        return out

    port_total = agg(issuer_df, ["year"])
    by_sector = agg(issuer_df, ["year","sector"])
    by_country = agg(issuer_df, ["year","country"])
    portfolio_agg = pd.concat([
        port_total.assign(bucket="TOTAL"),
        by_sector.assign(bucket="SECTOR").rename(columns={"sector":"bucket_value"}),
        by_country.assign(bucket="COUNTRY").rename(columns={"country":"bucket_value"}),
    ], ignore_index=True)

    # Monte Carlo (portfolio ΔV)
    mc_out = pd.DataFrame()
    if mc_paths and mc_paths>0:
        rng = np.random.default_rng(42)
        rows = []
        for y in horizons:
            # draw path multipliers
            c_price = scenario_value(scen_tbl, y, "carbon_price", default=60.0)
            for k in range(mc_paths):
                cp = max(0.0, rng.normal(c_price, mc_carbon_sd * c_price))
                pt_shift = clip01(pd.Series([rng.normal(0.0, mc_pass_sd)])).iloc[0]
                elas_mult = max(0.0, rng.normal(1.0, mc_elas_sd))
                # recompute ΔV for each issuer approximately (linearized around deterministic result)
                sub = issuer_df[issuer_df["year"]==y].copy()
                # carbon cost scale factor
                scale_carbon = cp / sub["carbon_price_usd_t"].replace(0, np.nan)
                scale_carbon = scale_carbon.fillna(1.0).clip(0.0, 3.0)
                # pass-through increase reduces absorption: +pt_shift
                dEbitda_pt = pt_shift * sub["carbon_cost_usd"]
                # elasticity scale
                dRev_elas = (elas_mult - 1.0) * (- sub["price_increase_pct"] * sub["baseline_revenue_usd"])  # ∂Rev
                dEbitda_elas = dRev_elas * ( - (base_fin.set_index("issuer_id").loc[sub["issuer_id"], "margin"].values / 100.0) )
                # carbon scale on absorbed portion only
                dEbitda_c = (scale_carbon - 1.0) * sub["absorbed_cost_usd"] * (-1.0)
                # value delta approx
                waccv = base_fin.set_index("issuer_id").loc[sub["issuer_id"], "wacc"].values
                gv = base_fin.set_index("issuer_id").loc[sub["issuer_id"], "g"].values
                dV_mc = (dEbitda_pt + dEbitda_elas + dEbitda_c) / (waccv - gv)
                pv = sub["delta_value_usd"].values + dV_mc
                rows.append({"path": k, "year": y, "portfolio_delta_value_usd": float(np.nansum(pv))})
        mc_out = pd.DataFrame(rows)

    return {
        "issuer_impacts": issuer_df,
        "factor_attribution": attr_df,
        "portfolio_aggregates": portfolio_agg,
        "sensitivities": sens_df,
        "mc_distribution": mc_out,
    }


# ----------------------------- CLI -----------------------------

@dataclass
class Config:
    portfolio: str
    emissions: Optional[str]
    financials: Optional[str]
    rev_breakdown: Optional[str]
    macc: Optional[str]
    pass_through: Optional[str]
    elasticity: Optional[str]
    subsidies: Optional[str]
    policies: Optional[str]
    horizons: str
    scenario: Optional[str]
    s3_alpha: float
    spread_beta: float
    absorption_cap: float
    mc_paths: int
    mc_carbon_sd: float
    mc_pass_sd: float
    mc_elas_sd: float
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Climate transition risk engine — carbon, demand, valuation & credit")
    ap.add_argument("--portfolio", required=True)
    ap.add_argument("--emissions", default="")
    ap.add_argument("--financials", default="")
    ap.add_argument("--rev_breakdown", default="")
    ap.add_argument("--macc", default="")
    ap.add_argument("--pass_through", default="")
    ap.add_argument("--elasticity", default="")
    ap.add_argument("--subsidies", default="")
    ap.add_argument("--policies", default="")
    ap.add_argument("--horizons", default="2025,2030,2035,2040")
    ap.add_argument("--scenario", default="")
    ap.add_argument("--s3_alpha", type=float, default=0.5)
    ap.add_argument("--spread_beta", type=float, default=20.0)
    ap.add_argument("--absorption_cap", type=float, default=0.35)
    ap.add_argument("--mc_paths", type=int, default=0)
    ap.add_argument("--mc_carbon_sd", type=float, default=0.15)
    ap.add_argument("--mc_pass_sd", type=float, default=0.10)
    ap.add_argument("--mc_elas_sd", type=float, default=0.15)
    ap.add_argument("--outdir", default="out_transition")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    horizons = parse_years(args.horizons)

    # Load data
    PORT = load_portfolio(args.portfolio)
    EMI  = load_emissions(args.emissions) if args.emissions else pd.DataFrame()
    FIN  = load_financials(args.financials) if args.financials else pd.DataFrame()
    REV  = load_rev_breakdown(args.rev_breakdown) if args.rev_breakdown else pd.DataFrame()
    MACC = load_macc(args.macc) if args.macc else pd.DataFrame()
    PASS = load_pass_through(args.pass_through) if args.pass_through else pd.DataFrame()
    ELAS = load_elasticity(args.elasticity) if args.elasticity else pd.DataFrame()
    SUBS = load_subsidies(args.subsidies) if args.subsidies else pd.DataFrame()
    POL  = load_policies(args.policies) if args.policies else pd.DataFrame()

    scen_tbl = resolve_policy_table(POL, args.scenario or None, horizons)
    scen_tbl.to_csv(outdir / "scenario_path.csv", index=False)

    results = run_engine(
        portfolio=PORT, emissions=EMI, financials=FIN, rev_breakdown=REV,
        macc_df=MACC, pass_df=PASS, elas_df=ELAS, subs_df=SUBS,
        scen_tbl=scen_tbl, horizons=horizons, s3_alpha=args.s3_alpha,
        absorption_cap=args.absorption_cap, spread_beta=args.spread_beta,
        mc_paths=args.mc_paths, mc_carbon_sd=args.mc_carbon_sd,
        mc_pass_sd=args.mc_pass_sd, mc_elas_sd=args.mc_elas_sd
    )

    # Write outputs
    results["issuer_impacts"].to_csv(outdir / "issuer_impacts.csv", index=False)
    results["factor_attribution"].to_csv(outdir / "factor_attribution.csv", index=False)
    results["portfolio_aggregates"].to_csv(outdir / "portfolio_aggregates.csv", index=False)
    if not results["sensitivities"].empty:
        results["sensitivities"].to_csv(outdir / "sensitivities.csv", index=False)
    if not results["mc_distribution"].empty:
        results["mc_distribution"].to_csv(outdir / "mc_distribution.csv", index=False)

    # Summary JSON
    iss = results["issuer_impacts"]
    tot = iss.groupby("year", as_index=False).agg(delta_value_usd=("delta_value_usd","sum"),
                                                  delta_ebitda_usd=("delta_ebitda_usd","sum"),
                                                  mv_usd=("mv_usd","sum"))
    tot["delta_value_pct_mv"] = tot["delta_value_usd"] / tot["mv_usd"].replace(0, np.nan)
    # top contributors latest year
    y_last = max(parse_years(args.horizons))
    sub_last = iss[iss["year"]==y_last].copy()
    top5 = (sub_last.groupby("sector", as_index=False)["delta_value_usd"].sum()
            .sort_values("delta_value_usd").head(5).to_dict(orient="records"))
    summary = {
        "horizons": horizons,
        "portfolio_delta_value": tot.to_dict(orient="records"),
        "top5_sector_negative_contributors_latest": top5
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        portfolio=args.portfolio, emissions=(args.emissions or None), financials=(args.financials or None),
        rev_breakdown=(args.rev_breakdown or None), macc=(args.macc or None), pass_through=(args.pass_through or None),
        elasticity=(args.elasticity or None), subsidies=(args.subsidies or None), policies=(args.policies or None),
        horizons=args.horizons, scenario=(args.scenario or None), s3_alpha=args.s3_alpha, spread_beta=args.spread_beta,
        absorption_cap=args.absorption_cap, mc_paths=args.mc_paths, mc_carbon_sd=args.mc_carbon_sd,
        mc_pass_sd=args.mc_pass_sd, mc_elas_sd=args.mc_elas_sd, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Climate Transition Risk — Results ==")
    for _, r in tot.sort_values("year").iterrows():
        print(f"{int(r['year'])}: ΔV {r['delta_value_usd']:,.0f} USD ({r['delta_value_pct_mv']*100:,.2f}% of MV), ΔEBITDA {r['delta_ebitda_usd']:,.0f} USD")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
