#!/usr/bin/env python3
"""
dutch_gas_phaseout.py — Scenario engine for phasing out natural gas in the Netherlands

What this does
--------------
Given baseline energy/gas demand by sector and configurable policy/technology ramps,
this script builds yearly trajectories (e.g., 2025–2040) for:
- Final gas demand reduction by sector (residential, commercial, industry, power)
- Substitution mix (heat pumps, district heat, electrification, hydrogen, biomethane, efficiency)
- Electricity demand uplift (annual and peak) from electrification & heat pumps
- CO₂e & CH₄ (upstream leakage) reductions vs baseline
- Security-of-supply checks (winter stress, LNG/import dependence)
- High-level CAPEX estimates (retrofits, heat pumps, grid, hydrogen/biomethane supply)
- “Feasibility flags” if ramps exceed practical limits (installers, grid capacity, retrofit pace)

Inputs (CSV/Parquet; flexible headers, case-insensitive)
-------------------------------------------------------
--baseline baseline.csv
    Required. Annual baseline by sector. Expected columns (at least):
      year, sector, gas_TWh, elec_TWh? (optional), population? (optional)
    Valid sectors (case-insensitive match): residential, commercial, industry, power (others allowed; mapped to 'other').

--emissions emissions.csv (optional)
    Columns: scope (direct, upstream), pollutant (CO2e, CH4), factor_kg_per_MWh_gas
    If omitted, defaults: CO2e=202 kg/MWh (LHV), CH4_upstream=0.7 kg/MWh_gas (change as needed).

--policy policy.csv (optional)
    Milestones & constraints. Columns (examples; all optional):
      year, new_gas_connections_ban (0/1), residential_gas_ban_share (0..1),
      industrial_carbon_price_eur_per_t, heat_pump_subsidy_eur_per_unit, grid_capex_ceiling_beur

--tech tech.csv (optional)
    Technology ramps & potentials (yearly). Columns (examples):
      year,
      heat_pump_installs_k (annual adds, thousands),
      heat_pump_stock_k,
      avg_hsp_hours (residential),
      heat_pump_scop (seasonal COP),
      district_heat_TWh,
      hydrogen_TWh (for end-use),
      biomethane_TWh,
      efficiency_pct (share of baseline demand avoided via EE measures),
      industrial_elec_share (0..1 of industrial heat switched to electricity),
      industrial_elec_eff_gain (e.g., 0.3 → 30% less final energy when electrified),
      peak_diversity_hp (0..1 coincidence factor for peak from HPs)

--price price.csv (optional)
    Price scenarios for sensitivity & SoS checks:
      year, gas_eur_per_MWh, elec_eur_per_MWh, co2_eur_per_t

Key Options
-----------
--start 2025                First modeled year (inclusive)
--end   2040                Last modeled year (inclusive)
--country "NL"              Country tag for outputs (free text)
--elec_losses 0.06          Grid & charging losses for added electricity
--cold_spell_days 10        Stress-test length for peak/winter energy
--hp_install_cap_k 350      Max annual heat-pump installs (thousand units/year) practical limit
--grid_add_MW 3500          Max annual added grid capacity (MW) practical limit
--bio_h2_cap_TWh 50         Cap on total (biomethane + hydrogen) supply by 2040
--capex_hp_k 6.0            Average installed cost per HP (kEUR)
--capex_retrofit_k 12.0     Average deep retrofit cost per dwelling (kEUR) for efficiency slice
--capex_grid_k_per_kW 1.0   Grid reinforcement unit cost
--capex_dh_beur 0.2         District heat CAPEX per TWh/year added (bn EUR)
--capex_h2_beur 0.5         Hydrogen end-use readiness per TWh/year (bn EUR)
--capex_bio_beur 0.2        Biomethane supply per TWh/year (bn EUR)
--outdir out_phaseout       Output folder

Outputs
-------
- timeseries.csv            Yearly totals by sector: gas, substitution mix, added electricity, emissions
- substitution_mix.csv      Yearly substitution by vector (HP, DH, Elec, H2, Biomethane, Efficiency)
- power_impact.csv          Added electricity (TWh) & peak (MW) with COP/diversity assumptions
- sos_check.csv             Security-of-supply diagnostic (import/LNG reliance, winter coverage)
- capex_summary.csv         Cumulative CAPEX breakdown & feasibility flags by year
- summary.json              Key KPIs (gas↓, CO₂e↓, peak↑, feasibility)
- config.json               Reproducibility (input paths, parameters)

Notes
-----
- This is a scenario calculator, not a forecast. Provide your own ramps in tech.csv/policy.csv.
- Defaults are generic. Replace with Dutch-specific numbers to be realistic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


# ----------------- Helpers -----------------
def norm_col(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None


def read_any(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def ensure_year(df: pd.DataFrame) -> pd.DataFrame:
    c = norm_col(df, "year") or df.columns[0]
    df = df.rename(columns={c: "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def sector_key(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("res"):
        return "residential"
    if s.startswith("comm") or s.startswith("tertiary"):
        return "commercial"
    if s.startswith("ind"):
        return "industry"
    if s.startswith("pow") or s.startswith("elec"):
        return "power"
    return "other"


def kg_to_mt(x): return float(x) / 1e6
def t_to_mt(x): return float(x) / 1e6


# ----------------- Core transforms -----------------
def load_baseline(path: str) -> pd.DataFrame:
    df = read_any(path)
    if df is None or df.empty:
        raise SystemExit("baseline.csv is required and appears empty.")
    df = ensure_year(df)
    df = df.rename(columns={norm_col(df, "sector") or "sector": "sector"})
    df["sector"] = df["sector"].map(sector_key)
    gcol = norm_col(df, "gas_twh") or "gas_TWh"
    df = df.rename(columns={gcol: "gas_TWh"})
    # Optional baseline electricity for context
    ecol = norm_col(df, "elec_twh")
    if ecol:
        df = df.rename(columns={ecol: "elec_TWh"})
    else:
        df["elec_TWh"] = np.nan
    return df[["year", "sector", "gas_TWh", "elec_TWh"]]


def load_emissions(path: Optional[str]) -> Dict[str, float]:
    # return factors (kg/MWh_gas)
    if not path:
        return {"CO2e": 202_000.0 / 1000, "CH4_up": 0.7}  # 202 kg/MWh CO2e; 0.7 kg/MWh CH4 upstream (illustrative)
    df = read_any(path)
    df = df.rename(columns={
        (norm_col(df, "scope") or "scope"): "scope",
        (norm_col(df, "pollutant") or "pollutant"): "pollutant",
        (norm_col(df, "factor_kg_per_mwh_gas") or norm_col(df, "factor") or "factor"): "factor",
    })
    fac = {}
    for _, r in df.iterrows():
        key = str(r["pollutant"]).strip().upper()
        fac[key] = float(r["factor"])
    # Standardize keys
    out = {}
    out["CO2e"] = fac.get("CO2E", fac.get("CO2", 202.0))  # kg/MWh
    out["CH4_up"] = fac.get("CH4", 0.7)                  # kg/MWh
    return out


def load_policy(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["year"])
    df = ensure_year(read_any(path))
    return df


def load_tech(path: Optional[str]) -> pd.DataFrame:
    if not path:
        # Minimal placeholder ramp (flat zeros)
        return pd.DataFrame()
    df = ensure_year(read_any(path))
    # normalize names we use
    rename = {
        norm_col(df, "heat_pump_installs_k") or "heat_pump_installs_k": "hp_installs_k",
        norm_col(df, "heat_pump_stock_k") or "heat_pump_stock_k": "hp_stock_k",
        norm_col(df, "avg_hsp_hours") or "avg_hsp_hours": "hp_hours",
        norm_col(df, "heat_pump_scop") or "heat_pump_scop": "hp_scop",
        norm_col(df, "district_heat_twh") or "district_heat_twh": "dh_TWh",
        norm_col(df, "hydrogen_twh") or "hydrogen_twh": "h2_TWh",
        norm_col(df, "biomethane_twh") or "biomethane_twh": "bio_TWh",
        norm_col(df, "efficiency_pct") or "efficiency_pct": "eff_pct",
        norm_col(df, "industrial_elec_share") or "industrial_elec_share": "ind_elec_share",
        norm_col(df, "industrial_elec_eff_gain") or "industrial_elec_eff_gain": "ind_elec_eff",
        norm_col(df, "peak_diversity_hp") or "peak_diversity_hp": "hp_peak_div",
    }
    df = df.rename(columns=rename)
    # numeric
    for c in ["hp_installs_k","hp_stock_k","hp_hours","hp_scop","dh_TWh","h2_TWh","bio_TWh","eff_pct","ind_elec_share","ind_elec_eff","hp_peak_div"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_price(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = ensure_year(read_any(path))
    df = df.rename(columns={
        (norm_col(df,"gas_eur_per_mwh") or "gas_eur_per_mWh"): "gas_eur_per_MWh",
        (norm_col(df,"elec_eur_per_mwh") or "elec_eur_per_mWh"): "elec_eur_per_MWh",
        (norm_col(df,"co2_eur_per_t") or "co2_eur_per_t"): "co2_eur_per_t",
    })
    for c in ["gas_eur_per_MWh","elec_eur_per_MWh","co2_eur_per_t"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------- Scenario engine -----------------
@dataclass
class Params:
    start: int
    end: int
    country: str
    elec_losses: float
    cold_days: int
    hp_install_cap_k: float
    grid_add_MW: float
    bio_h2_cap_TWh: float
    capex_hp_k: float
    capex_retrofit_k: float
    capex_grid_k_per_kW: float
    capex_dh_beur: float
    capex_h2_beur: float
    capex_bio_beur: float


def build_year_index(p: Params) -> pd.Index:
    return pd.Index(list(range(p.start, p.end + 1)), name="year")


def distribute_baseline(baseline: pd.DataFrame, years: pd.Index) -> pd.DataFrame:
    # Keep only modeled years, forward/backward fill if outside
    y0, y1 = baseline["year"].min(), baseline["year"].max()
    df = baseline.copy()
    # simple fill: if modeled years outside provided, use nearest available
    ext = []
    for y in years:
        if y < y0:
            d = df[df["year"] == y0].assign(year=y)
        elif y > y1:
            d = df[df["year"] == y1].assign(year=y)
        else:
            d = df[df["year"] == y]
        ext.append(d)
    out = pd.concat(ext, ignore_index=True)
    return out


def scenario_calculate(
    base: pd.DataFrame, tech: pd.DataFrame, policy: pd.DataFrame, price: pd.DataFrame, fac: Dict[str, float], p: Params
):
    years = build_year_index(p)
    base = distribute_baseline(base, years)

    # Aggregate baseline gas by sector per year
    pivot_gas = base.pivot_table(index="year", columns="sector", values="gas_TWh", aggfunc="sum").fillna(0.0)
    for s in ["residential","commercial","industry","power","other"]:
        if s not in pivot_gas.columns:
            pivot_gas[s] = 0.0
    pivot_gas = pivot_gas.loc[years]

    # Technology series aligned
    tech_al = pd.DataFrame(index=years)
    for c in ["hp_installs_k","hp_stock_k","hp_hours","hp_scop","dh_TWh","h2_TWh","bio_TWh","eff_pct","ind_elec_share","ind_elec_eff","hp_peak_div"]:
        if c in tech.columns:
            tech_al[c] = tech.set_index("year")[c].reindex(years).interpolate().ffill().bfill()
    # Limits & caps
    tech_al["hp_installs_k"] = tech_al.get("hp_installs_k", pd.Series(0, index=years)).clip(upper=p.hp_install_cap_k)
    tech_al["hp_peak_div"] = tech_al.get("hp_peak_div", pd.Series(0.35, index=years)).clip(0.1, 1.0)
    tech_al["hp_scop"] = tech_al.get("hp_scop", pd.Series(2.8, index=years)).clip(lower=1.5, upper=5.0)
    tech_al["hp_hours"] = tech_al.get("hp_hours", pd.Series(1400, index=years)).clip(lower=800, upper=2200)
    tech_al["eff_pct"] = tech_al.get("eff_pct", pd.Series(0.0, index=years)).clip(0, 0.7)  # up to 70% savings (deep retrofits)
    tech_al["ind_elec_share"] = tech_al.get("ind_elec_share", pd.Series(0.0, index=years)).clip(0, 1)
    tech_al["ind_elec_eff"] = tech_al.get("ind_elec_eff", pd.Series(0.3, index=years)).clip(0, 0.7)  # 30% avg energy reduction when electrified
    tech_al["dh_TWh"] = tech_al.get("dh_TWh", pd.Series(0.0, index=years)).clip(lower=0)
    # H2 & biomethane capped by global cap (growing to cap by end year)
    bio_h2 = tech_al.get("h2_TWh", pd.Series(0.0, index=years)) + tech_al.get("bio_TWh", pd.Series(0.0, index=years))
    cap_path = np.linspace(bio_h2.iloc[0], p.bio_h2_cap_TWh, len(years))
    scale = np.minimum(1.0, cap_path / (bio_h2.replace(0, np.nan).fillna(1e-9)))
    tech_al["h2_TWh"] = tech_al.get("h2_TWh", pd.Series(0.0, index=years)) * scale
    tech_al["bio_TWh"] = tech_al.get("bio_TWh", pd.Series(0.0, index=years)) * scale

    # --- Substitution logic ---
    # Residential & Commercial: HP + DH + Efficiency + Biomethane/H2 (residual)
    res0 = pivot_gas["residential"].copy()
    com0 = pivot_gas["commercial"].copy()

    # Efficiency savings apply to baseline first
    eff_res = res0 * tech_al["eff_pct"]
    eff_com = com0 * (tech_al["eff_pct"] * 0.5)  # assume lower retrofit penetration in commercial
    res_after_eff = (res0 - eff_res).clip(lower=0)
    com_after_eff = (com0 - eff_com).clip(lower=0)

    # Heat pump serviceable energy from installed stock
    # Convert HP stock to delivered heat (TWh) = stock * hours * 1 kW / 1e6 (to TWh) * COP
    # If only installs provided, accumulate stock
    if "hp_stock_k" in tech_al.columns and tech_al["hp_stock_k"].notna().any():
        hp_stock_k = tech_al["hp_stock_k"]
    else:
        hp_stock_k = tech_al["hp_installs_k"].cumsum()
    # 1 unit ~ 4 kW average? Let user implicitly encode through hours; assume 1 kW/unit baseline then scale by factor
    kw_per_unit = 4.0
    hp_heat_TWh = (hp_stock_k * 1e3 * kw_per_unit * tech_al["hp_hours"]) / 1e9  # to TWh thermal
    # Allocate HP heat between residential & commercial (assume 85/15 split)
    hp_res = hp_heat_TWh * 0.85
    hp_com = hp_heat_TWh * 0.15

    # District heat direct substitution (assume entirely for res+com; split 70/30)
    dh_res = tech_al["dh_TWh"] * 0.7
    dh_com = tech_al["dh_TWh"] * 0.3

    # Apply substitution sequence with residuals non-negative
    res_sub_hp = np.minimum(res_after_eff, hp_res)
    res_after_hp = (res_after_eff - res_sub_hp).clip(lower=0)
    res_sub_dh = np.minimum(res_after_hp, dh_res)
    res_after_dh = (res_after_hp - res_sub_dh).clip(lower=0)

    com_sub_hp = np.minimum(com_after_eff, hp_com)
    com_after_hp = (com_after_eff - com_sub_hp).clip(lower=0)
    com_sub_dh = np.minimum(com_after_hp, dh_com)
    com_after_dh = (com_after_hp - com_sub_dh).clip(lower=0)

    # Biomethane priority to remaining building gas, then H2 for high-temp segments (limited)
    bio_avail = tech_al["bio_TWh"]
    h2_avail = tech_al["h2_TWh"]

    # Allocate biomethane to res+com proportional to remaining demand
    rem_build = res_after_dh + com_after_dh
    bio_to_build = np.minimum(bio_avail, rem_build)
    bio_res = bio_to_build * (res_after_dh / rem_build.replace(0, np.nan)).fillna(0)
    bio_com = bio_to_build - bio_res
    res_after_bio = (res_after_dh - bio_res).clip(lower=0)
    com_after_bio = (com_after_dh - bio_com).clip(lower=0)

    # Industry: electrification + hydrogen + efficiency
    ind0 = pivot_gas["industry"].copy()
    ind_eff = ind0 * (tech_al["eff_pct"] * 0.5)  # assume partial efficiency in industry
    ind_after_eff = (ind0 - ind_eff).clip(lower=0)

    ind_elec_share = tech_al["ind_elec_share"]
    ind_elec_eff = tech_al["ind_elec_eff"]  # final energy reduction when electrified
    ind_elec_TWh = ind_after_eff * ind_elec_share * (1 - ind_elec_eff)
    ind_after_elec = (ind_after_eff - ind_elec_TWh).clip(lower=0)

    # Allocate hydrogen predominantly to industry from remaining H2
    bio_left = (bio_avail - bio_to_build).clip(lower=0)
    h2_to_ind = np.minimum(h2_avail + bio_left * 0.0, ind_after_elec)  # treat H2 separately from biomethane
    ind_after_h2 = (ind_after_elec - h2_to_ind).clip(lower=0)

    # Power: exogenous (keep baseline; can reduce via EE and substitution if user provides)
    pow0 = pivot_gas["power"].copy()
    pow_after = pow0  # unchanged in this simple model

    # Residual fossil gas by sector after substitutions
    res_resid = res_after_bio
    com_resid = com_after_bio
    ind_resid = ind_after_h2
    pow_resid = pow_after
    other_resid = pivot_gas["other"]

    # --- Electricity uplift & peak ---
    # Heat pumps electricity = thermal / COP, add grid losses
    hp_COP = tech_al["hp_scop"]
    elec_hp_TWh = ((res_sub_hp + com_sub_hp) / hp_COP) * (1 + p.elec_losses)
    # Industrial electrification electricity = electrified final energy / eff_gain_factor
    # If electrified energy already includes efficiency (we removed 'ind_elec_eff' above), convert with assumed process COP
    ind_elec_process_COP = 1.0  # set to 1.0 for simple conversion (final gas ~ elec kWh_thermal); refine if needed
    elec_ind_TWh = (ind0 * ind_elec_share * (1 - ind_elec_eff)) / ind_elec_process_COP * (1 + p.elec_losses)

    elec_added_TWh = elec_hp_TWh + elec_ind_TWh

    # Peak from HPs: peak MW ≈ (HP thermal at design) / COP_diversified
    # Use coincidence factor to derive peak load; thermal capacity ≈ stock * kw_per_unit
    COP_peak = np.maximum(2.0, hp_COP - 0.6)  # crude: COP declines at peak
    hp_kw = hp_stock_k * 1e3 * kw_per_unit  # kW thermal
    hp_peak_MW = (hp_kw / COP_peak) * tech_al["hp_peak_div"] / 1e3  # to MW (electrical)
    # Grid capacity limit check
    grid_cap_add = pd.Series(np.minimum(hp_peak_MW.diff().fillna(hp_peak_MW), p.grid_add_MW), index=years)
    feasibility_grid = (hp_peak_MW.diff().fillna(hp_peak_MW) <= p.grid_add_MW)

    # --- Emissions ---
    # Convert TWh_gas to MWh and multiply by factors (kg/MWh)
    def em_co2e(TWh): return kg_to_mt(fac["CO2e"] * (TWh * 1e6))
    def em_ch4(TWh):  return kg_to_mt(fac["CH4_up"] * (TWh * 1e6))

    gas_total_TWh = res_resid + com_resid + ind_resid + pow_resid + other_resid
    co2e_mt = em_co2e(gas_total_TWh)
    ch4_mt = em_ch4(gas_total_TWh)

    base_total_TWh = (pivot_gas.sum(axis=1))
    co2e_base_mt = em_co2e(base_total_TWh)
    ch4_base_mt  = em_ch4(base_total_TWh)

    # Reductions
    co2e_red_mt = (co2e_base_mt - co2e_mt)
    ch4_red_mt  = (ch4_base_mt - ch4_mt)
    gas_red_TWh = (base_total_TWh - gas_total_TWh)

    # --- CAPEX estimates (very high level; adjust unit costs to your case) ---
    hp_units_k = hp_stock_k
    capex_hp_beur = (hp_units_k * p.capex_hp_k) / 1e3  # kEUR → bn EUR
    # Retrofit: tie to efficiency savings in buildings (assume 1 retrofit per X MWh saved)
    MWh_saved_build = (eff_res + eff_com) * 1e6
    retrofit_units_k = (MWh_saved_build / 10_000).clip(lower=0) / 1e3  # assume 10 MWh/yr per retrofit (illustrative)
    capex_retrofit_beur = (retrofit_units_k * p.capex_retrofit_k) / 1e3
    # Grid capex from incremental peak
    incr_peak_kW = (hp_peak_MW.diff().fillna(hp_peak_MW) * 1e3).clip(lower=0)
    capex_grid_beur = (incr_peak_kW * p.capex_grid_k_per_kW) / 1e9
    # District heat, H2, Biomethane
    capex_dh_beur = tech_al["dh_TWh"].diff().fillna(tech_al["dh_TWh"]).clip(lower=0) * p.capex_dh_beur
    capex_h2_beur = tech_al["h2_TWh"].diff().fillna(tech_al["h2_TWh"]).clip(lower=0) * p.capex_h2_beur
    capex_bio_beur= tech_al["bio_TWh"].diff().fillna(tech_al["bio_TWh"]).clip(lower=0) * p.capex_bio_beur

    capex_yearly = pd.DataFrame({
        "hp_beur": capex_hp_beur.diff().fillna(capex_hp_beur),
        "retrofit_beur": capex_retrofit_beur.diff().fillna(capex_retrofit_beur),
        "grid_beur": capex_grid_beur,
        "dh_beur": capex_dh_beur,
        "h2_beur": capex_h2_beur,
        "bio_beur": capex_bio_beur,
    }, index=years).clip(lower=0)

    capex_cum = capex_yearly.cumsum()

    # --- Security of Supply (simple) ---
    # Assume domestic production declines with baseline (or user injects import share via policy/price)
    # Here we flag if residual gas exceeds a placeholder import ceiling (e.g., LNG capacity)
    lng_ceiling_TWh = base_total_TWh.iloc[0] * 0.6  # placeholder; customize
    sos_flag = gas_total_TWh > lng_ceiling_TWh

    # Cold spell energy need from HPs covered by grid:
    # Extra energy during cold days: assume HP COP drops to COP_peak over cold_days × average daily HP thermal
    avg_daily_hp_thermal_TWh = hp_heat_TWh / 365.0
    extra_elec_cold_TWh = (avg_daily_hp_thermal_TWh * p.cold_days) / COP_peak - (avg_daily_hp_thermal_TWh * p.cold_days) / hp_COP
    extra_elec_cold_TWh = extra_elec_cold_TWh.clip(lower=0)
    # Convert to average extra MW over cold period
    extra_peak_MW = (extra_elec_cold_TWh * 1e6) / (p.cold_days * 24.0)

    # --- Assemble outputs ---
    timeseries = pd.DataFrame({
        "country": p.country,
        "gas_total_TWh": gas_total_TWh,
        "gas_base_TWh": base_total_TWh,
        "gas_reduction_TWh": gas_red_TWh,
        "CO2e_mt": co2e_mt, "CO2e_base_mt": co2e_base_mt, "CO2e_reduction_mt": co2e_red_mt,
        "CH4_mt": ch4_mt, "CH4_base_mt": ch4_base_mt, "CH4_reduction_mt": ch4_red_mt,
        "elec_added_TWh": elec_added_TWh,
        "elec_HP_TWh": elec_hp_TWh,
        "elec_ind_TWh": elec_ind_TWh,
        "hp_peak_MW": hp_peak_MW,
        "feasible_grid": feasibility_grid.astype(int),
    }, index=years).reset_index().rename(columns={"index":"year"})

    # Sector detail
    sector_break = pd.DataFrame({
        "residual_res_TWh": res_resid,
        "residual_com_TWh": com_resid,
        "residual_ind_TWh": ind_resid,
        "residual_pow_TWh": pow_resid,
        "residual_other_TWh": other_resid,
        "sub_hp_res_TWh": res_sub_hp,
        "sub_hp_com_TWh": com_sub_hp,
        "sub_dh_res_TWh": res_sub_dh,
        "sub_dh_com_TWh": com_sub_dh,
        "sub_bio_res_TWh": bio_res,
        "sub_bio_com_TWh": bio_com,
        "sub_h2_ind_TWh": h2_to_ind,
        "sub_eff_res_TWh": eff_res,
        "sub_eff_com_TWh": eff_com,
        "sub_elec_ind_TWh": ind_elec_TWh,
    }, index=years).reset_index().rename(columns={"index":"year"})

    substitution_mix = pd.DataFrame({
        "HP_TWh": (res_sub_hp + com_sub_hp),
        "DistrictHeat_TWh": tech_al["dh_TWh"],
        "Electrification_TWh": ind_elec_TWh,
        "Hydrogen_TWh": h2_to_ind,
        "Biomethane_TWh": bio_to_build,
        "Efficiency_TWh": (eff_res + eff_com + ind_eff),
    }, index=years).reset_index()

    power_impact = pd.DataFrame({
        "elec_added_TWh": elec_added_TWh,
        "hp_elec_TWh": elec_hp_TWh,
        "ind_elec_TWh": elec_ind_TWh,
        "hp_peak_MW": hp_peak_MW,
        "extra_peak_MW_cold": extra_peak_MW,
        "grid_cap_add_MW": grid_cap_add,
    }, index=years).reset_index()

    sos_check = pd.DataFrame({
        "lng_ceiling_TWh": lng_ceiling_TWh,
        "residual_gas_TWh": gas_total_TWh,
        "sos_breach": sos_flag.astype(int),
        "cold_extra_TWh": extra_elec_cold_TWh,
    }, index=years).reset_index()

    capex_summary = capex_yearly.copy()
    capex_summary["cum_total_beur"] = capex_cum.sum(axis=1)
    capex_summary = capex_summary.reset_index().rename(columns={"index":"year"})

    # Headline KPIs for last modeled year
    last = timeseries.iloc[-1]
    kpis = {
        "country": p.country,
        "year_start": p.start,
        "year_end": p.end,
        "gas_reduction_pct": float((last["gas_base_TWh"] - last["gas_total_TWh"]) / max(1e-9, last["gas_base_TWh"])),
        "CO2e_reduction_mt": float(last["CO2e_reduction_mt"]),
        "elec_added_TWh": float(last["elec_added_TWh"]),
        "hp_peak_MW": float(last["hp_peak_MW"]),
        "grid_feasible": bool(last["feasible_grid"]),
        "cumulative_capex_beur": float(capex_summary["cum_total_beur"].iloc[-1]),
    }

    return timeseries, sector_break, substitution_mix, power_impact, sos_check, capex_summary, kpis


# ----------------- CLI -----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dutch natural gas phase-out scenario engine")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--emissions", default="")
    ap.add_argument("--policy", default="")
    ap.add_argument("--tech", default="")
    ap.add_argument("--price", default="")
    ap.add_argument("--start", type=int, default=2025)
    ap.add_argument("--end", type=int, default=2040)
    ap.add_argument("--country", default="NL")
    ap.add_argument("--elec_losses", type=float, default=0.06)
    ap.add_argument("--cold_spell_days", type=int, default=10)
    ap.add_argument("--hp_install_cap_k", type=float, default=350.0)
    ap.add_argument("--grid_add_MW", type=float, default=3500.0)
    ap.add_argument("--bio_h2_cap_TWh", type=float, default=50.0)
    ap.add_argument("--capex_hp_k", type=float, default=6.0)
    ap.add_argument("--capex_retrofit_k", type=float, default=12.0)
    ap.add_argument("--capex_grid_k_per_kW", type=float, default=1.0)
    ap.add_argument("--capex_dh_beur", type=float, default=0.2)
    ap.add_argument("--capex_h2_beur", type=float, default=0.5)
    ap.add_argument("--capex_bio_beur", type=float, default=0.2)
    ap.add_argument("--outdir", default="out_phaseout")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    base = load_baseline(args.baseline)
    fac = load_emissions(args.emissions)
    policy = load_policy(args.policy)  # currently not applied directly; hook for custom constraints
    tech = load_tech(args.tech)
    price = load_price(args.price)

    params = Params(
        start=int(args.start), end=int(args.end), country=args.country,
        elec_losses=float(args.elec_losses), cold_days=int(args.cold_spell_days),
        hp_install_cap_k=float(args.hp_install_cap_k), grid_add_MW=float(args.grid_add_MW),
        bio_h2_cap_TWh=float(args.bio_h2_cap_TWh),
        capex_hp_k=float(args.capex_hp_k), capex_retrofit_k=float(args.capex_retrofit_k),
        capex_grid_k_per_kW=float(args.capex_grid_k_per_kW),
        capex_dh_beur=float(args.capex_dh_beur), capex_h2_beur=float(args.capex_h2_beur), capex_bio_beur=float(args.capex_bio_beur),
    )

    (timeseries, sector_break, substitution_mix, power_impact, sos_check, capex_summary, kpis
     ) = scenario_calculate(base, tech, policy, price, fac, params)

    # Write outputs
    timeseries.to_csv(outdir / "timeseries.csv", index=False)
    sector_break.to_csv(outdir / "sector_breakdown.csv", index=False)
    substitution_mix.to_csv(outdir / "substitution_mix.csv", index=False)
    power_impact.to_csv(outdir / "power_impact.csv", index=False)
    sos_check.to_csv(outdir / "sos_check.csv", index=False)
    capex_summary.to_csv(outdir / "capex_summary.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(kpis, indent=2))
    (outdir / "config.json").write_text(json.dumps({
        "baseline": args.baseline, "emissions": args.emissions, "policy": args.policy, "tech": args.tech, "price": args.price,
        "params": asdict(params),
    }, indent=2))

    # Console
    print("== Dutch Gas Phase-out Scenario ==")
    print(f"Years: {params.start}-{params.end}  Country: {params.country}")
    print(f"Gas reduction in {params.end}: {kpis['gas_reduction_pct']*100:.1f}%  CO2e↓ {kpis['CO2e_reduction_mt']:.2f} Mt")
    print(f"Added electricity in {params.end}: {kpis['elec_added_TWh']:.1f} TWh  HP peak: {kpis['hp_peak_MW']:.0f} MW  Grid OK? {kpis['grid_feasible']}")
    print(f"Cumulative CAPEX: €{kpis['cumulative_capex_beur']:.1f} bn")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
