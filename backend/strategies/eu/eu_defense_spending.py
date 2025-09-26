#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eu_defense_spending.py — EU defense spending analytics, gaps to NATO targets, and forward projections

What this does
--------------
Given country-level budgets, GDP and (optionally) procurement programs & industry capacity,
this script produces historical and forward-looking views of:
- Defense spend (EUR bn) and as % of GDP
- Gap to NATO 2% of GDP target (and any custom target path)
- Equipment procurement share and gap to 20% NATO guideline
- Forward projection to a target year under configurable ramps
- Program funding coverage vs equipment budgets (funding gap by year)
- Very simple munitions demand vs industry capacity check
- EU aggregate rollups and country summaries

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--budgets budgets.csv   (required)
    Columns (suggested):
      country, year, defense_eur_bn, gdp_eur_bn,
      equipment_share (0..1 optional), personnel_share, o&m_share
    Notes:
      - If equipment_share is missing, we infer from personnel_share and o&m_share, else fallback to 0.20.

--programs programs.csv (optional; procurement pipeline)
    Columns:
      country, program, start_year, end_year, cost_eur_bn, category(optional: Ammunition/Missile/Vehicle/Other)

--capacity capacity.csv (optional; industry capacity for quick stress)
    Columns:
      country, category(Ammunition/Missile/Vehicle/Other), annual_capacity_eur_bn

--assumptions assumptions.csv (optional; country overrides)
    Columns:
      country, start_year(optional), target2pct_year(optional), target2pct_value(optional, default 0.02),
      equip20pct_year(optional), equip_target(optional, default 0.20)

Key options
-----------
--start 2020                First modeled year (inclusive)
--end 2035                  Last modeled year (inclusive)
--default-equip 0.20        Default equipment share if unknown
--target2pct-year 2028      Year by which each country ramps to 2% (unless overridden per-country)
--target2pct 0.02           Target defense % of GDP
--equip20pct-year 2028      Year by which each country ramps to 20% equipment share (unless overridden)
--equip-target 0.20         Target equipment share
--eu-members "AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IE,IT,LV,LT,LU,MT,NL,PL,PT,RO,SK,SI,ES,SE"
--outdir out_eu_defense

Outputs
-------
- spend_timeseries.csv        Country-year panel with spend, %GDP, equipment share and gaps
- country_summary.csv         Latest-year snapshot with gaps and required uplifts to meet targets
- eu_aggregate.csv            EU rollup by year (spend, %GDP-weighted, equipment share)
- program_funding_gap.csv     By country-year: equipment budget vs program needs (gap/surplus)
- capacity_check.csv          Category-level demand vs capacity (very coarse)
- summary.json                Headline KPIs
- config.json                 Reproducibility dump

Notes
-----
- This is a *scenario calculator*, not an official dataset. Bring your own official time series.
- All currency fields assumed EUR bn (billions). GDP must match that unit for % calculations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


# ----------------- helpers -----------------
def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None


def read_csv_maybe(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


def years_index(start: int, end: int) -> pd.Index:
    return pd.Index(list(range(int(start), int(end) + 1)), name="year")


def coalesce(a, b, default=None):
    if a is not None and not (isinstance(a, float) and np.isnan(a)):
        return a
    if b is not None and not (isinstance(b, float) and np.isnan(b)):
        return b
    return default


# ----------------- inputs -----------------
def load_budgets(path: str, start: int, end: int, default_equip: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df, "country") or "country"): "country",
        (ncol(df, "year") or "year"): "year",
        (ncol(df, "defense_eur_bn") or ncol(df, "defence_eur_bn") or "defense_eur_bn"): "defense_eur_bn",
        (ncol(df, "gdp_eur_bn") or "gdp_eur_bn"): "gdp_eur_bn",
        (ncol(df, "equipment_share") or "equipment_share"): "equipment_share",
        (ncol(df, "personnel_share") or "personnel_share"): "personnel_share",
        (ncol(df, "o&m_share") or ncol(df, "om_share") or "o&m_share"): "om_share",
    }
    df = df.rename(columns=ren)
    df["country"] = df["country"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ["defense_eur_bn", "gdp_eur_bn", "equipment_share", "personnel_share", "om_share"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill equipment share
    if "equipment_share" not in df.columns:
        df["equipment_share"] = np.nan
    # Where both personnel & o&m provided, derive equipment_share = 1 - (personnel + o&m)
    if "personnel_share" in df.columns and "om_share" in df.columns:
        est = 1.0 - (df["personnel_share"].fillna(0) + df["om_share"].fillna(0))
        df["equipment_share"] = df["equipment_share"].fillna(est)
    df["equipment_share"] = df["equipment_share"].fillna(float(default_equip)).clip(0, 1)

    # Restrict to modeled years; forward/back fill per country
    yr = years_index(start, end)
    out = []
    for ctry, g in df.groupby("country"):
        g = g.set_index("year").sort_index()
        g = g.reindex(yr).interpolate().ffill().bfill()
        g["country"] = ctry
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def load_programs(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["country", "program", "start_year", "end_year", "cost_eur_bn", "category"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df, "country") or "country"): "country",
        (ncol(df, "program") or "program"): "program",
        (ncol(df, "start_year") or "start_year"): "start_year",
        (ncol(df, "end_year") or "end_year"): "end_year",
        (ncol(df, "cost_eur_bn") or "cost_eur_bn"): "cost_eur_bn",
        (ncol(df, "category") or "category"): "category",
    }
    df = df.rename(columns=ren)
    for c in ["start_year", "end_year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["cost_eur_bn"] = pd.to_numeric(df.get("cost_eur_bn", np.nan), errors="coerce")
    df["category"] = df.get("category", "Other").fillna("Other")
    return df


def load_capacity(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["country", "category", "annual_capacity_eur_bn"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df, "country") or "country"): "country",
        (ncol(df, "category") or "category"): "category",
        (ncol(df, "annual_capacity_eur_bn") or "annual_capacity_eur_bn"): "annual_capacity_eur_bn",
    }
    df = df.rename(columns=ren)
    df["annual_capacity_eur_bn"] = pd.to_numeric(df["annual_capacity_eur_bn"], errors="coerce")
    return df


def load_assumptions(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["country", "start_year", "target2pct_year", "target2pct_value", "equip20pct_year", "equip_target"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df, "country") or "country"): "country",
        (ncol(df, "start_year") or "start_year"): "start_year",
        (ncol(df, "target2pct_year") or "target2pct_year"): "target2pct_year",
        (ncol(df, "target2pct_value") or "target2pct_value"): "target2pct_value",
        (ncol(df, "equip20pct_year") or "equip20pct_year"): "equip20pct_year",
        (ncol(df, "equip_target") or "equip_target"): "equip_target",
    }
    df = df.rename(columns=ren)
    for c in ["start_year", "target2pct_year", "equip20pct_year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["target2pct_value", "equip_target"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------- scenario engine -----------------
@dataclass
class Params:
    start: int
    end: int
    target2pct_year: int
    target2pct: float
    equip20pct_year: int
    equip_target: float
    eu_members: List[str]
    default_equip: float
    outdir: str


def ramp_to_target(series: pd.Series, target_year: int, target_value: float) -> pd.Series:
    """Linear ramp from first valid year in series to target value at target_year; hold flat after."""
    s = series.copy()
    idx = s.index.astype(int)
    y0 = int(idx[0])
    # build simple linear toward target_year
    m = (target_value - s.loc[y0]) / max(1, (target_year - y0))
    projected = []
    for y in idx:
        if y <= target_year:
            projected.append(s.loc[y0] + m * (y - y0))
        else:
            projected.append(target_value)
    return pd.Series(projected, index=idx)


def project_country(g: pd.DataFrame, p: Params, overrides: Optional[pd.Series]) -> pd.DataFrame:
    """Return panel for one country including projected %GDP and equipment share paths."""
    g = g.set_index("year").sort_index()
    # Build target years/values (apply overrides if provided)
    t2_year = int(coalesce(overrides.get("target2pct_year") if overrides is not None else None, p.target2pct_year, p.target2pct_year))
    t2_val = float(coalesce(overrides.get("target2pct_value") if overrides is not None else None, p.target2pct, p.target2pct))
    e20_year = int(coalesce(overrides.get("equip20pct_year") if overrides is not None else None, p.equip20pct_year, p.equip20pct_year))
    e_target = float(coalesce(overrides.get("equip_target") if overrides is not None else None, p.equip_target, p.equip_target))

    # Current %GDP series from inputs
    pct_series = (g["defense_eur_bn"] / g["gdp_eur_bn"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equip_series = g["equipment_share"].clip(0, 1)

    # Ramp forward
    yidx = g.index
    pct_proj = ramp_to_target(pd.Series(pct_series.iloc[0], index=yidx), t2_year, t2_val)
    # If history already above target in some years, keep max of history vs ramp to avoid falling
    pct_proj = np.maximum(pct_proj.values, pct_series.values)

    equip_proj = ramp_to_target(pd.Series(equip_series.iloc[0], index=yidx), e20_year, e_target)
    equip_proj = np.maximum(equip_proj.values, equip_series.values)

    out = g.copy()
    out["pct_gdp"] = pct_series.values
    out["pct_gdp_proj"] = pct_proj
    out["equipment_share_proj"] = equip_proj

    # Compute required defense and equipment budgets to hit targets (bn EUR)
    out["defense_needed_eur_bn"] = (out["gdp_eur_bn"] * out["pct_gdp_proj"]).round(3)
    out["defense_gap_eur_bn"] = (out["defense_needed_eur_bn"] - out["defense_eur_bn"]).round(3)

    out["equip_budget_eur_bn"] = (out["defense_eur_bn"] * out["equipment_share"]).round(3)
    out["equip_budget_needed_eur_bn"] = (out["defense_needed_eur_bn"] * out["equipment_share_proj"]).round(3)
    out["equip_gap_eur_bn"] = (out["equip_budget_needed_eur_bn"] - out["equip_budget_eur_bn"]).round(3)

    return out.reset_index()


def programs_annualized(programs: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Spread program cost evenly over years between start_year and end_year (inclusive)."""
    if programs.empty:
        return pd.DataFrame(columns=["country", "year", "program_need_eur_bn", "category"])
    rows = []
    for _, r in programs.iterrows():
        sy = int(r["start_year"]); ey = int(r["end_year"])
        if not (sy <= ey):
            sy, ey = ey, sy
        span = max(1, ey - sy + 1)
        per_year = float(r.get("cost_eur_bn", 0.0)) / span
        for y in range(max(start, sy), min(end, ey) + 1):
            rows.append({
                "country": r["country"], "year": y,
                "program_need_eur_bn": per_year,
                "category": r.get("category", "Other")
            })
    df = pd.DataFrame(rows)
    return df.groupby(["country", "year", "category"], as_index=False)["program_need_eur_bn"].sum()


def capacity_check(program_needs: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Compare country/category program needs vs stated annual capacity on the SAME category labels."""
    if program_needs.empty or capacity.empty:
        return pd.DataFrame(columns=["country", "category", "need_eur_bn", "capacity_eur_bn", "gap_eur_bn"])
    need = program_needs.groupby(["country", "category"])["program_need_eur_bn"].sum().reset_index()
    cap = capacity.groupby(["country", "category"])["annual_capacity_eur_bn"].sum().reset_index()
    merged = need.merge(cap, on=["country", "category"], how="left").fillna({"annual_capacity_eur_bn": 0.0})
    merged = merged.rename(columns={"program_need_eur_bn": "need_eur_bn", "annual_capacity_eur_bn": "capacity_eur_bn"})
    merged["gap_eur_bn"] = (merged["need_eur_bn"] - merged["capacity_eur_bn"]).round(3)
    return merged.sort_values(["country", "category"])


# ----------------- CLI -----------------
@dataclass
class Config:
    budgets: str
    programs: Optional[str]
    capacity: Optional[str]
    assumptions: Optional[str]
    start: int
    end: int
    default_equip: float
    target2pct_year: int
    target2pct: float
    equip20pct_year: int
    equip_target: float
    eu_members: List[str]
    outdir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EU defense spending analytics & projections")
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--programs", default="")
    ap.add_argument("--capacity", default="")
    ap.add_argument("--assumptions", default="")
    ap.add_argument("--start", type=int, default=2020)
    ap.add_argument("--end", type=int, default=2035)
    ap.add_argument("--default-equip", type=float, default=0.20)
    ap.add_argument("--target2pct-year", type=int, default=2028)
    ap.add_argument("--target2pct", type=float, default=0.02)
    ap.add_argument("--equip20pct-year", type=int, default=2028)
    ap.add_argument("--equip-target", type=float, default=0.20)
    ap.add_argument("--eu-members", default="AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IE,IT,LV,LT,LU,MT,NL,PL,PT,RO,SK,SI,ES,SE")
    ap.add_argument("--outdir", default="out_eu_defense")
    return ap.parse_args()


# ----------------- main -----------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    eu_members = [x.strip().upper() for x in str(args.eu_members).split(",") if x.strip()]
    p = Params(
        start=args.start, end=args.end,
        target2pct_year=args.target2pct_year, target2pct=args.target2pct,
        equip20pct_year=args.equip20pct_year, equip_target=args.equip_target,
        eu_members=eu_members, default_equip=args.default_equip, outdir=args.outdir
    )

    budgets = load_budgets(args.budgets, p.start, p.end, p.default_equip)
    programs = load_programs(args.programs) if args.programs else pd.DataFrame()
    capacity = load_capacity(args.capacity) if args.capacity else pd.DataFrame()
    assumptions = load_assumptions(args.assumptions) if args.assumptions else pd.DataFrame()

    # Per-country projections
    spend_rows = []
    for ctry, g in budgets.groupby("country"):
        ov = assumptions[assumptions["country"] == ctry].iloc[0] if not assumptions[assumptions["country"] == ctry].empty else pd.Series(dtype=object)
        panel = project_country(g.copy(), p, ov)
        panel["country"] = ctry
        spend_rows.append(panel)
    spend = pd.concat(spend_rows, ignore_index=True)

    # EU aggregate
    spend["is_eu"] = spend["country"].str.upper().isin(p.eu_members).astype(int)
    eu = (spend[spend["is_eu"] == 1]
          .groupby("year", as_index=False)
          .agg(
              defense_eur_bn=("defense_eur_bn", "sum"),
              gdp_eur_bn=("gdp_eur_bn", "sum"),
              defense_needed_eur_bn=("defense_needed_eur_bn", "sum"),
              equip_budget_eur_bn=("equip_budget_eur_bn", "sum"),
              equip_budget_needed_eur_bn=("equip_budget_needed_eur_bn", "sum"),
          ))
    eu["pct_gdp"] = (eu["defense_eur_bn"] / eu["gdp_eur_bn"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eu["pct_gdp_proj"] = (eu["defense_needed_eur_bn"] / eu["gdp_eur_bn"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eu["equip_share"] = (eu["equip_budget_eur_bn"] / eu["defense_eur_bn"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eu["equip_share_proj"] = (eu["equip_budget_needed_eur_bn"] / eu["defense_needed_eur_bn"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eu["defense_gap_eur_bn"] = (eu["defense_needed_eur_bn"] - eu["defense_eur_bn"]).round(3)
    eu["equip_gap_eur_bn"] = (eu["equip_budget_needed_eur_bn"] - eu["equip_budget_eur_bn"]).round(3)

    # Country latest snapshot (last modeled year)
    last_year = int(spend["year"].max())
    snap = spend[spend["year"] == last_year].copy()
    snap["gap_to_2pct_eur_bn"] = (snap["defense_needed_eur_bn"] - snap["defense_eur_bn"]).round(3)
    snap["gap_to_20pct_equip_eur_bn"] = (snap["equip_budget_needed_eur_bn"] - snap["equip_budget_eur_bn"]).round(3)
    country_summary = snap[[
        "country", "year",
        "defense_eur_bn", "gdp_eur_bn", "pct_gdp", "pct_gdp_proj",
        "gap_to_2pct_eur_bn",
        "equipment_share", "equipment_share_proj", "equip_budget_eur_bn", "equip_budget_needed_eur_bn", "gap_to_20pct_equip_eur_bn"
    ]].sort_values("gap_to_2pct_eur_bn", ascending=False)

    # Program needs vs equipment budgets (by country-year)
    prog_yr = programs_annualized(programs, p.start, p.end)
    if not prog_yr.empty:
        eq_budget_year = spend.groupby(["country", "year"], as_index=False).agg(
            equip_budget_eur_bn=("equip_budget_eur_bn", "sum"),
            equip_budget_needed_eur_bn=("equip_budget_needed_eur_bn", "sum")
        )
        prog_gap = prog_yr.groupby(["country", "year"], as_index=False)["program_need_eur_bn"].sum() \
            .merge(eq_budget_year, on=["country", "year"], how="left")
        prog_gap["funding_gap_vs_actual_eur_bn"] = (prog_gap["program_need_eur_bn"] - prog_gap["equip_budget_eur_bn"]).round(3)
        prog_gap["funding_gap_vs_needed_eur_bn"] = (prog_gap["program_need_eur_bn"] - prog_gap["equip_budget_needed_eur_bn"]).round(3)
    else:
        prog_gap = pd.DataFrame(columns=["country", "year", "program_need_eur_bn", "equip_budget_eur_bn", "equip_budget_needed_eur_bn", "funding_gap_vs_actual_eur_bn", "funding_gap_vs_needed_eur_bn"])

    # Capacity check (category-level, very coarse)
    if not prog_yr.empty and not capacity.empty:
        cap_tbl = capacity_check(prog_yr, capacity)
    else:
        cap_tbl = pd.DataFrame(columns=["country", "category", "need_eur_bn", "capacity_eur_bn", "gap_eur_bn"])

    # Write outputs
    spend.to_csv(outdir / "spend_timeseries.csv", index=False)
    country_summary.to_csv(outdir / "country_summary.csv", index=False)
    eu.to_csv(outdir / "eu_aggregate.csv", index=False)
    prog_gap.to_csv(outdir / "program_funding_gap.csv", index=False)
    cap_tbl.to_csv(outdir / "capacity_check.csv", index=False)

    # KPIs
    eu_last = eu[eu["year"] == last_year].iloc[0] if not eu.empty else pd.Series(dtype=float)
    kpi = {
        "last_year": last_year,
        "eu_defense_eur_bn": float(eu_last.get("defense_eur_bn", np.nan)) if not eu.empty else None,
        "eu_pct_gdp": float(eu_last.get("pct_gdp", np.nan)) if not eu.empty else None,
        "eu_defense_gap_eur_bn": float(eu_last.get("defense_gap_eur_bn", np.nan)) if not eu.empty else None,
        "countries": int(spend["country"].nunique()),
        "program_rows": int(len(programs)),
        "capacity_rows": int(len(capacity)),
        "top_5_gaps": country_summary[["country", "gap_to_2pct_eur_bn"]].head(5).set_index("country")["gap_to_2pct_eur_bn"].round(2).to_dict(),
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        budgets=args.budgets, programs=args.programs or None, capacity=args.capacity or None, assumptions=args.assumptions or None,
        start=p.start, end=p.end, default_equip=p.default_equip,
        target2pct_year=p.target2pct_year, target2pct=p.target2pct,
        equip20pct_year=p.equip20pct_year, equip_target=p.equip_target,
        eu_members=p.eu_members, outdir=args.outdir
    )), indent=2))

    # Console
    print("== EU Defense Spending Analytics ==")
    print(f"Modeled years: {p.start}-{p.end}  Countries: {kpi['countries']}")
    if not eu.empty:
        print(f"EU {last_year}: spend €{eu_last['defense_eur_bn']:.0f} bn  |  %GDP {eu_last['pct_gdp']*100:.2f}%  |  Gap to target €{eu_last['defense_gap_eur_bn']:.1f} bn")
    if kpi["top_5_gaps"]:
        print("Top gaps to 2% (bn):", kpi["top_5_gaps"])
    print("Outputs in:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
