#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# patent_cliff_pharma.py
#
# Patent/Exclusivity Cliff Analyzer for Pharma & Biotech
# ------------------------------------------------------
# What this script does
# - Ingests a product-level CSV: company, ticker, drug, revenue, patent/exclusivity dates, modality (small-mol vs biologic)
# - Derives LOE (loss of exclusivity) date per product and aggregates revenue-at-risk by year
# - Projects post-LOE revenue erosion using configurable curves
# - Rolls up to company level and builds Base / Bear / Bull scenarios
# - Produces sensitivity tables (erosion speed × terminal share), plus optional plots
#
# Quick start
# ----------
# python patent_cliff_pharma.py \
#   --products products.csv \
#   --horizon 10 \
#   --asof 2025-09-06 \
#   --base-growth 0.03 --bear-growth 0.00 --bull-growth 0.05 \
#   --plot
#
# Input schema (CSV; extra columns are preserved)
# -----------------------------------------------
# company,ticker,drug,region,modality,rev_usd_yr,rev_year,patent_expiry,exclusivity_end,expected_loe,erosion_profile,erosion_speed,terminal_share,competitor_count,notes
# ACME Pharma,ACME,DrugA,Global,small_molecule,2500000000,2024-12-31,2026-05-01,,
# ACME Pharma,ACME,DrugB,US,biologic,1200000000,2024-12-31,,2030-01-01,,biosimilar,medium,0.35,3,
# Notes:
# - rev_usd_yr: trailing-12m (or latest FY) revenue for the row (in USD)
# - rev_year: the fiscal date that rev_usd_yr corresponds to (YYYY-MM-DD or YYYY)
# - patent_expiry / exclusivity_end: most relevant jurisdiction for your modeling
# - expected_loe (optional): override LOE date if you have a better estimate
# - modality: one of {small_molecule, biologic} (sets default erosion parameters)
# - erosion_profile (optional): {step, linear, exp, s_curve}; per-product override
# - erosion_speed (optional): {slow, medium, fast} or numeric half-life months for exp/s_curve
# - terminal_share (optional): fraction of pre-LOE revenue that persists long-run (e.g., 0.20)
# - competitor_count (optional): integer; used to auto-tilt erosion speed
#
# Outputs (./artifacts/patent_cliff/*)
# ------------------------------------
# products_clean.csv
# loe_schedule.csv               (product-level LOE, modality, parameters)
# revenue_at_risk_by_year.csv    (company×year rev at risk & residual)
# company_forecasts.csv          (Base/Bear/Bull total revenue projections)
# sensitivity_grid.csv           (sweep over erosion speed × terminal share)
# plots/*.png                    (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from dateutil import parser as dtp


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    products_file: str
    asof: Optional[pd.Timestamp]
    horizon_years: int
    base_growth: float
    bear_growth: float
    bull_growth: float
    base_profile: str
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "patent_cliff_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def _parse_date(x):
    if pd.isna(x) or str(x).strip()=="":
        return pd.NaT
    try:
        d = dtp.parse(str(x), dayfirst=False, yearfirst=True)
        return pd.Timestamp(d.date())
    except Exception:
        try:
            # maybe just a year
            y = int(str(x)[:4])
            return pd.Timestamp(f"{y}-12-31")
        except Exception:
            return pd.NaT

def _num(x):
    if pd.isna(x): return np.nan
    try:
        return float(str(x).replace(",","").replace("_",""))
    except Exception:
        return np.nan

def norm_modality(s: str) -> str:
    s = (s or "").strip().lower()
    if "bio" in s: return "biologic"
    if "small" in s or "chem" in s: return "small_molecule"
    return "unknown"


# ----------------------------- Load & clean -----------------------------

def load_products(path: str, asof: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardize columns
    cols = {c.lower().strip(): c for c in df.columns}
    df.columns = [c.lower().strip() for c in df.columns]

    # required minimum
    need = {"company","drug","rev_usd_yr"}
    if not need.issubset(df.columns):
        raise SystemExit("products file must include at least: company, drug, rev_usd_yr")

    # defaults
    if "ticker" not in df.columns: df["ticker"] = ""
    if "region" not in df.columns: df["region"] = "Global"
    if "modality" not in df.columns: df["modality"] = "unknown"

    # numerics
    for c in ["rev_usd_yr","terminal_share","competitor_count"]:
        if c in df.columns: df[c] = df[c].apply(_num)

    # strings / enums
    df["company"] = df["company"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["drug"] = df["drug"].astype(str).str.strip()
    df["modality"] = df["modality"].apply(norm_modality)

    # dates
    for c in ["rev_year","patent_expiry","exclusivity_end","expected_loe"]:
        if c in df.columns:
            df[c] = df[c].apply(_parse_date)
        else:
            df[c] = pd.NaT

    # LOE
    df["loe_date"] = df["expected_loe"].where(df["expected_loe"].notna(), df[["patent_expiry","exclusivity_end"]].min(axis=1))
    if df["loe_date"].isna().all():
        print("[WARN] No LOE dates found; set expected_loe/patent_expiry/exclusivity_end in your CSV.")
    # reference revenue year
    if "rev_year" not in df.columns or df["rev_year"].isna().all():
        df["rev_year"] = (asof.normalize() if asof is not None else pd.Timestamp.today().normalize())
    # sanity
    df["rev_usd_yr"] = df["rev_usd_yr"].fillna(0.0)
    return df


# ----------------------------- Erosion modeling -----------------------------

def pick_defaults(row) -> Dict[str, float]:
    """
    Default parameters by modality & heuristics
    - generic small molecules: steep early erosion (e.g., 60–80% yr1), terminal share 10–20%
    - biosimilars: more measured (e.g., 25–50% yr1), terminal share 20–40%
    """
    modality = row.get("modality","unknown")
    comp = row.get("competitor_count", np.nan)
    speed = (row.get("erosion_speed") or "").strip().lower() if isinstance(row.get("erosion_speed"), str) else row.get("erosion_speed")

    if modality == "small_molecule":
        base = {"profile":"exp", "half_life_m": 6, "terminal_share": 0.15}
    elif modality == "biologic":
        base = {"profile":"s_curve", "half_life_m": 12, "terminal_share": 0.30}
    else:
        base = {"profile":"exp", "half_life_m": 9, "terminal_share": 0.20}

    # competitor tilt
    try:
        if pd.notna(comp) and comp >= 5: base["half_life_m"] *= 0.8
        if pd.notna(comp) and comp <= 1: base["half_life_m"] *= 1.2
    except Exception:
        pass

    # user overrides
    if isinstance(speed, str):
        if speed in ("fast","f","hi","high"): base["half_life_m"] *= 0.75
        if speed in ("slow","lo","low"): base["half_life_m"] *= 1.25
    else:
        # numeric half-life months
        if pd.notna(speed) and float(speed) > 0:
            base["half_life_m"] = float(speed)

    if pd.notna(row.get("terminal_share")):
        base["terminal_share"] = float(row["terminal_share"])

    if isinstance(row.get("erosion_profile"), str) and row["erosion_profile"]:
        base["profile"] = row["erosion_profile"].strip().lower()

    return base


def erosion_curve(months: np.ndarray, profile: str, half_life_m: float, terminal_share: float) -> np.ndarray:
    """
    Returns the share of pre-LOE revenue that remains at month t (0..N),
    asymptoting to terminal_share. 1.0 at t=0 (just before LOE).
    """
    m = np.maximum(0.0, months.astype(float))
    tau = max(0.1, float(half_life_m))

    if profile == "step":
        # immediate drop to terminal_share
        rem = np.where(m <= 1, 1.0, terminal_share)
    elif profile == "linear":
        # linear to terminal over, say, 24 months
        T = 24.0
        rem = 1.0 - (1.0-terminal_share) * np.minimum(1.0, m/T)
    elif profile == "s_curve":
        # logistic approach from 1→terminal with midpoint at tau
        k = 4.0/tau  # steepness
        rem = terminal_share + (1-terminal_share) / (1 + np.exp(k*(m - tau)))
    else:
        # exponential half-life
        lam = np.log(2.0)/tau
        rem = terminal_share + (1-terminal_share)*np.exp(-lam*m)
    return np.clip(rem, terminal_share, 1.05)


def project_product(row: pd.Series, asof: pd.Timestamp, horizon_years: int,
                    scenario_bump: float = 0.0, speed_mult: float = 1.0, term_bump: float = 0.0
                   ) -> pd.DataFrame:
    """
    Produce monthly projection for a single product, then sum to years.
    scenario_bump: growth pre-LOE (e.g., +/−5% p.a. before LOE)
    speed_mult: multiply half-life ( <1 faster, >1 slower )
    term_bump: add to terminal share (e.g., ±0.05)
    """
    base = pick_defaults(row)
    profile = base["profile"]
    hl = base["half_life_m"] * float(speed_mult)
    term = float(np.clip(base["terminal_share"] + term_bump, 0.0, 1.0))

    start = pd.Timestamp(asof.normalize())
    end = start + pd.DateOffset(years=horizon_years)
    months = pd.period_range(start=start, end=end, freq="M").to_timestamp("M")
    loe = row.get("loe_date", pd.NaT)
    loe = loe if pd.notna(loe) else end + pd.Timedelta(days=1)  # push beyond horizon if unknown

    # monthly revenue baseline = last FY level spread monthly (simple; could season)
    base_rev_m = float(row["rev_usd_yr"]) / 12.0
    series = pd.DataFrame(index=months, data={"rev": base_rev_m})

    # pre-LOE growth (scenario_bump, annualized, applied monthly until LOE)
    pre_growth_m = (1.0 + scenario_bump)**(1/12) - 1.0
    series.loc[series.index < loe, "rev"] = base_rev_m * (1.0 + pre_growth_m) ** np.arange((series.index < loe).sum())

    # post-LOE erosion
    post = series.index >= loe
    if post.any():
        m_since = np.arange(post.sum())
        rem_share = erosion_curve(m_since, profile, hl, term)
        # revenue transitions at LOE: first post-LOE month = month 0 of curve
        series.loc[post, "rev"] = base_rev_m * rem_share

    # yearly roll-up
    yearly = series.resample("Y").sum().rename_axis("year")
    yearly.index = yearly.index.year
    yearly["company"] = row["company"]
    yearly["ticker"] = row.get("ticker","")
    yearly["drug"]   = row["drug"]
    yearly["loe_year"] = int(loe.year) if pd.notna(loe) else np.nan
    yearly["profile"] = profile
    yearly["half_life_m"] = hl
    yearly["terminal_share"] = term
    return yearly.reset_index()


# ----------------------------- Portfolio aggregation -----------------------------

def build_all_products(df: pd.DataFrame, asof: pd.Timestamp, horizon_years: int,
                       scenario_bump: float, speed_mult: float, term_bump: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append(project_product(r, asof, horizon_years, scenario_bump, speed_mult, term_bump))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def revenue_at_risk(products_proj: pd.DataFrame, asof_year: int) -> pd.DataFrame:
    """
    Rev at risk = prior-year revenue − projected revenue (only for years >= LOE year).
    """
    p = products_proj.copy()
    p = p.sort_values(["company","drug","year"])
    p["prev_rev"] = p.groupby(["company","drug"])["rev"].shift(1)
    p["at_risk"] = np.where(p["year"] >= p["loe_year"], (p["prev_rev"] - p["rev"]).clip(lower=0.0), 0.0)
    out = p.groupby(["company","ticker","year"]).agg(
        rev_sum=("rev","sum"),
        rev_at_risk=("at_risk","sum")
    ).reset_index()
    return out


def company_total_forecasts(products_proj: pd.DataFrame,
                            base_growth: float, bear_growth: float, bull_growth: float,
                            asof_year: int) -> pd.DataFrame:
    """
    Combine product projections + ex-portfolio growth overlay:
    - Sum all modeled products per company
    - For simplicity, grow non-modeled revenue bucket at user growth (we infer a base from asof-year if provided)
    """
    p = products_proj.copy()
    # infer modeled base (asof_year) per company
    base_modeled = p[p["year"] == asof_year].groupby(["company","ticker"])["rev"].sum().rename("modeled_asof")
    # Project modeled part directly from p
    mod = p.groupby(["company","ticker","year"])["rev"].sum().rename("modeled_rev").reset_index()

    # Build non-modeled bucket = 0 by default (user can add an extra row "Other" product in CSV if they want a baseline)
    # Here we simply apply growth to asof modeled to demonstrate headroom; for realistic split, add an "Other" product.
    base_df = base_modeled.reset_index()
    frames = []
    for scen_name, g in [("Base",base_growth), ("Bear",bear_growth), ("Bull",bull_growth)]:
        # simple compounding from asof_year
        years = sorted(mod["year"].unique())
        other = []
        for _, row in base_df.iterrows():
            comp, tick, m0 = row["company"], row["ticker"], float(row["modeled_asof"])
            for y in years:
                t = y - asof_year
                if t < 0: continue
                other.append({"company": comp, "ticker": tick, "year": y, "other_rev": m0 * ((1+g)**t)})
        other_df = pd.DataFrame(other)
        merged = mod.merge(other_df, on=["company","ticker","year"], how="left").fillna({"other_rev":0.0})
        merged["scenario"] = scen_name
        # total = modeled (from erosion model) + "other" growth proxy
        merged["total_rev"] = merged["modeled_rev"] + merged["other_rev"]
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


# ----------------------------- Plots -----------------------------

def make_plots(loe_sched: pd.DataFrame, risk: pd.DataFrame, forecasts: pd.DataFrame, outdir: str):
    if plt is None: return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # LOE histogram by year
    if not loe_sched.empty:
        fig1 = plt.figure(figsize=(9,5)); ax1 = plt.gca()
        counts = loe_sched["loe_year"].dropna().astype(int).value_counts().sort_index()
        ax1.bar(counts.index, counts.values)
        ax1.set_title("Count of LOEs by year"); ax1.set_xlabel("Year"); ax1.set_ylabel("Products")
        plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "loe_hist.png"), dpi=140); plt.close(fig1)

    # Revenue at risk (Top companies)
    if not risk.empty:
        top = (risk.groupby("company")["rev_at_risk"].sum().sort_values(ascending=False).head(8)).index
        sub = risk[risk["company"].isin(top)]
        fig2 = plt.figure(figsize=(10,6)); ax2 = plt.gca()
        for comp in sub["company"].unique():
            s = sub[sub["company"]==comp].set_index("year")["rev_at_risk"]
            ax2.plot(s.index, s.values, marker="o", label=comp)
        ax2.set_title("Revenue at risk by year (top companies)")
        ax2.set_ylabel("USD"); ax2.legend()
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "rev_at_risk.png"), dpi=140); plt.close(fig2)

    # Company forecast examples (first 6)
    if not forecasts.empty:
        for comp in forecasts["company"].unique()[:6]:
            s = forecasts[forecasts["company"]==comp]
            fig3 = plt.figure(figsize=(9,5)); ax3 = plt.gca()
            for scen in ["Bear","Base","Bull"]:
                t = s[s["scenario"]==scen].set_index("year")["total_rev"]
                ax3.plot(t.index, t.values, marker="o", label=scen)
            ax3.set_title(f"Total revenue projection — {comp}")
            ax3.set_ylabel("USD"); ax3.legend()
            plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", f"forecast_{comp[:24]}.png"), dpi=140); plt.close(fig3)


# ----------------------------- Sensitivity -----------------------------

def sensitivity_grid(df: pd.DataFrame, asof: pd.Timestamp, horizon_years: int,
                     speed_mults: List[float], term_bumps: List[float]) -> pd.DataFrame:
    rows = []
    for sm in speed_mults:
        for tb in term_bumps:
            proj = build_all_products(df, asof, horizon_years, scenario_bump=0.0, speed_mult=sm, term_bump=tb)
            agg = proj.groupby(["year"])["rev"].sum().rename("portfolio_rev").reset_index()
            rows.append(agg.assign(speed_mult=sm, term_bump=tb))
    return pd.concat(rows, ignore_index=True)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Pharma patent/exclusivity cliff analyzer")
    ap.add_argument("--products", dest="products_file", required=True, help="CSV with product-level revenues & dates")
    ap.add_argument("--asof", type=str, default=None, help="As-of date (YYYY-MM-DD); defaults to today")
    ap.add_argument("--horizon", dest="horizon_years", type=int, default=10, help="Projection horizon in years")
    ap.add_argument("--base-growth", type=float, default=0.03, help="Annual pre-LOE growth for Base scenario (overlay)")
    ap.add_argument("--bear-growth", type=float, default=0.00, help="Annual growth for Bear (lower)")
    ap.add_argument("--bull-growth", type=float, default=0.05, help="Annual growth for Bull (higher)")
    ap.add_argument("--base-profile", type=str, default="exp", help="Fallback erosion profile if not inferable")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        products_file=args.products_file,
        asof=(pd.Timestamp(dtp.parse(args.asof)) if args.asof else pd.Timestamp.today().normalize()),
        horizon_years=int(max(1, args.horizon_years)),
        base_growth=float(args.base_growth),
        bear_growth=float(args.bear_growth),
        bull_growth=float(args.bull_growth),
        base_profile=args.base_profile.strip().lower(),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    # Load & clean
    prod = load_products(cfg.products_file, cfg.asof)
    prod.to_csv(os.path.join(cfg.outdir, "products_clean.csv"), index=False)

    # LOE schedule snapshot
    loe = prod.copy()
    loe["loe_year"] = loe["loe_date"].dt.year
    loe_out = loe[["company","ticker","drug","region","modality","rev_usd_yr","rev_year","loe_date","loe_year",
                   "patent_expiry","exclusivity_end","expected_loe","erosion_profile","erosion_speed","terminal_share","competitor_count"]]
    loe_out.to_csv(os.path.join(cfg.outdir, "loe_schedule.csv"), index=False)

    # Build scenarios
    scenarios = {
        "Bear":  {"pre_growth": cfg.bear_growth, "speed_mult": 0.8, "term_bump": -0.05},
        "Base":  {"pre_growth": cfg.base_growth, "speed_mult": 1.0, "term_bump":  0.00},
        "Bull":  {"pre_growth": cfg.bull_growth, "speed_mult": 1.2, "term_bump":  0.05},
    }

    all_proj = []
    for name, prm in scenarios.items():
        pr = build_all_products(prod, cfg.asof, cfg.horizon_years,
                                scenario_bump=prm["pre_growth"], speed_mult=prm["speed_mult"], term_bump=prm["term_bump"])
        pr["scenario"] = name
        all_proj.append(pr)
    products_proj = pd.concat(all_proj, ignore_index=True)
    products_proj.to_csv(os.path.join(cfg.outdir, "products_projection.csv"), index=False)

    # Revenue at risk (Base scenario)
    base_proj = products_proj[products_proj["scenario"]=="Base"].copy()
    rar = revenue_at_risk(base_proj, cfg.asof.year)
    rar.to_csv(os.path.join(cfg.outdir, "revenue_at_risk_by_year.csv"), index=False)

    # Company total forecasts (modeled + simple overlay)
    forecasts = company_total_forecasts(products_proj[products_proj["scenario"]=="Base"],
                                        cfg.base_growth, cfg.bear_growth, cfg.bull_growth,
                                        cfg.asof.year)
    forecasts.to_csv(os.path.join(cfg.outdir, "company_forecasts.csv"), index=False)

    # Sensitivity grid for the whole portfolio
    sens = sensitivity_grid(prod, cfg.asof, cfg.horizon_years,
                            speed_mults=[0.7, 1.0, 1.3], term_bumps=[-0.05, 0.0, 0.05])
    sens.to_csv(os.path.join(cfg.outdir, "sensitivity_grid.csv"), index=False)

    # Plots
    if cfg.plot:
        make_plots(loe_out, rar, forecasts, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== LOE count by year ===")
    print(loe_out["loe_year"].dropna().astype(int).value_counts().sort_index().to_string())

    print("\n=== Revenue at risk (Base scenario, next 7 years) ===")
    print(rar[rar["year"] <= (cfg.asof.year + 7)].pivot_table(index="year", columns="company",
          values="rev_at_risk", aggfunc="sum").fillna(0).round(0).to_string())

    print("\n=== Example company forecasts (Base/Bear/Bull totals) ===")
    for comp in forecasts["company"].unique()[:5]:
        sub = forecasts[forecasts["company"]==comp]
        snap = sub.pivot_table(index="year", columns="scenario", values="total_rev", aggfunc="sum")
        print(f"\n-- {comp} --")
        print(snap.round(0).to_string())

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()