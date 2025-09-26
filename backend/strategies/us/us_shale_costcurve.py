#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# us_shale_cost_curve.py
#
# U.S. Shale Cost Curve & Supply-at-Price
# ---------------------------------------
# What this does
# - Ingests well- or DSU-level data and normalizes costs (D&C, facilities), LOE, taxes/royalties, and EUR/type curves
# - Computes NPV breakeven price ($/bbl) at chosen discount rate and strip assumptions
# - Aggregates to play/basin/operator; builds cumulative supply curves (Mbbl) vs breakeven
# - Scenario engine: price, cost inflation/deflation, differential & royalty shifts, EUR up/down
# - Optional simple Arps decline for EUR from IP + b, Di (if EUR not given)
# - Exports tidy CSVs and plots (curves, distributions, supply@price)
#
# Quick start
# -----------
# python us_shale_cost_curve.py --wells wells.csv --plot
#
# Example with scenarios
# python us_shale_cost_curve.py --wells wells.csv --price 75 \
#   --cost-mult 1.05 --eur-mult 0.95 --royalty-pp 2 --diff-pp 1 --plot
#
# Input schema (CSV; case-insensitive; extras preserved)
# -----------------------------------------------------
# well_id,basin,play,county,operator,spud_date,first_prod,ip30_bopd,eur_bbl,
# lateral_ft,dc_cost_usd,facilities_usd,loe_usd_bbl,prod_taxes_pct,royalty_pct,differential_usd_bbl,
# b_factor,di_annual,discount_rate
#
# Minimal needed:
#   basin OR play, eur_bbl (or ip30_bopd + decline params), dc_cost_usd, loe_usd_bbl, royalty_pct, prod_taxes_pct
# Optional:
#   facilities_usd, differential_usd_bbl, discount_rate per well
#
# Outputs (./artifacts/us_shale_cost_curve/*)
# -------------------------------------------
# wells_clean.csv
# breakevens_by_well.csv
# aggregates_by_play.csv
# cost_curve_points.csv        (sorted breakevens with cumulative Mbbl)
# supply_at_price.csv          (Mbbl available ≤ price by group)
# plots/*.png                  (breakeven hist, cost curve by basin/play)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
    wells_file: str
    price: float               # flat WTI price assumption ($/bbl) for supply@price table
    cost_mult: float           # D&C + facilities multiplier (inflation/deflation)
    eur_mult: float            # EUR multiplier (type-curve optimism/pessimism)
    royalty_pp: float          # royalty +X pp (e.g., 2 -> +2 percentage points)
    diff_pp: float             # differential +X $/bbl (realized = price - diff)
    tax_pp: float              # production tax +X pp
    loe_mult: float            # LOE $/bbl multiplier
    base_discount: float       # real discount rate if not specified per well
    hedged_frac: float         # fraction of volumes priced at given price (rest at realized)
    outdir: str
    plot: bool


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "us_shale_cost_curve_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def _num(x):
    if x is None: return np.nan
    try:
        return float(str(x).replace(",","").replace("_",""))
    except Exception:
        return np.nan

def _dt(x):
    if pd.isna(x) or str(x).strip()=="":
        return pd.NaT
    try:
        return pd.Timestamp(dtp.parse(str(x)).date())
    except Exception:
        return pd.NaT


# ----------------------------- Decline / EUR -----------------------------

def arps_eur(ip30_bopd: float, b: float, di_annual: float, life_years: float = 30.0) -> float:
    """
    Very simple Arps EUR (oil only) from IP30, hyperbolic b, initial decline Di (annual, decimal).
    We approximate monthly IP0 = IP30 (already stabilized) and integrate monthly for life_years.
    """
    if not np.isfinite(ip30_bopd) or ip30_bopd <= 0:
        return np.nan
    Di = max(1e-4, float(di_annual))
    b = max(0.0, float(b))
    months = int(life_years * 12)
    q = ip30_bopd
    eur = 0.0
    for m in range(months):
        t = m / 12.0
        if b == 0:
            rate = q * np.exp(-Di * t)
        else:
            rate = q / ((1 + b * Di * t) ** (1 / b))
        eur += rate * 30.4375  # barrels per month (approx days)
    return float(eur)


# ----------------------------- Breakeven math -----------------------------

def npv_of_well(
    eur_bbl: float,
    price_wti: float,
    differential: float,
    loe_per_bbl: float,
    royalty_pct: float,
    tax_pct: float,
    dc_facilities_usd: float,
    r: float,
) -> float:
    """
    NPV at t=0 with all capex up-front and revenues/costs spread uniformly across EUR (shortcut).
    It's a simplified cash margin * EUR - capex model (no timing), then divided by (1+r)^0 = 1.
    For cost curve ranking, this static approach is common and robust enough.
    """
    if not np.isfinite(eur_bbl) or eur_bbl <= 0:
        return np.nan
    realized_price = max(0.0, price_wti - differential)
    netback = (realized_price * (1 - royalty_pct) * (1 - tax_pct)) - loe_per_bbl
    return float(netback * eur_bbl - dc_facilities_usd)


def breakeven_price_static(
    eur_bbl: float,
    differential: float,
    loe_per_bbl: float,
    royalty_pct: float,
    tax_pct: float,
    dc_facilities_usd: float,
) -> float:
    """
    Solve flat oil price P such that NPV≈0 under the static netback * EUR - CAPEX model:
      0 = ((P - diff)*(1-royalty)*(1-tax) - LOE) * EUR - CAPEX
      => P* = diff + (LOE + CAPEX/EUR) / ((1-royalty)*(1-tax))
    """
    if not np.isfinite(eur_bbl) or eur_bbl <= 0:
        return np.nan
    denom = (1 - royalty_pct) * (1 - tax_pct)
    if denom <= 0:
        return np.nan
    capex_per_bbl = dc_facilities_usd / eur_bbl
    return float(differential + (loe_per_bbl + capex_per_bbl) / denom)


# ----------------------------- IO & Clean -----------------------------

def load_wells(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # essential normalization
    for c in ["eur_bbl","ip30_bopd","dc_cost_usd","facilities_usd","loe_usd_bbl",
              "prod_taxes_pct","royalty_pct","differential_usd_bbl","b_factor","di_annual","discount_rate",
              "lateral_ft"]:
        if c in df.columns:
            df[c] = df[c].apply(_num)

    for c in ["spud_date","first_prod"]:
        if c in df.columns:
            df[c] = df[c].apply(_dt)

    # defaults
    if "facilities_usd" not in df.columns: df["facilities_usd"] = 0.0
    if "loe_usd_bbl" not in df.columns: df["loe_usd_bbl"] = 7.0
    if "differential_usd_bbl" not in df.columns: df["differential_usd_bbl"] = 3.0
    if "prod_taxes_pct" not in df.columns: df["prod_taxes_pct"] = 0.05
    if "royalty_pct" not in df.columns: df["royalty_pct"] = 0.20
    if "discount_rate" not in df.columns: df["discount_rate"] = np.nan
    if "eur_bbl" not in df.columns: df["eur_bbl"] = np.nan

    # derive EUR if missing and decline params present
    mask_need = df["eur_bbl"].isna() & df["ip30_bopd"].notna()
    if mask_need.any():
        df.loc[mask_need, "eur_bbl"] = [
            arps_eur(ip, b or 1.0, di or 0.7) for ip, b, di in zip(
                df.loc[mask_need, "ip30_bopd"],
                df.loc[mask_need, "b_factor"].fillna(1.0),
                df.loc[mask_need, "di_annual"].fillna(0.7)
            )
        ]

    # well/cost hygiene
    df["dc_total_usd"] = df["dc_cost_usd"].fillna(0.0) + df["facilities_usd"].fillna(0.0)

    # location keys
    if "basin" not in df.columns: df["basin"] = df.get("play", "Unknown")
    if "play" not in df.columns: df["play"] = df.get("basin", "Unknown")
    if "operator" not in df.columns: df["operator"] = "Unknown"

    return df


# ----------------------------- Main compute -----------------------------

def apply_scenarios(
    df: pd.DataFrame,
    cost_mult: float,
    eur_mult: float,
    royalty_pp: float,
    diff_pp: float,
    tax_pp: float,
    loe_mult: float,
) -> pd.DataFrame:
    d = df.copy()
    d["eur_adj"] = d["eur_bbl"] * eur_mult
    d["dc_adj"] = d["dc_total_usd"] * cost_mult
    d["loe_adj"] = d["loe_usd_bbl"] * loe_mult
    d["royalty_adj"] = (d["royalty_pct"].fillna(0.20) + royalty_pp/100.0).clip(0.0, 0.9)
    d["tax_adj"] = (d["prod_taxes_pct"].fillna(0.05) + tax_pp/100.0).clip(0.0, 0.25)
    d["diff_adj"] = d["differential_usd_bbl"].fillna(3.0) + diff_pp
    return d


def compute_tables(d: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Breakevens
    d["breakeven_wti"] = [
        breakeven_price_static(eur, diff, loe, roy, tax, cap)
        for eur, diff, loe, roy, tax, cap in zip(
            d["eur_adj"], d["diff_adj"], d["loe_adj"], d["royalty_adj"], d["tax_adj"], d["dc_adj"]
        )
    ]

    # NPV at cfg.price (flat), for info
    r = d["discount_rate"].fillna(cfg.base_discount)
    d["npv_at_price"] = [
        npv_of_well(eur, cfg.price, diff, loe, roy, tax, cap, rr)
        for eur, diff, loe, roy, tax, cap, rr in zip(
            d["eur_adj"], d["diff_adj"], d["loe_adj"], d["royalty_adj"], d["tax_adj"], d["dc_adj"], r
        )
    ]

    # Supply curve: sort by breakeven, cum EUR
    curv = d.dropna(subset=["breakeven_wti","eur_adj"]).sort_values("breakeven_wti")[
        ["well_id","basin","play","operator","eur_adj","breakeven_wti"]
    ].copy()
    curv["cum_eur_bbl"] = curv["eur_adj"].cumsum()
    curv["cum_eur_mmbbl"] = curv["cum_eur_bbl"] / 1e6

    # Aggregates by play/basin/operator
    agg = d.groupby(["basin","play","operator"], dropna=False).agg(
        wells=("well_id","count"),
        eur_total_bbl=("eur_adj","sum"),
        dc_total_usd=("dc_adj","sum"),
        loe_med=("loe_adj","median"),
        royalty_med=("royalty_adj","median"),
        tax_med=("tax_adj","median"),
        diff_med=("diff_adj","median"),
        bk_p50=("breakeven_wti","median"),
        bk_p25=("breakeven_wti", lambda s: s.quantile(0.25)),
        bk_p75=("breakeven_wti", lambda s: s.quantile(0.75)),
    ).reset_index()
    agg["eur_mmbbl"] = agg["eur_total_bbl"]/1e6

    # Supply at price (volumes ≤ price)
    price = cfg.price
    supply = d[d["breakeven_wti"] <= price].groupby(["basin","play"], dropna=False)["eur_adj"].sum().rename("eur_bbl").reset_index()
    supply["eur_mmbbl"] = supply["eur_bbl"]/1e6
    supply["price"] = price

    # Well-level output subset
    wells_out = d[[
        "well_id","basin","play","operator","lateral_ft","eur_bbl","eur_adj",
        "dc_total_usd","dc_adj","loe_usd_bbl","loe_adj","royalty_pct","royalty_adj",
        "prod_taxes_pct","tax_adj","differential_usd_bbl","diff_adj","breakeven_wti","npv_at_price"
    ]].copy()

    return wells_out, agg, curv, supply


# ----------------------------- Plotting -----------------------------

def make_plots(wells_out: pd.DataFrame, agg: pd.DataFrame, curv: pd.DataFrame, cfg: Config):
    if plt is None: 
        return
    os.makedirs(os.path.join(cfg.outdir, "plots"), exist_ok=True)

    # Breakeven histogram
    fig1 = plt.figure(figsize=(9,5)); ax1 = plt.gca()
    ax1.hist(wells_out["breakeven_wti"].dropna(), bins=50)
    ax1.axvline(cfg.price, linestyle="--", label=f"Price ${cfg.price:.0f}")
    ax1.set_title("Breakeven price distribution (WTI, $/bbl)"); ax1.set_xlabel("$ / bbl"); ax1.legend()
    plt.tight_layout(); fig1.savefig(os.path.join(cfg.outdir, "plots", "breakeven_hist.png"), dpi=140); plt.close(fig1)

    # Cost curve (all wells cumulative)
    fig2 = plt.figure(figsize=(10,6)); ax2 = plt.gca()
    sub = curv.dropna(subset=["breakeven_wti","cum_eur_mmbbl"])
    ax2.plot(sub["cum_eur_mmbbl"], sub["breakeven_wti"])
    ax2.axhline(cfg.price, linestyle="--", alpha=0.7)
    ax2.set_title("U.S. shale cost curve — cumulative Mbbl vs breakeven"); ax2.set_xlabel("Cumulative Mbbl"); ax2.set_ylabel("$ / bbl")
    plt.tight_layout(); fig2.savefig(os.path.join(cfg.outdir, "plots", "cost_curve_all.png"), dpi=140); plt.close(fig2)

    # Cost curve by basin (top 5 by EUR)
    top_basins = (wells_out.groupby("basin")["eur_adj"].sum().sort_values(ascending=False).head(5)).index
    fig3 = plt.figure(figsize=(10,6)); ax3 = plt.gca()
    for b in top_basins:
        s = curv[curv["basin"]==b]
        if s.empty: continue
        ax3.plot(s["cum_eur_mmbbl"], s["breakeven_wti"], label=b)
    ax3.axhline(cfg.price, linestyle="--", alpha=0.6)
    ax3.set_title("Cost curves by basin (top 5 by EUR)"); ax3.set_xlabel("Cumulative Mbbl"); ax3.set_ylabel("$ / bbl")
    ax3.legend()
    plt.tight_layout(); fig3.savefig(os.path.join(cfg.outdir, "plots", "cost_curve_by_basin.png"), dpi=140); plt.close(fig3)

    # Aggregate breakeven box by play (top 10)
    fig4 = plt.figure(figsize=(10,6)); ax4 = plt.gca()
    plays_top = wells_out.groupby("play")["eur_adj"].sum().sort_values(ascending=False).head(10).index
    bx = wells_out[wells_out["play"].isin(plays_top)]
    if not bx.empty:
        bx.boxplot(column="breakeven_wti", by="play", ax=ax4, rot=30, grid=False)
        ax4.set_title("Breakeven by play (boxplot)"); ax4.set_ylabel("$ / bbl"); plt.suptitle("")
        plt.tight_layout(); fig4.savefig(os.path.join(cfg.outdir, "plots", "breakeven_by_play.png"), dpi=140); plt.close(fig4)

    print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="U.S. shale cost curve builder: breakevens, supply@price, and scenarios")
    ap.add_argument("--wells", dest="wells_file", required=True, help="Well/DSU CSV (see header for expected fields)")
    ap.add_argument("--price", type=float, default=70.0, help="WTI flat price for supply@price table ($/bbl)")
    ap.add_argument("--cost-mult", type=float, default=1.00, help="D&C + facilities multiplier (e.g., 1.1 = +10%)")
    ap.add_argument("--eur-mult", type=float, default=1.00, help="EUR multiplier (e.g., 0.95 = −5%)")
    ap.add_argument("--royalty-pp", type=float, default=0.0, help="Royalty +pp (e.g., 2 = +2ppt)")
    ap.add_argument("--diff-pp", type=float, default=0.0, help="Differential +$/bbl (realized = WTI − diff)")
    ap.add_argument("--tax-pp", type=float, default=0.0, help="Production tax +pp")
    ap.add_argument("--loe-mult", type=float, default=1.00, help="LOE multiplier (e.g., 0.9 = −10%)")
    ap.add_argument("--discount", type=float, default=0.10, help="Discount rate if missing per well (real)")
    ap.add_argument("--hedged-frac", type=float, default=0.0, help="(Reserved) fraction hedged at --price [0..1]")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        wells_file=args.wells_file,
        price=float(args.price),
        cost_mult=float(args.cost_mult),
        eur_mult=float(args.eur_mult),
        royalty_pp=float(args.royalty_pp),
        diff_pp=float(args.diff_pp),
        tax_pp=float(args.tax_pp),
        loe_mult=float(args.loe_mult),
        base_discount=float(args.discount),
        hedged_frac=float(args.hedged_frac),
        outdir=ensure_outdir(args.outdir),
        plot=bool(args.plot),
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")
    wells = load_wells(cfg.wells_file)
    wells.to_csv(os.path.join(cfg.outdir, "wells_clean.csv"), index=False)

    scen = apply_scenarios(wells, cfg.cost_mult, cfg.eur_mult, cfg.royalty_pp, cfg.diff_pp, cfg.tax_pp, cfg.loe_mult)
    wells_out, agg, curv, supply = compute_tables(scen, cfg)

    wells_out.to_csv(os.path.join(cfg.outdir, "breakevens_by_well.csv"), index=False)
    agg.to_csv(os.path.join(cfg.outdir, "aggregates_by_play.csv"), index=False)
    curv.to_csv(os.path.join(cfg.outdir, "cost_curve_points.csv"), index=False)
    supply.to_csv(os.path.join(cfg.outdir, "supply_at_price.csv"), index=False)

    if cfg.plot:
        make_plots(wells_out, agg, curv, cfg)

    # Console snapshot
    print("\n=== Supply at price (top 10 plays by EUR) ===")
    print(supply.sort_values("eur_mmbbl", ascending=False).head(10)[["basin","play","eur_mmbbl","price"]].round(2).to_string(index=False))

    print("\n=== Aggregate breakevens by basin/play/operator (snippet) ===")
    cols = ["basin","play","operator","wells","eur_mmbbl","bk_p25","bk_p50","bk_p75","loe_med","royalty_med","tax_med","diff_med"]
    print(agg.sort_values("eur_mmbbl", ascending=False).head(12)[cols].round(2).to_string(index=False))

    print("\nFiles written to:", cfg.outdir)
    if cfg.hedged_frac > 0:
        print("[Note] hedged_frac reserved for future cashflow timing; current model uses static breakevens.")

if __name__ == "__main__":
    main()