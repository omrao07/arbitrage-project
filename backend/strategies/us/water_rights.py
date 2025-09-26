#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# water_rights.py
#
# Western Water Rights — portfolio, curtailment risk & valuation
# --------------------------------------------------------------
# What this does
# - Ingests: rights (priority/seniority, volumes, type), hydrology (flows/SWE/reservoir),
#   optional demand/value curves by sector, and optional market price history.
# - Computes: seniority ladder, reliability curves, curtailment thresholds vs flow percentiles,
#   Monte Carlo shortage paths, expected yield (AF) & cash flows (lease/sale).
# - Allocates water to demands each year to maximize value (greedy merit-order) subject to
#   physical/administrative constraints.
# - Values portfolio (NPV) under Base / Dry / Wet scenarios.
# - Exports tidy CSVs and optional plots.
#
# Quick start
# ----------
# python water_rights.py --rights rights.csv --hydro hydro.csv --plot
#
# Example with demand & prices
# python water_rights.py --rights rights.csv --hydro hydro.csv --demand demand.csv \
#   --prices prices.csv --years 30 --mc 5000 --discount 0.06 --plot
#
# Expected CSV schemas (case-insensitive; extra cols preserved)
# -------------------------------------------------------------
# rights.csv
#   right_id,owner,basin,river,source,priority_date,right_type,admin_number,
#   max_volume_af, max_rate_cfs, reliability_hint, lease_price_usd_af, sale_price_usd_af, notes
#   - right_type ∈ {surface, groundwater, storage, instream, exchange, contract}
#   - reliability_hint (0..1) optional (e.g., historic avg diversion / decreed)
#
# hydro.csv  (annual or monthly; we aggregate to water-year)
#   date,basin,river,flow_cfs,flow_af, snow_swe_in, reservoir_af, drought_index
#
# demand.csv (optional; value curves per sector in $/AF)
#   sector,basin,river,priority, value_usd_af, max_need_af
#   - Higher 'priority' consumes first in allocation (e.g., Muni > Ag > Indust > Env).
#
# prices.csv (optional; spot/lease market series)
#   date,basin,river,price_usd_af
#
# Outputs (./artifacts/water_rights/*)
# -----------------------------------
# rights_clean.csv
# hydro_wateryears.csv
# seniority_ladder.csv
# reliability_curves.csv
# mc_paths_summary.csv
# allocation_by_year.csv
# portfolio_valuation.csv
# plots/*.png (if --plot)
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
    rights_file: str
    hydro_file: str
    demand_file: Optional[str]
    prices_file: Optional[str]
    years: int
    mc_paths: int
    discount: float
    outdir: str
    plot: bool


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "water_rights_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def _num(x):
    if pd.isna(x): return np.nan
    try:
        return float(str(x).replace(",","").replace("_","").replace("$",""))
    except Exception:
        return np.nan

def parse_date(x):
    if pd.isna(x) or str(x).strip()=="":
        return pd.NaT
    try:
        return pd.Timestamp(dtp.parse(str(x)).date())
    except Exception:
        return pd.NaT


# ----------------------------- Loaders -----------------------------

def load_rights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Ensure required columns
    req = {"right_id","basin","river","priority_date"}
    if not req.issubset(df.columns):
        raise SystemExit(f"rights.csv must contain: {sorted(list(req))}")
    # Coerce
    df["priority_date"] = df["priority_date"].apply(parse_date)
    for c in ["admin_number","max_volume_af","max_rate_cfs","reliability_hint","lease_price_usd_af","sale_price_usd_af"]:
        if c in df.columns: df[c] = df[c].apply(_num)
    df["right_type"] = df.get("right_type","surface").astype(str).str.lower()
    df["owner"] = df.get("owner","").astype(str)
    # Seniority score: earlier date => higher seniority (smaller score)
    df["seniority_score"] = df["priority_date"].rank(method="min").astype(int)
    # If admin_number provided, prefer it
    if "admin_number" in df.columns and df["admin_number"].notna().any():
        # Normalize admin numbers so lower = more senior
        df["seniority_score"] = df["admin_number"].rank(method="min").astype(int)
    # Default prices if missing (fallback; can be overwritten by prices.csv)
    df["lease_price_usd_af"] = df.get("lease_price_usd_af", np.nan).fillna(250.0)
    df["sale_price_usd_af"]  = df.get("sale_price_usd_af", np.nan).fillna(5000.0)
    # Reliability prior
    df["reliability_hint"] = df.get("reliability_hint", np.nan)
    return df.sort_values(["basin","river","seniority_score"])


def load_hydro(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise SystemExit("hydro.csv must include 'date'")
    df["date"] = df["date"].apply(parse_date)
    for c in ["flow_cfs","flow_af","snow_swe_in","reservoir_af","drought_index"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    # Build water year (Oct–Sep commonly; adjust if needed)
    df["wy"] = (df["date"] + pd.offsets.MonthBegin(-3)).dt.year  # shifts Oct-Dec into next WY
    # Use AF if provided; else integrate CFS → AF-month; then sum to WY
    if "flow_af" not in df.columns or df["flow_af"].isna().all():
        if "flow_cfs" not in df.columns:
            raise SystemExit("Need either flow_af or flow_cfs in hydro.csv.")
        # Rough monthly AF from CFS: cfs * days_in_month * 1.9835 (AF/day per cfs)
        grp = df.groupby([df["date"].dt.to_period("M"), "basin","river"])["flow_cfs"].mean().reset_index()
        grp["date"] = grp["date"].dt.to_timestamp("M")
        days = grp["date"].dt.days_in_month
        grp["flow_af"] = grp["flow_cfs"] * days * 1.9835
        df = df.drop(columns=["flow_af"], errors="ignore").merge(
            grp[["date","basin","river","flow_af"]], on=["date","basin","river"], how="left"
        )
        df["wy"] = (df["date"] + pd.offsets.MonthBegin(-3)).dt.year
    # Aggregate to water year per basin/river
    wy = df.groupby(["wy","basin","river"], as_index=False).agg(
        flow_af=("flow_af","sum"),
        snow_swe_in=("snow_swe_in","mean"),
        reservoir_af=("reservoir_af","mean"),
        drought_index=("drought_index","mean")
    )
    # Percentiles by basin/river using historical WY totals
    wy["p_flow"] = wy.groupby(["basin","river"])["flow_af"].rank(pct=True)
    return wy.sort_values(["basin","river","wy"])


def load_demand(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["value_usd_af","max_need_af"]:
        if c in df.columns: df[c] = df[c].apply(_num)
    if "priority" not in df.columns: df["priority"] = 2
    df["sector"] = df.get("sector","Ag").astype(str).str.title()
    return df.sort_values(["priority","value_usd_af"], ascending=[True, False])


def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = df["date"].apply(parse_date)
        # yearly median by basin/river
        df["wy"] = (df["date"] + pd.offsets.MonthBegin(-3)).dt.year
        px = df.groupby(["wy","basin","river"], as_index=False)["price_usd_af"].median()
        return px
    return pd.DataFrame()


# ----------------------------- Core logic -----------------------------

def seniority_ladder(rights: pd.DataFrame) -> pd.DataFrame:
    """
    Return ladder per basin/river with cumulative decreed volume and implied cutoff
    at various hydrology percentiles.
    """
    ladders = []
    for (b, r), g in rights.groupby(["basin","river"]):
        g = g.sort_values("seniority_score")
        g["cum_decreed_af"] = g["max_volume_af"].fillna(0).cumsum()
        g["rank"] = np.arange(1, len(g)+1)
        ladders.append(g)
    return pd.concat(ladders, ignore_index=True)


def reliability_from_history(ladder: pd.DataFrame, wy: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reliability curve for each right as share of years where available flow
    covered cumulative volume up to that right (very simplified, basin × river bucket).
    """
    out = []
    for (b, r), g in ladder.groupby(["basin","river"]):
        hist = wy[(wy["basin"]==b) & (wy["river"]==r)]
        if hist.empty: continue
        avail = hist["flow_af"].values
        for _, row in g.iterrows():
            decreed = float(row.get("max_volume_af", 0) or 0)
            cum_need = float(row.get("cum_decreed_af", 0) or 0)
            # Reliability = P(flow >= cum_need) (ignores return flows/storage/admin)
            rel = np.mean(avail >= cum_need) if cum_need>0 else 1.0
            out.append({
                "right_id": row["right_id"], "basin": b, "river": r,
                "reliability_hist": rel, "cum_decreed_af": cum_need, "decreed_af": decreed
            })
    rel_df = pd.DataFrame(out)
    return ladder.merge(rel_df, on=["right_id","basin","river","cum_decreed_af"], how="left")


def draw_hydro_paths(wy: pd.DataFrame, years: int, npaths: int, rng: np.random.Generator) -> Dict[Tuple[str,str], np.ndarray]:
    """
    Monte Carlo generator: resample historical WY flow_AF with replacement (iid bootstrap).
    Returns dict {(basin,river): matrix [years × npaths]}.
    """
    mats: Dict[Tuple[str,str], np.ndarray] = {}
    for (b, r), g in wy.groupby(["basin","river"]):
        hist = g["flow_af"].dropna().values
        if len(hist) == 0: continue
        idx = rng.integers(0, len(hist), size=(years, npaths))
        mats[(b,r)] = hist[idx]
    return mats


def allocate_value(year_flow_by_br: Dict[Tuple[str,str], float],
                   rights: pd.DataFrame,
                   demand: pd.DataFrame,
                   prices_wy: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Allocate available flow to rights in seniority order; then allocate delivered AF to sector demands
    by value priority to estimate $ value created that year.
    Very simplified conveyance/no-return-flow representation.
    """
    rows = []
    # Pre-index demands
    demand_keyed = {}
    if not demand.empty:
        for (b, r), g in demand.groupby(["basin","river"]):
            demand_keyed[(b,r)] = g.sort_values(["priority","value_usd_af"], ascending=[True, False]).copy()

    for (b, r), available in year_flow_by_br.items():
        # Price signal (lease market) if provided
        px = np.nan
        if not prices_wy.empty:
            m = prices_wy[(prices_wy["basin"]==b) & (prices_wy["river"]==r)]
            if not m.empty:
                px = float(m["price_usd_af"].median())
        # Seniority subset
        g = rights[(rights["basin"]==b) & (rights["river"]==r)].sort_values("seniority_score").copy()
        remaining = float(available)
        # Deliveries to each right (capped by decreed)
        g["delivered_af"] = 0.0
        for i, row in g.iterrows():
            if remaining <= 0: break
            decreed = float(row.get("max_volume_af", 0) or 0)
            take = float(min(decreed, remaining))
            g.at[i, "delivered_af"] = take
            remaining -= take
        # Demand allocation (value creation)
        total_value = 0.0
        if (b,r) in demand_keyed:
            needtbl = demand_keyed[(b,r)].copy()
            pool = g["delivered_af"].sum()
            for j, drow in needtbl.iterrows():
                if pool <= 0: break
                need = float(drow.get("max_need_af", 0) or 0)
                take = min(need, pool)
                total_value += take * float(drow.get("value_usd_af", 0) or 0)
                pool -= take
        else:
            # If no demand table, proxy value via lease_price on delivered AF
            lease_price = g["lease_price_usd_af"].median() if "lease_price_usd_af" in g.columns else (px if not np.isnan(px) else 0.0)
            total_value = g["delivered_af"].sum() * float(lease_price or 0.0)
        # Append rows
        for _, rr in g.iterrows():
            rows.append({
                "basin": b, "river": r, "right_id": rr["right_id"],
                "decreed_af": rr["max_volume_af"], "delivered_af": rr["delivered_af"],
                "lease_price_usd_af": rr.get("lease_price_usd_af", px),
                "year_value_usd": rr["delivered_af"] * float(rr.get("lease_price_usd_af", px) or 0.0)
            })
    alloc = pd.DataFrame(rows)
    return alloc, float(alloc["year_value_usd"].sum()) if not alloc.empty else 0.0


def portfolio_npv(paths: Dict[Tuple[str,str], np.ndarray],
                  rights: pd.DataFrame,
                  demand: pd.DataFrame,
                  prices_wy: pd.DataFrame,
                  discount: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each Monte Carlo path, allocate annually & compute present value; return per-right stats
    and summary by path.
    """
    rng = np.random.default_rng(42)  # deterministic for repeatability
    years = list(range(paths[list(paths.keys())[0]].shape[0])) if paths else []
    per_right_accrual = {}  # (right_id) -> [pv across paths]
    path_summary = []
    for pidx in range(paths[list(paths.keys())[0]].shape[1] if paths else 0):
        pv = 0.0
        right_cash = {}
        for t in years:
            # Build available flow this year per basin/river
            year_flow_by_br = {br: float(paths[br][t, pidx]) for br in paths}
            alloc, tot_val = allocate_value(year_flow_by_br, rights, demand, prices_wy)
            # discount to t=0
            pv += tot_val / ((1+discount)**t)
            # accumulate by right
            for _, row in alloc.iterrows():
                rid = row["right_id"]
                cf = row["year_value_usd"] / ((1+discount)**t)
                right_cash[rid] = right_cash.get(rid, 0.0) + cf
        # record
        path_summary.append({"path": pidx, "pv_usd": pv})
        # stash to per_right
        for rid, cf in right_cash.items():
            per_right_accrual.setdefault(rid, []).append(cf)

    # Summaries
    if per_right_accrual:
        stats = []
        for rid, arr in per_right_accrual.items():
            arr = np.array(arr, dtype=float)
            stats.append({
                "right_id": rid,
                "pv_mean": float(np.nanmean(arr)),
                "pv_p10": float(np.nanpercentile(arr, 10)),
                "pv_p50": float(np.nanpercentile(arr, 50)),
                "pv_p90": float(np.nanpercentile(arr, 90)),
            })
        per_right = pd.DataFrame(stats)
    else:
        per_right = pd.DataFrame(columns=["right_id","pv_mean","pv_p10","pv_p50","pv_p90"])

    paths_df = pd.DataFrame(path_summary)
    return per_right, paths_df


# ----------------------------- Scenario builder -----------------------------

def make_scenarios(wy: pd.DataFrame, years: int, mc_paths: int, seed: int = 123) -> Dict[str, Dict[Tuple[str,str], np.ndarray]]:
    """
    Build flow matrices for Base (bootstrap), Dry (bias −20%), Wet (+20%).
    """
    rng = np.random.default_rng(seed)
    base = draw_hydro_paths(wy, years, mc_paths, rng)
    # Apply multiplicative biases for Dry/Wet
    dry = {k: v * 0.8 for k, v in base.items()}
    wet = {k: v * 1.2 for k, v in base.items()}
    return {"Base": base, "Dry": dry, "Wet": wet}


# ----------------------------- Plotting -----------------------------

def make_plots(wy: pd.DataFrame, ladder: pd.DataFrame, rel: pd.DataFrame, alloc: pd.DataFrame,
               per_right_val: pd.DataFrame, paths_df: pd.DataFrame, outdir: str):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Hydro distribution by basin/river
    fig1 = plt.figure(figsize=(9,5)); ax1 = plt.gca()
    for (b,r), g in wy.groupby(["basin","river"]):
        ax1.hist(g["flow_af"], bins=20, alpha=0.4, label=f"{b}-{r}")
    ax1.set_title("Water-year flow distribution (AF)"); ax1.set_xlabel("AF")
    if wy["basin"].nunique()*wy["river"].nunique() <= 6: ax1.legend()
    plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "wy_flow_hist.png"), dpi=140); plt.close(fig1)

    # Seniority vs cumulative decreed
    fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
    for (b,r), g in ladder.groupby(["basin","river"]):
        ax2.step(g["cum_decreed_af"], g["seniority_score"], where="post", label=f"{b}-{r}")
    ax2.set_title("Seniority ladder (score ↓ = more senior)"); ax2.set_xlabel("Cumulative decreed AF"); ax2.set_ylabel("Seniority score")
    plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "seniority_ladder.png"), dpi=140); plt.close(fig2)

    # Reliability curve (hist-based)
    if "reliability_hist" in rel.columns:
        fig3 = plt.figure(figsize=(9,5)); ax3 = plt.gca()
        ax3.scatter(rel["cum_decreed_af"], 100*rel["reliability_hist"], s=12)
        ax3.set_title("Reliability vs cumulative decreed (hist)"); ax3.set_xlabel("Cumulative decreed AF"); ax3.set_ylabel("Reliability %")
        plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "reliability_curve.png"), dpi=140); plt.close(fig3)

    # Allocation snapshot (latest run)
    if not alloc.empty:
        fig4 = plt.figure(figsize=(9,5)); ax4 = plt.gca()
        alloc.groupby("basin")["delivered_af"].sum().plot(kind="bar", ax=ax4)
        ax4.set_title("Delivered AF by basin (one-year snapshot)")
        plt.tight_layout(); fig4.savefig(os.path.join(outdir, "plots", "delivered_by_basin.png"), dpi=140); plt.close(fig4)

    # Per-right PV distribution (boxplot)
    if not per_right_val.empty:
        fig5 = plt.figure(figsize=(9,5)); ax5 = plt.gca()
        per_right_val[["pv_p10","pv_p50","pv_p90"]].plot(kind="box", ax=ax5)
        ax5.set_title("Per-right PV distribution (MC percentiles)")
        plt.tight_layout(); fig5.savefig(os.path.join(outdir, "plots", "per_right_pv_box.png"), dpi=140); plt.close(fig5)

    # Path PV histogram
    if not paths_df.empty:
        fig6 = plt.figure(figsize=(9,5)); ax6 = plt.gca()
        ax6.hist(paths_df["pv_usd"]/1e6, bins=30)
        ax6.set_title("Portfolio PV across MC paths"); ax6.set_xlabel("PV (USD millions)")
        plt.tight_layout(); fig6.savefig(os.path.join(outdir, "plots", "portfolio_pv_hist.png"), dpi=140); plt.close(fig6)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Water rights: seniority, curtailment risk & valuation toolkit")
    ap.add_argument("--rights", dest="rights_file", required=True, help="rights.csv (see header for schema)")
    ap.add_argument("--hydro", dest="hydro_file", required=True, help="hydro.csv (daily/monthly/annual hydrology)")
    ap.add_argument("--demand", dest="demand_file", default=None, help="Optional demand/value curve CSV")
    ap.add_argument("--prices", dest="prices_file", default=None, help="Optional price history CSV")
    ap.add_argument("--years", type=int, default=20, help="Projection horizon (water years)")
    ap.add_argument("--mc", dest="mc_paths", type=int, default=3000, help="Monte Carlo paths")
    ap.add_argument("--discount", type=float, default=0.07, help="Discount rate (real)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        rights_file=args.rights_file,
        hydro_file=args.hydro_file,
        demand_file=args.demand_file,
        prices_file=args.prices_file,
        years=int(max(5, args.years)),
        mc_paths=int(max(100, args.mc_paths)),
        discount=float(args.discount),
        outdir=ensure_outdir(args.outdir),
        plot=bool(args.plot),
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    # Load
    rights = load_rights(cfg.rights_file)
    hydro_wy = load_hydro(cfg.hydro_file)
    demand = load_demand(cfg.demand_file)
    prices = load_prices(cfg.prices_file)

    rights.to_csv(os.path.join(cfg.outdir, "rights_clean.csv"), index=False)
    hydro_wy.to_csv(os.path.join(cfg.outdir, "hydro_wateryears.csv"), index=False)

    # Build ladder & reliability
    ladder = seniority_ladder(rights)
    rel = reliability_from_history(ladder, hydro_wy)
    rel.to_csv(os.path.join(cfg.outdir, "reliability_curves.csv"), index=False)

    # One-year illustrative allocation using latest median flows by basin/river
    median_flow = hydro_wy.groupby(["basin","river"])["flow_af"].median()
    year_flow_by_br = {k: float(v) for k, v in median_flow.items()}
    alloc_snap, val_snap = allocate_value(year_flow_by_br, rights, demand, prices)
    alloc_snap.to_csv(os.path.join(cfg.outdir, "allocation_by_year.csv"), index=False)

    # Monte Carlo scenarios
    scenarios = make_scenarios(hydro_wy, cfg.years, cfg.mc_paths)
    portfolio_rows = []
    per_right_tables = []
    for name, mats in scenarios.items():
        per_right_val, paths_df = portfolio_npv(mats, rights, demand, prices, cfg.discount)
        if not per_right_val.empty:
            per_right_val["scenario"] = name
            per_right_tables.append(per_right_val)
        if not paths_df.empty:
            paths_df["scenario"] = name
            # Portfolio stats
            portfolio_rows.append({
                "scenario": name,
                "pv_mean": float(paths_df["pv_usd"].mean()),
                "pv_p10": float(np.percentile(paths_df["pv_usd"], 10)),
                "pv_p50": float(np.percentile(paths_df["pv_usd"], 50)),
                "pv_p90": float(np.percentile(paths_df["pv_usd"], 90)),
            })
        # Save per-scenario path distributions
        if not paths_df.empty:
            paths_df.to_csv(os.path.join(cfg.outdir, f"mc_paths_summary_{name}.csv"), index=False)

    per_right_all = pd.concat(per_right_tables, ignore_index=True) if per_right_tables else pd.DataFrame()
    if not per_right_all.empty:
        per_right_all.to_csv(os.path.join(cfg.outdir, "per_right_valuation.csv"), index=False)

    portfolio_tbl = pd.DataFrame(portfolio_rows)
    portfolio_tbl.to_csv(os.path.join(cfg.outdir, "portfolio_valuation.csv"), index=False)

    # Plots
    if cfg.plot:
        # Use Base scenario for snapshot figures
        base_paths = scenarios["Base"]
        # Build a quick one-path allocation for illustration
        if base_paths:
            # take first path, first year
            any_br = next(iter(base_paths))
            one_year_flows = {br: float(base_paths[br][0,0]) for br in base_paths}
            alloc_one, _ = allocate_value(one_year_flows, rights, demand, prices)
        else:
            alloc_one = pd.DataFrame()
        make_plots(hydro_wy, ladder, rel, alloc_one, per_right_all, pd.concat(
            [pd.read_csv(os.path.join(cfg.outdir, f)) for f in os.listdir(cfg.outdir) if f.startswith("mc_paths_summary_")],
            ignore_index=True
        ) if any(fn.startswith("mc_paths_summary_") for fn in os.listdir(cfg.outdir)) else pd.DataFrame(), cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== Portfolio PV by scenario (USD, millions) ===")
    if not portfolio_tbl.empty:
        pt = portfolio_tbl.copy()
        for c in ["pv_mean","pv_p10","pv_p50","pv_p90"]:
            pt[c] = pt[c]/1e6
        print(pt.round(2).to_string(index=False))
    else:
        print("N/A")

    print("\n=== Top 10 rights by PV (Base) ===")
    if not per_right_all.empty:
        base = per_right_all[per_right_all["scenario"]=="Base"].sort_values("pv_mean", ascending=False).head(10)
        print(base[["right_id","pv_mean","pv_p10","pv_p50","pv_p90"]].div(1e6).round(2).rename(columns=lambda c: c+" (USDm)" if c.startswith("pv_") else c).to_string(index=False))
    else:
        print("N/A")

    print("\nFiles written to:", cfg.outdir)
    print("\n[Note] Model is intentionally simplified (no return flows, carriage losses, call administration, "
          "augmentation/augmentation credits, or reservoir carryover). Use as directional risk/valuation tool and "
          "layer jurisdiction-specific admin rules if needed.")

if __name__ == "__main__":
    main()