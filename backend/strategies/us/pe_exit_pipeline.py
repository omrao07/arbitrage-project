#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pe_exit_pipeline.py
#
# Private Equity Exit Pipeline & Distributions Forecaster
# -------------------------------------------------------
# What this does
# - Ingests a deal-level CSV: company, fund, entry date/EV/equity, KPIs, status, target exit type/EV/timing
# - Scores **exit readiness** (0–100) using hold period, growth, leverage & market window proxies
# - Builds a **12–36 month exit pipeline** by type (M&A / IPO / Recap / Secondary) with probability-weighted EV
# - Converts EV→equity proceeds using net debt and ownership; allocates to funds; estimates **distributions** by month
# - Computes projected **MOIC, DPI, TVPI** at fund level under Base / Bear / Bull scenarios
# - Optional quick **market window** gauges from broad tickers (e.g., SPY, VIXY, IPO ETF); or skip if offline
# - Exports tidy CSVs and optional PNG plots
#
# Usage
# -----
# python pe_exit_pipeline.py \
#   --deals deals.csv \
#   --asof 2025-09-06 \
#   --horizon-m 24 \
#   --scenario Base \
#   --plot
#
# CSV schema (columns; extras preserved)
# --------------------------------------
# fund,company,sector,region,status,entry_date,entry_ev,entry_equity,ownership_pct,net_debt,ebitda_ltm,rev_ltm,ebitda_cagr_3y,rev_cagr_3y,leverage_turns,hold_target_m, \
# target_exit_type,target_exit_date,target_exit_ev,alt_exit_type,alt_exit_ev,alt_exit_date, \
# ipo_ready,strategic_interest,buyer_count,notes
# - status ∈ {Active, Exited, WrittenOff, Hold}
# - dates in YYYY-MM-DD; *_ev in USD; ownership_pct in decimals (e.g., 0.72)
# - Leverage_turns = Net Debt / EBITDA (numeric). ipo_ready/strategic_interest in {0..1} if provided.
#
# Outputs (./artifacts/pe_exit_pipeline/*)
# ---------------------------------------
# deals_clean.csv
# readiness_scores.csv
# exit_pipeline_monthly.csv        (probability-weighted EV & equity proceeds by month × exit type)
# fund_distributions.csv           (monthly distributions $ and cumulative DPI)
# fund_kpis_summary.csv            (current & projected MOIC/DPI/TVPI by fund & scenario)
# plots/*.png                      (if --plot)
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
    deals_file: str
    asof: pd.Timestamp
    horizon_m: int
    scenario: str
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "pe_exit_pipeline_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def _parse_date(x):
    if pd.isna(x) or str(x).strip()=="":
        return pd.NaT
    try:
        return pd.Timestamp(dtp.parse(str(x)).date())
    except Exception:
        return pd.NaT

def _num(x):
    if pd.isna(x): return np.nan
    try:
        return float(str(x).replace(",","").replace("_",""))
    except Exception:
        return np.nan


# ----------------------------- Load & clean -----------------------------

REQUIRED = {"fund","company","entry_date","entry_ev","entry_equity","ownership_pct"}

def load_deals(path: str, asof: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(list(missing))}")

    # Coerce
    for c in ["entry_ev","entry_equity","ownership_pct","net_debt","ebitda_ltm","rev_ltm",
              "ebitda_cagr_3y","rev_cagr_3y","leverage_turns","target_exit_ev","alt_exit_ev",
              "ipo_ready","strategic_interest","buyer_count","hold_target_m"]:
        if c in df.columns: df[c] = df[c].apply(_num)

    for c in ["entry_date","target_exit_date","alt_exit_date"]:
        if c in df.columns: df[c] = df[c].apply(_parse_date)

    # Defaults & hygiene
    if "status" not in df.columns: df["status"] = "Active"
    if "net_debt" not in df.columns: df["net_debt"] = np.where(df["ebitda_ltm"].notna() & df["leverage_turns"].notna(),
                                                               df["ebitda_ltm"]*df["leverage_turns"], np.nan)
    df["fund"] = df["fund"].astype(str).str.strip()
    df["company"] = df["company"].astype(str).str.strip()
    df["sector"] = df.get("sector","").astype(str).str.title()
    df["region"] = df.get("region","").astype(str).str.upper()

    # Derived
    df["hold_months"] = ((asof - df["entry_date"]).dt.days / 30.44).round(1)
    df["net_debt"] = df["net_debt"].fillna(0.0)
    # Current implied EV/Equity if a target is provided and post-money signal is missing
    df["target_exit_type"] = df.get("target_exit_type","").astype(str).str.title().replace({"": "M&A"})
    df["alt_exit_type"] = df.get("alt_exit_type","").astype(str).str.title()
    # Sanity on ownership [0..1]
    df["ownership_pct"] = df["ownership_pct"].clip(lower=0.0, upper=1.0)
    return df


# ----------------------------- Readiness model -----------------------------

def readiness_score(row: pd.Series, asof: pd.Timestamp) -> float:
    """
    0..100 score combining:
      + Hold months vs target
      + Growth (EBITDA CAGR)
      + Leverage (penalize high debt)
      + Market flags: ipo_ready / strategic_interest / buyer_count
    """
    hm = float(row.get("hold_months", 0) or 0)
    ht = float(row.get("hold_target_m", 36) or 36)
    hold_factor = np.clip(hm / max(6.0, ht), 0.0, 1.5)  # >1 if over target
    growth = float(row.get("ebitda_cagr_3y", 0) or 0)   # e.g., 0.12 = 12%
    growth_score = np.clip((growth + 0.05) / 0.25, 0.0, 1.2)  # -5%→0; +20%→1
    lev = float(row.get("leverage_turns", 0) or 0)
    lev_score = np.clip(1.0 - (lev-3)/4.0, 0.0, 1.0)    # 3x→1; 7x→0
    ipo = float(row.get("ipo_ready", 0) or 0)
    strat = float(row.get("strategic_interest", 0) or 0)
    buyers = float(row.get("buyer_count", 1) or 1)
    buyside = np.clip( (0.4*strat + 0.2*ipo + 0.4*np.tanh((buyers-1)/3.0)), 0.0, 1.2 )

    base = 100.0 * (0.35*hold_factor + 0.30*growth_score + 0.20*lev_score + 0.15*buyside) / (0.35+0.30+0.20+0.15)
    # Clamp
    return float(np.clip(base, 0.0, 100.0))


def score_all(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    d = df.copy()
    d["readiness"] = d.apply(lambda r: readiness_score(r, asof), axis=1)
    # Exit probabilities for next 12 months by type (heuristic)
    # Base: prob_next12 = sigmoid(readiness); allocation by declared target type
    z = (d["readiness"] - 55.0) / 12.0
    d["p_exit_12m"] = 1.0 / (1.0 + np.exp(-z))
    # Type weights
    def type_split(r):
        tgt = (r.get("target_exit_type") or "M&A").title()
        alt = (r.get("alt_exit_type") or "").title()
        if tgt == "Ipo": tgt = "IPO"
        if alt == "Ipo": alt = "IPO"
        w = {"M&A":0.6, "IPO":0.25, "Recap":0.1, "Secondary":0.05}
        if tgt in w: w[tgt] += 0.25
        if alt in w: w[alt] += 0.10
        # normalize
        s = sum(w.values()); return {k: v/s for k,v in w.items()}
    splits = d.apply(type_split, axis=1)
    d["w_ma"] = [s["M&A"] for s in splits]
    d["w_ipo"] = [s["IPO"] for s in splits]
    d["w_recap"] = [s["Recap"] for s in splits]
    d["w_secondary"] = [s["Secondary"] for s in splits]
    return d


# ----------------------------- Scenario EVs & timing -----------------------------

SCEN_MULT = {
    # factor applied to declared target_exit_ev if present (or entry_ev as proxy)
    "Bear": 0.85,
    "Base": 1.00,
    "Bull": 1.15,
}

def scenario_ev(row: pd.Series, scenario: str) -> float:
    base = row.get("target_exit_ev", np.nan)
    if pd.isna(base) or base <= 0:
        # fallback: apply a simple EV growth from entry using EBITDA CAGR if available and hold period
        ev0 = float(row.get("entry_ev", 0) or 0)
        cagr = float(row.get("ebitda_cagr_3y", 0) or 0)
        months = float(row.get("hold_months", 0) or 0)
        years = max(0.0, months/12.0)
        base = ev0 * ((1 + cagr) ** years)
    return float(base * SCEN_MULT.get(scenario, 1.0))


def expected_exit_months_dist(row: pd.Series, asof: pd.Timestamp) -> Dict[pd.Timestamp, float]:
    """
    Spreads the 12m exit probability across months with a front-loaded shape near target_exit_date.
    Returns {month_end: probability in that month}.
    """
    p12 = float(row.get("p_exit_12m", 0) or 0)
    if p12 <= 0: return {}
    # month grid
    months = pd.period_range(asof, asof + pd.DateOffset(months=12), freq="M").to_timestamp("M")
    # center around target_exit_date if present
    center = row.get("target_exit_date", pd.NaT)
    if pd.isna(center):
        # center at 9 months from now
        center = (asof + pd.DateOffset(months=9))
    # Gaussian-like weights
    mu = pd.Timestamp(center).to_period("M").to_timestamp("M")
    sigma = pd.Timedelta(days=30*3).days  # ~3m std dev
    xs = np.array([(m - mu).days for m in months], dtype=float)
    w = np.exp(-0.5*(xs/sigma)**2)
    w = w / w.sum()
    probs = (p12 * w)
    return {m: float(p) for m, p in zip(months, probs)}


# ----------------------------- Pipeline construction -----------------------------

def build_pipeline(df: pd.DataFrame, asof: pd.Timestamp, scenario: str, horizon_m: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - pipeline_monthly: probability-weighted exit EV & equity proceeds by month × type
      - readiness table per deal
    """
    d = score_all(df, asof)

    # probability mass per month for next 12 months; after that we push residual evenly to remaining horizon
    months = pd.period_range(asof, asof + pd.DateOffset(months=horizon_m), freq="M").to_timestamp("M")
    rows = []

    for _, r in d.iterrows():
        ev_s = scenario_ev(r, scenario)
        # Equity value at exit for sponsor
        net_debt = float(r.get("net_debt", 0) or 0)
        equity = max(0.0, ev_s - net_debt) * float(r.get("ownership_pct", 0) or 0)
        # 12m distribution
        d12 = expected_exit_months_dist(r, asof)
        # residual probability within horizon
        p12 = sum(d12.values())
        premass = {m: d12.get(m, 0.0) for m in months[:13]}  # first 13 month-ends inc. current
        residual_p = max(0.0, 1.0 - p12) * 0.35  # assume 35% chance within months 13..horizon
        tail_n = max(0, len(months) - len(premass))
        tail_mass = np.repeat(residual_p/(tail_n or 1), tail_n) if tail_n else np.array([])
        prob_by_m = list(premass.values()) + tail_mass.tolist()
        prob_by_m = prob_by_m[:len(months)]
        # type splits
        w = {"M&A": r["w_ma"], "IPO": r["w_ipo"], "Recap": r["w_recap"], "Secondary": r["w_secondary"]}

        for m, p in zip(months, prob_by_m):
            if p <= 0: continue
            for tname, wt in w.items():
                rows.append({
                    "fund": r["fund"], "company": r["company"], "sector": r.get("sector",""),
                    "month": m, "exit_type": tname,
                    "prob": p*wt,
                    "pw_ev": ev_s * p * wt,
                    "pw_equity": equity * p * wt
                })

    pipeline = pd.DataFrame(rows)
    if pipeline.empty:
        return pd.DataFrame(), d[["fund","company","readiness","p_exit_12m"]]

    # aggregate by month × type × fund
    agg = pipeline.groupby(["fund","month","exit_type"]).agg(
        prob=("prob","sum"),
        pw_ev=("pw_ev","sum"),
        pw_equity=("pw_equity","sum")
    ).reset_index()

    # readiness table
    ready = d[["fund","company","sector","hold_months","ebitda_cagr_3y","leverage_turns","readiness","p_exit_12m"]].copy()
    return agg, ready


# ----------------------------- Distributions & Fund KPIs -----------------------------

def distributions_from_pipeline(pipeline: pd.DataFrame) -> pd.DataFrame:
    """
    Converts probability-weighted equity into a monthly distributions schedule by fund.
    """
    if pipeline.empty: return pd.DataFrame()
    dist = pipeline.groupby(["fund","month"]).agg(
        dist_pw_usd=("pw_equity","sum")
    ).reset_index()
    # cumulative & DPI proxy (needs paid-in capital; we infer from entry_equity if provided by fund)
    return dist


def fund_capital_from_deals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate paid-in capital (PIC) by fund from entry_equity (simple, ignores fees/recycling).
    """
    cap = df.groupby("fund")["entry_equity"].sum().rename("pic_est").reset_index()
    return cap


def fund_kpis(pipeline: pd.DataFrame, df_deals: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Compute current MOIC/DPI/TVPI and projected DPI path using pipeline PW distributions.
    """
    cap = fund_capital_from_deals(df_deals)
    dist = distributions_from_pipeline(pipeline)
    # Current realized proceeds if any 'Exited' rows with realized equity; otherwise 0
    realized = df_deals[df_deals["status"].str.upper().eq("EXITED")].copy()
    realized_amt = realized.groupby("fund")["entry_equity"].sum().rename("realized_proxy")  # proxy if actual proceeds unknown

    # Aggregate monthly distributions forward
    if dist.empty:
        df = cap.merge(realized_amt, on="fund", how="left").fillna({"realized_proxy":0.0})
        df["dpi_now"] = df["realized_proxy"] / df["pic_est"].replace(0,np.nan)
        df["tvpi_now"] = np.nan
        return df

    # Build monthly DPI
    dist = dist.merge(cap, on="fund", how="left")
    dist = dist.sort_values(["fund","month"])
    dist["cum_dist"] = dist.groupby("fund")["dist_pw_usd"].cumsum()
    dist["dpi"] = dist["cum_dist"] / dist["pic_est"].replace(0,np.nan)

    # Snapshot KPIs table at last projected month per fund
    snap = dist.groupby("fund").tail(1)[["fund","cum_dist","dpi"]].rename(columns={"cum_dist":"cum_dist_proj","dpi":"dpi_proj"})
    # Simple TVPI proxy = (cum_dist + NAV_est)/PIC; here we approximate NAV_est as remaining unrealized at entry cost
    pic = cap.set_index("fund")["pic_est"]
    unreal = df_deals.groupby("fund")["entry_equity"].sum() - dist.groupby("fund")["dist_pw_usd"].sum()
    tvpi = (dist.groupby("fund")["dist_pw_usd"].sum() + np.maximum(unreal, 0)) / pic
    tvpi = tvpi.replace([np.inf,-np.inf], np.nan)

    out = cap.merge(snap, on="fund", how="left")
    out["tvpi_proj"] = tvpi.values
    # Current “now” DPI if we treat zero as-of distributions
    out["dpi_now"] = 0.0
    return out


# ----------------------------- Plotting -----------------------------

def make_plots(pipeline: pd.DataFrame, ready: pd.DataFrame, kpis: pd.DataFrame, outdir: str):
    if plt is None: 
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Exit mix over horizon
    if not pipeline.empty:
        fig1 = plt.figure(figsize=(10,6)); ax1 = plt.gca()
        mix = pipeline.groupby(["month","exit_type"])["pw_ev"].sum().unstack().fillna(0.0)
        mix.plot(ax=ax1)
        ax1.set_title("Probability-weighted Exit EV by Type")
        ax1.set_ylabel("USD"); plt.tight_layout()
        fig1.savefig(os.path.join(outdir, "plots", "exit_mix.png"), dpi=140); plt.close(fig1)

        # Distributions curve (aggregate)
        fig2 = plt.figure(figsize=(10,5)); ax2 = plt.gca()
        dist = pipeline.groupby("month")["pw_equity"].sum().cumsum()
        ax2.plot(dist.index, dist.values, marker="o")
        ax2.set_title("Projected Cumulative Distributions (PW)"); ax2.set_ylabel("USD")
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "cum_distributions.png"), dpi=140); plt.close(fig2)

    # Readiness scatter
    if not ready.empty:
        fig3 = plt.figure(figsize=(8,5)); ax3 = plt.gca()
        ax3.scatter(ready["hold_months"], ready["readiness"], alpha=0.8)
        ax3.set_title("Exit Readiness vs Hold Months"); ax3.set_xlabel("Hold months"); ax3.set_ylabel("Readiness (0–100)")
        plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "readiness_scatter.png"), dpi=140); plt.close(fig3)

    # Fund DPI snapshot
    if not kpis.empty and "dpi_proj" in kpis.columns:
        fig4 = plt.figure(figsize=(8,5)); ax4 = plt.gca()
        kpis.sort_values("dpi_proj").plot(x="fund", y="dpi_proj", kind="bar", ax=ax4, legend=False)
        ax4.set_title("Projected DPI by Fund"); ax4.set_ylabel("DPI"); ax4.set_xlabel("")
        plt.tight_layout(); fig4.savefig(os.path.join(outdir, "plots", "dpi_by_fund.png"), dpi=140); plt.close(fig4)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="PE exit pipeline & fund distributions forecaster")
    ap.add_argument("--deals", dest="deals_file", required=True, help="Deal-level CSV (see header for schema)")
    ap.add_argument("--asof", type=str, default=None, help="As-of date (YYYY-MM-DD); default today")
    ap.add_argument("--horizon-m", type=int, default=24, help="Projection horizon in months (12–36 typical)")
    ap.add_argument("--scenario", choices=["Bear","Base","Bull"], default="Base", help="EV scenario multiplier")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        deals_file=args.deals_file,
        asof=(pd.Timestamp(dtp.parse(args.asof)) if args.asof else pd.Timestamp.today().normalize()),
        horizon_m=int(max(6, args.horizon_m)),
        scenario=args.scenario,
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir),
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    deals = load_deals(cfg.deals_file, cfg.asof)
    deals.to_csv(os.path.join(cfg.outdir, "deals_clean.csv"), index=False)

    pipeline, ready = build_pipeline(deals, cfg.asof, cfg.scenario, cfg.horizon_m)
    if not pipeline.empty:
        pipeline.to_csv(os.path.join(cfg.outdir, "exit_pipeline_monthly.csv"), index=False)
    ready.to_csv(os.path.join(cfg.outdir, "readiness_scores.csv"), index=False)

    kpis = fund_kpis(pipeline, deals, cfg.asof)
    if not kpis.empty:
        kpis.to_csv(os.path.join(cfg.outdir, "fund_kpis_summary.csv"), index=False)

    if cfg.plot:
        make_plots(pipeline, ready, kpis, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== Readiness (top 10) ===")
    cols = ["fund","company","sector","hold_months","ebitda_cagr_3y","leverage_turns","readiness","p_exit_12m"]
    print(ready.sort_values("readiness", ascending=False).head(10)[cols].round(3).to_string(index=False))

    if not pipeline.empty:
        print("\n=== Next 12 months (PW equity by exit type) ===")
        soon = pipeline[pipeline["month"] <= (cfg.asof + pd.DateOffset(months=12)).to_period("M").to_timestamp("M")]
        snap = soon.groupby("exit_type")["pw_equity"].sum().sort_values(ascending=False)
        print((snap/1e6).round(2).rename("USDm").to_string())

        print("\n=== Fund distributions (first 12 months, PW) ===")
        dist = pipeline.groupby(["fund","month"])["pw_equity"].sum().reset_index()
        dist12 = dist[dist["month"] <= (cfg.asof + pd.DateOffset(months=12)).to_period("M").to_timestamp("M")]
        for f in dist12["fund"].unique():
            s = dist12[dist12["fund"]==f].set_index("month")["pw_equity"].cumsum()
            print(f"\n-- {f} --")
            print((s/1e6).round(2).rename("CumDist USDm").to_string())

    if not kpis.empty:
        print("\n=== Fund KPI snapshot ===")
        print(kpis.round(3).to_string(index=False))

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()