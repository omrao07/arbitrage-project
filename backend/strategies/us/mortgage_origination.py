#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mortgage_origination.py
#
# Mortgage Origination & Pipeline Analytics
# ----------------------------------------
# What this does
# - Ingest a loan-level CSV (locks, funding, pricing, fees, status, simple performance flags)
# - Builds KPIs for the **pipeline** (apps → locks → funded) and **production** (by channel/product/vintage)
# - Computes pull-through & fallout, cycle times, WA note rate / price, gain-on-sale economics
# - “As-of” pipeline snapshot with stage aging buckets
# - Cohort prepayment (simple SMM/CPR) & early payment default (EPD) flags
# - Repurchase / buyback summary
# - Optional plots
#
# Inputs (CSV)
# ------------
# Required columns (names can vary slightly—see RENAMES below):
#   application_date, lock_date, close_date or funded_date, status
#   channel, product, purpose, fico, ltv, dti, loan_amount
#   note_rate (or rate_note), points_bps, lender_credit_bps, price_premium_bps
#   orig_fee_bps, srp_bps, servicing_strip_bps, upfront_cost_usd
# Optional (good to have):
#   property_state, occupancy, term_months, rate_type, execution, best_eff_price_bps, mandatory_price_bps
#   tba_coupon, tba_price, hedge_pnl_usd, coverage_dv01
#   first_payment_date, payoff_date, current_upb, delinquency_status, epd_flag, repurchase_flag, buyback_date
#
# Usage
# -----
# python mortgage_origination.py --loans loans.csv --asof 2025-09-05 --plot
#
# Outputs (./artifacts/mortgage_origination/*)
# -------------------------------------------
#   inputs_clean.csv
#   kpis_summary.csv
#   pipeline_snapshot.csv
#   cycle_times.csv
#   profitability_by_segment.csv
#   production_mix.csv
#   cohort_prepayment.csv
#   repurchase_summary.csv
#   plots/*.png (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib python-dateutil
#
# Notes
# -----
# - All basis point fields interpreted as *bps* (e.g., 75 = 0.75%).
# - Gain-on-sale (per-loan, rough): price_premium + SRP + orig_fee − lender_credit − points − upfront_cost/UPB
#   Minus optional hedge PnL allocation if provided (per-loan via proportional UPB share).
# - Pull-through = funded / locks by cohort (lock month default). Fallout = 1 − pull-through.

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

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
    loans_file: str
    asof: Optional[str]
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

RENAMES = {
    "rate_note": "note_rate",
    "funded_date": "close_date",
    "amount": "loan_amount",
    "loan_amt": "loan_amount",
    "fico_score": "fico",
    "ltv_pct": "ltv",
    "dti_pct": "dti",
}

DATE_COLS = [
    "application_date", "lock_date", "close_date", "first_payment_date", "payoff_date", "buyback_date"
]

BPS_COLS = [
    "points_bps", "lender_credit_bps", "price_premium_bps", "orig_fee_bps",
    "srp_bps", "servicing_strip_bps", "best_eff_price_bps", "mandatory_price_bps"
]

NUM_COLS = [
    "loan_amount", "fico", "ltv", "dti", "note_rate", "upfront_cost_usd", "current_upb",
    "hedge_pnl_usd", "coverage_dv01"
] + BPS_COLS


def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "mortgage_origination_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def _parse_date(x):
    if pd.isna(x) or x == "":
        return pd.NaT
    try:
        return pd.Timestamp(dtp.parse(str(x)).date())
    except Exception:
        return pd.NaT


def read_loans(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    for k, v in RENAMES.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # parse dates
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = df[c].apply(_parse_date)

    # numeric coercion
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # strings
    for c in ["status", "channel", "product", "purpose", "property_state", "occupancy", "rate_type", "execution"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()

    # basic hygiene
    if "loan_amount" not in df.columns:
        raise SystemExit("Missing required column: loan_amount (or amount/loan_amt)")
    if "note_rate" not in df.columns:
        raise SystemExit("Missing required column: note_rate (or rate_note)")
    if "application_date" not in df.columns:
        raise SystemExit("Missing required column: application_date")
    if "lock_date" not in df.columns:
        print("[WARN] lock_date missing. Deriving pipeline on application_date only.")
        df["lock_date"] = pd.NaT
    if "close_date" not in df.columns:
        print("[WARN] close_date/funded_date missing. Funded production KPIs will be limited.")
        df["close_date"] = pd.NaT

    # default missing fee columns to 0
    for c in BPS_COLS + ["upfront_cost_usd", "hedge_pnl_usd", "coverage_dv01"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # status normalization
    if "status" in df.columns:
        df["status"] = df["status"].str.upper().replace({
            "APPROVED": "LOCKED",
            "FUNDED": "FUNDED",
            "CLOSED": "FUNDED",
            "WITHDRAWN": "WITHDRAWN",
            "DENIED": "DENIED",
            "EXPIRED": "EXPIRED",
        })
    else:
        df["status"] = np.where(df["close_date"].notna(), "FUNDED",
                         np.where(df["lock_date"].notna(), "LOCKED", "APPLICATION"))

    # cohorts
    df["app_month"] = df["application_date"].dt.to_period("M").astype(str)
    df["lock_month"] = df["lock_date"].dt.to_period("M").astype(str)
    df["fund_month"] = df["close_date"].dt.to_period("M").astype(str)

    return df


def bps_to_dec(x):
    return (x / 10000.0) if pd.notna(x) else np.nan


def compute_gain_on_sale(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Convert bps fields to decimals
    for c in BPS_COLS:
        if c in d.columns:
            d[c + "_dec"] = bps_to_dec(d[c])
        else:
            d[c + "_dec"] = 0.0

    # Components
    premium = d["price_premium_bps_dec"] if "price_premium_bps_dec" in d.columns else 0.0
    srp = d["srp_bps_dec"] if "srp_bps_dec" in d.columns else 0.0
    orig_fee = d["orig_fee_bps_dec"] if "orig_fee_bps_dec" in d.columns else 0.0
    lender_credit = d["lender_credit_bps_dec"] if "lender_credit_bps_dec" in d.columns else 0.0
    points = d["points_bps_dec"] if "points_bps_dec" in d.columns else 0.0

    upb = d["loan_amount"].astype(float)
    per_loan_hedge = 0.0
    if "hedge_pnl_usd" in d.columns and d["hedge_pnl_usd"].abs().sum() != 0 and upb.sum() != 0:
        # allocate hedge PnL by UPB share (negative PnL reduces gain)
        per_loan_hedge = d["hedge_pnl_usd"] / upb.replace(0, np.nan)
        per_loan_hedge = per_loan_hedge.fillna(0.0)

    upfront = (d["upfront_cost_usd"] / upb.replace(0, np.nan)).fillna(0.0)

    # GoS in decimal price terms
    d["gain_on_sale_px"] = (premium + srp + orig_fee - lender_credit - points) - upfront - per_loan_hedge

    # Also in $ per loan
    d["gain_on_sale_usd"] = d["gain_on_sale_px"] * upb

    # Channel/product aggregates will use this
    return d


def cycle_times(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["app_to_lock_days"] = (d["lock_date"] - d["application_date"]).dt.days
    d["lock_to_fund_days"] = (d["close_date"] - d["lock_date"]).dt.days
    d["app_to_fund_days"] = (d["close_date"] - d["application_date"]).dt.days
    return d[["app_to_lock_days","lock_to_fund_days","app_to_fund_days"]]


def pullthrough_fallout(df: pd.DataFrame, cohort: str = "lock_month") -> pd.DataFrame:
    d = df.copy()
    # Consider only loans with a lock date for pull-through
    locked = d[d["lock_date"].notna()]
    funded = d[d["close_date"].notna()]
    grp = locked.groupby(cohort).agg(locks_upb=("loan_amount","sum"), locks_cnt=("loan_amount","count")).reset_index()
    fund = funded.groupby(cohort).agg(fund_upb=("loan_amount","sum"), fund_cnt=("loan_amount","count")).reset_index()
    out = grp.merge(fund, on=cohort, how="left").fillna(0)
    out["pull_through_upb"] = out["fund_upb"] / out["locks_upb"].replace(0, np.nan)
    out["pull_through_cnt"] = out["fund_cnt"] / out["locks_cnt"].replace(0, np.nan)
    out["fallout_upb"] = 1.0 - out["pull_through_upb"]
    out["fallout_cnt"] = 1.0 - out["pull_through_cnt"]
    out.rename(columns={cohort: "cohort"}, inplace=True)
    return out


def pipeline_snapshot(df: pd.DataFrame, asof: Optional[pd.Timestamp]) -> pd.DataFrame:
    d = df.copy()
    if asof is None:
        asof = pd.Timestamp.today().normalize()
    # Stage inference
    d["stage"] = np.where(d["close_date"].notna() & (d["close_date"] <= asof), "Funded",
                  np.where(d["lock_date"].notna() & (d["lock_date"] <= asof), "Locked",
                  np.where(d["application_date"].notna() & (d["application_date"] <= asof), "Application", "Future")))

    snap = d[d["application_date"] <= asof]
    # Aging since stage start
    snap["stage_start_date"] = np.where(snap["stage"]=="Funded", snap["close_date"],
                                 np.where(snap["stage"]=="Locked", snap["lock_date"], snap["application_date"]))
    snap["aging_days"] = (asof - pd.to_datetime(snap["stage_start_date"])).dt.days

    # Buckets
    bins = [-1, 7, 15, 30, 45, 60, 90, 99999]
    labels = ["0-7","8-15","16-30","31-45","46-60","61-90","90+"]
    snap["aging_bucket"] = pd.cut(snap["aging_days"], bins=bins, labels=labels)

    piv = snap.pivot_table(index=["stage","aging_bucket"],
                           values=["loan_amount"], aggfunc=["count","sum"]).reset_index()
    piv.columns = ["stage","aging_bucket","loans_cnt","upb_sum"]
    return piv.sort_values(["stage","aging_bucket"])


def production_mix(df: pd.DataFrame) -> pd.DataFrame:
    funded = df[df["close_date"].notna()].copy()
    if funded.empty:
        return pd.DataFrame()
    keys = [k for k in ["fund_month","channel","product","purpose","rate_type"] if k in funded.columns]
    mix = funded.groupby(keys).agg(
        loans_cnt=("loan_amount","count"),
        upb_sum=("loan_amount","sum"),
        wa_rate=("note_rate", lambda x: np.average(x, weights=funded.loc[x.index,"loan_amount"]))
    ).reset_index()
    return mix.sort_values(["fund_month","channel","product"])


def profitability_by_segment(df_gos: pd.DataFrame) -> pd.DataFrame:
    funded = df_gos[df_gos["close_date"].notna()].copy()
    if funded.empty:
        return pd.DataFrame()
    keys = [k for k in ["fund_month","channel","product","purpose","rate_type","execution"] if k in funded.columns]
    prof = funded.groupby(keys).agg(
        loans_cnt=("loan_amount","count"),
        upb_sum=("loan_amount","sum"),
        wa_rate=("note_rate", lambda x: np.average(x, weights=funded.loc[x.index,"loan_amount"])),
        gos_px=("gain_on_sale_px", lambda x: np.average(x, weights=funded.loc[x.index,"loan_amount"])),
        gos_usd=("gain_on_sale_usd","sum")
    ).reset_index()
    return prof.sort_values(["fund_month","channel","product"])


def cohort_prepayment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very rough cohort SMM/CPR using fund_month vintages:
    - For each vintage, track payoffs within next N months (observed).
    - SMM ≈ monthly payoff UPB / (beginning UPB − scheduled principal); we ignore scheduled here, so it's a rough proxy.
    """
    funded = df[df["close_date"].notna()].copy()
    if funded.empty or "payoff_date" not in funded.columns:
        return pd.DataFrame()

    funded["vintage"] = funded["fund_month"]
    # expand into monthly grid from first_payment_date (fallback fund date) to payoff_date or as-of
    rows = []
    for _, r in funded.iterrows():
        upb0 = r["loan_amount"]
        fp = r["first_payment_date"] if pd.notna(r.get("first_payment_date", pd.NaT)) else r["close_date"]
        end = r["payoff_date"] if pd.notna(r.get("payoff_date", pd.NaT)) else pd.NaT
        if pd.isna(fp):
            continue
        # count first 12 months for EPD as well
        for m in range(1, 25):  # cap at 24 months window for reporting
            month = (fp + pd.offsets.MonthEnd(m)).to_period("M").strftime("%Y-%m")
            payoff = 1.0 if (pd.notna(end) and (end.to_period("M").strftime("%Y-%m") == month)) else 0.0
            rows.append({"vintage": r["vintage"], "month": month, "upb": upb0, "payoff_flag": payoff})
    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame()

    grp = panel.groupby(["vintage","month"]).agg(
        upb_sum=("upb","sum"),
        payoffs=("payoff_flag","sum")
    ).reset_index()
    # SMM proxy: payoff UPB / beg UPB (use count-based proxy if UPB alignment is too rough)
    # Assume average UPB per loan within vintage is stable; scale payoffs by avg UPB
    vint_upb = panel.groupby("vintage")["upb"].sum().to_dict()
    vint_avg = panel.groupby("vintage")["upb"].mean().to_dict()
    grp["smm_proxy"] = (grp["payoffs"] * pd.Series([vint_avg[v] for v in grp["vintage"]])) / pd.Series([vint_upb[v] for v in grp["vintage"]])
    grp["cpr_proxy"] = 1 - (1 - grp["smm_proxy"])**12
    return grp


def repurchase_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "repurchase_flag" not in d.columns and "epd_flag" not in d.columns:
        return pd.DataFrame()
    d["repurchase_flag"] = d.get("repurchase_flag", 0).fillna(0).astype(int)
    d["epd_flag"] = d.get("epd_flag", 0).fillna(0).astype(int)
    keys = [k for k in ["fund_month","channel","product","purpose"] if k in d.columns]
    rep = d.groupby(keys).agg(
        loans_cnt=("loan_amount","count"),
        upb_sum=("loan_amount","sum"),
        repurchases=("repurchase_flag","sum"),
        epd=("epd_flag","sum")
    ).reset_index()
    rep["repurchase_rate_cnt"] = rep["repurchases"] / rep["loans_cnt"].replace(0, np.nan)
    rep["epd_rate_cnt"] = rep["epd"] / rep["loans_cnt"].replace(0, np.nan)
    return rep


# ----------------------------- Plotting -----------------------------

def make_plots(kpis: pd.DataFrame, pull: pd.DataFrame, prof: pd.DataFrame, outdir: str):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Pull-through over time
    if not pull.empty:
        fig1 = plt.figure(figsize=(9,5)); ax1 = plt.gca()
        x = pd.to_datetime(pull["cohort"] + "-01")
        ax1.plot(x, 100*pull["pull_through_upb"], marker="o", label="UPB")
        ax1.plot(x, 100*pull["pull_through_cnt"], marker="s", label="Count", alpha=0.7)
        ax1.set_title("Pull-through (locks → funded)"); ax1.set_ylabel("%"); ax1.legend()
        plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "pull_through.png"), dpi=140); plt.close(fig1)

    # Gain-on-sale by month (WA)
    if not prof.empty and "gos_px" in prof.columns:
        gos = (prof.groupby("fund_month").apply(lambda g: np.average(g["gos_px"], weights=g["upb_sum"])).reset_index(name="gos_px"))
        fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
        ax2.plot(pd.to_datetime(gos["fund_month"]+"-01"), 10000*gos["gos_px"], marker="o")
        ax2.set_title("Gain-on-sale (WA, bps)"); ax2.set_ylabel("bps")
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "gos_bps.png"), dpi=140); plt.close(fig2)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Mortgage origination & pipeline analytics")
    ap.add_argument("--loans", required=True, help="Loan-level CSV")
    ap.add_argument("--asof", type=str, default=None, help="As-of date for pipeline snapshot (YYYY-MM-DD)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        loans_file=args.loans,
        asof=(pd.Timestamp(dtp.parse(args.asof)) if args.asof else None),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir),
    )
    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    df0 = read_loans(cfg.loans_file)
    df0.to_csv(os.path.join(cfg.outdir, "inputs_clean.csv"), index=False)

    # Gain-on-sale & cycle times
    df_gos = compute_gain_on_sale(df0)
    ct = cycle_times(df_gos)
    ct.to_csv(os.path.join(cfg.outdir, "cycle_times.csv"), index=False)

    # KPIs summary (simple headliners)
    total_apps = df_gos["loan_amount"].where(df_gos["application_date"].notna()).sum()
    total_locks = df_gos["loan_amount"].where(df_gos["lock_date"].notna()).sum()
    total_funded = df_gos["loan_amount"].where(df_gos["close_date"].notna()).sum()
    pt = pullthrough_fallout(df_gos, cohort="lock_month")
    pt_latest = pt.tail(1)

    kpis = {
        "apps_upb": float(total_apps or 0),
        "locks_upb": float(total_locks or 0),
        "funded_upb": float(total_funded or 0),
        "pull_through_upb_latest": float(pt_latest["pull_through_upb"].iloc[0]) if not pt_latest.empty else np.nan,
        "avg_app_to_fund_days": float(ct["app_to_fund_days"].replace([np.inf,-np.inf], np.nan).dropna().mean()) if not ct.empty else np.nan,
        "avg_note_rate_funded": float(np.average(df_gos.loc[df_gos["close_date"].notna(), "note_rate"],
                                                weights=df_gos.loc[df_gos["close_date"].notna(),"loan_amount"])) if df_gos["close_date"].notna().any() else np.nan,
        "wa_gos_bps_funded": float(10000*np.average(df_gos.loc[df_gos["close_date"].notna(),"gain_on_sale_px"],
                                                    weights=df_gos.loc[df_gos["close_date"].notna(),"loan_amount"])) if df_gos["close_date"].notna().any() else np.nan
    }
    pd.DataFrame([kpis]).to_csv(os.path.join(cfg.outdir, "kpis_summary.csv"), index=False)

    # Pipeline snapshot as-of
    snap = pipeline_snapshot(df_gos, cfg.asof)
    snap.to_csv(os.path.join(cfg.outdir, "pipeline_snapshot.csv"), index=False)

    # Production mix (funded)
    mix = production_mix(df_gos)
    if not mix.empty:
        mix.to_csv(os.path.join(cfg.outdir, "production_mix.csv"), index=False)

    # Profitability by segment
    prof = profitability_by_segment(df_gos)
    if not prof.empty:
        prof.to_csv(os.path.join(cfg.outdir, "profitability_by_segment.csv"), index=False)

    # Cohort prepayment (rough)
    prepay = cohort_prepayment(df_gos)
    if not prepay.empty:
        prepay.to_csv(os.path.join(cfg.outdir, "cohort_prepayment.csv"), index=False)

    # Repurchase / EPD summary
    rep = repurchase_summary(df_gos)
    if not rep.empty:
        rep.to_csv(os.path.join(cfg.outdir, "repurchase_summary.csv"), index=False)

    # Plots
    if cfg.plot:
        make_plots(pd.DataFrame([kpis]), pt, prof, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== KPIs (headline) ===")
    print(pd.read_csv(os.path.join(cfg.outdir, "kpis_summary.csv")).round(2).to_string(index=False))

    if not pt.empty:
        print("\n=== Pull-through by lock cohort (last 6) ===")
        print(pt.tail(6).assign(
            pull_through_upb=lambda x: (100*x["pull_through_upb"]).round(1),
            pull_through_cnt=lambda x: (100*x["pull_through_cnt"]).round(1)
        )[["cohort","locks_cnt","fund_cnt","pull_through_upb","pull_through_cnt"]].to_string(index=False))

    if not prof.empty:
        print("\n=== Profitability by segment (latest 6 rows) ===")
        tmp = prof.copy()
        tmp["gos_bps"] = (10000*tmp["gos_px"]).round(1)
        print(tmp[["fund_month","channel","product","purpose","rate_type","execution","gos_bps","gos_usd","upb_sum","loans_cnt"]]
              .tail(6).to_string(index=False))

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()