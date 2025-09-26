#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# streaming_subscribers_wars.py
#
# Streaming Subscriber Wars — market share, net adds, LTV & scenarios
# ------------------------------------------------------------------
# What this does
# - Ingests a quarterly panel for streaming platforms (subs, revenue, ARPU, churn, CAC, content spend)
# - Cleans + aligns series, computes:
#     • Net adds, market share, share of net adds
#     • ARPU mix, take-rate/gross margin (if cost given), unit economics
#     • Simple LTV (quarterly cohort-style) & CAC payback
#     • QoQ / YoY growth of subs & revenue
# - Scenario engine:
#     • Price change → ARPU lift with a churn elasticity assumption
#     • Crackdown / bundle / sports add-on → churn ↓ or ARPU ↑ for chosen quarters
# - Lightweight forecast (next 8 quarters): ARPU and churn trend persistence + optional scenario shocks
# - Exports tidy CSVs and optional plots (share, net adds, LTV, forecast)
#
# Usage
# -----
# python streaming_subscribers_wars.py \
#   --panel subs.csv \
#   --scenarios scenarios.csv \
#   --forecast-q 8 \
#   --plot
#
# Input schemas
# -------------
# 1) subs.csv  (wide or long accepted; case-insensitive)
#   date,platform,subs,rev,arpu,cac,churn_q,gross_margin,content_spend,marketing_spend
#   - date: quarter end (YYYY-MM-DD) or YYYYQx
#   - subs: paying subscribers (millions or absolute; set --subs-in-millions if millions)
#   - rev: revenue for the platform (same quarter)
#   - arpu: revenue per user per month (optional; inferred as rev / (subs*3) if missing)
#   - cac: customer acquisition cost per gross add (optional)
#   - churn_q: quarterly churn rate (decimal; e.g., 0.06)
#   - gross_margin: decimal (0.4 = 40%). If missing, inferred as (rev - content_spend - marketing_spend)/rev if possible.
#
# 2) scenarios.csv (optional; rows of shocks)
#   platform,date,kind,value,notes
#   - kind ∈ {price_hike_pct, churn_delta_pp, arpu_delta_pct, bundle_uplift_pct}
#   - value: e.g., 10 (for +10%), or -2 (for −2 percentage points churn)
#
# Outputs (./artifacts/streaming_subscribers_wars/*)
# --------------------------------------------------
# cleaned_panel.csv
# metrics_quarterly.csv
# ltv_unit_econ.csv
# share_table.csv
# forecast.csv
# plots/*.png   (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib python-dateutil

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
    panel_file: str
    scenarios_file: Optional[str]
    subs_in_millions: bool
    discount_rate_q: float
    elasticity: float       # churn elasticity to price (Δchurn_pp ≈ elasticity × price_pct_change)
    forecast_q: int
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "streaming_subscribers_wars_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def _parse_date(x):
    if pd.isna(x) or str(x).strip()=="":
        return pd.NaT
    s = str(x).strip()
    try:
        if "Q" in s.upper():
            y = int(s[:4]); q = int(s[-1])
            # use quarter end
            month = {1:3,2:6,3:9,4:12}[q]
            return pd.Timestamp(f"{y}-{month:02d}-01") + pd.offsets.MonthEnd(0)
        return pd.Timestamp(dtp.parse(s).date())
    except Exception:
        return pd.NaT

def _num(x):
    if pd.isna(x): return np.nan
    try:
        return float(str(x).replace(",","").replace("_",""))
    except Exception:
        return np.nan


# ----------------------------- Load & clean -----------------------------

def load_panel(path: str, subs_in_millions: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Accept wide format
    if "platform" not in df.columns and "date" in df.columns and any(c not in {"date"} for c in df.columns):
        # Melt all non-date columns as platforms with subs; revenue columns must be named like rev_<platform>, etc.
        # Simpler: require long for more variables; here we assume long format already
        pass
    if "platform" not in df.columns:
        raise SystemExit("panel must include 'platform' column (long format).")

    # base fields
    for c in ["subs","rev","arpu","cac","churn_q","gross_margin","content_spend","marketing_spend"]:
        if c in df.columns:
            df[c] = df[c].apply(_num)

    if "date" not in df.columns:
        raise SystemExit("panel must include 'date' column.")
    df["date"] = df["date"].apply(_parse_date)
    df["platform"] = df["platform"].astype(str).str.strip().str.title()

    # Normalize units
    if subs_in_millions:
        df["subs"] = df["subs"] * 1_000_000.0

    # Infer ARPU if missing (monthly)
    if "arpu" not in df.columns or df["arpu"].isna().all():
        if "rev" not in df.columns or df["rev"].isna().all():
            raise SystemExit("Need arpu or rev to compute metrics.")
        # rev is quarterly; arpu ≈ rev / (subs * 3)
        df["arpu"] = df["rev"] / (df["subs"] * 3.0)
    else:
        # If rev missing, infer from subs*arpu*3
        if "rev" not in df.columns or df["rev"].isna().all():
            df["rev"] = df["subs"] * df["arpu"] * 3.0

    # Gross margin
    if "gross_margin" not in df.columns or df["gross_margin"].isna().all():
        if ("content_spend" in df.columns or "marketing_spend" in df.columns) and "rev" in df.columns:
            c = df.get("content_spend", 0.0).fillna(0.0)
            m = df.get("marketing_spend", 0.0).fillna(0.0)
            df["gross_margin"] = (df["rev"] - c - m) / df["rev"].replace(0,np.nan)
        else:
            df["gross_margin"] = np.nan

    # Sort & set index
    df = df.sort_values(["platform","date"]).reset_index(drop=True)
    return df


# ----------------------------- Core metrics -----------------------------

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Net adds
    d["subs_prev"] = d.groupby("platform")["subs"].shift(1)
    d["net_adds"] = d["subs"] - d["subs_prev"]
    # Market totals & share
    totals = d.groupby("date")["subs"].sum().rename("market_subs")
    d = d.merge(totals, on="date", how="left")
    d["share"] = d["subs"] / d["market_subs"].replace(0,np.nan)
    # Share of net adds
    net_tot = d.groupby("date")["net_adds"].sum().rename("market_netadds")
    d = d.merge(net_tot, on="date", how="left")
    d["share_of_netadds"] = d["net_adds"] / d["market_netadds"].replace(0,np.nan)

    # QoQ / YoY
    d["subs_qoq"] = d.groupby("platform")["subs"].pct_change(1)
    d["subs_yoy"] = d.groupby("platform")["subs"].pct_change(4)
    d["rev_qoq"]  = d.groupby("platform")["rev"].pct_change(1)
    d["rev_yoy"]  = d.groupby("platform")["rev"].pct_change(4)

    # LTV (quarterly model): ARPU_m * 3 * GM / (disc + churn_q_effective)
    # If churn_q missing, estimate from net adds if gross adds known (we don't have), fallback: median per platform or 5%/q
    d["churn_q"] = d["churn_q"].groupby(d["platform"]).apply(lambda s: s.fillna(s.median()))
    d["churn_q"] = d["churn_q"].fillna(0.05)
    d["gross_margin"] = d["gross_margin"].fillna(d.groupby("platform")["gross_margin"].transform("median"))
    d["gross_margin"] = d["gross_margin"].fillna(0.35)

    # discount rate per quarter
    disc = d.attrs.get("_disc_q", 0.02)
    d["ltv_q"] = (d["arpu"] * 3.0 * d["gross_margin"]) / (disc + d["churn_q"])
    # CAC payback (quarters)
    if "cac" in d.columns:
        d["payback_q"] = d["cac"] / (d["arpu"] * 3.0 * d["gross_margin"])
    else:
        d["payback_q"] = np.nan

    # Unit contribution per sub per quarter
    d["unit_contrib_q"] = d["arpu"] * 3.0 * d["gross_margin"]

    # Rankers
    d["rank_share"] = d.groupby("date")["share"].rank(ascending=False, method="min")
    d["rank_netadds"] = d.groupby("date")["net_adds"].rank(ascending=False, method="min")

    return d


# ----------------------------- Scenarios -----------------------------

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    sc = pd.read_csv(path)
    sc.columns = [c.strip().lower() for c in sc.columns]
    need = {"platform","date","kind","value"}
    if not need.issubset(sc.columns):
        raise SystemExit("scenarios.csv must include: platform,date,kind,value")
    sc["platform"] = sc["platform"].astype(str).str.strip().str.title()
    sc["date"] = sc["date"].apply(_parse_date)
    sc["kind"] = sc["kind"].astype(str).str.strip().lower()
    sc["value"] = sc["value"].apply(_num)
    return sc


def apply_scenarios(df: pd.DataFrame, sc: pd.DataFrame, elasticity: float, disc_q: float) -> pd.DataFrame:
    if sc.empty:
        out = df.copy()
        out.attrs["_disc_q"] = disc_q
        return compute_metrics(out)

    d = df.copy()
    # For each scenario row, apply to that platform/date (and forward if needed)
    for _, r in sc.iterrows():
        mask = (d["platform"] == r["platform"]) & (d["date"] == r["date"])
        if not mask.any():  # allow nearest-forward if exact quarter missing
            nearest = d[(d["platform"]==r["platform"])].iloc[(d[(d["platform"]==r["platform"])]['date'] - r["date"]).abs().argmin()].name
            mask = False
            # Use exact only to avoid accidental mass updates
            continue
        kind = r["kind"]; val = float(r["value"] or 0)
        if kind == "price_hike_pct":
            # ARPU ↑ by pct; churn increases by elasticity * pct (in pp)
            d.loc[mask, "arpu"] = d.loc[mask, "arpu"] * (1 + val/100.0)
            d.loc[mask, "churn_q"] = (d.loc[mask, "churn_q"].fillna(0.05) + (elasticity * val/100.0)).clip(lower=0.0)
        elif kind == "arpu_delta_pct":
            d.loc[mask, "arpu"] = d.loc[mask, "arpu"] * (1 + val/100.0)
        elif kind == "churn_delta_pp":
            d.loc[mask, "churn_q"] = (d.loc[mask, "churn_q"].fillna(0.05) + val/100.0).clip(lower=0.0)
        elif kind == "bundle_uplift_pct":
            # ARPU ↑ and churn ↓ modestly
            d.loc[mask, "arpu"] = d.loc[mask, "arpu"] * (1 + val/100.0)
            d.loc[mask, "churn_q"] = (d.loc[mask, "churn_q"].fillna(0.05) - (0.3*val/100.0)).clip(lower=0.0)

    d.attrs["_disc_q"] = disc_q
    return compute_metrics(d)


# ----------------------------- Simple forecast -----------------------------

def forecast(df: pd.DataFrame, horizon_q: int, disc_q: float) -> pd.DataFrame:
    if horizon_q <= 0:
        return pd.DataFrame()
    # For each platform, extend last observation with ARPU/churn drift (last 4q trend) and subs via net adds trend
    last_date = df["date"].max()
    dates = pd.period_range(last_date + pd.offsets.QuarterEnd(0), periods=horizon_q, freq="Q").to_timestamp("Q")

    rows = []
    for p, g in df.groupby("platform"):
        g = g.sort_values("date")
        if g.empty: continue
        last = g.iloc[-1]
        subs = float(last["subs"])
        arpu = float(last["arpu"])
        churn_q = float(last.get("churn_q", 0.05) if pd.notna(last.get("churn_q", np.nan)) else 0.05)
        gm = float(last.get("gross_margin", 0.35) if pd.notna(last.get("gross_margin", np.nan)) else 0.35)
        # Trends
        arpu_trend = g["arpu"].pct_change().tail(4).mean(skipna=True)
        churn_trend = (g["churn_q"].diff().tail(4).mean(skipna=True) if "churn_q" in g.columns else 0.0)
        netadds_trend = g["net_adds"].tail(4).mean(skipna=True)
        if np.isnan(netadds_trend): netadds_trend = 0.0
        if np.isnan(arpu_trend): arpu_trend = 0.0
        if np.isnan(churn_trend): churn_trend = 0.0

        for i, dte in enumerate(dates, 1):
            # subs evolve by last net adds trend, damped as share approaches 1 (saturation)
            market_last = df[df["date"]==df["date"].max()]["subs"].sum()
            share = subs / max(market_last, 1.0)
            damp = max(0.4, 1 - 0.8*share)  # heuristic
            subs = max(0.0, subs + damp * netadds_trend)
            # ARPU & churn evolve
            arpu *= (1 + 0.5*arpu_trend)  # half-speed persistence
            churn_q = max(0.0, churn_q + 0.5*churn_trend)
            rev = subs * arpu * 3.0
            ltv_q = (arpu * 3.0 * gm) / (disc_q + churn_q) if (disc_q + churn_q) > 0 else np.nan
            rows.append({"platform": p, "date": dte, "subs": subs, "arpu": arpu, "churn_q": churn_q,
                         "gross_margin": gm, "rev": rev, "ltv_q": ltv_q})

    fc = pd.DataFrame(rows)
    if fc.empty: return fc
    # Recompute share fields with historical market size extended by forecast
    hist_last = df.groupby("date")["subs"].sum().rename("market_subs")
    fc = fc.merge(hist_last, left_on="date", right_index=True, how="left")
    # For new dates not in history, market_subs = sum of forecasted platforms at that date
    for dte, group in fc.groupby("date"):
        if pd.isna(group["market_subs"]).all():
            total = group["subs"].sum()
            fc.loc[fc["date"]==dte, "market_subs"] = total
    fc["share"] = fc["subs"] / fc["market_subs"].replace(0,np.nan)
    return fc.sort_values(["platform","date"])


# ----------------------------- Plotting -----------------------------

def make_plots(metrics: pd.DataFrame, share: pd.DataFrame, ltv: pd.DataFrame, fc: pd.DataFrame, outdir: str):
    if plt is None: 
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Market share (top 6 platforms)
    fig1 = plt.figure(figsize=(10,6)); ax1 = plt.gca()
    top = metrics.groupby("platform")["subs"].last().sort_values(ascending=False).head(6).index
    sub = metrics[metrics["platform"].isin(top)]
    for p in top:
        s = sub[sub["platform"]==p].set_index("date")["share"]
        ax1.plot(s.index, 100*s.values, label=p)
    ax1.set_title("Market share of subscribers"); ax1.set_ylabel("%"); ax1.legend()
    plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "market_share.png"), dpi=140); plt.close(fig1)

    # Net adds stacked area
    fig2 = plt.figure(figsize=(10,6)); ax2 = plt.gca()
    pivot = metrics.pivot_table(index="date", columns="platform", values="net_adds", aggfunc="sum")
    pivot.fillna(0.0).plot.area(ax=ax2)
    ax2.set_title("Net adds by platform (stacked)"); ax2.set_ylabel("subs")
    plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "net_adds_stacked.png"), dpi=140); plt.close(fig2)

    # LTV vs CAC scatter (latest quarter)
    fig3 = plt.figure(figsize=(8,5)); ax3 = plt.gca()
    last = ltv.sort_values("date").groupby("platform").tail(1)
    ax3.scatter(last["ltv_q"], last["cac"], s=60)
    for _, r in last.iterrows():
        ax3.annotate(r["platform"], (r["ltv_q"], r["cac"]), fontsize=8, xytext=(5,5), textcoords="offset points")
    ax3.set_title("LTV (quarterly model) vs CAC"); ax3.set_xlabel("LTV (USD)"); ax3.set_ylabel("CAC (USD)")
    plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "ltv_vs_cac.png"), dpi=140); plt.close(fig3)

    # Forecast paths (subs)
    if not fc.empty:
        fig4 = plt.figure(figsize=(10,6)); ax4 = plt.gca()
        for p in fc["platform"].unique():
            s_hist = metrics[metrics["platform"]==p].set_index("date")["subs"]
            s_fc = fc[fc["platform"]==p].set_index("date")["subs"]
            ax4.plot(s_hist.index, s_hist.values, label=f"{p} hist")
            ax4.plot(s_fc.index, s_fc.values, linestyle="--", label=f"{p} fc")
        ax4.set_title("Subscribers — history & forecast"); ax4.legend(ncol=2)
        plt.tight_layout(); fig4.savefig(os.path.join(outdir, "plots", "subs_forecast.png"), dpi=140); plt.close(fig4)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Streaming Subscriber Wars: market share, net adds, unit economics & scenarios")
    ap.add_argument("--panel", dest="panel_file", required=True, help="CSV with platform quarterly panel")
    ap.add_argument("--scenarios", dest="scenarios_file", default=None, help="Optional scenarios CSV")
    ap.add_argument("--subs-in-millions", action="store_true", help="Interpret 'subs' column as millions")
    ap.add_argument("--discount-rate-q", type=float, default=0.02, help="Quarterly discount rate for LTV (e.g., 0.02)")
    ap.add_argument("--elasticity", type=float, default=0.20, help="Churn elasticity to price (%pp churn per 100% price)")
    ap.add_argument("--forecast-q", type=int, default=8, help="Quarters to forecast ahead")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        panel_file=args.panel_file,
        scenarios_file=args.scenarios_file,
        subs_in_millions=bool(args.subs_in_millions),
        discount_rate_q=float(args.discount_rate_q),
        elasticity=float(args.elasticity),
        forecast_q=int(max(0, args.forecast_q)),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir),
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    panel = load_panel(cfg.panel_file, cfg.subs_in_millions)
    panel.attrs["_disc_q"] = cfg.discount_rate_q
    panel.to_csv(os.path.join(cfg.outdir, "cleaned_panel.csv"), index=False)

    scenarios = load_scenarios(cfg.scenarios_file) if cfg.scenarios_file else pd.DataFrame()

    metrics = apply_scenarios(panel, scenarios, cfg.elasticity, cfg.discount_rate_q)
    metrics.to_csv(os.path.join(cfg.outdir, "metrics_quarterly.csv"), index=False)

    # LTV/unit econ table (subset of columns)
    ltv_cols = ["date","platform","subs","rev","arpu","gross_margin","churn_q","ltv_q","cac","payback_q","unit_contrib_q","share","share_of_netadds","subs_qoq","subs_yoy","rev_qoq","rev_yoy"]
    ltv_tbl = metrics[ltv_cols].copy()
    ltv_tbl.to_csv(os.path.join(cfg.outdir, "ltv_unit_econ.csv"), index=False)

    # Share table
    share_tbl = metrics.pivot_table(index="date", columns="platform", values="share", aggfunc="last")
    share_tbl.to_csv(os.path.join(cfg.outdir, "share_table.csv"))

    # Forecast
    fc = forecast(metrics[["platform","date","subs","arpu","churn_q","gross_margin","rev"]], cfg.forecast_q, cfg.discount_rate_q)
    if not fc.empty:
        fc.to_csv(os.path.join(cfg.outdir, "forecast.csv"), index=False)

    # Plots
    if cfg.plot:
        make_plots(metrics, share_tbl, ltv_tbl, fc, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshots
    print("\n=== Latest quarter league table ===")
    last_q = metrics["date"].max()
    snap = metrics[metrics["date"]==last_q][["platform","subs","net_adds","share","arpu","churn_q","ltv_q","payback_q"]]
    snap = snap.sort_values("subs", ascending=False)
    snap["share_%"] = (100*snap["share"]).round(2)
    print(snap[["platform","subs","net_adds","share_%","arpu","churn_q","ltv_q","payback_q"]].round(3).to_string(index=False))

    if not fc.empty:
        print("\n=== Forecast (next 4 quarters, subs by platform) ===")
        nxt = fc.groupby(["platform"]).head(4).pivot_table(index="date", columns="platform", values="subs", aggfunc="last")
        print(nxt.round(0).to_string())

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()