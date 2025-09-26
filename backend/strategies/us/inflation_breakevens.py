#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# inflation_breakevens.py
#
# Breakeven inflation toolkit:
# - Pulls nominal & TIPS constant-maturity yields from FRED (optional)
# - Or ingests your own yields (CSV)
# - Computes spot breakevens, forwards (e.g., 5y5y), carry/roll proxies
# - Optional simple liquidity-premium & seasonality adjustments
# - Exports tidy CSVs and (optionally) plots
#
# Examples
# --------
# # 1) Pure FRED pull (daily):
# python inflation_breakevens.py --fred --start 2010-01-01 --plot
#
# # 2) Use your own curve snapshots:
# python inflation_breakevens.py --yields my_curve.csv --asof 2025-09-05 --plot
#
# my_curve.csv format (wide or long):
#   # wide
#   date,nom_2y,nom_5y,nom_7y,nom_10y,nom_30y,tips_5y,tips_10y,tips_30y
#   2025-09-05,3.95,3.65,3.55,3.50,3.90,1.80,1.60,1.55
#   # long
#   date,kind,tenor,rate
#   2025-09-05,nom,5y,3.65
#   2025-09-05,tips,5y,1.80
#
# Outputs
# -------
# outdir/
#   fred_raw.csv                   (if --fred)
#   curve_clean.csv                (aligned nom/tips per tenor)
#   breakevens_spot.csv            (spot BE per tenor)
#   breakevens_forwards.csv        (2y2y, 5y5y, 10y10y if tenors available)
#   carry_roll_proxy.csv           (breakeven carry/roll approximations)
#   plots/*.png                    (optional)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib pandas_datareader python-dateutil
#
# Notes
# -----
# - “Breakeven” ≈ nominal yield − real (TIPS) yield at same maturity.
# - Forward XyYy using simple-yield algebra: f_{a,b} = ((R_{a+b}(a+b) − R_a*a) / b).
#   Apply separately to nominal and real, then take (nom_fwd − real_fwd).
# - A very rough seasonality adjustment (optional) can be applied by adding a
#   user-provided monthly factor (bps) to carry for the months that will be in-the-window.
# - Liquidity premium (LP) option subtracts a user-set number of bps from spot BE
#   for chosen short tenors (e.g., 10 bps from 5y) to approximate TIPS liquidity effects.

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from dateutil import parser as dtp


# --------------------------- Config ---------------------------

@dataclass
class Config:
    yields_file: Optional[str]
    fred: bool
    start: Optional[str]
    end: Optional[str]
    asof: Optional[str]
    lp_bps_5y: float
    lp_bps_10y: float
    seasonality_file: Optional[str]
    plot: bool
    outdir: str


# --------------------------- IO helpers ---------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "inflation_breakevens_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def tenor_norm(s: str) -> str:
    s = str(s).strip().lower()
    if s.endswith("y"): return s
    if s.endswith("yr"): return s[:-2]+"y"
    if s.endswith("year"): return s[:-4]+"y"
    if s.endswith("years"): return s[:-5]+"y"
    return s


def read_yields_user(path: str) -> pd.DataFrame:
    """
    Returns a tidy long df: date, kind ('nom'|'tips'), tenor ('2y','5y','7y','10y','30y'), rate (decimal)
    Accepts wide or long input.
    """
    df = pd.read_csv(path)
    if {"date","kind","tenor","rate"} <= set(df.columns):
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"])
        x["kind"] = x["kind"].str.lower().str.strip()
        x["tenor"] = x["tenor"].astype(str).apply(tenor_norm)
        x["rate"] = pd.to_numeric(x["rate"], errors="coerce")/100.0 if x["rate"].abs().max()>1.5 else pd.to_numeric(x["rate"], errors="coerce")
        return x.sort_values(["date","kind","tenor"])
    # wide path
    df["date"] = pd.to_datetime(df["date"])
    long_rows = []
    for c in df.columns:
        if c == "date": continue
        v = c.lower().strip()
        if v.startswith("nom_"):
            tenor = tenor_norm(v.split("_",1)[1])
            kind = "nom"
        elif v.startswith("tips_") or v.startswith("real_"):
            tenor = tenor_norm(v.split("_",1)[1])
            kind = "tips"
        else:
            continue
        for d, val in zip(df["date"], df[c]):
            long_rows.append({"date": d, "kind": kind, "tenor": tenor, "rate": pd.to_numeric(val, errors="coerce")/100.0 if abs(pd.to_numeric(val, errors="coerce"))>1.5 else pd.to_numeric(val, errors="coerce")})
    x = pd.DataFrame(long_rows).dropna(subset=["rate"])
    return x.sort_values(["date","kind","tenor"])


def pull_fred(start: Optional[str], end: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Pull daily constant maturity yields:
      Nominal: DGS2, DGS5, DGS7, DGS10, DGS30
      TIPS   : DFII5, DFII7, DFII10, DFII30 (2y real is no longer published; omit)
    """
    if pdr is None:
        print("[WARN] pandas_datareader not available; skipping FRED.")
        return None
    series = {
        "nom_2y":"DGS2", "nom_5y":"DGS5", "nom_7y":"DGS7", "nom_10y":"DGS10", "nom_30y":"DGS30",
        "tips_5y":"DFII5", "tips_7y":"DFII7", "tips_10y":"DFII10", "tips_30y":"DFII30"
    }
    out = []
    for col, code in series.items():
        try:
            s = pdr.DataReader(code, "fred", start or "2003-01-01", end)
            s.columns = [col]
            out.append(s)
        except Exception as e:
            print(f"[WARN] FRED fetch failed for {code}: {e}")
    if not out:
        return None
    df = pd.concat(out, axis=1)
    # convert percent → decimal
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")/100.0
    return df.dropna(how="all")


def tidy_from_fred(df: pd.DataFrame) -> pd.DataFrame:
    """Convert fred wide to tidy long (date, kind, tenor, rate)"""
    rows = []
    for c in df.columns:
        if c.startswith("nom_"):
            kind = "nom"; tenor = c.split("_",1)[1]
        elif c.startswith("tips_"):
            kind = "tips"; tenor = c.split("_",1)[1]
        else:
            continue
        for d, v in df[c].items():
            if pd.isna(v): continue
            rows.append({"date": d, "kind": kind, "tenor": tenor, "rate": float(v)})
    return pd.DataFrame(rows).sort_values(["date","kind","tenor"])


def read_seasonality(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    seasonality CSV format:
      month,carry_bps
      1,2.0
      2,1.5
      ...
    Positive adds to carry in that month (very rough).
    """
    if not path:
        return None
    s = pd.read_csv(path)
    if not {"month","carry_bps"} <= set(s.columns):
        raise SystemExit("seasonality file must have columns: month, carry_bps")
    s["month"] = pd.to_numeric(s["month"], errors="coerce").astype(int)
    s["carry_bps"] = pd.to_numeric(s["carry_bps"], errors="coerce").fillna(0.0)
    return s


# --------------------------- Core math ---------------------------

def pivot_curve(tidy: pd.DataFrame) -> pd.DataFrame:
    """Wide on a given date: index=date, columns like nom_5y, tips_5y, ..."""
    wide = tidy.pivot_table(index="date", columns=["kind","tenor"], values="rate", aggfunc="last")
    # Flatten columns
    wide.columns = [f"{k}_{t}" for k,t in wide.columns]
    return wide.sort_index()


def spot_breakevens(wide: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=wide.index)
    for ten in ["2y","5y","7y","10y","30y"]:
        n = f"nom_{ten}"; r = f"tips_{ten}"
        if (n in wide.columns) and (r in wide.columns):
            out[f"BE_{ten}"] = (wide[n] - wide[r])
    return out.dropna(how="all")


def simple_forward(r_a: pd.Series, r_b: pd.Series, a_years: float, b_years: float) -> pd.Series:
    """
    f_{a,b} using simple yields (not compounding convention sensitive):
        f = ((R_{a+b}*(a+b) - R_a*a)/b)
    """
    return ((r_b*(a_years+b_years) - r_a*a_years) / b_years)


def forward_breakevens(wide: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(index=wide.index)
    # 5y5y if 5y and 10y present
    if {"nom_5y","nom_10y","tips_5y","tips_10y"} <= set(wide.columns):
        f_nom = simple_forward(wide["nom_5y"], wide["nom_10y"], 5.0, 5.0)
        f_real = simple_forward(wide["tips_5y"], wide["tips_10y"], 5.0, 5.0)
        res["BE_5y5y"] = f_nom - f_real
        # also classic algebra on BE directly (approx): 2*BE10 - BE5
        res["BE_5y5y_from_BE"] = 2.0*(wide["nom_10y"]-wide["tips_10y"]) - (wide["nom_5y"]-wide["tips_5y"])
    # 2y2y if 2y & 4y (approx via 2y and 5y) → use 2y & 5y as proxy for 2y3y; if 7y present do 2y5y via 7y
    if {"nom_2y","nom_5y","tips_2y","tips_5y"} <= set(wide.columns):
        f_nom_2y3y = simple_forward(wide["nom_2y"], wide["nom_5y"], 2.0, 3.0)
        f_real_2y3y = simple_forward(wide["tips_2y"], wide["tips_5y"], 2.0, 3.0)
        res["BE_2y3y"] = f_nom_2y3y - f_real_2y3y
    if {"nom_10y","nom_30y","tips_10y","tips_30y"} <= set(wide.columns):
        f_nom_10y10y = simple_forward(wide["nom_10y"], wide["nom_30y"], 10.0, 20.0)
        f_real_10y10y = simple_forward(wide["tips_10y"], wide["tips_30y"], 10.0, 20.0)
        res["BE_10y10y"] = f_nom_10y10y - f_real_10y10y
    return res.dropna(how="all")


def apply_liquidity_premium(be_spot: pd.DataFrame, lp_5y_bps: float, lp_10y_bps: float) -> pd.DataFrame:
    adj = be_spot.copy()
    if "BE_5y" in adj.columns and lp_5y_bps != 0:
        adj["BE_5y_LPadj"] = adj["BE_5y"] - (lp_5y_bps/1e4)
    if "BE_10y" in adj.columns and lp_10y_bps != 0:
        adj["BE_10y_LPadj"] = adj["BE_10y"] - (lp_10y_bps/1e4)
    return adj


def carry_roll_proxy(be_spot: pd.DataFrame, curve_date_index: pd.DatetimeIndex,
                     seasonality: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Heuristic carry+roll proxy:
      - Carry ≈ slope between nearby tenors (e.g., BE_5y vs BE_10y) scaled for 1M roll
      - Roll ≈ where a 5y note drifts along the curve after 1 month: linear interp between 4y11m & 5y1m (approx)
    We implement a simple 1M proxy:
      carry_5y ≈ (BE_10y - BE_5y)/5 * (1/12)
      carry_10y ≈ (BE_10y - BE_5y)/5 * (1/12)  (using same local slope)
    Seasonality: add monthly factor (bps) for the month after 'date' if table provided.
    """
    out = pd.DataFrame(index=be_spot.index)
    if {"BE_5y","BE_10y"} <= set(be_spot.columns):
        slope = (be_spot["BE_10y"] - be_spot["BE_5y"]) / 5.0  # per-year slope
        out["carry_5y_1m"] = slope/12.0
        out["carry_10y_1m"] = slope/12.0
    # Seasonality
    if seasonality is not None and not seasonality.empty:
        # Map date→ next month factor
        next_month = pd.Index([ (d.to_pydatetime().replace(day=15) + pd.offsets.MonthEnd(1)).month for d in curve_date_index ])
        m = pd.Series(next_month, index=curve_date_index).map(dict(zip(seasonality["month"], seasonality["carry_bps"]/1e4)))
        m = m.fillna(0.0)
        if "carry_5y_1m" in out.columns: out["carry_5y_1m"] = out["carry_5y_1m"] + m.reindex(out.index).fillna(0.0)
        if "carry_10y_1m" in out.columns: out["carry_10y_1m"] = out["carry_10y_1m"] + m.reindex(out.index).fillna(0.0)
    return out


# --------------------------- Plotting ---------------------------

def make_plots(be_spot: pd.DataFrame, be_fwd: pd.DataFrame, outdir: str):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Spot BE (5y/10y)
    common = [c for c in ["BE_5y","BE_10y","BE_30y"] if c in be_spot.columns]
    if common:
        fig1 = plt.figure(figsize=(10,5)); ax = plt.gca()
        (100*be_spot[common]).plot(ax=ax)
        ax.set_title("Spot breakevens"); ax.set_ylabel("%")
        plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "spot_breakevens.png"), dpi=140); plt.close(fig1)

    # 5y5y forward
    for_col = [c for c in ["BE_5y5y","BE_5y5y_from_BE"] if c in be_fwd.columns]
    if for_col:
        fig2 = plt.figure(figsize=(10,5)); ax2 = plt.gca()
        (100*be_fwd[for_col]).plot(ax=ax2)
        ax2.set_title("5y5y forward breakeven"); ax2.set_ylabel("%")
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "forward_5y5y.png"), dpi=140); plt.close(fig2)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Inflation breakevens: spot, forwards, carry/roll")
    ap.add_argument("--yields", dest="yields_file", default=None, help="CSV with nominal/TIPS yields")
    ap.add_argument("--fred", action="store_true", help="Pull from FRED (DGSx, DFIIx)")
    ap.add_argument("--start", type=str, default=None, help="Start date for FRED/history")
    ap.add_argument("--end", type=str, default=None, help="End date")
    ap.add_argument("--asof", type=str, default=None, help="If using CSV with many dates, pick one (YYYY-MM-DD)")
    ap.add_argument("--lp-bps-5y", type=float, default=0.0, help="Subtract this from 5y BE (liquidity premium)")
    ap.add_argument("--lp-bps-10y", type=float, default=0.0, help="Subtract this from 10y BE (liquidity premium)")
    ap.add_argument("--seasonality", type=str, default=None, help="Optional CSV with month,carry_bps")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        yields_file=args.yields_file,
        fred=bool(args.fred),
        start=args.start,
        end=args.end,
        asof=args.asof,
        lp_bps_5y=float(args.lp_bps_5y),
        lp_bps_10y=float(args.lp_bps_10y),
        seasonality_file=args.seasonality,
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir),
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    # Load data
    tidy = pd.DataFrame(columns=["date","kind","tenor","rate"])
    fred_raw = None
    if cfg.fred:
        fred_raw = pull_fred(cfg.start, cfg.end)
        if fred_raw is not None:
            fred_raw.to_csv(os.path.join(cfg.outdir, "fred_raw.csv"))
            tidy = tidy_from_fred(fred_raw)
    if cfg.yields_file:
        user_tidy = read_yields_user(cfg.yields_file)
        if cfg.asof:
            asof_ts = pd.Timestamp(dtp.parse(cfg.asof))
            user_tidy = user_tidy[user_tidy["date"] == asof_ts]
        tidy = pd.concat([tidy, user_tidy], ignore_index=True) if not tidy.empty else user_tidy

    if tidy.empty:
        raise SystemExit("No data. Provide --fred or --yields file.")

    # Pivot to daily wide
    wide_all = pivot_curve(tidy)
    wide_all.to_csv(os.path.join(cfg.outdir, "curve_clean.csv"))

    # Breakeven spot & forwards
    be_spot = spot_breakevens(wide_all)
    be_fwd = forward_breakevens(wide_all)

    # Liquidity premium adjustment
    be_spot_adj = apply_liquidity_premium(be_spot, cfg.lp_bps_5y, cfg.lp_bps_10y)

    # Carry/roll (1M heuristic) + optional seasonality
    seasonality = read_seasonality(cfg.seasonality_file)
    crr = carry_roll_proxy(be_spot_adj, be_spot_adj.index, seasonality)

    # Save
    be_spot_adj.to_csv(os.path.join(cfg.outdir, "breakevens_spot.csv"))
    be_fwd.to_csv(os.path.join(cfg.outdir, "breakevens_forwards.csv"))
    crr.to_csv(os.path.join(cfg.outdir, "carry_roll_proxy.csv"))

    # Plots
    if cfg.plot:
        make_plots(be_spot_adj, be_fwd, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot (last row)
    last = be_spot_adj.dropna(how="all").tail(1)
    if not last.empty:
        dt = last.index[-1].date()
        cols = [c for c in ["BE_2y","BE_5y","BE_10y","BE_30y","BE_5y_LPadj","BE_10y_LPadj"] if c in last.columns]
        print("\n=== Latest spot BE (%) on", dt, "===")
        print((100*last[cols]).round(2).to_string(index=False))
    if not be_fwd.empty:
        last_f = be_fwd.dropna(how="all").tail(1)
        dtf = last_f.index[-1].date()
        print("\n=== Latest forward BE (%) on", dtf, "===")
        print((100*last_f).round(2).to_string())

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()