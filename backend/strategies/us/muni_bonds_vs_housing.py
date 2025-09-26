#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# muni_bonds_vs_housing.py
#
# Municipal Bonds ↔ Housing Macro Toolkit
# ---------------------------------------
# What this does
# - Ingests *your* muni & treasury yields (e.g., AAA muni 5y/10y and UST 5y/10y) and/or pulls common housing series from FRED
# - Builds muni/UST ratios & muni spreads, then studies their relationship with housing activity:
#     • Housing starts (HOUST), building permits (PERMIT)
#     • Case-Shiller home price index (CSUSHPINSA) & 30y mortgage rate (MORTGAGE30US)
# - Creates rolling correlations, lead/lag cross-correlations, and simple regressions
# - Optional segmentation by states/MSAs if you provide panel CSVs
# - Exports tidy CSVs + plots
#
# Why ratios? Muni/UST (e.g., 10y muni ÷ 10y UST) is the common tax-equivalent gauge. We also compute muni − UST spreads.
#
# Inputs (choose any)
# -------------------
# 1) Your own curve snapshots/time series CSV (recommended)
#    Columns (wide or long both accepted):
#      # wide
#      date,muni_5y,muni_10y,ust_5y,ust_10y
#      2020-01-31,1.15,1.35,1.31,1.52
#      # long
#      date,kind,tenor,rate
#      2020-01-31,muni,10y,1.35
#      2020-01-31,ust,10y,1.52
#
# 2) FRED pull for housing + mortgage rate (muni yields must be in your CSV)
#    HOUST, PERMIT, CSUSHPINSA, MORTGAGE30US, DGS10 (backup for ust_10y if needed)
#
# Usage
# -----
# python muni_bonds_vs_housing.py \
#   --yields muni_ust_curve.csv \
#   --fred --start 2000-01-01 --plot
#
# Optional: only your CSV (no FRED) if you already merged everything
# python muni_bonds_vs_housing.py --yields my_merged.csv --plot
#
# Outputs (./artifacts/muni_bonds_vs_housing/*)
# --------------------------------------------
#   curve_clean.csv                 (muni & UST wide panel, % as decimals)
#   muni_metrics.csv                (ratios, spreads, z-scores)
#   housing_raw.csv                 (FRED pulls, if selected)
#   merged_monthly.csv              (monthly merge for analysis)
#   rolling_corr_24m.csv            (rolling Pearson 24m windows)
#   xcorr_table.csv                 (lead/lag max correlations)
#   regressions.csv                 (simple OLS summaries)
#   plots/*.png                     (term ratio, corr panels, scatter, etc.)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib pandas_datareader statsmodels python-dateutil
#
# Notes
# -----
# - All rates should be in **percent** (e.g., 1.50) or decimal (0.015); auto-detection converts to decimal.
# - If you don’t have muni yields by tenor, you can still study housing vs mortgage/UST—ratios will be missing.
# - This is research-grade: for trading/reporting, validate series definitions (e.g., AAA GO vs MMD curve).

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

import statsmodels.api as sm
from dateutil import parser as dtp


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    yields_file: Optional[str]
    fred: bool
    start: Optional[str]
    end: Optional[str]
    roll_window: int
    max_lag: int
    plot: bool
    outdir: str


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "muni_bonds_vs_housing_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def _rate_to_decimal(s: pd.Series) -> pd.Series:
    """Auto-convert percent→decimal if series looks like percentages."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().abs().median() > 1.5:  # likely in %
        return s / 100.0
    return s


def read_yields_user(path: str) -> pd.DataFrame:
    """
    Accept wide or long. Return a monthly wide panel with columns:
    muni_5y, muni_10y, ust_5y, ust_10y (decimals), indexed by month-end.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if {"date","kind","tenor","rate"} <= set(df.columns):
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"])
        x["kind"] = x["kind"].str.strip().str.lower()
        x["tenor"] = x["tenor"].str.strip().str.lower()
        x["rate"] = _rate_to_decimal(x["rate"])
        wide = x.pivot_table(index="date", columns=["kind","tenor"], values="rate", aggfunc="last")
        wide.columns = [f"{k}_{t}" for k,t in wide.columns]
    else:
        if "date" not in df.columns:
            raise SystemExit("yields CSV must include 'date' plus rate columns or a long (date,kind,tenor,rate).")
        df["date"] = pd.to_datetime(df["date"])
        wide = df.set_index("date").copy()
        # normalize common names
        ren = {
            "muni_5yr":"muni_5y","muni_10yr":"muni_10y",
            "ust_5yr":"ust_5y","ust_10yr":"ust_10y","dgs10":"ust_10y"
        }
        wide.rename(columns=ren, inplace=True)
        for c in wide.columns:
            wide[c] = _rate_to_decimal(wide[c])
    # Monthly end frequency
    wide = wide.resample("M").last()
    # Keep only the columns of interest if present
    keep = [c for c in wide.columns if c in {"muni_5y","muni_10y","ust_5y","ust_10y"}]
    if not keep:
        print("[WARN] No recognized muni/UST columns found; proceeding with whatever is present.")
        keep = list(wide.columns)
    return wide[keep].sort_index()


def pull_fred_housing(start: Optional[str], end: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Pulls common housing/mortgage series (monthly where possible).
    - HOUST: Total Housing Starts
    - PERMIT: Building Permits
    - CSUSHPINSA: Case-Shiller US (NSA) (monthly)
    - MORTGAGE30US: 30y mortgage rate (weekly → monthly last)
    - DGS10: 10y Treasury (daily → monthly last; only for backup)
    """
    if pdr is None:
        print("[WARN] pandas_datareader not available; skip FRED.")
        return None
    series = ["HOUST","PERMIT","CSUSHPINSA","MORTGAGE30US","DGS10"]
    frames = []
    for code in series:
        try:
            s = pdr.DataReader(code, "fred", start or "1990-01-01", end)
            s.columns = [code]
            frames.append(s)
        except Exception as e:
            print(f"[WARN] FRED fetch failed for {code}: {e}")
    if not frames:
        return None
    df = pd.concat(frames, axis=1)
    # convert to monthly: last observation of the month
    monthly = df.resample("M").last()
    # Convert % to decimal for rates
    for c in ["MORTGAGE30US","DGS10"]:
        if c in monthly.columns:
            monthly[c] = _rate_to_decimal(monthly[c])
    return monthly


# ----------------------------- Metrics -----------------------------

def muni_metrics(curve: pd.DataFrame, fred: Optional[pd.DataFrame]) -> pd.DataFrame:
    d = curve.copy()
    # Fill UST 10y from FRED if missing
    if ("ust_10y" not in d.columns) and (fred is not None) and ("DGS10" in fred.columns):
        d = d.join(fred[["DGS10"]], how="left")
        d["ust_10y"] = d["DGS10"]
        d.drop(columns=["DGS10"], inplace=True)

    # Ratios & spreads
    for tenor in ["5y","10y"]:
        mcol = f"muni_{tenor}"
        ucol = f"ust_{tenor}"
        if mcol in d.columns and ucol in d.columns:
            d[f"ratio_{tenor}"] = d[mcol] / d[ucol].replace(0, np.nan)
            d[f"spread_{tenor}_bps"] = 10000 * (d[mcol] - d[ucol])
    # Z-scores (rolling 5y = 60m)
    for col in [c for c in d.columns if c.startswith("ratio_") or c.startswith("spread_")]:
        r = d[col].rolling(60)
        d[col + "_z"] = (d[col] - r.mean()) / r.std(ddof=1)
    return d


def build_merge(muni: pd.DataFrame, fred: Optional[pd.DataFrame]) -> pd.DataFrame:
    if fred is None:
        merged = muni.copy()
    else:
        merged = muni.join(fred, how="left")
    # rename for clarity
    rn = {
        "HOUST":"housing_starts", "PERMIT":"permits",
        "CSUSHPINSA":"case_shiller_nsa",
        "MORTGAGE30US":"mortgage30y", "DGS10":"ust10y_fred"
    }
    merged.rename(columns=rn, inplace=True)

    # Growth/YoY
    for col in ["housing_starts","permits","case_shiller_nsa"]:
        if col in merged.columns:
            merged[col + "_yoy"] = merged[col].pct_change(12)
    return merged


def rolling_corr(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    out = {}
    # correlate muni ratios/spreads with housing series YoY or levels
    muni_cols = [c for c in df.columns if c.startswith("ratio_") or c.startswith("spread_")]
    house_cols = [c for c in df.columns if c in {"housing_starts_yoy","permits_yoy","case_shiller_nsa_yoy"}]
    if not house_cols:
        house_cols = [c for c in df.columns if c in {"housing_starts","permits","case_shiller_nsa"}]
    for m in muni_cols:
        for h in house_cols:
            name = f"corr_{m}_vs_{h}_{window}m"
            out[name] = df[m].rolling(window).corr(df[h])
    return pd.DataFrame(out).dropna(how="all")


def xcorr_leadlag(a: pd.Series, b: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """
    Cross-correlation for lags in [-max_lag .. +max_lag], where lag>0 means 'a' leads 'b' by lag months.
    """
    a, b = a.align(b, join="inner")
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k > 0:
            s1, s2 = a.iloc[:-k], b.iloc[k:]
        elif k < 0:
            s1, s2 = a.iloc[-k:], b.iloc[:k]
        else:
            s1, s2 = a, b
        r = s1.corr(s2)
        rows.append({"lag": k, "corr": r})
    df = pd.DataFrame(rows)
    best = df.iloc[df["corr"].abs().idxmax()]
    df.attrs["best_lag"] = int(best["lag"])
    df.attrs["best_corr"] = float(best["corr"])
    return df


def regressions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple OLS examples (monthly):
      1) starts_yoy ~ ratio_10y (lagged) + mortgage30y (level)
      2) permits_yoy ~ spread_10y_bps (lagged) + mortgage30y (level)
    """
    rows = []
    if "housing_starts_yoy" in df.columns:
        X = pd.concat([
            df.get("ratio_10y").shift(6),
            df.get("mortgage30y")
        ], axis=1).dropna()
        X.columns = ["ratio10y_l6", "mortgage30y"]
        Y = df["housing_starts_yoy"].reindex(X.index)
        if Y.dropna().shape[0] > 24:
            Xc = sm.add_constant(X)
            res = sm.OLS(Y, Xc, missing="drop").fit()
            rows.append({"model": "starts_yoy ~ ratio10y_l6 + mortgage30y",
                         "nobs": int(res.nobs),
                         "r2": float(res.rsquared),
                         "coef_const": float(res.params.get("const", np.nan)),
                         "coef_ratio10y_l6": float(res.params.get("ratio10y_l6", np.nan)),
                         "coef_mortgage30y": float(res.params.get("mortgage30y", np.nan))})
    if "permits_yoy" in df.columns:
        X = pd.concat([
            df.get("spread_10y_bps").shift(6),
            df.get("mortgage30y")
        ], axis=1).dropna()
        X.columns = ["spread10y_bps_l6", "mortgage30y"]
        Y = df["permits_yoy"].reindex(X.index)
        if Y.dropna().shape[0] > 24:
            Xc = sm.add_constant(X)
            res = sm.OLS(Y, Xc, missing="drop").fit()
            rows.append({"model": "permits_yoy ~ spread10y_bps_l6 + mortgage30y",
                         "nobs": int(res.nobs),
                         "r2": float(res.rsquared),
                         "coef_const": float(res.params.get("const", np.nan)),
                         "coef_spread10y_bps_l6": float(res.params.get("spread10y_bps_l6", np.nan)),
                         "coef_mortgage30y": float(res.params.get("mortgage30y", np.nan))})
    return pd.DataFrame(rows)


# ----------------------------- Plotting -----------------------------

def make_plots(curve: pd.DataFrame, merged: pd.DataFrame, roll24: pd.DataFrame, xcorrs: Dict[str, pd.DataFrame], outdir: str):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Ratios & spreads
    to_plot = [c for c in curve.columns if c.startswith("ratio_") or c.startswith("spread_")]
    if to_plot:
        fig1 = plt.figure(figsize=(10,5)); ax1 = plt.gca()
        curve[to_plot].plot(ax=ax1)
        ax1.set_title("Muni/UST ratios & spreads (monthly)"); ax1.set_ylabel("ratio / bps")
        plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "ratios_spreads.png"), dpi=140); plt.close(fig1)

    # Mortgage rate & housing starts YoY
    cols = [c for c in ["mortgage30y","housing_starts_yoy","permits_yoy"] if c in merged.columns]
    if cols:
        fig2 = plt.figure(figsize=(10,5)); ax2 = plt.gca()
        merged[cols].plot(ax=ax2)
        ax2.set_title("Mortgage rate & housing YoY"); ax2.set_ylabel("rate / YoY")
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "mortgage_housing.png"), dpi=140); plt.close(fig2)

    # Rolling correlations
    if not roll24.empty:
        fig3 = plt.figure(figsize=(10,6)); ax3 = plt.gca()
        roll24.plot(ax=ax3)
        ax3.axhline(0, linestyle="--", color="k", alpha=0.5)
        ax3.set_title("Rolling 24M correlations: muni metrics vs housing"); ax3.set_ylabel("corr")
        plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "rolling_corr_24m.png"), dpi=140); plt.close(fig3)

    # Scatter: ratio_10y vs starts_yoy
    if {"ratio_10y","housing_starts_yoy"} <= set(merged.columns):
        fig4 = plt.figure(figsize=(6,5)); ax4 = plt.gca()
        x, y = merged["ratio_10y"], merged["housing_starts_yoy"]
        ax4.scatter(x, y, s=12, alpha=0.7)
        ax4.set_title("ratio_10y vs housing_starts_yoy"); ax4.set_xlabel("ratio_10y"); ax4.set_ylabel("starts YoY")
        plt.tight_layout(); fig4.savefig(os.path.join(outdir, "plots", "scatter_ratio10y_vs_starts.png"), dpi=140); plt.close(fig4)

    # Lead/Lag bar summaries
    if xcorrs:
        # Build a compact figure
        fig5 = plt.figure(figsize=(9,5)); ax5 = plt.gca()
        labels, vals = [], []
        for k, df in xcorrs.items():
            labels.append(k + f" (best lag {df.attrs.get('best_lag',0)})")
            vals.append(df.attrs.get("best_corr", np.nan))
        ax5.barh(labels, vals)
        ax5.set_title("Best lead/lag correlations")
        plt.tight_layout(); fig5.savefig(os.path.join(outdir, "plots", "xcorr_best.png"), dpi=140); plt.close(fig5)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Muni Bonds vs Housing: ratios, correlations, lead/lag & OLS")
    ap.add_argument("--yields", dest="yields_file", default=None, help="CSV with muni/UST yields (see header for formats)")
    ap.add_argument("--fred", action="store_true", help="Pull housing/mortgage series from FRED")
    ap.add_argument("--start", type=str, default=None, help="History start date (YYYY-MM-DD) for FRED/resampling")
    ap.add_argument("--end", type=str, default=None, help="End date")
    ap.add_argument("--roll-window", type=int, default=24, help="Rolling correlation window in months")
    ap.add_argument("--max-lag", type=int, default=12, help="Max lead/lag (months) for cross-correlation")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        yields_file=args.yields_file,
        fred=bool(args.fred),
        start=args.start,
        end=args.end,
        roll_window=int(max(6, args.roll_window)),
        max_lag=int(max(1, args.max_lag)),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    # Load user yields (muni/UST)
    curve = pd.DataFrame()
    if cfg.yields_file:
        curve = read_yields_user(cfg.yields_file)
        if curve.empty:
            raise SystemExit("No usable columns found in --yields CSV.")
        curve.to_csv(os.path.join(cfg.outdir, "curve_clean.csv"))

    # FRED (housing & mortgage)
    fred = pull_fred_housing(cfg.start, cfg.end) if cfg.fred else None
    if fred is not None and not fred.empty:
        fred.to_csv(os.path.join(cfg.outdir, "housing_raw.csv"))

    if curve.empty and (fred is None or fred.empty):
        raise SystemExit("Nothing to analyze. Provide --yields and/or --fred.")

    # Build muni metrics (ratios, spreads, z-scores)
    muni = muni_metrics(curve, fred)
    muni.to_csv(os.path.join(cfg.outdir, "muni_metrics.csv"))

    # Merge with housing
    merged = build_merge(muni, fred)
    merged.to_csv(os.path.join(cfg.outdir, "merged_monthly.csv"))

    # Rolling correlations
    roll = rolling_corr(merged, cfg.roll_window)
    if not roll.empty:
        roll.to_csv(os.path.join(cfg.outdir, "rolling_corr_24m.csv"))

    # Lead/Lag cross-correlations
    xcorrs = {}
    if "ratio_10y" in merged.columns:
        if "housing_starts_yoy" in merged.columns:
            xcorrs["ratio10y→startsYoY"] = xcorr_leadlag(merged["ratio_10y"], merged["housing_starts_yoy"], cfg.max_lag)
        if "permits_yoy" in merged.columns:
            xcorrs["ratio10y→permitsYoY"] = xcorr_leadlag(merged["ratio_10y"], merged["permits_yoy"], cfg.max_lag)
    if "spread_10y_bps" in merged.columns and "permits_yoy" in merged.columns:
        xcorrs["spread10y→permitsYoY"] = xcorr_leadlag(merged["spread_10y_bps"], merged["permits_yoy"], cfg.max_lag)

    if xcorrs:
        # Concatenate & save detailed tables
        cat = []
        for k, df in xcorrs.items():
            tmp = df.copy(); tmp["pair"] = k; cat.append(tmp)
        xct = pd.concat(cat, ignore_index=True)
        xct.to_csv(os.path.join(cfg.outdir, "xcorr_table.csv"), index=False)

    # Regressions
    reg = regressions(merged)
    if not reg.empty:
        reg.to_csv(os.path.join(cfg.outdir, "regressions.csv"), index=False)

    # Plots
    if cfg.plot:
        make_plots(muni, merged, roll, xcorrs, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    # Show latest ratios/spreads + housing YoY
    last = merged.dropna(how="all").tail(1)
    if not last.empty:
        dt = last.index[-1].date()
        cols = [c for c in ["ratio_5y","ratio_10y","spread_5y_bps","spread_10y_bps",
                            "housing_starts_yoy","permits_yoy","mortgage30y"] if c in last.columns]
        print(f"\n=== Latest snapshot ({dt}) ===")
        print(last[cols].round(4).to_string(index=False))
    if not reg.empty:
        print("\n=== Regression quick view ===")
        print(reg.round(4).to_string(index=False))

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()