#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# students_loans.py
#
# Student Loans ↔ Consumption & Credit Toolkit
# --------------------------------------------
# What this script does
# - Loads your student-loan time series (balances, payments, delinquency) from CSV
#   and/or pulls common macro series from FRED (best-effort with multiple code tries).
# - Builds per-capita/household burden metrics and a proxy for payment “shock.”
# - Studies relationships vs consumption proxies (retail sales control, PCE),
#   credit tightness (Senior Loan Officer Survey), and card delinquencies (optional).
# - Computes rolling correlations, lead/lag cross-correlations, event windows
#   around the 2020 pause and 2023 restart, and simple OLS regressions.
# - Exports tidy CSVs and optional plots.
#
# Examples
# --------
# python students_loans.py --fred --start 2015-01-01 --plot
#
# python students_loans.py \
#   --loans my_student_loans.csv \
#   --fred \
#   --start 2010-01-01 \
#   --plot
#
# Your CSV formats (flexible)
# ---------------------------
# 1) Wide (monthly or quarterly)
#    date,loan_outstanding,payment_flow_monthly,delinquency_rate
#    2019-12-31,1600000000000,6500000000,0.093
#
# 2) Long
#    date,series,value
#    2019-12-31,loan_outstanding,1.6e12
#    2019-12-31,payment_flow_monthly,6.5e9
#
# Outputs (./artifacts/students_loans/*)
# --------------------------------------
#   raw_user.csv
#   fred_raw.csv
#   merged_monthly.csv
#   burden_metrics.csv
#   rolling_corr_24m.csv
#   xcorr_table.csv
#   regressions.csv
#   events_windows.csv
#   plots/*.png   (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy pandas_datareader matplotlib statsmodels python-dateutil

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
    loans_file: Optional[str]
    fred: bool
    start: Optional[str]
    end: Optional[str]
    roll_window: int
    max_lag: int
    plot: bool
    outdir: str


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "students_loans_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("_", ""))
    except Exception:
        return np.nan


def read_user_loans(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if {"date","series","value"} <= set(df.columns):
        # long → wide
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = df["value"].apply(_num)
        wide = df.pivot_table(index="date", columns="series", values="value", aggfunc="last")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        # numeric coercion
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        wide = df.set_index("date")
    else:
        raise SystemExit("Your CSV must have either (date,series,value) or a 'date' column with wide metrics.")
    # monthly end frequency
    wide = wide.resample("M").last().sort_index()
    # standardize expected keys if someone used variants
    rename = {
        "loans_outstanding": "loan_outstanding",
        "balance": "loan_outstanding",
        "payments": "payment_flow_monthly",
        "payment_flow": "payment_flow_monthly",
        "delinq_rate": "delinquency_rate",
        "delinquency": "delinquency_rate",
    }
    wide.rename(columns=rename, inplace=True)
    return wide


# ----------------------------- FRED pulls (best-effort) -----------------------------

FRED_CANDIDATES = {
    # We’ll try these codes in order; if none works, we skip that concept.
    "student_loans_total": ["SLOAS", "SLOASQ027S", "SLOASQ027SBOG"],    # outstanding student loans (levels; frequency varies)
    "retail_control": ["RRSFS", "RSXFS"],                                 # Advance Retail Sales & Food Services / Retail excl. autos, etc.
    "pce_nominal": ["PCE", "PCECC96"],                                    # Nominal PCE (PCE), real chained (PCECC96)
    "pop": ["CNP16OV", "POP"],                                            # Civilian noninstitutional pop, Total population
    "avg_hourly_earnings": ["CES0500000003", "CES0500000003A"],           # Private sector AHE
    "credit_card_delinquency": ["DRCCLACBS", "DRCCLACBSNOSS"],            # Delinq. rate on credit cards (banks)
    "loan_standards_consumer": ["DRTSCLCC"],                               # SLOOS: Net % tightening consumer credit card loans
}

def pull_first_ok(code_list: List[str], start: Optional[str], end: Optional[str]) -> Optional[pd.Series]:
    if pdr is None:
        return None
    for code in code_list:
        try:
            s = pdr.DataReader(code, "fred", start or "2000-01-01", end)
            s.columns = [code]
            # convert to monthly (last) for non-monthly series
            s = s.resample("M").last()
            return s[code].rename(code)
        except Exception:
            continue
    return None

def pull_fred_bundle(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    frames = []
    for k, codes in FRED_CANDIDATES.items():
        s = pull_first_ok(codes, start, end)
        if s is not None:
            frames.append(s.to_frame())
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    return df


# ----------------------------- Metrics -----------------------------

def build_metrics(user: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """
    Merge user + FRED, compute derived metrics:
      - Per-capita student debt
      - Payment-to-income proxy
      - Real retail control (deflate by AHE*pop, rough) & YoY
      - Payment pause/resume dummies (Mar-2020..Sep-2023 pause, Oct-2023+ resume)
    """
    merged = user.copy() if user is not None else pd.DataFrame()
    if fred is not None and not fred.empty:
        merged = merged.join(fred, how="outer") if not merged.empty else fred.copy()
    if merged.empty:
        raise SystemExit("No data to analyze. Provide --loans and/or --fred.")

    # Fill directionally (no forward-filling financials aggressively)
    merged = merged.sort_index()

    # Per-capita debt
    if "loan_outstanding" in merged.columns and any(c in merged.columns for c in ["CNP16OV","POP"]):
        pop = merged.get("CNP16OV", merged.get("POP"))
        merged["debt_per_capita"] = merged["loan_outstanding"] / pop.replace(0, np.nan)

    # Payment proxy intensity vs wages: payments / (AHE * population * 160 hours * (4.33 weeks))
    if "payment_flow_monthly" in merged.columns and "CES0500000003" in merged.columns or "CES0500000003A" in merged.columns:
        ahe = merged.get("CES0500000003", merged.get("CES0500000003A"))
        if any(c in merged.columns for c in ["CNP16OV","POP"]):
            pop = merged.get("CNP16OV", merged.get("POP"))
            denom = ahe * pop * (160 * 4.33)  # rough monthly total hours for full-time equivalent; proxy
            merged["payment_burden_vs_wages"] = merged["payment_flow_monthly"] / denom.replace(0, np.nan)

    # Real retail control proxy: use RRSFS (already nominal index-like) or RSXFS; YoY
    # PCE too: we compute YoY for both
    for nm in ["RRSFS", "RSXFS", "PCE", "PCECC96"]:
        if nm in merged.columns:
            merged[nm + "_yoy"] = merged[nm].pct_change(12)

    # Student-loan YoY and Δ
    for nm in ["loan_outstanding", "SLOAS", "SLOASQ027S", "SLOASQ027SBOG"]:
        if nm in merged.columns:
            merged[nm + "_yoy"] = merged[nm].pct_change(12)
            merged[nm + "_mom"] = merged[nm].pct_change(1)

    # Pause/resume windows
    idx = merged.index
    merged["pause_dummy"] = ((idx >= pd.Timestamp("2020-03-31")) & (idx <= pd.Timestamp("2023-09-30"))).astype(int)
    merged["resume_dummy"] = (idx >= pd.Timestamp("2023-10-31")).astype(int)

    return merged


def rolling_corr(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    # Correlate student loan metrics with consumption/credit proxies
    left = []
    right = []
    # left metrics (debt/per-capita, payments burden, Δ balances)
    for c in ["debt_per_capita","payment_burden_vs_wages",
              "loan_outstanding_mom","SLOAS_mom","SLOASQ027S_mom","SLOASQ027SBOG_mom"]:
        if c in df.columns: left.append(c)
    # right metrics (retail/PCE YoY, credit delinq, standards)
    for c in ["RRSFS_yoy","RSXFS_yoy","PCE_yoy","PCECC96_yoy","DRCCLACBS","DRCCLACBSNOSS","DRTSCLCC"]:
        if c in df.columns: right.append(c)
    out = {}
    for l in left:
        for r in right:
            name = f"corr_{l}_vs_{r}_{window}m"
            out[name] = df[l].rolling(window).corr(df[r])
    return pd.DataFrame(out).dropna(how="all")


def xcorr_leadlag(a: pd.Series, b: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    a, b = a.align(b, join="inner")
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k > 0:
            s1, s2 = a.iloc[:-k], b.iloc[k:]
        elif k < 0:
            s1, s2 = a.iloc[-k:], b.iloc[:k]
        else:
            s1, s2 = a, b
        rows.append({"lag": k, "corr": s1.corr(s2)})
    df = pd.DataFrame(rows)
    if df["corr"].notna().any():
        best = df.iloc[df["corr"].abs().idxmax()]
        df.attrs["best_lag"] = int(best["lag"])
        df.attrs["best_corr"] = float(best["corr"])
    return df


def regressions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example OLS:
      - Retail control YoY ~ Δ loan outstanding (lagged) + resume_dummy
      - PCE YoY ~ payment_burden_vs_wages (lagged) + pause_dummy
    """
    rows = []
    # Retail control (prefer RRSFS_yoy, else RSXFS_yoy)
    yname = "RRSFS_yoy" if "RRSFS_yoy" in df.columns else ("RSXFS_yoy" if "RSXFS_yoy" in df.columns else None)
    if yname is not None:
        X = pd.concat([
            df.get("loan_outstanding_mom").shift(3),  # 3m lag
            df.get("resume_dummy")
        ], axis=1)
        X.columns = ["dLoans_m3", "resume"]
        Y = df[yname]
        dat = pd.concat([Y, X], axis=1).dropna()
        if len(dat) > 36:
            res = sm.OLS(dat[yname], sm.add_constant(dat[["dLoans_m3","resume"]])).fit()
            rows.append({
                "model": f"{yname} ~ dLoans_m3 + resume",
                "nobs": int(res.nobs), "r2": float(res.rsquared),
                "b_dLoans_m3": float(res.params.get("dLoans_m3", np.nan)),
                "t_dLoans_m3": float(res.tvalues.get("dLoans_m3", np.nan)),
                "b_resume": float(res.params.get("resume", np.nan)),
                "t_resume": float(res.tvalues.get("resume", np.nan)),
            })
    # PCE vs payment burden
    if "PCE_yoy" in df.columns and "payment_burden_vs_wages" in df.columns:
        X = pd.concat([df["payment_burden_vs_wages"].shift(3), df.get("pause_dummy")], axis=1)
        X.columns = ["burden_l3","pause"]
        dat = pd.concat([df["PCE_yoy"], X], axis=1).dropna()
        if len(dat) > 36:
            res = sm.OLS(dat["PCE_yoy"], sm.add_constant(dat[["burden_l3","pause"]])).fit()
            rows.append({
                "model": "PCE_yoy ~ burden_l3 + pause",
                "nobs": int(res.nobs), "r2": float(res.rsquared),
                "b_burden_l3": float(res.params.get("burden_l3", np.nan)),
                "t_burden_l3": float(res.tvalues.get("burden_l3", np.nan)),
                "b_pause": float(res.params.get("pause", np.nan)),
                "t_pause": float(res.tvalues.get("pause", np.nan)),
            })
    return pd.DataFrame(rows)


# ----------------------------- Events study -----------------------------

def event_windows(df: pd.DataFrame, series: str, pre: int = 12, post: int = 18,
                  events: List[pd.Timestamp] = [pd.Timestamp("2020-03-31"), pd.Timestamp("2023-10-31")]) -> pd.DataFrame:
    """
    Build event-relative path (monthly) for chosen series around pause and restart.
    """
    if series not in df.columns:
        return pd.DataFrame()
    s = df[series].dropna()
    rows = []
    for ev in events:
        if ev not in s.index:
            # snap to closest
            try:
                idx = s.index[(s.index - ev).abs().argmin()]
            except Exception:
                continue
        else:
            idx = ev
        center = s.index.get_loc(idx)
        # Build relative window based on integer positions
        pos = s.index.get_indexer([idx])[0]
        start = max(0, pos - pre); end = min(len(s)-1, pos + post)
        win = s.iloc[start:end+1]
        rel = np.arange(start - pos, end - pos + 1)
        rows.append(pd.DataFrame({"event": ev.date(), "tau_m": rel, series: win.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------- Plotting -----------------------------

def make_plots(merged: pd.DataFrame, roll24: pd.DataFrame, xcorrs: Dict[str, pd.DataFrame],
               events_tbl: pd.DataFrame, outdir: str):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Key levels
    fig1 = plt.figure(figsize=(10,5)); ax1 = plt.gca()
    cols = [c for c in ["loan_outstanding","SLOAS","PCE","RRSFS","RSXFS"] if c in merged.columns]
    if cols:
        merged[cols].plot(ax=ax1)
        ax1.set_title("Levels: Student loans & consumption proxies")
        plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "levels.png"), dpi=140); plt.close(fig1)

    # Burden metrics
    if "debt_per_capita" in merged.columns or "payment_burden_vs_wages" in merged.columns:
        fig2 = plt.figure(figsize=(10,5)); ax2 = plt.gca()
        sub = merged[[c for c in ["debt_per_capita","payment_burden_vs_wages"] if c in merged.columns]]
        sub.plot(ax=ax2)
        ax2.axvspan(pd.Timestamp("2020-03-31"), pd.Timestamp("2023-09-30"), color="grey", alpha=0.2, label="Pause")
        ax2.axvline(pd.Timestamp("2023-10-31"), linestyle="--", alpha=0.6, label="Resume")
        ax2.set_title("Burden metrics & policy window"); ax2.legend()
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "burden.png"), dpi=140); plt.close(fig2)

    # Rolling correlations
    if not roll24.empty:
        fig3 = plt.figure(figsize=(10,5)); ax3 = plt.gca()
        roll24.plot(ax=ax3)
        ax3.axhline(0, linestyle="--", alpha=0.6)
        ax3.set_title("Rolling 24M correlations: loans vs consumption/credit")
        plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "rolling_corr.png"), dpi=140); plt.close(fig3)

    # Event windows
    if not events_tbl.empty:
        for ev in events_tbl["event"].unique():
            fig4 = plt.figure(figsize=(9,4)); ax4 = plt.gca()
            sub = events_tbl[events_tbl["event"] == ev].set_index("tau_m")
            sub.plot(ax=ax4)
            ax4.axvline(0, linestyle="--", alpha=0.6)
            ax4.set_title(f"Event path around {ev}")
            plt.tight_layout(); fig4.savefig(os.path.join(outdir, "plots", f"event_{ev}.png"), dpi=140); plt.close(fig4)

    # Lead/Lag summary bars
    if xcorrs:
        fig5 = plt.figure(figsize=(9,5)); ax5 = plt.gca()
        labels, vals = [], []
        for k, df in xcorrs.items():
            labels.append(k + f" (best {df.attrs.get('best_lag',0)})")
            vals.append(df.attrs.get("best_corr", np.nan))
        ax5.barh(labels, vals)
        ax5.set_title("Best lead/lag correlations")
        plt.tight_layout(); fig5.savefig(os.path.join(outdir, "plots", "xcorr_best.png"), dpi=140); plt.close(fig5)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Student loans vs consumption & credit: burdens, correlations, events, OLS")
    ap.add_argument("--loans", dest="loans_file", default=None, help="CSV with loan_outstanding/payment_flow_monthly/…")
    ap.add_argument("--fred", action="store_true", help="Pull macro series from FRED (best-effort)")
    ap.add_argument("--start", type=str, default=None, help="History start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date")
    ap.add_argument("--roll-window", type=int, default=24, help="Rolling correlation window (months)")
    ap.add_argument("--max-lag", type=int, default=12, help="Max lead/lag months for xcorr")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        loans_file=args.loans_file,
        fred=bool(args.fred),
        start=args.start,
        end=args.end,
        roll_window=int(max(12, args.roll_window)),
        max_lag=int(max(3, args.max_lag)),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    user = pd.DataFrame()
    if cfg.loans_file:
        user = read_user_loans(cfg.loans_file)
        if not user.empty:
            user.to_csv(os.path.join(cfg.outdir, "raw_user.csv"))

    fred = pd.DataFrame()
    if cfg.fred:
        fred = pull_fred_bundle(cfg.start, cfg.end)
        if not fred.empty:
            fred.to_csv(os.path.join(cfg.outdir, "fred_raw.csv"))

    merged = build_metrics(user if not user.empty else None, fred if not fred.empty else None)
    merged = merged.loc[(merged.index >= (cfg.start or "2000-01-01")) & (merged.index <= (cfg.end or merged.index.max()))]
    merged.to_csv(os.path.join(cfg.outdir, "merged_monthly.csv"))

    # Build burden snapshot table
    keep_cols = [c for c in ["loan_outstanding","debt_per_capita","payment_flow_monthly","payment_burden_vs_wages",
                             "RRSFS_yoy","RSXFS_yoy","PCE_yoy","PCECC96_yoy","DRCCLACBS","DRTSCLCC",
                             "pause_dummy","resume_dummy"] if c in merged.columns]
    merged[keep_cols].to_csv(os.path.join(cfg.outdir, "burden_metrics.csv"))

    # Rolling correlations
    roll = rolling_corr(merged, cfg.roll_window)
    if not roll.empty:
        roll.to_csv(os.path.join(cfg.outdir, "rolling_corr_24m.csv"))

    # Lead/Lag vs consumption YoY
    xcorrs = {}
    # choose a loans delta
    L = None
    for nm in ["loan_outstanding_mom","SLOAS_mom","SLOASQ027S_mom","SLOASQ027SBOG_mom"]:
        if nm in merged.columns:
            L = merged[nm]; break
    if L is not None:
        for y in ["RRSFS_yoy","RSXFS_yoy","PCE_yoy","PCECC96_yoy"]:
            if y in merged.columns:
                xcorrs[f"{L.name}→{y}"] = xcorr_leadlag(L, merged[y], cfg.max_lag)
    if xcorrs:
        cat = []
        for k, df in xcorrs.items():
            tmp = df.copy(); tmp["pair"] = k; cat.append(tmp)
        pd.concat(cat, ignore_index=True).to_csv(os.path.join(cfg.outdir, "xcorr_table.csv"), index=False)

    # Regressions
    reg = regressions(merged)
    if not reg.empty:
        reg.to_csv(os.path.join(cfg.outdir, "regressions.csv"), index=False)

    # Event study around pause/resume for consumption YoY
    ev_tbls = []
    tgt = "RRSFS_yoy" if "RRSFS_yoy" in merged.columns else ("PCE_yoy" if "PCE_yoy" in merged.columns else None)
    if tgt:
        ev_tbl = event_windows(merged, tgt, pre=12, post=18)
        if not ev_tbl.empty:
            ev_tbl.to_csv(os.path.join(cfg.outdir, "events_windows.csv"), index=False)
            ev_tbls.append(ev_tbl)
    # Plots
    if cfg.plot:
        make_plots(merged, roll, xcorrs, (pd.concat(ev_tbls, ignore_index=True) if ev_tbls else pd.DataFrame()), cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== Latest snapshot ===")
    last = merged.dropna(how="all").tail(1)
    if not last.empty:
        dt = last.index[-1].date()
        cols = [c for c in ["loan_outstanding","debt_per_capita","payment_flow_monthly","payment_burden_vs_wages",
                            "RRSFS_yoy","PCE_yoy"] if c in last.columns]
        print(f"As of {dt}:")
        print(last[cols].round(4).to_string(index=False))

    if not reg.empty:
        print("\n=== OLS quick view ===")
        print(reg.round(4).to_string(index=False))

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()