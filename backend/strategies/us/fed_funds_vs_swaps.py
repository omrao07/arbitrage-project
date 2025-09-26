#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fed_funds_vs_swaps.py
#
# Compare the **Fed Funds path** vs **USD OIS (SOFR) swap curve** and spreads.
# You can:
#   1) Pull daily EFFR & SOFR from FRED (optional), compute SOFR–EFFR basis history
#   2) Load a swaps sheet (par OIS rates by tenor) and build a discount/forward curve
#   3) Map the OIS forwards to an implied policy-rate path and compare to EFFR
#   4) Export tidy CSVs and optional plots
#
# Why this design?
# ----------------
# "Real" OIS swap quotes are not reliably free via public APIs. This tool lets you
# bring your own swap quotes (e.g., broker sheet) but still gives a full curve
# toolkit. If you *do* want history, it can at least pull EFFR/SOFR from FRED.
#
# Inputs
# ------
# --swaps FILE (CSV)   Par OIS swap quotes you provide (required for curve work):
#     date, tenor, rate_bps
#   - 'date' can be one or many dates (YYYY-MM-DD). If many, use --asof to choose.
#   - 'tenor' like 1M,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,30Y (case-insensitive)
#   - 'rate_bps' par fixed leg in basis points (SOFR OIS, annualized)
#
# Optional (FRED pull):
# --fred               Pulls EFFR (FEDFUNDS) and SOFR (SOFR) daily from FRED
# --start YYYY-MM-DD   Start date (for FRED pull & exports)
# --end   YYYY-MM-DD   End date
# --fred-api-key KEY   If you have one; otherwise pandas_datareader usually works unauth
#
# Curve options
# -------------
# --asof YYYY-MM-DD    Which date to use from the swaps file (default: max(date))
# --ois-daycount 360|365   Daycount for quoting (default 360)
# --effr-spread-bps X  Offset to translate SOFR forwards to an implied EFFR (default 0)
#                      (Historically SOFR ≈ EFFR - few bps; set your basis)
#
# Outputs
# -------
# outdir/
#   fred_effr_sofr.csv            (if --fred)
#   swaps_clean.csv
#   curve_bootstrap.csv           (zero/discount/forward)
#   implied_policy_path.csv       (monthly fwd overnight & implied EFFR)
#   basis_curve.csv               (par OIS minus avg implied EFFR over each tenor)
#   plots/*.png                   (optional --plot)
#
# Usage examples
# --------------
# python fed_funds_vs_swaps.py --swaps my_swaps.csv --asof 2025-09-05 --plot
# python fed_funds_vs_swaps.py --swaps sheet.csv --fred --start 2023-01-01 --effr-spread-bps 5 --plot
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib pandas_datareader python-dateutil
#
# Notes
# -----
# - Curve bootstrapping here is a *clean* OIS toy model:
#   • Annual-pay fixed vs compounded overnight floating (SOFR)
#   • Linear interpolation on discount factors between pillars
#   • ACT/365f accrual for compounding internally; external quote DC selectable
# - For trading, use professional curve engines; this is research-grade.

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Optional FRED pull
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

from dateutil import parser as dtp
from dateutil.relativedelta import relativedelta


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    swaps_file: str
    fred: bool
    fred_api_key: Optional[str]
    start: Optional[str]
    end: Optional[str]
    asof: Optional[str]
    ois_dc: int
    effr_spread_bps: float
    plot: bool
    outdir: str


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, f"fed_funds_vs_swaps_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def parse_tenor(t: str) -> relativedelta:
    s = str(t).strip().upper()
    if s.endswith("M"):
        return relativedelta(months=int(s[:-1]))
    if s.endswith("Y"):
        return relativedelta(years=int(s[:-1]))
    raise ValueError(f"Unsupported tenor '{t}'. Use like 1M,3M,6M,1Y,2Y,5Y,10Y.")


def tenor_to_months(t: str) -> int:
    d = parse_tenor(t)
    return d.years * 12 + d.months


def read_swaps(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"date","tenor","rate_bps"}
    if not req.issubset(df.columns):
        raise SystemExit(f"swaps CSV must include {sorted(list(req))}")
    df["date"] = pd.to_datetime(df["date"])
    df["tenor"] = df["tenor"].astype(str).str.upper().str.strip()
    df["months"] = df["tenor"].apply(tenor_to_months)
    df["rate"] = pd.to_numeric(df["rate_bps"], errors="coerce") / 1e4  # bps → decimal
    df = df.dropna(subset=["rate"]).sort_values(["date","months"])
    return df


def pull_fred(start: Optional[str], end: Optional[str], api_key: Optional[str]) -> Optional[pd.DataFrame]:
    if pdr is None:
        print("[WARN] pandas_datareader not available; skipping FRED pull.")
        return None
    os.environ.setdefault("FRED_API_KEY", api_key or "")
    s = start or "2018-01-01"
    fred_series = {"EFFR":"FEDFUNDS", "SOFR":"SOFR"}
    out = []
    for name, code in fred_series.items():
        try:
            x = pdr.DataReader(code, "fred", s, end)
            x.columns = [name]
            out.append(x)
        except Exception as e:
            print(f"[WARN] FRED fetch failed for {code}: {e}")
    if not out:
        return None
    df = pd.concat(out, axis=1)
    df.to_csv(os.path.join(cfg.outdir, "fred_effr_sofr.csv"))
    return df


# ----------------------------- Curve math -----------------------------

def yearfrac(d0: pd.Timestamp, d1: pd.Timestamp, basis: str = "ACT/365F") -> float:
    # Minimal ACT/365F
    return (d1 - d0).days / 365.0


def build_schedule(asof: pd.Timestamp, months: int, freq: str = "A") -> List[pd.Timestamp]:
    """
    Build cashflow/payment dates from asof to asof + tenor (months).
    For simplicity use annual fixed pay (freq="A") and monthly overnight comp buckets.
    """
    end = asof + relativedelta(months=months)
    pays = []
    cur = asof + relativedelta(years=1)
    while cur < end or abs((cur - end).days) < 2:
        if cur > end: cur = end
        pays.append(cur)
        cur = cur + relativedelta(years=1)
    if not pays or pays[-1] != end:
        pays.append(end)
    return pays


def bootstrap_ois(asof: pd.Timestamp, pillars: pd.DataFrame, dc_fixed: int = 360
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap discount factors from par OIS quotes.
    Assumptions:
      - Fixed leg: annual-pay, daycount 30/360 or ACT/360 equivalent via yearfracs with dc_fixed
      - Floating leg: compounded overnight equals discount factor ratio (risk-neutral)
      - Linear interpolation on discount factors between pillar maturities
    Returns:
      zeros: period, months, zero_rate (cont), DF
      forwards: monthly forward overnight rates (annualized)
      cashflows: helper table used in the solve (for debugging)
    """
    # Pillars must be sorted by maturity
    p = pillars.sort_values("months").copy().reset_index(drop=True)
    # Discount factor dictionary (start DF = 1)
    D = {asof: 1.0}
    # Container for solved pillar DFs
    solved = []

    # We'll approximate year fraction for fixed coupons using dc_fixed
    def yf(a, b): return (b - a).days / (365.0 if dc_fixed == 365 else 360.0)

    for i, row in p.iterrows():
        Tm = int(row["months"])
        R = float(row["rate"])  # decimal
        pay_dates = build_schedule(asof, Tm, "A")
        # Solve DF at final maturity so that PV_fixed = PV_float (par)
        # PV_float ≈ 1 - DF(T); PV_fixed = R * Σ DF(t_i) * yearfrac_i
        accruals = []
        dfs = []
        prev = asof
        for t in pay_dates:
            acc = yf(prev, t)
            accruals.append(acc)
            # Interpolate DF for t if earlier pillars exist
            if t in D:
                dfs.append(D[t])
            else:
                # Linear in time on log(DF) between last solved and (assume) flat to t
                # If no prior pillar -> simple exponential with guess using R
                if solved:
                    last_T, last_D = solved[-1]
                    last_date = asof + relativedelta(months=last_T)
                    # simple linear on time between last and this payment with flat forward = last implied
                    tau = yearfrac(last_date, t)
                    f_last = -np.log(last_D) / yearfrac(asof, last_date) if yearfrac(asof, last_date) > 0 else R
                    df_guess = last_D * np.exp(-f_last * tau)
                else:
                    df_guess = np.exp(-R * yearfrac(asof, t))
                dfs.append(df_guess)
            prev = t

        # Let X = DF(Tm) be unknown; replace last DF with X and solve:
        # R * Σ_i (DF_i * accr_i) = 1 - X
        # -> R * (S_without_last + X*accr_last) = 1 - X
        S_wo = np.sum(np.array(dfs[:-1]) * np.array(accruals[:-1]))
        accr_last = accruals[-1]

        # Solve linear equation: R*(S_wo + X*accr_last) = 1 - X
        # => R*S_wo + R*accr_last*X = 1 - X
        # => X*(R*accr_last + 1) = 1 - R*S_wo
        numerator = 1.0 - R * S_wo
        denom = (R * accr_last + 1.0)
        X = numerator / denom
        # Guardrails
        X = float(np.clip(X, 1e-6, 1.0))
        D[asof + relativedelta(months=Tm)] = X
        solved.append((Tm, X))

    # Build zeros table
    rows = []
    for m, x in solved:
        t = asof + relativedelta(months=m)
        tau = yearfrac(asof, t)
        z = -np.log(x) / max(tau, 1e-12)  # continuous comp
        rows.append({"date": t, "months": m, "DF": x, "zero_cont": z, "zero_simple": np.expm1(z)/z if z!=0 else 0.0})
    zeros = pd.DataFrame(rows).sort_values("months")

    # Monthly forward overnight from DF(t)
    # f(t_i, t_{i+1}) ≈ -ln(DF_{i+1}/DF_{i}) / Δt  (annualized, cont)
    # Create monthly grid
    if len(solved) == 0:
        raise SystemExit("No pillars solved; check swaps file.")
    max_m = int(max(m for m,_ in solved))
    grid = [asof + relativedelta(months=i) for i in range(1, max_m + 1)]
    df_series = pd.Series(
        {d: (D[d] if d in D else float(np.interp(
            (d - asof).days,
            [ (asof + relativedelta(months=m)).toordinal() - asof.toordinal() for m,_ in solved ],
            [ v for _,v in solved ]
        ))) for d in grid}
    ).sort_index()
    fwd = []
    prev_d, prev_df = asof, 1.0
    for d, cur_df in df_series.items():
        dt = yearfrac(prev_d, d)
        f = -np.log(cur_df / prev_df) / max(dt, 1e-12)
        fwd.append({"date": d, "months": (d.year - asof.year)*12 + d.month - asof.month, "fwd_overnight": f})
        prev_d, prev_df = d, cur_df
    forwards = pd.DataFrame(fwd)

    # Cashflows/debug (optional)
    cash = pd.DataFrame(solved, columns=["months","DF"]).assign(asof=str(asof.date()))
    return zeros, forwards, cash


def implied_policy_from_forwards(forwards: pd.DataFrame, effr_spread_bps: float) -> pd.DataFrame:
    """Translate SOFR forwards to an implied EFFR by adding a constant spread."""
    df = forwards.copy()
    df["implied_effr"] = df["fwd_overnight"] + (effr_spread_bps / 1e4)
    return df[["date","months","fwd_overnight","implied_effr"]]


def par_basis_vs_implied(par: pd.DataFrame, forwards: pd.DataFrame, asof: pd.Timestamp, dc_fixed: int) -> pd.DataFrame:
    """
    Compare each par OIS quote to the *average* implied EFFR over its life
    (simple average of monthly forwards for a quick read).
    """
    res = []
    for _, r in par.iterrows():
        m = int(r["months"])
        window = forwards[forwards["months"] <= m]
        if window.empty: continue
        avg_implied = float(window["implied_effr"].mean())
        res.append({
            "tenor": r["tenor"],
            "months": m,
            "par_ois": float(r["rate"]),
            "avg_implied_effr": avg_implied,
            "basis_par_minus_avg_effr": float(r["rate"] - avg_implied)
        })
    return pd.DataFrame(res).sort_values("months")


# ----------------------------- Plotting -----------------------------

def make_plots(zeros: pd.DataFrame, forwards: pd.DataFrame, basis: pd.DataFrame, fred_df: Optional[pd.DataFrame], outdir: str, asof: pd.Timestamp):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Term structure
    fig1 = plt.figure(figsize=(9,5)); ax = plt.gca()
    ax.plot(zeros["months"]/12.0, 100*zeros["zero_cont"], marker="o", label="Zero (cont, %)")
    ax.plot(forwards["months"]/12.0, 100*forwards["fwd_overnight"], alpha=0.6, label="Monthly fwd ON (%, ann)")
    ax.set_title(f"OIS term structure @ {asof.date()}"); ax.set_xlabel("Years"); ax.set_ylabel("%")
    ax.legend(); plt.tight_layout()
    fig1.savefig(os.path.join(outdir, "plots", "curve_term_structure.png"), dpi=140); plt.close(fig1)

    # Basis curve
    if not basis.empty:
        fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
        ax2.plot(basis["months"]/12.0, 10000*basis["basis_par_minus_avg_effr"], marker="s")
        ax2.axhline(0, linestyle="--", color="k", alpha=0.6)
        ax2.set_title("Par OIS – avg implied EFFR (bps)"); ax2.set_xlabel("Years"); ax2.set_ylabel("bps")
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", "basis_curve.png"), dpi=140); plt.close(fig2)

    # SOFR–EFFR historical basis (if FRED pulled)
    if fred_df is not None and {"SOFR","EFFR"}.issubset(fred_df.columns):
        basis_hist = (fred_df["SOFR"] - fred_df["EFFR"]).dropna()
        fig3 = plt.figure(figsize=(9,5)); ax3 = plt.gca()
        (basis_hist*100).plot(ax=ax3)
        ax3.axhline(0, linestyle="--", color="k", alpha=0.5)
        ax3.set_title("SOFR – EFFR (bps)"); ax3.set_ylabel("bps")
        plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "sofr_effr_basis_history.png"), dpi=140); plt.close(fig3)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Fed Funds vs OIS Swaps: curve toolkit & basis comparisons")
    ap.add_argument("--swaps", required=True, help="CSV with par SOFR OIS quotes: date,tenor,rate_bps")
    ap.add_argument("--fred", action="store_true", help="Pull EFFR & SOFR from FRED for history")
    ap.add_argument("--fred-api-key", type=str, default=None, help="Optional FRED API key")
    ap.add_argument("--start", type=str, default=None, help="Start date for FRED pull/exports")
    ap.add_argument("--end", type=str, default=None, help="End date for FRED pull/exports")
    ap.add_argument("--asof", type=str, default=None, help="Curve as-of date (default: max date in swaps)")
    ap.add_argument("--ois-daycount", type=int, choices=[360,365], default=360, help="Fixed leg DC (quote conv)")
    ap.add_argument("--effr-spread-bps", type=float, default=0.0,
                    help="Add this spread to OIS forwards to proxy implied EFFR")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    global cfg
    cfg = Config(
        swaps_file=args.swaps,
        fred=bool(args.fred),
        fred_api_key=args.fred_api_key,
        start=args.start,
        end=args.end,
        asof=args.asof,
        ois_dc=int(args.ois_daycount),
        effr_spread_bps=float(args.effr_spread_bps),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )
    print(f"[INFO] Writing to: {cfg.outdir}")

    # Swaps
    swaps_all = read_swaps(cfg.swaps_file)
    asof = pd.to_datetime(cfg.asof) if cfg.asof else swaps_all["date"].max()
    swaps = swaps_all[swaps_all["date"] == asof].copy()
    if swaps.empty:
        raise SystemExit(f"No swaps for as-of {asof.date()}")

    swaps.to_csv(os.path.join(cfg.outdir, "swaps_clean.csv"), index=False)

    # FRED (optional)
    fred_df = pull_fred(cfg.start, cfg.end, cfg.fred_api_key) if cfg.fred else None

    # Curve bootstrap
    zeros, forwards, cash = bootstrap_ois(asof, swaps[["months","rate","tenor"]], dc_fixed=cfg.ois_dc)
    zeros.to_csv(os.path.join(cfg.outdir, "curve_bootstrap.csv"), index=False)
    forwards = implied_policy_from_forwards(forwards, cfg.effr_spread_bps)
    forwards.to_csv(os.path.join(cfg.outdir, "implied_policy_path.csv"), index=False)

    # Par OIS vs implied EFFR averages (basis)
    par_basis = par_basis_vs_implied(swaps, forwards, asof, cfg.ois_dc)
    par_basis.to_csv(os.path.join(cfg.outdir, "basis_curve.csv"), index=False)

    # Plots
    if cfg.plot:
        make_plots(zeros, forwards, par_basis, fred_df, cfg.outdir, asof)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== Curve snapshot ===")
    print(zeros.assign(years=zeros["months"]/12.0, zero_pct=lambda x: 100*x["zero_cont"])
              .loc[:, ["months","years","zero_pct","DF"]].round(4).to_string(index=False))

    print("\n=== Basis (par OIS – avg implied EFFR) in bps ===")
    if not par_basis.empty:
        print(par_basis.assign(bps=lambda x: 10000*x["basis_par_minus_avg_effr"])
                    .loc[:, ["tenor","months","bps"]].round(2).to_string(index=False))
    else:
        print("No basis computed (check inputs).")

    print("\nDone. Files in:", cfg.outdir)


if __name__ == "__main__":
    main()