#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipo_subscription.py — Live IPO book analytics (India): subs ratios, pace-forecast & retail allotment odds
--------------------------------------------------------------------------------------------------------

What this does
==============
Given an IPO details file and a live/cumulative bids file (by investor category),
this script:

1) Normalizes inputs (dates, categories, lot size, reserved shares) to a tidy panel.
2) Computes **subscription ratios** (QIB, NII/HNI, Retail, Employee, Overall) over time.
3) Fits a simple **pace model** to forecast end-of-issue subscription for each category:
     log(Sub_t) = α·log(t/T) + log(Sub_final)  → estimate α & Sub_final
   • Robust to sparse/zero early prints (category-specific priors for α).
   • 80% intervals via OLS variance on the intercept (log-space).
4) Estimates **retail allotment probability** and expected lots per application,
   given lots available and (optional) applications. If not provided, it infers
   apps from retail bids / lot_size.
5) Exports tidy CSVs + JSON summary.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--ipo ipo.csv            REQUIRED (issue details)
  Columns (any subset; extras ignored). Script will map by fuzzy names:
    issue, name, symbol
    open_date, close_date
    price_low, price_high  (or price_band_low/high)
    lot_size (or lot)
    total_shares (or total_qty/issue_size_shares)  [if absent, use sum of category shares]
    qib_pct, nii_pct, retail_pct, emp_pct          [% of net offer]  (optional)
    qib_shares, nii_shares, retail_shares, emp_shares (optional, overrides %)
    anchor_shares (optional; excluded from live subs)
    max_lots (optional; for per-application cap; default=1 for retail odds calc)

--book book.csv          REQUIRED (cumulative bids by timestamp)
  Columns (min):
    timestamp (or date_time/time/date)  — any parseable datetime
    qib_bids, nii_bids, retail_bids [, emp_bids]    (in **shares**, cumulative)
  Notes:
    • Use cumulative totals at each print; the script deduplicates identical timestamps.
    • If only some categories are present, others are treated as NaN.

--history history.csv    OPTIONAL pace hints (category, t_frac, sub_frac)
  Columns:
    category (QIB/NII/Retail/Emp), t_frac (0..1), sub_frac (0..1)
  Used to seed priors for α by simple regression across past deals; safe to omit.

Key CLI
-------
--now "YYYY-MM-DD HH:MM"    As-of time (default: latest in book.csv)
--outdir out_ipo            Output directory (default)
--retail_apps 500000        Override number of retail applications (PANs) for odds calc
--alpha_qib 1.8             Prior α for QIB (steeper late surge)
--alpha_nii 1.2             Prior α for NII
--alpha_ret 1.0             Prior α for Retail
--alpha_emp 1.0             Prior α for Employee
--ci 0.80                   Forecast credible interval level (default 80%)

Outputs
-------
- panel.csv                 Time panel: bids, t_frac, subs by category, rolling forecast of final subs
- forecast_final.csv        Per-category forecast: Sub_final (point & interval), α estimate, fit stats
- retail_allotment.csv      Lots math, inferred apps, probability of ≥1 lot, expected lots/app
- summary.json              Headline snapshot (now vs forecast), odds, config echo
- config.json               Run configuration for reproducibility

Method notes
------------
• Subscription ratio = cumulative_bids_shares / reserved_shares (ex-anchor).
• Pace model is intentionally simple and transparent. If book is 0 for long,
  priors are used with conservative uplift; forecasts are never < current subs.
• Retail allotment (SEBI practice):
    - If applications <= lots_available → everyone gets pro-rata (≥1 lot possible).
    - If applications  > lots_available → lottery for **one lot** per winning app.
  We report:
    prob_one_lot ≈ min(1, lots_available / applications)
    expected_lots_per_app ≈ min(max_lots, (reserved_shares / (applications*lot_size)))
  (Use --retail_apps to override inference.)

DISCLAIMER
----------
For research/monitoring only. Uses simplifying assumptions; verify before financial decisions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# ----------------------------- utilities -----------------------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return first matching column via exact/lower/contains."""
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        lc = cand.lower()
        if lc in low: return low[lc]
    for cand in cands:
        t = cand.lower()
        for c in df.columns:
            if t in str(c).lower(): return c
    return None

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def safe_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")

def clip_nonneg(a: float) -> float:
    return float(a) if (a==a and a>=0) else 0.0

# ----------------------------- loaders -----------------------------

def load_ipo(path: str) -> Dict:
    df = pd.read_csv(path)
    # use first row
    r = df.iloc[0]

    def pick(*names, default=None):
        for nm in names:
            c = ncol(df, nm)
            if c: return r[c]
        return default

    issue = pick("issue","name","symbol","company","issuer","ipo")
    open_date = to_dt(pd.Series([pick("open_date","open","start_date","bidding_open")])).iloc[0]
    close_date= to_dt(pd.Series([pick("close_date","close","end_date","bidding_close")])).iloc[0]

    price_low  = safe_num(pd.Series([pick("price_low","price_band_low","lower_price","lower_band","low")])).iloc[0]
    price_high = safe_num(pd.Series([pick("price_high","price_band_high","upper_price","upper_band","high")])).iloc[0]
    lot_size   = int(safe_num(pd.Series([pick("lot_size","lot","market_lot","lotshares")])).fillna(0).iloc[0] or 0)
    max_lots   = int(safe_num(pd.Series([pick("max_lots","max_app_lots","max_lot_per_app")])).fillna(1).iloc[0] or 1)

    total_shares = safe_num(pd.Series([pick("total_shares","issue_size_shares","total_qty","net_offer_shares")])).iloc[0]
    anchor_shares= safe_num(pd.Series([pick("anchor_shares","qib_anchor_shares","anchor")])).iloc[0]

    # category shares / percents
    pct = {
        "QIB": safe_num(pd.Series([pick("qib_pct","qib_percent")])).iloc[0],
        "NII": safe_num(pd.Series([pick("nii_pct","hni_pct","nii_percent","hni_percent")])).iloc[0],
        "Retail": safe_num(pd.Series([pick("retail_pct","r_ii_pct","retail_percent")])).iloc[0],
        "Employee": safe_num(pd.Series([pick("emp_pct","employee_pct")])).iloc[0]
    }
    shares = {
        "QIB": safe_num(pd.Series([pick("qib_shares","qib_qty")])).iloc[0],
        "NII": safe_num(pd.Series([pick("nii_shares","hni_shares","nii_qty","hni_qty")])).iloc[0],
        "Retail": safe_num(pd.Series([pick("retail_shares","retail_qty")])).iloc[0],
        "Employee": safe_num(pd.Series([pick("emp_shares","employee_shares","emp_qty")])).iloc[0]
    }

    # Compute reserved shares by category (exclude anchors)
    reserved: Dict[str, float] = {}
    net_offer = (total_shares or 0) - (anchor_shares or 0)
    if not pd.isna(shares["QIB"]) or not pd.isna(shares["NII"]) or not pd.isna(shares["Retail"]) or not pd.isna(shares["Employee"]):
        for k, v in shares.items():
            reserved[k] = clip_nonneg(v)
        # If total missing, sum categories
        if not total_shares or pd.isna(total_shares):
            total_shares = float(np.nansum(list(reserved.values())) + (anchor_shares or 0))
            net_offer = (total_shares or 0) - (anchor_shares or 0)
    elif net_offer and any([not pd.isna(v) for v in pct.values()]):
        for k, p in pct.items():
            reserved[k] = clip_nonneg((p or 0)/100.0 * net_offer)
    else:
        raise ValueError("ipo.csv is missing both category shares and percents, and total/net offer is unclear.")

    # Sanity: ensure Retail exists
    for k in ["QIB","NII","Retail","Employee"]:
        reserved.setdefault(k, np.nan)

    out = {
        "issue": str(issue) if issue is not None else "",
        "open_date": pd.Timestamp(open_date) if not pd.isna(open_date) else None,
        "close_date": pd.Timestamp(close_date) if not pd.isna(close_date) else None,
        "price_low": float(price_low) if price_low==price_low else None,
        "price_high": float(price_high) if price_high==price_high else None,
        "lot_size": int(lot_size),
        "max_lots": int(max_lots),
        "anchor_shares": float(anchor_shares) if anchor_shares==anchor_shares else 0.0,
        "total_shares": float(total_shares) if total_shares==total_shares else None,
        "reserved": reserved,
        "net_offer": float(net_offer) if net_offer==net_offer else float(np.nansum([v for v in reserved.values() if v==v]))
    }
    if not out["open_date"] or not out["close_date"]:
        raise ValueError("ipo.csv must include open_date and close_date.")
    if out["lot_size"] <= 0:
        raise ValueError("ipo.csv must include a positive lot_size.")
    return out

def load_book(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = ncol(df, "timestamp","date_time","time","datetime","date")
    if not ts: raise ValueError("book.csv needs a 'timestamp' column (timestamp/date_time/time).")
    df = df.rename(columns={ts:"timestamp"})
    df["timestamp"] = to_dt(df["timestamp"])

    def pick(name: str) -> Optional[str]:
        return ncol(df, name, name+"_bids", name+"_shares", name+"_cum", name+"_bid", name+"_applied", name+"_apps")

    mapping = {
        "QIB": pick("qib") or pick("qii") or pick("qib_fii") or pick("qib_total"),
        "NII": pick("nii") or pick("hni") or pick("bnii") or pick("snii"),
        "Retail": pick("retail") or pick("rii"),
        "Employee": pick("emp") or pick("employee")
    }
    keep = ["timestamp"]
    for k, c in mapping.items():
        if c:
            df[c] = safe_num(df[c])
            keep.append(c)
    df = df[keep].dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # rename to canonical
    for k, c in mapping.items():
        if c: df = df.rename(columns={c: k})
        else: df[k] = np.nan

    return df.reset_index(drop=True)

# ----------------------------- pace model -----------------------------

@dataclass
class PaceFit:
    alpha: float
    s_final: float
    se_alpha: float
    se_log_sfinal: float
    n: int

def fit_power_pace(t_frac: pd.Series, sub: pd.Series, alpha_prior: float) -> PaceFit:
    """
    Fit: log(sub_t + eps) = a * log(t_frac) + b;  Sub_final = exp(b)
    Only use t where 0 < t_frac < 1 and sub_t > 0.
    If insufficient points, return prior-based uplift.
    """
    df = pd.DataFrame({"t": t_frac, "s": sub}).dropna()
    df = df[(df["t"] > 0) & (df["t"] < 1) & (df["s"] > 0)]
    if df.shape[0] < 3:
        # fallback: anchor alpha to prior; infer s_final from latest
        if sub.dropna().empty or t_frac.dropna().empty:
            return PaceFit(alpha=alpha_prior, s_final=float(sub.dropna().iloc[-1] if not sub.dropna().empty else 0.0),
                           se_alpha=np.nan, se_log_sfinal=np.nan, n=0)
        t_now = float(t_frac.dropna().iloc[-1])
        s_now = float(sub.dropna().iloc[-1])
        f_now = max(t_now, 1e-3)**alpha_prior
        s_final = s_now / f_now
        return PaceFit(alpha=alpha_prior, s_final=float(max(s_final, s_now)), se_alpha=np.nan, se_log_sfinal=np.nan, n=df.shape[0])

    x = np.log(df["t"].values.reshape(-1,1))
    y = np.log(df["s"].values.reshape(-1,1))
    X = np.column_stack([np.ones_like(x), x])
    XtX = X.T @ X
    XtY = X.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ XtY
    resid = y - X @ beta
    n = X.shape[0]; k = X.shape[1]
    dof = max(1, n-k)
    sigma2 = float((resid.T @ resid)/dof)
    cov = XtX_inv * sigma2
    a = float(beta[1,0]); b = float(beta[0,0])
    se_a = float(np.sqrt(cov[1,1]))
    se_b = float(np.sqrt(cov[0,0]))
    # shrink alpha a bit toward prior if unstable
    if not np.isfinite(se_a) or se_a > 1.0:
        a = (0.7*a + 0.3*alpha_prior)

    s_final = float(np.exp(b))
    # never forecast below latest sub
    s_final = float(max(s_final, float(sub.dropna().iloc[-1])))
    return PaceFit(alpha=a, s_final=s_final, se_alpha=se_a, se_log_sfinal=se_b, n=n)

def ci_from_logpoint(log_mean: float, se: float, z: float=1.2816) -> Tuple[float,float]:
    """Return (low, high) on exp-scale for log-mean ± z*se."""
    if not np.isfinite(se): return (float(np.exp(log_mean)), float(np.exp(log_mean)))
    return float(np.exp(log_mean - z*se)), float(np.exp(log_mean + z*se))

# ----------------------------- retail allotment math -----------------------------

def retail_allotment_math(reserved_shares: float, lot_size: int, max_lots: int,
                          bids_shares_now: float, apps_override: Optional[int]=None) -> Dict:
    lots_available = int(np.floor((reserved_shares or 0) / max(1, lot_size)))
    # infer applications (PANs). Assume avg lots/app ≈ 1 when oversubscribed; otherwise infer from ratio.
    if apps_override and apps_override > 0:
        apps = int(apps_override)
    else:
        avg_lots_per_app = 1.0
        if bids_shares_now and lot_size>0:
            est_apps = float(bids_shares_now) / float(lot_size * avg_lots_per_app)
        else:
            est_apps = 0.0
        apps = int(np.ceil(est_apps))
    subs_now = (bids_shares_now or 0) / (reserved_shares or np.nan)
    # Lottery probability if demand > supply
    if apps <= 0 or lots_available <= 0:
        prob_one_lot = 0.0
    else:
        prob_one_lot = min(1.0, lots_available / apps)
    # Expected lots per app (pro-rata if undersubscribed)
    if apps <= 0: exp_lots = 0.0
    else:
        exp_lots = min(max_lots, (reserved_shares or 0) / (apps * lot_size))
    return {
        "lots_available": lots_available,
        "apps_inferred": apps,
        "subs_now": float(subs_now) if subs_now==subs_now else None,
        "prob_one_lot": float(prob_one_lot),
        "expected_lots_per_app": float(exp_lots)
    }

# ----------------------------- orchestration -----------------------------

@dataclass
class Config:
    ipo: str
    book: str
    history: Optional[str]
    now: Optional[str]
    outdir: str
    retail_apps: Optional[int]
    alpha_qib: float
    alpha_nii: float
    alpha_ret: float
    alpha_emp: float
    ci: float

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IPO subscription tracker & forecast (India)")
    ap.add_argument("--ipo", required=True, help="IPO details CSV")
    ap.add_argument("--book", required=True, help="Live/cumulative bids CSV (timestamp + category bids in shares)")
    ap.add_argument("--history", default="", help="Optional pace history CSV (category,t_frac,sub_frac)")
    ap.add_argument("--now", default="", help="As-of time (YYYY-MM-DD HH:MM); default latest in book")
    ap.add_argument("--outdir", default="out_ipo")
    ap.add_argument("--retail_apps", type=int, default=0, help="Override retail applications (PANs)")
    ap.add_argument("--alpha_qib", type=float, default=1.8)
    ap.add_argument("--alpha_nii", type=float, default=1.2)
    ap.add_argument("--alpha_ret", type=float, default=1.0)
    ap.add_argument("--alpha_emp", type=float, default=1.0)
    ap.add_argument("--ci", type=float, default=0.80, help="Forecast interval level (default 0.80)")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    IPO = load_ipo(args.ipo)
    BOOK = load_book(args.book)
    if BOOK.empty:
        raise ValueError("book.csv appears empty after parsing.")
    # As-of cutoff
    t_now = (pd.to_datetime(args.now) if args.now else BOOK["timestamp"].max())
    BOOK = BOOK[BOOK["timestamp"] <= t_now]
    if BOOK.empty:
        raise ValueError("No book prints at or before --now.")
    # Time fraction in [0,1]
    T0, T1 = IPO["open_date"], IPO["close_date"]
    total_span = (T1 - T0).total_seconds()
    BOOK["t_frac"] = (BOOK["timestamp"] - T0).dt.total_seconds() / max(1.0, total_span)
    BOOK["t_frac"] = BOOK["t_frac"].clip(lower=0, upper=1)

    # Reserved shares
    res = IPO["reserved"]
    res_Q = res.get("QIB", np.nan)
    res_N = res.get("NII", np.nan)
    res_R = res.get("Retail", np.nan)
    res_E = res.get("Employee", np.nan)

    # Subscription ratios over time
    for k, denom in [("QIB", res_Q), ("NII", res_N), ("Retail", res_R), ("Employee", res_E)]:
        if k in BOOK.columns and denom==denom and denom>0:
            BOOK[f"sub_{k}"] = BOOK[k] / denom
        else:
            BOOK[f"sub_{k}"] = np.nan
    # Overall (exclude anchors): sum bids / net_offer
    BOOK["bids_total_known"] = BOOK[["QIB","NII","Retail","Employee"]].sum(axis=1, skipna=True)
    BOOK["sub_Overall"] = BOOK["bids_total_known"] / max(IPO["net_offer"] or np.nan, 1)

    # Pace model per category
    priors = {"QIB": args.alpha_qib, "NII": args.alpha_nii, "Retail": args.alpha_ret, "Employee": args.alpha_emp}
    fits: Dict[str,PaceFit] = {}
    forecasts: Dict[str,Dict] = {}
    z = 1.2816 if args.ci >= 0.80 else 1.6449  # crude mapping 80%/90%
    for cat in ["QIB","NII","Retail","Employee"]:
        sub = BOOK[f"sub_{cat}"]
        if sub.dropna().empty:
            continue
        fit = fit_power_pace(BOOK["t_frac"], sub, priors[cat])
        fits[cat] = fit
        lo, hi = ci_from_logpoint(np.log(max(fit.s_final, 1e-12)), fit.se_log_sfinal if fit.se_log_sfinal==fit.se_log_sfinal else 0.35, z=z)
        forecasts[cat] = {
            "alpha": fit.alpha, "alpha_se": fit.se_alpha, "n_points": fit.n,
            "sub_final_point": fit.s_final,
            "sub_final_lo": lo, "sub_final_hi": hi
        }
        # Attach running forecast to panel
        f_now = BOOK["t_frac"].replace(0, 1e-3)**max(1e-6, fit.alpha)
        BOOK[f"sub_{cat}_final_forecast"] = BOOK[f"sub_{cat}"] / f_now

    # Retail allotment odds (as-of now)
    asof = BOOK.iloc[-1]
    retail_odds = {}
    if res_R==res_R and res_R>0:
        retail_odds = retail_allotment_math(
            reserved_shares=res_R,
            lot_size=int(IPO["lot_size"]),
            max_lots=int(IPO["max_lots"] or 1),
            bids_shares_now=float(asof["Retail"]) if "Retail" in BOOK.columns else 0.0,
            apps_override=int(args.retail_apps) if args.retail_apps and args.retail_apps>0 else None
        )
    # Save panel
    panel_cols = ["timestamp","t_frac","QIB","NII","Retail","Employee",
                  "sub_QIB","sub_NII","sub_Retail","sub_Employee","sub_Overall"]
    for cat in ["QIB","NII","Retail","Employee"]:
        c = f"sub_{cat}_final_forecast"
        if c in BOOK.columns: panel_cols.append(c)
    BOOK[panel_cols].to_csv(outdir/"panel.csv", index=False)

    # Forecast table
    rows = []
    for cat, f in forecasts.items():
        rows.append({
            "category": cat,
            "sub_now": float(asof.get(f"sub_{cat}", np.nan)) if f"sub_{cat}" in BOOK.columns else np.nan,
            **f
        })
    fc_df = pd.DataFrame(rows)
    if not fc_df.empty:
        fc_df.to_csv(outdir/"forecast_final.csv", index=False)

    # Retail allotment export
    if retail_odds:
        pd.DataFrame([{
            "lot_size": int(IPO["lot_size"]),
            "max_lots": int(IPO["max_lots"] or 1),
            **retail_odds
        }]).to_csv(outdir/"retail_allotment.csv", index=False)

    # Summary JSON
    latest = {
        "timestamp": str(asof["timestamp"]),
        "t_frac": float(asof["t_frac"]),
        "subs_now": {
            "QIB": float(asof.get("sub_QIB")) if asof.get("sub_QIB")==asof.get("sub_QIB") else None,
            "NII": float(asof.get("sub_NII")) if asof.get("sub_NII")==asof.get("sub_NII") else None,
            "Retail": float(asof.get("sub_Retail")) if asof.get("sub_Retail")==asof.get("sub_Retail") else None,
            "Employee": float(asof.get("sub_Employee")) if asof.get("sub_Employee")==asof.get("sub_Employee") else None,
            "Overall": float(asof.get("sub_Overall")) if asof.get("sub_Overall")==asof.get("sub_Overall") else None
        }
    }
    summary = {
        "issue": IPO["issue"],
        "window": {"open": str(IPO["open_date"]), "close": str(IPO["close_date"])},
        "price_band": {"low": IPO["price_low"], "high": IPO["price_high"]},
        "lot_size": int(IPO["lot_size"]),
        "net_offer_shares_ex_anchor": float(IPO["net_offer"]),
        "reserved_shares": {k: float(v) if v==v else None for k,v in IPO["reserved"].items()},
        "as_of": latest,
        "forecasts": forecasts,
        "retail_allotment": retail_odds,
        "config": {
            "alpha_priors": {"QIB": args.alpha_qib, "NII": args.alpha_nii, "Retail": args.alpha_ret, "Employee": args.alpha_emp},
            "ci_level": args.ci,
            "retail_apps_override": int(args.retail_apps or 0)
        }
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        ipo=args.ipo, book=args.book, history=(args.history or None), now=(str(t_now) if t_now else None),
        outdir=args.outdir, retail_apps=(args.retail_apps or None),
        alpha_qib=args.alpha_qib, alpha_nii=args.alpha_nii, alpha_ret=args.alpha_ret, alpha_emp=args.alpha_emp,
        ci=args.ci
    ))
    (outdir/"config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== IPO Subscription Tracker ==")
    print(f"Issue: {summary['issue']} | Window: {summary['window']['open']} → {summary['window']['close']}")
    print(f"As-of: {latest['timestamp']}  (t_frac={latest['t_frac']:.3f})")
    s_now = summary["as_of"]["subs_now"]
    s_now_disp = ", ".join([f"{k}:{(v if v is not None else float('nan')):.2f}x" for k,v in s_now.items() if v is not None])
    print("Current subs:", s_now_disp)
    if rows:
        for r in rows:
            print(f"Forecast {r['category']}: {r['sub_final_point']:.2f}x  (α={r['alpha']:.2f}, n={r['n_points']})")
    if retail_odds:
        print(f"Retail odds: apps≈{retail_odds['apps_inferred']:,}, lots={retail_odds['lots_available']:,}, "
              f"P(≥1 lot)≈{retail_odds['prob_one_lot']*100:.1f}% | E[lots/app]≈{retail_odds['expected_lots_per_app']:.2f}")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
