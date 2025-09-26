#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gold_loans_vs_imports.py — Do Indian gold loans substitute for gold imports?
-----------------------------------------------------------------------------

What this does
==============
Given time series for (a) gold loans (e.g., NBFC/Bank outstanding or flows) and
(b) gold imports (value or tonnes), this script:

1) Cleans data to MONTHLY frequency and aligns series (optional gold price & USD/INR).
2) If imports are in value, converts to **implied tonnes** using price & FX:
      tons ≈ (imports_value_in_USD / gold_price_USD_per_oz) / 32,150.7466
3) Builds diagnostics:
   • Δlog relationships, rolling correlations/betas (short/med/long windows)
   • Lead–lag table: Corr(Δlog Loans_{t−k}, Δlog Imports_t) for k∈[−L..L]
   • Simple elasticities via OLS:
       Δlog(ImportsProxy)_t ~ α + β_L * Δlog(Loans_t) + β_P * Δlog(Price_t) + β_FX * Δlog(USDINR_t)
   • “Substitution events”: months where loans jump but imports fall (and vice versa)
4) Produces tidy CSVs + a JSON summary (latest stats + elasticities).

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--loans loans.csv          REQUIRED
  Columns: date, value
  Optional: type/segment columns are ignored.

--imports imports.csv      REQUIRED
  Columns: date, <one of: value_in_inr | value_in_usd | tons>
  Use --imports_col to force; otherwise the script guesses common names:
    value_in_inr:  "value", "imports_inr", "inr", "value_inr", "amount_inr"
    value_in_usd:  "usd", "value_usd", "imports_usd"
    tons:          "tons", "tonnes", "volume_ton", "qty_tonnes"

--price price.csv          OPTIONAL (if imports are value; gold LBM price)
  Columns: date, price_usd_per_oz  (use --price_col to override)

--fx fx.csv                OPTIONAL (if imports are INR; USD/INR)
  Columns: date, usd_inr   (use --fx_col to override)  (units: INR per 1 USD)

CLI (example)
-------------
python gold_loans_vs_imports.py \
  --loans loans.csv --imports imports.csv --price price.csv --fx fx.csv \
  --imports_col value_inr --price_col GOLD_OZ --fx_col USDINR \
  --start 2014-01-01 --wshort 6 --wmed 12 --wlong 24 --lags 12 \
  --outdir out_gold

Outputs
-------
- panel_monthly.csv        Monthly aligned panel with levels & Δlogs (loans, imports_proxy, price, fx)
- rolling_stats.csv        Rolling corr/beta between Δlog(loans) and Δlog(imports_proxy) (S/M/L)
- leadlag_corr.csv         Lead–lag correlation table (lags × corr)
- regression_coeffs.csv    OLS elasticities with std errors & t-stats
- events.csv               Substitution/divergence event flags
- summary.json             Headline diagnostics & latest estimates
- config.json              Run configuration for reproducibility

Notes
-----
• Imports proxy priority: tons → value_usd(+price) → value_inr(+fx,+price).
• If you don’t provide price (and imports aren’t in tons), the proxy falls back
  to value growth — fine for correlation/lead–lag, but *not* comparable to tonnes.
• All analytics use monthly Δlogs (approximate % changes). Windows are in months.

DISCLAIMER
----------
Research tooling with simplifying assumptions (e.g., linear elasticities, no endogeneity fix).
Validate before policy or trading use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

OZT_PER_TON = 32150.7466  # troy ounces per metric ton

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return the first matching column (case-insensitive, fuzzy)."""
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        if cand.lower() in low: return low[cand.lower()]
    # fuzzy contains
    for cand in cands:
        t = cand.lower()
        for c in df.columns:
            if t in str(c).lower(): return c
    return None

def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def resample_monthly_sum(df: pd.DataFrame, col: str) -> pd.Series:
    return df.set_index("date")[col].resample("M").sum(min_count=1)

def resample_monthly_mean(df: pd.DataFrame, col: str) -> pd.Series:
    return df.set_index("date")[col].resample("M").mean()

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    minp = max(3, window // 2)
    return x.rolling(window, min_periods=minp).corr(y)

def roll_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta for y ~ a + b*x (one regressor)."""
    minp = max(3, window // 2)
    x_ = x.astype(float); y_ = y.astype(float)
    mx = x_.rolling(window, min_periods=minp).mean()
    my = y_.rolling(window, min_periods=minp).mean()
    cov = (x_*y_).rolling(window, min_periods=minp).mean() - mx*my
    varx = (x_*x_).rolling(window, min_periods=minp).mean() - mx*mx
    return cov / varx.replace(0, np.nan)

def leadlag_corr(flow: pd.Series, ret: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            f = flow.shift(k); r = ret
        else:
            f = flow; r = ret.shift(-k)
        c = f.corr(r)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

def ols(y: pd.Series, X: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, float]:
    """
    OLS with intercept included in X if provided.
    Returns (beta, cov_beta, sigma2).
    """
    Y = y.values.reshape(-1,1)
    Xv = X.values
    XT = Xv.T
    XtX = XT @ Xv
    XtY = XT @ Y
    XtX_inv = np.linalg.pinv(XtX)
    B = XtX_inv @ XtY                # (k×1)
    resid = Y - Xv @ B               # (n×1)
    n, k = Xv.shape
    dof = max(1, n - k)
    sigma2 = float((resid.T @ resid) / dof)
    covB = XtX_inv * sigma2
    beta = pd.Series(B.ravel(), index=X.columns, dtype=float)
    cov = pd.DataFrame(covB, index=X.columns, columns=X.columns)
    return beta, cov, sigma2


# ----------------------------- loaders -----------------------------

def load_loans(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    val = ncol(df, "value", "loans", "amount", "outstanding", "flows", "loan_value")
    if not dt or not val:
        raise ValueError("loans.csv must have date and value columns (e.g., 'date', 'value').")
    df = df.rename(columns={dt:"date", val:"loans_val"})
    df["date"] = to_month(df["date"])
    df["loans_val"] = safe_num(df["loans_val"])
    # monthly: sum if flows, else mean; we default to mean to avoid double-counting — user can pre-aggregate.
    # Heuristic: if >50% of values are negative or near zero, treat as flows → sum.
    flows_like = (df["loans_val"] <= 0).mean() > 0.1
    agg = resample_monthly_sum if flows_like else resample_monthly_mean
    s = agg(df[["date","loans_val"]].dropna(), "loans_val")
    return pd.DataFrame({"date": s.index, "loans_val": s.values})

def load_imports(path: str, imports_col_hint: str="") -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    if not dt: raise ValueError("imports.csv needs a 'date' column.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month(df["date"])
    # Detect column type
    if imports_col_hint:
        col = ncol(df, imports_col_hint)
        if not col: raise ValueError(f"imports_col '{imports_col_hint}' not found.")
        return _reshape_imports(df, col, imports_col_hint)
    # Guess
    tons_c = ncol(df, "tons", "tonnes", "qty_tonnes", "volume_ton")
    usd_c  = ncol(df, "usd", "value_usd", "imports_usd")
    inr_c  = ncol(df, "value_inr", "imports_inr", "inr", "value", "amount_inr")
    if tons_c:
        return _reshape_imports(df, tons_c, "tons")
    if usd_c:
        return _reshape_imports(df, usd_c, "value_usd")
    if inr_c:
        return _reshape_imports(df, inr_c, "value_inr")
    raise ValueError("imports.csv: could not find columns for tons/value_usd/value_inr.")

def _reshape_imports(df: pd.DataFrame, col: str, kind: str) -> pd.DataFrame:
    s = df[["date", col]].rename(columns={col: kind})
    s[kind] = safe_num(s[kind])
    # monthly sum (imports are flows)
    s = s.set_index("date").resample("M").sum(min_count=1).reset_index()
    return s

def load_price(path: Optional[str], price_col_hint: str="") -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    if not dt: raise ValueError("price.csv needs a date column.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month(df["date"])
    pc = ncol(df, price_col_hint) if price_col_hint else (ncol(df,"price_usd_per_oz") or ncol(df,"gold") or ncol(df,"XAUUSD") or ncol(df,"price"))
    if not pc:
        # pick first numeric non-date
        num = [c for c in df.columns if c!="date" and pd.api.types.is_numeric_dtype(df[c])]
        if not num: raise ValueError("price.csv: no numeric price column found.")
        pc = num[0]
    df = df[["date", pc]].rename(columns={pc:"gold_usd_oz"})
    df["gold_usd_oz"] = safe_num(df["gold_usd_oz"])
    df = df.set_index("date").resample("M").mean().reset_index()
    return df

def load_fx(path: Optional[str], fx_col_hint: str="") -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    if not dt: raise ValueError("fx.csv needs a date column.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month(df["date"])
    fc = ncol(df, fx_col_hint) if fx_col_hint else (ncol(df,"usd_inr") or ncol(df,"dollar_inr") or ncol(df,"inr_per_usd") or ncol(df,"USDINR"))
    if not fc:
        num = [c for c in df.columns if c!="date" and pd.api.types.is_numeric_dtype(df[c])]
        if not num: raise ValueError("fx.csv: no numeric fx column found.")
        fc = num[0]
    df = df[["date", fc]].rename(columns={fc:"usd_inr"})
    df["usd_inr"] = safe_num(df["usd_inr"])
    df = df.set_index("date").resample("M").mean().reset_index()
    return df


# ----------------------------- core calc -----------------------------

def build_panel(loans: pd.DataFrame, imp: pd.DataFrame, price: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    # Merge monthly
    panel = pd.DataFrame({"date": pd.date_range(
        start=min(loans["date"].min(), imp["date"].min()),
        end=max(loans["date"].max(), imp["date"].max()),
        freq="M"
    )})
    panel = panel.merge(loans, on="date", how="left")
    panel = panel.merge(imp, on="date", how="left")
    if not price.empty:
        panel = panel.merge(price, on="date", how="left")
    if not fx.empty:
        panel = panel.merge(fx, on="date", how="left")

    # Decide imports proxy
    have_tons = "tons" in panel.columns and panel["tons"].notna().any()
    have_usd  = "value_usd" in panel.columns and panel["value_usd"].notna().any()
    have_inr  = "value_inr" in panel.columns and panel["value_inr"].notna().any()

    panel["imports_proxy"] = np.nan
    proxy_note = None

    if have_tons:
        panel["imports_proxy"] = safe_num(panel["tons"])
        proxy_note = "tons (as-provided)"
    elif have_usd and ("gold_usd_oz" in panel.columns):
        panel["imports_proxy"] = (safe_num(panel["value_usd"]) / panel["gold_usd_oz"]) / OZT_PER_TON
        proxy_note = "implied tons from USD/value and gold price"
    elif have_inr and ("usd_inr" in panel.columns) and ("gold_usd_oz" in panel.columns):
        usd_val = safe_num(panel["value_inr"]) / panel["usd_inr"]
        panel["imports_proxy"] = (usd_val / panel["gold_usd_oz"]) / OZT_PER_TON
        proxy_note = "implied tons from INR/value, FX, gold price"
    else:
        # fallback: use value growth as proxy (no unit conversion)
        src = "value_usd" if have_usd else ("value_inr" if have_inr else None)
        if src is None:
            raise ValueError("Imports proxy cannot be constructed: provide tons or value with price (and FX if INR).")
        panel["imports_proxy"] = safe_num(panel[src])
        proxy_note = f"value-based proxy ({src}) — NOTE: not tonnes"

    # Δlogs (approx % changes)
    panel["dlog_loans"]   = dlog(panel["loans_val"])
    panel["dlog_imports"] = dlog(panel["imports_proxy"])
    panel["dlog_price"]   = dlog(panel["gold_usd_oz"]) if "gold_usd_oz" in panel.columns else np.nan
    panel["dlog_fx"]      = dlog(panel["usd_inr"]) if "usd_inr" in panel.columns else np.nan

    # Clean
    panel = panel.dropna(subset=["loans_val", "imports_proxy"], how="any")
    return panel, proxy_note

def rolling_stats(panel: pd.DataFrame, wshort: int, wmed: int, wlong: int) -> pd.DataFrame:
    df = panel.set_index("date")
    R = []
    for w, lab in [(wshort, "short"), (wmed, "med"), (wlong, "long")]:
        rc = roll_corr(df["dlog_loans"], df["dlog_imports"], w)
        b  = roll_beta(df["dlog_imports"], df["dlog_loans"], w)
        R.append(pd.DataFrame({"date": df.index,
                               f"corr_{lab}": rc.values,
                               f"beta_{lab}": b.values}).set_index("date"))
    out = pd.concat(R, axis=1).reset_index()
    return out

def leadlag(panel: pd.DataFrame, lags: int) -> pd.DataFrame:
    df = panel.set_index("date")
    return leadlag_corr(df["dlog_loans"], df["dlog_imports"], lags)

def regress_elasticity(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Δlog(imports) on Δlog(loans), Δlog(price), Δlog(fx) — drop missing controls automatically.
    """
    df = panel.dropna(subset=["dlog_imports", "dlog_loans"], how="any").copy()
    Xcols = ["const", "dlog_loans"]
    X = pd.DataFrame({"const": 1.0, "dlog_loans": df["dlog_loans"].values}, index=df.index)
    if df["dlog_price"].notna().any():
        X["dlog_price"] = df["dlog_price"]
        Xcols.append("dlog_price")
    if df["dlog_fx"].notna().any():
        X["dlog_fx"] = df["dlog_fx"]
        Xcols.append("dlog_fx")
    beta, cov, sigma2 = ols(df["dlog_imports"], X[Xcols])
    se = np.sqrt(np.diag(cov.values))
    res = pd.DataFrame({
        "var": X[Xcols].columns,
        "coef": beta.values,
        "std_err": se,
        "t_stat": beta.values / np.where(se==0, np.nan, se)
    })
    return res

def find_events(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Flag months where loans↑ while imports↓ and vice versa (Δlogs).
    """
    df = panel.copy().dropna(subset=["dlog_loans", "dlog_imports"])
    p75_loan = np.nanpercentile(df["dlog_loans"], 75)
    p25_loan = np.nanpercentile(df["dlog_loans"], 25)
    p75_imp  = np.nanpercentile(df["dlog_imports"], 75)
    p25_imp  = np.nanpercentile(df["dlog_imports"], 25)

    cond_sub = (df["dlog_loans"] >= p75_loan) & (df["dlog_imports"] <= p25_imp)
    cond_rest= (df["dlog_loans"] <= p25_loan) & (df["dlog_imports"] >= p75_imp)

    events = []
    for d, row in df[cond_sub].iterrows():
        events.append({"date": d, "type": "LoansUp_ImportsDown",
                       "dlog_loans": float(row["dlog_loans"]), "dlog_imports": float(row["dlog_imports"])})
    for d, row in df[cond_rest].iterrows():
        events.append({"date": d, "type": "LoansDown_ImportsUp",
                       "dlog_loans": float(row["dlog_loans"]), "dlog_imports": float(row["dlog_imports"])})
    return pd.DataFrame(events).sort_values("date")


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    loans: str
    imports: str
    price: Optional[str]
    fx: Optional[str]
    imports_col: Optional[str]
    price_col: Optional[str]
    fx_col: Optional[str]
    start: Optional[str]
    end: Optional[str]
    wshort: int
    wmed: int
    wlong: int
    lags: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Gold loans vs gold imports (monthly) — substitution analytics")
    ap.add_argument("--loans", required=True, help="CSV with gold loans time series (date,value)")
    ap.add_argument("--imports", required=True, help="CSV with gold imports (tons OR value_usd OR value_inr)")
    ap.add_argument("--price", default="", help="CSV with gold price (USD/oz)")
    ap.add_argument("--fx", default="", help="CSV with USD/INR (INR per 1 USD)")
    ap.add_argument("--imports_col", default="", help="Force which imports column to use: tons|value_usd|value_inr")
    ap.add_argument("--price_col", default="", help="Column in price.csv (default: price_usd_per_oz/gold)")
    ap.add_argument("--fx_col", default="", help="Column in fx.csv (default: usd_inr/USDINR)")
    ap.add_argument("--start", default="", help="Start YYYY-MM")
    ap.add_argument("--end", default="", help="End YYYY-MM")
    ap.add_argument("--wshort", type=int, default=6, help="Rolling short window (months)")
    ap.add_argument("--wmed", type=int, default=12, help="Rolling medium window (months)")
    ap.add_argument("--wlong", type=int, default=24, help="Rolling long window (months)")
    ap.add_argument("--lags", type=int, default=12, help="Lead–lag max lags (months)")
    ap.add_argument("--outdir", default="out_gold", help="Output directory")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load
    LOANS = load_loans(args.loans)
    IMP   = load_imports(args.imports, imports_col_hint=(args.imports_col or ""))
    PRICE = load_price(args.price, price_col_hint=(args.price_col or "")) if args.price else pd.DataFrame()
    FX    = load_fx(args.fx, fx_col_hint=(args.fx_col or "")) if args.fx else pd.DataFrame()

    # Date filters
    if args.start:
        start = pd.to_datetime(args.start).to_period("M").to_timestamp("M")
        for df in [LOANS, IMP, PRICE, FX]:
            if not df.empty: df.drop(df[df["date"] < start].index, inplace=True)
    if args.end:
        end = pd.to_datetime(args.end).to_period("M").to_timestamp("M")
        for df in [LOANS, IMP, PRICE, FX]:
            if not df.empty: df.drop(df[df["date"] > end].index, inplace=True)

    # Panel
    panel, proxy_note = build_panel(LOANS, IMP, PRICE, FX)
    if panel.empty or panel.shape[0] < 12:
        raise ValueError("Insufficient overlapping monthly data (need ≥12 months after alignment).")
    panel.to_csv(outdir / "panel_monthly.csv", index=False)

    # Rolling stats
    rolling = rolling_stats(panel, args.wshort, args.wmed, args.wlong)
    rolling.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag
    ll = leadlag(panel, args.lags)
    ll.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Elasticity regression
    regress = regress_elasticity(panel)
    regress.to_csv(outdir / "regression_coeffs.csv", index=False)

    # Events
    events = find_events(panel)
    if not events.empty:
        events.to_csv(outdir / "events.csv", index=False)

    # Summary
    latest = panel.dropna(subset=["dlog_loans","dlog_imports"]).tail(1)
    last_date = None if latest.empty else str(latest["date"].iloc[0].date())
    beta_s, beta_m, beta_l = (rolling.dropna(subset=[c]).iloc[-1][c] if not rolling.dropna(subset=[c]).empty else np.nan
                              for c in ["beta_short","beta_med","beta_long"])
    corr_s, corr_m, corr_l = (rolling.dropna(subset=[c]).iloc[-1][c] if not rolling.dropna(subset=[c]).empty else np.nan
                              for c in ["corr_short","corr_med","corr_long"])

    # Pull elasticity for loans if present
    b_loans = float(regress.loc[regress["var"]=="dlog_loans","coef"].iloc[0]) if (regress["var"]=="dlog_loans").any() else np.nan
    se_loans = float(regress.loc[regress["var"]=="dlog_loans","std_err"].iloc[0]) if (regress["var"]=="dlog_loans").any() else np.nan
    t_loans  = float(regress.loc[regress["var"]=="dlog_loans","t_stat"].iloc[0]) if (regress["var"]=="dlog_loans").any() else np.nan

    # Simple scenario: +10% MoM in loans
    scen_imp_pct = float(b_loans * 0.10) if b_loans==b_loans else None

    summary = {
        "rows": int(panel.shape[0]),
        "date_range": { "start": str(panel["date"].min().date()), "end": str(panel["date"].max().date()) },
        "imports_proxy_note": proxy_note,
        "latest": {
            "date": last_date,
            "dlog_loans_pct": (float(latest["dlog_loans"].iloc[0])*100 if not latest.empty else None),
            "dlog_imports_pct": (float(latest["dlog_imports"].iloc[0])*100 if not latest.empty else None),
        },
        "rolling": {
            "beta_short": float(beta_s) if beta_s==beta_s else None,
            "beta_med": float(beta_m) if beta_m==beta_m else None,
            "beta_long": float(beta_l) if beta_l==beta_l else None,
            "corr_short": float(corr_s) if corr_s==corr_s else None,
            "corr_med": float(corr_m) if corr_m==corr_m else None,
            "corr_long": float(corr_l) if corr_l==corr_l else None,
        },
        "elasticity": {
            "loans_coef": b_loans,
            "loans_se": se_loans,
            "loans_t": t_loans
        },
        "scenario_plus10pct_loans_effect_on_imports_pct": scen_imp_pct
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config
    cfg = asdict(Config(
        loans=args.loans, imports=args.imports, price=(args.price or None), fx=(args.fx or None),
        imports_col=(args.imports_col or None), price_col=(args.price_col or None), fx_col=(args.fx_col or None),
        start=(args.start or None), end=(args.end or None),
        wshort=int(args.wshort), wmed=int(args.wmed), wlong=int(args.wlong), lags=int(args.lags),
        outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Gold Loans vs Imports ==")
    print(f"Monthly rows: {summary['rows']} | {summary['date_range']['start']} → {summary['date_range']['end']}")
    print(f"Imports proxy: {proxy_note}")
    print(f"Rolling beta (S/M/L): {summary['rolling']['beta_short']:.2f}, {summary['rolling']['beta_med']:.2f}, {summary['rolling']['beta_long']:.2f}")
    if scen_imp_pct is not None:
        print(f"Elasticity: 10% ↑ in loans ⇒ {scen_imp_pct*100:+.1f}% change in imports (model)")

if __name__ == "__main__":
    main()
