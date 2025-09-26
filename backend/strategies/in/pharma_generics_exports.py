#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pharma_generics_exports.py — India pharma generics exports: pricing pressure, FX, FDA flow & scenarios
-----------------------------------------------------------------------------------------------------

What this does
==============
Given monthly (or higher-frequency) data on **pharma exports** alongside **US generic pricing**,
**USD/INR**, **FDA approvals/warnings**, and **API/input costs**, this script:

1) Cleans & aligns everything to **monthly**.
2) Builds transforms:
   • Δlog(exports_usd), YoY(exports_usd)
   • Δlog(USDINR)  (↑ = INR depreciation)
   • Generic price pressure via either:
       a) price index → Δlog(price_index)  (negatively related to exports if erosion)
       b) explicit erosion series → erosion_pp (percentage points; + = more erosion)
   • FDA activity: Δlog1p(ANDA approvals) and counts of warning letters / OAI / import alerts
   • API/input cost indices Δlog
3) Diagnostics
   • Rolling correlations (short/med/long) vs exports
   • Lead–lag tables Corr(X_{t−k}, Δlog Exports_t)
4) Distributed-lag elasticities (Newey–West SE)
   Δlog(Exports)_t ~ Σ β_fx·Δlog(USDINR)_{t−i}
                      + Σ β_price·(price_pressure)_{t−i}
                      + Σ β_anda·Δlog1p(ANDA)_{t−i}
                      + Σ β_api·Δlog(API_costs)_{t−i}
5) Scenarios
   • INR depreciation +5% / +10%
   • Price erosion shock ±5pp (worsen/improve)
   • API cost shock −10%
6) Optional event-study around policy/FDA events.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--exports exports.csv        REQUIRED (monthly preferred)
  Columns (any subset; extras ignored):
    date,
    exports_usd, pharma_exports_usd, generic_exports_usd, formulations_usd, api_exports_usd
    units, volume, qty  (for ASP; optional)
    us_exports_usd, eu_exports_usd, row_exports_usd (optional splits)

--pricing pricing.csv        OPTIONAL (US generic pricing)
  EITHER an index:
    date, us_generic_price_index, price_index, eris_index, mck_qgi, iqvia_generic_index
  OR an erosion series (pp or %):
    price_erosion, erosion_yoy, us_generic_erosion, gpi_erosion
  (If erosion looks like %, it will be normalized to **pp** in [−100, +100] terms.)

--fx fx.csv                  OPTIONAL
  Columns:
    date, usdinr  (or inverse inr_usd/inrusd → auto-inverted)

--fda fda.csv                OPTIONAL
  Columns:
    date, anda_approvals (or approvals), warning_letters, oai, import_alerts

--api api.csv                OPTIONAL (input costs/logistics proxies)
  Columns:
    date, api_cost_index, solvent_index, freight_index, power_price, china_api_index ...

--events events.csv          OPTIONAL (event study)
  Columns:
    date, label

Key CLI
-------
--lags 12                    Lag horizon for d-lag & lead–lag (months)
--windows 6,12,24            Rolling windows
--start / --end              Date filters (YYYY-MM-DD)
--outdir out_generics
--min_obs 36                 Minimum observations for regressions

Outputs
-------
- panel.csv                  Aligned monthly panel (levels & transforms)
- rolling_stats.csv          Rolling corr of Δlog exports vs drivers
- leadlag_corr.csv           Lead–lag tables Corr(X_{t−k}, Δlog Exports_t)
- dlag_regression.csv        D-lag regression (coef, se, t) per regressor & lag
- scenarios.csv              Shock mapping (INR +5/+10, erosion ±5pp, API −10%)
- event_study.csv            Average Δlog around events (if provided)
- summary.json               Headline diagnostics & last snapshot
- config.json                Run configuration

DISCLAIMER
----------
Research tooling. Results depend on input construction (e.g., pricing series, coverage).
Validate before investment, hedging, or policy use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
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

def to_month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def dlog1p(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (np.log1p(s) - np.log1p(s.shift(1)))

def yoy_log(s: pd.Series, periods: int=12) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s) - np.log(s.shift(periods))

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(6, window//3)).corr(y)

def leadlag_corr(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            xx = x.shift(k); yy = y
        else:
            xx = x; yy = y.shift(-k)
        c = xx.corr(yy)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

def ols_beta_se(X: np.ndarray, y: np.ndarray):
    XtX = X.T @ X
    XtY = X.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ XtY
    resid = y - X @ beta
    return beta, resid, XtX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
    """Newey–West HAC with Bartlett kernel."""
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1)+1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))


# ----------------------------- loaders -----------------------------

def load_exports(path: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    ex = (ncol(df, "exports_usd","pharma_exports_usd","generic_exports_usd",
               "formulations_usd","drug_exports_usd","total_exports_usd","value_usd") )
    if not ex:
        raise ValueError("exports.csv must contain a USD exports column (e.g., exports_usd).")
    vol = ncol(df, "units","volume","qty","quantity","packs")
    df[ex] = safe_num(df[ex])
    if vol: df[vol] = safe_num(df[vol])
    out = df[["date", ex] + ([vol] if vol else [])].rename(columns={ex:"exports_usd", (vol or "units"):"units"})
    # add optional splits if present
    for c in ["us_exports_usd","eu_exports_usd","row_exports_usd","api_exports_usd","formulations_usd"]:
        cc = ncol(df, c)
        if cc:
            out[c] = safe_num(df[cc])
    # ASP proxy
    if "units" in out.columns:
        out["asp_usd"] = out["exports_usd"] / out["units"].replace(0, np.nan)
    return out.sort_values("date"), "exports_usd", ("units" if "units" in out.columns else None)

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    fx = ncol(df, "usdinr","usd_inr","inr_per_usd","fx")
    if not fx:
        inv = ncol(df, "inrusd","inr_usd","usdperinr")
        if not inv:
            raise ValueError("fx.csv must have usdinr (or inverse).")
        df["usdinr"] = 1.0 / safe_num(df[inv])
    else:
        df["usdinr"] = safe_num(df[fx])
    return df[["date","usdinr"]].sort_values("date")

def load_pricing(path: Optional[str]) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """
    Return DF with either:
      - price_index (level) and dlog_price_index
      - erosion_pp (percentage points), normalized to pp (not fraction)
    """
    if not path: return pd.DataFrame(), None, None
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    idx = ncol(df, "us_generic_price_index","price_index","generic_index","iqvia_generic_index","ers_generic_index")
    ero = ncol(df, "price_erosion","erosion_yoy","generic_erosion","gpi_erosion")
    out = pd.DataFrame({"date": df["date"]})
    idx_used = ero_used = None
    if idx:
        out["price_index"] = safe_num(df[idx])
        out["dlog_price_index"] = dlog(out["price_index"])
        idx_used = "price_index"
    if ero:
        x = safe_num(df[ero])
        # normalize: if looks like −7.5 (percent), keep as pp; if −0.075 (fraction), convert to pp
        med = np.nanmedian(x.values.astype(float))
        if np.isfinite(med) and abs(med) < 1.0:
            x = x * 100.0
        out["erosion_pp"] = x
        ero_used = "erosion_pp"
    return out.drop_duplicates(subset=["date"]).sort_values("date"), idx_used, ero_used

def load_fda(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    apps = ncol(df, "anda_approvals","approvals","anda","gdufa_approvals")
    warn = ncol(df, "warning_letters","wl")
    oai  = ncol(df, "oai","official_action_indicated","oai_count")
    imp  = ncol(df, "import_alerts","import_alert","ia")
    keep = ["date"]
    if apps: df = df.rename(columns={apps:"anda_approvals"}); df["anda_approvals"] = safe_num(df["anda_approvals"]); keep.append("anda_approvals")
    if warn: df = df.rename(columns={warn:"warning_letters"}); df["warning_letters"] = safe_num(df["warning_letters"]); keep.append("warning_letters")
    if oai:  df = df.rename(columns={oai:"oai"}); df["oai"] = safe_num(df["oai"]); keep.append("oai")
    if imp:  df = df.rename(columns={imp:"import_alerts"}); df["import_alerts"] = safe_num(df["import_alerts"]); keep.append("import_alerts")
    out = df[keep].groupby("date", as_index=False).sum()
    return out.sort_values("date")

def load_api(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])
    return df.groupby("date", as_index=False).mean().sort_values("date")

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    lab = ncol(df, "label","event","name") or "label"
    df = df.rename(columns={dt:"date", lab:"label"})
    df["date"] = to_month_end(df["date"])
    df["label"] = df["label"].astype(str)
    return df[["date","label"]].dropna().sort_values("date")


# ----------------------------- core construction -----------------------------

def build_panel(EX: pd.DataFrame, FX: pd.DataFrame, PR: pd.DataFrame, FDA: pd.DataFrame, API: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    df = EX.copy()
    for d in [FX, PR, FDA, API]:
        if not d.empty:
            df = df.merge(d, on="date", how="left")
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    # transforms
    df["dlog_exports"] = dlog(df["exports_usd"])
    df["yoy_exports"] = yoy_log(df["exports_usd"])
    if "usdinr" in df.columns:
        df["dlog_fx"] = dlog(df["usdinr"])
    if "price_index" in df.columns:
        if "dlog_price_index" not in df.columns:
            df["dlog_price_index"] = dlog(df["price_index"])
    # price pressure (higher = worse)
    if "erosion_pp" in df.columns:
        df["price_pressure"] = df["erosion_pp"]  # already pp
    elif "dlog_price_index" in df.columns:
        df["price_pressure"] = -100.0 * df["dlog_price_index"]  # approx: −Δlog(index) in pp
    else:
        df["price_pressure"] = np.nan

    if "anda_approvals" in df.columns:
        df["dlog1p_anda"] = dlog1p(df["anda_approvals"])
    for c in [col for col in df.columns if c.endswith("_index") and col!="price_index"]:
        df[f"dlog_{c}"] = dlog(df[c])

    # ASP transform if units present
    if "asp_usd" in df.columns:
        df["dlog_asp"] = dlog(df["asp_usd"])

    mapping = {
        "exports": "dlog_exports",
        "fx": "dlog_fx" if "dlog_fx" in df.columns else None,
        "price_pressure": "price_pressure" if "price_pressure" in df.columns else None,
        "anda": "dlog1p_anda" if "dlog1p_anda" in df.columns else None,
    }
    # add first available API cost driver to mapping (optional)
    api_dlogs = [c for c in df.columns if c.startswith("dlog_") and any(k in c for k in ["api_cost","solvent","freight","power","china_api"])]
    if api_dlogs:
        mapping["api_costs"] = api_dlogs[0]

    return df, mapping


# ----------------------------- analytics -----------------------------

def rolling_stats(panel: pd.DataFrame, mapping: Dict[str,str], windows: List[int]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    y = idx.get("dlog_exports")
    if y is None: return pd.DataFrame()
    for name, col in mapping.items():
        if not col or col not in idx.columns: continue
        if name == "exports": continue
        x = idx[col]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({"driver": name, "column": col, "window": w, "tag": tag,
                         "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan})
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, mapping: Dict[str,str], lags: int) -> pd.DataFrame:
    rows = []
    y = panel.get("dlog_exports")
    if y is None or y.dropna().empty: return pd.DataFrame()
    for name, col in mapping.items():
        if name == "exports" or not col or col not in panel.columns: continue
        tab = leadlag_corr(panel[col], y, lags)
        tab["driver"] = name
        tab["column"] = col
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","driver","column"])

def dlag_regression(panel: pd.DataFrame, mapping: Dict[str,str], L: int, min_obs: int) -> pd.DataFrame:
    """
    Δlog(Exports)_t on lags (0..L) of drivers with HAC(SE).
    """
    if "dlog_exports" not in panel.columns: return pd.DataFrame()
    df = panel.copy()
    dep = "dlog_exports"
    Xparts = [pd.Series(1.0, index=df.index, name="const")]
    names = ["const"]
    # include FX, price pressure, ANDA, first API cost if available
    drivers = ["fx","price_pressure","anda","api_costs"]
    for drv in drivers:
        col = mapping.get(drv)
        if not col or col not in df.columns: continue
        for l in range(0, L+1):
            nm = f"{col}_l{l}"
            Xparts.append(df[col].shift(l).rename(nm))
            names.append(nm)
    X = pd.concat(Xparts, axis=1)
    XY = pd.concat([df[dep].rename("dep"), X], axis=1).dropna()
    if XY.shape[0] < max(min_obs, 5*X.shape[1]):
        return pd.DataFrame()
    yv = XY["dep"].values.reshape(-1,1)
    Xv = XY.drop(columns=["dep"]).values
    beta, resid, XtX_inv = ols_beta_se(Xv, yv)
    se = hac_se(Xv, resid, XtX_inv, L=max(6, L))
    out = []
    for i, nm in enumerate(names):
        out.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                    "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "lags": L})
    # cumulative effects (0..L) for key drivers
    for key in ["dlog_fx", "price_pressure", mapping.get("api_costs","")]:
        if not key: continue
        idxs = [i for i, nm in enumerate(names) if nm.startswith(f"{key}_l")]
        if idxs:
            bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
            out.append({"var": f"{key}_cum_0..L", "coef": bsum, "se": ses,
                        "t_stat": bsum/(ses if ses>0 else np.nan), "lags": L})
    return pd.DataFrame(out)

def scenarios(reg: pd.DataFrame) -> pd.DataFrame:
    """
    Map shocks:
      FX: +5%, +10% INR depreciation  → use cum β_fx
      Price erosion: ±5pp             → use cum β_price (pp → multiply by 0.01)
      API costs: −10%                 → use cum β_api (Δlog = ln(0.9) ≈ −0.1053)
    All impacts reported as **percentage impact on exports growth** (Δlog * 100).
    """
    if reg.empty: return pd.DataFrame()
    def get_cum(prefix: str) -> Optional[float]:
        row = reg[reg["var"] == prefix]
        return float(row["coef"].iloc[0]) if not row.empty else None

    b_fx   = get_cum("dlog_fx_cum_0..L")
    b_pr   = None
    # find price cum key (price_pressure)
    r = reg[reg["var"].str.contains("^price_pressure_cum_0..L$", regex=True)]
    if not r.empty: b_pr = float(r["coef"].iloc[0])
    # api cum: first var containing "api_cost" and "cum"
    r2 = reg[reg["var"].str.contains("api_cost.*_cum_0..L")]
    b_api = float(r2["coef"].iloc[0]) if not r2.empty else None

    rows = []
    # FX shocks
    if b_fx is not None:
        for sh in [0.05, 0.10]:
            rows.append({"shock": f"INR depreciation {int(sh*100)}%", "impact_pp": b_fx * sh * 100.0})
    # Price erosion shocks (pp → decimal)
    if b_pr is not None:
        for pp in [ +5.0, -5.0 ]:
            rows.append({"shock": f"Price erosion {pp:+.0f}pp", "impact_pp": b_pr * (pp/100.0) * 100.0})
    # API cost shock −10%
    if b_api is not None:
        dl = np.log(0.9)  # ≈ −0.1053
        rows.append({"shock": "API cost −10%", "impact_pp": b_api * dl * 100.0})
    return pd.DataFrame(rows)

def event_study(panel: pd.DataFrame, events: pd.DataFrame, window: int=6) -> pd.DataFrame:
    if events.empty or "dlog_exports" not in panel.columns:
        return pd.DataFrame()
    rows = []
    dates = panel["date"].values
    for _, ev in events.iterrows():
        d0 = pd.to_datetime(ev["date"])
        lbl = str(ev["label"])
        if len(dates)==0: continue
        idx = np.argmin(np.abs(dates - np.datetime64(d0)))
        anchor = pd.to_datetime(dates[idx])
        sl = panel[(panel["date"] >= anchor - pd.offsets.DateOffset(months=window)) &
                   (panel["date"] <= anchor + pd.offsets.DateOffset(months=window))]
        for _, r in sl.iterrows():
            h = (r["date"].to_period("M") - anchor.to_period("M")).n
            rows.append({"event": lbl, "event_date": anchor, "h": int(h),
                         "dlog_exports": float(r.get("dlog_exports", np.nan)) if pd.notna(r.get("dlog_exports", np.nan)) else np.nan,
                         "price_pressure": float(r.get("price_pressure", np.nan)) if pd.notna(r.get("price_pressure", np.nan)) else np.nan})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = (df.groupby(["event","event_date","h"])
             .agg({"dlog_exports":"mean","price_pressure":"mean"})
             .reset_index()
             .sort_values(["event","h"]))
    return out


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    exports: str
    pricing: Optional[str]
    fx: Optional[str]
    fda: Optional[str]
    api: Optional[str]
    events: Optional[str]
    lags: int
    windows: str
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pharma generics exports vs pricing, FX, FDA & API costs")
    ap.add_argument("--exports", required=True)
    ap.add_argument("--pricing", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--fda", default="")
    ap.add_argument("--api", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--windows", default="6,12,24")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_generics")
    ap.add_argument("--min_obs", type=int, default=36)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    EX, ex_col, vol_col = load_exports(args.exports)
    FX = load_fx(args.fx) if args.fx else pd.DataFrame()
    PR, idx_used, ero_used = load_pricing(args.pricing) if args.pricing else (pd.DataFrame(), None, None)
    FDA = load_fda(args.fda) if args.fda else pd.DataFrame()
    API = load_api(args.api) if args.api else pd.DataFrame()
    EVENTS = load_events(args.events) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        for df in [EX, FX, PR, FDA, API]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [EX, FX, PR, FDA, API]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    PANEL, mapping = build_panel(EX, FX, PR, FDA, API)
    if PANEL["dlog_exports"].dropna().shape[0] < args.min_obs:
        raise ValueError("Insufficient overlapping months after alignment (need ≥ min_obs on Δlog exports).")
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Rolling & lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, mapping, windows)
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)
    LL = leadlag_tables(PANEL, mapping, lags=int(args.lags))
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Distributed-lag regression
    REG = dlag_regression(PANEL, mapping, L=int(args.lags), min_obs=int(args.min_obs))
    if not REG.empty: REG.to_csv(outdir / "dlag_regression.csv", index=False)

    # Scenarios
    SCN = scenarios(REG) if not REG.empty else pd.DataFrame()
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Event study
    ES = event_study(PANEL, EVENTS, window=max(6, int(args.lags)//2)) if not EVENTS.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Best lead/lag summary
    best_ll = {}
    if not LL.empty:
        for drv, g in LL.dropna(subset=["corr"]).groupby("driver"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[drv] = {"lag": int(row["lag"]), "corr": float(row["corr"]), "column": row["column"]}

    # Summary
    last = PANEL.dropna(subset=["exports_usd"]).tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "columns_used": {
            "exports": ex_col,
            "volume": vol_col,
            "fx": ("usdinr" if "usdinr" in PANEL.columns else None),
            "pricing_index": idx_used,
            "erosion_series": ero_used
        },
        "latest": {
            "date": str(last["date"].date()),
            "exports_usd": float(last["exports_usd"]),
            "asp_usd": (float(last["asp_usd"]) if "asp_usd" in PANEL.columns and pd.notna(last["asp_usd"]) else None),
            "dlog_exports_pct": float(last.get("dlog_exports", np.nan))*100 if pd.notna(last.get("dlog_exports", np.nan)) else None,
            "price_pressure_pp": float(last.get("price_pressure", np.nan)) if pd.notna(last.get("price_pressure", np.nan)) else None
        },
        "rolling_windows": windows,
        "leadlag_best": best_ll,
        "regression_terms": REG["var"].tolist() if not REG.empty else [],
        "scenarios": SCN.to_dict(orient="records") if not SCN.empty else []
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        exports=args.exports, pricing=(args.pricing or None), fx=(args.fx or None),
        fda=(args.fda or None), api=(args.api or None), events=(args.events or None),
        lags=int(args.lags), windows=args.windows, start=(args.start or None), end=(args.end or None),
        outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Pharma Generics Exports ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']}")
    if best_ll:
        for k, st in best_ll.items():
            print(f"Lead–lag {k}: max |corr| at lag {st['lag']:+d} → {st['corr']:+.2f}")
    if not SCN.empty:
        for _, r in SCN.iterrows():
            print(f"{r['shock']}: impact ≈ {r['impact_pp']:+.2f} pp on Δlog(exports)")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
