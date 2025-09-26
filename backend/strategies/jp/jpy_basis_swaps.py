#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jpy_basis_swaps.py — JPY cross-currency basis (USD/JPY, EUR/JPY): drivers, CIP, funding & scenarios
---------------------------------------------------------------------------------------------------

What this does
==============
Builds a **daily (or monthly)** panel for JPY cross-currency basis swaps and computes:
- Curve snapshots by tenor (1M–5Y), changes, slopes, and **carry/roll-down**
- Covered Interest Parity (CIP) **deviation** using spot & forwards vs OIS curves
- **Hedged funding costs** (USD→JPY, JPY→USD) from basis & forwards
- **Elasticities** of basis to stress proxies (USD FRA–OIS, repo diffs, FX, issuance, quarter ends)
  using OLS with Newey–West (HAC) SEs
- **Event study** around quarter ends, FY-end, BoJ/Fed dates (if provided)
- **Scenario engine** (one-step) that shocks drivers → Δbasis by tenor, funding impact, carry P&L
- **Historical VaR/ES** on basis changes for chosen tenor

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--basis basis.csv            REQUIRED (daily preferred; long form)
  Columns:
    date, pair[, ccy_pair], tenor, basis_bp[, basis]
  Notes:
    • pair values like USDJPY or EURJPY. basis_bp sign as quoted in market (usually negative in stress).

--ois ois.csv                OPTIONAL (daily or monthly)
  Columns:
    date, ccy, tenor, rate_pct[, ois_pct, r]
  Notes:
    • ccy in {USD, JPY, EUR}. Tenors matching basis tenors help (1M,3M,6M,1Y,2Y,5Y).
    • Used for CIP and funding-cost comparisons.

--fx fx.csv                  OPTIONAL (daily or monthly)
  Columns:
    date, USDJPY[, JPYUSD]
  Notes:
    • If forward points provided (see --fwd), CIP uses forwards. Else CIP falls back to OIS diff + basis.

--fwd forwards.csv           OPTIONAL (daily or monthly; USDJPY forwards)
  Columns:
    date, tenor, fwd_points[, fwd_jpy_points, points]
  Notes:
    • Points are **in JPY per USD** (same unit as spot). Forward = spot + points.

--fraois fraois.csv          OPTIONAL (stress proxy)
  Columns:
    date, usd_fraois_bp[, fra_ois_bp, usd_fra_ois_bp]
  Notes:
    • Used as USD funding stress driver.

--repo repo.csv              OPTIONAL
  Columns:
    date, ccy, repo_pct[, gc_pct, r]
  Notes:
    • We build repo_diff = repo_usd - repo_jpy as a collateral/funding driver.

--issuance issuance.csv      OPTIONAL (Samurai/Uridashi/JPY bank or USD bank issuance)
  Columns:
    date, ccy, amount_usd[, amt, amount], type[, label]
  Notes:
    • We compute rolling sums (e.g., 1M, 3M) as a supply driver.

--events events.csv          OPTIONAL
  Columns:
    date, type[, label]

CLI (key)
---------
--pair USDJPY                Pair to analyze (USDJPY or EURJPY)
--tenors 1M,3M,6M,1Y,2Y,5Y   Tenors of interest (comma-separated)
--freq daily|monthly         Aggregation for analysis outputs (default daily)
--roll 60                    Rolling window (obs units) for beta/corr/carry estimates
--reg_lags 5                 Newey–West HAC lags (daily≈5–10, monthly≈3)
--notional 100000000         Notional used for carry/roll approximations (JPY)
--scenario_fraois 20         Shock to USD FRA–OIS in bp
--scenario_repo_diff 10      Shock to (repo_USD - repo_JPY) in bp
--scenario_fx 3              % change in USDJPY (+ = USD↑/JPY↓)
--scenario_issuance 2000     Additional issuance (USD mm) over last month
--tenor_for_var 3M           Tenor used for VaR/ES & scenario carry
--outdir out_basis           Output directory
--start / --end              Inclusive filters (YYYY-MM-DD)

Outputs
-------
panel_long.csv              Tidy joined panel (date, pair, tenor, basis_bp, drivers…)
curve_wide.csv              Wide curve by tenor (per date)
cip_deviation.csv           CIP deviations by tenor (bp)
funding_costs.csv           Hedged funding rates (JPY↔USD) by tenor
reg_elasticities.csv        Basis ~ drivers OLS (HAC/NW SEs)
event_study.csv             ±window CARs in Δbasis around events (or quarter ends)
carry_roll.csv              Carry/roll-down and slope metrics by tenor
scenarios.csv               One-step scenario: Δbasis, funding deltas, carry P&L
stress_vares.csv            Historical VaR/ES for Δbasis (chosen tenor)
summary.json, config.json   Metadata and configuration echo

DISCLAIMER: Research tooling. Validate sign conventions & quoting with your data source before decisions.
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
    if df is None or df.empty: return None
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        if cand.lower() in low: return low[cand.lower()]
    # fuzzy contains
    for cand in cands:
        key = cand.lower()
        for c in df.columns:
            if key in str(c).lower(): return c
    return None

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def eom(s: pd.Series) -> pd.Series:
    return to_dt(s) + pd.offsets.MonthEnd(0)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(x: pd.Series) -> pd.Series:
    x = safe_num(x).replace(0, np.nan)
    return np.log(x) - np.log(x.shift(1))

def pct_change(x: pd.Series) -> pd.Series:
    x = safe_num(x); return x.pct_change()

def tenor_to_years(t: str) -> float:
    t = str(t).upper().strip()
    if t.endswith("W"): return float(t[:-1]) * 7.0 / 365.0
    if t.endswith("M"): return float(t[:-1]) / 12.0
    if t.endswith("Y"): return float(t[:-1])
    return np.nan

def standardize_tenor(t: str) -> str:
    t = str(t).upper().strip()
    # normalize like 1M,3M,6M,1Y,2Y,5Y
    if t.endswith("MO"): t = t[:-2] + "M"
    if t.endswith("YR"): t = t[:-2] + "Y"
    return t

# Newey–West (HAC) utilities
def ols_beta_resid(X: np.ndarray, y: np.ndarray):
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XTX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XTX_inv: np.ndarray, L: int) -> np.ndarray:
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1) + 1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XTX_inv @ S @ XTX_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


# ----------------------------- loaders -----------------------------

def load_basis(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df,"date"); p = ncol(df,"pair","ccy_pair"); t = ncol(df,"tenor"); b = ncol(df,"basis_bp","basis")
    if not (d and p and t and b): raise ValueError("basis.csv needs date, pair, tenor, basis_bp.")
    df = df.rename(columns={d:"date", p:"pair", t:"tenor", b:"basis_bp"})
    df["date"] = to_dt(df["date"])
    df["tenor"] = df["tenor"].map(standardize_tenor)
    df["basis_bp"] = safe_num(df["basis_bp"])
    return df.dropna(subset=["date","pair","tenor","basis_bp"]).sort_values(["date","pair","tenor"])

def load_ois(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"ccy","currency"); t = ncol(df,"tenor"); r = ncol(df,"rate_pct","ois_pct","r")
    if not (d and c and t and r): raise ValueError("ois.csv needs date, ccy, tenor, rate_pct.")
    df = df.rename(columns={d:"date", c:"ccy", t:"tenor", r:"rate_pct"})
    df["date"] = to_dt(df["date"])
    df["tenor"] = df["tenor"].map(standardize_tenor)
    df["rate_pct"] = safe_num(df["rate_pct"])
    return df.dropna(subset=["date","ccy","tenor","rate_pct"]).sort_values(["date","ccy","tenor"])

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); u = ncol(df,"USDJPY","usdjpy"); j = ncol(df,"JPYUSD","jpyusd")
    if not d: raise ValueError("fx.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = to_dt(df["date"])
    if u:
        df["USDJPY"] = safe_num(df[u])
    elif j:
        df["USDJPY"] = 1.0 / safe_num(df[j])
    else:
        num = [c for c in df.columns if c!="date"]
        if not num: raise ValueError("Provide USDJPY or JPYUSD in fx.csv.")
        df["USDJPY"] = safe_num(df[num[0]])
    df["r_fx"] = dlog(df["USDJPY"])
    return df[["date","USDJPY","r_fx"]].sort_values("date")

def load_forwards(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"tenor"); f = ncol(df,"fwd_points","fwd_jpy_points","points")
    if not (d and t and f): raise ValueError("forwards.csv needs date, tenor, fwd_points (JPY units).")
    df = df.rename(columns={d:"date", t:"tenor", f:"fwd_points"})
    df["date"] = to_dt(df["date"])
    df["tenor"] = df["tenor"].map(standardize_tenor)
    df["fwd_points"] = safe_num(df["fwd_points"])
    return df.dropna(subset=["date","tenor","fwd_points"]).sort_values(["date","tenor"])

def load_fraois(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); f = ncol(df,"usd_fraois_bp","fra_ois_bp","usd_fra_ois_bp")
    if not (d and f): raise ValueError("fraois.csv needs date and usd_fraois_bp.")
    df = df.rename(columns={d:"date", f:"usd_fraois_bp"})
    df["date"] = to_dt(df["date"])
    df["usd_fraois_bp"] = safe_num(df["usd_fraois_bp"])
    return df.dropna(subset=["date","usd_fraois_bp"]).sort_values("date")

def load_repo(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"ccy","currency"); r = ncol(df,"repo_pct","gc_pct","r")
    if not (d and c and r): raise ValueError("repo.csv needs date, ccy, repo_pct.")
    df = df.rename(columns={d:"date", c:"ccy", r:"repo_pct"})
    df["date"] = to_dt(df["date"])
    df["repo_pct"] = safe_num(df["repo_pct"])
    # pivot to USD, JPY repo
    pv = df.pivot_table(index="date", columns="ccy", values="repo_pct", aggfunc="mean").reset_index()
    pv.columns.name = None
    # build diff
    if {"USD","JPY"}.issubset(pv.columns):
        pv["repo_diff_bp"] = (pv["USD"] - pv["JPY"]) * 100.0
    return pv

def load_issuance(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"ccy","currency"); a = ncol(df,"amount_usd","amt","amount")
    if not (d and c and a): raise ValueError("issuance.csv needs date, ccy, amount_usd.")
    df = df.rename(columns={d:"date", c:"ccy", a:"amount_usd"})
    df["date"] = to_dt(df["date"])
    df["amount_usd"] = safe_num(df["amount_usd"])
    # keep USD & JPY issuance totals
    pv = (df.groupby(["date","ccy"], as_index=False)["amount_usd"].sum()
            .pivot_table(index="date", columns="ccy", values="amount_usd", aggfunc="sum").reset_index())
    pv.columns.name = None
    # supply proxy: USD issuance by JP banks could be in data; here generic totals
    if "USD" in pv.columns: pv["iss_usd_1m"] = pv["USD"].rolling(21, min_periods=5).sum()
    if "JPY" in pv.columns: pv["iss_jpy_1m"] = pv["JPY"].rolling(21, min_periods=5).sum()
    return pv

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"type"); l = ncol(df,"label","event")
    if not d: raise ValueError("events.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = to_dt(df["date"])
    if t: df = df.rename(columns={t:"type"})
    if l: df = df.rename(columns={l:"label"})
    if "type" not in df.columns: df["type"] = "EVENT"
    if "label" not in df.columns: df["label"] = df["type"]
    return df.sort_values("date")


# ----------------------------- panel & transforms -----------------------------

def resample_freq(df: pd.DataFrame, freq: str, how: str="last") -> pd.DataFrame:
    if df.empty: return df
    g = df.set_index("date")
    if how == "sum":
        out = g.resample("M").sum().reset_index()
    elif how == "mean":
        out = g.resample("M").mean().reset_index()
    else:
        out = g.resample("M").last().reset_index()
    out["date"] = eom(out["date"])
    return out

def build_panel(pair: str, tenors: List[str], freq: str,
                BASIS: pd.DataFrame, OIS: pd.DataFrame, FX: pd.DataFrame,
                FWD: pd.DataFrame, FRAOIS: pd.DataFrame, REPO: pd.DataFrame, ISS: pd.DataFrame) -> pd.DataFrame:
    df = BASIS[BASIS["pair"].str.upper()==pair.upper()].copy()
    df = df[df["tenor"].isin(tenors)]
    # pivot basis to columns per tenor for curve file; keep long for panel
    # Join drivers
    if not FX.empty: df = df.merge(FX, on="date", how="left")
    if not FRAOIS.empty: df = df.merge(FRAOIS, on="date", how="left")
    if not REPO.empty: df = df.merge(REPO[["date","repo_diff_bp"]], on="date", how="left")
    if not ISS.empty:
        cols = [c for c in ISS.columns if c!="date"]
        df = df.merge(ISS[["date"]+cols], on="date", how="left")

    # If monthly frequency requested, resample everything after expanding tenor columns
    if freq.lower().startswith("month"):
        # resample basis per tenor
        parts = []
        for t in tenors:
            dft = df[df["tenor"]==t].drop(columns=["tenor"]).set_index("date").resample("M").last().reset_index()
            dft["date"] = eom(dft["date"]); dft["tenor"] = t
            parts.append(dft)
        df = pd.concat(parts, ignore_index=True).sort_values(["date","tenor"])

    # Δbasis and slope (per tenor)
    df["dbasis_bp"] = df.groupby("tenor")["basis_bp"].diff()
    # slope e.g. 5Y-1Y, 1Y-3M if available
    if {"1Y","5Y"}.issubset(set(tenors)):
        y5 = df[df["tenor"]=="5Y"][["date","basis_bp"]].rename(columns={"basis_bp":"b5"})
        y1 = df[df["tenor"]=="1Y"][["date","basis_bp"]].rename(columns={"basis_bp":"b1"})
        sl = y5.merge(y1, on="date", how="inner")
        sl["slope_5y_1y_bp"] = sl["b5"] - sl["b1"]
        df = df.merge(sl[["date","slope_5y_1y_bp"]], on="date", how="left")

    # Seasonality flags: quarter-end, FY-end (Mar)
    end_of_month = df["date"].dt.is_month_end
    df["is_qend"] = ((df["date"].dt.month % 3)==0) & end_of_month
    df["is_fyend"] = (df["date"].dt.month==3) & end_of_month

    # Repo diff forward/backfill
    if "repo_diff_bp" in df.columns:
        df["repo_diff_bp"] = df["repo_diff_bp"].ffill()

    return df.sort_values(["date","tenor"])


# ----------------------------- CIP & funding costs -----------------------------

def compute_cip(pair: str, tenors: List[str], PANEL: pd.DataFrame, OIS: pd.DataFrame,
                FX: pd.DataFrame, FWD: pd.DataFrame) -> pd.DataFrame:
    """
    CIP deviation (bp):
      CIP_dev ≈ [ (F/S - 1)/T  -  (r_ccy1 - r_ccy2) ] * 1e4
      For USDJPY: ccy1=USD, ccy2=JPY. F,S in JPY/USD; T in years; r in decimal.
    If forwards absent: proxy CIP_dev ≈ basis_bp (uses market basis as deviation proxy).
    """
    pair = pair.upper()
    if FWD.empty or FX.empty or OIS.empty:
        # fallback using basis itself
        tmp = (PANEL[["date","tenor","basis_bp"]].copy()
               .assign(cip_dev_bp=lambda x: x["basis_bp"]))
        return tmp.rename(columns={"basis_bp":"basis_bp_input"})

    # Build OIS pivot by ccy & tenor on dates
    O = OIS.copy()
    O["tenor"] = O["tenor"].map(standardize_tenor)
    OUS = O[O["ccy"].str.upper()=="USD"].pivot_table(index="date", columns="tenor", values="rate_pct", aggfunc="mean")
    OJP = O[O["ccy"].str.upper()=="JPY"].pivot_table(index="date", columns="tenor", values="rate_pct", aggfunc="mean")

    # merge spot & forwards
    FX_ = FX[["date","USDJPY"]].copy().set_index("date")
    F = FWD.copy()
    F["tenor"] = F["tenor"].map(standardize_tenor)
    Fp = F.pivot_table(index="date", columns="tenor", values="fwd_points", aggfunc="mean")

    rows = []
    for dt in sorted(set(PANEL["date"])):
        s = FX_.reindex([dt]).squeeze()
        if pd.isna(s): continue
        for t in tenors:
            T = tenor_to_years(t)
            if not np.isfinite(T) or T<=0: continue
            fp = Fp.reindex([dt])[t].squeeze() if t in Fp.columns else np.nan
            r_usd = OUS.reindex([dt])[t].squeeze()/100.0 if t in OUS.columns else np.nan
            r_jpy = OJP.reindex([dt])[t].squeeze()/100.0 if t in OJP.columns else np.nan
            if np.any(np.isnan([fp, r_usd, r_jpy])): continue
            Fwd = float(s + fp)  # JPY per USD
            prem = (Fwd / float(s) - 1.0) / T
            cip_dev = (prem - (r_usd - r_jpy)) * 10000.0
            rows.append({"date": dt, "tenor": t, "cip_dev_bp": float(cip_dev)})
    C = pd.DataFrame(rows).sort_values(["date","tenor"])
    # attach basis observed (for comparison)
    B = PANEL[["date","tenor","basis_bp"]]
    out = C.merge(B, on=["date","tenor"], how="left")
    return out

def funding_costs(pair: str, tenors: List[str], PANEL: pd.DataFrame, OIS: pd.DataFrame,
                  CIP: pd.DataFrame) -> pd.DataFrame:
    """
    Hedged funding costs using CIP intuition:
      • Hedged USD → JPY cash yield ≈ JPY OIS + CIP_dev (bp)
      • Hedged JPY → USD cash yield ≈ USD OIS - CIP_dev (bp)
    If CIP unavailable, use market basis_bp as proxy for CIP_dev.
    """
    pair = pair.upper()
    if OIS.empty:
        return pd.DataFrame()

    O = OIS.copy()
    O["tenor"] = O["tenor"].map(standardize_tenor)
    OUS = O[O["ccy"].str.upper()=="USD"].pivot_table(index="date", columns="tenor", values="rate_pct", aggfunc="mean")
    OJP = O[O["ccy"].str.upper()=="JPY"].pivot_table(index="date", columns="tenor", values="rate_pct", aggfunc="mean")

    # pick CIP if available else basis as proxy
    if not CIP.empty and "cip_dev_bp" in CIP.columns:
        C = CIP.copy().rename(columns={"cip_dev_bp":"dev_bp"})
    else:
        C = PANEL[["date","tenor","basis_bp"]].rename(columns={"basis_bp":"dev_bp"})

    rows = []
    for _, r in C.iterrows():
        dt, t, dev = r["date"], r["tenor"], float(r["dev_bp"])
        T = tenor_to_years(t)
        if not np.isfinite(T): continue
        r_usd = OUS.reindex([dt])[t].squeeze() if t in OUS.columns else np.nan
        r_jpy = OJP.reindex([dt])[t].squeeze() if t in OJP.columns else np.nan
        if np.any(np.isnan([r_usd, r_jpy])): continue
        usd_to_jpy = r_jpy + dev/100.0
        jpy_to_usd = r_usd - dev/100.0
        rows.append({"date": dt, "tenor": t, "hedged_usd_to_jpy_pct": float(usd_to_jpy),
                     "hedged_jpy_to_usd_pct": float(jpy_to_usd),
                     "ois_usd_pct": float(r_usd), "ois_jpy_pct": float(r_jpy), "dev_bp": float(dev)})
    return pd.DataFrame(rows).sort_values(["date","tenor"])


# ----------------------------- regression: basis ~ drivers -----------------------------

def regress_basis(panel: pd.DataFrame, tenor: str, lags: int, min_obs: int=100) -> pd.DataFrame:
    """
    Δbasis_t (bp) ~ α + β1*ΔFRAOIS + β2*repo_diff_bp + β3*ΔFX + β4*issuance + β5*Qend + β6*FYend + ε
    HAC/NW SEs with 'lags'.
    """
    df = panel[panel["tenor"]==tenor].copy()
    if df.empty: return pd.DataFrame()
    y = df["dbasis_bp"]
    # drivers (differences where sensible)
    Xlist = [np.ones((len(df),1))]; names = ["const"]
    if "usd_fraois_bp" in df.columns:
        x = df["usd_fraois_bp"].diff()
        Xlist.append(x.values.reshape(-1,1)); names.append("d_usd_fraois_bp")
    if "repo_diff_bp" in df.columns:
        Xlist.append(df[["repo_diff_bp"]].values); names.append("repo_diff_bp")
    if "r_fx" in df.columns:
        Xlist.append(df[["r_fx"]].values); names.append("r_fx")
    if "iss_usd_1m" in df.columns:
        Xlist.append(df[["iss_usd_1m"]].fillna(0.0).values); names.append("iss_usd_1m")
    # seasonal dummies
    Xlist.append(df[["is_qend"]].astype(int).values); names.append("is_qend")
    Xlist.append(df[["is_fyend"]].astype(int).values); names.append("is_fyend")

    X = np.hstack(Xlist)
    mask = ~y.isna() & np.isfinite(X).all(axis=1)
    X = X[mask]; yv = y[mask].values.reshape(-1,1)
    if len(yv) < max(min_obs, 10 + X.shape[1]):
        return pd.DataFrame()
    beta, resid, XTX_inv = ols_beta_resid(X, yv)
    se = hac_se(X, resid, XTX_inv, L=max(1, lags))
    rows = []
    for i, nm in enumerate(names):
        b = float(beta[i,0]); s = float(se[i]); t = b/s if s>0 else np.nan
        rows.append({"tenor": tenor, "var": nm, "coef": b, "se": s, "t_stat": t, "n": int(len(yv)), "lags": int(lags)})
    return pd.DataFrame(rows)

def pick_coef(EL: pd.DataFrame, tenor: str, var: str, default: float) -> float:
    r = EL[(EL["tenor"]==tenor) & (EL["var"]==var)]
    return float(r["coef"].iloc[0]) if not r.empty else float(default)


# ----------------------------- carry / roll-down -----------------------------

def carry_roll(panel: pd.DataFrame, tenors: List[str], notional: float, roll_window: int=60) -> pd.DataFrame:
    """
    Approximate carry/roll for a long-basis position:
      carry ≈ slope * Δt * notional_annuity
    where slope ≈ (b_short - b_long)/(T_long - T_short),
    and notional_annuity ~ T (years). Δt is 1/roll_window of a year for daily data.
    This is a **very rough** approximation for intuition.
    """
    df = panel.copy()
    out = []
    # choose adjacent tenor pairs for slope
    pairs = [("3M","1Y"), ("1Y","5Y")]
    for short, long_ in pairs:
        if short not in tenors or long_ not in tenors: continue
        s = df[df["tenor"]==short][["date","basis_bp"]].rename(columns={"basis_bp":f"b_{short}"})
        l = df[df["tenor"]==long_][["date","basis_bp"]].rename(columns={"basis_bp":f"b_{long_}"})
        m = s.merge(l, on="date", how="inner")
        Tshort, Tlong = tenor_to_years(short), tenor_to_years(long_)
        m["slope_bp_per_year"] = (m[f"b_{short}"] - m[f"b_{long_}"]) / (Tlong - Tshort)
        # assume Δt = 1/roll_window years per step (daily approx)
        dt_year = 1.0 / max(1, roll_window)
        annuity = Tlong  # rough
        m["carry_bp_equiv"] = m["slope_bp_per_year"] * dt_year
        m["carry_jpy"] = (m["carry_bp_equiv"] / 10000.0) * notional * annuity
        m["pair"] = f"{short}-{long_}"
        out.append(m[["date","pair","slope_bp_per_year","carry_bp_equiv","carry_jpy"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------- event study -----------------------------

def quarter_end_events(dates: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(sorted(set(dates.dt.to_period("M").dt.to_timestamp("M"))))})
    df = df[df["date"].dt.is_quarter_end]
    df["type"] = "QUARTER_END"
    df["label"] = "Quarter End"
    return df

def event_study(panel: pd.DataFrame, events: pd.DataFrame, tenor: str, window: int=5) -> pd.DataFrame:
    """
    CAR of Δbasis over ±window days around events.
    """
    ser = (panel[panel["tenor"]==tenor].set_index("date")["dbasis_bp"].dropna())
    if ser.empty or events.empty: return pd.DataFrame()
    rows = []
    idx = ser.index
    for _, ev in events.iterrows():
        dt = pd.Timestamp(ev["date"])
        if dt not in idx:
            pos = idx.searchsorted(dt)
            if pos>=len(idx): continue
            dt = idx[pos]
        i = idx.get_loc(dt)
        L = max(0, i-window); R = min(len(idx)-1, i+window)
        car = float(ser.iloc[L:R+1].sum())
        rows.append({"event_date": str(idx[i].date()), "type": ev.get("type","EVENT"),
                     "label": ev.get("label", ev.get("type","EVENT")), "tenor": tenor, "CAR_dbasis_bp": car})
    return pd.DataFrame(rows).sort_values(["event_date","type"])


# ----------------------------- scenarios -----------------------------

def run_scenario(tenor: str, EL: pd.DataFrame, current_basis_bp: float,
                 shock_fraois_bp: float, shock_repo_diff_bp: float, shock_fx_pct: float,
                 shock_issuance_usd: float, notional: float, annuity_years: float) -> Dict[str, float]:
    """
    Linearized Δbasis from regression elasticities, then compute carry P&L and funding deltas.
    """
    d_fra = pick_coef(EL, tenor, "d_usd_fraois_bp", default=0.30)
    b_repo = pick_coef(EL, tenor, "repo_diff_bp", default=0.10)
    b_fx = pick_coef(EL, tenor, "r_fx", default=50.0)  # dbp per 1.0 in r_fx (~log return). Prior is large; adjust with data.
    b_iss = pick_coef(EL, tenor, "iss_usd_1m", default=0.0)

    # Convert FX % shock to log-return
    r_fx = np.log1p(shock_fx_pct/100.0)
    delta_bp = d_fra * shock_fraois_bp + b_repo * shock_repo_diff_bp + b_fx * r_fx + b_iss * shock_issuance_usd
    # Carry P&L approximation
    pnl_carry = (delta_bp / 10000.0) * notional * annuity_years
    return {"tenor": tenor, "delta_basis_bp": float(delta_bp),
            "new_basis_bp": float(current_basis_bp + delta_bp),
            "carry_pnl_jpy": float(pnl_carry)}


# ----------------------------- VaR/ES -----------------------------

def var_es_hist(series: pd.Series, alpha: float=0.05) -> Tuple[float,float]:
    x = series.dropna().values
    if len(x) < 100: return (np.nan, np.nan)
    q = np.quantile(x, alpha)
    es = x[x<=q].mean() if np.any(x<=q) else np.nan
    return float(q), float(es)


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    basis: str
    ois: Optional[str]
    fx: Optional[str]
    forwards: Optional[str]
    fraois: Optional[str]
    repo: Optional[str]
    issuance: Optional[str]
    events: Optional[str]
    pair: str
    tenors: List[str]
    freq: str
    roll: int
    reg_lags: int
    notional: float
    scenario_fraois: float
    scenario_repo_diff: float
    scenario_fx: float
    scenario_issuance: float
    tenor_for_var: str
    outdir: str
    start: Optional[str]
    end: Optional[str]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="JPY cross-currency basis toolkit")
    ap.add_argument("--basis", required=True)
    ap.add_argument("--ois", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--forwards", default="")
    ap.add_argument("--fraois", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--issuance", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--pair", default="USDJPY", choices=["USDJPY","EURJPY"])
    ap.add_argument("--tenors", default="1M,3M,6M,1Y,2Y,5Y")
    ap.add_argument("--freq", default="daily", choices=["daily","monthly"])
    ap.add_argument("--roll", type=int, default=60)
    ap.add_argument("--reg_lags", type=int, default=5)
    ap.add_argument("--notional", type=float, default=100_000_000.0)
    ap.add_argument("--scenario_fraois", type=float, default=20.0)
    ap.add_argument("--scenario_repo_diff", type=float, default=10.0)
    ap.add_argument("--scenario_fx", type=float, default=3.0)
    ap.add_argument("--scenario_issuance", type=float, default=2000.0, help="USD mm over last month")
    ap.add_argument("--tenor_for_var", default="3M")
    ap.add_argument("--outdir", default="out_basis")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    tenors = [standardize_tenor(x) for x in str(args.tenors).split(",") if x.strip()]

    BASIS = load_basis(args.basis)
    OIS   = load_ois(args.ois) if args.ois else pd.DataFrame()
    FX    = load_fx(args.fx) if args.fx else pd.DataFrame()
    FWD   = load_forwards(args.forwards) if args.forwards else pd.DataFrame()
    FRAOIS= load_fraois(args.fraois) if args.fraois else pd.DataFrame()
    REPO  = load_repo(args.repo) if args.repo else pd.DataFrame()
    ISS   = load_issuance(args.issuance) if args.issuance else pd.DataFrame()
    EVT   = load_events(args.events) if args.events else pd.DataFrame()

    # Time filters
    if args.start:
        s = to_dt(pd.Series([args.start])).iloc[0]
        for df in [BASIS,OIS,FX,FWD,FRAOIS,REPO,ISS,EVT]:
            if not df.empty: df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = to_dt(pd.Series([args.end])).iloc[0]
        for df in [BASIS,OIS,FX,FWD,FRAOIS,REPO,ISS,EVT]:
            if not df.empty: df.drop(df[df["date"] > e].index, inplace=True)

    # Panel
    P = build_panel(args.pair, tenors, args.freq, BASIS, OIS, FX, FWD, FRAOIS, REPO, ISS)
    if P.empty: raise ValueError("Panel is empty after filters/joins. Check inputs & pair/tenors.")
    P.to_csv(outdir / "panel_long.csv", index=False)

    # Wide curve export
    CURVE = (P.pivot_table(index="date", columns="tenor", values="basis_bp", aggfunc="last")
               .sort_index())
    CURVE.to_csv(outdir / "curve_wide.csv")

    # CIP
    CIP = compute_cip(args.pair, tenors, P, OIS, FX, FWD)
    if not CIP.empty: CIP.to_csv(outdir / "cip_deviation.csv", index=False)

    # Hedged funding
    FUND = funding_costs(args.pair, tenors, P, OIS, CIP)
    if not FUND.empty: FUND.to_csv(outdir / "funding_costs.csv", index=False)

    # Regressions per tenor
    EL_list = []
    for t in tenors:
        el = regress_basis(P, t, lags=int(args.reg_lags), min_obs=(60 if args.freq=="daily" else 24))
        if not el.empty: EL_list.append(el)
    EL = pd.concat(EL_list, ignore_index=True) if EL_list else pd.DataFrame()
    if not EL.empty: EL.to_csv(outdir / "reg_elasticities.csv", index=False)

    # Carry / roll-down
    CR = carry_roll(P, tenors, notional=float(args.notional), roll_window=int(args.roll))
    if not CR.empty: CR.to_csv(outdir / "carry_roll.csv", index=False)

    # Events: if none provided, synth quarter-ends
    if EVT.empty:
        EVT = quarter_end_events(P["date"])
    ES = event_study(P, EVT, tenor=(args.tenor_for_var if args.tenor_for_var in tenors else tenors[0]), window=5)
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Scenario
    tenor_use = args.tenor_for_var if args.tenor_for_var in tenors else tenors[0]
    latest_basis = float(P[P["tenor"]==tenor_use]["basis_bp"].dropna().iloc[-1])
    annuity = tenor_to_years(tenor_use) if np.isfinite(tenor_to_years(tenor_use)) else 1.0
    SC = run_scenario(
        tenor=tenor_use, EL=(EL if not EL.empty else pd.DataFrame()),
        current_basis_bp=latest_basis,
        shock_fraois_bp=float(args.scenario_fraois),
        shock_repo_diff_bp=float(args.scenario_repo_diff),
        shock_fx_pct=float(args.scenario_fx),
        shock_issuance_usd=float(args.scenario_issuance),
        notional=float(args.notional), annuity_years=float(annuity)
    )
    pd.DataFrame([SC]).to_csv(outdir / "scenarios.csv", index=False)

    # VaR/ES on Δbasis for chosen tenor
    series = P[P["tenor"]==tenor_use].set_index("date")["dbasis_bp"]
    var5, es5 = var_es_hist(series, alpha=0.05)
    ST = pd.DataFrame([{
        "tenor": tenor_use,
        "VaR5_dbasis_bp": var5,
        "ES5_dbasis_bp": es5,
        "mean_dbasis_bp": float(series.mean()) if series.notna().any() else np.nan,
        "sd_dbasis_bp": float(series.std(ddof=0)) if series.notna().any() else np.nan,
        "n_obs": int(series.dropna().shape[0])
    }])
    ST.to_csv(outdir / "stress_vares.csv", index=False)

    # Summary
    latest_row = P.sort_values("date").tail(1).iloc[0]
    summ = {
        "sample": {
            "start": str(P["date"].min().date()),
            "end": str(P["date"].max().date()),
            "freq": args.freq,
            "n_points": int(P["date"].nunique())
        },
        "pair": args.pair,
        "tenors": tenors,
        "latest": {
            "date": str(latest_row["date"].date()),
            "basis_bp_by_tenor": {t: (float(CURVE.loc[CURVE.index.max(), t]) if t in CURVE.columns and CURVE.index.size>0 and pd.notna(CURVE.loc[CURVE.index.max(), t]) else None) for t in tenors}
        },
        "scenario_inputs": {
            "fraois_bp": float(args.scenario_fraois),
            "repo_diff_bp": float(args.scenario_repo_diff),
            "fx_%": float(args.scenario_fx),
            "issuance_usd_mm": float(args.scenario_issuance),
            "tenor": tenor_use,
            "notional_jpy": float(args.notional)
        },
        "scenario_outputs": SC,
        "risk": {
            "VaR5_dbasis_bp": var5,
            "ES5_dbasis_bp": es5
        },
        "outputs": {
            "panel_long": "panel_long.csv",
            "curve_wide": "curve_wide.csv",
            "cip_deviation": "cip_deviation.csv" if not CIP.empty else None,
            "funding_costs": "funding_costs.csv" if not FUND.empty else None,
            "reg_elasticities": "reg_elasticities.csv" if not EL.empty else None,
            "carry_roll": "carry_roll.csv" if not CR.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "scenarios": "scenarios.csv",
            "stress_vares": "stress_vares.csv"
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2))

    # Config echo
    cfg = asdict(Config(
        basis=args.basis, ois=(args.ois or None), fx=(args.fx or None), forwards=(args.forwards or None),
        fraois=(args.fraois or None), repo=(args.repo or None), issuance=(args.issuance or None),
        events=(args.events or None), pair=args.pair, tenors=tenors, freq=args.freq, roll=int(args.roll),
        reg_lags=int(args.reg_lags), notional=float(args.notional),
        scenario_fraois=float(args.scenario_fraois), scenario_repo_diff=float(args.scenario_repo_diff),
        scenario_fx=float(args.scenario_fx), scenario_issuance=float(args.scenario_issuance),
        tenor_for_var=args.tenor_for_var, outdir=args.outdir,
        start=(args.start or None), end=(args.end or None)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== JPY Basis Toolkit ==")
    print(f"Pair: {args.pair} | Sample ({args.freq}): {summ['sample']['start']} → {summ['sample']['end']} ({summ['sample']['n_points']} pts)")
    print(f"Scenario ({tenor_use}): Δbasis {SC['delta_basis_bp']:.1f} bp → {SC['new_basis_bp']:.1f} bp | Carry P&L ≈ ¥{SC['carry_pnl_jpy']:,.0f}")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
