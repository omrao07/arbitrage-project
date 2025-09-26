#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cultural_content_exports.py — Soft-power exports: drivers, gravity, events & scenarios
-------------------------------------------------------------------------------------

What this does
==============
A research toolkit to analyze **cultural content exports** (music/film/TV/anime/games/books, etc.)
from a chosen **origin** country into destination markets. It builds a monthly panel from exports,
releases, platform & social signals, FX, policy, and macro data to:

1) Clean & align a market–month panel
   • Exports (USD), units, categories; origin → destination mapping
   • Platform access/penetration, social interest, FX, policy flags
   • Macro: GDP, population, internet penetration; optional distance & language

2) Event studies
   • Δ exports / social / platform ranks around major releases (±W months)

3) Elasticities (panel regressions with HAC/Newey–West SEs)
   • dlog(exports) ~ Σ_{l=0..L} [dlog(social) + platform + FX] + policy + seasonals + controls

4) Gravity model (cross-section or panel)
   • ln(1+exports) ~ ln(GDP_origin) + ln(GDP_dest) + ln(pop_dest) + ln(distance) + language + internet + platform

5) Scenarios & projections
   • FX shock (%), platform availability/penetration change, policy loosening/tightening
   • Forward path projections for exports by market (H months)

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--exports exports.csv            REQUIRED
  Columns (any subset):
    date, market_country[, dest_country], origin_country[, origin], category, revenue_usd[, exports_usd], units

--releases releases.csv          OPTIONAL
  Columns:
    date, title, category, origin_country, market_country[, country], channel[, platform], franchise[, ip], budget_usd

--social social.csv              OPTIONAL
  Columns:
    date, market_country[, country], category[, scope], search_idx[, social_idx]

--platform platform.csv          OPTIONAL
  Columns:
    date, market_country[, country], category[, scope], rank[, chart_rank], availability[, avail](0/1),
    subs_penetration[, subs_pct]   # % households/users with subscribing platform access

--fx fx.csv                      OPTIONAL
  Columns:
    date, country, usd_local[, fx_usd_local]      # USD per local currency (higher = stronger USD)

--policy policy.csv              OPTIONAL
  Columns:
    date, country, type[, event_type], label[, event], value  # e.g., QUOTA_UP / QUOTA_DOWN / BAN / VISA_EASE / PROMO

--macro macro.csv                OPTIONAL
  Columns:
    date[, year], country, gdp_usd[, gdp_ppp_usd], population[, pop], internet_pct[, internet_pen]

--distance distance.csv          OPTIONAL (for gravity)
  Columns:
    origin_country, market_country[, dest_country], distance_km[, km], common_language[, lang_shared](0/1)

CLI (key)
---------
--origin "KR"                    ISO country code of the content origin to analyze (e.g., KR, JP, US)
--category "ALL"                 Filter to a category (e.g., MUSIC, FILM, TV, ANIME, GAMES, BOOKS, or ALL)
--freq monthly                   Resample frequency (monthly only supported)
--lags 3                         Lags for distributed-lag elasticities
--event_window 3                 ± months around release for event studies
--topk_releases 50               Number of largest releases (by budget/proxy) to include in event study
--fx_shock_pct 0                 FX shock in % (higher = stronger USD vs local)
--avail_shift_pp 0.0             Platform availability shift in percentage points (additive to availability 0–1)
--subs_shift_pp 0.0              Platform penetration shift in pp
--policy_shift "NONE"            Policy scenario: NONE | LOOSEN | TIGHTEN (affects policy index)
--horizon 12                     Projection horizon in months from last sample month
--start / --end                  Sample filters (YYYY-MM-DD; month granularity OK)
--outdir out_culture             Output directory
--min_obs 60                     Minimum obs per market for regressions (reduce if sparse)

Outputs
-------
- panel.csv                      Cleaned market–month panel
- event_study.csv                Release-centered deltas by market
- elasticity.csv                 Distributed-lag regression (per-market & pooled)
- gravity.csv                    Gravity model coefficients
- scenarios.csv                  Forward projections under shocks
- summary.json                   Headline metrics
- config.json                    Echo of run configuration

Assumptions & notes
-------------------
• Monthly frequency; daily/weekly inputs are aggregated by month-end.  
• If `macro` or `distance` files are missing, related covariates are skipped.  
• Elasticities are reduced-form; validate robustness before use.  
• Currency: `revenue_usd` assumed already USD; FX enters as affordability/sentiment proxy in destination.

DISCLAIMER
----------
Research tooling with simplifying assumptions. Inspect data quality/coverage before operational decisions.
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

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def to_date(s: pd.Series) -> pd.Series:
    return to_dt(s).dt.date

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def eom(d: pd.Series) -> pd.Series:
    dt = to_dt(d)
    return (dt + pd.offsets.MonthEnd(0))

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s) - np.log(s.shift(1))

def winsor(s: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def standardize(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return s * 0.0
    return (s - mu) / sd

# OLS + HAC (Newey–West)
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

def load_exports(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df, "date"); mc = ncol(df, "market_country","dest_country","country")
    oc = ncol(df, "origin_country","origin")
    cat = ncol(df, "category")
    rev = ncol(df, "revenue_usd","exports_usd","usd")
    unt = ncol(df, "units")
    if not (d and mc and oc and rev):
        raise ValueError("exports.csv must include date, market_country, origin_country, revenue_usd.")
    df = df.rename(columns={d:"date", mc:"market_country", oc:"origin_country", rev:"revenue_usd"})
    if cat: df = df.rename(columns={cat:"category"})
    if unt: df = df.rename(columns={unt:"units"})
    df["date"] = eom(df["date"])
    df["market_country"] = df["market_country"].astype(str).str.upper().str.strip()
    df["origin_country"] = df["origin_country"].astype(str).str.upper().str.strip()
    df["revenue_usd"] = safe_num(df["revenue_usd"])
    if "units" in df.columns: df["units"] = safe_num(df["units"])
    if "category" not in df.columns: df["category"] = "ALL"
    return df

def load_releases(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); t = ncol(df, "title"); cat = ncol(df, "category")
    oc = ncol(df, "origin_country","origin"); mc = ncol(df, "market_country","country")
    ch = ncol(df, "channel","platform"); fr = ncol(df, "franchise","ip")
    bud = ncol(df, "budget_usd","budget")
    if not (d and t and oc): raise ValueError("releases.csv needs date, title, origin_country.")
    df = df.rename(columns={d:"date", t:"title"})
    if cat: df = df.rename(columns={cat:"category"})
    if oc: df = df.rename(columns={oc:"origin_country"})
    if mc: df = df.rename(columns={mc:"market_country"})
    if ch: df = df.rename(columns={ch:"channel"})
    if fr: df = df.rename(columns={fr:"franchise"})
    if bud: df = df.rename(columns={bud:"budget_usd"})
    df["date"] = eom(df["date"])
    for c in ["origin_country","market_country","category","channel","franchise"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    if "market_country" not in df.columns: df["market_country"] = "ALL"
    if "category" not in df.columns: df["category"] = "ALL"
    if "budget_usd" in df.columns: df["budget_usd"] = safe_num(df["budget_usd"])
    return df

def load_social(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); mc = ncol(df, "market_country","country")
    cat = ncol(df, "category","scope"); s = ncol(df, "search_idx","social_idx","index")
    if not (d and s): raise ValueError("social.csv needs date and search_idx.")
    df = df.rename(columns={d:"date", s:"search_idx"})
    if mc: df = df.rename(columns={mc:"market_country"})
    if cat: df = df.rename(columns={cat:"category"})
    df["date"] = eom(df["date"])
    if "market_country" not in df.columns: df["market_country"] = "ALL"
    if "category" not in df.columns: df["category"] = "ALL"
    df["search_idx"] = safe_num(df["search_idx"])
    df["market_country"] = df["market_country"].astype(str).str.upper().str.strip()
    df["category"] = df["category"].astype(str).str.upper().str.strip()
    return df

def load_platform(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); mc = ncol(df, "market_country","country")
    cat = ncol(df, "category","scope")
    rk = ncol(df, "rank","chart_rank"); av = ncol(df, "availability","avail")
    sp = ncol(df, "subs_penetration","subs_pct","penetration")
    if not d: raise ValueError("platform.csv needs date.")
    df = df.rename(columns={d:"date"})
    if mc: df = df.rename(columns={mc:"market_country"})
    if cat: df = df.rename(columns={cat:"category"})
    if rk: df = df.rename(columns={rk:"rank"})
    if av: df = df.rename(columns={av:"availability"})
    if sp: df = df.rename(columns={sp:"subs_penetration"})
    df["date"] = eom(df["date"])
    df["market_country"] = df.get("market_country","ALL").astype(str).str.upper().str.strip()
    df["category"] = df.get("category","ALL").astype(str).str.upper().str.strip()
    if "rank" in df.columns: df["rank"] = safe_num(df["rank"])
    if "availability" in df.columns: df["availability"] = safe_num(df["availability"]).clip(0,1)
    if "subs_penetration" in df.columns: df["subs_penetration"] = safe_num(df["subs_penetration"]).clip(0,100)
    return df

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); c = ncol(df, "country"); fx = ncol(df, "usd_local","fx_usd_local","usd_per_local")
    if not (d and c and fx): raise ValueError("fx.csv needs date, country, usd_local.")
    df = df.rename(columns={d:"date", c:"country", fx:"usd_local"})
    df["date"] = eom(df["date"])
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["usd_local"] = safe_num(df["usd_local"])
    return df

def load_policy(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); c = ncol(df, "country"); typ = ncol(df, "type","event_type")
    lab = ncol(df, "label","event"); val = ncol(df, "value")
    if not (d and c and (typ or lab)): raise ValueError("policy.csv needs date, country, and type/label.")
    df = df.rename(columns={d:"date", c:"country"})
    if typ: df = df.rename(columns={typ:"type"})
    if lab: df = df.rename(columns={lab:"label"})
    if val: df = df.rename(columns={val:"value"})
    df["date"] = eom(df["date"])
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["type"] = df.get("type","").astype(str).str.upper().str.strip()
    df["label"] = df.get("label","").astype(str).str.upper().str.strip()
    if "value" in df.columns: df["value"] = safe_num(df["value"])
    return df

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date","year"); c = ncol(df, "country")
    g = ncol(df, "gdp_usd","gdp_ppp_usd"); p = ncol(df, "population","pop")
    it = ncol(df, "internet_pct","internet_pen")
    if not (d and c): raise ValueError("macro.csv needs date/year and country.")
    df = df.rename(columns={d:"date", c:"country"})
    if g: df = df.rename(columns={g:"gdp_usd"})
    if p: df = df.rename(columns={p:"population"})
    if it: df = df.rename(columns={it:"internet_pct"})
    df["date"] = eom(df["date"])
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    for k in ["gdp_usd","population","internet_pct"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_distance(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    o = ncol(df, "origin_country"); m = ncol(df, "market_country","dest_country"); km = ncol(df, "distance_km","km")
    cl = ncol(df, "common_language","lang_shared")
    if not (o and m and km): raise ValueError("distance.csv needs origin_country, market_country, distance_km.")
    df = df.rename(columns={o:"origin_country", m:"market_country", km:"distance_km"})
    if cl: df = df.rename(columns={cl:"common_language"})
    df["origin_country"] = df["origin_country"].astype(str).str.upper().str.strip()
    df["market_country"] = df["market_country"].astype(str).str.upper().str.strip()
    df["distance_km"] = safe_num(df["distance_km"])
    if "common_language" in df.columns: df["common_language"] = safe_num(df["common_language"]).clip(0,1)
    return df


# ----------------------------- panel construction -----------------------------

def build_panel(EXP: pd.DataFrame, REL: pd.DataFrame, SOC: pd.DataFrame, PLT: pd.DataFrame,
                FX: pd.DataFrame, POL: pd.DataFrame, MAC: pd.DataFrame,
                origin: str, category: str) -> pd.DataFrame:
    # Filter by origin & category
    df = EXP[EXP["origin_country"] == origin].copy()
    if category and category.upper() != "ALL":
        df = df[df["category"].str.upper() == category.upper()]
    # monthly exports by market
    agg = (df.groupby(["date","market_country"], as_index=False)
             .agg(revenue_usd=("revenue_usd","sum"),
                  units=("units","sum") if "units" in df.columns else ("revenue_usd","size")))
    panel = agg.copy()
    # Attach social
    if not SOC.empty:
        soc = SOC.copy()
        if category and category.upper() != "ALL":
            soc = soc[(soc["category"]==category.upper()) | (soc["category"]=="ALL")]
        panel = panel.merge(soc.groupby(["date","market_country"], as_index=False)["search_idx"].mean(),
                            on=["date","market_country"], how="left")
    # Attach platform
    if not PLT.empty:
        pl = PLT.copy()
        if category and category.upper() != "ALL":
            pl = pl[(pl["category"]==category.upper()) | (pl["category"]=="ALL")]
        gp = pl.groupby(["date","market_country"], as_index=False).agg(
            availability=("availability","mean") if "availability" in pl.columns else ("date","size"),
            subs_penetration=("subs_penetration","mean") if "subs_penetration" in pl.columns else ("date","size"),
            rank=("rank","mean") if "rank" in pl.columns else ("date","size")
        )
        panel = panel.merge(gp, on=["date","market_country"], how="left")
    # FX (destination affordability proxy)
    if not FX.empty:
        panel = panel.merge(FX.rename(columns={"country":"market_country"})[["date","market_country","usd_local"]],
                            on=["date","market_country"], how="left")
    # Macro (destination)
    if not MAC.empty:
        m = MAC.rename(columns={"country":"market_country"})
        panel = panel.merge(m[["date","market_country"] + [c for c in ["gdp_usd","population","internet_pct"] if c in m.columns]],
                            on=["date","market_country"], how="left")
    # Policy: create a policy index per month (loosen positive, tighten negative)
    if not POL.empty:
        pol = POL.rename(columns={"country":"market_country"}).copy()
        pol["sign"] = 0.0
        pol.loc[pol["type"].str.contains("LOOSEN|PROMO|VISA", na=False), "sign"] = 1.0
        pol.loc[pol["type"].str.contains("TIGHTEN|BAN|QUOTA_DOWN", na=False), "sign"] = -1.0
        pol_month = pol.groupby(["date","market_country"], as_index=False)["sign"].sum().rename(columns={"sign":"policy_index"})
        panel = panel.merge(pol_month, on=["date","market_country"], how="left")
    # Releases: create count of origin releases per market-month (or ALL markets if unspecified)
    if not REL.empty:
        rel = REL.copy()
        rel = rel[rel["origin_country"]==origin]
        if category and category.upper() != "ALL":
            rel = rel[(rel["category"]==category.upper()) | (rel["category"]=="ALL")]
        relc = rel.groupby(["date","market_country"], as_index=False).agg(
            n_releases=("title","nunique"),
            big_release_score=("budget_usd","sum") if "budget_usd" in rel.columns else ("title","size")
        )
        panel = panel.merge(relc, on=["date","market_country"], how="left")
    # Fill NA
    for c in ["search_idx","availability","subs_penetration","rank","usd_local","gdp_usd","population","internet_pct","policy_index","n_releases","big_release_score"]:
        if c in panel.columns:
            panel[c] = panel[c].astype(float)
    panel = panel.sort_values(["market_country","date"]).reset_index(drop=True)
    # Derived
    panel["dlog_exports"] = panel.groupby("market_country")["revenue_usd"].apply(dlog).reset_index(level=0, drop=True)
    if "search_idx" in panel.columns:
        panel["dlog_social"] = panel.groupby("market_country")["search_idx"].apply(dlog).reset_index(level=0, drop=True)
    if "usd_local" in panel.columns:
        panel["dlog_fx"] = panel.groupby("market_country")["usd_local"].apply(dlog).reset_index(level=0, drop=True)
    if "rank" in panel.columns:
        panel["rank_z"] = panel.groupby("market_country")["rank"].transform(standardize)
    if "subs_penetration" in panel.columns:
        panel["subs_pen_z"] = panel.groupby("market_country")["subs_penetration"].transform(standardize)
    # Month seasonality
    panel["month"] = panel["date"].dt.month
    return panel

# ----------------------------- event study -----------------------------

def pick_top_releases(REL: pd.DataFrame, origin: str, category: str, topk: int) -> pd.DataFrame:
    if REL.empty: return pd.DataFrame()
    r = REL[REL["origin_country"]==origin].copy()
    if category and category.upper() != "ALL":
        r = r[(r["category"]==category.upper()) | (r["category"]=="ALL")]
    if "budget_usd" in r.columns:
        r["size"] = safe_num(r["budget_usd"]).fillna(0.0)
    else:
        r["size"] = 1.0
    r = r.sort_values("size", ascending=False).head(int(topk))
    # ensure market_country present
    if "market_country" not in r.columns:
        r["market_country"] = "ALL"
    return r[["date","market_country","title","category","channel","franchise","size"]].copy()

def event_study(panel: pd.DataFrame, rel_top: pd.DataFrame, window: int) -> pd.DataFrame:
    if panel.empty or rel_top.empty: return pd.DataFrame()
    idx = panel.set_index(["market_country","date"]).sort_index()
    rows = []
    for _, ev in rel_top.iterrows():
        mk = ev["market_country"]
        dates = sorted(idx.loc[mk].index) if mk in idx.index.levels[0] else []
        if not dates: 
            continue
        # nearest month to release
        dt = pd.Timestamp(ev["date"])
        t0 = min(dates, key=lambda x: abs(pd.Timestamp(x)-dt))
        t_pos = dates.index(t0)
        for h in range(-window, window+1):
            j = t_pos + h
            if j < 0 or j >= len(dates): continue
            dd = dates[j]
            row = {"market_country": mk, "title": ev["title"], "h": h, "date": str(pd.Timestamp(dd).date())}
            # deltas vs mean of pre-window
            g = idx.loc[(mk, dd)]
            row["dlog_exports"] = float(g.get("dlog_exports", np.nan))
            row["dlog_social"]  = float(g.get("dlog_social", np.nan)) if "dlog_social" in g.index else np.nan
            row["rank_z"]       = float(g.get("rank_z", np.nan)) if "rank_z" in g.index else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty: return df
    out = []
    for (mk, title), g in df.groupby(["market_country","title"]):
        base = g[g["h"]<0][["dlog_exports","dlog_social","rank_z"]].mean(numeric_only=True)
        for _, r in g.iterrows():
            out.append({
                "market_country": mk, "title": title, "h": int(r["h"]), "date": r["date"],
                "delta_dlog_exports": float(r.get("dlog_exports", np.nan) - base.get("dlog_exports", np.nan)) if pd.notna(r.get("dlog_exports", np.nan)) else np.nan,
                "delta_dlog_social":  float(r.get("dlog_social", np.nan)  - base.get("dlog_social", np.nan))  if pd.notna(r.get("dlog_social", np.nan)) else np.nan,
                "delta_rank_z":       float(r.get("rank_z", np.nan)       - base.get("rank_z", np.nan))       if pd.notna(r.get("rank_z", np.nan)) else np.nan
            })
    return pd.DataFrame(out).sort_values(["market_country","title","h"])

# ----------------------------- elasticities -----------------------------

def build_dlag_matrix(g: pd.DataFrame, L: int, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.DataFrame(index=g.index)
    names = []
    for c in cols:
        if c not in g.columns: 
            continue
        for l in range(0, L+1):
            nm = f"{c}_l{l}"
            X[nm] = g[c].shift(l)
            names.append(nm)
    return X, names

def elasticities(panel: pd.DataFrame, L: int, min_obs: int) -> pd.DataFrame:
    if panel.empty or "dlog_exports" not in panel.columns: return pd.DataFrame()
    rows = []
    # per-market regressions
    for mk, g in panel.groupby("market_country"):
        g = g.sort_values("date")
        y = g["dlog_exports"]
        X1, names1 = build_dlag_matrix(g, L, cols=[c for c in ["dlog_social","dlog_fx"] if c in g.columns])
        # levels (contemp)
        add = []
        if "availability" in g.columns: add.append("availability")
        if "subs_pen_z" in g.columns: add.append("subs_pen_z")
        if "rank_z" in g.columns: add.append("rank_z")
        if "policy_index" in g.columns: add.append("policy_index")
        X2 = g[add] if add else pd.DataFrame(index=g.index)
        # month dummies (seasonality)
        D = pd.get_dummies(g["month"].astype(int), prefix="m", drop_first=True)
        X = pd.concat([pd.Series(1.0, index=g.index, name="const"), X1, X2, D], axis=1).dropna()
        YY = y.loc[X.index]
        if X.shape[0] < max(min_obs, 5*X.shape[1]//3):
            continue
        beta, resid, XTX_inv = ols_beta_resid(X.values, YY.values.reshape(-1,1))
        se = hac_se(X.values, resid, XTX_inv, L=max(6, L))
        for i, nm in enumerate(X.columns):
            rows.append({"market_country": mk, "var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                         "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "n": int(X.shape[0]), "lags": int(L)})
        # cumulative elasticities for dlog_social and dlog_fx
        for base in ["dlog_social","dlog_fx"]:
            idxs = [i for i, nm in enumerate(X.columns) if nm.startswith(base+"_l")]
            if idxs:
                bsum = float(np.sum([beta[i,0] for i in idxs]))
                sesq = float(np.sum([se[i]**2 for i in idxs]))
                rows.append({"market_country": mk, "var": base+"_cum_0..L", "coef": bsum, "se": np.sqrt(sesq),
                             "t_stat": bsum/np.sqrt(sesq) if sesq>0 else np.nan, "n": int(X.shape[0]), "lags": int(L)})
    return pd.DataFrame(rows)

# ----------------------------- gravity model -----------------------------

def gravity_model(panel: pd.DataFrame, DIST: pd.DataFrame, origin: str) -> pd.DataFrame:
    if panel.empty: return pd.DataFrame()
    # use last 12 months average per market
    recent = panel.groupby("market_country").tail(12) if panel["market_country"].nunique() > 0 else panel
    g = (recent.groupby("market_country", as_index=False)
               .agg(exports_usd=("revenue_usd","mean"),
                    gdp_usd=("gdp_usd","mean") if "gdp_usd" in panel.columns else ("revenue_usd","size"),
                    population=("population","mean") if "population" in panel.columns else ("revenue_usd","size"),
                    internet_pct=("internet_pct","mean") if "internet_pct" in panel.columns else ("revenue_usd","size"),
                    availability=("availability","mean") if "availability" in panel.columns else ("revenue_usd","size")))
    # merge distance
    if not DIST.empty:
        dd = DIST[DIST["origin_country"]==origin][["market_country","distance_km"] + (["common_language"] if "common_language" in DIST.columns else [])]
        g = g.merge(dd, on="market_country", how="left")
    # build design
    y = np.log1p(g["exports_usd"].astype(float).values).reshape(-1,1)
    cols = []
    Xparts = [np.ones((g.shape[0],1))]
    cols.append("const")
    if "gdp_usd" in g.columns:
        Xparts.append(np.log(g["gdp_usd"].replace(0,np.nan)).fillna(0).values.reshape(-1,1)); cols.append("ln_gdp_dest")
    if "population" in g.columns:
        Xparts.append(np.log(g["population"].replace(0,np.nan)).fillna(0).values.reshape(-1,1)); cols.append("ln_pop_dest")
    if "distance_km" in g.columns:
        Xparts.append(np.log(g["distance_km"].replace(0,np.nan)).fillna(g["distance_km"].median()).values.reshape(-1,1)); cols.append("ln_distance")
    if "internet_pct" in g.columns:
        Xparts.append(g["internet_pct"].fillna(g["internet_pct"].median()).values.reshape(-1,1)); cols.append("internet_pct")
    if "availability" in g.columns:
        Xparts.append(g["availability"].fillna(0).values.reshape(-1,1)); cols.append("platform_availability")
    if "common_language" in g.columns:
        Xparts.append(g["common_language"].fillna(0).values.reshape(-1,1)); cols.append("common_language")
    X = np.concatenate(Xparts, axis=1)
    beta, resid, XTX_inv = ols_beta_resid(X, y)
    se = hac_se(X, resid, XTX_inv, L=0)
    out = []
    for i, nm in enumerate(cols):
        out.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i]), "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                    "n_markets": int(g.shape[0])})
    return pd.DataFrame(out)

# ----------------------------- scenarios -----------------------------

def pick_cum_beta(EL: pd.DataFrame, mk: str, var: str) -> Optional[float]:
    r = EL[(EL["market_country"]==mk) & (EL["var"]==var)]
    if not r.empty: return float(r["coef"].iloc[0])
    # fallback to pooled average
    r2 = EL[EL["var"]==var]
    return float(r2["coef"].mean()) if not r2.empty else None

def scenario(panel: pd.DataFrame, EL: pd.DataFrame,
             fx_shock_pct: float, avail_shift_pp: float, subs_shift_pp: float, policy_shift: str,
             horizon: int) -> pd.DataFrame:
    if panel.empty: return pd.DataFrame()
    last_dt = panel["date"].max()
    months = pd.date_range(last_dt + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
    rows = []
    for mk, g in panel.groupby("market_country"):
        g = g.sort_values("date")
        base_rev = float(g["revenue_usd"].iloc[-1]) if not g.empty else np.nan
        b_soc = pick_cum_beta(EL, mk, "dlog_social_cum_0..L") or 0.15
        b_fx  = pick_cum_beta(EL, mk, "dlog_fx_cum_0..L") or -0.10
        b_av  = float(EL[(EL["market_country"]==mk) & (EL["var"]=="availability")]["coef"].mean()) if "availability" in panel.columns else 0.05
        b_sub = float(EL[(EL["market_country"]==mk) & (EL["var"]=="subs_pen_z")]["coef"].mean()) if "subs_pen_z" in panel.columns else 0.02
        b_pol = float(EL[(EL["market_country"]==mk) & (EL["var"]=="policy_index")]["coef"].mean()) if "policy_index" in panel.columns else 0.03
        # scenario shocks (assumed constant over horizon)
        dlog_fx = np.log(1.0 + fx_shock_pct/100.0)
        d_av = avail_shift_pp/100.0
        d_subz = (subs_shift_pp/100.0)  # approx z-shift if std~1; treat as small
        d_pol = {"NONE":0.0, "LOOSEN": +1.0, "TIGHTEN": -1.0}.get(str(policy_shift).upper(), 0.0)
        rev = base_rev
        for dt in months:
            dlog_rev = (b_fx * dlog_fx) + (b_av * d_av) + (b_sub * d_subz) + (b_pol * d_pol)
            # social boost set to 0 here (unless the user adds a social shock in future)
            rev = rev * np.exp(dlog_rev) if rev==rev else np.nan
            rows.append({"market_country": mk, "date": dt.date(), "revenue_usd": float(rev) if rev==rev else np.nan,
                         "dlog_rev": float(dlog_rev), "fx_shock_pct": fx_shock_pct,
                         "avail_shift_pp": avail_shift_pp, "subs_shift_pp": subs_shift_pp, "policy_shift": policy_shift})
    return pd.DataFrame(rows).sort_values(["market_country","date"])

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    exports: str
    releases: Optional[str]
    social: Optional[str]
    platform: Optional[str]
    fx: Optional[str]
    policy: Optional[str]
    macro: Optional[str]
    distance: Optional[str]
    origin: str
    category: str
    freq: str
    lags: int
    event_window: int
    topk_releases: int
    fx_shock_pct: float
    avail_shift_pp: float
    subs_shift_pp: float
    policy_shift: str
    horizon: int
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cultural content exports — drivers, gravity, events & scenarios")
    ap.add_argument("--exports", required=True)
    ap.add_argument("--releases", default="")
    ap.add_argument("--social", default="")
    ap.add_argument("--platform", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--policy", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--distance", default="")
    ap.add_argument("--origin", required=True, help="ISO country code of origin, e.g., KR/JP/US")
    ap.add_argument("--category", default="ALL")
    ap.add_argument("--freq", default="monthly", choices=["monthly"])
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--topk_releases", type=int, default=50)
    ap.add_argument("--fx_shock_pct", type=float, default=0.0)
    ap.add_argument("--avail_shift_pp", type=float, default=0.0)
    ap.add_argument("--subs_shift_pp", type=float, default=0.0)
    ap.add_argument("--policy_shift", default="NONE", choices=["NONE","LOOSEN","TIGHTEN"])
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_culture")
    ap.add_argument("--min_obs", type=int, default=60)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    EXP = load_exports(args.exports)
    REL = load_releases(args.releases) if args.releases else pd.DataFrame()
    SOC = load_social(args.social) if args.social else pd.DataFrame()
    PLT = load_platform(args.platform) if args.platform else pd.DataFrame()
    FX  = load_fx(args.fx) if args.fx else pd.DataFrame()
    POL = load_policy(args.policy) if args.policy else pd.DataFrame()
    MAC = load_macro(args.macro) if args.macro else pd.DataFrame()
    DST = load_distance(args.distance) if args.distance else pd.DataFrame()

    # Time filters
    if args.start:
        s = eom(pd.Series([args.start])).iloc[0]
        for df in [EXP, REL, SOC, PLT, FX, POL, MAC]:
            if not df.empty: df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = eom(pd.Series([args.end])).iloc[0]
        for df in [EXP, REL, SOC, PLT, FX, POL, MAC]:
            if not df.empty: df.drop(df[df["date"] > e].index, inplace=True)

    # Panel
    PANEL = build_panel(EXP, REL, SOC, PLT, FX, POL, MAC, origin=args.origin.upper(), category=args.category.upper())
    if PANEL.empty:
        raise ValueError("Panel is empty. Check origin/category filters and input coverage.")
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Event study
    EVTOP = pick_top_releases(REL, args.origin.upper(), args.category.upper(), topk=int(args.topk_releases)) if not REL.empty else pd.DataFrame()
    ES = event_study(PANEL, EVTOP, window=int(args.event_window)) if not EVTOP.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Elasticities
    EL = elasticities(PANEL, L=int(args.lags), min_obs=int(args.min_obs))
    if not EL.empty: EL.to_csv(outdir / "elasticity.csv", index=False)

    # Gravity
    GR = gravity_model(PANEL, DST, origin=args.origin.upper()) if not DST.empty else pd.DataFrame()
    if not GR.empty: GR.to_csv(outdir / "gravity.csv", index=False)

    # Scenarios
    SCN = scenario(PANEL, EL if not EL.empty else pd.DataFrame(), fx_shock_pct=float(args.fx_shock_pct),
                   avail_shift_pp=float(args.avail_shift_pp), subs_shift_pp=float(args.subs_shift_pp),
                   policy_shift=str(args.policy_shift), horizon=int(args.horizon))
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    key = {
        "origin": args.origin.upper(),
        "category": args.category.upper(),
        "markets": int(PANEL["market_country"].nunique()),
        "sample_start": str(PANEL["date"].min().date()),
        "sample_end": str(PANEL["date"].max().date()),
        "has_social": "dlog_social" in PANEL.columns,
        "has_platform": "availability" in PANEL.columns or "rank" in PANEL.columns,
        "has_fx": "dlog_fx" in PANEL.columns,
        "has_macro": ("gdp_usd" in PANEL.columns) or ("population" in PANEL.columns),
        "has_policy": "policy_index" in PANEL.columns,
        "has_gravity": not GR.empty,
        "has_event_study": not ES.empty,
        "has_elasticity": not EL.empty,
        "has_scenarios": not SCN.empty
    }
    summary = {
        "key": key,
        "files": {
            "panel": "panel.csv",
            "event_study": "event_study.csv" if not ES.empty else None,
            "elasticity": "elasticity.csv" if not EL.empty else None,
            "gravity": "gravity.csv" if not GR.empty else None,
            "scenarios": "scenarios.csv" if not SCN.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        exports=args.exports, releases=(args.releases or None), social=(args.social or None),
        platform=(args.platform or None), fx=(args.fx or None), policy=(args.policy or None),
        macro=(args.macro or None), distance=(args.distance or None), origin=args.origin.upper(),
        category=args.category.upper(), freq=args.freq, lags=int(args.lags), event_window=int(args.event_window),
        topk_releases=int(args.topk_releases), fx_shock_pct=float(args.fx_shock_pct),
        avail_shift_pp=float(args.avail_shift_pp), subs_shift_pp=float(args.subs_shift_pp),
        policy_shift=args.policy_shift, horizon=int(args.horizon), start=(args.start or None),
        end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Cultural Content Exports ==")
    print(f"Origin: {key['origin']} | Category: {key['category']} | Markets: {key['markets']} | Sample: {key['sample_start']} → {key['sample_end']}")
    if key["has_elasticity"]:
        print("Elasticities estimated (see elasticity.csv).")
    if key["has_gravity"]:
        print("Gravity model estimated (see gravity.csv).")
    if key["has_event_study"]:
        print(f"Event study available (±{args.event_window} months).")
    if key["has_scenarios"]:
        print(f"Scenario horizon: {args.horizon} months | FX {args.fx_shock_pct:+.1f}% | avail +{args.avail_shift_pp:.1f}pp | subs +{args.subs_shift_pp:.1f}pp | policy {args.policy_shift}")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
