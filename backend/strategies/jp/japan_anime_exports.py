#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
japan_anime_exports.py — Quantifying Japan's anime export revenues, drivers, events & scenarios
-----------------------------------------------------------------------------------------------

What this does
==============
Builds a **monthly destination-country panel** for Japan’s anime exports and related proxies,
then estimates elasticities, runs event studies, computes a market opportunity/risk score,
and projects forward scenarios.

It ingests:
- revenue by destination (licensing/streaming/box office/merch proxies)
- trade-based proxies (manga/books, toys/figures, BD/DVD) by HS buckets (optional)
- platform availability (catalog size, simulcasts, subscription price)
- FX (USDJPY or JPYUSD; optional JPY REER)
- demand proxies (GDP/income/entertainment spend/internet)
- global/region interest (Google Trends/YouTube views for “anime”)
- shipping/freight (for physical merch), policy/censorship, piracy indicators
- releases/events (major titles/platform launches), with optional country targeting

Core outputs
------------
1) panel.csv                        — destination–month feature matrix
2) elasticity_revenue.csv           — HAC/NW elasticities for revenue
3) event_study.csv                  — CARs of Δlog(revenue) around releases & platform shocks
4) risk_opportunity.csv             — 0–100 composite (market size, openness, platform, piracy, FX)
5) scenarios.csv                    — H-month projections under user shocks
6) stress_vares.csv                 — Monte Carlo VaR/ES for monthly export revenue
7) summary.json, config.json        — run metadata

DISCLAIMER: Research tooling with simplifying assumptions. Validate with local data before use.
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

def winsor(x: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

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

def load_revenue(path: str) -> pd.DataFrame:
    """
    revenue.csv columns (any subset; monthly or higher freq):
      date, country[, market], rev_total_usd, license_usd, streaming_usd, boxoffice_usd, merch_usd
    """
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country","market")
    if not (d and c): raise ValueError("revenue.csv needs date and country.")
    df = df.rename(columns={d:"date", c:"country"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    ren = {
        ncol(df,"rev_total_usd","revenue_usd","export_revenue_usd"): "rev_total_usd",
        ncol(df,"license_usd","licensing_usd"): "license_usd",
        ncol(df,"streaming_usd","svod_usd","avod_usd","ott_usd"): "streaming_usd",
        ncol(df,"boxoffice_usd","theatrical_usd"): "boxoffice_usd",
        ncol(df,"merch_usd","merchandise_usd"): "merch_usd",
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for k in ["rev_total_usd","license_usd","streaming_usd","boxoffice_usd","merch_usd"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # derive rev_total if missing
    if "rev_total_usd" not in df.columns:
        parts = [c for c in ["license_usd","streaming_usd","boxoffice_usd","merch_usd"] if c in df.columns]
        if parts:
            df["rev_total_usd"] = df[parts].sum(axis=1)
    return (df.groupby(["date","country"], as_index=False).sum(numeric_only=True))

def load_trade(path: Optional[str]) -> pd.DataFrame:
    """
    trade.csv columns (optional):
      date, country, hs_code[, commodity], value_usd
      map HS → proxies: manga/books (4901/490199), toys/figures (9503), video media (852349/8523*)
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); hs = ncol(df,"hs_code","hs6","hs")
    v = ncol(df,"value_usd","export_usd","cif_usd")
    if not (d and c and hs and v): raise ValueError("trade.csv needs date,country,hs_code,value_usd.")
    df = df.rename(columns={d:"date", c:"country", hs:"hs_code", v:"value_usd"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str); df["value_usd"] = safe_num(df["value_usd"])
    df["hs_code"] = df["hs_code"].astype(str)
    # tag buckets
    def tag(h:str) -> str:
        h = str(h)
        if h.startswith(("9503","9504")): return "toys_figures_usd"
        if h.startswith(("4901","4902","490199")): return "manga_books_usd"
        if h.startswith(("8523","852349","852380")): return "video_media_usd"
        return "other_usd"
    df["bucket"] = df["hs_code"].apply(tag)
    T = (df.groupby(["date","country","bucket"], as_index=False)["value_usd"].sum())
    PIV = T.pivot_table(index=["date","country"], columns="bucket", values="value_usd", aggfunc="sum").reset_index()
    for col in ["toys_figures_usd","manga_books_usd","video_media_usd"]:
        if col not in PIV.columns: PIV[col] = np.nan
    return PIV

def load_platform(path: Optional[str]) -> pd.DataFrame:
    """
    platform.csv columns:
      date, country, titles_available[, catalog_anime_titles], simulcast_titles, platform_count, sub_price_usd
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    if not (d and c): raise ValueError("platform.csv needs date,country.")
    df = df.rename(columns={d:"date", c:"country"})
    ren = {
        ncol(df,"titles_available","catalog_anime_titles","anime_titles"): "titles_available",
        ncol(df,"simulcast_titles","simulcasts","same_day_titles"): "simulcast_titles",
        ncol(df,"platform_count","services_count","ott_count"): "platform_count",
        ncol(df,"sub_price_usd","subscription_price_usd","price_usd"): "sub_price_usd",
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    for k in ["titles_available","simulcast_titles","platform_count","sub_price_usd"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_fx(path: Optional[str]) -> pd.DataFrame:
    """
    fx.csv columns (any one):
      date, USDJPY or date, JPYUSD; optional: JPY_REER
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); u = ncol(df,"USDJPY","usdjpy","USD_JPY"); j = ncol(df,"JPYUSD","jpyusd","JPY_USD")
    r = ncol(df,"JPY_REER","reer_jpy","jpy_reer")
    if not d: raise ValueError("fx.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    if u:
        df["USDJPY"] = safe_num(df[u])
    elif j:
        df["USDJPY"] = 1.0 / safe_num(df[j])
    else:
        # try first non-date numeric
        num = [c for c in df.columns if c != "date"]
        if not num: raise ValueError("fx.csv must include USDJPY or JPYUSD.")
        df["USDJPY"] = safe_num(df[num[0]])
    if r: df = df.rename(columns={r:"JPY_REER"})
    df["r_fx_usdjpy"] = dlog(df["USDJPY"])
    if "JPY_REER" in df.columns: df["r_jpy_reer"] = dlog(df["JPY_REER"])
    return df[["date","USDJPY"] + ([ "JPY_REER" ] if "JPY_REER" in df.columns else []) + ["r_fx_usdjpy"] + (["r_jpy_reer"] if "r_jpy_reer" in df.columns else [])]

def load_macro(path: Optional[str]) -> pd.DataFrame:
    """
    macro.csv columns:
      date, country, gdp_usd[, gdp_pc_usd], entertainment_spend_pc_usd, internet_penetration
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    if not (d and c): raise ValueError("macro.csv needs date,country.")
    df = df.rename(columns={d:"date", c:"country"})
    ren = {
        ncol(df,"gdp_usd"): "gdp_usd",
        ncol(df,"gdp_pc_usd","income_pc_usd"): "gdp_pc_usd",
        ncol(df,"entertainment_spend_pc_usd","ent_spend_pc_usd","media_spend_pc_usd"): "ent_spend_pc_usd",
        ncol(df,"internet_penetration","internet_pct","broadband_pct"): "internet_penetration"
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    for k in ["gdp_usd","gdp_pc_usd","ent_spend_pc_usd","internet_penetration"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_interest(path: Optional[str]) -> pd.DataFrame:
    """
    interest.csv columns:
      date, country, anime_search_idx[, youtube_views_anime, tiktok_views_anime]
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    if not (d and c): raise ValueError("interest.csv needs date,country.")
    df = df.rename(columns={d:"date", c:"country"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    ren = {
        ncol(df,"anime_search_idx","google_trends_anime"): "anime_search_idx",
        ncol(df,"youtube_views_anime","yt_anime_views"): "youtube_views_anime",
        ncol(df,"tiktok_views_anime","tt_anime_views"): "tiktok_views_anime"
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for k in ["anime_search_idx","youtube_views_anime","tiktok_views_anime"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_shipping(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); f = ncol(df,"freight_usd_per_ton","freight_idx","shipping_cost")
    if not d or not f: raise ValueError("shipping.csv needs date and freight.")
    df = df.rename(columns={d:"date", f:"freight"})
    df["date"] = eom(df["date"]); df["freight"] = safe_num(df["freight"])
    return df

def load_policy(path: Optional[str]) -> pd.DataFrame:
    """
    policy.csv columns:
      date, country, type[, value]
      type ∈ {CENSOR_EASE, CENSOR_TIGHTEN, QUOTA, TAX_UP, TAX_DOWN, PLATFORM_LAUNCH, PLATFORM_EXIT}
      value optional; inferred sign if missing.
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); t = ncol(df,"type","event"); v = ncol(df,"value")
    if not (d and c and t): raise ValueError("policy.csv needs date,country,type.")
    df = df.rename(columns={d:"date", c:"country", t:"type"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    df["type"] = df["type"].astype(str).str.upper().str.strip()
    if v: df = df.rename(columns={v:"value"})
    if "value" not in df.columns:
        df["value"] = 0.0
        df.loc[df["type"].str.contains("EASE|DOWN|LAUNCH"), "value"] = +1.0
        df.loc[df["type"].str.contains("TIGHTEN|UP|EXIT|QUOTA"), "value"] = -1.0
    df["value"] = safe_num(df["value"])
    return df

def load_piracy(path: Optional[str]) -> pd.DataFrame:
    """
    piracy.csv columns:
      date, country, piracy_index[, takedowns]
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); p = ncol(df,"piracy_index","piracy_score"); tk = ncol(df,"takedowns")
    if not (d and c and (p or tk)): raise ValueError("piracy.csv needs date,country and piracy_index or takedowns.")
    df = df.rename(columns={d:"date", c:"country"})
    if p: df = df.rename(columns={p:"piracy_index"})
    if tk: df = df.rename(columns={tk:"takedowns"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    for k in ["piracy_index","takedowns"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_releases(path: Optional[str]) -> pd.DataFrame:
    """
    releases.csv columns:
      date[, overseas_date], title, studio[, franchise], impact_score[, country]  # country optional for targeted promos
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"title"); s = ncol(df,"studio"); f = ncol(df,"franchise"); i = ncol(df,"impact_score","score")
    c = ncol(df,"country")
    if not (d and t): raise ValueError("releases.csv needs date,title.")
    df = df.rename(columns={d:"date"})
    if s: df = df.rename(columns={s:"studio"})
    if f: df = df.rename(columns={f:"franchise"})
    if i: df = df.rename(columns={i:"impact_score"})
    if c: df = df.rename(columns={c:"country"})
    df["date"] = eom(df["date"])
    if "country" in df.columns: df["country"] = df["country"].astype(str)
    if "impact_score" not in df.columns: df["impact_score"] = 1.0
    df["impact_score"] = safe_num(df["impact_score"]).fillna(1.0)
    return df


# ----------------------------- panel construction -----------------------------

def build_panel(REV: pd.DataFrame, TR: pd.DataFrame, PLAT: pd.DataFrame, FX: pd.DataFrame,
                MAC: pd.DataFrame, INT: pd.DataFrame, SH: pd.DataFrame,
                POL: pd.DataFrame, PIR: pd.DataFrame) -> pd.DataFrame:
    P = REV.copy()
    # Merge trade buckets
    if not TR.empty: P = P.merge(TR, on=["date","country"], how="left")
    # Platform
    if not PLAT.empty: P = P.merge(PLAT, on=["date","country"], how="left")
    # Macro/demand
    if not MAC.empty: P = P.merge(MAC, on=["date","country"], how="left")
    # Interest
    if not INT.empty: P = P.merge(INT, on=["date","country"], how="left")
    # Policy (monthly index: sum of signs)
    if not POL.empty:
        polm = POL.groupby(["date","country"], as_index=False)["value"].sum().rename(columns={"value":"policy_index"})
        P = P.merge(polm, on=["date","country"], how="left")
    # Piracy
    if not PIR.empty: P = P.merge(PIR, on=["date","country"], how="left")
    # FX (same for all destinations; join on date)
    if not FX.empty: P = P.merge(FX, on=["date"], how="left")

    # Derived growth rates
    P = P.sort_values(["country","date"])
    if "rev_total_usd" in P.columns:
        P["dlog_rev"] = P.groupby("country")["rev_total_usd"].apply(dlog).reset_index(level=0, drop=True)
    for k in ["license_usd","streaming_usd","boxoffice_usd","merch_usd",
              "toys_figures_usd","manga_books_usd","video_media_usd",
              "titles_available","simulcast_titles","platform_count","sub_price_usd",
              "anime_search_idx","youtube_views_anime","tiktok_views_anime",
              "freight","gdp_pc_usd","ent_spend_pc_usd","internet_penetration"]:
        if k in P.columns and k not in ["internet_penetration"]:
            P["dlog_"+k] = P.groupby("country")[k].apply(dlog).reset_index(level=0, drop=True)
    # normalize piracy so that higher = worse; derivative in levels
    if "piracy_index" in P.columns:
        P["piracy_index"] = safe_num(P["piracy_index"])
    # month dummies
    P["month"] = P["date"].dt.month
    return P


# ----------------------------- regressions (elasticities) -----------------------------

def build_dlag_matrix(g: pd.DataFrame, L: int, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.DataFrame(index=g.index)
    names = []
    for c in cols:
        if c not in g.columns: continue
        for l in range(0, L+1):
            nm = f"{c}_l{l}"
            X[nm] = g[c].shift(l)
            names.append(nm)
    return X, names

def regress_elasticities(P: pd.DataFrame, L: int, min_obs: int) -> pd.DataFrame:
    """
    Δlog(revenue) ~ FX + platform + interest + macro + shipping + policy + piracy + seasonality
    """
    if P.empty or "dlog_rev" not in P.columns: return pd.DataFrame()
    rows = []
    for cty, g in P.groupby("country"):
        g = g.sort_values("date").copy()
        y = g["dlog_rev"]
        if y.notna().sum() < max(min_obs, 24): continue

        # drivers in dlog (use those available)
        dlog_cols = [c for c in [
            "r_fx_usdjpy", "r_jpy_reer",
            "dlog_titles_available","dlog_simulcast_titles","dlog_platform_count","dlog_sub_price_usd",
            "dlog_anime_search_idx","dlog_youtube_views_anime",
            "dlog_gdp_pc_usd","dlog_ent_spend_pc_usd",
            "dlog_freight"
        ] if c in g.columns]
        X1, _ = build_dlag_matrix(g, L, cols=dlog_cols)

        # level-type drivers (policy/piracy/internet)
        add = []
        if "policy_index" in g.columns: add.append("policy_index")
        if "piracy_index" in g.columns: add.append("piracy_index")
        if "internet_penetration" in g.columns: add.append("internet_penetration")
        X2 = g[add] if add else pd.DataFrame(index=g.index)

        # seasonality
        D = pd.get_dummies(g["month"].astype(int), prefix="m", drop_first=True)

        X = pd.concat([pd.Series(1.0, index=g.index, name="const"), X1, X2, D], axis=1).dropna()
        Y = y.loc[X.index]
        if X.shape[0] < max(min_obs, 5*X.shape[1]//3): 
            continue

        beta, resid, XTX_inv = ols_beta_resid(X.values, Y.values.reshape(-1,1))
        se = hac_se(X.values, resid, XTX_inv, L=max(6, L))

        for i, nm in enumerate(X.columns):
            rows.append({"country": cty, "var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                         "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                         "n": int(X.shape[0]), "lags": int(L)})

        # cumulative elasticities for key dlog drivers
        for base in ["r_fx_usdjpy","r_jpy_reer","dlog_titles_available","dlog_simulcast_titles",
                     "dlog_anime_search_idx","dlog_youtube_views_anime","dlog_ent_spend_pc_usd"]:
            idxs = [i for i, nm in enumerate(X.columns) if nm.startswith(base+"_l")]
            if idxs:
                bsum = float(np.sum([beta[i,0] for i in idxs]))
                sesq = float(np.sum([se[i]**2 for i in idxs]))
                rows.append({"country": cty, "var": base+"_cum_0..L", "coef": bsum, "se": np.sqrt(sesq),
                             "t_stat": bsum/np.sqrt(sesq) if sesq>0 else np.nan,
                             "n": int(X.shape[0]), "lags": int(L)})
    return pd.DataFrame(rows).sort_values(["country","var"])


# ----------------------------- event studies -----------------------------

def zscore(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w, min_periods=max(6, w//3)).mean()
    sd = s.rolling(w, min_periods=max(6, w//3)).std(ddof=0)
    return (s - mu) / (sd + 1e-12)

def car(series: pd.Series, anchor: pd.Timestamp, pre: int, post: int) -> float:
    idx = series.index
    if anchor not in idx:
        pos = idx.searchsorted(anchor)
        if pos >= len(idx): return np.nan
        anchor = idx[pos]
    i0 = idx.searchsorted(anchor) - pre
    i1 = idx.searchsorted(anchor) + post
    i0 = max(0, i0); i1 = min(len(idx)-1, i1)
    return float(series.iloc[i0:i1+1].sum())

def event_study(P: pd.DataFrame, REL: pd.DataFrame, PLAT: pd.DataFrame,
                window: int=2, z_window: int=12, z_thr: float=1.0) -> pd.DataFrame:
    """
    Events:
      - Release dates (global or country-targeted)
      - Platform capacity shocks (z-scores on Δlog titles/catalog)
      - Large FX shocks (z on r_fx_usdjpy)
    """
    if P.empty or "dlog_rev" not in P.columns: return pd.DataFrame()
    rows = []
    # Country loops to compute CARs
    for cty, g in P.groupby("country"):
        ser = g.set_index("date")["dlog_rev"].dropna().sort_index()
        if ser.empty: continue
        events = []

        # Releases: if targeted country present, filter; else treat as global
        if not REL.empty:
            if "country" in REL.columns:
                R = REL[(REL["country"].astype(str).str.lower()==str(cty).lower()) | (REL["country"].isna())]
            else:
                R = REL
            for _, r in R.iterrows():
                events.append(("RELEASE", pd.Timestamp(r["date"])))

        # Platform shocks
        if not PLAT.empty and "dlog_titles_available" in g.columns:
            z = zscore(g.set_index("date")["dlog_titles_available"].dropna(), w=z_window)
            ev = z[abs(z)>=z_thr]
            for dt in ev.index:
                events.append(("PLATFORM", pd.Timestamp(dt)))

        # FX shocks
        if "r_fx_usdjpy" in g.columns:
            z = zscore(g.set_index("date")["r_fx_usdjpy"].dropna(), w=z_window)
            ev = z[abs(z)>=z_thr]
            for dt in ev.index:
                events.append(("FX", pd.Timestamp(dt)))

        # Compute CARs
        idx = ser.index
        for lab, dt in events:
            if not isinstance(dt, pd.Timestamp): continue
            rows.append({"country": cty, "event": lab, "event_date": str(dt.date()),
                         "CAR_dlog_rev": car(ser, dt, window, window)})
    return pd.DataFrame(rows).sort_values(["country","event_date","event"])


# ----------------------------- risk & opportunity score -----------------------------

def risk_opportunity(P: pd.DataFrame) -> pd.DataFrame:
    """
    Components (higher = better unless noted):
      - Size: GDP_pc, ent_spend_pc
      - Platform: titles_available, simulcasts, platform_count
      - Interest: anime_search_idx, youtube_views_anime
      - Openness: policy_index (higher better), internet_penetration
      - Friction (negative): piracy_index, freight
      - Stability (negative): vol of dlog_rev (12m)
    Score scaled 0..100 per country using min-max within sample.
    """
    if P.empty: return pd.DataFrame()
    g = P.sort_values(["country","date"]).copy()
    # 12m vol
    g["rev_vol_12m"] = (g.groupby("country")["dlog_rev"]
                          .transform(lambda s: s.rolling(12, min_periods=6).std(ddof=0)))
    # last obs per country
    last = g.groupby("country").tail(1).set_index("country")
    # normalize helper
    def norm(s: pd.Series, invert: bool=False) -> pd.Series:
        x = s.copy()
        lo, hi = np.nanmin(x.values), np.nanmax(x.values)
        z = (x - lo) / (hi - lo + 1e-12)
        if invert: z = 1.0 - z
        return z.fillna(0.0)

    comps = {}
    if "gdp_pc_usd" in last.columns: comps["size1"] = norm(last["gdp_pc_usd"])
    if "ent_spend_pc_usd" in last.columns: comps["size2"] = norm(last["ent_spend_pc_usd"])
    if "titles_available" in last.columns: comps["plat1"] = norm(last["titles_available"])
    if "simulcast_titles" in last.columns: comps["plat2"] = norm(last["simulcast_titles"])
    if "platform_count" in last.columns: comps["plat3"] = norm(last["platform_count"])
    if "anime_search_idx" in last.columns: comps["int1"] = norm(last["anime_search_idx"])
    if "youtube_views_anime" in last.columns: comps["int2"] = norm(last["youtube_views_anime"])
    if "policy_index" in last.columns: comps["open1"] = norm(last["policy_index"])
    if "internet_penetration" in last.columns: comps["open2"] = norm(last["internet_penetration"])
    if "piracy_index" in last.columns: comps["fric1"] = norm(last["piracy_index"], invert=True)
    if "freight" in last.columns: comps["fric2"] = norm(last["freight"], invert=True)
    if "rev_vol_12m" in last.columns: comps["stab"] = norm(last["rev_vol_12m"], invert=True)

    DF = pd.DataFrame(comps)
    if DF.empty: return pd.DataFrame()
    # weights
    w = {
        "size1":0.15, "size2":0.10,
        "plat1":0.10, "plat2":0.05, "plat3":0.05,
        "int1":0.15, "int2":0.05,
        "open1":0.10, "open2":0.05,
        "fric1":0.10, "fric2":0.05,
        "stab":0.05
    }
    # align missing weights to 0
    for k in list(w.keys()):
        if k not in DF.columns: w.pop(k, None)
    # score
    score = np.sum([DF[c]*w[c] for c in w], axis=0) / (sum(w.values())+1e-12)
    OUT = DF.copy()
    OUT["score_0_100"] = 100.0 * score
    OUT = OUT.reset_index().rename(columns={"index":"country"})
    return OUT.sort_values("score_0_100", ascending=False)


# ----------------------------- scenarios -----------------------------

def pick_country_coef(EL: pd.DataFrame, country: str, var: str, default: float) -> float:
    r = EL[(EL["country"]==country) & (EL["var"]==var)]
    if not r.empty: return float(r["coef"].iloc[0])
    r2 = EL[EL["var"]==var]
    return float(r2["coef"].mean()) if not r2.empty else float(default)

def run_scenarios(P: pd.DataFrame, EL: pd.DataFrame,
                  fx_pct: float, titles_pct: float, interest_pct: float,
                  policy_shift: float, piracy_shift: float,
                  release_uplift_pp: float, horizon: int) -> pd.DataFrame:
    """
    Log-linear projection from latest month by country.
    - fx_pct: % change in USDJPY (+ = USD stronger)
    - titles_pct: % change in titles_available
    - interest_pct: % change in anime_search_idx
    - policy_shift: additive change in policy_index units
    - piracy_shift: additive change in piracy_index (negative = improvement)
    - release_uplift_pp: additive percentage points to growth at t+1, decays geometrically by 50% per month
    """
    if P.empty: return pd.DataFrame()
    last = P.groupby("country").tail(1).set_index("country")
    rows = []
    for cty, r in last.iterrows():
        base = float(r.get("rev_total_usd", np.nan))
        if not np.isfinite(base) or base <= 0: continue
        # elasticities (use cumulative if available)
        e_fx  = pick_country_coef(EL, cty, "r_fx_usdjpy_cum_0..L", default=+0.20) if "r_fx_usdjpy" in P.columns else 0.0
        e_tit = pick_country_coef(EL, cty, "dlog_titles_available_cum_0..L", default=+0.30) if "dlog_titles_available" in P.columns else 0.0
        e_int = pick_country_coef(EL, cty, "dlog_anime_search_idx_cum_0..L", default=+0.25) if "dlog_anime_search_idx" in P.columns else 0.0
        b_pol = pick_country_coef(EL, cty, "policy_index", default=+0.015) if "policy_index" in P.columns else 0.0
        b_pir = pick_country_coef(EL, cty, "piracy_index", default=-0.010) if "piracy_index" in P.columns else 0.0

        dv = (e_fx * np.log1p(fx_pct/100.0) +
              e_tit * np.log1p(titles_pct/100.0) +
              e_int * np.log1p(interest_pct/100.0) +
              b_pol * policy_shift +
              b_pir * piracy_shift)
        decay = 1.0
        lvl = base
        for h in range(1, horizon+1):
            add = (release_uplift_pp/100.0) * decay if h == 1 or decay > 1e-6 else 0.0
            lvl = float(lvl * np.exp(dv + add))
            rows.append({"country": cty, "h_month": h, "rev_total_usd": lvl,
                         "fx_pct": fx_pct, "titles_pct": titles_pct, "interest_pct": interest_pct,
                         "policy_shift": policy_shift, "piracy_shift": piracy_shift,
                         "release_uplift_pp": release_uplift_pp})
            decay *= 0.5
    return pd.DataFrame(rows).sort_values(["country","h_month"])


# ----------------------------- stress (VaR/ES) -----------------------------

def stress_var_es(P: pd.DataFrame, EL: pd.DataFrame, n_sims: int=10000) -> pd.DataFrame:
    """
    One-month revenue distribution using historical driver covariances and elasticities.
    Drivers used (if present): r_fx_usdjpy, dlog_titles_available, dlog_anime_search_idx
    """
    need = [c for c in ["r_fx_usdjpy","dlog_titles_available","dlog_anime_search_idx"] if c in P.columns]
    if not need: return pd.DataFrame()
    # Build driver return matrix across all countries (average by date)
    R = (P.groupby("date")[need].mean().dropna())
    if R.shape[0] < 24: return pd.DataFrame()
    mu = R.mean().values; cov = R.cov().values
    L = np.linalg.cholesky(cov + 1e-12*np.eye(len(need)))
    rng = np.random.default_rng(42)
    shocks = rng.standard_normal(size=(n_sims, len(need))) @ L.T + mu

    last = P.groupby("country").tail(1)
    out = []
    for cty, r in last.groupby("country"):
        base = float(r["rev_total_usd"].iloc[0]) if "rev_total_usd" in r.columns else np.nan
        if not np.isfinite(base) or base <= 0: continue
        # elasticities vector (use country-specific cum where possible, else priors)
        e = []
        for nm in need:
            key = nm+"_cum_0..L" if nm != "r_fx_usdjpy" else "r_fx_usdjpy_cum_0..L"
            prior = {"r_fx_usdjpy": 0.20, "dlog_titles_available": 0.30, "dlog_anime_search_idx": 0.25}[nm]
            e.append(pick_country_coef(EL, cty, key, default=prior))
        e = np.array(e)
        # simulate
        X = base * np.exp(shocks @ e)
        x = np.array(X)
        x_sorted = np.sort(x)
        var5 = float(np.percentile(x, 5))
        es5 = float(x_sorted[:max(1,int(0.05*len(x_sorted)))].mean())
        out.append({"country": cty, "VaR_5pct_rev_usd": var5, "ES_5pct_rev_usd": es5,
                    "mean_rev_usd": float(x.mean()), "sd_rev_usd": float(x.std(ddof=0)),
                    "n_sims": int(n_sims)})
    return pd.DataFrame(out).sort_values("VaR_5pct_rev_usd")


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    revenue: str
    trade: Optional[str]
    platform: Optional[str]
    fx: Optional[str]
    macro: Optional[str]
    interest: Optional[str]
    shipping: Optional[str]
    policy: Optional[str]
    piracy: Optional[str]
    releases: Optional[str]
    lags: int
    horizon: int
    n_sims: int
    z_window: int
    z_threshold: float
    event_window: int
    fx_pct: float
    titles_pct: float
    interest_pct: float
    policy_shift: float
    piracy_shift: float
    release_uplift_pp: float
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Japan Anime Exports — revenues, drivers, events & scenarios")
    ap.add_argument("--revenue", required=True)
    ap.add_argument("--trade", default="")
    ap.add_argument("--platform", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--interest", default="")
    ap.add_argument("--shipping", default="")
    ap.add_argument("--policy", default="")
    ap.add_argument("--piracy", default="")
    ap.add_argument("--releases", default="")
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n_sims", type=int, default=10000)
    ap.add_argument("--z_window", type=int, default=12)
    ap.add_argument("--z_threshold", type=float, default=1.0)
    ap.add_argument("--event_window", type=int, default=2)
    # Scenario shocks
    ap.add_argument("--fx_pct", type=float, default=0.0, help="% change in USDJPY (+ = USD up)")
    ap.add_argument("--titles_pct", type=float, default=0.0, help="% change in titles_available")
    ap.add_argument("--interest_pct", type=float, default=0.0, help="% change in anime_search_idx")
    ap.add_argument("--policy_shift", type=float, default=0.0, help="Δ policy_index (units)")
    ap.add_argument("--piracy_shift", type=float, default=0.0, help="Δ piracy_index (negative = improvement)")
    ap.add_argument("--release_uplift_pp", type=float, default=0.0, help="Additive uplift to monthly growth at t+1 (pp)")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_anime")
    ap.add_argument("--min_obs", type=int, default=36)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    REV = load_revenue(args.revenue)
    TR  = load_trade(args.trade)       if args.trade      else pd.DataFrame()
    PLT = load_platform(args.platform) if args.platform   else pd.DataFrame()
    FX  = load_fx(args.fx)             if args.fx         else pd.DataFrame()
    MAC = load_macro(args.macro)       if args.macro      else pd.DataFrame()
    INT = load_interest(args.interest) if args.interest   else pd.DataFrame()
    SH  = load_shipping(args.shipping) if args.shipping   else pd.DataFrame()
    POL = load_policy(args.policy)     if args.policy     else pd.DataFrame()
    PIR = load_piracy(args.piracy)     if args.piracy     else pd.DataFrame()
    REL = load_releases(args.releases) if args.releases   else pd.DataFrame()

    # Time filters (month-end aligned)
    if args.start:
        s = eom(pd.Series([args.start])).iloc[0]
        for df in [REV, TR, PLT, FX, MAC, INT, SH, POL, PIR, REL]:
            if not df.empty: df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = eom(pd.Series([args.end])).iloc[0]
        for df in [REV, TR, PLT, FX, MAC, INT, SH, POL, PIR, REL]:
            if not df.empty: df.drop(df[df["date"] > e].index, inplace=True)

    # Panel
    P = build_panel(REV, TR, PLT, FX, MAC, INT, SH, POL, PIR)
    if P.empty:
        raise ValueError("Panel is empty after merges/filters. Check inputs.")
    P.to_csv(outdir / "panel.csv", index=False)

    # Elasticities
    EL = regress_elasticities(P, L=int(args.lags), min_obs=int(args.min_obs))
    if not EL.empty: EL.to_csv(outdir / "elasticity_revenue.csv", index=False)

    # Events
    ES = event_study(P, REL, PLT, window=int(args.event_window),
                     z_window=int(args.z_window), z_thr=float(args.z_threshold))
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Risk/Opportunity score (latest)
    RISK = risk_opportunity(P)
    if not RISK.empty: RISK.to_csv(outdir / "risk_opportunity.csv", index=False)

    # Scenarios
    SCN = run_scenarios(P, EL if not EL.empty else pd.DataFrame(),
                        fx_pct=float(args.fx_pct), titles_pct=float(args.titles_pct),
                        interest_pct=float(args.interest_pct),
                        policy_shift=float(args.policy_shift), piracy_shift=float(args.piracy_shift),
                        release_uplift_pp=float(args.release_uplift_pp),
                        horizon=int(args.horizon))
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Stress VaR/ES
    ST = stress_var_es(P, EL if not EL.empty else pd.DataFrame(), n_sims=int(args.n_sims))
    if not ST.empty: ST.to_csv(outdir / "stress_vares.csv", index=False)

    # Summary
    summary = {
        "sample": {
            "start": str(P["date"].min().date()),
            "end": str(P["date"].max().date()),
            "months": int(P["date"].nunique())
        },
        "destinations": int(P["country"].nunique()),
        "has_trade_proxies": bool(not TR.empty),
        "has_platform": bool(not PLT.empty),
        "has_fx": bool(not FX.empty),
        "has_interest": bool(not INT.empty),
        "has_policy": bool(not POL.empty),
        "has_piracy": bool(not PIR.empty),
        "outputs": {
            "panel": "panel.csv",
            "elasticity_revenue": "elasticity_revenue.csv" if not EL.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "risk_opportunity": "risk_opportunity.csv" if not RISK.empty else None,
            "scenarios": "scenarios.csv" if not SCN.empty else None,
            "stress_vares": "stress_vares.csv" if not ST.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        revenue=args.revenue, trade=(args.trade or None), platform=(args.platform or None),
        fx=(args.fx or None), macro=(args.macro or None), interest=(args.interest or None),
        shipping=(args.shipping or None), policy=(args.policy or None), piracy=(args.piracy or None),
        releases=(args.releases or None), lags=int(args.lags), horizon=int(args.horizon),
        n_sims=int(args.n_sims), z_window=int(args.z_window), z_threshold=float(args.z_threshold),
        event_window=int(args.event_window), fx_pct=float(args.fx_pct), titles_pct=float(args.titles_pct),
        interest_pct=float(args.interest_pct), policy_shift=float(args.policy_shift),
        piracy_shift=float(args.piracy_shift), release_uplift_pp=float(args.release_uplift_pp),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Japan Anime Exports Toolkit ==")
    print(f"Destinations: {summary['destinations']} | Sample: {summary['sample']['start']} → {summary['sample']['end']}")
    if summary["outputs"]["elasticity_revenue"]: print("Elasticities estimated (HAC/NW).")
    if summary["outputs"]["event_study"]: print(f"Event study done (±{args.event_window} months).")
    if summary["outputs"]["risk_opportunity"]: print("Risk/Opportunity score computed (latest).")
    if summary["outputs"]["scenarios"]: print(f"Scenarios projected ({args.horizon} months).")
    if summary["outputs"]["stress_vares"]: print(f"Stress VaR/ES computed (n={args.n_sims}).")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
