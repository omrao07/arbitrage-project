#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cherry_blossom_tourism.py — Bloom timing → travel demand, pricing & scenarios
-----------------------------------------------------------------------------

What this does
==============
A research toolkit to connect **sakura (cherry blossom) bloom timing** with **tourism
demand and pricing** across cities (Japan or elsewhere with seasonal blooms).

It ingests bloom calendars, arrivals/bookings, weather, FX, social interest, and events
to build a city–date panel, estimate elasticities, run event studies, and simulate
forward scenarios (e.g., bloom shifts, FX shocks, rain anomalies, visa/holiday changes).

Main capabilities
-----------------
1) Bloom calendar & alignment
   • First/Full bloom dates per city & year; climatology vs anomaly (days early/late)
   • Gaussian "bloom score" around full bloom (peak intensity over ±window days)
   • Weekend/holiday alignment around full bloom

2) Daily/weekly panel
   • Arrivals (intl/dom), hotel occupancy, ADR / room rate, airfare index
   • Drivers: bloom_score, weather (temp/rain), FX (e.g., USDJPY↑ = weaker JPY),
     social interest index, events (visa/holidays), weekday dummies

3) Event studies
   • Δ arrivals/bookings/ADR around full bloom date (±K days/weeks)

4) Elasticities (distributed-lag with Newey–West SEs)
   • dlog(arrivals) ~ Σ_{l=0..L} bloom_score_{t−l} + controls (FX, rain, temp, social, events)
   • dlog(ADR)      ~ Σ_{l=0..L} bloom_score_{t−l} + controls

5) Scenarios
   • Shift full bloom by ±D days per city/season
   • FX shock (% move in USDJPY – positive = JPY weakens)
   • Rainfall shock (%), temp shock (°C), holiday/visa bonus around peak
   • Forward projection of arrivals, occupancy, ADR and RevPAR over horizon

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--blooms blooms.csv              REQUIRED
  Columns:
    city, year[, season], first_bloom_date, full_bloom_date
    climatology_full_date (optional, long-run median per city)

--arrivals arrivals.csv          RECOMMENDED
  Columns:
    date, city, arrivals_total[, arrivals_intl, arrivals_domestic]

--bookings bookings.csv          OPTIONAL
  Columns:
    date, city,
    hotel_occ_pct[, occ_pct], adr_usd[, room_rate_usd], airfare_idx[, airfare]

--weather weather.csv            OPTIONAL
  Columns:
    date, city, avg_temp_c[, temp_c], rainfall_mm[, rain_mm], wind_kph (optional)

--fx fx.csv                      OPTIONAL
  Columns:
    date, usd_jpy[, fx_usd_local]     # higher usd_jpy = weaker JPY

--social social.csv              OPTIONAL
  Columns:
    date, city[, scope], search_idx[, social_idx]   # interest index (normalize 0–100)

--events events.csv              OPTIONAL
  Columns:
    date, city[, scope], label[, type]  # type: HOLIDAY/VISA/PROMO/SHOCK ...

CLI (key)
---------
--freq daily|weekly              Output frequency (default daily)
--bloom_window 14                +/- days around full bloom for score
--sigma 5.0                      Gaussian sigma (days) for bloom score
--lags 7                         Max lag for distributed-lag regressions (daily; try 3 for weekly)
--event_window 7                 Half-window for event studies (in chosen freq units)
--scenario "Tokyo:+5;Kyoto:-3"   Shift full bloom by ±days (semicolon sep; use ALL:*:+3)
--fx_shock_pct 0                 % change in USDJPY (e.g., +5 → JPY weaker by 5%)
--rain_shock_pct 0               % change in rainfall level
--temp_shock_c 0                 °C additive shock to temperature
--holiday_bonus_pp 0.0           pp boost to bloom_score if full bloom at/near holiday/weekend
--horizon 28                     Days (or weeks) to project from last observed date
--start / --end                  Sample filters (YYYY-MM-DD)
--outdir out_sakura              Output directory
--min_obs 120                    Minimum obs per city for regressions (reduce if weekly)

Outputs
-------
- bloom_calendar.csv             Clean bloom calendar, anomalies & alignment stats
- tourism_panel.csv             City–date panel with drivers & y-variables
- event_study.csv                Δ around full bloom for arrivals/ADR/occ
- elasticity_arrivals.csv        D-lag regression results for arrivals_total
- elasticity_adr.csv             D-lag regression results for ADR
- scenarios.csv                  Forward projections under scenario(s)
- summary.json                   Headline metrics
- config.json                    Echo of run configuration

Assumptions & notes
-------------------
• If climatology_full_date not provided, uses city median full bloom date by year.
• Bloom score S(t) = exp(-0.5 * (Δdays / sigma)^2) within +/- bloom_window, else 0.
• Weekend/holiday alignment adds holiday_bonus_pp to S(t) at (t==full_bloom) and ±1 day (daily),
  or at the full-bloom week (weekly).
• Elasticities are reduced-form correlations; validate before use.

DISCLAIMER
----------
This is research tooling with simplifying assumptions. Inspect data quality and robustness
before making operational/financial decisions.
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

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.date

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def yoy(s: pd.Series, periods: int) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s) - np.log(s.shift(periods))

def is_weekend(d: pd.Series) -> pd.Series:
    return (pd.to_datetime(d).dt.dayofweek >= 5).astype(int)

def winsor(s: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

# OLS + HAC (Newey–West)
def ols_beta_se(X: np.ndarray, y: np.ndarray):
    XtX = X.T @ X
    XtY = X.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ XtY
    resid = y - X @ beta
    return beta, resid, XtX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
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

def load_blooms(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    c = ncol(df, "city"); y = ncol(df, "year","season"); f1 = ncol(df, "first_bloom_date","first_bloom"); ff = ncol(df, "full_bloom_date","full_bloom")
    clim = ncol(df, "climatology_full_date","climatology")
    if not (c and y and f1 and ff):
        raise ValueError("blooms.csv must include city, year, first_bloom_date, full_bloom_date.")
    df = df.rename(columns={c:"city", y:"year", f1:"first_bloom_date", ff:"full_bloom_date"})
    df["city"] = df["city"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["first_bloom_date"] = to_date(df["first_bloom_date"])
    df["full_bloom_date"]  = to_date(df["full_bloom_date"])
    if clim:
        df = df.rename(columns={clim:"climatology_full_date"})
        df["climatology_full_date"] = to_date(df["climatology_full_date"])
    # fallback climatology = city median full bloom over years
    if "climatology_full_date" not in df.columns or df["climatology_full_date"].isna().all():
        med = (df.groupby("city")["full_bloom_date"]
                 .apply(lambda s: pd.to_datetime(s.dropna()).dt.dayofyear.median()))
        df = df.merge(med.rename("city_median_doy").reset_index(), on="city", how="left")
        def med_to_date(y, doy):
            try:
                base = pd.Timestamp(year=int(y), month=1, day=1)
                return (base + pd.Timedelta(days=int(doy)-1)).date()
            except Exception:
                return pd.NaT
        df["climatology_full_date"] = [med_to_date(yy, d) for yy, d in zip(df["year"], df["city_median_doy"])]
        df.drop(columns=["city_median_doy"], inplace=True)
    # anomalies
    df["full_bloom_doy"] = pd.to_datetime(df["full_bloom_date"]).dt.dayofyear
    df["climatology_doy"] = pd.to_datetime(df["climatology_full_date"]).dt.dayofyear
    df["bloom_anom_days"] = (pd.to_datetime(df["full_bloom_date"]) - pd.to_datetime(df["climatology_full_date"])).dt.days
    return df[["city","year","first_bloom_date","full_bloom_date","climatology_full_date","bloom_anom_days","full_bloom_doy","climatology_doy"]].sort_values(["city","year"])

def load_arrivals(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); c = ncol(df, "city"); a = ncol(df, "arrivals_total","arrivals")
    ai = ncol(df, "arrivals_intl","international"); ad = ncol(df, "arrivals_domestic","domestic")
    if not (d and c and a): raise ValueError("arrivals.csv needs date, city, arrivals_total.")
    df = df.rename(columns={d:"date", c:"city", a:"arrivals_total"})
    df["date"] = to_date(df["date"]); df["city"] = df["city"].astype(str).str.strip()
    if ai: df = df.rename(columns={ai:"arrivals_intl"}); df["arrivals_intl"] = safe_num(df["arrivals_intl"])
    if ad: df = df.rename(columns={ad:"arrivals_domestic"}); df["arrivals_domestic"] = safe_num(df["arrivals_domestic"])
    df["arrivals_total"] = safe_num(df["arrivals_total"])
    return df.sort_values(["city","date"])

def load_bookings(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); c = ncol(df, "city")
    occ = ncol(df, "hotel_occ_pct","occ_pct","occupancy_pct"); adr = ncol(df, "adr_usd","room_rate_usd","adr")
    air = ncol(df, "airfare_idx","airfare")
    if not (d and c): raise ValueError("bookings.csv needs date and city.")
    df = df.rename(columns={d:"date", c:"city"})
    df["date"] = to_date(df["date"]); df["city"] = df["city"].astype(str).str.strip()
    if occ: df = df.rename(columns={occ:"hotel_occ_pct"}); df["hotel_occ_pct"] = safe_num(df["hotel_occ_pct"])
    if adr: df = df.rename(columns={adr:"adr_usd"}); df["adr_usd"] = safe_num(df["adr_usd"])
    if air: df = df.rename(columns={air:"airfare_idx"}); df["airfare_idx"] = safe_num(df["airfare_idx"])
    return df.sort_values(["city","date"])

def load_weather(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); c = ncol(df, "city")
    t = ncol(df, "avg_temp_c","temp_c","temperature_c"); r = ncol(df, "rainfall_mm","rain_mm","precip_mm")
    w = ncol(df, "wind_kph","wind_speed_kph")
    if not (d and c): raise ValueError("weather.csv needs date and city.")
    df = df.rename(columns={d:"date", c:"city"})
    df["date"] = to_date(df["date"]); df["city"] = df["city"].astype(str).str.strip()
    if t: df = df.rename(columns={t:"temp_c"}); df["temp_c"] = safe_num(df["temp_c"])
    if r: df = df.rename(columns={r:"rain_mm"}); df["rain_mm"] = safe_num(df["rain_mm"])
    if w: df = df.rename(columns={w:"wind_kph"}); df["wind_kph"] = safe_num(df["wind_kph"])
    return df.sort_values(["city","date"])

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date")
    fx = ncol(df, "usd_jpy","fx_usd_local","usd/local")
    if not (d and fx): raise ValueError("fx.csv needs date and usd_jpy (or fx_usd_local).")
    df = df.rename(columns={d:"date", fx:"usd_jpy"})
    df["date"] = to_date(df["date"])
    df["usd_jpy"] = safe_num(df["usd_jpy"])
    return df.sort_values("date")

def load_social(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); c = ncol(df, "city")
    s = ncol(df, "search_idx","social_idx","index")
    if not (d and s): raise ValueError("social.csv needs date and search_idx.")
    df = df.rename(columns={d:"date", s:"search_idx"})
    if c: df = df.rename(columns={c:"city"}) ; df["city"] = df["city"].astype(str).str.strip()
    df["date"] = to_date(df["date"]); df["search_idx"] = safe_num(df["search_idx"])
    return df.sort_values(["city","date"] if "city" in df.columns else ["date"])

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); c = ncol(df, "city","scope")
    lab = ncol(df, "label","event","name") or "label"
    typ = ncol(df, "type","kind") or None
    if not d: raise ValueError("events.csv needs date.")
    df = df.rename(columns={d:"date", lab:"label"})
    if c: df = df.rename(columns={c:"city"})
    if typ: df = df.rename(columns={typ:"type"})
    df["date"] = to_date(df["date"])
    if "city" in df.columns: df["city"] = df["city"].astype(str).str.strip()
    if "type" not in df.columns: df["type"] = ""
    return df.sort_values(["city","date"] if "city" in df.columns else ["date"])


# ----------------------------- constructions -----------------------------

def gaussian_bloom_profile(dates: pd.Series, full_date: pd.Timestamp, sigma: float, window: int) -> np.ndarray:
    if pd.isna(full_date):
        return np.zeros(len(dates))
    dt = pd.to_datetime(dates)
    fd = pd.to_datetime(full_date)
    delta = (dt - fd).dt.days.astype(float)
    inside = np.abs(delta) <= window
    score = np.zeros(len(dates))
    score[inside] = np.exp(-0.5 * (delta[inside] / float(sigma))**2)
    return score

def build_bloom_score(blooms: pd.DataFrame, cities: List[str], dates: pd.DatetimeIndex,
                      sigma: float, window: int, events: pd.DataFrame, freq: str, holiday_bonus_pp: float) -> pd.DataFrame:
    rows = []
    # holiday dates per city (HOLIDAY type or all events if type blank and label contains 'holiday')
    evt = pd.DataFrame()
    if not events.empty:
        evt = events.copy()
        if "city" not in evt.columns:
            evt["city"] = "ALL"
        evt["is_holiday"] = evt["type"].str.upper().eq("HOLIDAY") | evt["label"].str.upper().str.contains("HOLIDAY", na=False)
        evt["date"] = pd.to_datetime(evt["date"])
    for city in cities:
        # get per-year full bloom date for this city
        bb = blooms[blooms["city"]==city]
        # map each date to its year
        for d in dates:
            y = d.year
            fb = bb[bb["year"]==y]["full_bloom_date"]
            full = pd.to_datetime(fb.iloc[0]) if not fb.empty else pd.NaT
            s = gaussian_bloom_profile(pd.Series([d]), full, sigma=sigma, window=window)[0]
            # Weekend + holiday alignment bonuses on/around full bloom
            bonus = 0.0
            if not pd.isna(full) and (abs((d - pd.to_datetime(full)).days) <= (1 if freq.startswith("d") else 0)):
                # weekend
                if d.dayofweek >= 5:
                    bonus += holiday_bonus_pp
                # holiday (global or city)
                if not evt.empty:
                    ee = evt[(evt["is_holiday"]) & ((evt["city"]==city) | (evt["city"]=="ALL"))]
                    if not ee.empty:
                        if (ee["date"].dt.date == d.date()).any():
                            bonus += holiday_bonus_pp
            rows.append({"date": d, "city": city, "bloom_score": float(min(1.0, s + bonus))})
    S = pd.DataFrame(rows)
    return S

def resample_panel(df: pd.DataFrame, freq: str, by_cols: List[str], agg: Dict[str,str]) -> pd.DataFrame:
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    out = []
    for keys, g in df2.groupby(by_cols):
        gg = g.set_index("date").sort_index().resample("W-SUN" if freq.startswith("w") else "D").agg(agg)
        for i, k in enumerate(by_cols):
            gg[k] = keys[i] if isinstance(keys, tuple) else keys
        out.append(gg.reset_index().rename(columns={"index":"date"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def build_panel(blooms: pd.DataFrame,
                ARR: pd.DataFrame, BKG: pd.DataFrame, WTH: pd.DataFrame,
                FX: pd.DataFrame, SOC: pd.DataFrame,
                events: pd.DataFrame, freq: str,
                sigma: float, window: int, holiday_bonus_pp: float) -> pd.DataFrame:
    # universe of cities and dates from available series
    cities = sorted(set(
        ([*ARR["city"].unique()] if not ARR.empty else []) +
        ([*BKG["city"].unique()] if not BKG.empty else []) +
        ([*WTH["city"].unique()] if not WTH.empty else []) +
        ([*SOC["city"].unique()] if (not SOC.empty and "city" in SOC.columns) else []) +
        ([*blooms["city"].unique()] if not blooms.empty else [])
    ))
    # date range
    dt_min = min([
        (pd.to_datetime(ARR["date"]).min() if not ARR.empty else None),
        (pd.to_datetime(BKG["date"]).min() if not BKG.empty else None),
        (pd.to_datetime(WTH["date"]).min() if not WTH.empty else None),
        (pd.to_datetime(SOC["date"]).min() if not SOC.empty else None),
    ] + [pd.to_datetime(blooms["full_bloom_date"]).min() if not blooms.empty else None])
    dt_max = max([
        (pd.to_datetime(ARR["date"]).max() if not ARR.empty else None),
        (pd.to_datetime(BKG["date"]).max() if not BKG.empty else None),
        (pd.to_datetime(WTH["date"]).max() if not WTH.empty else None),
        (pd.to_datetime(SOC["date"]).max() if not SOC.empty else None),
    ] + [pd.to_datetime(blooms["full_bloom_date"]).max() if not blooms.empty else None])
    if pd.isna(dt_min) or pd.isna(dt_max):
        raise ValueError("Insufficient dates to build panel. Check inputs.")
    dates = pd.date_range(dt_min, dt_max, freq="D")

    # bloom score per city-date
    S = build_bloom_score(blooms, cities, dates, sigma=sigma, window=window, events=events, freq=freq, holiday_bonus_pp=holiday_bonus_pp)

    # merge all series (daily)
    def tidy(df, cols):
        if df.empty: return pd.DataFrame(columns=["date","city"]+cols)
        d = df.copy(); d["date"] = to_dt(d["date"])
        return d[["date","city"]+cols]
    panel = S.copy()
    if not ARR.empty:
        panel = panel.merge(tidy(ARR, [c for c in ["arrivals_total","arrivals_intl","arrivals_domestic"] if c in ARR.columns]),
                            on=["date","city"], how="left")
    if not BKG.empty:
        panel = panel.merge(tidy(BKG, [c for c in ["hotel_occ_pct","adr_usd","airfare_idx"] if c in BKG.columns]),
                            on=["date","city"], how="left")
    if not WTH.empty:
        panel = panel.merge(tidy(WTH, [c for c in ["temp_c","rain_mm","wind_kph"] if c in WTH.columns]),
                            on=["date","city"], how="left")
    if not SOC.empty:
        if "city" in SOC.columns:
            panel = panel.merge(tidy(SOC, ["search_idx"]), on=["date","city"], how="left")
        else:
            # city-agnostic social index
            panel = panel.merge(SOC.rename(columns={"date":"date"})[["date","search_idx"]], on="date", how="left")
    # FX is date-only
    if not FX.empty:
        FX2 = FX.copy(); FX2["date"] = to_dt(FX2["date"])
        panel = panel.merge(FX2[["date","usd_jpy"]], on="date", how="left")

    # calendar features
    panel["dow"] = pd.to_datetime(panel["date"]).dt.dayofweek
    panel["is_weekend"] = (panel["dow"]>=5).astype(int)
    # proximity to full bloom (days)
    fb_map = blooms.set_index(["city","year"])["full_bloom_date"].to_dict()
    year = pd.to_datetime(panel["date"]).dt.year
    dd = []
    for i, r in panel.iterrows():
        fb = fb_map.get((r["city"], year.iloc[i]), pd.NaT)
        dd.append((pd.to_datetime(r["date"]) - pd.to_datetime(fb)).days if pd.notna(fb) else np.nan)
    panel["days_to_full"] = dd

    # resample to weekly if requested
    if freq.startswith("w"):
        agg = {
            "bloom_score":"mean","arrivals_total":"sum","arrivals_intl":"sum","arrivals_domestic":"sum",
            "hotel_occ_pct":"mean","adr_usd":"mean","airfare_idx":"mean",
            "temp_c":"mean","rain_mm":"sum","wind_kph":"mean","search_idx":"mean","usd_jpy":"mean",
            "is_weekend":"sum","days_to_full":"mean"
        }
        panel = resample_panel(panel, freq="weekly", by_cols=["city"], agg=agg)

    # transforms
    for col in ["arrivals_total","arrivals_intl","arrivals_domestic","adr_usd","airfare_idx","search_idx","usd_jpy"]:
        if col in panel.columns:
            panel[f"dlog_{col}"] = panel.groupby("city")[col].apply(dlog).reset_index(level=0, drop=True)
    # rainfall shock
    if "rain_mm" in panel.columns:
        panel["rain_mm"] = panel["rain_mm"].fillna(0.0)
    return panel.sort_values(["city","date"])

def bloom_calendar_stats(blooms: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    # weekend/holiday alignment for the exact full bloom date
    out = blooms.copy()
    out["full_bloom_weekend"] = pd.to_datetime(out["full_bloom_date"]).dt.dayofweek >= 5
    if events.empty:
        out["full_bloom_holiday"] = False
    else:
        E = events.copy(); E["date"] = pd.to_datetime(E["date"])
        out["full_bloom_holiday"] = [
            ((E[(E["type"].str.upper()=="HOLIDAY") | (E["label"].str.upper().str.contains("HOLIDAY", na=False))]["date"].dt.date
             == pd.to_datetime(d).date()).any()) for d in out["full_bloom_date"]
        ]
    return out

# ----------------------------- event study -----------------------------

def event_study(panel: pd.DataFrame, blooms: pd.DataFrame, window: int, freq: str) -> pd.DataFrame:
    """
    For each city-year, compute deltas around full bloom for:
      dlog_arrivals_total, dlog_adr_usd, hotel_occ_pct
    Δ vs average of pre-window (h<0).
    """
    if panel.empty: return pd.DataFrame()
    rows = []
    idx = panel.set_index(["city","date"]).sort_index()
    for _, b in blooms.iterrows():
        city = b["city"]; fb = pd.to_datetime(b["full_bloom_date"])
        # locate closest panel date
        if (city, fb) not in idx.index:
            # choose nearest date in same week if weekly
            if freq.startswith("w"):
                # find week that contains fb
                week = fb - pd.Timedelta(days=fb.weekday())  # Monday anchor; panel is W-SUN, but ok for nearest
                # take nearest date for city
                dates_c = idx.loc[city].index if city in idx.index.levels[0] else []
                if len(dates_c)==0: continue
                fb_use = min(dates_c, key=lambda d: abs(pd.to_datetime(d)-week))
            else:
                # choose nearest day
                dates_c = idx.loc[city].index if city in idx.index.levels[0] else []
                if len(dates_c)==0: continue
                fb_use = min(dates_c, key=lambda d: abs(pd.to_datetime(d)-fb))
        else:
            fb_use = fb
        # window h=-w..+w
        dates_c = sorted(idx.loc[city].index) if city in idx.index.levels[0] else []
        if len(dates_c)==0: continue
        try:
            i0 = dates_c.index(fb_use)
        except ValueError:
            # nearest index
            i0 = min(range(len(dates_c)), key=lambda i: abs(pd.to_datetime(dates_c[i])-fb_use))
        for h in range(-window, window+1):
            j = i0 + h
            if j < 0 or j >= len(dates_c): continue
            dt = dates_c[j]
            r = idx.loc[(city, dt)]
            rec = {"city": city, "full_bloom": pd.to_datetime(fb).date(), "h": h, "date": pd.to_datetime(dt).date()}
            rec["dlog_arrivals_total"] = float(r.get("dlog_arrivals_total", np.nan)) if "dlog_arrivals_total" in r.index else np.nan
            rec["dlog_adr_usd"] = float(r.get("dlog_adr_usd", np.nan)) if "dlog_adr_usd" in r.index else np.nan
            rec["hotel_occ_pct"] = float(r.get("hotel_occ_pct", np.nan)) if "hotel_occ_pct" in r.index else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    out = []
    for (city, fb), g in df.groupby(["city","full_bloom"]):
        base = g[g["h"]<0][["dlog_arrivals_total","dlog_adr_usd","hotel_occ_pct"]].mean(numeric_only=True)
        for _, r in g.iterrows():
            rr = {"city": city, "full_bloom": fb, "h": int(r["h"])}
            for c in ["dlog_arrivals_total","dlog_adr_usd","hotel_occ_pct"]:
                rr[f"delta_{c}"] = float(r.get(c, np.nan) - base.get(c, np.nan)) if pd.notna(r.get(c, np.nan)) else np.nan
            out.append(rr)
    return pd.DataFrame(out).sort_values(["city","full_bloom","h"])

# ----------------------------- regressions (elasticities) -----------------------------

def dlag_regression(panel: pd.DataFrame, dep_col: str, L: int, min_obs: int) -> pd.DataFrame:
    """
    Per-city regression with HAC SEs:
      dep_t = α + Σ_{l=0..L} β_l * bloom_score_{t−l} + γ * dlog(usd_jpy) + δ * rain_mm + θ * temp_c + φ * dlog(search_idx) + ε
    dep_col ∈ {"dlog_arrivals_total","dlog_adr_usd"}.
    """
    if dep_col not in panel.columns or "bloom_score" not in panel.columns:
        return pd.DataFrame()
    out = []
    for city, g in panel.groupby("city"):
        g = g.sort_values("date")
        dep = g[dep_col]
        Xparts = [pd.Series(1.0, index=g.index, name="const")]
        names = ["const"]
        # bloom score lags
        for l in range(0, L+1):
            nm = f"bloom_l{l}"
            Xparts.append(g["bloom_score"].shift(l).rename(nm)); names.append(nm)
        # controls (Δlog or level)
        if "usd_jpy" in g.columns and g["usd_jpy"].notna().sum() >= min_obs//2:
            Xparts.append(np.log(g["usd_jpy"]).diff().rename("dlog_usd_jpy")); names.append("dlog_usd_jpy")
        if "rain_mm" in g.columns and g["rain_mm"].notna().sum() >= min_obs//2:
            Xparts.append(g["rain_mm"].rename("rain_mm")); names.append("rain_mm")
        if "temp_c" in g.columns and g["temp_c"].notna().sum() >= min_obs//2:
            Xparts.append(g["temp_c"].rename("temp_c")); names.append("temp_c")
        if "search_idx" in g.columns and g["search_idx"].notna().sum() >= min_obs//2:
            Xparts.append(dlog(g["search_idx"]).rename("dlog_search")); names.append("dlog_search")
        X = pd.concat(Xparts, axis=1)
        XY = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        if XY.shape[0] < max(min_obs, 5*X.shape[1]):
            continue
        yv = XY["dep"].values.reshape(-1,1)
        Xv = XY.drop(columns=["dep"]).values
        beta, resid, XtX_inv = ols_beta_se(Xv, yv)
        se = hac_se(Xv, resid, XtX_inv, L=max(6, L))
        for i, nm in enumerate(names):
            out.append({"city": city, "dep": dep_col, "var": nm,
                        "coef": float(beta[i,0]), "se": float(se[i]),
                        "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                        "n": int(XY.shape[0]), "lags": int(L)})
        # cumulative bloom effect
        idxs = [i for i,nm in enumerate(names) if nm.startswith("bloom_l")]
        if idxs:
            bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
            out.append({"city": city, "dep": dep_col, "var": "bloom_cum_0..L",
                        "coef": bsum, "se": ses,
                        "t_stat": bsum/(ses if ses>0 else np.nan), "n": int(XY.shape[0]), "lags": int(L)})
    return pd.DataFrame(out)

# ----------------------------- scenarios -----------------------------

def parse_scenarios(s: str) -> Dict[str, int]:
    """
    "Tokyo:+5;Kyoto:-3;ALL:+2" -> {"Tokyo": 5, "Kyoto": -3, "ALL": 2}
    """
    if not s: return {}
    out = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        if ":" in p:
            city, val = p.split(":")
            out[city.strip()] = int(float(val.strip()))
    return out

def scenario_shift_blooms(blooms: pd.DataFrame, shifts: Dict[str,int]) -> pd.DataFrame:
    if not shifts: return blooms.copy()
    B = blooms.copy()
    for i, r in B.iterrows():
        city = r["city"]
        dd = shifts.get(city, shifts.get("ALL", 0))
        if dd:
            B.at[i, "full_bloom_date"] = (pd.to_datetime(r["full_bloom_date"]) + pd.Timedelta(days=int(dd))).date()
            B.at[i, "first_bloom_date"] = (pd.to_datetime(r["first_bloom_date"]) + pd.Timedelta(days=int(dd))).date()
            B.at[i, "bloom_anom_days"] = int(r.get("bloom_anom_days", 0)) + int(dd)
    return B

def pick_cum_beta(df: pd.DataFrame, city: str, dep: str) -> Optional[float]:
    if df.empty: return None
    r = df[(df["city"]==city) & (df["dep"]==dep) & (df["var"]=="bloom_cum_0..L")]
    return float(r["coef"].iloc[0]) if not r.empty else None

def scenario_apply(panel: pd.DataFrame, blooms: pd.DataFrame,
                   EL_ARR: pd.DataFrame, EL_ADR: pd.DataFrame,
                   fx_shock_pct: float, rain_shock_pct: float, temp_shock_c: float,
                   horizon: int, sigma: float, window: int, holiday_bonus_pp: float, freq: str) -> pd.DataFrame:
    """
    Shift bloom dates → recompute bloom_score path from last observed date forward,
    then map to arrivals/ADR using cumulative elasticities. FX/rain/temp shocks apply additively.
    """
    if panel.empty: return pd.DataFrame()
    last_date = pd.to_datetime(panel["date"]).max()
    cities = sorted(panel["city"].unique())
    f_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq=("W-SUN" if freq.startswith("w") else "D"))
    S = build_bloom_score(blooms, cities, f_dates, sigma=sigma, window=window, events=pd.DataFrame(), freq=freq, holiday_bonus_pp=holiday_bonus_pp)
    rows = []
    for city in cities:
        # latest levels
        g = panel[panel["city"]==city].sort_values("date")
        if g.empty: continue
        arr0 = float(g["arrivals_total"].iloc[-1]) if "arrivals_total" in g.columns else np.nan
        adr0 = float(g["adr_usd"].iloc[-1]) if "adr_usd" in g.columns else np.nan
        occ0 = float(g["hotel_occ_pct"].iloc[-1]) if "hotel_occ_pct" in g.columns else np.nan
        # coefficients
        b_arr = pick_cum_beta(EL_ARR, city, "dlog_arrivals_total")
        b_adr = pick_cum_beta(EL_ADR, city, "dlog_adr_usd")
        if b_arr is None: b_arr = 0.25
        if b_adr is None: b_adr = 0.10
        dlog_fx = np.log(1.0 + fx_shock_pct/100.0) if fx_shock_pct else 0.0
        rain_mult = (1.0 + rain_shock_pct/100.0) if rain_shock_pct else 1.0
        temp_add = temp_shock_c if temp_shock_c else 0.0
        # baseline bloom average score near last period for normalization (avoid massive jumps)
        base_bs = float(g["bloom_score"].tail(7 if not freq.startswith("w") else 1).mean()) if "bloom_score" in g.columns else 0.0
        for d in f_dates:
            bs = float(S[(S["city"]==city) & (S["date"]==d)]["bloom_score"].iloc[0]) if not S.empty and ((S["city"]==city) & (S["date"]==d)).any() else 0.0
            d_bs = bs - base_bs  # change vs late-history average
            dlog_arr = b_arr * d_bs + (-0.15)*np.log(rain_mult) + (0.10)*temp_add + (0.20)*dlog_fx  # signs: rain hurts, warmer slightly helps, weaker JPY helps intl arrivals
            dlog_adr = b_adr * d_bs + (0.05)*dlog_fx + (0.02)*temp_add
            arr0 = arr0 * np.exp(dlog_arr) if arr0==arr0 else np.nan
            adr0 = adr0 * np.exp(dlog_adr) if adr0==adr0 else np.nan
            # a simple RevPAR proxy if occupancy available
            revpar = (adr0 * (occ0/100.0)) if (adr0==adr0 and occ0==occ0) else np.nan
            rows.append({"city": city, "date": d.date(), "bloom_score": bs, "dlog_arrivals": dlog_arr, "dlog_adr": dlog_adr,
                         "arrivals_total": float(arr0) if arr0==arr0 else np.nan,
                         "adr_usd": float(adr0) if adr0==adr0 else np.nan,
                         "revpar_proxy": float(revpar) if revpar==revpar else np.nan,
                         "fx_shock_pct": fx_shock_pct, "rain_shock_pct": rain_shock_pct, "temp_shock_c": temp_shock_c})
    return pd.DataFrame(rows).sort_values(["city","date"])


# ----------------------------- CLI / orchestration -----------------------------

@dataclass
class Config:
    blooms: str
    arrivals: Optional[str]
    bookings: Optional[str]
    weather: Optional[str]
    fx: Optional[str]
    social: Optional[str]
    events: Optional[str]
    freq: str
    bloom_window: int
    sigma: float
    lags: int
    event_window: int
    scenario_raw: Optional[str]
    fx_shock_pct: float
    rain_shock_pct: float
    temp_shock_c: float
    holiday_bonus_pp: float
    horizon: int
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cherry blossom bloom timing → tourism demand, pricing & scenarios")
    ap.add_argument("--blooms", required=True)
    ap.add_argument("--arrivals", default="")
    ap.add_argument("--bookings", default="")
    ap.add_argument("--weather", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--social", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--freq", default="daily", choices=["daily","weekly"])
    ap.add_argument("--bloom_window", type=int, default=14)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--lags", type=int, default=7)
    ap.add_argument("--event_window", type=int, default=7)
    ap.add_argument("--scenario", dest="scenario_raw", default="")
    ap.add_argument("--fx_shock_pct", type=float, default=0.0)
    ap.add_argument("--rain_shock_pct", type=float, default=0.0)
    ap.add_argument("--temp_shock_c", type=float, default=0.0)
    ap.add_argument("--holiday_bonus_pp", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_sakura")
    ap.add_argument("--min_obs", type=int, default=120)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    BLOOM = load_blooms(args.blooms)
    ARR   = load_arrivals(args.arrivals) if args.arrivals else pd.DataFrame()
    BKG   = load_bookings(args.bookings) if args.bookings else pd.DataFrame()
    WTH   = load_weather(args.weather) if args.weather else pd.DataFrame()
    FX    = load_fx(args.fx) if args.fx else pd.DataFrame()
    SOC   = load_social(args.social) if args.social else pd.DataFrame()
    EVT   = load_events(args.events) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        s = pd.to_datetime(args.start).date()
        for df in [ARR,BKG,WTH,FX,SOC,EVT]:
            if not df.empty:
                df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = pd.to_datetime(args.end).date()
        for df in [ARR,BKG,WTH,FX,SOC,EVT]:
            if not df.empty:
                df.drop(df[df["date"] > e].index, inplace=True)

    # Save bloom calendar with alignment
    CAL = bloom_calendar_stats(BLOOM, EVT)
    CAL.to_csv(outdir / "bloom_calendar.csv", index=False)

    # Base panel
    PANEL = build_panel(BLOOM, ARR, BKG, WTH, FX, SOC, EVT, freq=args.freq,
                        sigma=float(args.sigma), window=int(args.bloom_window), holiday_bonus_pp=float(args.holiday_bonus_pp))
    if not PANEL.empty: PANEL.to_csv(outdir / "tourism_panel.csv", index=False)

    # Event study
    ES = event_study(PANEL, BLOOM, window=int(args.event_window), freq=args.freq) if not PANEL.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Elasticities
    EL_ARR = dlag_regression(PANEL, dep_col="dlog_arrivals_total", L=int(args.lags), min_obs=int(args.min_obs)) if ("dlog_arrivals_total" in PANEL.columns) else pd.DataFrame()
    if not EL_ARR.empty: EL_ARR.to_csv(outdir / "elasticity_arrivals.csv", index=False)
    EL_ADR = dlag_regression(PANEL, dep_col="dlog_adr_usd",       L=int(args.lags), min_obs=int(args.min_obs)) if ("dlog_adr_usd" in PANEL.columns) else pd.DataFrame()
    if not EL_ADR.empty: EL_ADR.to_csv(outdir / "elasticity_adr.csv", index=False)

    # Scenarios
    SCN = pd.DataFrame()
    shifts = parse_scenarios(args.scenario_raw) if args.scenario_raw else {}
    if shifts or any([args.fx_shock_pct, args.rain_shock_pct, args.temp_shock_c]):
        BLOOM_S = scenario_shift_blooms(BLOOM, shifts)
        SCN = scenario_apply(PANEL, BLOOM_S, EL_ARR if not EL_ARR.empty else pd.DataFrame(),
                             EL_ADR if not EL_ADR.empty else pd.DataFrame(),
                             fx_shock_pct=float(args.fx_shock_pct),
                             rain_shock_pct=float(args.rain_shock_pct),
                             temp_shock_c=float(args.temp_shock_c),
                             horizon=int(args.horizon),
                             sigma=float(args.sigma), window=int(args.bloom_window),
                             holiday_bonus_pp=float(args.holiday_bonus_pp), freq=args.freq)
        if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    latest = PANEL.groupby("city").tail(1) if not PANEL.empty else pd.DataFrame()
    key = {
        "n_cities": int(PANEL["city"].nunique()) if not PANEL.empty else 0,
        "sample_start": str(pd.to_datetime(PANEL["date"]).min().date()) if not PANEL.empty else None,
        "sample_end": str(pd.to_datetime(PANEL["date"]).max().date()) if not PANEL.empty else None,
        "has_arrivals": "dlog_arrivals_total" in PANEL.columns,
        "has_adr": "dlog_adr_usd" in PANEL.columns,
        "has_fx": "usd_jpy" in PANEL.columns,
        "freq": args.freq
    }
    summary = {
        "key": key,
        "latest_snapshot": [
            {
                "city": r["city"],
                "date": str(pd.to_datetime(r["date"]).date()),
                "arrivals_total": float(r.get("arrivals_total", np.nan)) if "arrivals_total" in PANEL.columns else None,
                "adr_usd": float(r.get("adr_usd", np.nan)) if "adr_usd" in PANEL.columns else None,
                "hotel_occ_pct": float(r.get("hotel_occ_pct", np.nan)) if "hotel_occ_pct" in PANEL.columns else None,
                "bloom_score": float(r.get("bloom_score", np.nan))
            } for _, r in (latest.sort_values("city").iterrows() if not latest.empty else [])
        ],
        "reg_files": {
            "arrivals": "elasticity_arrivals.csv" if not EL_ARR.empty else None,
            "adr": "elasticity_adr.csv" if not EL_ADR.empty else None
        },
        "has_event_study": bool(not ES.empty),
        "has_scenario": bool(not SCN.empty)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        blooms=args.blooms, arrivals=(args.arrivals or None), bookings=(args.bookings or None),
        weather=(args.weather or None), fx=(args.fx or None), social=(args.social or None),
        events=(args.events or None), freq=args.freq, bloom_window=int(args.bloom_window), sigma=float(args.sigma),
        lags=int(args.lags), event_window=int(args.event_window), scenario_raw=(args.scenario_raw or None),
        fx_shock_pct=float(args.fx_shock_pct), rain_shock_pct=float(args.rain_shock_pct),
        temp_shock_c=float(args.temp_shock_c), holiday_bonus_pp=float(args.holiday_bonus_pp),
        horizon=int(args.horizon), start=(args.start or None), end=(args.end or None),
        outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Cherry Blossom → Tourism ==")
    print(f"Cities: {key['n_cities']} | Sample: {key['sample_start']} → {key['sample_end']} | Freq: {key['freq']}")
    if summary["has_event_study"]:
        print(f"Event study output: event_study.csv (±{args.event_window} {('days' if args.freq=='daily' else 'weeks')})")
    if summary["has_scenario"]:
        print(f"Scenario horizon: {args.horizon} {('days' if args.freq=='daily' else 'weeks')} | shifts: {args.scenario_raw or 'none'}")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
