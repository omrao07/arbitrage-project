#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
textile_export_quota.py — Quota tracking, utilization, price effects & scenarios
-------------------------------------------------------------------------------

What this does
==============
A research toolkit to analyze **textile export quotas / TRQs** by product & destination.
It ingests quota windows and trade/firm data to compute:

1) Quota tracking & early warnings
   • Per window: allocated vs shipped, utilization %, run-rate, days-to-exhaust
   • Forecast exhaustion date using recent run-rate
   • Price/UV (unit value) premia when quotas tighten
   • Firm exposure (share of exports tied to constrained windows), if firm IDs exist

2) Monthly product–destination panel
   • Exports (qty/value), unit values, orderbook flow, shares by destination
   • Inputs/FX/freight overlays (cotton, yarn/fabric, USD crosses, container rates)

3) Event studies around policy changes (impose/raise/relax/auction)
   • Δ orders, Δ shipments, Δ unit values in ±K periods around event

4) Elasticities via distributed-lag regressions (Newey–West SEs)
   • dlog(exports) ~ Σ Δquota_availability + controls (FX, input prices, freight)
   • dlog(unit_value) ~ Σ Δquota_availability (pricing power)

5) Substitution diagnostics
   • Shift of volumes across destinations when a destination quota tightens

6) Scenarios
   • Tighten/relax specific quotas by X% → projected impacts on shipments & UV
   • Competitor quota changes (cross-destination substitution)
   • FX shocks or input-cost shocks

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--quotas quotas.csv            REQUIRED (one row per quota window)
  Columns (any subset; keys for matching shipments):
    window_start, window_end           # dates (inclusive)
    destination[, country, market]     # e.g., "US", "EU"
    product[, hs6, category]           # HS-6 or product family
    allocated_qty                      # SAME UNIT as shipments qty
    quota_type                         # 'TRQ' / 'absolute' (optional)
    in_quota_tariff_pct[, out_quota_tariff_pct] (optional)
    revision_date (optional)
    firm (optional; if firm-specific allocation)

--shipments shipments.csv      RECOMMENDED (transactional or monthly)
  Columns:
    date, destination[, country], product[, hs6],
    qty[, quantity_kg, quantity_ton], value_usd[, value],
    firm (optional)

--orders orders.csv            OPTIONAL (order book / invoices)
  Columns:
    date, destination, product, order_qty[, value_usd], cancellations_pct (optional), firm (optional)

--prices prices.csv            OPTIONAL (inputs/FX/freight)
  Columns:
    date,
    cotton_idx[, cotton_usd], yarn_price[, fabric_price],
    fx_usd_local[, usd_inr, usd_try, ...], freight_idx[, container_rate_usd],
    electricity_idx (optional)

--competitor competitor.csv    OPTIONAL (competitor quotas / signals)
  Columns:
    date, destination, product, quota_tightness_idx[, competitor_allocation, competitor_utilization_pct]

--events events.csv            OPTIONAL (policy changes/announcements)
  Columns:
    date, destination[, product], label[, type]  # type: IMPOSE/RAISE/RELAX/AUCTION/EXTEND

Key CLI
-------
--freq monthly|quarterly       Output frequency (default monthly)
--lags 6                       Max lag for d-lag regressions
--event_window 3               Half-window (periods) for event studies
--util_warn 0.85               Utilization threshold for warning
--rr_days 28                   Run-rate lookback (days)
--scenario "+10:US:HS6109,+0:EU:HS6110"  Quota shocks "±PCT:DEST:PRODUCT" (comma-separated)
--fx_shock_pct 0               Scenario FX shock (% USD appreciation vs local; sign as +)
--input_shock_pct 0            Scenario cotton/yarn shock (%)
--start / --end                Sample filters (YYYY-MM-DD)
--outdir out_textiles          Output directory
--min_obs 24                   Minimum obs for regressions

Outputs
-------
- window_utilization.csv       One row per quota window (+ firm if present)
- monthly_panel.csv            Product–destination monthly panel (qty/value/UV + drivers)
- firm_exposure.csv            Firm-level exposure to tight quotas (if firm available)
- event_study.csv              Δ around events for qty / UV / orders
- elasticities_qty.csv         d-lag regression for exports growth
- elasticities_uv.csv          d-lag regression for unit values
- substitution.csv             Destination share shifts when a destination tightens
- scenarios.csv                Scenario impacts on qty & UV
- warnings.csv                 Early-warning signals (utilization/run-rate/exhaustion)
- summary.json                 Headline metrics & pointers
- config.json                  Echo of run configuration

Assumptions & Notes
-------------------
• Matching key is (destination, product) with flexible column names (hs6 or product).
• Shipments/allocated units must be consistent (e.g., kg/tons). The script will not convert.
• If shipments are daily, they are aggregated to month/quarter ends by --freq.
• Run-rate uses the last rr_days of daily shipments; with monthly data, uses last 1 period.
• Elasticities are reduced-form and illustrative; validate before operational use.

DISCLAIMER
----------
This is research tooling. Validate column mappings, units, and robustness before decisions.
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

def to_period_end(s: pd.Series, freq: str) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    return (dt.dt.to_period("M").dt.to_timestamp("M") if freq.startswith("M")
            else dt.dt.to_period("Q").dt.to_timestamp("Q"))

def to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def yoy(s: pd.Series, periods: int) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s) - np.log(s.shift(periods))

def d(s: pd.Series) -> pd.Series:
    return s.astype(float).diff()

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

def load_quotas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ws = ncol(df, "window_start","start","quota_start")
    we = ncol(df, "window_end","end","quota_end")
    dst= ncol(df, "destination","country","market")
    prod = ncol(df, "product","hs6","category","hs_code")
    alloc = ncol(df, "allocated_qty","allocation_qty","quota_qty","allocated")
    qt = ncol(df, "quota_type","type")
    inq = ncol(df, "in_quota_tariff_pct","in_quota_tariff")
    ouq = ncol(df, "out_quota_tariff_pct","out_quota_tariff")
    rev = ncol(df, "revision_date","revised","announce_date")
    firm = ncol(df, "firm","exporter")
    if not (ws and we and dst and prod and alloc):
        raise ValueError("quotas.csv must include window_start, window_end, destination, product, allocated_qty.")
    df = df.rename(columns={ws:"window_start", we:"window_end", dst:"destination", prod:"product", alloc:"allocated_qty"})
    df["window_start"] = to_datetime(df["window_start"])
    df["window_end"]   = to_datetime(df["window_end"])
    df["allocated_qty"] = safe_num(df["allocated_qty"])
    if qt:  df = df.rename(columns={qt:"quota_type"})
    if inq: df = df.rename(columns={inq:"in_quota_tariff_pct"}); df["in_quota_tariff_pct"] = safe_num(df["in_quota_tariff_pct"])
    if ouq: df = df.rename(columns={ouq:"out_quota_tariff_pct"}); df["out_quota_tariff_pct"] = safe_num(df["out_quota_tariff_pct"])
    if rev: df = df.rename(columns={rev:"revision_date"}); df["revision_date"] = to_datetime(df["revision_date"])
    if firm: df = df.rename(columns={firm:"firm"})
    # window id for grouping
    df["window_id"] = (df["destination"].astype(str).str.upper().str.strip() + "|" +
                       df["product"].astype(str).str.upper().str.strip() + "|" +
                       df["window_start"].dt.strftime("%Y-%m-%d") + "→" + df["window_end"].dt.strftime("%Y-%m-%d"))
    return df

def load_shipments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt  = ncol(df, "date","ship_date","invoice_date")
    dst = ncol(df, "destination","country","market")
    prod= ncol(df, "product","hs6","category","hs_code")
    qty = ncol(df, "qty","quantity","quantity_kg","quantity_ton")
    val = ncol(df, "value_usd","value","usd_value")
    firm= ncol(df, "firm","exporter")
    if not (dt and dst and prod and qty and val):
        raise ValueError("shipments.csv needs date, destination, product, qty, value_usd.")
    df = df.rename(columns={dt:"date", dst:"destination", prod:"product", qty:"qty", val:"value_usd"})
    df["date"] = to_datetime(df["date"])
    for c in ["qty","value_usd"]:
        df[c] = safe_num(df[c])
    df["unit_value_usd"] = df["value_usd"] / df["qty"].replace(0,np.nan)
    if firm: df = df.rename(columns={firm:"firm"})
    return df

def load_orders(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date")
    dst = ncol(df, "destination","country","market")
    prod= ncol(df, "product","hs6","category")
    oq  = ncol(df, "order_qty","orders_qty","qty")
    ov  = ncol(df, "value_usd","value")
    can = ncol(df, "cancellations_pct","cancel_pct")
    firm= ncol(df, "firm","exporter")
    if not (dt and dst and prod and oq):
        raise ValueError("orders.csv needs date, destination, product, order_qty.")
    df = df.rename(columns={dt:"date", dst:"destination", prod:"product", oq:"order_qty"})
    df["date"] = to_datetime(df["date"]); df["order_qty"] = safe_num(df["order_qty"])
    if ov: df = df.rename(columns={ov:"order_value_usd"}); df["order_value_usd"] = safe_num(df["order_value_usd"])
    if can: df = df.rename(columns={can:"cancellations_pct"}); df["cancellations_pct"] = safe_num(df["cancellations_pct"])
    if firm: df = df.rename(columns={firm:"firm"})
    return df

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date")
    if not dt: raise ValueError("prices.csv needs date.")
    df = df.rename(columns={dt:"date"}); df["date"] = to_datetime(df["date"])
    ren = {}
    for src,out in [
        ("cotton_idx","cotton_idx"), ("cotton_usd","cotton_usd"),
        ("yarn_price","yarn_price"), ("fabric_price","fabric_price"),
        ("fx_usd_local","fx_usd_local"), ("usd_inr","fx_usd_local"),
        ("freight_idx","freight_idx"), ("container_rate_usd","container_rate_usd"),
        ("electricity_idx","electricity_idx")
    ]:
        c = ncol(df, src)
        if c: ren[c]=out
    df = df.rename(columns=ren)
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    return df

def load_competitor(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date")
    dst= ncol(df, "destination","country")
    prod= ncol(df, "product","hs6","category")
    if not (dt and dst and prod): raise ValueError("competitor.csv needs date, destination, product.")
    df = df.rename(columns={dt:"date", dst:"destination", prod:"product"})
    df["date"] = to_datetime(df["date"])
    for c in df.columns:
        if c not in ["date","destination","product"]:
            df[c] = safe_num(df[c])
    return df

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date")
    dst= ncol(df, "destination","country")
    prod= ncol(df, "product","hs6","category")
    lab= ncol(df, "label","event","name") or "label"
    typ= ncol(df, "type","kind") or None
    if not dt: raise ValueError("events.csv needs date.")
    df = df.rename(columns={dt:"date", lab:"label"})
    if dst: df = df.rename(columns={dst:"destination"})
    if prod:df = df.rename(columns={prod:"product"})
    if typ: df = df.rename(columns={typ:"type"})
    df["date"] = to_datetime(df["date"])
    if "type" not in df.columns: df["type"] = ""
    return df.sort_values("date")


# ----------------------------- constructions -----------------------------

def assign_windows(ship: pd.DataFrame, Q: pd.DataFrame) -> pd.DataFrame:
    """
    For each shipment row, attach the matching quota window (by destination & product whose date ∈ [start,end]).
    If multiple firm-specific and general windows exist, prefer firm-specific match.
    """
    if ship.empty or Q.empty: return pd.DataFrame()
    # Pre-split by destination|product to reduce join size
    ship["dst_prod"] = ship["destination"].astype(str).str.upper().str.strip()+"|"+ship["product"].astype(str).str.upper().str.strip()
    Q["dst_prod"]    = Q["destination"].astype(str).str.upper().str.strip()+"|"+Q["product"].astype(str).str.upper().str.strip()
    rows = []
    # Build interval lookup per key
    for key, qg in Q.groupby("dst_prod"):
        sg = ship[ship["dst_prod"]==key]
        if sg.empty: continue
        # firm-specific first
        if "firm" in sg.columns and "firm" in qg.columns:
            # try firm match; else fallback to generic (NaN firm in quotas)
            # concat firm-specific and generic
            parts = []
            f_specific = sg.merge(qg.dropna(subset=["firm"]), on=["dst_prod","firm"], how="left", suffixes=("","_q"))
            f_generic  = sg.merge(qg[qg["firm"].isna()] if qg["firm"].isna().any() else qg, on=["dst_prod"], how="left", suffixes=("","_q"))
            parts = [f_specific, f_generic]
            cand = pd.concat(parts, ignore_index=True, sort=False)
        else:
            cand = sg.merge(qg, on="dst_prod", how="left", suffixes=("","_q"))
        # filter by interval
        cand = cand[(cand["date"]>=cand["window_start"]) & (cand["date"]<=cand["window_end"])]
        # prefer firm-specific when duplicates (drop duplicates by earliest end date)
        cand.sort_values(["date","firm","window_end"], inplace=True)
        # deduplicate per shipment row id (use index)
        cand["row_id"] = cand.index
        best = cand.drop_duplicates(subset=["date","destination","product","qty","value_usd","firm"], keep="first")
        rows.append(best)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["within_window"] = True
    return out

def window_utilization(Q: pd.DataFrame, ship_win: pd.DataFrame,
                       util_warn: float, rr_days: int) -> pd.DataFrame:
    """
    Compute shipped qty within each window, utilization %, run-rate (recent), and forecast exhaustion.
    """
    # shipped within window
    if ship_win is None or ship_win.empty:
        shipped = pd.DataFrame(columns=["window_id","shipped_qty","shipped_value_usd","avg_uv_usd"])
    else:
        shipped = (ship_win.groupby("window_id", as_index=False)
                          .agg(shipped_qty=("qty","sum"),
                               shipped_value_usd=("value_usd","sum"),
                               avg_uv_usd=("unit_value_usd","mean")))
    # join
    W = Q.merge(shipped, on="window_id", how="left")
    W["shipped_qty"] = W["shipped_qty"].fillna(0.0)
    W["utilization_pct"] = W["shipped_qty"] / W["allocated_qty"].replace(0,np.nan)
    # days & run-rate
    W["window_days"] = (W["window_end"] - W["window_start"]).dt.days.clip(lower=1)
    # recent run rate: use last rr_days shipments within window (if daily). If shipments monthly, fallback to last period per destination/product.
    rr = []
    if ship_win is not None and not ship_win.empty:
        for wid, g in ship_win.groupby("window_id"):
            g = g.sort_values("date")
            # pick last rr_days
            if (g["date"].dt.normalize().nunique() > 20):  # assume daily-ish
                cutoff = g["date"].max() - pd.Timedelta(days=rr_days)
                recent = g[g["date"]>cutoff]["qty"].sum()
                days = max(1, (g["date"].max() - cutoff).days)
                rr_val = recent / days
            else:
                # monthly-ish fallback: last period shipment divided by period days ~ 30
                last = g[g["date"]==g["date"].max()]["qty"].sum()
                rr_val = last / 30.0
            rr.append({"window_id": wid, "run_rate_qty_per_day": rr_val})
    RR = pd.DataFrame(rr)
    W = W.merge(RR, on="window_id", how="left")
    # forecast exhaustion date
    fc_rows = []
    for _, r in W.iterrows():
        shipped = float(r.get("shipped_qty",0.0))
        alloc   = float(r.get("allocated_qty", np.nan))
        start   = r["window_start"]; end = r["window_end"]
        rrpd    = float(r.get("run_rate_qty_per_day", np.nan))
        if np.isnan(alloc) or np.isnan(rrpd) or rrpd<=0:
            fc_rows.append(np.nan)
            continue
        remaining = max(0.0, alloc - shipped)
        days_needed = remaining / rrpd if rrpd>0 else np.inf
        ex_date = r["date"] if "date" in W.columns else pd.NaT  # not used
        exhaustion = r["window_start"] + pd.to_timedelta(int(np.ceil(days_needed)), unit="D")
        # cap at window end
        exhaustion = min(exhaustion, end)
        fc_rows.append(exhaustion)
    W["forecast_exhaustion"] = fc_rows
    W["days_to_exhaustion"] = (W["forecast_exhaustion"] - pd.Timestamp.today().normalize()).dt.days
    W["flag_util_high"] = (W["utilization_pct"] >= util_warn)
    W["flag_exhaust_early"] = (W["forecast_exhaustion"].notna()) & (W["forecast_exhaustion"] < W["window_end"])
    # warnings table
    warn = W[["window_id","destination","product","window_start","window_end","allocated_qty","shipped_qty",
              "utilization_pct","forecast_exhaustion","flag_util_high","flag_exhaust_early"]].copy()
    return W.sort_values(["destination","product","window_start"]), warn.sort_values(["flag_util_high","flag_exhaust_early"], ascending=False)

def monthly_panel(ship: pd.DataFrame, orders: pd.DataFrame, prices: pd.DataFrame,
                  competitor: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate to product–destination by month/quarter."""
    if ship.empty: return pd.DataFrame()
    ship["period"] = to_period_end(ship["date"], "M" if freq.startswith("m") else "Q")
    pan = (ship.groupby(["period","destination","product"], as_index=False)
                .agg(qty=("qty","sum"),
                     value_usd=("value_usd","sum"),
                     uv_usd=("unit_value_usd","mean")))
    # Orders
    if not orders.empty:
        orders["period"] = to_period_end(orders["date"], "M" if freq.startswith("m") else "Q")
        og = orders.groupby(["period","destination","product"], as_index=False).agg(
            order_qty=("order_qty","sum"),
            order_value_usd=("order_value_usd","sum")
        )
        pan = pan.merge(og, on=["period","destination","product"], how="left")
    # Drivers (date-only or by dest/product)
    if not prices.empty:
        prices["period"] = to_period_end(prices["date"], "M" if freq.startswith("m") else "Q")
        pr = prices.drop(columns=[c for c in ["date"] if c in prices.columns]).groupby("period", as_index=False).mean(numeric_only=True)
        pan = pan.merge(pr, on="period", how="left")
    if not competitor.empty:
        competitor["period"] = to_period_end(competitor["date"], "M" if freq.startswith("m") else "Q")
        cg = competitor.groupby(["period","destination","product"], as_index=False).mean(numeric_only=True)
        pan = pan.merge(cg, on=["period","destination","product"], how="left")
    # Transforms
    pan = pan.sort_values(["destination","product","period"])
    pan["dlog_qty"] = pan.groupby(["destination","product"])["qty"].apply(dlog).reset_index(level=[0,1], drop=True)
    pan["dlog_uv"]  = pan.groupby(["destination","product"])["uv_usd"].apply(dlog).reset_index(level=[0,1], drop=True)
    yo = 12 if freq.startswith("m") else 4
    pan["yoy_qty"]  = pan.groupby(["destination","product"])["qty"].apply(lambda s: yoy(s, yo)).reset_index(level=[0,1], drop=True)
    pan["yoy_uv"]   = pan.groupby(["destination","product"])["uv_usd"].apply(lambda s: yoy(s, yo)).reset_index(level=[0,1], drop=True)
    # Destination shares per product
    tot = pan.groupby(["period","product"], as_index=False)["qty"].sum().rename(columns={"qty":"qty_total_prod"})
    pan = pan.merge(tot, on=["period","product"], how="left")
    pan["dest_share"] = pan["qty"] / pan["qty_total_prod"].replace(0,np.nan)
    return pan

# ----------------------------- event study -----------------------------

def event_study(pan: pd.DataFrame, events: pd.DataFrame, window: int) -> pd.DataFrame:
    if pan.empty or events.empty: return pd.DataFrame()
    rows = []
    idx = pan.set_index(["period","destination","product"]).sort_index()
    dates = sorted(pan["period"].unique())
    for _, ev in events.iterrows():
        d0 = to_period_end(pd.Series([ev["date"]]), "M").iloc[0]
        # choose closest available period
        d0_use = max([t for t in dates if t <= d0], default=None)
        if d0_use is None: continue
        # filter scope
        scope = idx
        if not pd.isna(ev.get("destination", np.nan)):
            scope = scope.loc[(slice(None), str(ev["destination"]), slice(None)), :]
        if not pd.isna(ev.get("product", np.nan)):
            scope = scope.loc[(slice(None), slice(None), str(ev["product"])), :]
        # walk h=-w..+w
        for (per,dst,prd), row in scope.iterrows():
            if per not in dates: continue
            # compute relative h wrt d0_use
            try:
                h = dates.index(per) - dates.index(d0_use)
            except ValueError:
                continue
            if -window <= h <= window:
                rows.append({"event_date": d0_use, "h": h, "destination": dst, "product": prd,
                             "label": ev.get("label",""), "type": ev.get("type",""),
                             "dlog_qty": row.get("dlog_qty", np.nan), "dlog_uv": row.get("dlog_uv", np.nan),
                             "order_qty": row.get("order_qty", np.nan)})
    df = pd.DataFrame(rows)
    if df.empty: return df
    # Δ vs pre-event average (h<0)
    out = []
    for (dst, prd, d0), g in df.groupby(["destination","product","event_date"]):
        base = g[g["h"]<0][["dlog_qty","dlog_uv","order_qty"]].mean(numeric_only=True)
        for _, r in g.iterrows():
            rec = {"destination":dst, "product":prd, "event_date":d0, "h":int(r["h"]), "label":r["label"], "type":r["type"]}
            for c in ["dlog_qty","dlog_uv","order_qty"]:
                val = r.get(c, np.nan); rec[f"delta_{c}"] = float(val - base.get(c, np.nan)) if pd.notna(val) else np.nan
            out.append(rec)
    return pd.DataFrame(out).sort_values(["destination","product","event_date","h"])

# ----------------------------- regressions -----------------------------

def build_quota_availability(pan: pd.DataFrame, Q: pd.DataFrame) -> pd.DataFrame:
    """
    Map each period-product-destination to 'quota availability share' within its active windows:
      availability = remaining_qty / allocated_qty   (0..1, clipped)
    Aggregates by averaging across windows overlapping the period midpoint.
    """
    if pan.empty or Q.empty: return pd.DataFrame()
    mid = pan["period"] - pd.offsets.MonthEnd(0) + pd.offsets.Day(15)
    P = pan.assign(mid_date=mid)
    Q2 = Q[["destination","product","window_start","window_end","allocated_qty"]].copy()
    Q2["key"] = (Q2["destination"].astype(str).str.upper().str.strip()+"|"+
                 Q2["product"].astype(str).str.upper().str.strip())
    rows = []
    # pre-compute shipped-to-date within window using shipments mapping is heavy; approximate with utilization_pct from window_utilization if already computed externally
    # fallback: assume linear consumption → remaining at mid-date = allocated * (1 - elapsed/window_days)
    for _, r in P.iterrows():
        key = str(r["destination"]).upper().strip()+"|"+str(r["product"]).upper().strip()
        cand = Q2[(Q2["key"]==key) & (Q2["window_start"]<=r["mid_date"]) & (Q2["window_end"]>=r["mid_date"])]
        if cand.empty:
            avail = np.nan
        else:
            # linear depletion proxy
            rr = []
            for _, q in cand.iterrows():
                elapsed = (r["mid_date"] - q["window_start"]).days
                total   = max(1, (q["window_end"] - q["window_start"]).days)
                remain_share = max(0.0, 1.0 - elapsed/total)
                rr.append(remain_share)
            avail = float(np.mean(rr)) if rr else np.nan
        rows.append(avail)
    P["quota_availability_share"] = rows
    return P.drop(columns=["mid_date"])

def dlag_regression(pan: pd.DataFrame, dep_col: str, L: int, min_obs: int) -> pd.DataFrame:
    """
    Panel-by-(destination,product) regression with HAC SEs:
      dep_t = α + Σ_{l=0..L} β_l * Δquota_availability_{t−l}
                    + γ * dlog(fx_usd_local) + δ * dlog(cotton/yarn/freight)
    """
    if dep_col not in pan.columns or "quota_availability_share" not in pan.columns:
        return pd.DataFrame()
    out = []
    for (dst,prd), g in pan.groupby(["destination","product"]):
        g = g.sort_values("period")
        dep = g[dep_col]
        Xparts = [pd.Series(1.0, index=g.index, name="const")]
        names = ["const"]
        qa = g["quota_availability_share"]
        dqa = qa.diff()
        for l in range(0, L+1):
            nm = f"dqa_l{l}"
            Xparts.append(dqa.shift(l).rename(nm)); names.append(nm)
        # controls (Δlog where possible)
        controls = []
        for c in ["fx_usd_local","cotton_idx","cotton_usd","yarn_price","freight_idx","container_rate_usd","electricity_idx"]:
            if c in g.columns:
                s = g[c]
                if s.notna().sum() >= min_obs//2:
                    nm = f"dlog_{c}"
                    Xparts.append(np.log(s).diff().rename(nm))
                    names.append(nm)
                    controls.append(nm)
        X = pd.concat(Xparts, axis=1)
        XY = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        if XY.shape[0] < max(min_obs, 5*X.shape[1]): 
            continue
        yv = XY["dep"].values.reshape(-1,1)
        Xv = XY.drop(columns=["dep"]).values
        beta, resid, XtX_inv = ols_beta_se(Xv, yv)
        se = hac_se(Xv, resid, XtX_inv, L=max(6, L))
        for i, nm in enumerate(names):
            out.append({"destination":dst, "product":prd, "dep":dep_col, "var":nm,
                        "coef": float(beta[i,0]), "se": float(se[i]),
                        "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                        "n": int(XY.shape[0]), "lags": int(L)})
        # cumulative effect of dqa 0..L
        idxs = [i for i,nm in enumerate(names) if nm.startswith("dqa_l")]
        if idxs:
            bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
            out.append({"destination":dst, "product":prd, "dep":dep_col, "var":"dqa_cum_0..L",
                        "coef": bsum, "se": ses, "t_stat": bsum/(ses if ses>0 else np.nan),
                        "n": int(XY.shape[0]), "lags": int(L)})
    return pd.DataFrame(out)


# ----------------------------- substitution -----------------------------

def substitution_diag(pan: pd.DataFrame, util: pd.DataFrame, util_thresh: float=0.9) -> pd.DataFrame:
    """
    When destination A is tight (util >= thresh) for product p in a period,
    measure Δ share to other destinations in t..t+1 relative to t-1..t baseline.
    """
    if pan.empty or util.empty: return pd.DataFrame()
    # Construct period-level utilization flag by aligning period midpoint to windows
    mid = pan["period"] - pd.offsets.MonthEnd(0) + pd.offsets.Day(15)
    pan2 = pan.assign(mid_date=mid)
    tight = []
    for _, r in pan2.iterrows():
        # find any window covering mid_date
        mask = (util["destination"].astype(str).str.upper()==str(r["destination"]).upper()) & \
               (util["product"].astype(str).str.upper()==str(r["product"]).upper()) & \
               (util["window_start"]<=r["mid_date"]) & (util["window_end"]>=r["mid_date"])
        U = util[mask]
        is_tight = (U["utilization_pct"]>=util_thresh).any() if not U.empty else False
        tight.append(is_tight)
    pan2["tight_here"] = tight
    # For each (product,period) where some destination tight, compute share changes
    rows = []
    dates = sorted(pan2["period"].unique())
    for prd, gp in pan2.groupby("product"):
        for t in range(1, len(dates)-1):
            per = dates[t]
            g = gp[gp["period"]==per]
            if not g["tight_here"].any(): 
                continue
            prev = gp[gp["period"]==dates[t-1]][["destination","dest_share"]].set_index("destination")
            nxt  = gp[gp["period"]==dates[t+1]][["destination","dest_share"]].set_index("destination")
            cur  = g[["destination","dest_share","tight_here"]].set_index("destination")
            # change in shares
            for dst in cur.index:
                rows.append({"product": prd, "period": per, "destination": dst,
                             "tight_here": bool(cur.loc[dst]["tight_here"]),
                             "d_share_forward": float(nxt["dest_share"].get(dst, np.nan) - cur["dest_share"].get(dst, np.nan)),
                             "d_share_back": float(cur["dest_share"].get(dst, np.nan) - prev["dest_share"].get(dst, np.nan))})
    return pd.DataFrame(rows).sort_values(["product","period","destination"])


# ----------------------------- scenarios -----------------------------

def parse_scenarios(s: str) -> List[Tuple[float,str,str]]:
    """
    "+10:US:HS6109,-20:EU:HS6110" -> [(+10,'US','HS6109'), (-20,'EU','HS6110')]
    Use '*' to wildcard destination or product.
    """
    if not s: return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        pct, dst, prd = p.split(":")
        out.append((float(pct), dst.strip(), prd.strip()))
    return out

def scenario_apply(pan: pd.DataFrame, elast_qty: pd.DataFrame, elast_uv: pd.DataFrame,
                   shocks: List[Tuple[float,str,str]], fx_shock_pct: float, input_shock_pct: float, horizon: int=6) -> pd.DataFrame:
    """
    Map quota shocks (±%) into Δlog qty & UV using cumulative coefficients on Δquota_availability (dqa_cum_0..L).
    Approximate mapping: a +10% quota change ≈ +10pp availability (level) → proxy as dqa sum over horizon.
    FX and input shocks map via their Δlog coefficients if available.
    """
    if pan.empty or not shocks: return pd.DataFrame()
    last = pan.groupby(["destination","product"]).tail(1)
    rows = []
    # fetch cumulative betas
    def pick_beta(df, dst, prd):
        r = df[(df["destination"]==dst) & (df["product"]==prd) & (df["var"]=="dqa_cum_0..L")]
        return float(r["coef"].iloc[0]) if not r.empty else None

    for _, r in last.iterrows():
        dst, prd = r["destination"], r["product"]
        # match shocks
        applicable = [sc for sc in shocks if (sc[1] in [dst,"*"]) and (sc[2] in [prd,"*"])]
        if not applicable: continue
        # combine (sum) shocks for this pair
        dq = sum(sc[0] for sc in applicable) / 100.0  # as fraction
        bq = pick_beta(elast_qty, dst, prd)
        bu = pick_beta(elast_uv,  dst, prd)
        # fallbacks
        if bq is None: bq = 0.30   # quantity responds +0.30 per +1.0 availability (heuristic)
        if bu is None: bu = 0.10   # UV responds +0.10 per +1.0 availability (pricing power fades)
        # FX / inputs (use simple pass-through heuristics if no regression)
        bfx = 0.20  # +Δlog(FX USD/local) reduces competitiveness; we treat positive FX as USD up (local dep.)
        binp = -0.15  # +Δlog(input) hurts margin → may cut UV or qty
        dlog_fx   = np.log(1.0 + fx_shock_pct/100.0) if fx_shock_pct else 0.0
        dlog_inp  = np.log(1.0 + input_shock_pct/100.0) if input_shock_pct else 0.0
        # impacts (level shift applied over horizon)
        dlog_qty = bq * dq + (-bfx)*dlog_fx + (binp)*dlog_inp
        dlog_uv  = bu * dq + (bfx)*dlog_fx + (-0.3*binp)*dlog_inp
        # build simple forward path
        qty = r["qty"]; uv = r["uv_usd"]
        for t in range(1, horizon+1):
            qty = qty * np.exp(dlog_qty) if pd.notna(qty) else np.nan
            uv  = uv  * np.exp(dlog_uv)  if pd.notna(uv)  else np.nan
            rows.append({"destination": dst, "product": prd, "t": t,
                         "date0": str(r["period"].date()),
                         "shock_availability": dq, "dlog_qty": dlog_qty, "dlog_uv": dlog_uv,
                         "qty": float(qty) if pd.notna(qty) else np.nan,
                         "uv_usd": float(uv) if pd.notna(uv) else np.nan})
    return pd.DataFrame(rows)


# ----------------------------- CLI / main -----------------------------

@dataclass
class Config:
    quotas: str
    shipments: str
    orders: Optional[str]
    prices: Optional[str]
    competitor: Optional[str]
    events: Optional[str]
    freq: str
    lags: int
    event_window: int
    util_warn: float
    rr_days: int
    scenario: Optional[str]
    fx_shock_pct: float
    input_shock_pct: float
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Textile export quotas: tracking, effects & scenarios")
    ap.add_argument("--quotas", required=True)
    ap.add_argument("--shipments", required=True)
    ap.add_argument("--orders", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--competitor", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","quarterly"])
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--event_window", type=int, default=3)
    ap.add_argument("--util_warn", type=float, default=0.85)
    ap.add_argument("--rr_days", type=int, default=28)
    ap.add_argument("--scenario", default="")
    ap.add_argument("--fx_shock_pct", type=float, default=0.0)
    ap.add_argument("--input_shock_pct", type=float, default=0.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_textiles")
    ap.add_argument("--min_obs", type=int, default=24)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq = "M" if args.freq.startswith("m") else "Q"

    Q  = load_quotas(args.quotas)
    S  = load_shipments(args.shipments)
    O  = load_orders(args.orders) if args.orders else pd.DataFrame()
    P  = load_prices(args.prices) if args.prices else pd.DataFrame()
    C  = load_competitor(args.competitor) if args.competitor else pd.DataFrame()
    EV = load_events(args.events) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        t0 = pd.to_datetime(args.start)
        for df in [Q, S, O, P, C, EV]:
            if df is not None and not df.empty and ("date" in df.columns or "window_start" in df.columns):
                if "date" in df.columns:
                    df.drop(df[df["date"] < t0].index, inplace=True)
                if "window_start" in df.columns:
                    df.drop(df[df["window_end"] < t0].index, inplace=True)
    if args.end:
        t1 = pd.to_datetime(args.end)
        for df in [Q, S, O, P, C, EV]:
            if df is not None and not df.empty and ("date" in df.columns or "window_end" in df.columns):
                if "date" in df.columns:
                    df.drop(df[df["date"] > t1].index, inplace=True)
                if "window_end" in df.columns:
                    df.drop(df[df["window_start"] > t1].index, inplace=True)

    # Map shipments to windows & compute utilization
    SW = assign_windows(S, Q)
    WUTIL, WARN = window_utilization(Q, SW, util_warn=float(args.util_warn), rr_days=int(args.rr_days))
    if not WUTIL.empty: WUTIL.to_csv(outdir / "window_utilization.csv", index=False)
    if not WARN.empty: WARN.to_csv(outdir / "warnings.csv", index=False)

    # Monthly panel (product–destination)
    PAN = monthly_panel(S, O, P, C, args.freq)
    # Attach quota availability proxy
    if not PAN.empty and not Q.empty:
        PAN = build_quota_availability(PAN, Q)
    if not PAN.empty: PAN.to_csv(outdir / "monthly_panel.csv", index=False)

    # Firm exposure (if firm present)
    FIRM = pd.DataFrame()
    if "firm" in S.columns:
        # firm's share of qty falling into windows with high utilization or early exhaustion
        tight_wids = WUTIL[(WUTIL["utilization_pct"]>=args.util_warn) | (WUTIL["flag_exhaust_early"])]["window_id"].unique().tolist()
        if SW is not None and not SW.empty and tight_wids:
            f = (SW[SW["window_id"].isin(tight_wids)]
                    .groupby("firm", as_index=False)
                    .agg(tight_qty=("qty","sum"), tight_value_usd=("value_usd","sum")))
            total = S.groupby("firm", as_index=False).agg(total_qty=("qty","sum"), total_value_usd=("value_usd","sum"))
            FIRM = f.merge(total, on="firm", how="right")
            FIRM["exposure_share_qty"] = FIRM["tight_qty"].fillna(0) / FIRM["total_qty"].replace(0,np.nan)
            FIRM["exposure_share_value"] = FIRM["tight_value_usd"].fillna(0) / FIRM["total_value_usd"].replace(0,np.nan)
            FIRM.fillna(0, inplace=True)
            FIRM.to_csv(outdir / "firm_exposure.csv", index=False)

    # Event study
    ES = event_study(PAN, EV, window=int(args.event_window)) if not PAN.empty and not EV.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Elasticities
    EL_QTY = dlag_regression(PAN, dep_col="dlog_qty", L=int(args.lags), min_obs=int(args.min_obs)) if not PAN.empty else pd.DataFrame()
    if not EL_QTY.empty: EL_QTY.to_csv(outdir / "elasticities_qty.csv", index=False)
    EL_UV  = dlag_regression(PAN, dep_col="dlog_uv",  L=int(args.lags), min_obs=int(args.min_obs)) if not PAN.empty else pd.DataFrame()
    if not EL_UV.empty: EL_UV.to_csv(outdir / "elasticities_uv.csv", index=False)

    # Substitution
    SUB = substitution_diag(PAN, WUTIL) if (not PAN.empty and not WUTIL.empty) else pd.DataFrame()
    if not SUB.empty: SUB.to_csv(outdir / "substitution.csv", index=False)

    # Scenarios
    SCN = pd.DataFrame()
    shocks = parse_scenarios(args.scenario) if args.scenario else []
    if shocks:
        SCN = scenario_apply(PAN, EL_QTY if not EL_QTY.empty else pd.DataFrame(),
                             EL_UV if not EL_UV.empty else pd.DataFrame(),
                             shocks=shocks, fx_shock_pct=float(args.fx_shock_pct),
                             input_shock_pct=float(args.input_shock_pct), horizon=6)
        if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    latest_util = {}
    if not WUTIL.empty:
        # latest windows by end date
        top = WUTIL.sort_values("window_end").tail(5)
        for _, r in top.iterrows():
            latest_util[r["window_id"]] = {"dest": r["destination"], "prod": r["product"],
                                           "util": float(r.get("utilization_pct", np.nan)),
                                           "forecast_exhaustion": str(r.get("forecast_exhaustion", "")) if pd.notna(r.get("forecast_exhaustion", np.nan)) else None}
    key_stats = {
        "n_windows": int(len(WUTIL)) if not WUTIL.empty else 0,
        "n_tight_windows": int((WUTIL["utilization_pct"]>=args.util_warn).sum()) if not WUTIL.empty else 0,
        "n_warn": int(WARN.shape[0]) if not WARN.empty else 0,
        "panel_periods": int(PAN["period"].nunique()) if not PAN.empty else 0,
        "event_study": bool(not ES.empty),
        "scenario": bool(not SCN.empty)
    }
    summary = {
        "date_run": str(pd.Timestamp.today().date()),
        "latest_windows": latest_util,
        "key_stats": key_stats,
        "files": {
            "window_utilization": "window_utilization.csv" if not WUTIL.empty else None,
            "monthly_panel": "monthly_panel.csv" if not PAN.empty else None,
            "firm_exposure": "firm_exposure.csv" if not FIRM.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "elasticities_qty": "elasticities_qty.csv" if not EL_QTY.empty else None,
            "elasticities_uv": "elasticities_uv.csv" if not EL_UV.empty else None,
            "substitution": "substitution.csv" if not SUB.empty else None,
            "scenarios": "scenarios.csv" if not SCN.empty else None,
            "warnings": "warnings.csv" if not WARN.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        quotas=args.quotas, shipments=args.shipments, orders=(args.orders or None),
        prices=(args.prices or None), competitor=(args.competitor or None), events=(args.events or None),
        freq=args.freq, lags=int(args.lags), event_window=int(args.event_window),
        util_warn=float(args.util_warn), rr_days=int(args.rr_days), scenario=(args.scenario or None),
        fx_shock_pct=float(args.fx_shock_pct), input_shock_pct=float(args.input_shock_pct),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Textile Export Quotas ==")
    print(f"Windows: {key_stats['n_windows']} | Tight (≥{args.util_warn:.0%}): {key_stats['n_tight_windows']} | Warnings: {key_stats['n_warn']}")
    if latest_util:
        print("Recent windows:")
        for wid, info in latest_util.items():
            print(f"  {info['dest']}-{info['prod']} | util {info['util']:.1%} | exhaustion {info['forecast_exhaustion']}")
    if key_stats["event_study"]:
        print(f"Event study written (±{args.event_window} {args.freq[0]}).")
    if key_stats["scenario"]:
        print("Scenario outputs available (see scenarios.csv).")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
