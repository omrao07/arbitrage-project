#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eurozone_pmi_surprise.py — PMI surprise analytics for Eurozone & member states

What this does
--------------
Given PMI release data (actual, consensus) and optional country weights and market prices,
this script computes:
- Surprise = actual − consensus for each release (Manufacturing/Services/Composite; Flash/Final)
- Rolling-standardized surprise (z-score) per (country, sector, stage)
- Eurozone-aggregate surprise from country releases using GDP/exports weights (optional)
- Release calendar table (with timezone-aware timestamps)
- Event study: asset returns around release times (intraday minutes and/or daily)
- Simple impact regressions: asset return ~ standardized surprise (+ dummies)

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--releases releases.csv  (required)
  Columns (suggested):
    date, time (HH:MM, local), tz (IANA, e.g., Europe/Brussels), country (DE/FR/IT/ES/EZ...),
    sector (Manufacturing/Services/Composite), stage (Flash/Final), actual, consensus, previous (optional),
    source (S&P Global/Markit/…)
  Notes:
    • If time or tz is missing, default time=09:00 and tz=Europe/Brussels.
    • country 'EZ' or 'EA' denotes Eurozone-wide PMI (if present). Aggregation can be computed from country releases.

--weights weights.csv   (optional; for Eurozone aggregation when EZ headline not provided)
  Columns:
    country, weight  (should sum ~1 across included countries; script will normalize if not)

--markets markets.csv   (optional; for event study)
  Columns (two formats accepted):
    A) Intraday:  timestamp (UTC ISO), asset, price   [or return]
    B) Daily:     date (YYYY-MM-DD), asset, close     [or return]
  If price/close provided, returns are computed as log-diffs. Units assumed in natural logs.

Key options
-----------
--asof 2025-09-06
--rolling_n 36               Rolling window length (# of releases) for z-scores
--event_win_mins 60          Intraday event window: [0, +N] minutes after release
--event_horizon_pre 30       Intraday pre-window minutes used for baseline return (subtract if >0)
--event_win_days 1           Daily event window: [0, +N] trading days after release
--only_flash 0               If 1, restrict analytics to Flash stage
--outdir out_pmi

Outputs
-------
- releases_enriched.csv      One row per release with surprises, z-scores, UTC timestamp
- ez_aggregate.csv           Eurozone-aggregated surprises from country releases (if weights provided)
- event_intraday.csv         Intraday event-window returns per asset & release (if intraday data present)
- event_daily.csv            Daily event-window returns per asset & release (if daily data present)
- regressions.csv            OLS: asset return ~ standardized surprise (+ sector/stage dummies)
- summary.json               Headline KPIs (latest releases, top surprises)
- config.json                Reproducibility dump

Notes
-----
- No external data is fetched. Provide your own CSVs.
- Event study is simplistic (no risk adjustment). Use with care.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ----------------------------- helpers -----------------------------
def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def to_timestamp(dt: pd.Series, tm: pd.Series, tz: pd.Series) -> pd.Series:
    """
    Combine date, time, and timezone to UTC-aware pandas Timestamps.
    """
    d = pd.to_datetime(dt, errors="coerce")
    # Parse times; default 09:00 if missing
    t = pd.to_datetime(tm.fillna("09:00"), format="%H:%M", errors="coerce").dt.time
    base = pd.to_datetime(d.dt.strftime("%Y-%m-%d") + " " + pd.Series([x.strftime("%H:%M") if pd.notna(x) else "09:00" for x in t]), utc=False, errors="coerce")
    # Attach timezone per row (vectorized best-effort)
    out = []
    for b, zone in zip(base, tz.fillna("Europe/Brussels")):
        try:
            out.append(b.tz_localize(str(zone)).tz_convert("UTC"))
        except Exception:
            try:
                out.append(b.tz_localize("Europe/Brussels").tz_convert("UTC"))
            except Exception:
                out.append(pd.NaT)
    return pd.Series(out)

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def standardize_groupwise(df: pd.DataFrame, value_col: str, by_cols: List[str], rolling_n: int) -> pd.Series:
    """
    Rolling z-score by group: (x - mean_lastN) / std_lastN, excluding current obs.
    """
    def roll_z(s: pd.Series) -> pd.Series:
        mu = s.shift(1).rolling(rolling_n, min_periods=max(5, rolling_n//3)).mean()
        sd = s.shift(1).rolling(rolling_n, min_periods=max(5, rolling_n//3)).std(ddof=1)
        return (s - mu) / (sd + 1e-12)
    return df.sort_values("release_dt_utc").groupby(by_cols, group_keys=False)[value_col].apply(roll_z)

def sanitize_country(c: str) -> str:
    if c is None: return ""
    x = str(c).strip().upper()
    return {"EA":"EZ","EUROZONE":"EZ","EURO AREA":"EZ","EMU":"EZ"}.get(x, x)

def stage_norm(s: str) -> str:
    if s is None: return ""
    x = str(s).strip().title()
    return "Flash" if "Flash" in x else ("Final" if "Final" in x else x)

def sector_norm(s: str) -> str:
    if s is None: return ""
    x = str(s).strip().title()
    if x.startswith("Manu"): return "Manufacturing"
    if x.startswith("Serv"): return "Services"
    if x.startswith("Comp"): return "Composite"
    return x

# ----------------------------- loaders -----------------------------
def load_releases(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"time") or "time"): "time",
        (ncol(df,"tz") or "tz"): "tz",
        (ncol(df,"country") or ncol(df,"area") or "country"): "country",
        (ncol(df,"sector") or "sector"): "sector",
        (ncol(df,"stage") or "stage"): "stage",
        (ncol(df,"actual") or "actual"): "actual",
        (ncol(df,"consensus") or ncol(df,"survey") or "consensus"): "consensus",
        (ncol(df,"previous") or "previous"): "previous",
        (ncol(df,"source") or "source"): "source",
    }
    df = df.rename(columns=ren)
    df["country"] = df["country"].apply(sanitize_country)
    df["sector"] = df["sector"].apply(sector_norm)
    df["stage"] = df["stage"].apply(stage_norm)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "time" not in df or df["time"].isna().all():
        df["time"] = "09:00"
    if "tz" not in df:
        df["tz"] = "Europe/Brussels"
    df["release_dt_utc"] = to_timestamp(df["date"], df["time"], df["tz"])
    for c in ["actual","consensus","previous"]:
        if c in df.columns: df[c] = num(df[c])
    # Drop rows without actual or consensus
    df = df.dropna(subset=["actual","consensus"])
    # Ensure basic columns
    if "sector" not in df.columns: df["sector"] = "Composite"
    if "stage" not in df.columns: df["stage"] = "Flash"
    return df

def load_weights(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={(ncol(df,"country") or "country"):"country", (ncol(df,"weight") or "weight"):"weight"})
    df["country"] = df["country"].apply(sanitize_country)
    df["weight"] = num(df["weight"])
    s = df["weight"].sum()
    if s and np.isfinite(s) and s > 0:
        df["weight_norm"] = df["weight"] / s
    else:
        df["weight_norm"] = np.nan
    return df[["country","weight_norm"]].rename(columns={"weight_norm":"weight"})

def load_markets(path: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (intraday_df, daily_df)
    """
    if not path:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(path)
    # Try infer intraday vs daily by presence of 'timestamp' vs 'date'
    ts_col = ncol(df, "timestamp")
    date_col = ncol(df, "date")
    asset_col = ncol(df, "asset") or "asset"
    price_col = ncol(df, "price") or ncol(df, "close") or "price"
    ret_col = ncol(df, "return") or ncol(df, "ret") or "return"
    if ts_col:
        intra = df.rename(columns={ts_col:"timestamp", asset_col:"asset", price_col:"price", ret_col:"return"})
        intra["timestamp"] = pd.to_datetime(intra["timestamp"], utc=True, errors="coerce")
        if "return" not in intra or intra["return"].isna().all():
            intra = intra.sort_values(["asset","timestamp"])
            intra["return"] = intra.groupby("asset")["price"].apply(lambda s: np.log(s) - np.log(s.shift(1)))
        intra = intra.dropna(subset=["timestamp","asset","return"])
    else:
        intra = pd.DataFrame()
    if date_col:
        daily = df.rename(columns={date_col:"date", asset_col:"asset", price_col:"close", ret_col:"return"})
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.date
        if "return" not in daily or daily["return"].isna().all():
            daily = daily.sort_values(["asset","date"])
            daily["return"] = daily.groupby("asset")["close"].apply(lambda s: np.log(s) - np.log(s.shift(1)))
        daily = daily.dropna(subset=["date","asset","return"])
    else:
        daily = pd.DataFrame()
    return intra, daily

# ----------------------------- core logic -----------------------------
def compute_surprises(releases: pd.DataFrame, rolling_n: int, only_flash: bool) -> pd.DataFrame:
    df = releases.copy()
    if only_flash:
        df = df[df["stage"] == "Flash"]
    df["surprise"] = df["actual"] - df["consensus"]
    # Grouping for standardization
    by = ["country","sector","stage"]
    df = df.sort_values(["country","sector","stage","release_dt_utc"])
    df["surprise_z"] = standardize_groupwise(df, "surprise", by, rolling_n)
    # Directional helpers
    df["above_consensus"] = (df["surprise"] > 0).astype(int)
    df["abs_z"] = df["surprise_z"].abs()
    return df

def aggregate_ez_from_countries(enriched: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate country releases into Eurozone (EZ) by weighted average surprise on the SAME date/sector/stage.
    If an EZ headline exists for that timestamp, it will also be included for comparison.
    """
    if weights.empty:
        return pd.DataFrame()
    w = weights.set_index("country")["weight"].to_dict()
    # country rows only (exclude existing EZ to avoid double counting)
    ctr = enriched[enriched["country"] != "EZ"]
    # For each (date, sector, stage), compute weighted avg surprise and z
    grp = ctr.groupby(["date","sector","stage"])
    rows = []
    for (d, sec, stg), g in grp:
        ww = np.array([w.get(c, np.nan) for c in g["country"]])
        if np.isnan(ww).all():
            continue
        ww = ww / np.nansum(ww)
        spr = np.nansum(ww * g["surprise"].values)
        # For z, a simple approach: weighted average of z-scores (not perfect, but indicative)
        z = np.nansum(ww * g["surprise_z"].fillna(0).values)
        # Take earliest release time among constituents as aggregate timestamp
        ts = g["release_dt_utc"].min()
        rows.append({"date": d, "release_dt_utc": ts, "country": "EZ_agg", "sector": sec, "stage": stg,
                     "surprise": spr, "surprise_z": z, "n_countries": int(len(g))})
    ez = pd.DataFrame(rows)
    # Add observed EZ headline (if present) for compare
    ez_headline = enriched[enriched["country"] == "EZ"][["date","release_dt_utc","sector","stage","surprise","surprise_z"]].copy()
    ez_headline = ez_headline.rename(columns={"surprise":"surprise_ez_headline", "surprise_z":"surprise_z_ez_headline"})
    out = ez.merge(ez_headline, on=["date","sector","stage"], how="left", suffixes=("",""))
    return out.sort_values(["date","sector","stage"])

def event_study_intraday(enriched: pd.DataFrame, intraday: pd.DataFrame,
                         win_post_min: int, win_pre_min: int) -> pd.DataFrame:
    """
    Sum intraday returns from t=0 (release) to +win_post_min.
    If win_pre_min>0, subtract pre-window [-win_pre_min,0) to baseline-adjust.
    """
    if intraday.empty:
        return pd.DataFrame()
    rows = []
    intra = intraday.copy()
    intra = intra.dropna(subset=["timestamp","asset","return"])
    # index for faster slicing
    intra = intra.sort_values(["asset","timestamp"]).set_index("timestamp")
    for rid, r in enriched.dropna(subset=["release_dt_utc"]).iterrows():
        t0 = r["release_dt_utc"]
        t1 = t0 + pd.Timedelta(minutes=int(win_post_min))
        pre0 = t0 - pd.Timedelta(minutes=int(win_pre_min)) if win_pre_min and win_pre_min>0 else None
        for asset, g in intra.groupby("asset"):
            # post window
            post_sum = float(g.loc[t0:t1, "return"].sum()) if t0 in g.index or (t0 < g.index.max()) else np.nan
            # pre window
            pre_sum = float(g.loc[pre0:t0, "return"].sum()) if pre0 is not None else 0.0
            if not np.isfinite(post_sum):
                continue
            adj = post_sum - (pre_sum if np.isfinite(pre_sum) else 0.0)
            rows.append({
                "release_dt_utc": t0, "date": r["date"], "country": r["country"], "sector": r["sector"], "stage": r["stage"],
                "surprise": r["surprise"], "surprise_z": r["surprise_z"],
                "asset": asset, "ret_intraday_adj": adj, "win_post_min": int(win_post_min), "win_pre_min": int(win_pre_min or 0)
            })
    return pd.DataFrame(rows)

def event_study_daily(enriched: pd.DataFrame, daily: pd.DataFrame, win_days: int) -> pd.DataFrame:
    """
    Sum daily returns from release date (calendar date) to +win_days.
    """
    if daily.empty:
        return pd.DataFrame()
    rows = []
    dly = daily.copy()
    dly = dly.dropna(subset=["date","asset","return"])
    # map release date to plain date (UTC → date)
    rel = enriched.dropna(subset=["date"]).copy()
    for rid, r in rel.iterrows():
        d0 = pd.to_datetime(r["date"]).date()
        for asset, g in dly.groupby("asset"):
            g2 = g[g["date"] >= d0].sort_values("date").head(win_days+1)
            if g2.empty: 
                continue
            rs = float(g2["return"].sum())
            rows.append({
                "date": d0, "country": r["country"], "sector": r["sector"], "stage": r["stage"],
                "surprise": r["surprise"], "surprise_z": r["surprise_z"],
                "asset": asset, "ret_daily": rs, "win_days": int(win_days)
            })
    return pd.DataFrame(rows)

def regress_impacts(events_df: pd.DataFrame, dep_col: str) -> pd.DataFrame:
    """
    OLS per-asset: dep_col ~ 1 + surprise_z + sector dummies + stage dummy (Flash)
    Returns one row per asset with coefficients and R^2.
    """
    if events_df.empty:
        return pd.DataFrame()
    out = []
    # build dummies
    d = events_df.copy()
    d["intercept"] = 1.0
    d["is_flash"] = (d["stage"] == "Flash").astype(int)
    # sector dummies
    for sec in ["Manufacturing","Services","Composite"]:
        d[f"sec_{sec}"] = (d["sector"] == sec).astype(int)
    assets = sorted(d["asset"].unique())
    Xcols = ["intercept","surprise_z","is_flash","sec_Manufacturing","sec_Services","sec_Composite"]
    for a in assets:
        g = d[d["asset"] == a].dropna(subset=[dep_col, "surprise_z"])
        if len(g) < 25:
            continue
        X = g[Xcols].values
        Y = g[dep_col].values
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            yhat = X @ beta
            r2 = 1.0 - np.sum((Y - yhat)**2) / max(1e-12, np.sum((Y - Y.mean())**2))
            out.append({
                "asset": a, "n": int(len(g)), "r2": float(r2),
                "beta_surprise_z": float(beta[1]),
                "beta_is_flash": float(beta[2]),
                "beta_sec_manu": float(beta[3]),
                "beta_sec_serv": float(beta[4]),
                "beta_sec_comp": float(beta[5]),
            })
        except Exception:
            continue
    return pd.DataFrame(out).sort_values("r2", ascending=False)

# ----------------------------- CLI -----------------------------
@dataclass
class Config:
    releases: str
    weights: Optional[str]
    markets: Optional[str]
    asof: str
    rolling_n: int
    event_win_mins: int
    event_horizon_pre: int
    event_win_days: int
    only_flash: bool
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Eurozone PMI surprise analytics")
    ap.add_argument("--releases", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--markets", default="")
    ap.add_argument("--asof", default="")
    ap.add_argument("--rolling_n", type=int, default=36)
    ap.add_argument("--event_win_mins", type=int, default=60)
    ap.add_argument("--event_horizon_pre", type=int, default=30)
    ap.add_argument("--event_win_days", type=int, default=1)
    ap.add_argument("--only_flash", type=int, default=0)
    ap.add_argument("--outdir", default="out_pmi")
    return ap.parse_args()

# ----------------------------- main -----------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    asof = pd.to_datetime(args.asof).date() if args.asof else None

    releases = load_releases(args.releases)
    if asof:
        releases = releases[releases["date"].dt.date <= asof]

    weights = load_weights(args.weights) if args.weights else pd.DataFrame()
    intra, daily = load_markets(args.markets) if args.markets else (pd.DataFrame(), pd.DataFrame())

    # Surprises & z
    enriched = compute_surprises(releases, args.rolling_n, bool(args.only_flash))
    enriched = enriched.sort_values(["release_dt_utc","country","sector","stage"])
    enriched.to_csv(outdir / "releases_enriched.csv", index=False)

    # EZ aggregation
    ez_agg = aggregate_ez_from_countries(enriched, weights) if not weights.empty else pd.DataFrame()
    if not ez_agg.empty:
        ez_agg.to_csv(outdir / "ez_aggregate.csv", index=False)

    # Event studies
    ev_intra = event_study_intraday(enriched, intra, args.event_win_mins, args.event_horizon_pre) if not intra.empty else pd.DataFrame()
    if not ev_intra.empty:
        ev_intra.to_csv(outdir / "event_intraday.csv", index=False)

    ev_daily = event_study_daily(enriched, daily, args.event_win_days) if not daily.empty else pd.DataFrame()
    if not ev_daily.empty:
        ev_daily.to_csv(outdir / "event_daily.csv", index=False)

    # Regressions
    reg_intra = regress_impacts(ev_intra, "ret_intraday_adj") if not ev_intra.empty else pd.DataFrame()
    reg_daily = regress_impacts(ev_daily, "ret_daily") if not ev_daily.empty else pd.DataFrame()
    regs = pd.concat([reg_intra.assign(freq="intraday"), reg_daily.assign(freq="daily")], ignore_index=True) if (not reg_intra.empty or not reg_daily.empty) else pd.DataFrame()
    if not regs.empty:
        regs.to_csv(outdir / "regressions.csv", index=False)

    # KPIs / summary
    latest_date = enriched["date"].max().date() if not enriched.empty else None
    top_recent = (enriched[enriched["date"].dt.date == latest_date]
                  .sort_values("abs_z", ascending=False)
                  .head(10)
                  [["country","sector","stage","surprise","surprise_z","release_dt_utc"]])
    kpi = {
        "latest_date": str(latest_date) if latest_date else None,
        "releases_total": int(len(enriched)),
        "countries": sorted(enriched["country"].unique().tolist()),
        "top_recent": top_recent.to_dict(orient="records") if not top_recent.empty else [],
        "has_ez_aggregate": bool(not ez_agg.empty),
        "event_intraday_rows": int(len(ev_intra)) if not ev_intra.empty else 0,
        "event_daily_rows": int(len(ev_daily)) if not ev_daily.empty else 0,
        "regressions": regs.to_dict(orient="records")[:10] if not regs.empty else [],
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        releases=args.releases, weights=args.weights or None, markets=args.markets or None,
        asof=args.asof or None, rolling_n=args.rolling_n, event_win_mins=args.event_win_mins,
        event_horizon_pre=args.event_horizon_pre, event_win_days=args.event_win_days,
        only_flash=bool(args.only_flash), outdir=args.outdir
    )), indent=2))

    # Console
    print("== Eurozone PMI Surprise ==")
    print(f"Releases: {kpi['releases_total']} | Latest date: {kpi['latest_date']}")
    if kpi["top_recent"]:
        print("Top surprises (latest date):")
        for r in kpi["top_recent"][:5]:
            z = r.get("surprise_z")
            print(f"  {r['country']} {r['sector']} {r['stage']}: surprise {r['surprise']:+.2f}, z {z:+.2f} at {r['release_dt_utc']}")
    if regs is not None and not regs.empty:
        best = regs.sort_values("r2", ascending=False).head(3)
        for _, row in best.iterrows():
            print(f"Impact {row['freq']} {row['asset']}: beta_surprise_z={row['beta_surprise_z']:+.4f}, R^2={row['r2']:.3f}")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
