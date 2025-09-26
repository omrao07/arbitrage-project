#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
french_pension_protests.py — Mobilization intensity, disruption analytics, and market/event-study toolkit
for the French pension reform protest cycles.

What this does
--------------
Given event logs (marches, blockades, clashes), strike participation, transport disruptions, sentiment,
and market prices, this script builds a daily timeline of “intensity”, detects national mobilization
days, summarizes by region/sector, and runs a simple event study on assets.

Core outputs
- events_clean.csv            Harmonized event log with parsed headcounts & flags
- timeline_daily.csv          Daily aggregates: participants, violence, disruption, INTENSITY index, R7 momentum
- regions.csv                 Per-region cumulative metrics and latest intensity
- sectors.csv                 Strike participation by sector (daily and cumulative)
- disruptions.csv             Transport disruption aggregates by mode/day
- mobilization_days.csv       Detected national mobilization days (threshold-based)
- event_study.csv             Asset returns in windows around mobilization days
- sentiment_joined.csv        Sentiment merged with intensity (if provided)
- summary.json                KPIs (latest day, peak intensity, counts)
- config.json                 Reproducibility dump

Assumptions / notes
- Bring your own CSVs; headers are case-insensitive and flexible.
- Participation numbers can be in plain numbers or strings like "80k", "1.2m".
- INTENSITY index is a winsorized min-max blend of four pillars:
  participation (50%), transport disruption (20%), violence/arrests (20%), strike breadth (10%).
- NATIONAL MOBILIZATION is flagged when (participants >= pct_of_peak * threshold) AND (≥ min_sectors active).

Inputs (CSV; flexible headers)
------------------------------
--events events.csv          REQUIRED
    Columns (suggested):
      date, time(optional), city, departement, region,
      crowd (or crowd_estimate / participants / turnout),
      violence(0/1), arrests, injuries, police_count, organizer, type, lat, lon, source

--strikes strikes.csv        OPTIONAL
    Columns:
      date, sector (e.g., Transport/Education/Energy/Refinery/Public), participation_rate or participants, sites_affected, note

--transport transport.csv    OPTIONAL
    Columns:
      date, mode (Rail/Metro/Road/Air/Port), cancellations, delays, pct_running (0..1 or 0..100), lines_closed

--sentiment sentiment.csv    OPTIONAL
    Columns:
      date, sentiment_score (-1..1), volume

--prices prices.csv          OPTIONAL (for event study)
    Columns:
      date (YYYY-MM-DD), asset, close  (or price / return)
    If price/close provided, log returns are computed.

Key options
-----------
--start 2023-01-01
--end   2025-12-31
--intensity_winsor 0.05                  Winsorization tails for normalization (e.g., 0.05=5%)
--mobilization_peak_pct 0.40             Participants threshold as % of sample peak for national day
--mobilization_min_sectors 3             Minimum distinct sectors with strike participation on that day
--event_win 3                            Event-study window (days) on each side of t=0 (national day)
--outdir out_fr_protests

Example
-------
python french_pension_protests.py --events events.csv --strikes strikes.csv --transport transport.csv \
  --sentiment sentiment.csv --prices prices.csv --start 2023-01-01 --end 2025-12-31 --outdir out_fr_protests
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
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.floor("D")

def num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def parse_human_number(x) -> float:
    """
    Parse '80k', '1.2m', '35 000', '35,000', '~50k' into float.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float)) and np.isfinite(x):
        return float(x)
    s = str(x).strip().lower()
    s = s.replace("~", "").replace("≈", "").replace("about", "").replace(" ", "")
    s = s.replace(",", "")
    try:
        mult = 1.0
        if s.endswith("k"):
            mult = 1e3; s = s[:-1]
        elif s.endswith("m"):
            mult = 1e6; s = s[:-1]
        return float(s) * mult
    except Exception:
        # last resort: extract digits
        import re
        digs = re.findall(r"[\d\.]+", s)
        if not digs:
            return np.nan
        try:
            return float(digs[0])
        except Exception:
            return np.nan

def pct01(x: pd.Series) -> pd.Series:
    """Normalize 0..1 if looks like percentages >1."""
    x = num_series(x)
    if x.dropna().max() is not None and x.dropna().max() > 1.5:
        return x / 100.0
    return x

def winsor_minmax(s: pd.Series, a: float = 0.05) -> pd.Series:
    """Winsorize and scale to 0..1."""
    x = s.astype(float).replace([np.inf, -np.inf], np.nan)
    lo = x.quantile(a) if len(x.dropna()) else np.nan
    hi = x.quantile(1 - a) if len(x.dropna()) else np.nan
    x = x.clip(lower=lo, upper=hi)
    den = (hi - lo) if (hi is not None and lo is not None and np.isfinite(hi) and np.isfinite(lo) and hi > lo) else np.nan
    return (x - lo) / den if np.isfinite(den) else x * np.nan

def safe_mean(x: pd.Series) -> float:
    return float(x.dropna().mean()) if x is not None and len(x.dropna()) else np.nan

def mode_upper(s: pd.Series) -> str:
    if s.empty: return ""
    v = s.dropna().astype(str).str.upper()
    if v.empty: return ""
    return v.mode().iloc[0] if not v.mode().empty else ""

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ----------------------------- loaders -----------------------------

def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"time") or "time"): "time",
        (ncol(df,"city") or "city"): "city",
        (ncol(df,"departement") or ncol(df,"department") or ncol(df,"dept") or "departement"): "departement",
        (ncol(df,"region") or "region"): "region",
        (ncol(df,"crowd") or ncol(df,"participants") or ncol(df,"turnout") or ncol(df,"crowd_estimate") or "crowd"): "crowd",
        (ncol(df,"violence") or ncol(df,"clashes") or "violence"): "violence",
        (ncol(df,"arrests") or "arrests"): "arrests",
        (ncol(df,"injuries") or "injuries"): "injuries",
        (ncol(df,"police_count") or ncol(df,"police") or "police_count"): "police_count",
        (ncol(df,"organizer") or "organizer"): "organizer",
        (ncol(df,"type") or "type"): "type",
        (ncol(df,"lat") or "lat"): "lat",
        (ncol(df,"lon") or ncol(df,"lng") or "lon"): "lon",
        (ncol(df,"source") or "source"): "source",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    # parse crowd numbers; allow "low" and "high" alternative columns if present
    alt_low = ncol(df, "crowd_low") or ncol(df, "participants_low")
    alt_high = ncol(df, "crowd_high") or ncol(df, "participants_high")
    if alt_low or alt_high:
        low = df[alt_low].apply(parse_human_number) if alt_low in df.columns else np.nan
        high = df[alt_high].apply(parse_human_number) if alt_high in df.columns else np.nan
        mid = pd.DataFrame({"low": low, "high": high}).mean(axis=1)
        df["crowd"] = df["crowd"].where(df["crowd"].notna(), mid)
    df["crowd"] = df["crowd"].apply(parse_human_number)
    for c in ["violence", "arrests", "injuries", "police_count", "lat", "lon"]:
        if c in df.columns:
            df[c] = num_series(df[c])
    # violence to 0/1 if categorical
    if "violence" in df.columns:
        v = df["violence"]
        if v.dropna().max() and v.dropna().max() > 1:
            df["violence"] = (v > 0).astype(int)
    return df

def load_strikes(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"sector") or "sector"): "sector",
        (ncol(df,"participation_rate") or ncol(df,"participation") or "participation_rate"): "participation_rate",
        (ncol(df,"participants") or ncol(df,"headcount") or "participants"): "participants",
        (ncol(df,"sites_affected") or "sites_affected"): "sites_affected",
        (ncol(df,"note") or "note"): "note",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    if "participation_rate" in df.columns:
        df["participation_rate"] = pct01(df["participation_rate"]).clip(0, 1)
    if "participants" in df.columns:
        df["participants"] = df["participants"].apply(parse_human_number)
    return df

def load_transport(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"mode") or "mode"): "mode",
        (ncol(df,"cancellations") or "cancellations"): "cancellations",
        (ncol(df,"delays") or "delays"): "delays",
        (ncol(df,"pct_running") or ncol(df,"percent_running") or "pct_running"): "pct_running",
        (ncol(df,"lines_closed") or "lines_closed"): "lines_closed",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    for c in ["cancellations","delays","lines_closed"]:
        if c in df.columns: df[c] = num_series(df[c])
    if "pct_running" in df.columns:
        df["pct_running"] = pct01(df["pct_running"]).clip(0, 1)
    return df

def load_sentiment(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"sentiment_score") or ncol(df,"sentiment") or "sentiment_score"): "sentiment_score",
        (ncol(df,"volume") or "volume"): "volume",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    for c in ["sentiment_score","volume"]:
        if c in df.columns: df[c] = num_series(df[c])
    return df

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"asset") or "asset"): "asset",
        (ncol(df,"close") or ncol(df,"price") or "close"): "close",
        (ncol(df,"return") or ncol(df,"ret") or "return"): "return",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    if "return" not in df.columns or df["return"].isna().all():
        df = df.sort_values(["asset","date"])
        df["return"] = df.groupby("asset")["close"].apply(lambda s: np.log(s) - np.log(s.shift(1)))
    df = df.dropna(subset=["date","asset","return"])
    return df

# ----------------------------- analytics -----------------------------

def aggregate_transport(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame(columns=["date","cancel_sum","delay_sum","lines_closed_sum","pct_running_avg","disruption_score"])
    g = (tr.groupby("date", as_index=False)
           .agg(cancel_sum=("cancellations","sum"),
                delay_sum=("delays","sum"),
                lines_closed_sum=("lines_closed","sum"),
                pct_running_avg=("pct_running","mean")))
    # Disruption score: more cancellations/delays & lower pct_running → higher score
    g["disruption_score_raw"] = (g["cancel_sum"].fillna(0) + g["delay_sum"].fillna(0) + g["lines_closed_sum"].fillna(0)) \
                                + (1.0 - g["pct_running_avg"].fillna(1.0)) * 100.0
    return g

def aggregate_strikes(st: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if st.empty:
        return pd.DataFrame(columns=["date","sectors","participation_avg","participants_sum"]), pd.DataFrame()
    # Daily summary
    daily = (st.groupby("date", as_index=False)
               .agg(sectors=("sector","nunique"),
                    participation_avg=("participation_rate","mean"),
                    participants_sum=("participants","sum")))
    # Sector cumulation
    by_sector = (st.groupby(["date","sector"], as_index=False)
                   .agg(participation=("participation_rate","mean"),
                        participants=("participants","sum"),
                        sites_affected=("sites_affected","sum")))
    return daily, by_sector

def aggregate_events(ev: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ev.empty:
        return pd.DataFrame(), pd.DataFrame()
    ev["violence_flag"] = ev.get("violence", 0).fillna(0).astype(int)
    daily = (ev.groupby("date", as_index=False)
               .agg(events=("city","count"),
                    participants=("crowd","sum"),
                    violence_incidents=("violence_flag","sum"),
                    arrests=("arrests","sum"),
                    injuries=("injuries","sum"),
                    police=("police_count","sum")))
    by_region = (ev.groupby("region", as_index=False)
                   .agg(events=("city","count"),
                        participants=("crowd","sum"),
                        violence_incidents=("violence_flag","sum"),
                        arrests=("arrests","sum")))
    return daily, by_region

def build_intensity(daily_ev: pd.DataFrame,
                    daily_tr: pd.DataFrame,
                    daily_st: pd.DataFrame,
                    winsor_a: float) -> pd.DataFrame:
    # Merge bases
    df = pd.DataFrame()
    cols = []
    if not daily_ev.empty:
        df = daily_ev.copy()
        cols = ["date"]
    else:
        return pd.DataFrame()
    if not daily_tr.empty:
        df = df.merge(daily_tr[["date","disruption_score_raw"]], on="date", how="left")
    if not daily_st.empty:
        df = df.merge(daily_st[["date","sectors","participation_avg","participants_sum"]], on="date", how="left")
    # Normalizations
    df["part_norm"] = winsor_minmax(df["participants"].fillna(0), winsor_a=winsor_a if (winsor_a:=winsor_a) else winsor_a)
    df["disr_norm"] = winsor_minmax(df.get("disruption_score_raw", pd.Series(np.nan, index=df.index)).fillna(0), winsor_a)
    # Violence/arrests (blend)
    viol_raw = df.get("violence_incidents", 0).fillna(0) + 0.5 * df.get("arrests", 0).fillna(0) + 0.25 * df.get("injuries", 0).fillna(0)
    df["viol_norm"] = winsor_minmax(viol_raw, winsor_a)
    # Strike breadth
    breadth_raw = df.get("sectors", 0).fillna(0) + 3.0 * df.get("participation_avg", 0).fillna(0)
    df["breadth_norm"] = winsor_minmax(breadth_raw, winsor_a)
    # INTENSITY (weighted)
    df["intensity"] = (0.50 * df["part_norm"] + 0.20 * df["disr_norm"] + 0.20 * df["viol_norm"] + 0.10 * df["breadth_norm"]).clip(0, 1)
    # 7-day momentum (R7-style): ratio of 7D sum today vs a week ago
    df = df.sort_values("date")
    s7 = df["intensity"].rolling(7, min_periods=3).sum()
    df["R7"] = s7 / s7.shift(7)
    return df

def detect_mobilization_days(df_int: pd.DataFrame, daily_st: pd.DataFrame,
                             peak_pct: float, min_sectors: int) -> pd.DataFrame:
    if df_int.empty:
        return pd.DataFrame()
    peak = float(df_int["participants"].max() or 0.0)
    thr = peak * float(peak_pct)
    # merge strike sectors
    base = df_int.merge(daily_st[["date","sectors"]], on="date", how="left") if not daily_st.empty else df_int.copy()
    base["is_mobilization"] = ((base["participants"] >= thr) & (base.get("sectors", 0).fillna(0) >= int(min_sectors))).astype(int)
    out = base[base["is_mobilization"] == 1][["date","participants","events","intensity","R7","sectors"]].copy()
    return out.sort_values("date")

def join_sentiment(intensity: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    if intensity.empty or sent.empty:
        return pd.DataFrame()
    j = intensity.merge(sent, on="date", how="left")
    return j

def event_study(prices: pd.DataFrame, events: pd.DataFrame, win: int) -> pd.DataFrame:
    if prices.empty or events.empty:
        return pd.DataFrame()
    evdays = set(pd.to_datetime(events["date"]).dt.date.tolist())
    rows = []
    for asset, g in prices.groupby("asset"):
        g = g.sort_values("date").reset_index(drop=True)
        g["date_only"] = g["date"].dt.date
        idx_map = {d: i for i, d in enumerate(g["date_only"])}
        for d in evdays:
            if d not in idx_map:
                continue
            i0 = idx_map[d]
            lo = max(0, i0 - win)
            hi = min(len(g) - 1, i0 + win)
            window = g.iloc[lo:hi+1].copy()
            window["t"] = range(lo - i0, hi - i0 + 1)
            window["is_event"] = (window["date_only"] == d).astype(int)
            # cum return over window & markers
            rows.extend([{
                "asset": asset,
                "event_date": str(d),
                "t": int(row["t"]),
                "return": float(row["return"])
            } for _, row in window.iterrows()])
    df = pd.DataFrame(rows)
    if df.empty: return df
    # Average across events per asset per t
    avg = (df.groupby(["asset","t"], as_index=False)
             .agg(mean_ret=("return","mean"),
                  med_ret=("return","median"),
                  count=("return","count")))
    return avg.sort_values(["asset","t"])

# ----------------------------- CLI / orchestration -----------------------------

@dataclass
class Config:
    events: str
    strikes: Optional[str]
    transport: Optional[str]
    sentiment: Optional[str]
    prices: Optional[str]
    start: str
    end: str
    intensity_winsor: float
    mobilization_peak_pct: float
    mobilization_min_sectors: int
    event_win: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="French pension protest analytics")
    ap.add_argument("--events", required=True)
    ap.add_argument("--strikes", default="")
    ap.add_argument("--transport", default="")
    ap.add_argument("--sentiment", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--intensity_winsor", type=float, default=0.05)
    ap.add_argument("--mobilization_peak_pct", type=float, default=0.40)
    ap.add_argument("--mobilization_min_sectors", type=int, default=3)
    ap.add_argument("--event_win", type=int, default=3)
    ap.add_argument("--outdir", default="out_fr_protests")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    events = load_events(args.events)
    strikes = load_strikes(args.strikes) if args.strikes else pd.DataFrame()
    transport = load_transport(args.transport) if args.transport else pd.DataFrame()
    sentiment = load_sentiment(args.sentiment) if args.sentiment else pd.DataFrame()
    prices = load_prices(args.prices) if args.prices else pd.DataFrame()

    # Filter period
    start = pd.to_datetime(args.start); end = pd.to_datetime(args.end)
    if not events.empty:
        events = events[(events["date"] >= start) & (events["date"] <= end)].copy()
    if not strikes.empty:
        strikes = strikes[(strikes["date"] >= start) & (strikes["date"] <= end)].copy()
    if not transport.empty:
        transport = transport[(transport["date"] >= start) & (transport["date"] <= end)].copy()
    if not sentiment.empty:
        sentiment = sentiment[(sentiment["date"] >= start) & (sentiment["date"] <= end)].copy()
    if not prices.empty:
        prices = prices[(prices["date"] >= start - pd.Timedelta(days=args.event_win+5)) & (prices["date"] <= end + pd.Timedelta(days=args.event_win+5))].copy()

    # Clean outputs
    ensure_cols(events, ["violence","arrests","injuries","police_count","departement","region"])
    events_clean = events.sort_values(["date","region","city"])
    events_clean.to_csv(outdir / "events_clean.csv", index=False)

    # Aggregations
    daily_ev, by_region = aggregate_events(events)
    daily_tr = aggregate_transport(transport)
    daily_st, st_sector = aggregate_strikes(strikes)

    # Intensity
    intensity = build_intensity(daily_ev, daily_tr, daily_st, winsor_a=args.intensity_winsor)
    if not intensity.empty:
        intensity.to_csv(outdir / "timeline_daily.csv", index=False)

    # Regions & sectors
    if not by_region.empty:
        by_region = by_region.sort_values("participants", ascending=False)
        by_region.to_csv(outdir / "regions.csv", index=False)
    if not st_sector.empty:
        st_sector.to_csv(outdir / "sectors.csv", index=False)

    # Disruptions detail
    if not daily_tr.empty:
        daily_tr.rename(columns={"disruption_score_raw":"disruption_score"}, inplace=True)
        daily_tr.to_csv(outdir / "disruptions.csv", index=False)

    # Mobilization detection
    mobil = detect_mobilization_days(intensity, daily_st, args.mobilization_peak_pct, args.mobilization_min_sectors) if not intensity.empty else pd.DataFrame()
    if not mobil.empty:
        mobil.to_csv(outdir / "mobilization_days.csv", index=False)

    # Sentiment join
    sent_join = join_sentiment(intensity, sentiment) if (not intensity.empty and not sentiment.empty) else pd.DataFrame()
    if not sent_join.empty:
        sent_join.to_csv(outdir / "sentiment_joined.csv", index=False)

    # Event study
    evt = event_study(prices, mobil if not mobil.empty else intensity[intensity["intensity"] >= intensity["intensity"].quantile(0.9)][["date"]], args.event_win) \
          if not prices.empty and not intensity.empty else pd.DataFrame()
    if not evt.empty:
        evt.to_csv(outdir / "event_study.csv", index=False)

    # KPIs
    latest = intensity["date"].max().date() if not intensity.empty else None
    peak_day = intensity.sort_values("intensity", ascending=False).head(1) if not intensity.empty else pd.DataFrame()
    kpi = {
        "period": {"start": args.start, "end": args.end},
        "events_rows": int(len(events)),
        "strike_rows": int(len(strikes)),
        "transport_rows": int(len(transport)),
        "sentiment_rows": int(len(sentiment)),
        "assets": sorted(prices["asset"].unique().tolist()) if not prices.empty else [],
        "latest_day": str(latest) if latest else None,
        "peak_intensity_day": str(peak_day["date"].iloc[0].date()) if not peak_day.empty else None,
        "peak_intensity_value": float(peak_day["intensity"].iloc[0]) if not peak_day.empty else None,
        "peak_participants": float(peak_day["participants"].iloc[0]) if not peak_day.empty else None,
        "mobilization_days_detected": int(len(mobil)) if not mobil.empty else 0
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))

    # Config dump
    cfg = asdict(Config(
        events=args.events, strikes=args.strikes or None, transport=args.transport or None,
        sentiment=args.sentiment or None, prices=args.prices or None,
        start=args.start, end=args.end, intensity_winsor=args.intensity_winsor,
        mobilization_peak_pct=args.mobilization_peak_pct, mobilization_min_sectors=args.mobilization_min_sectors,
        event_win=args.event_win, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== French Pension Protests Analytics ==")
    print(f"Window: {args.start} → {args.end} | rows(events/strikes/transport): {len(events)}/{len(strikes)}/{len(transport)}")
    if kpi["peak_intensity_day"]:
        print(f"Peak intensity on {kpi['peak_intensity_day']} | normalized {kpi['peak_intensity_value']:.3f} | participants≈{kpi['peak_participants']:.0f}")
    if not mobil.empty:
        sample = mobil.head(5)[["date","participants","intensity","sectors"]].copy()
        sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
        print("Mobilization days (sample):", sample.to_dict(orient="records")[:5])
    print("Outputs in:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
