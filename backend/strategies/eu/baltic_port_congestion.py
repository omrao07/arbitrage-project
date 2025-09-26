#!/usr/bin/env python3
"""
baltic_port_congestion.py — Compute Baltic Sea port congestion metrics from port-call / AIS data

What it does
------------
- Ingests port-call event logs (and optionally AIS positions) and computes:
  * Arrivals, berths, departures per day
  * Queue length (ships arrived, not yet berthed)
  * Waiting time (arrival→berth), dwell time (berth→depart)
  * Anchorage headcount snapshot (from events or AIS)
  * Backlog (ships berthed, not yet departed)
  * Congestion index vs baseline (z-score of waiting time / queue length)
- Focus on Baltic ports (filterable), but works for any ports if you pass metadata.

Inputs
------
--calls calls.csv              Port-call events (wide or long). Expected columns (case-insensitive):
                               - vessel_id (mmsi/imo/name ok), port, country, event, timestamp
                               - Optional: lat, lon, nav_status, area ("anchorage"/"berth"/"port")
                               Event values are flexible (e.g., "arrival", "arrive", "anchorage", "berth", "depart", "departure")

--ports ports.csv              Port metadata (optional, improves filtering):
                               - port, country, lat, lon, anch_radius_km, nominal_berth_cap (per day)

--positions positions.parquet  Optional AIS positions for anchorage headcount (columns: timestamp, vessel_id, lat, lon, sog, nav_status)

--start 2024-01-01             Filter window start (inclusive)
--end   2025-09-06             Filter window end (inclusive)
--baseline "2023-01-01:2023-12-31"  Period for baseline z-scores (arrivals & wait/dwell medians)

--baltic-only                   If set, only keep ports located in Baltic littoral countries (see list below)
--countries "SE,FI,DK,DE,PL,LT,LV,EE,RU"  Override country filter
--outdir out_baltic             Output folder

Outputs
-------
- port_daily.csv         Per-port daily KPIs
- congestion_index.csv   Per-port congestion indices vs baseline (z-scores)
- snapshot.csv           Latest-date snapshot across ports
- ships_waiting.csv      Vessels currently waiting (arrived, not yet berthed)
- ships_in_anchorage.csv If anchorage status derivable (from events/AIS)
- kpis.json              High-level summary and run configuration

Usage
-----
python baltic_port_congestion.py --calls calls.csv --ports ports.csv \
  --baseline "2023-01-01:2023-06-30" --baltic-only --outdir out_baltic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Constants / helpers
# -----------------------------
BALTIC_LITTORAL = {"SE","FI","DK","DE","PL","LT","LV","EE","RU"}  # Override via --countries as needed

EVENT_MAP = {
    # canonical -> fuzzy aliases (lowercased substring match)
    "arrival": ["arrival", "arrive", "port_arrival", "arrived", "anchorage_arrival"],
    "anchorage": ["anchorage", "anchor", "waiting", "outer", "nafp", "area=anchorage"],
    "berth": ["berth", "all_fast", "berthed", "start_cargo", "alongside"],
    "depart": ["departure", "depart", "sail", "port_departure", "all_clear", "unberthed", "end_cargo"],
}

def parse_span(span: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    a, b = span.split(":", 1)
    return pd.to_datetime(a.strip()), pd.to_datetime(b.strip())

def norm_col(df: pd.DataFrame, name: str) -> Optional[str]:
    for c in df.columns:
        if c.lower() == name:
            return c
    for c in df.columns:
        if name in c.lower():
            return c
    return None

def canonical_event(s: str) -> str:
    x = (s or "").strip().lower()
    for canon, aliases in EVENT_MAP.items():
        if any(a in x for a in aliases):
            return canon
    return x  # pass-through

def ensure_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

# -----------------------------
# I/O
# -----------------------------
def read_calls(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = norm_col(df, "timestamp") or norm_col(df, "time") or df.columns[0]
    df[ts] = ensure_datetime(df[ts])
    df = df.rename(columns={
        ts: "timestamp",
        (norm_col(df, "vessel_id") or norm_col(df, "mmsi") or norm_col(df, "imo") or norm_col(df, "ship") or "vessel"): "vessel",
        (norm_col(df, "port") or "port"): "port",
        (norm_col(df, "country") or "country"): "country",
        (norm_col(df, "event") or "event"): "event",
    })
    # canonicalize
    df["event"] = df["event"].astype(str).map(canonical_event)
    # optional area
    if norm_col(df, "area"):
        df = df.rename(columns={norm_col(df, "area"): "area"})
    else:
        df["area"] = np.where(df["event"].isin(["anchorage"]), "anchorage",
                       np.where(df["event"].isin(["berth"]), "berth", "port"))
    # sort
    cols = ["timestamp","vessel","port","country","event","area"]
    keep = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    df = df[keep].sort_values("timestamp")
    return df

def read_ports(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={
        (norm_col(df, "port") or "port"): "port",
        (norm_col(df, "country") or "country"): "country",
    })
    if norm_col(df, "lat"): df = df.rename(columns={norm_col(df, "lat"): "lat"})
    if norm_col(df, "lon") or norm_col(df, "lng"): df = df.rename(columns={(norm_col(df, "lon") or norm_col(df, "lng")): "lon"})
    for c in ("anch_radius_km","nominal_berth_cap"):
        if norm_col(df, c): df = df.rename(columns={norm_col(df, c): c})
    return df

def read_positions(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    ts = norm_col(df, "timestamp") or norm_col(df, "time") or df.columns[0]
    df[ts] = ensure_datetime(df[ts])
    df = df.rename(columns={ts:"timestamp",
                            (norm_col(df,"vessel_id") or norm_col(df,"mmsi") or norm_col(df,"imo") or "vessel"): "vessel",
                            (norm_col(df,"lat") or "lat"): "lat",
                            (norm_col(df,"lon") or norm_col(df,"lng") or "lon"): "lon",
                            })
    return df.sort_values("timestamp")

# -----------------------------
# Core computation
# -----------------------------
def compute_wait_dwell(call_seq: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[float], Optional[float]]:
    """
    Given events for (vessel, port) ordered by time, return:
    arrival_ts, berth_ts, wait_hours, dwell_hours
    Uses first arrival/anchorage before first berth; first berth before first depart.
    """
    arr = call_seq.loc[call_seq["event"].isin(["arrival","anchorage"])]
    berth = call_seq.loc[call_seq["event"].isin(["berth"])]
    dep = call_seq.loc[call_seq["event"].isin(["depart"])]

    arr_ts = arr["timestamp"].iloc[0] if not arr.empty else None
    berth_ts = berth["timestamp"].iloc[0] if not berth.empty else None
    dep_ts = dep["timestamp"].iloc[0] if not dep.empty else None

    wait_h = None
    dwell_h = None
    if arr_ts is not None and berth_ts is not None and berth_ts >= arr_ts:
        wait_h = float((berth_ts - arr_ts).total_seconds() / 3600.0)
    if berth_ts is not None and dep_ts is not None and dep_ts >= berth_ts:
        dwell_h = float((dep_ts - berth_ts).total_seconds() / 3600.0)
    return arr_ts, berth_ts, wait_h, dwell_h

def daily_kpis(calls: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per (port, date) KPIs: arrivals/berths/departures, queue/backlog, avg wait/dwell, anchorage count.
    Also returns per-vessel staging to build 'waiting' list.
    """
    df = calls.copy()
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= (end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

    df["date"] = df["timestamp"].dt.floor("D")

    # Count basic flows
    flow = df.groupby(["port","date","event"]).size().unstack("event").fillna(0).rename_axis(None, axis=1)
    for k in ["arrival","berth","depart","anchorage"]:
        if k not in flow.columns:
            flow[k] = 0
    flow = flow.reset_index()

    # Vessel-level first events per port
    grp_keys = ["vessel","port"]
    vessel_events = []
    for (v,p), g in df.sort_values("timestamp").groupby(grp_keys):
        arr_ts, berth_ts, wait_h, dwell_h = compute_wait_dwell(g)
        dep_ts = g.loc[g["event"]=="depart","timestamp"].min() if (g["event"]=="depart").any() else None
        vessel_events.append({
            "vessel": v, "port": p,
            "arr_ts": arr_ts, "berth_ts": berth_ts, "dep_ts": dep_ts,
            "wait_h": wait_h, "dwell_h": dwell_h
        })
    ve = pd.DataFrame(vessel_events)

    # Daily averages (wait/dwell computed using first occurrences landing that date)
    # Map each wait to the date of BERTH (when waiting resolves); dwell to DEPART date
    ve["wait_date"] = ve["berth_ts"].dt.floor("D")
    ve["dwell_date"] = ve["dep_ts"].dt.floor("D")

    wait_daily = ve.dropna(subset=["wait_date","wait_h"]).groupby(["port","wait_date"])["wait_h"].mean().rename("avg_wait_h").reset_index().rename(columns={"wait_date":"date"})
    dwell_daily = ve.dropna(subset=["dwell_date","dwell_h"]).groupby(["port","dwell_date"])["dwell_h"].mean().rename("avg_dwell_h").reset_index().rename(columns={"dwell_date":"date"})

    # Queue & backlog: run cumulative balance by date
    # queue_t = queue_{t-1} + arrivals_t - berths_t;  backlog_t = backlog_{t-1} + berths_t - dep_t
    base = flow[["port","date"]].copy()
    base = base.merge(flow[["port","date","arrival","berth","depart"]], on=["port","date"], how="left")
    base = base.sort_values(["port","date"]).fillna(0)
    base["queue"] = base.groupby("port").apply(lambda g: (g["arrival"] - g["berth"]).cumsum().clip(lower=0)).reset_index(level=0, drop=True)
    base["backlog"] = base.groupby("port").apply(lambda g: (g["berth"] - g["depart"]).cumsum().clip(lower=0)).reset_index(level=0, drop=True)

    # Anchorage headcount proxy from events: rolling current count of "anchorage" less a berth/dep on that day
    # Proxy: anch_count_t = anch_count_{t-1} + anchorage_t + arrival_t - berth_t (arrivals often go anchorage first)
    base["anch_count"] = base.groupby("port").apply(lambda g: (g["anchorage"] + g["arrival"] - g["berth"]).cumsum().clip(lower=0)).reset_index(level=0, drop=True)

    # Merge averages
    kpis = base.merge(wait_daily, on=["port","date"], how="left").merge(dwell_daily, on=["port","date"], how="left")
    kpis["arrivals"] = kpis["arrival"]; kpis["berths"] = kpis["berth"]; kpis["departures"] = kpis["depart"]
    kpis = kpis.drop(columns=["arrival","berth","depart"])

    return kpis.sort_values(["port","date"]), ve

def overlay_ais_anchorage_headcount(kpis: pd.DataFrame, positions: Optional[pd.DataFrame], ports_meta: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If AIS positions are present AND ports metadata has lat/lon & anch_radius_km,
    compute daily unique vessel count within anchorage radius per port.
    Returns (kpis_with_ais, ships_in_anchorage_df).
    """
    if positions is None or ports_meta is None or not {"lat","lon","anch_radius_km"}.issubset(set(ports_meta.columns)):
        return kpis, pd.DataFrame()

    # Simple haversine (km)
    def hav_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1 = np.radians(lat1); lon1 = np.radians(lon1)
        lat2 = np.radians(lat2); lon2 = np.radians(lon2)
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2*R*np.arcsin(np.sqrt(a))

    pos = positions.copy()
    pos["date"] = pos["timestamp"].dt.floor("D")

    # Build per-port vessel headcount
    recs = []
    ships_rows = []
    for _, row in ports_meta.dropna(subset=["lat","lon","anch_radius_km"]).iterrows():
        p, lat0, lon0, rad = row["port"], float(row["lat"]), float(row["lon"]), float(row["anch_radius_km"])
        sub = pos.copy()
        # distance to port center
        sub["dist_km"] = hav_km(sub["lat"].values, sub["lon"].values, lat0, lon0)
        in_ring = sub[sub["dist_km"] <= rad]
        if in_ring.empty:
            continue
        by_day = in_ring.groupby("date")["vessel"].nunique().rename("anch_ais_count")
        recs.append(by_day.to_frame().assign(port=p).reset_index())
        # ships listing (latest day)
        latest = in_ring[in_ring["date"] == in_ring["date"].max()]
        ships_rows.extend([{"date": d, "port": p, "vessel": v} for d, v in zip(latest["date"], latest["vessel"])])

    if not recs:
        return kpis, pd.DataFrame()

    anch = pd.concat(recs, axis=0, ignore_index=True)
    kpis2 = kpis.merge(anch, on=["port","date"], how="left")
    # prefer AIS count if available
    kpis2["anch_count"] = kpis2["anch_ais_count"].fillna(kpis2["anch_count"])
    ships_df = pd.DataFrame(ships_rows).drop_duplicates()
    return kpis2, ships_df

def baseline_stats(kpis: pd.DataFrame, span: Optional[str]) -> pd.DataFrame:
    """
    Build per-port baselines (median & MAD) for arrivals, avg_wait_h, queue over baseline span.
    """
    if not span:
        # Use entire history for baseline
        base = kpis.copy()
    else:
        a, b = parse_span(span)
        base = kpis[(kpis["date"] >= a) & (kpis["date"] <= b)].copy()
        if base.empty:
            base = kpis.copy()

    def mad(x):
        med = np.nanmedian(x)
        return np.nanmedian(np.abs(x - med)) * 1.4826

    agg = base.groupby("port").agg(
        arr_med=("arrivals","median"),
        arr_mad=("arrivals", mad),
        wait_med=("avg_wait_h","median"),
        wait_mad=("avg_wait_h", mad),
        queue_med=("queue","median"),
        queue_mad=("queue", mad),
    ).reset_index()
    # avoid zero MAD
    for c in ["arr_mad","wait_mad","queue_mad"]:
        agg[c] = agg[c].replace(0, np.nan)
        agg[c] = agg[c].fillna(agg[c].median() if np.isfinite(agg[c].median()) else 1.0)
    return agg

def congestion_indices(kpis: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """
    Z-scores per port per day:
      z_wait  = (avg_wait_h - wait_med) / wait_mad
      z_queue = (queue - queue_med) / queue_mad
      z_comb  = 0.5*z_wait + 0.5*z_queue
    """
    df = kpis.merge(base, on="port", how="left")
    df["z_wait"] = (df["avg_wait_h"] - df["wait_med"]) / (df["wait_mad"] + 1e-9)
    df["z_queue"] = (df["queue"] - df["queue_med"]) / (df["queue_mad"] + 1e-9)
    df["z_comb"] = 0.5 * df["z_wait"].fillna(0) + 0.5 * df["z_queue"].fillna(0)
    return df

def current_waiters(ve: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Vessels arrived but not yet berthed by 'asof' (waiting).
    """
    mask = (ve["arr_ts"].notna()) & ((ve["berth_ts"].isna()) | (ve["berth_ts"] > asof))
    cur = ve.loc[mask, ["vessel","port","arr_ts","berth_ts"]].copy()
    cur["asof"] = asof
    cur["waiting_hours"] = (cur["asof"] - cur["arr_ts"]).dt.total_seconds() / 3600.0
    return cur.sort_values(["port","waiting_hours"], ascending=[True, False])

# -----------------------------
# CLI
# -----------------------------
@dataclass
class Config:
    calls: str
    ports: Optional[str]
    positions: Optional[str]
    start: Optional[str]
    end: Optional[str]
    baseline: Optional[str]
    baltic_only: bool
    countries: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Baltic port congestion KPIs from port-call/AIS data")
    ap.add_argument("--calls", required=True, help="CSV of port-call events")
    ap.add_argument("--ports", default="", help="CSV of port metadata (port, country, lat, lon, anch_radius_km, nominal_berth_cap)")
    ap.add_argument("--positions", default="", help="AIS positions (parquet/csv) for anchorage headcounts (optional)")
    ap.add_argument("--start", default="", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="", help="End date (YYYY-MM-DD)")
    ap.add_argument("--baseline", default="", help="Baseline span 'YYYY-MM-DD:YYYY-MM-DD' for z-scores")
    ap.add_argument("--baltic-only", action="store_true", help="Restrict to Baltic littoral countries")
    ap.add_argument("--countries", default="", help="Override country code set, e.g., 'SE,FI,DK,DE,PL,LT,LV,EE,RU'")
    ap.add_argument("--outdir", default="out_baltic")
    return ap.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    cfg = Config(
        calls=args.calls,
        ports=args.ports or None,
        positions=args.positions or None,
        start=args.start or None,
        end=args.end or None,
        baseline=args.baseline or None,
        baltic_only=bool(args.baltic_only),
        countries=args.countries or None,
        outdir=args.outdir,
    )

    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)

    calls = read_calls(cfg.calls)
    ports_meta = read_ports(cfg.ports)
    positions = read_positions(cfg.positions) if cfg.positions else None

    # Filter by country if requested
    if cfg.baltic_only or cfg.countries:
        keep = set([c.strip().upper() for c in (cfg.countries.split(",") if cfg.countries else list(BALTIC_LITTORAL)) if c.strip()])
        if "country" in calls.columns:
            calls = calls[calls["country"].astype(str).str.upper().isin(keep)]
        if ports_meta is not None and "country" in ports_meta.columns:
            ports_meta = ports_meta[ports_meta["country"].astype(str).str.upper().isin(keep)]

    # Date filters
    start = pd.to_datetime(cfg.start) if cfg.start else None
    end = pd.to_datetime(cfg.end) if cfg.end else None

    kpis, ve = daily_kpis(calls, start, end)
    # AIS overlay (if available and metadata present)
    kpis, ships_anch = overlay_ais_anchorage_headcount(kpis, positions, ports_meta)

    # Baseline & congestion index
    base = baseline_stats(kpis, cfg.baseline)
    idx = congestion_indices(kpis, base)

    # Latest snapshot
    if not kpis.empty:
        asof = kpis["date"].max()
    else:
        asof = pd.Timestamp.now().floor("D")
    waiters = current_waiters(ve, asof)

    # Outputs
    kpis.to_csv(outdir / "port_daily.csv", index=False)
    idx.to_csv(outdir / "congestion_index.csv", index=False)
    snap = idx[idx["date"] == asof].copy().sort_values("z_comb", ascending=False)
    snap.to_csv(outdir / "snapshot.csv", index=False)
    waiters.to_csv(outdir / "ships_waiting.csv", index=False)
    if not ships_anch.empty:
        ships_anch.to_csv(outdir / "ships_in_anchorage.csv", index=False)

    # Headline summary
    summary = {
        "asof": str(asof.date()),
        "n_ports": int(kpis["port"].nunique()) if not kpis.empty else 0,
        "ports_most_congested_today": snap[["port","z_comb","avg_wait_h","queue"]].head(10).to_dict(orient="records") if not snap.empty else [],
        "config": asdict(cfg),
    }
    (outdir / "kpis.json").write_text(json.dumps(summary, indent=2))

    # Console
    print("== Baltic Port Congestion ==")
    print(f"AsoF: {asof.date()}  Ports: {summary['n_ports']}")
    if not snap.empty:
        print(snap[["port","z_comb","avg_wait_h","queue"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
