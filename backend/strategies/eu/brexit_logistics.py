#!/usr/bin/env python3
"""
brexit_logistics.py — Post-Brexit logistics analytics: delays, duties, and lane risk

What it does
------------
Given shipment event logs and reference tables, this tool computes:
- Lead times & on-time performance
- Border dwell (UK↔EU) and queue distributions per crossing (Dover, Holyhead, Calais, etc.)
- Additional costs from tariffs/duties (rules of origin aware) and customs fees
- FX-normalized cost baselines
- Lane risk scores (delay frequency × severity × volume)
- Scenario analysis (e.g., +X% tariffs, +Y hours customs, driver shortage multipliers)

Inputs
------
--shipments shipments.csv
    Long-form shipment events (wide tolerated). Expected columns (case-insensitive):
      shipment_id, event, timestamp, location, country, node
      optional: mode, carrier, incoterm, hs6, value, weight_kg, freight_cost, lane
    Typical events: pickup, depart_port, uk_exit, eu_entry, customs_release, delivery
    Use as many as you have; the script derives segments/dwells from available pairs.

--tariffs tariffs.csv (optional)
    Columns: hs6, origin, destination, rate_adval (in decimals, e.g., 0.04 for 4%)
    If missing, duty=0 unless overridden by scenarios.

--origin_qual origin_qual.csv (optional)
    Rules of origin qualification flags by shipment or HS.
    Either long (shipment_id, qualifies) or code-level (hs6, origin, qualifies).
    'qualifies' ∈ {0,1}. If 1, duty for that record = 0.

--fx fx.csv (optional)
    Daily FX to a base currency. Columns: date, curr, rate_to_base (e.g., GBP->EUR).
    Shipment monetary inputs assumed in their native 'currency' column if present; else base.

--lanes lanes.csv (optional)
    Mapping to standardize lane names/crossings.
    Columns: node/location -> lane, country_from, country_to, is_border (0/1)

--fees fees.csv (optional)
    Additional per-shipment fixed/variable costs (e.g., customs broker).
    Columns: name, amount, currency (opt), apply_if (expr over columns, e.g., "is_border==1 and mode=='road'")

Scenario knobs (optional)
-------------------------
--scen-tariff +0.02     Additive change to tariff rate (e.g., +0.02 = +2pp)
--scen-delay +2.0       Add Y hours to border dwell (applied to all UK↔EU crossings)
--scen-driver 1.10      Multiply border dwell by this factor (driver shortage impact)
--scen-broker +25       Add flat customs/broker fee in base currency

Outputs (to --outdir)
---------------------
- shipment_costs.csv        Per-shipment baseline & scenario costs; lead times; border dwell
- lane_daily.csv            Per-lane daily volumes, mean dwell, on-time rate
- lane_risk.csv             Lane risk scores (exposure × delay severity × frequency)
- dwell_hist.csv            Histogram (per lane) of border dwell hours
- kpis.json                 Headline KPIs (weighted averages, % late, added cost %)
- config.json               Run configuration

Usage
-----
python brexit_logistics.py \
  --shipments shipments.csv --tariffs tariffs.csv --origin_qual origin_qual.csv \
  --fx fx.csv --lanes lanes.csv --fees fees.csv \
  --scen-delay 1.5 --scen-tariff 0.01 --outdir out_brexit
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# ----------------- Utilities -----------------
def norm_col(df: pd.DataFrame, name: str) -> Optional[str]:
    for c in df.columns:
        if c.lower() == name:
            return c
    for c in df.columns:
        if name in c.lower():
            return c
    return None


def ensure_ts(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")


def hours_between(a: Optional[pd.Timestamp], b: Optional[pd.Timestamp]) -> Optional[float]:
    if a is None or b is None:
        return None
    if pd.isna(a) or pd.isna(b):
        return None
    return float((b - a).total_seconds() / 3600.0)


def read_csv_any(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


# ----------------- I/O readers -----------------
def read_shipments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = norm_col(df, "timestamp") or df.columns[0]
    df[ts] = ensure_ts(df[ts])
    df = df.rename(
        columns={
            ts: "timestamp",
            (norm_col(df, "shipment_id") or "shipment_id"): "shipment_id",
            (norm_col(df, "event") or "event"): "event",
            (norm_col(df, "location") or "location"): "location",
            (norm_col(df, "country") or "country"): "country",
            (norm_col(df, "node") or "node"): "node",
        }
    )
    # optional enrichments
    for k, std in [
        ("mode", "mode"),
        ("carrier", "carrier"),
        ("incoterm", "incoterm"),
        ("hs6", "hs6"),
        ("value", "value"),
        ("weight_kg", "weight_kg"),
        ("freight_cost", "freight_cost"),
        ("currency", "currency"),
        ("lane", "lane"),
        ("origin", "origin"),
        ("destination", "destination"),
    ]:
        c = norm_col(df, k)
        if c:
            df = df.rename(columns={c: std})
    # Lowercase events for matching
    df["event"] = df["event"].astype(str).str.lower()
    return df.sort_values(["shipment_id", "timestamp"])


def read_tariffs(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            (norm_col(df, "hs6") or "hs6"): "hs6",
            (norm_col(df, "origin") or "origin"): "origin",
            (norm_col(df, "destination") or "destination"): "destination",
            (norm_col(df, "rate_adval") or norm_col(df, "rate") or "rate"): "rate",
        }
    )
    df["hs6"] = df["hs6"].astype(str).str[:6]
    df["origin"] = df["origin"].astype(str).str.upper()
    df["destination"] = df["destination"].astype(str).str.upper()
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0.0)
    return df


def read_origin_qual(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "shipment_id" in cols and "qualifies" in cols:
        df = df.rename(columns={norm_col(df, "shipment_id"): "shipment_id", norm_col(df, "qualifies"): "qualifies"})
        df["qualifies"] = (pd.to_numeric(df["qualifies"], errors="coerce") > 0).astype(int)
        df["hs6"] = None
        df["origin"] = None
        return df
    # code-level
    df = df.rename(
        columns={
            (norm_col(df, "hs6") or "hs6"): "hs6",
            (norm_col(df, "origin") or "origin"): "origin",
            (norm_col(df, "qualifies") or "qualifies"): "qualifies",
        }
    )
    df["hs6"] = df["hs6"].astype(str).str[:6]
    df["origin"] = df["origin"].astype(str).str.upper()
    df["qualifies"] = (pd.to_numeric(df["qualifies"], errors="coerce") > 0).astype(int)
    return df


def read_fx(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            (norm_col(df, "date") or "date"): "date",
            (norm_col(df, "curr") or norm_col(df, "currency") or "currency"): "currency",
            (norm_col(df, "rate_to_base") or norm_col(df, "rate") or "rate"): "rate_to_base",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["currency"] = df["currency"].astype(str).str.upper()
    df["rate_to_base"] = pd.to_numeric(df["rate_to_base"], errors="coerce")
    return df


def read_lanes(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    # Flexible headers
    lane = norm_col(df, "lane") or "lane"
    node = norm_col(df, "node") or (norm_col(df, "location") or "location")
    df = df.rename(
        columns={
            lane: "lane",
            node: "node",
            (norm_col(df, "country_from") or "country_from"): "country_from",
            (norm_col(df, "country_to") or "country_to"): "country_to",
            (norm_col(df, "is_border") or "is_border"): "is_border",
        }
    )
    if "is_border" in df.columns:
        df["is_border"] = (pd.to_numeric(df["is_border"], errors="coerce") > 0).astype(int)
    return df


def read_fees(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            (norm_col(df, "name") or "name"): "name",
            (norm_col(df, "amount") or "amount"): "amount",
            (norm_col(df, "currency") or "currency"): "currency",
            (norm_col(df, "apply_if") or "apply_if"): "apply_if",
        }
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "currency" in df.columns:
        df["currency"] = df["currency"].astype(str).str.upper()
    else:
        df["currency"] = "BASE"
    return df


# ----------------- Core transforms -----------------
def derive_lane_and_border(ship: pd.DataFrame, lanes_map: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = ship.copy()
    # lane inference: prefer provided lane; else map by node/location; else country transitions
    if lanes_map is not None:
        m = lanes_map[["node", "lane", "is_border"]].drop_duplicates()
        # Use node if present, else location
        key = "node" if "node" in df.columns else "location"
        df = df.merge(m, how="left", left_on=key, right_on="node")
        df["lane"] = df.get("lane_x", df.get("lane", np.nan)).fillna(df.get("lane_y", np.nan))
        df = df.drop(columns=[c for c in ["lane_x", "lane_y", "node_y"] if c in df.columns]).rename(columns={"node_x": "node"})
    if "lane" not in df.columns:
        df["lane"] = df["location"].fillna(df["node"])
    # border detection: UK↔EU if consecutive countries differ with one being GB/UK and the other in EU27.
    df["country"] = df["country"].astype(str).str.upper()
    return df


def summarize_shipments(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build shipment-level aggregates:
      pickup_ts, uk_exit_ts, eu_entry_ts, customs_release_ts, delivery_ts
      border_dwell_h, lead_time_h, on_time (if promised_delivery present)
    """
    # If promised delivery exists, parse it
    promised_col = norm_col(events, "promised_delivery")
    if promised_col:
        events[promised_col] = ensure_ts(events[promised_col])

    recs = []
    for sid, g in events.groupby("shipment_id"):
        g = g.sort_values("timestamp")
        # pick canonical timestamps when available
        get_ts = lambda name_list: next((g.loc[g["event"] == n, "timestamp"].iloc[0] for n in name_list if (g["event"] == n).any()), None)
        pickup = get_ts(["pickup", "collection"])
        uk_exit = get_ts(["uk_exit", "gb_exit", "leave_uk", "customs_export_clearance"])
        eu_entry = get_ts(["eu_entry", "enter_eu", "fr_entry", "ie_entry", "import_arrival"])
        customs_rel = get_ts(["customs_release", "import_clearance", "ie_release", "fr_release"])
        delivery = get_ts(["delivery", "delivered", "pod"])

        border_dwell = hours_between(uk_exit, eu_entry) if (uk_exit and eu_entry and eu_entry >= uk_exit) else None
        import_dwell = hours_between(eu_entry, customs_rel) if (eu_entry and customs_rel and customs_rel >= eu_entry) else None
        lead_time = hours_between(pickup, delivery)

        rec = {
            "shipment_id": sid,
            "pickup_ts": pickup,
            "uk_exit_ts": uk_exit,
            "eu_entry_ts": eu_entry,
            "customs_release_ts": customs_rel,
            "delivery_ts": delivery,
            "border_dwell_h": border_dwell,
            "import_dwell_h": import_dwell,
            "lead_time_h": lead_time,
        }
        if promised_col:
            promised = g[promised_col].dropna().iloc[0] if g[promised_col].notna().any() else None
            rec["promised_delivery_ts"] = promised
            if promised and delivery:
                rec["on_time"] = int(delivery <= promised)
        recs.append(rec)
    return pd.DataFrame(recs)


def fx_to_base(amount: float, currency: str, date: pd.Timestamp, fx: Optional[pd.DataFrame]) -> float:
    if amount is None or pd.isna(amount):
        return 0.0
    if fx is None or currency is None or currency.upper() == "BASE":
        return float(amount)
    day = pd.to_datetime(date).floor("D")
    row = fx[(fx["currency"] == currency.upper()) & (fx["date"] == day)]
    if row.empty:
        # fallback: last available rate
        row = fx[fx["currency"] == currency.upper()].sort_values("date").iloc[-1:] if (fx["currency"] == currency.upper()).any() else None
    rate = float(row["rate_to_base"].iloc[0]) if row is not None and not row.empty else 1.0
    return float(amount) * rate


def compute_duties(
    ship_summary: pd.DataFrame,
    shipments_events: pd.DataFrame,
    tariffs: Optional[pd.DataFrame],
    origin_qual: Optional[pd.DataFrame],
    fx: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Returns a DataFrame keyed by shipment_id with:
      customs_value_base, duty_rate, duty_amount_base
    """
    # Build per-shipment commodity/monetary info (latest non-null in events)
    cols = {}
    for k in ["hs6", "value", "freight_cost", "currency", "origin", "destination"]:
        c = norm_col(shipments_events, k)
        if c:
            cols[k] = c
    info = shipments_events.groupby("shipment_id").agg({v: "last" for v in cols.values()})
    info = info.rename(columns={v: k for k, v in cols.items()})
    if "hs6" in info.columns:
        info["hs6"] = info["hs6"].astype(str).str[:6]
    # Determine customs value in BASE: value + freight (if provided)
    # Choose a date near EU entry for FX conversion
    entry_date = ship_summary.set_index("shipment_id")["eu_entry_ts"]
    info = info.join(entry_date, how="left")
    info["value"] = pd.to_numeric(info.get("value", 0.0), errors="coerce").fillna(0.0)
    info["freight_cost"] = pd.to_numeric(info.get("freight_cost", 0.0), errors="coerce").fillna(0.0)
    info["currency"] = info.get("currency", "BASE").fillna("BASE").astype(str)
    info["customs_value_base"] = [
        fx_to_base(info["value"].iat[i] + info["freight_cost"].iat[i], info["currency"].iat[i], entry_date.iloc[i], fx)
        for i in range(len(info))
    ]

    # Origin qualification
    qual_map = {}
    if origin_qual is not None:
        if "shipment_id" in [c.lower() for c in origin_qual.columns]:
            tmp = origin_qual.copy()
            tmp.columns = [c.lower() for c in tmp.columns]
            qual_map = dict(zip(tmp["shipment_id"], tmp["qualifies"]))
        else:
            # code-level: hs6+origin
            code_map = origin_qual.set_index(["hs6", "origin"])["qualifies"].to_dict()
            info["origin"] = info.get("origin", "").astype(str).str.upper()
            info["qualifies"] = [
                int(code_map.get((info["hs6"].iat[i], info["origin"].iat[i]), 0)) if ("hs6" in info.columns and "origin" in info.columns) else 0
                for i in range(len(info))
            ]
    info["qualifies"] = info.get("qualifies", 0)

    # Duty rate lookup
    rate = []
    for i, row in info.reset_index().iterrows():
        r = 0.0
        if int(row.get("qualifies", 0)) == 1:
            r = 0.0
        elif tariffs is not None and "hs6" in info.columns and "origin" in info.columns and "destination" in info.columns:
            hit = tariffs[
                (tariffs["hs6"] == str(row["hs6"])[:6])
                & (tariffs["origin"] == str(row.get("origin", "")).upper())
                & (tariffs["destination"] == str(row.get("destination", "")).upper())
            ]
            r = float(hit["rate"].iloc[0]) if not hit.empty else 0.0
        rate.append(r)
    info["duty_rate"] = rate

    # Compute duty (base currency)
    info["duty_amount_base"] = info["customs_value_base"] * info["duty_rate"]
    return info.reset_index()[["shipment_id", "customs_value_base", "duty_rate", "duty_amount_base"]]


def apply_fees_per_shipment(base_cost: float, meta: dict, fees: Optional[pd.DataFrame], fx: Optional[pd.DataFrame]) -> float:
    if fees is None or fees.empty:
        return 0.0
    # eval context is the meta dict
    add = 0.0
    for _, row in fees.iterrows():
        expr = str(row.get("apply_if", "")).strip()
        ok = True
        if expr and expr.lower() not in ("", "true", "nan"):
            try:
                ok = bool(eval(expr, {}, meta))
            except Exception:
                ok = False
        if ok:
            amt = float(row["amount"])
            curr = row.get("currency", "BASE")
            # Use delivery/pickup date if available for FX; else today
            day = meta.get("eu_entry_ts") or meta.get("pickup_ts") or pd.Timestamp.today()
            add += fx_to_base(amt, curr, day, fx)
    return add


def build_lane_daily(events: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    # Lane assigned per shipment: prefer explicit lane in events else 'UK<->EU'
    lane_map = (
        events[["shipment_id", "lane"]]
        .dropna()
        .groupby("shipment_id")
        .agg({"lane": "last"})
    )
    df = summary.set_index("shipment_id").join(lane_map, how="left")
    df["lane"] = df["lane"].fillna(np.where(df["uk_exit_ts"].notna() | df["eu_entry_ts"].notna(), "UK<->EU", "Domestic/Other"))
    # Date = EU entry date if border, else delivery date
    df["date"] = pd.to_datetime(
        np.where(df["eu_entry_ts"].notna(), df["eu_entry_ts"], df["delivery_ts"])
    ).astype("datetime64[ns]").astype("datetime64[D]")

    # On-time rate (if available)
    if "on_time" not in df.columns:
        df["on_time"] = np.nan

    grp = df.groupby(["lane", "date"])
    out = grp.agg(
        shipments=("lead_time_h", "size"),
        mean_lead_h=("lead_time_h", "mean"),
        mean_border_dwell_h=("border_dwell_h", "mean"),
        p95_border_dwell_h=("border_dwell_h", lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan),
        on_time_rate=("on_time", "mean"),
    ).reset_index()
    return out.sort_values(["lane", "date"])


def lane_risk_score(lane_daily: pd.DataFrame) -> pd.DataFrame:
    # Score: exposure (shipments) × severity (p95 dwell) × frequency (% days with dwell > 2h)
    scores = []
    for lane, g in lane_daily.groupby("lane"):
        exposure = float(g["shipments"].sum())
        p95 = float(np.nanmean(g["p95_border_dwell_h"]))
        freq = float(np.mean((g["mean_border_dwell_h"] > 2.0).fillna(False)))
        score = exposure * (p95 if np.isfinite(p95) else 0.0) * freq
        scores.append({"lane": lane, "exposure": exposure, "p95_dwell_h": p95, "freq_gt2h": freq, "risk_score": score})
    return pd.DataFrame(scores).sort_values("risk_score", ascending=False)


def dwell_histogram(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["lane"] = df.get("lane", "UK<->EU")
    bins = [0, 1, 2, 4, 8, 12, 24, 36, 48, 72, 96, np.inf]
    labels = ["<=1", "1-2", "2-4", "4-8", "8-12", "12-24", "24-36", "36-48", "48-72", "72-96", "96+"]
    recs = []
    for lane, g in df.groupby("lane"):
        cats = pd.cut(g["border_dwell_h"], bins=bins, labels=labels, include_lowest=True)
        t = cats.value_counts(dropna=True).reindex(labels).fillna(0).astype(int)
        for lab in labels:
            recs.append({"lane": lane, "bucket": lab, "count": int(t[lab])})
    return pd.DataFrame(recs)


# ----------------- Scenarios -----------------
@dataclass
class Scenario:
    tariff_add: float = 0.0   # additive to ad-valorem rate
    delay_add_h: float = 0.0  # added border dwell hours
    driver_mult: float = 1.0  # multiplicative border dwell
    broker_add_base: float = 0.0  # flat fee in BASE


def apply_scenarios(costs: pd.DataFrame, scen: Scenario) -> pd.DataFrame:
    df = costs.copy()
    # Adjust duty rate and recompute
    df["duty_rate_scen"] = (df["duty_rate"].fillna(0.0) + scen.tariff_add).clip(lower=0.0)
    df["duty_amount_base_scen"] = df["customs_value_base"].fillna(0.0) * df["duty_rate_scen"]

    # Delay scenarios
    bd = df["border_dwell_h"].fillna(0.0)
    df["border_dwell_h_scen"] = bd * scen.driver_mult + scen.delay_add_h

    # Add flat broker/customs cost
    df["scenario_extra_fees_base"] = df.get("extra_fees_base", 0.0) + float(scen.broker_add_base)

    # Total baseline vs scenario
    df["baseline_cost_base"] = df["freight_cost_base"].fillna(0.0) + df.get("extra_fees_base", 0.0) + df["duty_amount_base"].fillna(0.0)
    df["scenario_cost_base"] = df["freight_cost_base"].fillna(0.0) + df["scenario_extra_fees_base"].fillna(0.0) + df["duty_amount_base_scen"].fillna(0.0)
    df["added_cost_base"] = df["scenario_cost_base"] - df["baseline_cost_base"]
    return df


# ----------------- CLI & Orchestration -----------------
@dataclass
class Config:
    shipments: str
    tariffs: Optional[str]
    origin_qual: Optional[str]
    fx: Optional[str]
    lanes: Optional[str]
    fees: Optional[str]
    scen_tariff: float
    scen_delay: float
    scen_driver: float
    scen_broker: float
    outdir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Brexit logistics analytics: delays, duties, lane risk")
    ap.add_argument("--shipments", required=True)
    ap.add_argument("--tariffs", default="")
    ap.add_argument("--origin_qual", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--lanes", default="")
    ap.add_argument("--fees", default="")
    ap.add_argument("--scen-tariff", type=float, default=0.0)
    ap.add_argument("--scen-delay", type=float, default=0.0)
    ap.add_argument("--scen-driver", type=float, default=1.0)
    ap.add_argument("--scen-broker", type=float, default=0.0)
    ap.add_argument("--outdir", default="out_brexit")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        shipments=args.shipments,
        tariffs=args.tariffs or None,
        origin_qual=args.origin_qual or None,
        fx=args.fx or None,
        lanes=args.lanes or None,
        fees=args.fees or None,
        scen_tariff=float(args.scen_tariff),
        scen_delay=float(args.scen_delay),
        scen_driver=float(args.scen_driver),
        scen_broker=float(args.scen_broker),
        outdir=args.outdir,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    events = read_shipments(cfg.shipments)
    tariffs = read_tariffs(cfg.tariffs)
    origin_qual = read_origin_qual(cfg.origin_qual)
    fx = read_fx(cfg.fx)
    lanes_map = read_lanes(cfg.lanes)
    fees = read_fees(cfg.fees)

    # Derive lanes/border markers
    events = derive_lane_and_border(events, lanes_map)

    # Shipment summary
    summary = summarize_shipments(events)

    # Monetary: freight cost to base (per shipment)
    money_cols = {k: norm_col(events, k) for k in ("freight_cost", "currency")}
    ship_money = (
        events.groupby("shipment_id")
        .agg(
            freight_cost=(money_cols.get("freight_cost") or "freight_cost", "last"),
            currency=(money_cols.get("currency") or "currency", "last"),
        )
        if money_cols.get("freight_cost")
        else pd.DataFrame({"freight_cost": []})
    )
    # Convert freight to base using delivery date
    tmp = summary.set_index("shipment_id")[["delivery_ts", "eu_entry_ts", "pickup_ts"]]
    ship_money = ship_money.join(tmp, how="left")
    if not ship_money.empty:
        ship_money["currency"] = ship_money.get("currency", "BASE").fillna("BASE")
        ship_money["freight_cost_base"] = [
            fx_to_base(ship_money["freight_cost"].iat[i] if pd.notna(ship_money["freight_cost"].iat[i]) else 0.0,
                       ship_money["currency"].iat[i],
                       ship_money["delivery_ts"].iat[i] or ship_money["eu_entry_ts"].iat[i] or ship_money["pickup_ts"].iat[i] or pd.Timestamp.today(),
                       fx)
            for i in range(len(ship_money))
        ]
    else:
        ship_money = pd.DataFrame(index=summary["shipment_id"]).assign(freight_cost_base=0.0)

    # Duties
    duties = compute_duties(summary, events, tariffs, origin_qual, fx)

    # Fees (rule-based)
    meta_cols = [
        "pickup_ts",
        "uk_exit_ts",
        "eu_entry_ts",
        "customs_release_ts",
        "delivery_ts",
    ]
    meta_df = summary.set_index("shipment_id")[meta_cols]
    extra_fee_list = []
    for sid, mrow in meta_df.iterrows():
        meta = {k: mrow[k] for k in meta_cols}
        # add standard fields if used in expressions
        meta.update({"is_border": int(pd.notna(mrow["eu_entry_ts"]) and pd.notna(mrow["uk_exit_ts"]))})
        extra_fee_list.append({"shipment_id": sid, "extra_fees_base": apply_fees_per_shipment(0.0, meta, fees, fx)})
    fees_df = pd.DataFrame(extra_fee_list).set_index("shipment_id")

    # Merge costs
    costs = (
        summary.set_index("shipment_id")
        .join(ship_money[["freight_cost_base"]], how="left")
        .join(duties.set_index("shipment_id"), how="left")
        .join(fees_df, how="left")
        .fillna({"freight_cost_base": 0.0, "duty_amount_base": 0.0, "duty_rate": 0.0, "extra_fees_base": 0.0})
        .reset_index()
    )

    # Baseline totals
    costs["baseline_cost_base"] = costs["freight_cost_base"] + costs["extra_fees_base"] + costs["duty_amount_base"]

    # Scenario
    scen = Scenario(
        tariff_add=cfg.scen_tariff,
        delay_add_h=cfg.scen_delay,
        driver_mult=cfg.scen_driver,
        broker_add_base=cfg.scen_broker,
    )
    costs_scen = apply_scenarios(costs, scen)

    # Lane daily and risk
    lane_daily = build_lane_daily(events, summary)
    # attach lanes into summary for histogram
    lane_for_hist = (
        events[["shipment_id", "lane"]].dropna().groupby("shipment_id").last()
    )
    summary_hist = summary.set_index("shipment_id").join(lane_for_hist, how="left").reset_index()
    dwell_hist = dwell_histogram(summary_hist)
    lane_risk = lane_risk_score(lane_daily)

    # Headlines
    head = {
        "n_shipments": int(len(costs_scen)),
        "mean_lead_h": float(np.nanmean(costs_scen["lead_time_h"])),
        "mean_border_dwell_h": float(np.nanmean(costs_scen["border_dwell_h"])),
        "on_time_rate": float(np.nanmean(costs_scen.get("on_time", np.nan))) if "on_time" in costs_scen else None,
        "duty_rate_avg": float(np.nanmean(costs_scen["duty_rate"])) if "duty_rate" in costs_scen else 0.0,
        "added_cost_total_base": float(costs_scen["added_cost_base"].sum()),
        "added_cost_avg_per_shipment_base": float(costs_scen["added_cost_base"].mean()),
        "scenario": asdict(scen),
    }

    # Write outputs
    costs_scen.to_csv(outdir / "shipment_costs.csv", index=False)
    lane_daily.to_csv(outdir / "lane_daily.csv", index=False)
    lane_risk.to_csv(outdir / "lane_risk.csv", index=False)
    dwell_hist.to_csv(outdir / "dwell_hist.csv", index=False)
    (outdir / "kpis.json").write_text(json.dumps(head, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Console
    print("== Brexit Logistics Report ==")
    print(f"Shipments: {head['n_shipments']:,}  Mean lead (h): {head['mean_lead_h']:.1f}  Mean border dwell (h): {head['mean_border_dwell_h']:.1f}")
    if head.get("on_time_rate") is not None and not np.isnan(head["on_time_rate"]):
        print(f"On-time rate: {head['on_time_rate']*100:.1f}%")
    print(f"Avg duty rate: {head['duty_rate_avg']*100:.2f}%  Scenario added cost (avg, base): {head['added_cost_avg_per_shipment_base']:.2f}")
    print("\nTop risky lanes:")
    print(lane_risk.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
