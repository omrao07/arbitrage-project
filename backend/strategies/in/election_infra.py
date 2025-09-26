#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
election_infra.py — Local election infrastructure risk, capacity & scenario engine
----------------------------------------------------------------------------------

What this does
==============
Given precinct/site assets, dependencies, staffing, and turnout forecasts, this script:

1) Builds a dependency graph (sites ↔ power/network/vendor/facility nodes).
2) Scores risk per site & asset (physical, cyber, power, network, vendor, single-points).
3) Estimates baseline *throughput* and *queueing waits* using a simple M/M/c model
   across core stations (check-in, ballot issuance/marking, scanner/tabulator).
4) Applies disruption scenarios (e.g., power outage, vendor outage, equipment failure,
   staffing shortfall, weather) and recomputes capacity & waits.
5) Produces tidy CSVs and a JSON summary with prioritized mitigations (“playbook”).

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--assets assets.csv            REQUIRED
  Rows describe precinct or vote-center equipment & posture.
  Columns (examples, case-insensitive; extras ignored):
    precinct_id, county, site_name, voters_registered, is_vote_center (0/1),
    chk_in_stations, chk_in_thr_per_hr, chk_in_offline_ok (0/1),
    ballot_printers, bmds, scanner_units, issue_thr_per_hr, scan_thr_per_hr,
    backup_power (0/1), network_redundant (0/1), paper_pollbooks (0/1),
    vendor_epoll, vendor_ballot, opens (HH:MM), closes (HH:MM)

--dependencies deps.csv        OPTIONAL
  Directed edges "from" → "to" with a type; used to fan-out failure impact.
  Columns: src, dst, type   where type ∈ {power, network, vendor, facility}

--staffing staffing.csv        OPTIONAL
  Columns: precinct_id, pollworkers, techs

--turnout turnout.csv          OPTIONAL (hourly arrivals per site; else auto shape)
  Long: date, precinct_id, hour (0-23), arrivals
  Or:   precinct_id, hour, arrivals

--incidents incidents.csv      OPTIONAL (historical events to inform risk)
  Columns: precinct_id, date, type, severity (1-5)

--scenarios scenarios.csv      OPTIONAL (key,value pairs; multiple rows per scenario)
  Columns: scenario, key, value
  Keys (examples):
    power.outage.county=King
    power.outage.share=0.5             (share of sites *without* backup hit)
    vendor.outage=E-POLLCO
    vendor.degradation=0.7             (capacity multiplier if offline fallback)
    staffing.shortfall.pct=20
    equipment.fail.chk_in.pct=30
    equipment.fail.scanner.pct=15
    weather.impact.county=Snohomish
    hours.extend.minutes=60

CLI
---
Basic:
  python election_infra.py \
    --assets assets.csv --dependencies deps.csv --staffing staffing.csv \
    --turnout turnout.csv --scenarios scenarios.csv --scenario POWER \
    --outdir out_election

Outputs
-------
- baseline_capacity.csv    Per precinct: capacities, bottleneck, waits, peak hour
- risk_scores.csv          Per precinct/site risk components & composite score
- scenario_impacts.csv     Per scenario×precinct: new capacity, wait deltas, voters affected
- playbook.csv             Suggested mitigations sorted by benefit/feasibility
- summary.json             Headline metrics & top-risk sites
- config.json              Run configuration for reproducibility

Notes & assumptions
-------------------
• Queueing: M/M/c approximation on the *bottleneck* station (min of check-in, issue, scan).
  Arrival rate λ from hourly forecast; service rate μ per station from *_thr_per_hr.
• If turnout.csv missing, we synthesize a 13-hour triangular curve peaking mid-day.
• Risk scoring is heuristic & transparent in the code; adapt weights to your jurisdiction.
• This is **planning**/resilience analytics; it neither collects voter PII nor handles ballots.

DISCLAIMER
----------
Research tool with simplifying assumptions; validate with local procedures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ------------- helpers ----------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if str(c).lower() == t: return c
    for c in df.columns:
        if t in str(c).lower(): return c
    return None

def safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def to_time(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce").dt.time
    return out

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def mmc_wait_minutes(lam: float, mu: float, c: int) -> Tuple[float, float]:
    """
    M/M/c queue: returns (Wq_minutes, W_minutes). Units: per HOUR inputs.
    lam: arrival rate per hour, mu: service rate per hour per server, c: servers.
    """
    if c <= 0 or mu <= 0:
        return float('inf'), float('inf')
    rho = lam / (c * mu)
    if rho >= 1.0:
        return float('inf'), float('inf')
    # Erlang C
    def fact(n):
        return np.math.factorial(n)
    summ = sum([ (lam/mu)**k / fact(k) for k in range(c) ])
    term = ((lam/mu)**c / (fact(c) * (1 - rho)))
    P0 = 1.0 / (summ + term)
    Pw = term * P0
    Wq = Pw / (c * mu - lam)  # in hours
    W  = Wq + 1.0/mu          # total time in system (hours)
    return Wq*60.0, W*60.0

def triangular_load(shape_hours: List[int], total: float) -> pd.Series:
    """
    Symmetric triangular profile over provided hours; peaks mid-window.
    """
    n = len(shape_hours)
    x = np.arange(n)
    peak = (n - 1) / 2
    tri = 1.0 - np.abs(x - peak)/peak if peak != 0 else np.ones(n)
    tri = np.maximum(tri, 0)
    tri = tri / tri.sum()
    vals = total * tri
    return pd.Series(vals, index=shape_hours)

# ------------- loaders ----------------

def load_assets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"precinct_id") or "precinct_id"): "precinct_id",
        (ncol(df,"county") or "county"): "county",
        (ncol(df,"site_name") or "site_name"): "site_name",
        (ncol(df,"voters_registered") or "voters_registered"): "voters_registered",
        (ncol(df,"is_vote_center") or "is_vote_center"): "is_vote_center",
        (ncol(df,"chk_in_stations") or "chk_in_stations"): "chk_in_stations",
        (ncol(df,"chk_in_thr_per_hr") or "chk_in_thr_per_hr"): "chk_in_thr_per_hr",
        (ncol(df,"chk_in_offline_ok") or "chk_in_offline_ok"): "chk_in_offline_ok",
        (ncol(df,"ballot_printers") or "ballot_printers"): "ballot_printers",
        (ncol(df,"bmds") or "bmds"): "bmds",
        (ncol(df,"scanner_units") or "scanner_units"): "scanner_units",
        (ncol(df,"issue_thr_per_hr") or "issue_thr_per_hr"): "issue_thr_per_hr",
        (ncol(df,"scan_thr_per_hr") or "scan_thr_per_hr"): "scan_thr_per_hr",
        (ncol(df,"backup_power") or "backup_power"): "backup_power",
        (ncol(df,"network_redundant") or "network_redundant"): "network_redundant",
        (ncol(df,"paper_pollbooks") or "paper_pollbooks"): "paper_pollbooks",
        (ncol(df,"vendor_epoll") or "vendor_epoll"): "vendor_epoll",
        (ncol(df,"vendor_ballot") or "vendor_ballot"): "vendor_ballot",
        (ncol(df,"opens") or "opens"): "opens",
        (ncol(df,"closes") or "closes"): "closes",
    }
    df = df.rename(columns=ren)
    # types/defaults
    df["precinct_id"] = df["precinct_id"].astype(str)
    for c in ["voters_registered","chk_in_stations","ballot_printers","bmds","scanner_units"]:
        if c in df.columns: df[c] = safe_num(df[c]).fillna(0).astype(int)
    for c in ["chk_in_thr_per_hr","issue_thr_per_hr","scan_thr_per_hr"]:
        if c in df.columns: df[c] = safe_num(df[c]).fillna(np.nan)
    for c in ["is_vote_center","chk_in_offline_ok","backup_power","network_redundant","paper_pollbooks"]:
        if c in df.columns: df[c] = safe_num(df[c]).fillna(0).astype(int)
    for c in ["vendor_epoll","vendor_ballot","county","site_name"]:
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
    # hours
    if "opens" in df.columns:
        df["opens"] = pd.to_datetime(df["opens"], errors="coerce").dt.time
    else:
        df["opens"] = pd.to_datetime("07:00").time()
    if "closes" in df.columns:
        df["closes"] = pd.to_datetime(df["closes"], errors="coerce").dt.time
    else:
        df["closes"] = pd.to_datetime("20:00").time()
    # default throughputs if missing (per station per hour)
    df["chk_in_thr_per_hr"] = df["chk_in_thr_per_hr"].fillna(15.0)  # ~4 min per check-in
    df["issue_thr_per_hr"]  = df["issue_thr_per_hr"].fillna(20.0)   # printing/issue
    df["scan_thr_per_hr"]   = df["scan_thr_per_hr"].fillna(30.0)   # scanner throughput
    return df

def load_deps(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["src","dst","type"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"src") or "src"):"src", (ncol(df,"dst") or "dst"):"dst", (ncol(df,"type") or "type"):"type"}
    df = df.rename(columns=ren)
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    df["type"] = df["type"].astype(str).str.lower()
    return df

def load_staffing(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["precinct_id","pollworkers","techs"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"precinct_id") or "precinct_id"):"precinct_id",
           (ncol(df,"pollworkers") or "pollworkers"):"pollworkers",
           (ncol(df,"techs") or "techs"):"techs"}
    df = df.rename(columns=ren)
    df["precinct_id"] = df["precinct_id"].astype(str)
    for c in ["pollworkers","techs"]: df[c] = safe_num(df[c]).fillna(0).astype(int)
    return df

def load_turnout(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    # support both wide and long
    if ncol(df,"hour") is not None and ncol(df,"arrivals") is not None:
        ren = {(ncol(df,"precinct_id") or "precinct_id"):"precinct_id",
               (ncol(df,"hour") or "hour"):"hour",
               (ncol(df,"arrivals") or "arrivals"):"arrivals"}
        df = df.rename(columns=ren)
        df["precinct_id"] = df["precinct_id"].astype(str)
        df["hour"] = safe_num(df["hour"]).astype(int)
        df["arrivals"] = safe_num(df["arrivals"]).fillna(0.0)
        return df[["precinct_id","hour","arrivals"]]
    # else assume wide: precinct rows, hour_07, hour_08...
    if ncol(df,"precinct_id"):
        df = df.rename(columns={ncol(df,"precinct_id"):"precinct_id"})
        df["precinct_id"] = df["precinct_id"].astype(str)
        hours = []
        for c in df.columns:
            if c == "precinct_id": continue
            if any(k in str(c).lower() for k in ["hour_", "h", "hr"]):
                hours.append(c)
        if not hours:
            # maybe columns 7..20 are hours
            hours = [c for c in df.columns if str(c).isdigit()]
        rows = []
        for _, r in df.iterrows():
            for c in hours:
                try:
                    h = int(''.join([ch for ch in str(c) if ch.isdigit()])[:2])
                except Exception:
                    continue
                rows.append({"precinct_id": str(r["precinct_id"]), "hour": h, "arrivals": float(safe_num(pd.Series([r[c]])).iloc[0])})
        return pd.DataFrame(rows)
    return pd.DataFrame()

def load_incidents(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"precinct_id") or "precinct_id"):"precinct_id",
           (ncol(df,"date") or "date"):"date",
           (ncol(df,"type") or "type"):"type",
           (ncol(df,"severity") or "severity"):"severity"}
    df = df.rename(columns=ren)
    df["precinct_id"] = df["precinct_id"].astype(str)
    df["severity"] = safe_num(df["severity"]).fillna(1.0)
    return df

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"):"scenario",
           (ncol(df,"key") or "key"):"key",
           (ncol(df,"value") or "value"):"value"}
    df = df.rename(columns=ren)
    return df

# ------------- core calculations ----------------

def hours_open(opens_t, closes_t) -> List[int]:
    try:
        start = int(str(opens_t).split(':')[0])
        end   = int(str(closes_t).split(':')[0])
    except Exception:
        start, end = 7, 20
    if end <= start:
        end = start + 13
    return list(range(start, end))

def site_capacities(row: pd.Series) -> Dict[str, float]:
    """
    Per-hour capacity per stage and bottleneck.
    """
    chk = float(row["chk_in_stations"] * row["chk_in_thr_per_hr"])
    issue_units = max(int(row.get("ballot_printers", 0)) + int(row.get("bmds", 0)), 1)
    issue = float(issue_units * row["issue_thr_per_hr"])
    scan = float(max(int(row.get("scanner_units", 0)), 1) * row["scan_thr_per_hr"])
    bottleneck = min(chk, issue, scan)
    return {"cap_chk": chk, "cap_issue": issue, "cap_scan": scan, "cap_bottleneck": bottleneck}

def build_baseline_capacity(ASSETS: pd.DataFrame, TURNOUT: pd.DataFrame, STAFF: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ASSETS.iterrows():
        prec = r["precinct_id"]
        caps = site_capacities(r)
        hours = hours_open(r.get("opens"), r.get("closes"))
        # staffing adjustments (simple): +5% capacity per extra tech beyond 1 (cap at +15%)
        if not STAFF.empty:
            st = STAFF[STAFF["precinct_id"]==prec]
            if not st.empty:
                techs = int(st["techs"].iloc[0] or 0)
                bump = min(0.05*max(0, techs-1), 0.15)
                for k in caps:
                    caps[k] *= (1.0 + bump)
        # hourly arrivals
        if not TURNOUT.empty:
            tt = TURNOUT[TURNOUT["precinct_id"]==prec].copy()
            if tt.empty:
                total = float(r.get("voters_registered", 0)) * 0.6  # assume 60% turnout if unknown
                arr = triangular_load(hours, total)
            else:
                arr = (tt.groupby("hour")["arrivals"].sum()).reindex(hours).fillna(0.0)
        else:
            total = float(r.get("voters_registered", 0)) * 0.6
            arr = triangular_load(hours, total)

        # compute queueing waits on *bottleneck* station using λ peak
        lam_peak = float(arr.max())
        # choose c and μ from bottleneck component
        # (we approximate by mapping which stage is bottleneck)
        stage_caps = { "chk": ("chk_in_stations", "chk_in_thr_per_hr"),
                       "issue": ("ballot_printers", "issue_thr_per_hr"),
                       "scan": ("scanner_units", "scan_thr_per_hr") }
        # Determine stage with minimal capacity
        stage_cap_vals = {"chk": caps["cap_chk"], "issue": caps["cap_issue"], "scan": caps["cap_scan"]}
        bottleneck_stage = min(stage_cap_vals, key=stage_cap_vals.get)
        servers = int(r[stage_caps[bottleneck_stage][0]]) if stage_caps[bottleneck_stage][0] in r else 1
        if bottleneck_stage == "issue":
            servers = max(int(r.get("ballot_printers", 0)) + int(r.get("bmds", 0)), 1)
        mu = float(r[stage_caps[bottleneck_stage][1]])
        Wq_min, W_min = mmc_wait_minutes(lam_peak, mu, max(servers,1))
        rows.append({
            "precinct_id": prec,
            "county": r.get("county"),
            "site_name": r.get("site_name"),
            "is_vote_center": int(r.get("is_vote_center",0)),
            "hours_open": len(hours),
            "peak_hour": int(arr.idxmax()) if len(arr)>0 else np.nan,
            "arrivals_total": float(arr.sum()),
            "lambda_peak_per_hr": lam_peak,
            "cap_chk_per_hr": caps["cap_chk"],
            "cap_issue_per_hr": caps["cap_issue"],
            "cap_scan_per_hr": caps["cap_scan"],
            "cap_bottleneck_per_hr": caps["cap_bottleneck"],
            "bottleneck_stage": bottleneck_stage,
            "servers_bottleneck": int(servers),
            "mu_bottleneck_per_hr": float(mu),
            "wait_q_peak_min": float(Wq_min),
            "wait_total_peak_min": float(W_min)
        })
    return pd.DataFrame(rows)

def risk_scores(ASSETS: pd.DataFrame, DEPS: pd.DataFrame, INCIDENTS: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic composite risk per site:
      power_risk     = 1 if no backup_power else 0.2
      network_risk   = 0.7 if no redundancy else 0.2
      vendor_risk    = 0.6 if vendor_epoll set (dependency) else 0.3
      single_point   = 1 - min(1, log(1+stations)/log(1+target)) per stage
      history_bonus  = min(1, 0.1*severity_sum)  (incidents)
    Composite = w * components (weights below).
    """
    weights = {"power": 0.25, "network": 0.20, "vendor": 0.20, "single_point": 0.20, "history": 0.15}
    rows = []
    # precompute incident severities per site
    sev_map = {}
    if not INCIDENTS.empty:
        sev_map = INCIDENTS.groupby("precinct_id")["severity"].sum().to_dict()
    # dependency amplification: count distinct upstream deps
    upstream = {}
    if not DEPS.empty:
        for p in ASSETS["precinct_id"].astype(str):
            # simple BFS upstream of precinct p
            seen = set(); frontier = [p]
            parents = set()
            while frontier:
                cur = frontier.pop()
                for _, r in DEPS[DEPS["dst"]==cur].iterrows():
                    src = str(r["src"])
                    if src not in seen:
                        seen.add(src); parents.add(src); frontier.append(src)
            upstream[p] = len(parents)
    for _, r in ASSETS.iterrows():
        pid = str(r["precinct_id"])
        power_risk = 1.0 if int(r.get("backup_power",0))==0 else 0.2
        net_risk   = 0.7 if int(r.get("network_redundant",0))==0 else 0.2
        vend_risk  = 0.6 if str(r.get("vendor_epoll","")).strip() else 0.3
        # single point risk (fewer stations => higher risk)
        def spr(count, target=6):
            c = max(1, int(count or 0))
            return 1.0 - min(1.0, np.log(1+c)/np.log(1+target))
        sp = max(spr(r.get("chk_in_stations",1)), spr( (r.get("ballot_printers",0) or 0) + (r.get("bmds",0) or 0) ), spr(r.get("scanner_units",1)))
        hist = min(1.0, 0.1 * float(sev_map.get(pid, 0.0)))
        # dependency count bump
        dep_bump = 0.05 * float(upstream.get(pid, 0))
        comp = (weights["power"]*power_risk +
                weights["network"]*net_risk +
                weights["vendor"]*vend_risk +
                weights["single_point"]*sp +
                weights["history"]*hist)
        comp = min(1.0, comp + dep_bump)
        rows.append({
            "precinct_id": pid,
            "county": r.get("county"),
            "site_name": r.get("site_name"),
            "power_risk": power_risk,
            "network_risk": net_risk,
            "vendor_risk": vend_risk,
            "single_point_risk": sp,
            "history_risk": hist,
            "dependency_bump": dep_bump,
            "risk_composite_0to1": comp,
            "risk_score_0to100": comp*100.0
        })
    return pd.DataFrame(rows)

# ------------- scenarios ----------------

def apply_scenario_row(row: pd.Series, sc_keys: Dict[str, str]) -> Tuple[float, Dict[str,float], List[str]]:
    """
    Returns capacity_multiplier, stage_multipliers, notes
    """
    notes = []
    stage_mult = {"chk": 1.0, "issue": 1.0, "scan": 1.0}
    cap_mult = 1.0

    county_hit = str(sc_keys.get("power.outage.county","")).lower() in str(row.get("county","")).lower()
    power_share = float(sc_keys.get("power.outage.share", 1.0))
    if county_hit and int(row.get("backup_power",0))==0:
        cap_mult *= (1.0 - clip01(power_share))
        notes.append("power_outage_no_backup")

    # vendor e-pollbook outage
    vend = str(sc_keys.get("vendor.outage","")).strip()
    if vend and vend.lower() == str(row.get("vendor_epoll","")).strip().lower():
        degrade = float(sc_keys.get("vendor.degradation", 0.5))  # share of capacity retained if offline
        if int(row.get("chk_in_offline_ok",0))==1 or int(row.get("paper_pollbooks",0))==1:
            stage_mult["chk"] *= degrade
            notes.append("epoll_outage_offline_degrade")
        else:
            stage_mult["chk"] *= 0.0
            notes.append("epoll_outage_no_fallback")

    # staffing shortfall
    short_pct = float(sc_keys.get("staffing.shortfall.pct", 0.0))/100.0
    if short_pct > 0:
        stage_mult = {k: v*(1.0 - clip01(short_pct)) for k,v in stage_mult.items()}
        notes.append("staffing_shortfall")

    # equipment failures
    for comp, key in [("chk","equipment.fail.chk_in.pct"),
                      ("scan","equipment.fail.scanner.pct"),
                      ("issue","equipment.fail.issue.pct")]:
        pct = float(sc_keys.get(key, 0.0))/100.0
        if pct > 0:
            stage_mult[comp] *= (1.0 - clip01(pct))
            notes.append(f"{comp}_equipment_failure")

    # weather impact by county
    weather_cty = str(sc_keys.get("weather.impact.county","")).strip().lower()
    if weather_cty and weather_cty == str(row.get("county","")).strip().lower():
        stage_mult = {k: v*0.85 for k,v in stage_mult.items()}  # generic 15% degrade
        notes.append("weather_impact")

    return cap_mult, stage_mult, notes

def run_scenarios(ASSETS: pd.DataFrame, BASE: pd.DataFrame, TURNOUT: pd.DataFrame, SCEN: pd.DataFrame) -> pd.DataFrame:
    if SCEN.empty:
        return pd.DataFrame(columns=["scenario","precinct_id","cap_bottleneck_per_hr_new","wait_total_peak_min_new","delta_wait_min","voters_at_risk"])
    out_rows = []
    # construct scenario dicts
    for sc_name, g in SCEN.groupby("scenario"):
        keys = { str(k): str(v) for k, v in zip(g["key"], g["value"]) }
        # apply per site
        for _, site in ASSETS.iterrows():
            pid = site["precinct_id"]
            # base row
            base = BASE[BASE["precinct_id"]==pid]
            if base.empty: continue
            base = base.iloc[0]
            cap_mult, stage_mult, notes = apply_scenario_row(site, keys)
            # recompute stage capacities
            caps = site_capacities(site)
            caps["cap_chk"]   *= stage_mult["chk"]
            # issue stage uses printers+BMDs; degrade via issue key
            caps["cap_issue"] *= stage_mult["issue"]
            caps["cap_scan"]  *= stage_mult["scan"]
            caps["cap_bottleneck"] = min(caps["cap_chk"], caps["cap_issue"], caps["cap_scan"]) * cap_mult

            # arrivals (same as baseline)
            hours = hours_open(site.get("opens"), site.get("closes"))
            if not TURNOUT.empty:
                tt = TURNOUT[TURNOUT["precinct_id"]==pid]
                arr = (tt.groupby("hour")["arrivals"].sum()).reindex(hours).fillna(0.0) if not tt.empty else triangular_load(hours, float(site.get("voters_registered",0))*0.6)
            else:
                arr = triangular_load(hours, float(site.get("voters_registered",0))*0.6)

            lam_peak = float(arr.max())
            # Reconstruct servers & mu for bottleneck stage under multipliers
            stage_vals = {"chk": caps["cap_chk"], "issue": caps["cap_issue"], "scan": caps["cap_scan"]}
            bstage = min(stage_vals, key=stage_vals.get)
            if bstage == "chk":
                servers = max(1, int(site.get("chk_in_stations",1)))
                mu = float(site.get("chk_in_thr_per_hr",15.0)) * stage_mult["chk"] * cap_mult
            elif bstage == "issue":
                servers = max(1, int(site.get("ballot_printers",0)) + int(site.get("bmds",0)))
                mu = float(site.get("issue_thr_per_hr",20.0)) * stage_mult["issue"] * cap_mult
            else:
                servers = max(1, int(site.get("scanner_units",1)))
                mu = float(site.get("scan_thr_per_hr",30.0)) * stage_mult["scan"] * cap_mult
            Wq_new, W_new = mmc_wait_minutes(lam_peak, mu, servers)
            # voters potentially facing >30 min waits at peak (rough proxy)
            voters_risk = float(arr.max() * (1.0 if W_new>30 else 0.0))
            out_rows.append({
                "scenario": sc_name,
                "precinct_id": pid,
                "county": site.get("county"),
                "notes": "|".join(notes),
                "cap_bottleneck_per_hr_base": float(base["cap_bottleneck_per_hr"]),
                "cap_bottleneck_per_hr_new": float(caps["cap_bottleneck"]),
                "wait_total_peak_min_base": float(base["wait_total_peak_min"]),
                "wait_total_peak_min_new": float(W_new),
                "delta_wait_min": float(W_new - base["wait_total_peak_min"]),
                "voters_at_risk_peak": voters_risk
            })
    return pd.DataFrame(out_rows)

# ------------- mitigations / playbook ----------------

def playbook(BASE: pd.DataFrame, RISK: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a generic mitigation list per precinct with a simple cost-benefit heuristic.
    """
    rows = []
    for _, b in BASE.iterrows():
        pid = b["precinct_id"]
        r = RISK[RISK["precinct_id"]==pid]
        risk = float(r["risk_composite_0to1"].iloc[0]) if not r.empty else 0.0
        # Candidate mitigations
        cands = [
            ("Add 1 check-in station", "equipment", 2500, 0.20, "chk"),
            ("Deploy backup power (battery/generator)", "power", 1800, 0.25, "cap"),
            ("Enable paper pollbooks / offline mode drills", "procedure", 300, 0.15, "chk"),
            ("Add 1 scanner/tabulator", "equipment", 3500, 0.15, "scan"),
            ("Extend hours by 60 minutes", "policy", 0, 0.10, "cap"),
            ("Reallocate 2 poll workers from low-load site", "staffing", 0, 0.08, "cap"),
        ]
        for name, typ, cost, benefit, comp in cands:
            # crude benefit scaling by risk & bottleneck match
            match_bonus = 0.15 if comp == b["bottleneck_stage"] or comp=="cap" else 0.0
            eff = benefit + match_bonus
            score = (risk * eff) / (1.0 + cost/5000.0)
            rows.append({
                "precinct_id": pid, "mitigation": name, "type": typ,
                "est_cost_usd": cost, "benefit_score_0to1": eff,
                "priority_score": score,
                "bottleneck_stage": b["bottleneck_stage"],
                "wait_total_peak_min": float(b["wait_total_peak_min"])
            })
    pb = pd.DataFrame(rows)
    return (pb.sort_values(["priority_score","benefit_score_0to1"], ascending=False)
              .groupby("precinct_id").head(5).reset_index(drop=True))

# ------------- CLI / Orchestration ----------------

@dataclass
class Config:
    assets: str
    dependencies: Optional[str]
    staffing: Optional[str]
    turnout: Optional[str]
    incidents: Optional[str]
    scenarios: Optional[str]
    scenario: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Election infrastructure risk/capacity & scenario engine")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--dependencies", default="")
    ap.add_argument("--staffing", default="")
    ap.add_argument("--turnout", default="")
    ap.add_argument("--incidents", default="")
    ap.add_argument("--scenarios", default="", help="CSV of scenario key/values")
    ap.add_argument("--scenario", default="", help="Scenario name to run from scenarios.csv (if omitted, run all)")
    ap.add_argument("--outdir", default="out_election")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    ASSETS = load_assets(args.assets)
    DEPS   = load_deps(args.dependencies)
    STAFF  = load_staffing(args.staffing)
    TO     = load_turnout(args.turnout)
    INCI   = load_incidents(args.incidents)
    SCEN   = load_scenarios(args.scenarios)

    # Baseline capacity & waits
    BASE = build_baseline_capacity(ASSETS, TO, STAFF)
    BASE.to_csv(outdir / "baseline_capacity.csv", index=False)

    # Risk
    RISK = risk_scores(ASSETS, DEPS, INCI)
    RISK.to_csv(outdir / "risk_scores.csv", index=False)

    # Scenarios
    if not SCEN.empty:
        if args.scenario:
            SCEN = SCEN[SCEN["scenario"].astype(str)==args.scenario]
        IMP = run_scenarios(ASSETS, BASE, TO, SCEN)
        if not IMP.empty:
            IMP.to_csv(outdir / "scenario_impacts.csv", index=False)
    else:
        IMP = pd.DataFrame()

    # Playbook
    PB = playbook(BASE, RISK)
    PB.to_csv(outdir / "playbook.csv", index=False)

    # Summary
    top_risk = RISK.sort_values("risk_composite_0to1", ascending=False).head(10)
    long_waits = BASE.sort_values("wait_total_peak_min", ascending=False).head(10)
    summary = {
        "sites": int(ASSETS.shape[0]),
        "baseline": {
            "median_wait_peak_min": float(BASE["wait_total_peak_min"].median()),
            "sites_wait_gt_30min": int((BASE["wait_total_peak_min"]>30).sum()),
            "max_wait_site": {
                "precinct_id": str(long_waits.iloc[0]["precinct_id"]) if not long_waits.empty else None,
                "wait_total_peak_min": float(long_waits.iloc[0]["wait_total_peak_min"]) if not long_waits.empty else None
            } if not long_waits.empty else None
        },
        "risk": {
            "median_risk_score": float(RISK["risk_score_0to100"].median()),
            "top_risk_sites": top_risk[["precinct_id","site_name","risk_score_0to100"]].to_dict(orient="records")
        },
        "scenario_run": args.scenario or ("ALL" if not SCEN.empty else None),
        "scenario_summary": (
            IMP.groupby("scenario")["delta_wait_min"].median().to_dict()
            if not IMP.empty else {}
        )
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config
    cfg = asdict(Config(
        assets=args.assets, dependencies=(args.dependencies or None), staffing=(args.staffing or None),
        turnout=(args.turnout or None), incidents=(args.incidents or None),
        scenarios=(args.scenarios or None), scenario=(args.scenario or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Election Infrastructure ==")
    print(f"Sites: {ASSETS.shape[0]} | Median peak wait: {summary['baseline']['median_wait_peak_min']:.1f} min | ≥30 min: {summary['baseline']['sites_wait_gt_30min']}")
    if args.scenario and not SCEN.empty:
        med = IMP[IMP["scenario"]==args.scenario]["delta_wait_min"].median() if not IMP.empty else 0.0
        print(f"Scenario '{args.scenario}': median Δwait {med:+.1f} min across sites")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
