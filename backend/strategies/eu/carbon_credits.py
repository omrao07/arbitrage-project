#!/usr/bin/env python3
"""
carbon_credits.py — Portfolio analytics for voluntary carbon credits (VCC)

What it does
------------
- Loads registry/projects, ratings, quotes, and trades
- Builds current positions (by project & bucket), mark-to-market, and P&L
- Estimates fair value using a transparent *hedonic* model (type/country/vintage/quality)
- Computes quality score (additionality, permanence, leakage, MRV, co-benefits, governance)
- Aggregates exposures by project type, country, methodology, registry, and vintage buckets
- Produces forward curve (if quotes have tenor), and liquidity metrics
- Scenario shocks (policy/quality/vintage) and haircuts to revalue portfolio
- Risk concentration metrics (HHI), and holdings/retirements reconciliation

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--projects projects.csv
    Columns (recommended):
      project_id, registry, methodology, type, country, vintage_start, vintage_end,
      tonnes_issued, tonnes_remaining, buffer_frac
--ratings ratings.csv (optional but recommended)
    project_id, additionality, permanence, leakage, mrv, co_benefits, governance   # scores 0..5
--quotes quotes.csv (optional)
    date, project_id (or type), price_usd, volume, tenor_months, delivery
--trades trades.csv (optional; portfolio)
    date, project_id, side(buy/sell/retire), tonnes, price_usd, account

Key options
-----------
--asof 2025-09-06               Valuation date (default: today)
--vintage-buckets "≤2012,2013-2015,2016-2018,2019-2021,2022+,CORSIA"
--quality-weights "add:0.35,perm:0.2,leak:-0.1,mrv:0.15,coben:0.1,gov:0.1,age:-0.1"
--scen "low_quality:-0.15,policy_uplift:+0.08,nature_risk:-0.1,vintage_penalty:-0.05"
--haircut-liquidity 0.05        Extra haircut applied where no direct quote
--outdir out_carbon

Outputs
-------
- holdings.csv           Current positions by project (tonnes, cost, MV, P&L)
- mtm_by_bucket.csv      Aggregates by Type/Country/Methodology/Vintage/Registry
- quality_scores.csv     Project-level quality scores & components
- quotes_latest.csv      Latest quotes resolved per project and fallbacks
- forward_curve.csv      Tenor-wise average prices & volumes
- scenario_valuation.csv Revaluation table under scenarios
- summary.json           Headline KPIs (MV, P&L, exposures, HHI)
- config.json            Reproducibility dump

Notes
-----
- If a project has no direct quote, the script falls back to (type,country,vintage-bucket) averages,
  then to (type) averages, then to overall mean — applying a configurable *liquidity haircut*.
- Quality score ∈ [-1, +1] after weighting & normalization; used for optional hedonic uplift/discount.
- Vintage "age" penalty uses mid-vintage year vs asof year.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------- Helpers -----------------
def norm_col(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None


def read_csv_flex(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


def to_date(s):
    return pd.to_datetime(s, errors="coerce")


def parse_pairlist(s: str, aliases: Dict[str, str] = None) -> Dict[str, float]:
    """
    Parse "add:0.35,perm:0.2,leak:-0.1,..." into dict.
    """
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip().lower()
        if aliases and k in aliases:
            k = aliases[k]
        try:
            out[k] = float(v.strip().replace("+", ""))
        except ValueError:
            pass
    return out


def parse_spanlist(s: str) -> List[str]:
    """
    Example: "≤2012,2013-2015,2016-2018,2019-2021,2022+,CORSIA"
    """
    if not s:
        return ["≤2012", "2013-2015", "2016-2018", "2019-2021", "2022+"]
    return [x.strip() for x in s.split(",") if x.strip()]


def bucket_vintage(mid_year: float, buckets: List[str]) -> str:
    if pd.isna(mid_year):
        return "Unknown"
    for b in buckets:
        s = b.replace(" ", "")
        if s.startswith("≤"):
            y = int(s[1:])
            if mid_year <= y:
                return b
        elif s.endswith("+"):
            y = int(s[:-1])
            if mid_year >= y:
                return b
        elif "-" in s:
            a, c = s.split("-", 1)
            if int(a) <= mid_year <= int(c):
                return b
        elif s.upper() == "CORSIA":
            # Tagging handled elsewhere; keep bucket for summary grouping only if set by user
            continue
    return f"{int(mid_year)}"


def hhi(weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).values
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float(np.sum(w**2))


# ----------------- Quality score -----------------
def build_quality_scores(projects: pd.DataFrame, ratings: Optional[pd.DataFrame], asof: pd.Timestamp,
                         weights: Dict[str, float]) -> pd.DataFrame:
    # Prepare ratings table
    if ratings is None or ratings.empty:
        # Create zeros to avoid NaNs (neutral)
        rat = pd.DataFrame({"project_id": projects["project_id"].astype(str).unique()})
    else:
        rat = ratings.copy()
        if norm_col(rat, "project_id"):
            rat = rat.rename(columns={norm_col(rat, "project_id"): "project_id"})
        else:
            raise SystemExit("ratings.csv must contain project_id.")
    # Normalize important fields to 0..1
    # Map aliases
    rename_map = {
        (norm_col(rat, "additionality") or "additionality"): "add",
        (norm_col(rat, "permanence") or "permanence"): "perm",
        (norm_col(rat, "leakage") or "leakage"): "leak",
        (norm_col(rat, "mrv") or "mrv"): "mrv",
        (norm_col(rat, "co_benefits") or norm_col(rat, "coben") or "co_benefits"): "coben",
        (norm_col(rat, "governance") or "governance"): "gov",
    }
    rat = rat.rename(columns={k: v for k, v in rename_map.items() if k in rat.columns})
    for k in ["add", "perm", "leak", "mrv", "coben", "gov"]:
        if k not in rat.columns:
            rat[k] = np.nan
        rat[k] = pd.to_numeric(rat[k], errors="coerce")
        # If on a 0..5 scale, rescale to 0..1; otherwise assume already 0..1
        mx = np.nanmax(rat[k].values) if rat[k].notna().any() else 1.0
        if mx > 1.5:
            rat[k] = rat[k] / 5.0
    # Age (vintage) penalty feature
    pr = projects[["project_id", "vintage_start", "vintage_end"]].copy()
    pr["mid_year"] = (pd.to_numeric(pr["vintage_start"], errors="coerce") + pd.to_numeric(pr["vintage_end"], errors="coerce")) / 2.0
    age = (asof.year - pr["mid_year"]).clip(lower=0)
    rat = pr[["project_id", "mid_year"]].merge(rat, on="project_id", how="left")
    rat["age_years"] = age

    # Weighted score in [-1, +1]
    w = {
        "add": weights.get("add", 0.35),
        "perm": weights.get("perm", 0.2),
        "leak": weights.get("leak", -0.1),     # negative if more leakage -> worse
        "mrv": weights.get("mrv", 0.15),
        "coben": weights.get("coben", 0.1),
        "gov": weights.get("gov", 0.1),
        "age": weights.get("age", -0.1),       # older vintages discounted
    }
    # Normalize inputs: center around 0.5 (neutral) so score can be negative for <0.5
    def center(x): return (x.fillna(0.5) - 0.5) / 0.5  # maps 0..1 to -1..+1
    score = (
        w["add"] * center(rat["add"]) +
        w["perm"] * center(rat["perm"]) +
        w["leak"] * center(1 - rat["leak"].fillna(0.5)) +  # invert leakage: higher leakage is worse
        w["mrv"] * center(rat["mrv"]) +
        w["coben"] * center(rat["coben"]) +
        w["gov"] * center(rat["gov"]) +
        w["age"] * (-(rat["age_years"].fillna(0) / 12.0).clip(-2, 2))  # -1 per 12y, capped
    )
    rat["quality_score"] = score.clip(-1, 1)
    return rat


# ----------------- Pricing -----------------
def latest_quotes(quotes: Optional[pd.DataFrame], asof: pd.Timestamp) -> pd.DataFrame:
    if quotes is None or quotes.empty:
        return pd.DataFrame(columns=["date", "project_id", "type", "country", "vintage_bucket", "price_usd", "volume", "tenor_months"])
    q = quotes.copy()
    # normalize columns
    for col, std in [
        ("date", "date"), ("project_id", "project_id"), ("type", "type"),
        ("country", "country"), ("price_usd", "price_usd"), ("volume", "volume"),
        ("tenor_months", "tenor_months"), ("delivery", "delivery"),
    ]:
        c = norm_col(q, col)
        if c:
            q = q.rename(columns={c: std})
    q["date"] = to_date(q["date"]).fillna(to_date(q.get("delivery", None)))
    q = q[q["date"] <= asof].copy()
    # take last by project_id; if not present, keep type-level quotes
    # Prepare latest per key
    def last_by(key_cols):
        return (q.sort_values(["date"])
                .groupby(key_cols)
                .tail(1))
    latest_proj = last_by(["project_id"]) if "project_id" in q.columns else q.iloc[0:0]
    latest_type = last_by(["type", "country"]).query("project_id.isna() or project_id==project_id") if "type" in q.columns else q.iloc[0:0]
    latest = pd.concat([latest_proj, latest_type], ignore_index=True)
    # Forward curve from tenor
    curve = (q.dropna(subset=["tenor_months"])
             .groupby("tenor_months")
             .agg(avg_price=("price_usd", "mean"), total_vol=("volume", "sum"))
             .reset_index().sort_values("tenor_months"))
    return latest, curve


def resolve_prices(projects: pd.DataFrame, ratings: pd.DataFrame, quotes_latest: pd.DataFrame,
                   buckets: List[str], liquidity_haircut: float) -> pd.DataFrame:
    pr = projects.copy()
    pr["project_id"] = pr["project_id"].astype(str)
    # Harmonize attributes
    pr = pr.rename(columns={
        (norm_col(pr, "registry") or "registry"): "registry",
        (norm_col(pr, "methodology") or "methodology"): "methodology",
        (norm_col(pr, "type") or "type"): "type",
        (norm_col(pr, "country") or "country"): "country",
        (norm_col(pr, "vintage_start") or "vintage_start"): "vintage_start",
        (norm_col(pr, "vintage_end") or "vintage_end"): "vintage_end",
        (norm_col(pr, "buffer_frac") or "buffer_frac"): "buffer_frac",
        (norm_col(pr, "tonnes_remaining") or "tonnes_remaining"): "tonnes_remaining",
    })
    # Vintage bucket
    mid = (pd.to_numeric(pr["vintage_start"], errors="coerce") + pd.to_numeric(pr["vintage_end"], errors="coerce")) / 2.0
    pr["vintage_mid"] = mid
    pr["vintage_bucket"] = [bucket_vintage(x, buckets) for x in mid.fillna(0)]
    # Join quality
    pr = pr.merge(ratings[["project_id", "quality_score"]], on="project_id", how="left")
    # Attach best quote: project -> (type,country) -> (type) -> global mean
    q = quotes_latest.copy()
    # Prepare fallbacks
    project_price = q.dropna(subset=["project_id"])[["project_id", "price_usd"]].set_index("project_id")["price_usd"]
    type_country_price = q.dropna(subset=["type", "country"]).groupby(["type", "country"])["price_usd"].mean()
    type_price = q.dropna(subset=["type"]).groupby("type")["price_usd"].mean()
    global_mean = float(q["price_usd"].mean()) if "price_usd" in q.columns and not q["price_usd"].empty else np.nan

    resolved = []
    for _, row in pr.iterrows():
        pid = row["project_id"]
        typ = str(row.get("type", ""))
        ctry = str(row.get("country", ""))
        price = np.nan
        src = "none"
        if pid in project_price.index:
            price = float(project_price.loc[pid]); src = "project"
        elif (typ, ctry) in type_country_price.index:
            price = float(type_country_price.loc[(typ, ctry)]); src = "type_country"
        elif typ in type_price.index:
            price = float(type_price.loc[typ]); src = "type"
        else:
            price = global_mean; src = "global"
        # Liquidity haircut for non-project quotes
        if src != "project" and np.isfinite(price):
            price = price * (1.0 - float(liquidity_haircut))
        resolved.append((pid, price, src))
    rez = pd.DataFrame(resolved, columns=["project_id", "px_resolved", "px_source"])
    return pr.merge(rez, on="project_id", how="left")


# ----------------- Trades / positions -----------------
def build_positions(trades: Optional[pd.DataFrame], projects: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        # No portfolio: use projects' remaining as available inventory at zero cost
        return projects[["project_id", "tonnes_remaining"]].rename(columns={"tonnes_remaining": "tonnes"}).assign(
            cost_usd=0.0, side="inventory", avg_cost_usd=0.0
        )
    t = trades.copy()
    for col, std in [("date", "date"), ("project_id", "project_id"), ("side", "side"), ("tonnes", "tonnes"), ("price_usd", "price_usd"), ("account", "account")]:
        c = norm_col(t, col)
        if c:
            t = t.rename(columns={c: std})
    t["date"] = to_date(t["date"])
    t["side"] = t["side"].astype(str).str.lower().str.strip()
    t["tonnes"] = pd.to_numeric(t["tonnes"], errors="coerce").fillna(0.0)
    t["price_usd"] = pd.to_numeric(t.get("price_usd", 0.0), errors="coerce").fillna(0.0)
    # buys positive, sells/retire negative inventory
    sign = t["side"].map({"buy": 1, "b": 1, "sell": -1, "s": -1, "retire": -1}).fillna(0)
    t["qty"] = sign * t["tonnes"]
    pos = t.groupby("project_id").agg(tonnes=("qty", "sum"),
                                      cost_usd=("price_usd", lambda x: float(np.nan)),  # placeholder
                                      notional=("price_usd", lambda x: 0.0)).reset_index()
    # Average cost from trades (weighted by tonnes on buys only)
    buys = t[sign > 0].groupby("project_id").apply(lambda g: np.average(g["price_usd"], weights=g["tonnes"]) if g["tonnes"].sum() > 0 else np.nan)
    pos["avg_cost_usd"] = pos["project_id"].map(buys.to_dict()).fillna(0.0)
    return pos


# ----------------- Scenarios -----------------
@dataclass
class Scenarios:
    low_quality: float = -0.15     # apply to projects with quality_score < 0 (discount)
    policy_uplift: float = 0.0     # apply to CORSIA-eligible types, if identifiable by methodology/type keywords
    nature_risk: float = 0.0       # apply to Nature-based types (e.g., Forestry/REDD+)
    vintage_penalty: float = 0.0   # apply to vintages before 2016 (example)


def apply_scenarios(holdings: pd.DataFrame, scen: Scenarios) -> pd.DataFrame:
    df = holdings.copy()
    adj = np.ones(len(df))
    # low quality discount
    adj *= np.where(df["quality_score"] < 0, 1.0 + scen.low_quality, 1.0)
    # crude flags by type/methodology keywords
    mth = df.get("methodology", "").astype(str).str.upper()
    typ = df.get("type", "").astype(str).str.upper()
    corsia_flag = typ.str.contains("CORSIA") | mth.str.contains("CORSIA") | mth.str.contains("ACI|AER|CORSIA")
    nature_flag = typ.str.contains("FOREST|REDD|AFOLU|NATURE") | mth.str.contains("REDD|AFOLU|ARR")
    vintage_old = pd.to_numeric(df.get("vintage_start", np.nan), errors="coerce").fillna(0) < 2016
    adj *= np.where(corsia_flag, 1.0 + scen.policy_uplift, 1.0)
    adj *= np.where(nature_flag, 1.0 + scen.nature_risk, 1.0)
    adj *= np.where(vintage_old, 1.0 + scen.vintage_penalty, 1.0)

    df["px_scenario"] = df["px_resolved"] * adj
    df["mv_scenario"] = df["tonnes"].clip(lower=0) * df["px_scenario"]
    return df


# ----------------- CLI -----------------
@dataclass
class Config:
    projects: str
    ratings: Optional[str]
    quotes: Optional[str]
    trades: Optional[str]
    asof: str
    vintage_buckets: List[str]
    quality_weights: Dict[str, float]
    scen: Dict[str, float]
    haircut_liquidity: float
    outdir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Carbon credits portfolio analytics")
    ap.add_argument("--projects", required=True)
    ap.add_argument("--ratings", default="")
    ap.add_argument("--quotes", default="")
    ap.add_argument("--trades", default="")
    ap.add_argument("--asof", default="")
    ap.add_argument("--vintage-buckets", default="≤2012,2013-2015,2016-2018,2019-2021,2022+")
    ap.add_argument("--quality-weights", default="add:0.35,perm:0.2,leak:-0.1,mrv:0.15,coben:0.1,gov:0.1,age:-0.1")
    ap.add_argument("--scen", default="low_quality:-0.15,policy_uplift:+0.08,nature_risk:-0.10,vintage_penalty:-0.05")
    ap.add_argument("--haircut-liquidity", type=float, default=0.05)
    ap.add_argument("--outdir", default="out_carbon")
    return ap.parse_args()


# ----------------- Main -----------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    asof = pd.to_datetime(args.asof) if args.asof else pd.Timestamp.today().normalize()

    # Load inputs
    projects = read_csv_flex(args.projects)
    if projects is None or projects.empty:
        raise SystemExit("projects.csv is required and appears empty.")
    projects = projects.rename(columns={norm_col(projects, "project_id") or "project_id": "project_id"})
    ratings = read_csv_flex(args.ratings) if args.ratings else None
    quotes = read_csv_flex(args.quotes) if args.quotes else None
    trades = read_csv_flex(args.trades) if args.trades else None

    # Buckets & weights & scenarios
    buckets = parse_spanlist(args.vintage_buckets)
    w = parse_pairlist(args.quality_weights, aliases={"coben": "coben"})
    scen_map = parse_pairlist(args.scen)
    scen = Scenarios(
        low_quality=scen_map.get("low_quality", -0.15),
        policy_uplift=scen_map.get("policy_uplift", 0.0),
        nature_risk=scen_map.get("nature_risk", 0.0),
        vintage_penalty=scen_map.get("vintage_penalty", 0.0),
    )

    # Quality
    qtbl = build_quality_scores(projects, ratings, asof, w)
    qtbl.to_csv(outdir / "quality_scores.csv", index=False)

    # Quotes
    q_latest, fwd_curve = latest_quotes(quotes, asof)
    if isinstance(q_latest, tuple):
        # older signature guard
        pass
    (outdir / "quotes_latest.csv").write_text(q_latest.to_csv(index=False))

    # Resolve prices and build holdings
    prc = resolve_prices(projects, qtbl, q_latest, buckets, args.haircut_liquidity)

    pos = build_positions(trades, projects)
    # If both trades and projects remaining exist, join on project_id; default to trades if present
    hol = prc.merge(pos[["project_id", "tonnes", "avg_cost_usd"]], on="project_id", how="left")
    # If no trades, infer tonnes from projects.remaining
    hol["tonnes"] = hol["tonnes"].fillna(hol.get("tonnes_remaining", 0.0)).fillna(0.0)
    # Values
    hol["mv_usd"] = hol["tonnes"].clip(lower=0) * hol["px_resolved"]
    hol["pnl_usd"] = hol["tonnes"].clip(lower=0) * (hol["px_resolved"] - hol["avg_cost_usd"].fillna(0.0))

    # Aggregations and exposures
    hol["vintage_bucket"] = hol["vintage_bucket"].fillna("Unknown")
    hol["type"] = hol.get("type", "Unknown")
    hol["country"] = hol.get("country", "Unknown")
    hol["registry"] = hol.get("registry", "Unknown")
    hol["methodology"] = hol.get("methodology", "Unknown")

    hol_out_cols = ["project_id", "registry", "methodology", "type", "country",
                    "vintage_start", "vintage_end", "vintage_bucket",
                    "quality_score", "px_resolved", "px_source", "tonnes", "avg_cost_usd", "mv_usd", "pnl_usd"]
    hol[hol_out_cols].to_csv(outdir / "holdings.csv", index=False)

    grp_keys = [
        ["type"], ["country"], ["methodology"], ["registry"], ["vintage_bucket"],
        ["type", "country"], ["type", "vintage_bucket"]
    ]
    agg_rows = []
    for keys in grp_keys:
        g = hol.groupby(keys).agg(
            tonnes=("tonnes", "sum"),
            mv_usd=("mv_usd", "sum"),
            pnl_usd=("pnl_usd", "sum"),
            wavg_px=("px_resolved", lambda x: float(np.average(x, weights=hol.loc[x.index, "tonnes"].clip(lower=0))) if (hol.loc[x.index, "tonnes"].clip(lower=0).sum() > 0) else float(np.mean(x))),
            hhi_mv=("mv_usd", lambda x: hhi(x)),
            q_wavg=("quality_score", lambda x: float(np.average(x, weights=hol.loc[x.index, "tonnes"].clip(lower=0))) if (hol.loc[x.index, "tonnes"].clip(lower=0).sum() > 0) else float(np.nanmean(x))),
        ).reset_index()
        g["bucket"] = "+".join(keys)
        agg_rows.append(g)
    mtm_by_bucket = pd.concat(agg_rows, ignore_index=True)
    mtm_by_bucket.to_csv(outdir / "mtm_by_bucket.csv", index=False)

    # Forward curve
    if isinstance(fwd_curve, pd.DataFrame) and not fwd_curve.empty:
        fwd_curve.to_csv(outdir / "forward_curve.csv", index=False)

    # Scenario valuation
    scen_val = apply_scenarios(hol, scen)
    scen_val_out = scen_val[["project_id", "type", "country", "vintage_bucket", "tonnes", "px_resolved", "px_scenario", "mv_usd", "mv_scenario", "quality_score"]].copy()
    scen_val_out["delta_mv"] = scen_val_out["mv_scenario"] - scen_val_out["mv_usd"]
    scen_val_out.to_csv(outdir / "scenario_valuation.csv", index=False)

    # Summary
    mv_total = float(hol["mv_usd"].sum())
    pnl_total = float(hol["pnl_usd"].sum())
    exp_type = hol.groupby("type")["mv_usd"].sum().sort_values(ascending=False)
    exp_country = hol.groupby("country")["mv_usd"].sum().sort_values(ascending=False)
    exp_vintage = hol.groupby("vintage_bucket")["mv_usd"].sum().sort_values(ascending=False)
    summary = {
        "asof": str(asof.date()),
        "mv_total_usd": mv_total,
        "pnl_total_usd": pnl_total,
        "n_projects": int(hol["project_id"].nunique()),
        "quote_sources_share": hol["px_source"].value_counts(normalize=True).round(3).to_dict(),
        "hhi_type": hhi(exp_type),
        "hhi_country": hhi(exp_country),
        "hhi_vintage": hhi(exp_vintage),
        "top_types": exp_type.head(5).round(2).to_dict(),
        "top_countries": exp_country.head(5).round(2).to_dict(),
        "top_vintage_buckets": exp_vintage.head(5).round(2).to_dict(),
        "scenario": asdict(scen),
        "files": ["holdings.csv", "mtm_by_bucket.csv", "quality_scores.csv", "quotes_latest.csv", "forward_curve.csv", "scenario_valuation.csv"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "config.json").write_text(json.dumps({
        "projects": args.projects, "ratings": args.ratings, "quotes": args.quotes, "trades": args.trades,
        "asof": str(asof.date()),
        "vintage_buckets": buckets,
        "quality_weights": w,
        "scenarios": asdict(scen),
        "haircut_liquidity": args.haircut_liquidity,
    }, indent=2))

    # Console
    print("== Carbon Credits Portfolio ==")
    print(f"Asof: {summary['asof']}  MV: ${summary['mv_total_usd']:,.0f}  P&L: ${summary['pnl_total_usd']:,.0f}")
    print("Quote sources mix:", summary["quote_sources_share"])
    print("Top types:", summary["top_types"])
    print("Top countries:", summary["top_countries"])
    print("Top vintage buckets:", summary["top_vintage_buckets"])
    print(f"HHI(type)={summary['hhi_type']:.3f}  HHI(country)={summary['hhi_country']:.3f}  HHI(vintage)={summary['hhi_vintage']:.3f}")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
