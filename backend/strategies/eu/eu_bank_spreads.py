#!/usr/bin/env python3
"""
eu_bank_spreads.py — EU bank bond spread analytics (senior / covered / AT1)

What it does
------------
- Loads bond-level quotes (by ISIN) and benchmarks (govt or swaps) and computes:
    * Clean yield if only price/duration/coupon given (optional; otherwise use provided yield)
    * Z-spreads approximated as simple yield-over-benchmark: (y_bond - y_bench) in bp
    * Curve points by tenor (1–30Y), rolling stats, and outlier flags
    * Issuer / country / rating / tier aggregates (means, 10th/90th pctls)
    * DV01-weighted indices and spread changes (Δbp, z-score)
    * PCA on the cross-section of issuer curve spreads (level/slope/richness)
    * Stress table for user-specified parallel and slope shocks
- Optional CDS overlay to compare bond vs CDS basis
- Optional equity overlay to relate AT1 moves to equity drawdowns

Inputs
------
--bonds bonds.csv
    Required. Long- or wide-form quotes. Recommended columns (case-insensitive):
      date, isin, issuer, tier, country, rating, ccy, coupon, price, yield, duration, tenor_y, benchmark_key
    Notes:
      - date: YYYY-MM-DD
      - tier: one of {Senior, SNP, Covered, Tier2, AT1} (free text ok)
      - tenor_y: numeric years to maturity (or we infer from 'maturity' date if provided)
      - yield: clean yield in decimals (0.045 = 4.5%); if missing we'll try to infer from price/coupon (rough)
      - duration: Macaulay/Modified (we treat as modified); improves DV01 estimates
      - benchmark_key: joins to benchmark curve row (e.g., "EUR_SWAP", "DE_OIS", "OAT", etc.)

--bench bench.csv
    Required. Benchmark curve points per date. Columns:
      date, benchmark_key, tenor_y, yld   # yld in decimals (0.025 = 2.5%)

--cds cds.csv (optional)
    Columns: date, issuer, tenor_y, cds_bp

--equity equity.csv (optional)
    Columns: date, issuer, px (or ret), mktcap?  # used only for AT1 correlation snapshot

Key options
-----------
--tenors "1,2,3,5,7,10,15,20"   Tenor grid for bucketing & curve building
--window 60                     Rolling window for z-stats (trading days)
--base-ccy EUR                  Only for labeling; no FX math is done
--pca                           If set, runs PCA on issuer spread curves (per date)
--stresses "+25,+50,-25"        Parallel spread shocks in bp
--slope-stresses "+10x10-2s"    Slope shocks: +10bp @10y vs -2bp @2y etc. (comma-separated)
--outdir out_eu_banks

Outputs
-------
- bond_spreads.csv        Per ISIN & date with y_bond, y_bench, spread_bp, z, dv01, dv01_weight
- issuer_curve.csv        Issuer-tenor daily curve spreads (bp) on chosen grid
- aggregates.csv          Issuer / country / rating / tier aggregates (mean, p10, p90)
- index_dv01.csv          DV01-weighted indices by tier & rating bucket
- pca_components.csv      (if --pca) Level/Slope/Richness loadings per issuer and explained variance
- cds_basis.csv           (if cds) Bond-vs-CDS basis by issuer/tenor
- at1_equity_snap.csv     (if equity) AT1 spread vs equity return correlation snapshot
- stress_table.csv        Shocked index levels under parallel/slope stresses
- summary.json            Latest snapshot KPIs
- config.json             Reproducibility dump
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------- I/O helpers ----------------
def read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


def ncol(df: pd.DataFrame, name: str) -> Optional[str]:
    t = name.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None


def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


# -------------- Bond helpers --------------
def infer_tenor(df: pd.DataFrame) -> pd.Series:
    if ncol(df, "tenor_y"):
        return pd.to_numeric(df[ncol(df, "tenor_y")], errors="coerce")
    # try maturity
    mcol = ncol(df, "maturity")
    dcol = ncol(df, "date") or df.columns[0]
    if mcol:
        mat = to_date(df[mcol])
        dt = to_date(df[dcol])
        return (mat - dt).dt.days / 365.25
    return pd.Series(np.nan, index=df.index)


def approx_yield_from_price(price: float, coupon: float, tenor_y: float) -> Optional[float]:
    """
    Very rough yield approximation (annual-pay): y ≈ (coupon + (100 - price)/tenor) / ((100 + price)/2)
    price in clean price (per 100), coupon in % of par.
    """
    try:
        if any(pd.isna(x) for x in (price, coupon, tenor_y)) or tenor_y <= 0:
            return np.nan
        c = float(coupon) / 100.0
        p = float(price) / 100.0
        y = (c + (1.0 - p) / float(tenor_y)) / ((1.0 + p) / 2.0)
        return float(y)
    except Exception:
        return np.nan


def ensure_bond_yield(df: pd.DataFrame) -> pd.Series:
    ycol = ncol(df, "yield")
    if ycol:
        y = pd.to_numeric(df[ycol], errors="coerce")
        # Treat percentages if needed
        med = np.nanmedian(y.values)
        if np.isfinite(med) and med > 1.5:
            y = y / 100.0
        return y
    # try compute from price & coupon & tenor
    pcol = ncol(df, "price")
    ccol = ncol(df, "coupon")
    if pcol and ccol:
        tenor = infer_tenor(df)
        return pd.Series(
            [approx_yield_from_price(df[pcol].iat[i], df[ccol].iat[i], tenor.iat[i]) for i in range(len(df))],
            index=df.index,
        )
    raise SystemExit("Provide bond 'yield' or ('price' & 'coupon' & 'tenor/maturity').")


def dv01_from_duration(yield_dec: pd.Series, dur: pd.Series, notional: float = 100.0) -> pd.Series:
    """
    DV01 ≈ duration * price * 1bp. Using price≈par for simplicity → DV01 ≈ dur * notional * 0.0001
    If price available, you may multiply by price/100 for precision; here keep simple & consistent.
    """
    return pd.to_numeric(dur, errors="coerce").fillna(0.0) * (notional * 1e-4)


def coerce_tier(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.upper()
    # Simple normalization
    x = x.replace(
        {
            "SENIOR": "SENIOR",
            "SNP": "SNP",
            "SENIOR NON-PREFERRED": "SNP",
            "T2": "TIER2",
            "TIER2": "TIER2",
            "AT1": "AT1",
            "ADDITIONAL TIER 1": "AT1",
            "COVERED": "COVERED",
        }
    )
    return x


# -------------- Core computations --------------
def compute_spreads(bonds: pd.DataFrame, bench: pd.DataFrame, tenors: List[float], win: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Normalize columns
    dcol_b = ncol(bonds, "date") or bonds.columns[0]
    bonds = bonds.rename(
        columns={
            dcol_b: "date",
            (ncol(bonds, "isin") or "isin"): "isin",
            (ncol(bonds, "issuer") or "issuer"): "issuer",
            (ncol(bonds, "country") or "country"): "country",
            (ncol(bonds, "rating") or "rating"): "rating",
            (ncol(bonds, "ccy") or "ccy"): "ccy",
            (ncol(bonds, "tier") or "tier"): "tier",
            (ncol(bonds, "duration") or "duration"): "duration",
            (ncol(bonds, "benchmark_key") or "benchmark_key"): "benchmark_key",
        }
    )
    bonds["date"] = to_date(bonds["date"])
    bonds["tenor_y"] = infer_tenor(bonds)
    bonds["tier"] = coerce_tier(bonds.get("tier", pd.Series(index=bonds.index, dtype=str)).fillna("SENIOR"))

    # Ensure yields
    bonds["y_bond"] = ensure_bond_yield(bonds)

    # Bench
    bench = bench.rename(
        columns={
            (ncol(bench, "date") or bench.columns[0]): "date",
            (ncol(bench, "benchmark_key") or "benchmark_key"): "benchmark_key",
            (ncol(bench, "tenor_y") or "tenor_y"): "tenor_y",
            (ncol(bench, "yld") or ncol(bench, "yield") or "yld"): "y_bench",
        }
    )
    bench["date"] = to_date(bench["date"])
    bench["tenor_y"] = pd.to_numeric(bench["tenor_y"], errors="coerce")
    bench["y_bench"] = pd.to_numeric(bench["y_bench"], errors="coerce")

    # Interpolate benchmark to each bond's tenor per date/key
    def interp_group(g):
        # g: one benchmark_key × date group across tenors
        g = g.dropna(subset=["tenor_y", "y_bench"]).sort_values("tenor_y")
        return g

    bench_sorted = bench.groupby(["date", "benchmark_key"], group_keys=False).apply(interp_group)
    # prepare for merge_asof per (date, key)
    out_rows = []
    for (dt, key), g in bonds.groupby(["date", "benchmark_key"]):
        cur = g.copy()
        bcurve = bench_sorted[(bench_sorted["date"] == dt) & (bench_sorted["benchmark_key"] == key)]
        if bcurve.empty:
            cur["y_bench"] = np.nan
        else:
            # linear interpolation
            x = bcurve["tenor_y"].values
            y = bcurve["y_bench"].values
            t = cur["tenor_y"].astype(float).values
            yhat = np.interp(t, x, y, left=y[0], right=y[-1])
            cur["y_bench"] = yhat
        out_rows.append(cur)
    bonds2 = pd.concat(out_rows, ignore_index=True) if out_rows else bonds.assign(y_bench=np.nan)

    # Spreads (bp)
    bonds2["spread_bp"] = (bonds2["y_bond"] - bonds2["y_bench"]) * 1e4

    # DV01 estimate
    dur = pd.to_numeric(bonds2.get("duration", np.nan), errors="coerce")
    bonds2["dv01"] = dv01_from_duration(bonds2["y_bond"], dur)

    # Rolling z by ISIN (or issuer/tenor)
    bonds2 = bonds2.sort_values(["isin", "date"])
    def rolling_z(s):
        mu = s.rolling(win).mean()
        sd = s.rolling(win).std(ddof=1)
        return (s - mu) / (sd + 1e-12)

    bonds2["z_isin"] = bonds2.groupby("isin")["spread_bp"].apply(rolling_z).reset_index(level=0, drop=True)

    # Issuer-tenor grid (bucket tenor to nearest in 'tenors')
    tenarr = np.array(tenors, dtype=float)
    bonds2["tenor_bucket"] = tenarr[np.argmin(np.abs(bonds2["tenor_y"].values[:, None] - tenarr[None, :]), axis=1)]
    issuer_curve = (
        bonds2.dropna(subset=["spread_bp", "tenor_bucket"])
        .groupby(["date", "issuer", "tenor_bucket"])
        .agg(spread_bp=("spread_bp", "mean"))
        .reset_index()
        .pivot_table(index=["date", "issuer"], columns="tenor_bucket", values="spread_bp")
        .reset_index()
        .sort_values(["issuer", "date"])
    )
    issuer_curve.columns = ["date", "issuer"] + [f"t{int(t)}y_bp" for t in issuer_curve.columns[2:]]
    return bonds2, issuer_curve


# -------------- Aggregation / indices --------------
def aggregates(bonds2: pd.DataFrame) -> pd.DataFrame:
    # Simple issuer/country/rating/tier aggregates
    def pct(x, q): 
        x = x.dropna()
        return float(np.percentile(x, q)) if len(x) else np.nan

    grp_cols = [
        ["issuer"], ["country"], ["rating"], ["tier"],
        ["country", "tier"], ["rating", "tier"]
    ]
    out = []
    for cols in grp_cols:
        g = bonds2.groupby(cols).agg(
            count=("spread_bp", "size"),
            mean_bp=("spread_bp", "mean"),
            p10_bp=("spread_bp", lambda x: pct(x, 10)),
            p90_bp=("spread_bp", lambda x: pct(x, 90)),
            last_bp=("spread_bp", lambda x: x.dropna().iloc[-1] if x.notna().any() else np.nan),
            last_z=("z_isin", lambda x: x.dropna().iloc[-1] if x.notna().any() else np.nan),
        ).reset_index()
        g["group"] = "+".join(cols)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def dv01_indices(bonds2: pd.DataFrame) -> pd.DataFrame:
    # DV01-weighted average spread by date × {tier, rating bucket (IG/HighYield by heuristic)}
    df = bonds2.copy()
    rating = df.get("rating", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    # very rough IG/Non-IG split
    ig = rating.str.extract(r"(\d+)").astype(float)
    ig_flag = (ig[0] <= 3) | rating.str.contains("A|AA|AAA|BBB")  # heuristic
    df["bucket"] = np.where(ig_flag.fillna(False), "IG", "Non-IG")
    df["dv01w"] = df["dv01"].clip(lower=0)
    rows = []
    for (dt, tier, bucket), g in df.groupby(["date", "tier", "bucket"]):
        w = g["dv01w"].sum()
        if w <= 0 or not np.isfinite(w):
            avg = np.nan
            chg = np.nan
        else:
            avg = float(np.nansum(g["spread_bp"] * g["dv01w"]) / w)
            # daily change vs previous day within same tier/bucket
            rows.append({"date": dt, "tier": tier, "bucket": bucket, "dv01_index_bp": avg})
    idx = pd.DataFrame(rows).sort_values(["tier", "bucket", "date"])
    idx["d1_change_bp"] = idx.groupby(["tier", "bucket"])["dv01_index_bp"].diff()
    return idx


# -------------- PCA (optional) --------------
def pca_on_curves(issuer_curve: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build X (issuers × tenors) using last available date per issuer
    if issuer_curve.empty or issuer_curve.shape[1] < 4:
        return pd.DataFrame(), pd.DataFrame()
    last_date = issuer_curve["date"].max()
    X = issuer_curve[issuer_curve["date"] == last_date].copy()
    X = X.set_index("issuer").drop(columns=["date"])
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if X.shape[0] < 3 or X.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()
    # mean-center columns
    Xc = X - X.mean(axis=0)
    # SVD
    U, s, Vt = np.linalg.svd(Xc.values, full_matrices=False)
    expl = (s**2) / np.sum(s**2)
    # Loadings (tenor dimension)
    load = pd.DataFrame(Vt, columns=X.columns)
    load.index = [f"PC{i+1}" for i in range(len(load))]
    # Scores per issuer
    scores = pd.DataFrame(U * s, index=X.index)
    scores.columns = load.index
    # Explained variance
    ev = pd.DataFrame({"component": load.index, "explained_var": expl})
    # Pack components with tenor labels
    comps = load.reset_index().rename(columns={"index": "component"})
    return (
        pd.concat([scores.reset_index().rename(columns={"index": "issuer"})], axis=0),
        pd.concat([comps, ev], axis=1)
    )


# -------------- CDS basis (optional) --------------
def cds_basis(bonds2: pd.DataFrame, cds: pd.DataFrame) -> pd.DataFrame:
    if cds is None or cds.empty:
        return pd.DataFrame()
    c = cds.rename(
        columns={
            (ncol(cds, "date") or cds.columns[0]): "date",
            (ncol(cds, "issuer") or "issuer"): "issuer",
            (ncol(cds, "tenor_y") or "tenor_y"): "tenor_y",
            (ncol(cds, "cds_bp") or "cds_bp"): "cds_bp",
        }
    )
    c["date"] = to_date(c["date"])
    c["tenor_y"] = pd.to_numeric(c["tenor_y"], errors="coerce")
    c["cds_bp"] = pd.to_numeric(c["cds_bp"], errors="coerce")
    # Bucket issuer curve to same tenor rounding
    df = bonds2.copy()
    df["tenor_bucket"] = np.round(df["tenor_y"].astype(float))
    c["tenor_bucket"] = np.round(c["tenor_y"].astype(float))
    # Join on date/issuer/tenor_bucket
    merged = (
        df.groupby(["date", "issuer", "tenor_bucket"])["spread_bp"].mean().reset_index()
        .merge(c.groupby(["date", "issuer", "tenor_bucket"])["cds_bp"].mean().reset_index(),
               on=["date", "issuer", "tenor_bucket"], how="inner")
    )
    merged["basis_bp"] = merged["spread_bp"] - merged["cds_bp"]
    return merged.sort_values(["issuer", "date", "tenor_bucket"])


# -------------- AT1 vs Equity (optional) --------------
def at1_equity_corr(bonds2: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    if equity is None or equity.empty:
        return pd.DataFrame()
    e = equity.rename(
        columns={
            (ncol(equity, "date") or equity.columns[0]): "date",
            (ncol(equity, "issuer") or "issuer"): "issuer",
            (ncol(equity, "px") or ncol(equity, "ret") or "px"): "px",
        }
    )
    e["date"] = to_date(e["date"])
    e = e.sort_values(["issuer", "date"])
    # compute equity returns if price provided
    if "ret" not in e.columns:
        e["ret"] = e.groupby("issuer")["px"].pct_change()
    # AT1 spreads by issuer/date
    at1 = bonds2[bonds2["tier"] == "AT1"].groupby(["issuer", "date"])["spread_bp"].mean().reset_index()
    at1["d1_bp"] = at1.groupby("issuer")["spread_bp"].diff()
    # Join & rolling corr (30D)
    merged = at1.merge(e[["issuer", "date", "ret"]], on=["issuer", "date"], how="inner").sort_values(["issuer", "date"])
    def roll_corr(df):
        r = df["ret"]
        s = df["d1_bp"]
        out = pd.Series(np.nan, index=df.index)
        win = 30
        for i in range(len(df)):
            lo = max(0, i - win + 1)
            rr = r.iloc[lo:i+1]; ss = s.iloc[lo:i+1]
            if rr.notna().sum() > 5 and ss.notna().sum() > 5:
                out.iloc[i] = float(np.corrcoef(rr.fillna(0), ss.fillna(0))[0, 1])
        return out
    merged["rolling_corr"] = merged.groupby("issuer", group_keys=False).apply(roll_corr)
    return merged


# -------------- Stressing --------------
def stress_indices(idx: pd.DataFrame, parallels: List[int], slope_specs: List[str]) -> pd.DataFrame:
    """
    Apply simple shocks to DV01 index:
      - Parallel: add bp to 'dv01_index_bp'
      - Slope: add a*bps at long tenor, -b*bps at short — here we only label the scenario; index stays parallel
        (you can extend with tenor weights if you store them; for now we show parallel as proxy).
    """
    rows = []
    last = idx.sort_values("date").groupby(["tier", "bucket"]).tail(1)
    for _, r in last.iterrows():
        for p in parallels:
            rows.append({
                "tier": r["tier"], "bucket": r["bucket"], "scenario": f"PAR_{p:+d}bp",
                "index_bp_new": float(r["dv01_index_bp"] + p)
            })
        for s in slope_specs:
            rows.append({
                "tier": r["tier"], "bucket": r["bucket"], "scenario": f"SLOPE_{s}",
                "index_bp_new": float(r["dv01_index_bp"])  # placeholder proxy
            })
    return pd.DataFrame(rows)


# -------------- CLI / Orchestration --------------
@dataclass
class Config:
    bonds: str
    bench: str
    cds: Optional[str]
    equity: Optional[str]
    tenors: List[float]
    window: int
    base_ccy: str
    pca: bool
    stresses: List[int]
    slope_stresses: List[str]
    outdir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EU bank bond spread analytics")
    ap.add_argument("--bonds", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--cds", default="")
    ap.add_argument("--equity", default="")
    ap.add_argument("--tenors", default="1,2,3,5,7,10,15,20")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--base-ccy", default="EUR")
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--stresses", default="+25,+50,-25")
    ap.add_argument("--slope-stresses", default="+10x10-2s")
    ap.add_argument("--outdir", default="out_eu_banks")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    bonds = read_csv(args.bonds)
    bench = read_csv(args.bench)
    cds = read_csv(args.cds) if args.cds else None
    equity = read_csv(args.equity) if args.equity else None

    if bonds is None or bonds.empty or bench is None or bench.empty:
        raise SystemExit("bonds.csv and bench.csv are required and appear empty or missing.")

    tenors = [float(x.strip()) for x in str(args.tenors).split(",") if x.strip()]
    stresses = [int(x.replace("+","").strip()) for x in str(args.stresses).split(",") if x.strip()]
    slope_specs = [x.strip() for x in str(args.slope_stresses).split(",") if x.strip()]

    bonds2, issuer_curve = compute_spreads(bonds, bench, tenors, args.window)
    agg = aggregates(bonds2)
    idx = dv01_indices(bonds2)

    # Optional overlays
    cds_b = cds_basis(bonds2, cds) if cds is not None else pd.DataFrame()
    at1_eq = at1_equity_corr(bonds2, equity) if equity is not None else pd.DataFrame()

    # Optional PCA
    scores, comps = (pd.DataFrame(), pd.DataFrame())
    if args.pca:
        scores, comps = pca_on_curves(issuer_curve)

    # Stressing
    stress_tbl = stress_indices(idx, stresses, slope_specs)

    # Latest snapshot KPIs
    latest_date = bonds2["date"].max()
    last_univ = bonds2[bonds2["date"] == latest_date]
    kpi = {
        "latest_date": str(pd.to_datetime(latest_date).date()) if pd.notna(latest_date) else None,
        "n_isins": int(last_univ["isin"].nunique()),
        "n_issuers": int(last_univ["issuer"].nunique()),
        "avg_spread_bp": float(np.nanmean(last_univ["spread_bp"])),
        "p90_spread_bp": float(np.nanpercentile(last_univ["spread_bp"].dropna(), 90)) if last_univ["spread_bp"].notna().any() else None,
        "at1_wideners_top5": last_univ[last_univ["tier"]=="AT1"].groupby("issuer")["spread_bp"].mean().sort_values(ascending=False).head(5).round(1).to_dict(),
        "dv01_index_tiers": idx[idx["date"] == idx["date"].max()].pivot(index="tier", columns="bucket", values="dv01_index_bp").round(1).fillna(np.nan).to_dict(),
        "base_ccy": args.base_ccy,
    }

    # Write outputs
    bonds2.sort_values(["date","issuer","isin"]).to_csv(outdir / "bond_spreads.csv", index=False)
    issuer_curve.to_csv(outdir / "issuer_curve.csv", index=False)
    agg.to_csv(outdir / "aggregates.csv", index=False)
    idx.to_csv(outdir / "index_dv01.csv", index=False)
    if not cds_b.empty:
        cds_b.to_csv(outdir / "cds_basis.csv", index=False)
    if not at1_eq.empty:
        at1_eq.to_csv(outdir / "at1_equity_snap.csv", index=False)
    if args.pca and not comps.empty:
        scores.to_csv(outdir / "pca_scores.csv", index=False)
        comps.to_csv(outdir / "pca_components.csv", index=False)
    stress_tbl.to_csv(outdir / "stress_table.csv", index=False)

    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        bonds=args.bonds, bench=args.bench, cds=args.cds or None, equity=args.equity or None,
        tenors=tenors, window=args.window, base_ccy=args.base_ccy, pca=bool(args.pca),
        stresses=stresses, slope_stresses=slope_specs, outdir=args.outdir
    )), indent=2))

    # Console
    print("== EU Bank Spreads ==")
    print(f"Date: {kpi['latest_date']}  Universe: {kpi['n_isins']} ISINs / {kpi['n_issuers']} issuers")
    print(f"Avg spread: {kpi['avg_spread_bp']:.1f} bp  P90: {kpi['p90_spread_bp']:.1f} bp")
    print("Top AT1 wideners:", kpi["at1_wideners_top5"])
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
