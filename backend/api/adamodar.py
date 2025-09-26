#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adamodar.py
-----------
Utilities to work with Damodaran-style valuation datasets and run quick
ERP/CRP, cost of capital, DCF, and relative valuation analyses.

Key features
- Resilient CSV loaders (auto-discover by fuzzy filename match).
- Cleaning & harmonization: tickers, sectors, regions, dates, numerics.
- Equity Risk Premiums (top-down and bottom-up), Country Risk Premium (CRP).
- Betas (by company/sector/region), un/levered transformations.
- Cost of equity, cost of debt, WACC; CAPM & multi-beta flavors.
- DCF: FCFF/FCFE flows, growth/decay templates, terminal value variants.
- Relative valuation: EV/EBITDA, P/E, P/S, P/B comps and z-scores.
- Sector/Region aggregations + regression helpers.
- Curated export for your “models.yaml” and or “valuation/” pipeline.

Dependencies: numpy, pandas. Optional: scipy for stats.quantiles.
"""

from __future__ import annotations
import os
import re
import math
import glob
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_NUMERIC_COL_HINTS = (
    "ev", "ebit", "ebitda", "revenue", "sales", "net", "income", "eps",
    "cash", "debt", "capex", "wc", "wc_change", "fcff", "fcfe", "shares",
    "price", "mktcap", "equity", "assets", "book", "interest", "beta",
    "tax", "margin", "roa", "roe", "roc", "wacc", "ke", "kd", "g", "erp", "crp",
)

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if any(h in c.lower() for h in _NUMERIC_COL_HINTS):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _norm_ticker(x: str | float) -> str:
    if pd.isna(x): return ""
    s = str(x).strip().upper()
    # remove non-alnum except . and -
    s = re.sub(r"[^A-Z0-9\.\-]", "", s)
    return s

def _norm_sector(s: str | float) -> str:
    if pd.isna(s): return ""
    return re.sub(r"\s+", " ", str(s).strip().title())

def _norm_region(s: str | float) -> str:
    if pd.isna(s): return ""
    return re.sub(r"\s+", " ", str(s).strip().title())

def _infer_date_from_filename(path: str) -> Optional[pd.Timestamp]:
    m = re.search(r"(20\d{2})[^\d]?(0?[1-9]|1[0-2])", os.path.basename(path))
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(year=y, month=mo, day=1)
    return None

# -----------------------------------------------------------------------------
# Data registry/loader
# -----------------------------------------------------------------------------

@dataclass
class AdamodarData:
    root: str = "data/adamodar"
    cache: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Patterns mapped to canonical keys
    patterns: Dict[str, Iterable[str]] = field(default_factory=lambda: {
        # Cross-sectional “latest”
        "by_company": [r"by[_\s]?company\.csv$", r"globale?quities\.csv$", r"us\s+equities\.csv$"],
        "by_sector":  [r"by[_\s]?sector\.csv$", r"sectors\s+multiples\.csv$", r"pe\s+by\s+sector\.csv$", r"ev\s*ebitda\s*by\s*sector\.csv$"],
        "by_region":  [r"by[_\s]?region\.csv$", r"bottom\s*up\s*by\s*region\.csv$", r"y\s*region\.csv$"],

        # Timeseries
        "erp_ts":     [r"erp\s*(timeseries)?\.csv$", r"us\s+historical\s+erp\.csv$", r"annual\s+erp\.csv$", r"nthly\s+erp\.csv$"],
        "betas":      [r"betas\.csv$", r"202[2-5]/betas\.csv$"],
        "fundamentals":[r"fundamentals.*\.csv$", r"fundamentals\s+clean\.csv$", r"202[2-5]/fundamentals\.csv$"],

        # Multiples
        "multiples":  [r"ev\s*ebitda\.csv$", r"pe\s+ratios\.csv$", r"pb\s+ratios\.csv$", r"evenue\s+multiples\.csv$"],

        # Region/EM/DM
        "regions_em": [r"emerging\s+markets\.csv$", r"em\s+equities\s+regression\.csv$"],
        "regions_dev":[r"developed\s+markets\.csv$", r"developed\s+equities\s+regression\.csv$"],
        "regions_us": [r"us\s+equities\s+regression\.csv$"],

        # Risk premia pieces
        "rp":         [r"rp\.csv$", r"pull\s*crp\.py$"],

        # Growth & margins
        "margins":    [r"ebit\s*margin\.csv$", r"net\s*income\s*margin\.csv$", r"operating\s*margin\.csv$", r"cashflow\s*growth\.csv$", r"earnings\s*growth\.csv$", r"revenue\s*growth\.csv$"],
    })

    def _find(self, key: str) -> List[str]:
        pats = list(self.patterns.get(key, []))
        hits: List[str] = []
        for p in pats:
            for path in glob.glob(os.path.join(self.root, "**", "*"), recursive=True):
                if re.search(p, os.path.basename(path), flags=re.IGNORECASE):
                    hits.append(path)
        return sorted(set(hits))

    def load(self, key: str, prefer_latest: bool = True) -> pd.DataFrame:
        if key in self.cache:
            return self.cache[key].copy()
        paths = self._find(key)
        if not paths:
            raise FileNotFoundError(f"No files matched for key='{key}' in {self.root}")
        # choose by embedded date if available
        if prefer_latest:
            dated = [(p, _infer_date_from_filename(p)) for p in paths]
            dated.sort(key=lambda t: (t[1] or pd.Timestamp(1900,1,1)), reverse=True)
            path = dated[0][0]
        else:
            path = paths[0]

        df = pd.read_csv(path)
        # common cleaning
        for c in df.columns:
            if c.lower() in {"ticker", "symbol"}:
                df.rename(columns={c: "ticker"}, inplace=True)
            elif c.lower() in {"sector", "industry"}:
                df.rename(columns={c: "sector"}, inplace=True)
            elif "region" in c.lower() or "country" in c.lower():
                df.rename(columns={c: "region"}, inplace=True)
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].map(_norm_ticker)
        if "sector" in df.columns:
            df["sector"] = df["sector"].map(_norm_sector)
        if "region" in df.columns:
            df["region"] = df["region"].map(_norm_region)
        df = _coerce_numeric(df)
        self.cache[key] = df
        return df.copy()

    # Convenience wrappers
    def companies(self) -> pd.DataFrame: return self.load("by_company")
    def sectors(self) -> pd.DataFrame:   return self.load("by_sector")
    def regions(self) -> pd.DataFrame:   return self.load("by_region")
    def erp_ts(self) -> pd.DataFrame:    return self.load("erp_ts")
    def betas(self) -> pd.DataFrame:     return self.load("betas")
    def fundamentals(self) -> pd.DataFrame: return self.load("fundamentals")
    def multiples(self) -> pd.DataFrame: return self.load("multiples")
    def margins(self) -> pd.DataFrame:   return self.load("margins")
    def rp(self) -> pd.DataFrame:        return self.load("rp")

# -----------------------------------------------------------------------------
# Risk premia & cost of capital
# -----------------------------------------------------------------------------

def combine_erp(erp_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ERP timeseries into ['date','erp'] where ERP is decimal (e.g., 0.05).
    Accepts columns like ['Date','ERP'] or ['year','us_erp'] etc.
    """
    df = erp_ts.copy()
    # try to find date & erp-like column
    date_col = next((c for c in df.columns if c.lower() in {"date","year","month","period"}), df.columns[0])
    erp_col = next((c for c in df.columns if "erp" in c.lower()), df.columns[-1])
    df = df[[date_col, erp_col]].rename(columns={date_col: "date", erp_col: "erp"})
    # parse date/year
    if df["date"].dtype == object:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            df["date"] = pd.to_datetime(df["date"].astype(str) + "-12-31", errors="coerce")
    # scale if in percentage
    if df["erp"].abs().median() > 1.0:
        df["erp"] = df["erp"] / 100.0
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df

def country_risk_premium(base_crp: float, rating_notch_adj: float = 0.0) -> float:
    """
    Simple CRP helper: start from base spread (e.g., CDS-implied), adjust by sovereign rating notch if desired.
    """
    return float(max(0.0, base_crp + rating_notch_adj))

def ke_capm(rf: float, beta: float, erp: float, crp: float = 0.0, lambda_em: float = 1.0) -> float:
    """Cost of equity via CAPM plus scaled country risk premium (λ in [0,1])."""
    return float(rf + beta * erp + lambda_em * crp)

def kd_after_tax(pre_tax_rate: float, tax_rate: float) -> float:
    return float(pre_tax_rate * (1.0 - tax_rate))

def wacc(ke: float, kd_at: float, E: float, D: float) -> float:
    V = max(1e-12, E + D)
    return float(ke * (E / V) + kd_at * (D / V))

def unlever_beta(beta_l: float, debt: float, equity: float, tax_rate: float) -> float:
    """Hamada: β_u = β_l / (1 + (1 - t) D/E)."""
    if equity <= 0: return float("nan")
    return float(beta_l / (1.0 + (1.0 - tax_rate) * (debt / equity)))

def lever_beta(beta_u: float, debt: float, equity: float, tax_rate: float) -> float:
    """Hamada inverse: β_l = β_u * (1 + (1 - t) D/E)."""
    if equity <= 0: return float("nan")
    return float(beta_u * (1.0 + (1.0 - tax_rate) * (debt / equity)))

# -----------------------------------------------------------------------------
# DCF engines
# -----------------------------------------------------------------------------

@dataclass
class DCFInputs:
    # Base financials
    revenue: float
    ebit_margin: float
    tax_rate: float
    dep_amort: float
    capex: float
    wc_change: float
    # Capital structure
    debt: float
    cash: float
    shares: float
    rf: float
    erp: float
    beta: float
    crp: float = 0.0
    lambda_em: float = 1.0
    kd_pre_tax: float = 0.05
    # Growth path
    g_years: int = 5
    g_start: float = 0.08
    g_decay_to: float = 0.03  # fade to this by year g_years
    # Terminal
    g_terminal: float = 0.025
    tv_method: str = "gordon"  # 'gordon' or 'exit_multiple'
    exit_ev_ebitda: Optional[float] = None

def project_fcff(base: DCFInputs) -> pd.DataFrame:
    """
    Build FCFF projection over g_years with linear fade of growth & margins (optional).
    """
    years = np.arange(1, base.g_years + 1)
    g_path = np.linspace(base.g_start, base.g_decay_to, num=base.g_years)
    ebit_m = base.ebit_margin  # keep constant; could also fade if desired

    rev = [base.revenue]
    ebit = []
    tax = []
    fcff = []

    for t, g in zip(years, g_path):
        rev.append(rev[-1] * (1 + g))
        e = rev[-1] * ebit_m
        ta = e * base.tax_rate
        nopat = e - ta
        # FCFF = NOPAT + D&A - Capex - ΔWC
        f = nopat + base.dep_amort - base.capex - base.wc_change
        ebit.append(e); tax.append(ta); fcff.append(f)

    df = pd.DataFrame({
        "year": years,
        "revenue": rev[1:],
        "ebit": ebit,
        "tax": tax,
        "fcff": fcff,
    })
    return df

def dcf_valuation(base: DCFInputs) -> Dict[str, float | pd.DataFrame]:
    flows = project_fcff(base)
    ke = ke_capm(base.rf, base.beta, base.erp, base.crp, base.lambda_em)
    kd_at = kd_after_tax(base.kd_pre_tax, base.tax_rate)
    E = max(1e-9, (base.revenue * base.ebit_margin) * 12)  # rough proxy for equity scale (not used in PV)
    D = max(1e-9, base.debt)
    disc = wacc(ke, kd_at, E, D)

    # PV of FCFF
    flows["disc"] = (1.0 + disc) ** flows["year"]
    flows["pv_fcff"] = flows["fcff"] / flows["disc"]

    # Terminal Value
    if base.tv_method == "gordon":
        tv_fcff = flows["fcff"].iloc[-1] * (1 + base.g_terminal)
        tv = tv_fcff / max(1e-9, (disc - base.g_terminal))
    elif base.tv_method == "exit_multiple" and base.exit_ev_ebitda:
        ebitda = flows["ebit"].iloc[-1] + base.dep_amort
        tv = ebitda * base.exit_ev_ebitda
    else:
        raise ValueError("Invalid terminal value settings.")

    pv_tv = tv / ((1.0 + disc) ** flows["year"].iloc[-1])

    ev = float(flows["pv_fcff"].sum() + pv_tv)
    eq_value = ev - base.debt + base.cash
    price = eq_value / max(1e-9, base.shares)

    return {
        "ke": float(ke), "kd_after_tax": float(kd_at), "wacc": float(disc),
        "enterprise_value": float(ev), "equity_value": float(eq_value), "price": float(price),
        "flows": flows,
    }

# -----------------------------------------------------------------------------
# Relative valuation
# -----------------------------------------------------------------------------

def relval_comps(df_companies: pd.DataFrame, universe_filter: Optional[pd.Series] = None,
                 metrics: Tuple[str, ...] = ("ev_ebitda", "pe", "ps", "pb"),
                 winsor: float = 0.01) -> pd.DataFrame:
    """
    Compute relative valuation z-scores for common multiples across a peer set.
    Expected columns in df_companies: 'ticker','sector','region','ev','ebitda','price','eps','sales','book'
    """
    df = df_companies.copy()
    if universe_filter is not None:
        df = df[universe_filter].copy()

    # Build multiples if not present
    if "ev_ebitda" in metrics and "ev_ebitda" not in df.columns and {"ev","ebitda"} <= set(df.columns):
        df["ev_ebitda"] = df["ev"] / df["ebitda"].replace(0, np.nan)
    if "pe" in metrics and "pe" not in df.columns and {"price","eps"} <= set(df.columns):
        df["pe"] = df["price"] / df["eps"].replace(0, np.nan)
    if "ps" in metrics and "ps" not in df.columns and {"price","sales","shares"} <= set(df.columns):
        df["ps"] = (df["price"] * df["shares"]) / df["sales"].replace(0, np.nan)
    if "pb" in metrics and "pb" not in df.columns and {"price","book","shares"} <= set(df.columns):
        df["pb"] = (df["price"] * df["shares"]) / df["book"].replace(0, np.nan)

    # winsorize per sector to control industry structure
    out = []
    for g, gdf in df.groupby("sector", dropna=False):
        s = gdf.copy()
        for m in metrics:
            if m not in s.columns: 
                s[m] = np.nan
            ql, qh = s[m].quantile(winsor), s[m].quantile(1 - winsor)
            s[m] = s[m].clip(ql, qh)
            s[m + "_z"] = (s[m] - s[m].mean()) / (s[m].std(ddof=0) + 1e-12)
        out.append(s)
    out = pd.concat(out, axis=0).sort_index()

    out["rel_score"] = -out[[m + "_z" for m in metrics if m + "_z" in out.columns]].mean(axis=1)
    return out

# -----------------------------------------------------------------------------
# Aggregations & betas
# -----------------------------------------------------------------------------

def aggregate_by(df: pd.DataFrame, by: str, cols: Iterable[str], wcol: Optional[str] = None) -> pd.DataFrame:
    """Weighted or unweighted aggregation by 'sector'/'region'."""
    d = df.copy()
    if wcol and wcol in d.columns:
        w = d[wcol].replace({0: np.nan}).fillna(0)
        out = (d[cols].multiply(w, axis=0)).groupby(d[by]).sum() / (w.groupby(d[by]).sum().replace(0, np.nan))
    else:
        out = d.groupby(by)[cols].mean()
    out = out.reset_index()
    return out

def recompute_betas(df_returns: pd.DataFrame, benchmark_col: str, window: int = 252) -> pd.Series:
    """
    Rolling OLS beta of each column vs benchmark_col; returns last betas.
    df_returns: columns = tickers; expects daily returns.
    """
    b = {}
    bm = df_returns[benchmark_col]
    var_bm = bm.rolling(window).var().iloc[-1]
    for c in df_returns.columns:
        if c == benchmark_col: 
            continue
        cov = df_returns[c].rolling(window).cov(bm).iloc[-1]
        b[c] = float(cov / (var_bm + 1e-12))
    return pd.Series(b, name="beta")

# -----------------------------------------------------------------------------
# Curator/export
# -----------------------------------------------------------------------------

def curated_snapshot(data: AdamodarData) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of curated frames you can save to /valuations/curated.
    """
    out: Dict[str, pd.DataFrame] = {}
    try:
        out["companies"] = data.companies()
    except Exception as e:
        warnings.warn(f"companies: {e}")
    try:
        out["sectors"] = data.sectors()
    except Exception as e:
        warnings.warn(f"sectors: {e}")
    try:
        out["regions"] = data.regions()
    except Exception as e:
        warnings.warn(f"regions: {e}")
    try:
        out["erp_ts"] = combine_erp(data.erp_ts())
    except Exception as e:
        warnings.warn(f"erp_ts: {e}")
    try:
        out["multiples"] = data.multiples()
    except Exception as e:
        warnings.warn(f"multiples: {e}")
    return out

def save_curated(cur: Dict[str, pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for k, v in cur.items():
        v.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)

# -----------------------------------------------------------------------------
# Quick CLI (optional)
# -----------------------------------------------------------------------------

def _demo_company_relval(root: str, take: int = 50) -> pd.DataFrame:
    data = AdamodarData(root=root)
    comps = data.companies()
    if "mktcap" in comps.columns:
        top = comps.nlargest(take, "mktcap")
    else:
        top = comps.head(take)
    rel = relval_comps(top)
    return rel[["ticker","sector","rel_score","ev_ebitda_z","pe_z","ps_z","pb_z"]].sort_values("rel_score")

def _demo_dcf(root: str) -> Dict[str, float | pd.DataFrame]:
    data = AdamodarData(root=root)
    # pick a sample company (requires columns present; otherwise use placeholders)
    df = data.companies()
    row = df.iloc[0]
    eps = float(row.get("eps", 5.0))
    shares = float(row.get("shares", 1_000.0))
    price = float(row.get("price", 100.0))
    revenue = float(row.get("revenue", 10_000.0))
    ebit_m = float(row.get("ebit_margin", row.get("operating_margin", 0.15)))
    tax = float(row.get("tax", 0.23))
    dep = float(row.get("dep_amort", 500.0))
    capex = float(row.get("capex", 600.0))
    wc = float(row.get("wc_change", 50.0))
    debt = float(row.get("debt", 2_000.0))
    cash = float(row.get("cash", 500.0))
    beta = float(row.get("beta", 1.1))

    erp_df = combine_erp(data.erp_ts())
    erp = float(erp_df["erp"].iloc[-1]) if not erp_df.empty else 0.05
    rf = 0.04

    base = DCFInputs(
        revenue=revenue, ebit_margin=ebit_m, tax_rate=tax,
        dep_amort=dep, capex=capex, wc_change=wc,
        debt=debt, cash=cash, shares=shares,
        rf=rf, erp=erp, beta=beta,
        g_years=5, g_start=0.08, g_decay_to=0.04, g_terminal=0.025,
        tv_method="gordon"
    )
    return dcf_valuation(base)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Damodaran utilities")
    ap.add_argument("--root", default="data/adamodar", help="Root folder with CSVs")
    ap.add_argument("--mode", choices=["relval","dcf","curate"], default="relval")
    ap.add_argument("--take", type=int, default=50)
    ap.add_argument("--out", default="valuations/curated")
    args = ap.parse_args()

    if args.mode == "relval":
        df = _demo_company_relval(args.root, take=args.take)
        print(df.head(20).to_string(index=False))
    elif args.mode == "dcf":
        res = _demo_dcf(args.root)
        print({k: v for k, v in res.items() if k not in {"flows"}})
        print(res["flows"].head().to_string(index=False))#type:ignore
    elif args.mode == "curate":
        cur = curated_snapshot(AdamodarData(root=args.root))
        save_curated(cur, args.out)
        print(f"Saved curated snapshot to {args.out}")