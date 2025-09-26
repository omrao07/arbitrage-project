#!/usr/bin/env python3
"""
pull_erp.py
-----------
Compute Equity Risk Premiums (ERP) by region/country using multiple methods.

Inputs (any subset OK; script will gracefully fallback):
  - data/adamodar/curated/pe_ratios.csv
      cols: as_of, entity, entity_type, region, market_cap_usd, net_income_usd, pe_ratio
  - data/adamodar/curated/us_treasury_yields.csv
      cols: as_of, tenor, yield (decimal), region='US' (or blank)
  - data/adamodar/curated/inflation_expectations.csv  (optional)
      cols: as_of, region, exp_inflation (decimal)
  - data/adamodar/curated/buyback_dividend_yield.csv  (optional)
      cols: as_of, entity/region, dividend_yield, buyback_yield
  - data/adamodar/curated/earnings_growth.csv (optional, top-down growth)
      cols: as_of, region, long_run_growth (decimal)
  - data/adamodar/curated/crp.csv (optional, country risk premia to add for EM)
      cols: as_of, country, region, crp (decimal)

Outputs:
  - data/adamodar/curated/erp.csv
      cols: as_of, scope, scope_type, region, method, rf_nominal, rf_real,
            earnings_yield, payout_yield, growth, implied_erp, implied_real_erp, addl_spread, notes

Methods implemented:
  1) EY_minus_RF        : ERP = EarningsYield - RF
  2) Gordon_Payout      : ERP = (Div+Buyback) + g - RF
  3) Blended_Payout_EY  : ERP = w*(Div+Buyback+g) + (1-w)*(EY) - RF  (w=0.5 default)
  4) CAPE_Yield_minus_RF (if CAPE provided via pe_ratio)  -> same as method 1 with 'entity_type=index'
  5) Optional CRP add-on for EM: ERP += CRP

Notes:
  - EarningsYield = 1 / PE (trailing or forward depending on your pe_ratios feed)
  - RF defaults to US 10Y if region RF not found.
  - Real ERP = ERP - expected inflation (approx), or EY - RF_real if TIPS provided.
"""

import os
import argparse
from typing import Optional, Dict
import numpy as np
import pandas as pd
from datetime import date

CURATED_DIR = "data/adamodar/curated"
OUT_PATH = os.path.join(CURATED_DIR, "erp.csv")

# ---------- Helpers ----------

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if os.path.exists(path) else None

def latest_by_group(df: pd.DataFrame, group_cols, date_col="as_of") -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    idx = d.groupby(group_cols)[date_col].idxmax()
    return d.loc[idx].reset_index(drop=True)

def pick_rf(yields_df: pd.DataFrame, region: str, tenor_preference=("10Y","7Y","5Y")) -> Optional[float]:
    # try region match, then fallback to US
    for reg in (region, "US"):
        sub = yields_df[(yields_df.get("region","US").fillna("US")==reg)] # type: ignore
        if sub.empty: 
            continue
        sub = latest_by_group(sub, ["tenor"])
        for t in tenor_preference:
            row = sub[sub["tenor"].str.upper()==t]
            if not row.empty:
                return float(row["yield"].iloc[0])
    return None

def pick_infl(exp_df: Optional[pd.DataFrame], region: str) -> Optional[float]:
    if exp_df is None or exp_df.empty: 
        return None
    exp_df["as_of"] = pd.to_datetime(exp_df["as_of"])
    sub = exp_df[exp_df["region"].fillna("")==region]
    if sub.empty:
        sub = exp_df[exp_df["region"].fillna("")=="US"]
    if sub.empty:
        return None
    sub = sub.sort_values("as_of")
    return float(sub["exp_inflation"].iloc[-1])

def get_payout_yield(payout_df: Optional[pd.DataFrame], region: str) -> Optional[float]:
    if payout_df is None or payout_df.empty: return None
    df = payout_df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    # try region-level, else aggregate entities with that region if provided
    candidates = []
    if "region" in df.columns:
        candidates = df[df["region"].fillna("")==region]
    if candidates == [] or (isinstance(candidates, pd.DataFrame) and candidates.empty): # type: ignore
        return None
    candidates = candidates.sort_values("as_of") # type: ignore
    dy = float(candidates["dividend_yield"].iloc[-1]) if "dividend_yield" in candidates.columns else 0.0
    by = float(candidates["buyback_yield"].iloc[-1]) if "buyback_yield" in candidates.columns else 0.0
    return dy + by

def get_growth(growth_df: Optional[pd.DataFrame], region: str, default: float=0.02) -> float:
    if growth_df is None or growth_df.empty: 
        return default
    growth_df["as_of"] = pd.to_datetime(growth_df["as_of"])
    sub = growth_df[growth_df["region"].fillna("")==region]
    if sub.empty:
        sub = growth_df[growth_df["region"].fillna("")=="Global"]
    if sub.empty:
        return default
    sub = sub.sort_values("as_of")
    return float(sub["long_run_growth"].iloc[-1])

def get_crp(crp_df: Optional[pd.DataFrame], region: str) -> float:
    """
    Simple CRP adder: if region is EM/LatAm/EM Asia/EMEA, take a representative average
    else 0. You can refine by country later.
    """
    if crp_df is None or crp_df.empty:
        return 0.0
    # map common region aliases
    em_aliases = {"EM","Emerging","EM Asia","LatAm","EMEA","Emerging Markets"}
    if region not in em_aliases:
        return 0.0
    # take latest rows and average
    crp_df["as_of"] = pd.to_datetime(crp_df["as_of"])
    latest = crp_df.sort_values("as_of").groupby(["country","region"]).tail(1)
    val = latest["crp"].mean()
    return float(val) if pd.notna(val) else 0.0

# ---------- ERP Methods ----------

def method_ey_minus_rf(ey: float, rf: float) -> Optional[float]:
    if ey is None or rf is None: return None
    return float(ey - rf)

def method_gordon_payout(payout_yield: float, g: float, rf: float) -> Optional[float]:
    if payout_yield is None or g is None or rf is None: return None
    return float(payout_yield + g - rf)

def method_blended(payout_yield: Optional[float], g: Optional[float], ey: Optional[float], rf: float, w: float=0.5) -> Optional[float]:
    """
    Blended ERP: w*(payout+g) + (1-w)*EY - RF
    """
    if rf is None: return None
    if payout_yield is None or g is None or ey is None:
        return None
    blended_yield = w*(payout_yield + g) + (1.0 - w)*ey
    return float(blended_yield - rf)

# ---------- Main computation ----------

def compute_erp(pe_df: Optional[pd.DataFrame],
                yields_df: Optional[pd.DataFrame],
                infl_df: Optional[pd.DataFrame],
                payout_df: Optional[pd.DataFrame],
                growth_df: Optional[pd.DataFrame],
                crp_df: Optional[pd.DataFrame],
                regions=("US","Europe","Japan","EM","Global")) -> pd.DataFrame:
    rows = []
    today = str(date.today())

    # Prepare earnings yield from pe_ratios
    ey_map: Dict[str, float] = {}
    if pe_df is not None and not pe_df.empty:
        pe_df["as_of"] = pd.to_datetime(pe_df["as_of"])
        # dedupe by (region, entity_type=index) pref if present
        pe_latest = pe_df.sort_values("as_of").groupby(["region","entity","entity_type"]).tail(1)
        # prefer index-level EY per region; fallback to aggregate mean if needed
        for r in regions:
            # try region index first
            sub_idx = pe_latest[(pe_latest["region"]==r) & (pe_latest["entity_type"]=="index")]
            pe = None
            if not sub_idx.empty:
                pe = float(sub_idx["pe_ratio"].iloc[-1])
            else:
                sub_reg = pe_latest[(pe_latest["region"]==r)]
                if not sub_reg.empty:
                    pe = float(sub_reg["pe_ratio"].mean())
            if pe and pe > 0:
                ey_map[r] = 1.0 / pe

    # iterate regions
    for r in regions:
        rf = pick_rf(yields_df, r) if yields_df is not None else None
        exp_infl = pick_infl(infl_df, r)
        rf_real = (rf - exp_infl) if (rf is not None and exp_infl is not None) else None

        ey = ey_map.get(r, None)
        payout = get_payout_yield(payout_df, r)
        g = get_growth(growth_df, r, default=0.02)
        add_crp = get_crp(crp_df, r)  # often 0 for DM

        # Methods
        erp_ey = method_ey_minus_rf(ey, rf) # type: ignore
        erp_gordon = method_gordon_payout(payout, g, rf) # type: ignore
        erp_blend = method_blended(payout, g, ey, rf, w=0.5) # type: ignore

        # Real ERPs (approx): subtract exp. inflation from nominal ERP if rf_real missing
        implied_real_ey = (ey - (rf_real if rf_real is not None else (rf - (exp_infl or 0.0)))) if ey is not None and rf is not None else None
        implied_real_gordon = ((payout or 0.0) + (g or 0.0) - (rf_real if rf_real is not None else (rf - (exp_infl or 0.0)))) if rf is not None else None
        implied_real_blend = ( (0.5*((payout or 0.0) + (g or 0.0)) + 0.5*(ey or 0.0)) - (rf_real if rf_real is not None else (rf - (exp_infl or 0.0))) ) if rf is not None else None

        # Add CRP for EM (optional)
        def add_crp_if_needed(val: Optional[float]) -> Optional[float]:
            return (val + add_crp) if (val is not None and add_crp) else val

        out = [
            # EY - RF
            dict(as_of=today, scope=r, scope_type="region", region=r, method="EY_minus_RF",
                 rf_nominal=rf, rf_real=rf_real, earnings_yield=ey, payout_yield=payout, growth=g,
                 implied_erp=add_crp_if_needed(erp_ey),
                 implied_real_erp=add_crp_if_needed(implied_real_ey),
                 addl_spread=add_crp if add_crp else 0.0,
                 notes="ERP = EarningsYield - RF; EY from pe_ratios"),
            # Gordon
            dict(as_of=today, scope=r, scope_type="region", region=r, method="Gordon_Payout",
                 rf_nominal=rf, rf_real=rf_real, earnings_yield=ey, payout_yield=payout, growth=g,
                 implied_erp=add_crp_if_needed(erp_gordon),
                 implied_real_erp=add_crp_if_needed(implied_real_gordon),
                 addl_spread=add_crp if add_crp else 0.0,
                 notes="ERP = (Div+Buyback)+g - RF"),
            # Blended
            dict(as_of=today, scope=r, scope_type="region", region=r, method="Blended_Payout_EY_50_50",
                 rf_nominal=rf, rf_real=rf_real, earnings_yield=ey, payout_yield=payout, growth=g,
                 implied_erp=add_crp_if_needed(erp_blend),
                 implied_real_erp=add_crp_if_needed(implied_real_blend),
                 addl_spread=add_crp if add_crp else 0.0,
                 notes="ERP = 0.5*((Div+Buyback)+g) + 0.5*EY - RF"),
        ]
        rows.extend(out)

    return pd.DataFrame(rows)

# ---------- Runner ----------

def run(pe_path: str,
        yields_path: str,
        infl_path: Optional[str],
        payout_path: Optional[str],
        growth_path: Optional[str],
        crp_path: Optional[str],
        out_path: str = OUT_PATH) -> None:

    pe_df = load_csv_safe(pe_path)
    yld_df = load_csv_safe(yields_path)
    infl_df = load_csv_safe(infl_path) if infl_path else None
    pay_df = load_csv_safe(payout_path) if payout_path else None
    grw_df = load_csv_safe(growth_path) if growth_path else None
    crp_df = load_csv_safe(crp_path) if crp_path else None

    if yld_df is not None and "yield" in yld_df.columns:
        # ensure numeric decimal
        yld_df["yield"] = pd.to_numeric(yld_df["yield"], errors="coerce")

    df = compute_erp(pe_df, yld_df, infl_df, pay_df, grw_df, crp_df)

    ensure_dir(out_path)
    df.to_csv(out_path, index=False)
    print(f"✅ wrote {out_path} ({len(df)} rows)")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Compute Equity Risk Premiums (ERP) with multiple methods")
    ap.add_argument("--pe", default=os.path.join(CURATED_DIR,"pe_ratios.csv"), help="pe_ratios.csv path")
    ap.add_argument("--yields", default=os.path.join(CURATED_DIR,"us_treasury_yields.csv"), help="risk-free yields csv path")
    ap.add_argument("--infl", default=os.path.join(CURATED_DIR,"inflation_expectations.csv"), help="inflation expectations csv path (optional)")
    ap.add_argument("--payout", default=os.path.join(CURATED_DIR,"buyback_dividend_yield.csv"), help="dividend+buyback yields csv path (optional)")
    ap.add_argument("--growth", default=os.path.join(CURATED_DIR,"earnings_growth.csv"), help="long-run growth csv path (optional)")
    ap.add_argument("--crp", default=os.path.join(CURATED_DIR,"crp.csv"), help="country risk premia csv path (optional)")
    ap.add_argument("--out", default=OUT_PATH, help="output erp.csv path")
    args = ap.parse_args()

    run(args.pe, args.yields, args.infl, args.payout, args.growth, args.crp, args.out)

if __name__ == "__main__":
    main()