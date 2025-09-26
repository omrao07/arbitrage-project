#!/usr/bin/env python3
"""
build_curated.py
----------------
One-click builder for curated valuation & mispricing datasets.

Generates:
- data/adamodar/curated/pe_ratios.csv
- data/adamodar/curated/pb_ratios.csv
- data/adamodar/curated/revenue_multiples.csv
- data/adamodar/curated/ev_ebitda.csv
- data/adamodar/curated/mispricings_us.csv
- data/adamodar/curated/mispricings_em.csv
- data/adamodar/curated/mispricings_global.csv

Usage:
  python build_curated.py --all
  python build_curated.py --only pe_ratios ev_ebitda
  python build_curated.py --list
  python build_curated.py --outdir data/adamodar/curated --all
"""

import os
import sys
import argparse
from datetime import date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# --- Optional project helpers (falls back if not available) ---
try:
    from io import ensure_dir as _ensure_dir, save_csv as _save_csv  # type: ignore # your io.py
except Exception:  # pragma: no cover
    def _ensure_dir(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    def _save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
        _ensure_dir(path)
        df.to_csv(path, index=index)

TODAY = str(date.today())


# ---------- Utilities ----------

def _zscore_proxy(curr: pd.Series, avg: pd.Series, pct_sigma: float = 0.15) -> pd.Series:
    """
    Proxy z-score when you don't have a full history:
    z ≈ (x - μ) / (μ * pct_sigma)
    """
    denom = (avg * pct_sigma).replace(0, np.nan)
    return (curr - avg) / denom

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)

def _write(df: pd.DataFrame, out_path: str) -> None:
    _save_csv(df, out_path, index=False)
    print(f"✅ wrote {out_path} ({len(df)} rows)")


# ---------- Builders ----------

def build_pe_ratios(outdir: str) -> str:
    path = os.path.join(outdir, "pe_ratios.csv")
    rows = [
        (TODAY,"S&P500","index","US","",72e12,3.2e12,np.nan,18.0,np.nan,"S&P rich vs 10y avg"),
        (TODAY,"MSCI_EM","index","EM","",18e12,1.5e12,np.nan,13.5,np.nan,"EM trades at discount"),
        (TODAY,"MSCI_Europe","index","Europe","",15e12,1.1e12,np.nan,14.5,np.nan,"Europe below avg"),
        (TODAY,"US_Tech","sector","US","Tech",18e12,0.6e12,np.nan,25.0,np.nan,"Tech premium vs avg"),
        (TODAY,"US_Banks","sector","US","Banks",6.5e12,0.52e12,np.nan,11.0,np.nan,"Banks slightly rich"),
        (TODAY,"Apple","company","US","Tech",3.25e12,0.10e12,np.nan,28.0,np.nan,"Apple strong profitability"),
        (TODAY,"Petrobras","company","LatAm","Energy",1.0e11,0.02e12,np.nan,8.0,np.nan,"Discounted"),
    ]
    cols = ["as_of","entity","entity_type","region","sector",
            "market_cap_usd","net_income_usd","pe_ratio",
            "pe_hist_avg","pe_zscore","notes"]
    df = pd.DataFrame(rows, columns=cols)
    df["pe_ratio"] = _safe_div(df["market_cap_usd"], df["net_income_usd"])
    df["pe_zscore"] = _zscore_proxy(df["pe_ratio"], df["pe_hist_avg"])
    _write(df, path)
    return path

def build_pb_ratios(outdir: str) -> str:
    path = os.path.join(outdir, "pb_ratios.csv")
    rows = [
        (TODAY,"S&P500","index","US","",72e12,25e12,np.nan,2.60,np.nan,"S&P modestly rich"),
        (TODAY,"MSCI_EM","index","EM","",18e12,10e12,np.nan,1.90,np.nan,"EM a bit cheap"),
        (TODAY,"MSCI_Europe","index","Europe","",15e12,8.5e12,np.nan,1.90,np.nan,"Below avg"),
        (TODAY,"US_Banks","sector","US","Banks",6.5e12,4.5e12,np.nan,1.20,np.nan,"Above avg PB"),
        (TODAY,"US_Tech","sector","US","Tech",18e12,6e12,np.nan,2.50,np.nan,"Tech premium"),
        (TODAY,"Apple","company","US","Tech",3.25e12,0.75e12,np.nan,3.50,np.nan,"High ROE"),
        (TODAY,"Petrobras","company","LatAm","Energy",1.0e11,0.08e12,np.nan,1.50,np.nan,"Gov risk"),
    ]
    cols = ["as_of","entity","entity_type","region","sector",
            "market_cap_usd","book_value_usd","pb_ratio","pb_hist_avg","pb_zscore","notes"]
    df = pd.DataFrame(rows, columns=cols)
    df["pb_ratio"] = _safe_div(df["market_cap_usd"], df["book_value_usd"])
    df["pb_zscore"] = _zscore_proxy(df["pb_ratio"], df["pb_hist_avg"])
    _write(df, path)
    return path

def build_revenue_multiples(outdir: str) -> str:
    path = os.path.join(outdir, "revenue_multiples.csv")
    rows = [
        (TODAY,"S&P500","index","US","",72e12,4.0e12,np.nan,16.0,np.nan,"Index above avg"),
        (TODAY,"MSCI_EM","index","EM","",18e12,5.9e12,np.nan,3.5,np.nan,"EM slightly cheap"),
        (TODAY,"MSCI_Europe","index","Europe","",15e12,5.0e12,np.nan,3.2,np.nan,"Below avg"),
        (TODAY,"US_SaaS","sector","US","Tech",2.5e12,0.15e12,np.nan,12.0,np.nan,"High premium"),
        (TODAY,"US_Biotech","sector","US","Healthcare",1.2e12,0.3e12,np.nan,5.0,np.nan,"Discount"),
        (TODAY,"Apple","company","US","Tech",3.25e12,0.4e12,np.nan,7.0,np.nan,"Modestly rich"),
        (TODAY,"Petrobras","company","LatAm","Energy",1.0e11,1.0e11,np.nan,1.2,np.nan,"Discount"),
    ]
    cols = ["as_of","entity","entity_type","region","sector",
            "market_cap_usd","revenue_usd","psales_ratio","psales_hist_avg","psales_zscore","notes"]
    df = pd.DataFrame(rows, columns=cols)
    df["psales_ratio"] = _safe_div(df["market_cap_usd"], df["revenue_usd"])
    df["psales_zscore"] = _zscore_proxy(df["psales_ratio"], df["psales_hist_avg"])
    _write(df, path)
    return path

def build_ev_ebitda(outdir: str) -> str:
    path = os.path.join(outdir, "ev_ebitda.csv")
    rows = [
        (TODAY,"S&P500","index","US","",72e12,6e12,np.nan,11.0,np.nan,"Slightly rich"),
        (TODAY,"MSCI_EM","index","EM","",18e12,2.4e12,np.nan,8.0,np.nan,"Cheap vs hist"),
        (TODAY,"MSCI_Europe","index","Europe","",15e12,1.7e12,np.nan,9.5,np.nan,"Below avg"),
        (TODAY,"Tech Sector","sector","US","Tech",18e12,1.15e12,np.nan,14.5,np.nan,"US Tech premium"),
        (TODAY,"Energy Sector","sector","US","Energy",6e12,0.75e12,np.nan,7.0,np.nan,"Near avg"),
        (TODAY,"Apple","company","US","Tech",3.25e12,0.24e12,np.nan,12.0,np.nan,"Premium"),
        (TODAY,"Petrobras","company","LatAm","Energy",1.0e11,2.0e10,np.nan,6.5,np.nan,"Discount"),
    ]
    cols = ["as_of","entity","entity_type","region","sector","ev_usd","ebitda_usd",
            "ev_ebitda","ev_ebitda_hist_avg","ev_ebitda_zscore","notes"]
    df = pd.DataFrame(rows, columns=cols)
    df["ev_ebitda"] = _safe_div(df["ev_usd"], df["ebitda_usd"])
    df["ev_ebitda_zscore"] = _zscore_proxy(df["ev_ebitda"], df["ev_ebitda_hist_avg"])
    _write(df, path)
    return path

def build_mispricings_us(outdir: str) -> str:
    path = os.path.join(outdir, "mispricings_us.csv")
    rows = [
        ("2025-01-15","US","", "equity_index","S&P500","PE_vs_History",4600,4800,np.nan,-1.1,"short",90,"Fwd P/E > 85th pctile","Hedge via ES"),
        ("2025-01-15","US","", "sector","Tech_EV","EVEBITDA_vs_hist",14.5,15.6,np.nan,-0.8,"short",60,"Rich vs 10y avg","Pair vs Energy"),
        ("2025-01-15","US","", "rate","US_10Y","Term_Premium",-0.002,0.001,np.nan,-0.9,"long",60,"Excess TP","TY futures"),
        ("2025-01-15","US","", "credit","IG_OAS","CDS_vs_spread",1.05,1.20,np.nan,-1.3,"long",45,"Cash OAS > CDS","Primary supply"),
        ("2025-01-15","US","", "commodity","WTI","Real_Fair_Price",85,75,np.nan,0.7,"long",30,"Below fair","Event risk"),
    ]
    cols = ["as_of","region","state","asset_type","instrument","metric","fair_value","market_value",
            "mispricing_pct","zscore","signal","horizon_days","rationale","notes"]
    df = pd.DataFrame(rows, columns=cols)
    df["mispricing_pct"] = (df["fair_value"] / df["market_value"] - 1).replace([np.inf, -np.inf], np.nan)
    # Simple rule
    df["signal"] = df["mispricing_pct"].apply(lambda x: "long" if x > 0.03 else "short" if x < -0.03 else "flat")
    _write(df, path)
    return path

def build_mispricings_em(outdir: str) -> str:
    path = os.path.join(outdir, "mispricings_em.csv")
    rows = [
        ("2025-01-15","India","Emerging Asia","equity_index","NIFTY50","PE_vs_History",19800,21000,np.nan,-1.2,"short",90,"Rich vs hist avg","Pair vs EM"),
        ("2025-01-15","Brazil","LatAm","fx","BRLUSD","PPP",0.230,0.200,np.nan,1.1,"long",120,"~15% cheap to PPP","Carry +"),
        ("2025-01-15","China","Emerging Asia","equity_index","CSI300","ERP_gap",4200,3800,np.nan,0.8,"long",180,"ERP elevated","Policy support"),
        ("2025-01-15","South Africa","EMEA","credit","ZAF_5Y_CDS","CDS_vs_spread",180,210,np.nan,-1.6,"long",60,"Cash > CDS","Liquidity"),
    ]
    cols = ["as_of","country","region","asset_type","instrument","metric",
            "fair_value","market_value","mispricing_pct","zscore","signal","horizon_days","rationale","notes"]
    df = pd.DataFrame(rows, columns=cols)
    df["mispricing_pct"] = (df["fair_value"] / df["market_value"] - 1).replace([np.inf, -np.inf], np.nan)
    df["signal"] = df["mispricing_pct"].apply(lambda x: "long" if x > 0.03 else "short" if x < -0.03 else "flat")
    _write(df, path)
    return path

def build_mispricings_global(outdir: str) -> str:
    path = os.path.join(outdir, "mispricings_global.csv")
    rows = [
        ("2025-01-15","US","United States","equity_index","S&P500","PE_vs_History",4600,4800,np.nan,-1.1,"short",90,"Fwd P/E > 85th pctile","Hedge ES"),
        ("2025-01-15","Europe","Germany","equity_index","DAX","ERP_gap",16500,15500,np.nan,0.9,"long",120,"ERP spread > avg","Policy easing"),
        ("2025-01-15","Japan","Japan","equity_index","Nikkei225","PB_vs_History",34500,33500,np.nan,0.6,"long",180,"ROE improving","Rates capped"),
        ("2025-01-15","Global","","commodity","Brent_Oil","Real_Fair_Price",85,75,np.nan,0.7,"long",60,"Undervalued vs cost curve","Geo prem absent"),
    ]
    cols = ["as_of","region","country","asset_type","instrument","metric",
            "fair_value","market_value","mispricing_pct","zscore","signal","horizon_days","rationale","notes"]
    df = pd.DataFrame(rows, columns=cols)
    df["mispricing_pct"] = (df["fair_value"] / df["market_value"] - 1).replace([np.inf, -np.inf], np.nan)
    df["signal"] = df["mispricing_pct"].apply(lambda x: "long" if x > 0.03 else "short" if x < -0.03 else "flat")
    _write(df, path)
    return path


# ---------- Registry & CLI ----------

BUILDERS: Dict[str, Callable[[str], str]] = {
    "pe_ratios": build_pe_ratios,
    "pb_ratios": build_pb_ratios,
    "revenue_multiples": build_revenue_multiples,
    "ev_ebitda": build_ev_ebitda,
    "mispricings_us": build_mispricings_us,
    "mispricings_em": build_mispricings_em,
    "mispricings_global": build_mispricings_global,
}

def list_builders() -> None:
    print("Available curated builders:")
    for k in BUILDERS.keys():
        print(f"  - {k}")

def run_builders(names: List[str], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for n in names:
        if n not in BUILDERS:
            print(f"❌ unknown builder: {n}")
            continue
        try:
            BUILDERS[n](outdir)
        except Exception as e:
            print(f"💥 failed: {n} -> {e}", file=sys.stderr)

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build curated datasets")
    p.add_argument("--outdir", default="data/adamodar/curated", help="output directory")
    p.add_argument("--all", action="store_true", help="build all curated datasets")
    p.add_argument("--only", nargs="+", help="build only specific datasets (space-separated keys)")
    p.add_argument("--list", action="store_true", help="list available builders and exit")
    args = p.parse_args(argv)

    if args.list:
        list_builders()
        return 0

    if args.all:
        names = list(BUILDERS.keys())
        run_builders(names, args.outdir)
        return 0

    if args.only:
        run_builders(args.only, args.outdir)
        return 0

    p.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())