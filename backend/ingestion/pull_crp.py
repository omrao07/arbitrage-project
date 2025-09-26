#!/usr/bin/env python3
"""
pull_crp.py
-----------
Build Country Risk Premiums (CRP) dataset from CDS or ratings.

Inputs:
  - cds_spreads.csv (curated or raw)
  - sovereign_ratings.csv (optional backup)

Outputs:
  - data/adamodar/curated/crp.csv
"""

import os
import pandas as pd
import numpy as np
from datetime import date

CURATED_DIR = "data/adamodar/curated"
OUT_PATH = os.path.join(CURATED_DIR, "crp.csv")

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_cds(path: str) -> pd.DataFrame:
    """Load curated cds_spreads.csv or raw feed."""
    df = pd.read_csv(path)
    if "as_of" not in df.columns:
        raise ValueError("Expected 'as_of' col in cds_spreads")
    return df

def load_ratings(path: str) -> pd.DataFrame:
    """Optional: load ratings → default spreads map."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def compute_crp(cds: pd.DataFrame, ratings: pd.DataFrame, us_ticker="US_5Y", vol_adj: float = 1.5) -> pd.DataFrame:
    """Compute CRP for each country."""
    today = str(date.today())
    df = cds.copy()
    df = df[df["tenor"].str.upper()=="5Y"]  # focus on 5Y tenor

    # Extract US baseline CDS
    us_cds = df.loc[df["ticker"]==us_ticker, "spread_bps"]
    if us_cds.empty:
        raise ValueError(f"No US CDS found ({us_ticker})")
    us_val = us_cds.iloc[0]

    rows = []
    for _, r in df.iterrows():
        crp = max(0.0, (r["spread_bps"] - us_val) / 10000.0 * vol_adj)  # convert bps→decimal
        rows.append({
            "as_of": pd.to_datetime(r["as_of"]).strftime("%Y-%m-%d"),
            "country": r["name"],
            "region": r.get("region",""),
            "rating": r.get("rating",""),
            "cds_bps": r["spread_bps"],
            "us_cds_bps": us_val,
            "vol_adj": vol_adj,
            "crp": round(crp,4),
            "method": "CDS_spread",
            "source": r.get("source",""),
            "notes": f"{r['name']} CDS {r['spread_bps']}bps vs US {us_val}bps"
        })
    return pd.DataFrame(rows)

def save_crp(df: pd.DataFrame, path: str = OUT_PATH):
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"✅ wrote {path} ({len(df)} rows)")

if __name__ == "__main__":
    # Example: use curated cds_spreads.csv
    cds_path = os.path.join(CURATED_DIR,"cds_spreads.csv")
    ratings_path = os.path.join(CURATED_DIR,"sovereign_ratings.csv")

    cds = load_cds(cds_path)
    ratings = load_ratings(ratings_path)

    df = compute_crp(cds, ratings, us_ticker="US_5Y", vol_adj=1.5)
    save_crp(df)