#!/usr/bin/env python3
"""
pull_cds_spreads.py
-------------------
Fetch, clean, and save CDS spread data into curated schema.

Supports:
- Local CSV ingestion (e.g., downloaded Markit/Bloomberg files)
- API placeholder (wire in your vendor API later)
"""

import os
import pandas as pd
from datetime import date
from typing import Optional

OUT_PATH = "data/adamodar/curated/cds_spreads.csv"


# ---------- Loader ----------

def load_local_csv(path: str) -> pd.DataFrame:
    """
    Load raw CDS data from a CSV file.
    Expected columns: Date, Name, Region, Tenor, Spread (bps)
    """
    df = pd.read_csv(path)
    # Try common header names
    rename_map = {
        "Date": "as_of",
        "Entity": "name",
        "Region": "region",
        "Ticker": "ticker",
        "Tenor": "tenor",
        "Spread": "spread_bps",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    return df


def fetch_from_api() -> pd.DataFrame:
    """
    Placeholder for API call (Markit, Refinitiv, etc).
    Return schema: as_of, ticker, name, region, tenor, spread_bps
    """
    # TODO: replace with actual API code
    today = str(date.today())
    rows = [
        (today, "US_5Y", "United States", "US", "5Y", 32),
        (today, "AAPL_5Y", "Apple Inc", "US", "5Y", 45),
        (today, "TSLA_5Y", "Tesla Inc", "US", "5Y", 120),
    ]
    return pd.DataFrame(rows, columns=["as_of","ticker","name","region","tenor","spread_bps"])


# ---------- Cleaner ----------

def clean_cds(df: pd.DataFrame, source: str = "Markit") -> pd.DataFrame:
    df = df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.strftime("%Y-%m-%d")
    df["curve_type"] = df.get("curve_type", "senior_unsecured")
    df["source"] = source
    df["notes"] = df.get("notes", "")
    return df[["as_of","ticker","name","region","tenor","spread_bps","curve_type","source","notes"]]


# ---------- Writer ----------

def save_cds(df: pd.DataFrame, path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Wrote {path} ({len(df)} rows)")


# ---------- Main ----------

if __name__ == "__main__":
    # Option A: Load from API (placeholder)
    df = fetch_from_api()

    # Option B: Load from a downloaded file
    # df = load_local_csv("raw/markit_cds.csv")

    df = clean_cds(df)
    save_cds(df, OUT_PATH)