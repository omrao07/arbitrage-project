# feature-store/features/eq_returns_1d.py
"""
Feast feature definitions for daily equity returns (1-day horizon).

Data assumptions
----------------
- Partitioned parquet written by your equities pipeline, e.g.:
  data/equities/processed/returns/date=YYYY-MM-DD/ticker=XYZ/part-*.parquet

Required columns:
  ts (timestamp), ticker (string), sector (string), 
  ret_1d (float), ret_5d (float), ret_20d (float),
  vol_20d (float), mcap_usd (float)

Usage
-----
feast apply
feast materialize <start> <end>
"""

from datetime import timedelta
# type: ignore
from feast import Entity, FeatureView, Field# type: ignore
from feast.types import Float32, String# type: ignore
from feast.infra.offline_stores.file_source import FileSource# type: ignore
from feast.on_demand_feature_view import on_demand_feature_view# type: ignore
from feast import RequestSource# type: ignore

# ---------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------
try:
    from feature_store.entities import equity  # type: ignore
except Exception:
    equity = Entity(
        name="equity",
        join_keys=["ticker"],
        description="Equity ticker symbol (e.g., AAPL, TSLA)",
    )

# ---------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------
equity_returns_source = FileSource(
    name="equity_returns_source",
    path="data/equities/processed/returns/date=*/ticker=*/*.parquet",
    timestamp_field="ts",
    created_timestamp_column=None,
    file_format="parquet",
)

# ---------------------------------------------------------------------
# Feature View
# ---------------------------------------------------------------------
eq_returns_1d = FeatureView(
    name="eq_returns_1d",
    entities=[equity],
    ttl=timedelta(days=365 * 2),
    online=True,
    source=equity_returns_source,
    schema=[
        Field(name="ret_1d", dtype=Float32),
        Field(name="ret_5d", dtype=Float32),
        Field(name="ret_20d", dtype=Float32),
        Field(name="vol_20d", dtype=Float32),
        Field(name="mcap_usd", dtype=Float32),
        Field(name="sector", dtype=String),
    ],
    tags={"domain": "equities", "layer": "silver"},
    description="Equity 1-day returns, multi-horizon returns, and realized volatility.",
)

# ---------------------------------------------------------------------
# On-demand derived features
# ---------------------------------------------------------------------
req = RequestSource(
    name="eq_returns_request",
    schema=[
        Field(name="vol_threshold", dtype=Float32),  # user-specified threshold
    ],
)

@on_demand_feature_view(
    sources=[eq_returns_1d, req],
    schema=[
        Field(name="risk_adj_return", dtype=Float32),
        Field(name="is_high_vol", dtype=String),
    ],
    name="eq_returns_derived",
    description="Request-time derived equity return features.",
)
def eq_returns_derived(features_df):
    """
    risk_adj_return = ret_1d / vol_20d
    is_high_vol = 'Y' if vol_20d > threshold else 'N'
    """
    import numpy as np
    r1d = features_df["ret_1d"].astype("float32")
    vol = features_df["vol_20d"].replace({0.0: np.nan}).astype("float32")
    thresh = features_df.get("vol_threshold", 0.05).astype("float32")  # default 5%

    risk_adj = (r1d / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    high_vol = np.where(vol > thresh, "Y", "N")

    out = features_df[[]].copy()
    out["risk_adj_return"] = risk_adj
    out["is_high_vol"] = high_vol
    return out

# ---------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------
__all__ = [
    "equity",
    "eq_returns_1d",
    "eq_returns_derived",
]