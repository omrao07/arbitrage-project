# feature-store/features/credit_spread.py
"""
Feast feature definitions for Credit Spreads.

Data assumptions
---------------
- Partitioned parquet written by your pipelines, e.g.
  data/credit/processed/spreads/date=YYYY-MM-DD/issuer_id=.../part-*.parquet
  data/credit/processed/index_spreads/date=YYYY-MM-DD/index_id=.../part-*.parquet

Required columns (issuers):
  ts (timestamp), issuer_id (string), sector (string),
  oas_bps (float), z_spread_bps (float), duration_yrs (float),
  rating_num (int), spread_5d_change_bps (float), spread_20d_change_bps (float)

Required columns (indices):
  ts (timestamp), index_id (string), hy_oas_bps (float), ig_oas_bps (float),
  hy_z_bps (float), ig_z_bps (float), spread_trend_20d (float)

If you already defined entities in feature-store/entities.py, they’ll be used.
Otherwise, we create minimal fallback entities here.

Usage
-----
feast apply
feast materialize <start> <end>
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field # type: ignore
from feast.types import Float32, Int64, String# type: ignore
from feast.infra.offline_stores.file_source import FileSource# type: ignore
from feast.on_demand_feature_view import on_demand_feature_view# type: ignore
from feast import RequestSource# type: ignore

# ---------------------------------------------------------------------
# Entities (use your project entities if available)
# ---------------------------------------------------------------------
try:
    # Prefer entities defined in your repository
    from feature_store.entities import credit_issuer, credit_index  # type: ignore
except Exception:
    # Fallback minimal entities if imports not available
    credit_issuer = Entity(
        name="credit_issuer",
        join_keys=["issuer_id"],
        description="Issuer identifier (CUSIP/Ticker/UUID)"
    )
    credit_index = Entity(
        name="credit_index",
        join_keys=["index_id"],
        description="Credit index identifier (e.g., CDXHY, CDXIG)"
    )

# ---------------------------------------------------------------------
# Sources
#   You can switch paths to s3:// or gs:// — Feast file source supports fsspec URLs.
# ---------------------------------------------------------------------
issuer_spreads_source = FileSource(
    name="issuer_credit_spreads_source",
    path="data/credit/processed/spreads/date=*/issuer_id=*/*.parquet",
    timestamp_field="ts",
    created_timestamp_column=None,
    file_format="parquet",
)

index_spreads_source = FileSource(
    name="credit_index_spreads_source",
    path="data/credit/processed/index_spreads/date=*/index_id=*/*.parquet",
    timestamp_field="ts",
    created_timestamp_column=None,
    file_format="parquet",
)

# ---------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------
credit_spreads = FeatureView(
    name="credit_spreads",
    entities=[credit_issuer],
    ttl=timedelta(days=365 * 3),
    online=True,
    source=issuer_spreads_source,
    schema=[
        Field(name="oas_bps", dtype=Float32),
        Field(name="z_spread_bps", dtype=Float32),
        Field(name="duration_yrs", dtype=Float32),
        Field(name="rating_num", dtype=Int64),                # map AAA=1 .. D=10 (example)
        Field(name="sector", dtype=String),
        Field(name="spread_5d_change_bps", dtype=Float32),
        Field(name="spread_20d_change_bps", dtype=Float32),
    ],
    tags={"domain": "credit", "layer": "silver"},
    description="Issuer-level option-adjusted spread features and short-horizon momentum.",
)

credit_index_spreads = FeatureView(
    name="credit_index_spreads",
    entities=[credit_index],
    ttl=timedelta(days=365 * 5),
    online=True,
    source=index_spreads_source,
    schema=[
        Field(name="hy_oas_bps", dtype=Float32),
        Field(name="ig_oas_bps", dtype=Float32),
        Field(name="hy_z_bps", dtype=Float32),
        Field(name="ig_z_bps", dtype=Float32),
        Field(name="spread_trend_20d", dtype=Float32),  # e.g., HY − IG or index momentum
    ],
    tags={"domain": "credit", "layer": "silver"},
    description="Credit index spread levels (HY/IG) and 20-day trend.",
)

# ---------------------------------------------------------------------
# Optional: On-demand derived features
# ---------------------------------------------------------------------
# Request-time parameters to normalize spreads or set risk thresholds.
spread_req = RequestSource(
    name="credit_spread_request",
    schema=[
        Field(name="wide_spread_threshold_bps", dtype=Float32),  # e.g., 350 bps
    ],
)

@on_demand_feature_view(
    sources=[credit_spreads, spread_req],
    schema=[
        Field(name="oas_per_duration", dtype=Float32),
        Field(name="is_wide_spread", dtype=Int64),
    ],
    name="credit_spreads_derived",
    description="Request-time derived credit spread features.",
)
def credit_spreads_derived(features_df):
    """
    oas_per_duration: OAS normalized by duration (bps per year)
    is_wide_spread:   1 if OAS exceeds threshold, else 0
    """
    import numpy as np
    oas = features_df["oas_bps"].astype("float32")
    dur = features_df["duration_yrs"].replace({0.0: np.nan}).astype("float32")
    thresh = features_df.get("wide_spread_threshold_bps", 350.0).astype("float32")

    oas_per_duration = (oas / dur).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    is_wide_spread = (oas > thresh).astype("int64")

    out = features_df[[]].copy()
    out["oas_per_duration"] = oas_per_duration
    out["is_wide_spread"] = is_wide_spread
    return out

# ---------------------------------------------------------------------
# Registry export helper (optional)
# ---------------------------------------------------------------------
__all__ = [
    "credit_issuer",
    "credit_index",
    "credit_spreads",
    "credit_index_spreads",
    "credit_spreads_derived",
]