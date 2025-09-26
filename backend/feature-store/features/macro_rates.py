# feature-store/features/macro_rates.py
"""
Feast feature definitions for macro rates & yield curves.

Expected Parquet schema (daily or business-day):
  ts (timestamp)
  country (string, e.g., "US","EA","UK","JP")
  policy_rate (float, %)
  y2 (float, %)      # 2y yield
  y5 (float, %)
  y10 (float, %)
  y30 (float, %)
  be5y (float, %)    # 5y breakeven (optional)
  be10y (float, %)   # 10y breakeven (optional)
  term_spread_10y2y (float, %, optional if you already compute it)
  term_spread_5y2y  (float, %, optional)
  policy_rate_change_3m (float, bps, optional)
  y10_momentum_3m (float, bps or %, optional)

Usage
-----
feast apply
feast materialize 2023-01-01 2025-01-01
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field # type: ignore
from feast.types import Float32, String # type: ignore
from feast.infra.offline_stores.file_source import FileSource # type: ignore
from feast.on_demand_feature_view import on_demand_feature_view # type: ignore
from feast import RequestSource # type: ignore

# ---------------------------------------------------------------------
# Entity (reuse your repo entities if defined)
# ---------------------------------------------------------------------
try:
    from feature_store.entities import macro_country  # type: ignore
except Exception:
    macro_country = Entity(
        name="macro_country",
        join_keys=["country"],
        description="Country/region code for macro features (US, EA, UK, JP, ...)",
    )

# ---------------------------------------------------------------------
# Offline source (local/S3/GCS paths supported via fsspec when configured)
# ---------------------------------------------------------------------
macro_rates_source = FileSource(
    name="macro_rates_source",
    path="data/macro/processed/rates/date=*/country=*/*.parquet",
    timestamp_field="ts",
    created_timestamp_column=None,
    file_format="parquet",
)

# ---------------------------------------------------------------------
# Base features from the lakehouse
# ---------------------------------------------------------------------
macro_rates = FeatureView(
    name="macro_rates",
    entities=[macro_country],
    ttl=timedelta(days=365 * 5),
    online=True,
    source=macro_rates_source,
    schema=[
        Field(name="policy_rate", dtype=Float32),
        Field(name="y2", dtype=Float32),
        Field(name="y5", dtype=Float32),
        Field(name="y10", dtype=Float32),
        Field(name="y30", dtype=Float32),

        # Optional but supported if present in your Parquet
        Field(name="be5y", dtype=Float32),
        Field(name="be10y", dtype=Float32),
        Field(name="term_spread_10y2y", dtype=Float32),
        Field(name="term_spread_5y2y", dtype=Float32),
        Field(name="policy_rate_change_3m", dtype=Float32),
        Field(name="y10_momentum_3m", dtype=Float32),
    ],
    tags={"domain": "macro", "layer": "silver"},
    description="Policy rate & yield curve levels (2y/5y/10y/30y) with optional breakevens and precomputed term spreads.",
)

# ---------------------------------------------------------------------
# On-demand derived features
#   - Computes real 10y, slope/curvature if not already provided
#   - Simple regimes using thresholds you can override at request time
# ---------------------------------------------------------------------
od_request = RequestSource(
    name="macro_rates_request",
    schema=[
        Field(name="tighten_threshold_bps", dtype=Float32),  # default 25 bps
        Field(name="recession_slope_threshold", dtype=Float32),  # default 0.0 (%); slope < 0 inverted
    ],
)

@on_demand_feature_view(
    sources=[macro_rates, od_request],
    schema=[
        Field(name="real_10y", dtype=Float32),            # y10 - be10y (if be10y available)
        Field(name="slope_10y2y", dtype=Float32),         # y10 - y2
        Field(name="curvature_2_5_10", dtype=Float32),    # 2*y5 - (y2 + y10)
        Field(name="is_tightening", dtype=Float32),       # 1.0 if policy_rate_change_3m > threshold_bps
        Field(name="recession_flag", dtype=Float32),      # 1.0 if slope < thresh AND y10_momentum_3m < 0
    ],
    name="macro_rates_derived",
    description="Request-time derived macro rate features: real yields, slope/curvature, tightening and recession flags.",
)
def macro_rates_derived(df):
    import numpy as np
    as_f32 = lambda s, default=0.0: (s if s is not None else default).astype("float32")  # type: ignore # noqa: E731

    y2  = as_f32(df.get("y2"))
    y5  = as_f32(df.get("y5"))
    y10 = as_f32(df.get("y10"))
    be10 = df.get("be10y")
    be10 = be10.astype("float32") if be10 is not None else None

    # Derived: real 10y if breakeven is available
    real_10y = (y10 - be10) if be10 is not None else np.full_like(y10, np.nan, dtype="float32")

    # Derived slope & curvature (don’t rely on precomputed columns)
    slope_10y2y = (y10 - y2).astype("float32")
    curvature = (2.0 * y5 - (y2 + y10)).astype("float32")

    # Tightening flag from 3m policy change vs request threshold
    dpol3m = df.get("policy_rate_change_3m")
    dpol3m = dpol3m.astype("float32") if dpol3m is not None else np.zeros_like(y2, dtype="float32")
    thresh_bps = df.get("tighten_threshold_bps")
    tighten_thr = (thresh_bps.astype("float32") if thresh_bps is not None else np.full_like(y2, 25.0, dtype="float32"))
    is_tight = (dpol3m > tighten_thr).astype("float32")

    # Recession flag: (a) curve inverted vs threshold AND (b) long-end momentum negative
    slope_thr = df.get("recession_slope_threshold")
    slope_thr = slope_thr.astype("float32") if slope_thr is not None else np.full_like(y2, 0.0, dtype="float32")
    y10_momo = df.get("y10_momentum_3m")
    y10_momo = y10_momo.astype("float32") if y10_momo is not None else np.zeros_like(y2, dtype="float32")
    recession = ((slope_10y2y < slope_thr) & (y10_momo < 0.0)).astype("float32")

    out = df[[]].copy()
    out["real_10y"] = real_10y.astype("float32")
    out["slope_10y2y"] = slope_10y2y
    out["curvature_2_5_10"] = curvature
    out["is_tightening"] = is_tight
    out["recession_flag"] = recession
    return out

# ---------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------
__all__ = [
    "macro_country",
    "macro_rates",
    "macro_rates_derived",
]