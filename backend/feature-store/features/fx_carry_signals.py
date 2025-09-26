# feature-store/features/fx_carry_signals.py
"""
Feast feature definitions for FX carry / rate-differential signals.

Data assumptions
----------------
Partitioned Parquet produced by your FX pipeline, e.g.:
  data/fx/processed/carry/date=YYYY-MM-DD/pair=EURUSD/part-*.parquet

Required columns
  ts (timestamp, daily), pair (string, e.g., "EURUSD"),
  base (string), quote (string),
  ir_base_3m (float), ir_quote_3m (float),
  ir_base_1y (float), ir_quote_1y (float),
  carry_3m (float), carry_1y (float),        # rate differentials (base - quote), in %
  spot_ret_20d (float),                      # 1m momentum helper
  vol_20d (float)                            # realized spot vol (abs returns)

Usage
-----
feast apply
feast materialize 2023-01-01 2025-01-01
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field# type: ignore
from feast.types import Float32, String# type: ignore
from feast.infra.offline_stores.file_source import FileSourcev# type: ignore
from feast.on_demand_feature_view import on_demand_feature_view# type: ignore
from feast import RequestSource# type: ignore

# ---------------------------------------------------------------------
# Entity (reuse repository entity if present)
# ---------------------------------------------------------------------
try:
    from feature_store.entities import fx_pair  # type: ignore
except Exception:
    fx_pair = Entity(
        name="fx_pair",
        join_keys=["pair"],  # "EURUSD", "USDJPY", ...
        description="FX currency pair (base+quote, 6 letters)",
    )

# ---------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------
fx_carry_source = FileSource(# type: ignore
    name="fx_carry_source",
    path="data/fx/processed/carry/date=*/pair=*/*.parquet",  # local/S3/GCS via fsspec ok
    timestamp_field="ts",
    created_timestamp_column=None,
    file_format="parquet",
)

# ---------------------------------------------------------------------
# Feature View
# ---------------------------------------------------------------------
fx_carry_signals = FeatureView(
    name="fx_carry_signals",
    entities=[fx_pair],
    ttl=timedelta(days=365 * 3),
    online=True,
    source=fx_carry_source,
    schema=[
        Field(name="base", dtype=String),
        Field(name="quote", dtype=String),
        Field(name="ir_base_3m", dtype=Float32),
        Field(name="ir_quote_3m", dtype=Float32),
        Field(name="ir_base_1y", dtype=Float32),
        Field(name="ir_quote_1y", dtype=Float32),
        Field(name="carry_3m", dtype=Float32),     # base - quote (%)
        Field(name="carry_1y", dtype=Float32),     # base - quote (%)
        Field(name="spot_ret_20d", dtype=Float32), # momentum proxy
        Field(name="vol_20d", dtype=Float32),      # realized vol
    ],
    tags={"domain": "fx", "layer": "silver"},
    description="FX carry (rate differentials) and simple momentum/vol features.",
)

# ---------------------------------------------------------------------
# On-demand derived features
# ---------------------------------------------------------------------
req = RequestSource(
    name="fx_carry_request",
    schema=[
        Field(name="vol_floor", dtype=Float32),   # e.g., 0.02 => avoid divide-by-small-vol
        Field(name="blend_alpha", dtype=Float32), # 0..1 to mix carry & momentum
    ],
)

@on_demand_feature_view(
    sources=[fx_carry_signals, req],
    schema=[
        Field(name="carry_risk_adj", dtype=Float32),   # carry_1y / max(vol, floor)
        Field(name="carry_momo_blend", dtype=Float32), # alpha*carry + (1-alpha)*mom
    ],
    name="fx_carry_derived",
    description="Request-time derived FX carry signals.",
)
def fx_carry_derived(df):
    import numpy as np

    vol = df["vol_20d"].astype("float32")
    floor = df.get("vol_floor", 0.02).astype("float32")  # 2% vol floor default
    denom = np.maximum(vol, floor)

    carry1y = df["carry_1y"].astype("float32")
    mom1m = df["spot_ret_20d"].astype("float32")
    alpha = df.get("blend_alpha", 0.6).astype("float32") # tilt toward carry by default

    carry_risk_adj = (carry1y / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    carry_momo_blend = (alpha * carry1y + (1.0 - alpha) * mom1m).astype("float32")

    out = df[[]].copy()
    out["carry_risk_adj"] = carry_risk_adj
    out["carry_momo_blend"] = carry_momo_blend
    return out

# ---------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------
__all__ = [
    "fx_pair",
    "fx_carry_signals",
    "fx_carry_derived",
]