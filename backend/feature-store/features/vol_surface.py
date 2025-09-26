# feature-store/features/vol_surface.py
"""
Feast feature definitions for options volatility surface snapshots per underlier.

Expected Parquet schema (daily/business-day aggregates by underlier):
  ts (timestamp)
  symbol (string)           # underlier, e.g., AAPL, SPY, ES, BTC
  iv30 (float)              # 30d ATM implied vol (annualized, %)
  iv60 (float)
  iv90 (float)
  rv20 (float)              # 20d realized vol (close-close, %)
  rr25 (float)              # 25Δ risk–reversal (call – put, in vol points)
  bf25 (float)              # 25Δ butterfly (smile convexity, in vol points)
  skew_25d (float)          # put–call skew at 25Δ (convention-specific; keep sign consistent)
  vega_notional (float)     # optional: aggregate vega notionals traded
  dvol (float)              # optional: DVOL/VVIX style index for underlier (if available)

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
# Entity (reuse your project entity if one exists)
# ---------------------------------------------------------------------
try:
    from feature_store.entities import underlier  # type: ignore
except Exception:
    underlier = Entity(
        name="underlier",
        join_keys=["symbol"],
        description="Options underlier symbol (AAPL, SPY, ES, BTC, …)",
    )

# ---------------------------------------------------------------------
# Source (local/S3/GCS via fsspec supported by Feast config)
# ---------------------------------------------------------------------
vol_surface_source = FileSource(
    name="vol_surface_source",
    path="data/options/processed/vol_surface/date=*/symbol=*/*.parquet",
    timestamp_field="ts",
    created_timestamp_column=None,
    file_format="parquet",
)

# ---------------------------------------------------------------------
# Base (silver) features from lakehouse
# ---------------------------------------------------------------------
vol_surface = FeatureView(
    name="vol_surface",
    entities=[underlier],
    ttl=timedelta(days=365 * 3),
    online=True,
    source=vol_surface_source,
    schema=[
        Field(name="iv30", dtype=Float32),
        Field(name="iv60", dtype=Float32),
        Field(name="iv90", dtype=Float32),
        Field(name="rv20", dtype=Float32),
        Field(name="rr25", dtype=Float32),
        Field(name="bf25", dtype=Float32),
        Field(name="skew_25d", dtype=Float32),
        Field(name="vega_notional", dtype=Float32),
        Field(name="dvol", dtype=Float32),
    ],
    tags={"domain": "options", "layer": "silver"},
    description="ATM term structure (30/60/90), 25Δ risk-reversal/butterfly, skew and realized vol.",
)

# ---------------------------------------------------------------------
# On-demand derived signals (risk-adjusted + regimes)
# ---------------------------------------------------------------------
od_req = RequestSource(
    name="vol_surface_request",
    schema=[
        Field(name="crash_bf_threshold", dtype=Float32),    # e.g., 2.0 vol pts
        Field(name="richness_floor", dtype=Float32),         # min |carry| to flag rich/cheap
    ],
)

@on_demand_feature_view(
    sources=[vol_surface, od_req],
    schema=[
        Field(name="term_slope_90_30", dtype=Float32),       # iv90 - iv30
        Field(name="term_slope_60_30", dtype=Float32),       # iv60 - iv30
        Field(name="term_curvature", dtype=Float32),         # 2*iv60 - (iv30 + iv90)
        Field(name="carry_30_vs_rv20", dtype=Float32),       # iv30 - rv20
        Field(name="skew_norm", dtype=Float32),              # skew_25d / max(iv30,1e-6)
        Field(name="rr_norm", dtype=Float32),                # rr25 / max(iv30,1e-6)
        Field(name="bf_norm", dtype=Float32),                # bf25 / max(iv30,1e-6)
        Field(name="crash_risk_flag", dtype=Float32),        # 1 if bf25 > threshold
        Field(name="richness_flag", dtype=Float32),          # 1 if |carry| > floor
    ],
    name="vol_surface_derived",
    description="Derived term-structure/smile metrics, carry, and simple crash/richness flags.",
)
def vol_surface_derived(df):
    import numpy as np

    asf = lambda s, d=0.0: (s if s is not None else d).astype("float32")  # type: ignore # noqa: E731

    iv30 = asf(df.get("iv30"))
    iv60 = asf(df.get("iv60"))
    iv90 = asf(df.get("iv90"))
    rv20 = asf(df.get("rv20"))
    rr25 = asf(df.get("rr25"))
    bf25 = asf(df.get("bf25"))
    skew = asf(df.get("skew_25d"))

    eps = np.full_like(iv30, 1e-6, dtype="float32")
    denom = np.maximum(iv30, eps)

    # Term structure
    slope_90_30 = (iv90 - iv30).astype("float32")
    slope_60_30 = (iv60 - iv30).astype("float32")
    curvature = (2.0 * iv60 - (iv30 + iv90)).astype("float32")

    # Carry and normalized smile metrics
    carry = (iv30 - rv20).astype("float32")
    skew_norm = (skew / denom).astype("float32")
    rr_norm = (rr25 / denom).astype("float32")
    bf_norm = (bf25 / denom).astype("float32")

    # Simple flags (thresholds are request-time tunables)
    bf_thr = df.get("crash_bf_threshold")
    bf_thr = bf_thr.astype("float32") if bf_thr is not None else np.full_like(iv30, 2.0, dtype="float32")
    crash_flag = (bf25 > bf_thr).astype("float32")

    rich_floor = df.get("richness_floor")
    rich_floor = rich_floor.astype("float32") if rich_floor is not None else np.full_like(iv30, 1.5, dtype="float32")
    richness_flag = (np.abs(carry) > rich_floor).astype("float32")

    out = df[[]].copy()
    out["term_slope_90_30"] = slope_90_30
    out["term_slope_60_30"] = slope_60_30
    out["term_curvature"] = curvature
    out["carry_30_vs_rv20"] = carry
    out["skew_norm"] = skew_norm
    out["rr_norm"] = rr_norm
    out["bf_norm"] = bf_norm
    out["crash_risk_flag"] = crash_flag
    out["richness_flag"] = richness_flag
    return out

# ---------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------
__all__ = [
    "underlier",
    "vol_surface",
    "vol_surface_derived",
]