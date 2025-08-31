# tests/test_feature_store.py
import json
import math
import time
import importlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest # type: ignore

"""
Supported public APIs (any one is fine)

A) Class FeatureStore with methods:
   - put(entity: str, features: dict, as_of: int|datetime, ttl_s: Optional[int]=None)
   - get(entity: str, keys: List[str], as_of: int|datetime) -> dict
   - batch_get(entities: List[str], keys: List[str], as_of: int|datetime) -> dict[str, dict]
   - window(entity: str, keys: List[str], start: int|datetime, end: int|datetime) -> List[dict]
   - register_view(name: str, keys: List[str], ttl_s: Optional[int]=None, schema: dict|None=None)
   - materialize(view: str, entities: List[str], start: int|datetime, end: int|datetime, fn, **kw)
   - backfill(view: str, entities: List[str], start: int|datetime, end: int|datetime, fn, **kw)

B) Functions:
   - put(...), get(...), batch_get(...), window(...), register_view(...), materialize(...), backfill(...)

The tests will detect what you expose and adapt. If your names differ slightly,
adjust the small resolver helpers below (search for "RESOLVER").
"""

# ---------------------------------------------------------------------
# Import candidates — tweak if your module is elsewhere
# ---------------------------------------------------------------------
IMPORT_PATH_CANDIDATES = [
    "backend.features.feature_store",
    "backend.ml.feature_store",
    "backend.analytics.feature_store",
    "features.feature_store",
    "feature_store",
]

# ---------------------------------------------------------------------
# Load + resolve API
# ---------------------------------------------------------------------
def load_module():
    last = None
    for p in IMPORT_PATH_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import feature_store from {IMPORT_PATH_CANDIDATES} ({last})")

def _ts(dt_or_int):
    if isinstance(dt_or_int, (int, float)):
        return int(dt_or_int)
    if isinstance(dt_or_int, datetime):
        if dt_or_int.tzinfo is None:
            dt_or_int = dt_or_int.replace(tzinfo=timezone.utc)
        return int(dt_or_int.timestamp() * 1000)
    raise TypeError(f"Unsupported ts type: {type(dt_or_int)}")

class API:
    def __init__(self, mod):
        self.mod = mod
        self.obj = None
        # ====== RESOLVER: prefer class FeatureStore if present ==========
        if hasattr(mod, "FeatureStore"):
            FS = getattr(mod, "FeatureStore")
            try:
                self.obj = FS()
            except TypeError:
                self.obj = FS  # maybe static methods; we’ll call on class

    def has(self, name):
        if self.obj and hasattr(self.obj, name):
            return True
        return hasattr(self.mod, name)

    def call(self, name, *args, **kw):
        if self.obj and hasattr(self.obj, name):
            return getattr(self.obj, name)(*args, **kw)
        if hasattr(self.mod, name):
            return getattr(self.mod, name)(*args, **kw)
        raise AttributeError(name)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def api():
    mod = load_module()
    return API(mod)

@pytest.fixture()
def now_ms():
    return int(time.time() * 1000)

@pytest.fixture()
def ts_range(now_ms):
    start = now_ms - 2 * 60_000
    mid = now_ms - 60_000
    end = now_ms
    return start, mid, end

@pytest.fixture()
def entity():
    return "AAPL"

@pytest.fixture()
def schema_basic():
    # Example schema: {feature: python/type or str}
    return {
        "zscore": float,
        "rsi14": float,
        "sentiment": float,
        "adv": int,
        "sector": str,
    }

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_register_view_and_put_get_roundtrip(api, entity, ts_range, schema_basic):
    start, mid, end = ts_range

    if api.has("register_view"):
        api.call("register_view", name="alpha_core", keys=list(schema_basic.keys()), ttl_s=3600, schema=schema_basic)

    # write at two timestamps
    api.call("put", entity=entity, features={"zscore": -1.2, "rsi14": 31.0, "sentiment": 0.15, "adv": 8_000_000, "sector": "Tech"}, as_of=start)
    api.call("put", entity=entity, features={"zscore": -0.6, "rsi14": 42.5, "sentiment": 0.20}, as_of=end)

    # read as_of mid -> returns first row values (start candles)
    got_mid = api.call("get", entity=entity, keys=["zscore", "rsi14", "sentiment", "adv", "sector"], as_of=mid)
    assert isinstance(got_mid, dict)
    assert got_mid["zscore"] == pytest.approx(-1.2)
    assert got_mid["rsi14"] == pytest.approx(31.0)
    assert got_mid["adv"] == 8_000_000
    assert got_mid["sector"] == "Tech"

    # read as_of end -> latest values
    got_end = api.call("get", entity=entity, keys=["zscore", "rsi14", "sentiment", "adv", "sector"], as_of=end)
    assert got_end["zscore"] == pytest.approx(-0.6)
    assert got_end["rsi14"] == pytest.approx(42.5)
    assert got_end["sentiment"] == pytest.approx(0.20)
    # non-updated columns should forward-fill (if store supports it), else may be missing — accept either
    if "adv" in got_end:
        assert got_end["adv"] == 8_000_000


def test_ttl_expiry_optional(api, entity, now_ms):
    if not api.has("put") or not api.has("get"):
        pytest.skip("No put/get API")

    t_old = now_ms - 10_000
    t_new = now_ms

    # Insert with TTL 5s -> 5000ms
    api.call("put", entity=entity, features={"sentiment": 0.9}, as_of=t_old, ttl_s=5)
    api.call("put", entity=entity, features={"sentiment": 0.1}, as_of=t_new)

    # After a short wait, the old one should be expired when queried as_of=now
    got = api.call("get", entity=entity, keys=["sentiment"], as_of=now_ms)
    assert got["sentiment"] == pytest.approx(0.1)

    # If store enforces TTL strictly per as_of, reading at t_old+1s should still see 0.9 before expiry.
    got2 = api.call("get", entity=entity, keys=["sentiment"], as_of=t_old + 1000)
    # Accept either 0.9 (strict TTL) or 0.1 (if store always resolves to latest <= as_of ignoring TTL)
    assert got2["sentiment"] in (pytest.approx(0.9), pytest.approx(0.1))


def test_window_fetch_and_ordering(api, entity, ts_range):
    if not api.has("window"):
        pytest.skip("No window() API; skipping")

    start, mid, end = ts_range
    # Write three rows around the window
    api.call("put", entity=entity, features={"zscore": -2.0}, as_of=start - 10_000)
    api.call("put", entity=entity, features={"zscore": -1.0}, as_of=mid)
    api.call("put", entity=entity, features={"zscore":  0.5}, as_of=end + 5_000)

    out = api.call("window", entity=entity, keys=["zscore"], start=start, end=end)
    assert isinstance(out, list)
    # Expect only rows whose ts in [start, end]; and ascending by ts
    ts_list = [row.get("ts") or row.get("_ts") for row in out]
    assert ts_list == sorted(ts_list)
    # There should be at least the 'mid' record inside the window.
    assert any(abs(t - mid) <= 5 for t in ts_list)  # tolerate ms rounding


def test_batch_get_multiple_entities(api, now_ms):
    if not api.has("batch_get"):
        pytest.skip("No batch_get() API; skipping")

    e1, e2 = "AAPL", "MSFT"
    t1 = now_ms - 1000
    t2 = now_ms

    api.call("put", entity=e1, features={"rsi14": 30.0, "adv": 8_000_000}, as_of=t1)
    api.call("put", entity=e2, features={"rsi14": 55.0, "adv": 5_000_000}, as_of=t2)

    res = api.call("batch_get", entities=[e1, e2], keys=["rsi14", "adv"], as_of=t2)
    assert isinstance(res, dict) and e1 in res and e2 in res
    assert res[e1]["rsi14"] == pytest.approx(30.0)
    assert res[e2]["adv"] == 5_000_000


def test_schema_enforcement_optional(api, entity, now_ms, schema_basic):
    # If your store enforces schema (types / required keys), invalid writes should raise
    if not api.has("register_view"):
        pytest.skip("No register_view() => schema may not be supported.")

    api.call("register_view", name="risk_view", keys=["vol20", "beta"], ttl_s=3600,
             schema={"vol20": float, "beta": float})

    # valid
    api.call("put", entity=entity, features={"vol20": 0.28, "beta": 1.2}, as_of=now_ms)

    # invalid type
    with pytest.raises(Exception):
        api.call("put", entity=entity, features={"vol20": "oops", "beta": 1.0}, as_of=now_ms + 1)


def test_materialize_and_backfill_hooks(api, entity, ts_range):
    # We emulate a feature computation function "fn" that the store calls for each ts
    if not api.has("materialize") and not api.has("backfill"):
        pytest.skip("No materialize/backfill API")

    start, mid, end = ts_range

    def fake_fn(entity: str, ts: int, **kw) -> Dict[str, Any]:
        # deterministic features from ts to verify outputs
        return {
            "intraday_vol": round(((ts // 1000) % 1000) / 1000.0, 6),
            "hour": int(datetime.utcfromtimestamp(ts/1000).strftime("%H")),
        }

    # Either call materialize OR backfill depending on availability
    if api.has("materialize"):
        api.call("materialize", view="intraday", entities=[entity], start=start, end=end, fn=fake_fn)
    if api.has("backfill"):
        api.call("backfill", view="intraday", entities=[entity], start=start, end=end, fn=fake_fn)

    got = api.call("get", entity=entity, keys=["intraday_vol", "hour"], as_of=end)
    assert isinstance(got, dict)
    assert "intraday_vol" in got and "hour" in got
    assert 0.0 <= float(got["intraday_vol"]) <= 1.0
    assert 0 <= int(got["hour"]) <= 23


def test_upsert_idempotency_and_monotonic_timestamps(api, entity, now_ms):
    # Writing the same (entity, ts) twice should be idempotent (no duplicate rows),
    # and a later write should not affect earlier as_of reads.
    t = now_ms
    api.call("put", entity=entity, features={"sentiment": 0.2}, as_of=t)
    api.call("put", entity=entity, features={"sentiment": 0.2}, as_of=t)  # duplicate

    early = api.call("get", entity=entity, keys=["sentiment"], as_of=t - 1)
    assert early == {} or early.get("sentiment") is None  # nothing before t

    late = api.call("get", entity=entity, keys=["sentiment"], as_of=t + 1)
    assert late["sentiment"] == pytest.approx(0.2)


def test_json_round_trip_and_nan_handling(api, entity, now_ms):
    # Stores should remain JSON-serializable; NaN should be handled (omit or convert)
    api.call("put", entity=entity, features={"weird": float("nan"), "ok": 1.23}, as_of=now_ms)
    out = api.call("get", entity=entity, keys=["weird", "ok"], as_of=now_ms)
    # Ensure round-trip to JSON doesn't blow up
    s = json.dumps(out, allow_nan=True)
    assert isinstance(s, str) and len(s) > 2
    # If NaN preserved, JSON may render it as NaN if allow_nan=True — allow either None or NaN stringy
    assert "ok" in out and math.isfinite(float(out["ok"]))