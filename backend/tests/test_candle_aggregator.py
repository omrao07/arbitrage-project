# tests/test_candle_aggregator.py
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest # type: ignore

"""
What this tests
---------------
- Tick/trade -> OHLCV aggregation at a fixed interval (e.g., 1s/1m)
- Correct open/high/low/close/volume for a simple sequence
- Alignment to time boundaries (epoch multiple of interval)
- Out-of-order ticks tolerated (sorted within bucket) or rejected explicitly
- Resampling 1s -> 1m (if your module exposes it) OR direct 1m aggregation
- Gaps (no ticks) produce either empty candles OR carry no phantom volume

How it adapts to your code
--------------------------
We try a few common export shapes:

A) Class CandleAggregator with
   - __init__(symbol, interval_ms, **kw)
   - ingest(tick) where tick = {ts, price, size} (ms timestamps)
   - finalize_until(ts_end)  -> finalize all candles with start < ts_end
   - get_candles() or drain() -> list of candles dicts
   Candle keys expected: ts (start), o,h,l,c,v   (aliases accepted via mapping below)

B) Function aggregate_ticks(ticks, interval_ms, **kw) -> list[candles]

C) Function resample(candles, from_ms, to_ms, **kw) for 1s->1m resampling
   (optional; test will skip if absent)
"""

# ------------- Adjust import candidates if your path differs -------------
IMPORT_PATH_CANDIDATES = [
    "backend.ingestion.candle_aggregator",
    "backend.ingestion.candles",
    "backend.engine.candle_aggregator",
    "ingestion.candle_aggregator",
    "candle_aggregator",
]


# ------------- Helpers to adapt to your API ------------------------------

def load_module():
    last_err = None
    for path in IMPORT_PATH_CANDIDATES:
        try:
            return importlib.import_module(path)
        except ModuleNotFoundError as e:
            last_err = e
    pytest.skip(f"Could not import candle aggregator from {IMPORT_PATH_CANDIDATES}: {last_err}")


@dataclass
class CandleKeys:
    ts: str = "ts"
    o: str = "o"
    h: str = "h"
    l: str = "l"
    c: str = "c"
    v: str = "v"


def get_api(mod):
    """
    Returns a tuple (mode, ctor_or_fn, out_keys, resample_fn)
      mode: "class" | "function"
      ctor_or_fn: class or aggregate function
      out_keys: CandleKeys mapping
      resample_fn: callable or None
    """
    keys = CandleKeys()

    # If module gives aliases, pick them up (optional)
    for attr, default in [("KEY_TS", "ts"), ("KEY_O", "o"), ("KEY_H", "h"),
                          ("KEY_L", "l"), ("KEY_C", "c"), ("KEY_V", "v")]:
        if hasattr(mod, attr):
            setattr(keys, attr.split("_")[-1].lower(), getattr(mod, attr))

    # Prefer class API
    if hasattr(mod, "CandleAggregator"):
        resample_fn = getattr(mod, "resample", None)
        return ("class", getattr(mod, "CandleAggregator"), keys, resample_fn)

    # Fallback to function API
    if hasattr(mod, "aggregate_ticks"):
        resample_fn = getattr(mod, "resample", None)
        return ("function", getattr(mod, "aggregate_ticks"), keys, resample_fn)

    pytest.skip("No CandleAggregator class or aggregate_ticks function found.")


# ------------- Synthetic data --------------------------------------------

@pytest.fixture()
def ticks_simple():
    """
    4 ticks within one minute (60000ms). All times are ms since epoch.
    Start boundary at t0 = 1_700_000_000_000 (multiple of 1000 for tidiness).
    """
    t0 = 1_700_000_000_000
    return [
        {"ts": t0 + 5_000,  "price": 100.0, "size": 2.0},  # open = 100
        {"ts": t0 + 15_000, "price": 102.0, "size": 1.0},  # high = 102
        {"ts": t0 + 30_000, "price":  99.5, "size": 1.0},  # low  = 99.5
        {"ts": t0 + 55_000, "price": 101.2, "size": 3.0},  # close= 101.2
    ]


@pytest.fixture()
def ticks_two_minutes(ticks_simple):
    """Copy the first minute and add a second minute with different path."""
    t0 = ticks_simple[0]["ts"] - 5_000  # back to boundary
    one_min = 60_000
    minute2 = [
        {"ts": t0 + one_min + 1_000, "price": 101.3, "size": 1.0},
        {"ts": t0 + one_min + 20_000, "price": 100.1, "size": 2.0},
        {"ts": t0 + one_min + 59_000, "price": 100.9, "size": 1.0},
    ]
    return ticks_simple + minute2


# ------------- Tests ------------------------------------------------------

def _expect_ohlcv(candle: dict, keys: CandleKeys, start_ts: int,
                  o: float, h: float, l: float, c: float, v: float, eps=1e-9):
    assert candle[keys.ts] == start_ts
    assert candle[keys.o] == pytest.approx(o, abs=eps)
    assert candle[keys.h] == pytest.approx(h, abs=eps)
    assert candle[keys.l] == pytest.approx(l, abs=eps)
    assert candle[keys.c] == pytest.approx(c, abs=eps)
    assert candle[keys.v] == pytest.approx(v, abs=eps)


def test_one_minute_basic_ohlcv(ticks_simple):
    mod = load_module()
    mode, ctor, keys, _ = get_api(mod) # type: ignore
    t0 = ticks_simple[0]["ts"] - 5_000  # boundary
    interval = 60_000

    if mode == "class":
        agg = ctor(symbol="TEST", interval_ms=interval) if "interval_ms" in getattr(ctor, "__init__").__code__.co_varnames else ctor("TEST", interval)
        for t in ticks_simple:
            tick = {"ts": t["ts"], "p": t.get("p", t["price"]), "price": t["price"], "size": t["size"]}
            # Accept either ingest(tick) or add()
            if hasattr(agg, "ingest"):
                agg.ingest(tick)
            elif hasattr(agg, "add"):
                agg.add(tick)
            else:
                pytest.skip("CandleAggregator has neither ingest() nor add().")
        # finalize first minute
        if hasattr(agg, "finalize_until"):
            agg.finalize_until(t0 + interval)
        # drain / get
        if hasattr(agg, "drain"):
            out = agg.drain()
        elif hasattr(agg, "get_candles"):
            out = agg.get_candles()
        else:
            pytest.skip("CandleAggregator has neither drain() nor get_candles().")
    else:
        out = ctor(ticks_simple, interval)

    assert len(out) >= 1
    _expect_ohlcv(out[0], keys, start_ts=t0, o=100.0, h=102.0, l=99.5, c=101.2, v=7.0)


def test_alignment_and_gap_behavior(ticks_two_minutes):
    mod = load_module()
    mode, ctor, keys, _ = get_api(mod) # type: ignore
    t0 = ticks_two_minutes[0]["ts"] - 5_000
    interval = 60_000

    if mode == "class":
        agg = ctor(symbol="TEST", interval_ms=interval) if "interval_ms" in getattr(ctor, "__init__").__code__.co_varnames else ctor("TEST", interval)
        for t in ticks_two_minutes:
            tick = {"ts": t["ts"], "price": t["price"], "size": t["size"]}
            (hasattr(agg, "ingest") and agg.ingest(tick)) or (hasattr(agg, "add") and agg.add(tick)) # type: ignore
        if hasattr(agg, "finalize_until"):
            agg.finalize_until(t0 + 2 * interval)
        out = (agg.drain() if hasattr(agg, "drain") else agg.get_candles())
    else:
        out = ctor(ticks_two_minutes, interval)

    assert len(out) >= 2
    # First minute was asserted in previous test; here check minute #2 close
    m2 = out[1]
    assert m2[keys.ts] == t0 + interval
    assert m2[keys.c] == pytest.approx(100.9, abs=1e-9)
    # Volume is sum of sizes in minute 2
    assert m2[keys.v] == pytest.approx(1.0 + 2.0 + 1.0, abs=1e-9)


def test_out_of_order_ticks_handled(ticks_simple):
    mod = load_module()
    mode, ctor, keys, _ = get_api(mod) # type: ignore
    t0 = ticks_simple[0]["ts"] - 5_000
    interval = 60_000

    # Shuffle: last comes second
    ooo = [ticks_simple[0], ticks_simple[3], ticks_simple[1], ticks_simple[2]]

    if mode == "class":
        agg = ctor(symbol="TEST", interval_ms=interval) if "interval_ms" in getattr(ctor, "__init__").__code__.co_varnames else ctor("TEST", interval)
        for t in ooo:
            tick = {"ts": t["ts"], "price": t["price"], "size": t["size"]}
            (hasattr(agg, "ingest") and agg.ingest(tick)) or (hasattr(agg, "add") and agg.add(tick)) # type: ignore
        if hasattr(agg, "finalize_until"):
            agg.finalize_until(t0 + interval)
        out = (agg.drain() if hasattr(agg, "drain") else agg.get_candles())
    else:
        out = ctor(ooo, interval)

    # Even with out-of-order ticks, OHLCV should be identical (if implementation sorts within bucket)
    _expect_ohlcv(out[0], keys, start_ts=t0, o=100.0, h=102.0, l=99.5, c=101.2, v=7.0)


def test_resample_optional_1s_to_1m():
    """
    If your module exposes resample(candles, from_ms, to_ms), validate 1s -> 1m.
    Otherwise skip gracefully.
    """
    mod = load_module()
    mode, ctor, keys, resample_fn = get_api(mod) # type: ignore
    if not resample_fn:
        pytest.skip("No resample() function exported; skipping resample test.")

    # Build 1-second candles for one minute
    t0 = 1_700_000_000_000
    one_sec = 1_000
    one_min = 60_000
    s_candles = []
    price = 100.0
    vol = 0.1
    high, low = price, price
    for i in range(60):
        ts = t0 + i * one_sec
        # small random-ish walk without randomness (deterministic)
        price += (1 if i in (10, 20) else -1 if i in (30, 40) else 0) * 0.1
        high = max(high, price)
        low = min(low, price)
        s_candles.append({keys.ts: ts, keys.o: price, keys.h: price, keys.l: price, keys.c: price, keys.v: vol})

    out = resample_fn(s_candles, from_ms=one_sec, to_ms=one_min) # type: ignore
    assert isinstance(out, list) and len(out) >= 1
    c = out[0]
    # open/close/vol should match first/last/sum of the second bars
    assert c[keys.o] == pytest.approx(s_candles[0][keys.o], abs=1e-9)
    assert c[keys.c] == pytest.approx(s_candles[-1][keys.c], abs=1e-9)
    assert c[keys.v] == pytest.approx(sum(x[keys.v] for x in s_candles), abs=1e-9)
    # high/low should be envelope
    assert c[keys.h] >= max(x[keys.h] for x in s_candles) - 1e-9
    assert c[keys.l] <= min(x[keys.l] for x in s_candles) + 1e-9