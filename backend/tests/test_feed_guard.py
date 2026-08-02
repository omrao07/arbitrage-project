"""
FeedGuard tests: a missing, stale, or malformed feed must RAISE.

These lock down the data-layer half of D-2. The failure they prevent is not
"the system crashed" — it is "the system kept trading on a price that was
either invented or ten seconds old, and nothing in the UI said so".
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from backend.data.feed_guard import (
    FeedGuard,
    MalformedTickError,
    MissingFeedError,
    StaleDataError,
    clean_series,
)


@pytest.fixture
def now_ms():
    return time.time() * 1000.0


@pytest.fixture
def guard():
    return FeedGuard(max_staleness_ms=2000)


def test_fresh_tick_passes(guard, now_ms):
    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms}
    assert guard.validate_tick(tick, "NIFTY", now_ms=now_ms)["ltp"] == 100.0


def test_none_tick_raises(guard):
    with pytest.raises(MissingFeedError, match="No tick"):
        guard.validate_tick(None, "NIFTY")


def test_stale_tick_raises(guard, now_ms):
    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms - 5000}
    with pytest.raises(StaleDataError, match="stale"):
        guard.validate_tick(tick, "NIFTY", now_ms=now_ms)


def test_tick_just_inside_tolerance_passes(guard, now_ms):
    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms - 1900}
    assert guard.validate_tick(tick, "NIFTY", now_ms=now_ms)


def test_missing_timestamp_raises(guard):
    with pytest.raises(MalformedTickError, match="exchange_ts_ms"):
        guard.validate_tick({"ltp": 100.0}, "NIFTY")


def test_future_timestamp_raises(guard, now_ms):
    """Clock skew must be fatal — it would otherwise mask real staleness."""
    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms + 60_000}
    with pytest.raises(MalformedTickError, match="FUTURE"):
        guard.validate_tick(tick, "NIFTY", now_ms=now_ms)


@pytest.mark.parametrize("bad_price", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_price_raises(guard, now_ms, bad_price):
    tick = {"ltp": bad_price, "exchange_ts_ms": now_ms}
    with pytest.raises(MissingFeedError):
        guard.validate_tick(tick, "NIFTY", now_ms=now_ms)


def test_missing_price_raises(guard, now_ms):
    with pytest.raises(MissingFeedError, match="no ltp"):
        guard.validate_tick({"exchange_ts_ms": now_ms}, "NIFTY", now_ms=now_ms)


def test_non_numeric_price_raises(guard, now_ms):
    tick = {"ltp": "not-a-number", "exchange_ts_ms": now_ms}
    with pytest.raises(MalformedTickError, match="not numeric"):
        guard.validate_tick(tick, "NIFTY", now_ms=now_ms)


def test_crossed_book_raises(guard, now_ms):
    """bid > ask is bad data, not free money."""
    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms, "bid": 101.0, "ask": 99.0}
    with pytest.raises(MalformedTickError, match="crossed book"):
        guard.validate_tick(tick, "NIFTY", now_ms=now_ms)


def test_normal_book_passes(guard, now_ms):
    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms, "bid": 99.5, "ask": 100.5}
    assert guard.validate_tick(tick, "NIFTY", now_ms=now_ms)


# ── Per-instrument and per-regime tolerance ─────────────────────────────────

def test_per_instrument_override(now_ms):
    g = FeedGuard(max_staleness_ms=2000, overrides={"ILLIQUID": 10_000})
    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms - 5000}
    assert g.validate_tick(tick, "ILLIQUID", now_ms=now_ms)   # within its own 10s
    with pytest.raises(StaleDataError):
        g.validate_tick(tick, "NIFTY", now_ms=now_ms)          # but not the 2s default


def test_expiry_regime_tightens_tolerance(now_ms):
    """2s of staleness during a gamma spike is a different risk than at midday."""
    normal = FeedGuard(max_staleness_ms=2000, regime="normal")
    expiry = FeedGuard(max_staleness_ms=2000, regime="expiry")
    assert expiry.tolerance_for("NIFTY") < normal.tolerance_for("NIFTY")

    tick = {"ltp": 100.0, "exchange_ts_ms": now_ms - 1500}
    assert normal.validate_tick(tick, "NIFTY", now_ms=now_ms)
    with pytest.raises(StaleDataError):
        expiry.validate_tick(tick, "NIFTY", now_ms=now_ms)


# ── Observability ───────────────────────────────────────────────────────────

def test_stats_track_each_failure_mode(guard, now_ms):
    guard.validate_tick({"ltp": 1.0, "exchange_ts_ms": now_ms}, "X", now_ms=now_ms)
    for tick, exc in [
        (None, MissingFeedError),
        ({"ltp": 1.0, "exchange_ts_ms": now_ms - 9999}, StaleDataError),
        ({"ltp": 1.0}, MalformedTickError),
    ]:
        with pytest.raises(exc):
            guard.validate_tick(tick, "X", now_ms=now_ms)
    assert guard.stats == {"ok": 1, "missing": 1, "stale": 1, "malformed": 1}


def test_is_healthy_reflects_failure_ratio(guard, now_ms):
    for _ in range(99):
        guard.validate_tick({"ltp": 1.0, "exchange_ts_ms": now_ms}, "X", now_ms=now_ms)
    assert guard.is_healthy()
    with pytest.raises(MissingFeedError):
        guard.validate_tick(None, "X")
    assert guard.is_healthy(min_ok_ratio=0.95)      # 99/100
    assert not guard.is_healthy(min_ok_ratio=0.999)


# ── clean_series: forward-fill is a narrow, named carve-out ─────────────────

def test_clean_series_replaces_inf_with_nan():
    out = clean_series([1.0, np.inf, 3.0, -np.inf], "prev_close")
    assert int(out.isna().sum()) == 2


def test_clean_series_refuses_to_ffill_a_price_series():
    """Forward-filling a live price is the fabrication bug in a nicer costume."""
    with pytest.raises(MalformedTickError, match="refusing to forward-fill"):
        clean_series([1.0, np.nan, 3.0], "ltp", allow_ffill=True)


def test_clean_series_allows_ffill_for_named_reference_series():
    out = clean_series([100.0, np.nan, np.nan], "prev_day_oi", allow_ffill=True, ffill_limit=1)
    assert out.iloc[1] == 100.0      # filled, within limit
    assert bool(np.isnan(out.iloc[2]))  # limit respected, not filled forever


def test_clean_series_leaves_gaps_when_ffill_not_requested():
    out = clean_series([1.0, np.nan, 3.0], "ltp")
    assert bool(out.isna().iloc[1])
