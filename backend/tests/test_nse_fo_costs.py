"""
Cost-model tests. A wrong cost function makes every backtest a lie, so these
assert the structural invariants rather than just a golden number.

Note on the STT asymmetry, which is the whole point: STT is charged on the SELL
leg of an option and stamp duty on the BUY leg. Short-premium strategies
therefore pay the (higher, post-Apr-2026) STT on entry, on the larger premium.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.costs import nse_fo as C

LOT = 75  # NIFTY


# ── Structural invariants ───────────────────────────────────────────────────

@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("premium", [0.05, 1.0, 50.0, 500.0, 5000.0])
@pytest.mark.parametrize("lots", [1, 7, 100])
def test_option_costs_are_non_negative(premium, lots, side):
    c = C.option_costs(premium, lots, LOT, side)
    assert c.total > 0
    for field in ("brokerage", "stt", "exchange_txn", "sebi_fee", "gst", "stamp_duty"):
        assert getattr(c, field) >= 0


@pytest.mark.parametrize("premium", [0.05, 50.0, 5000.0])
def test_stt_applies_only_to_the_sell_leg(premium):
    assert C.option_costs(premium, 1, LOT, "SELL").stt > 0
    assert C.option_costs(premium, 1, LOT, "BUY").stt == 0


@pytest.mark.parametrize("premium", [0.05, 50.0, 5000.0])
def test_stamp_duty_applies_only_to_the_buy_leg(premium):
    assert C.option_costs(premium, 1, LOT, "BUY").stamp_duty > 0
    assert C.option_costs(premium, 1, LOT, "SELL").stamp_duty == 0


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_gst_is_18pct_of_brokerage_txn_and_sebi_only(side):
    """GST is a tax on the fees, not on STT/stamp. Getting this wrong inflates cost."""
    c = C.option_costs(100.0, 3, LOT, side)
    expected = C.GST_RATE * (c.brokerage + c.exchange_txn + c.sebi_fee)
    assert c.gst == pytest.approx(expected, rel=1e-12)


def test_total_equals_sum_of_components():
    c = C.option_costs(123.45, 4, LOT, "SELL")
    assert c.total == pytest.approx(
        c.brokerage + c.stt + c.exchange_txn + c.sebi_fee + c.gst + c.stamp_duty
    )


def test_cost_is_monotonic_in_premium():
    prev = 0.0
    for premium in (1.0, 10.0, 100.0, 1000.0):
        total = C.option_costs(premium, 1, LOT, "SELL").total
        assert total > prev
        prev = total


def test_cost_is_monotonic_in_lots():
    prev = 0.0
    for lots in (1, 2, 10, 50):
        total = C.option_costs(100.0, lots, LOT, "SELL").total
        assert total > prev
        prev = total


def test_sell_leg_costs_more_than_buy_leg_at_equal_premium():
    """STT (0.15%) dwarfs stamp duty (0.003%), so selling is the expensive side."""
    sell = C.option_costs(100.0, 1, LOT, "SELL").total
    buy = C.option_costs(100.0, 1, LOT, "BUY").total
    assert sell > buy


# ── Exact arithmetic against the published rate table ───────────────────────

def test_sell_leg_matches_hand_computation():
    premium, lots = 100.0, 1
    V = premium * lots * LOT                       # 7,500
    c = C.option_costs(premium, lots, LOT, "SELL")
    assert c.brokerage == pytest.approx(20.0)
    assert c.stt == pytest.approx(0.0015 * V)      # 11.25
    assert c.exchange_txn == pytest.approx(0.0003503 * V)
    assert c.sebi_fee == pytest.approx(0.000001 * V)
    assert c.stamp_duty == 0.0
    assert c.total == pytest.approx(37.96, abs=0.01)


def test_futures_brokerage_is_capped_at_the_flat_fee():
    """0.03% of turnover, capped at Rs 20."""
    small = C.future_costs(100.0, 1, 1, "BUY")     # 0.03% of 100 = 0.03
    assert small.brokerage == pytest.approx(0.03)
    large = C.future_costs(25_000.0, 1, LOT, "BUY")  # 0.03% of 1.875mm >> 20
    assert large.brokerage == pytest.approx(C.BROKERAGE_FLAT)


def test_futures_stt_is_sell_side_only():
    assert C.future_costs(25_000.0, 1, LOT, "SELL").stt > 0
    assert C.future_costs(25_000.0, 1, LOT, "BUY").stt == 0


# ── Vectorised path must agree with the scalar path ─────────────────────────

def test_vectorised_option_costs_match_scalar():
    premiums = np.array([10.0, 100.0, 250.0, 1000.0])
    sells = np.array([True, False, True, False])
    vec = C.option_costs_vec(premiums, np.ones(4), LOT, sells)
    for i, (p, s) in enumerate(zip(premiums, sells)):
        scalar = C.option_costs(float(p), 1, LOT, "SELL" if s else "BUY").total
        assert vec[i] == pytest.approx(scalar)


def test_vectorised_future_costs_match_scalar():
    prices = np.array([20_000.0, 25_000.0])
    sells = np.array([True, False])
    vec = C.future_costs_vec(prices, np.ones(2), LOT, sells)
    for i, (p, s) in enumerate(zip(prices, sells)):
        scalar = C.future_costs(float(p), 1, LOT, "SELL" if s else "BUY").total
        assert vec[i] == pytest.approx(scalar)


# ── Data-integrity: bad input HALTS, never returns a plausible number ───────

@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_premium_raises(bad):
    with pytest.raises(ValueError):
        C.option_costs_vec(np.array([bad]), np.array([1.0]), LOT, np.array([True]))


def test_invalid_lot_size_raises():
    with pytest.raises(ValueError):
        C.option_costs_vec(np.array([100.0]), np.array([1.0]), 0, np.array([True]))


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        C.option_costs(100.0, 1, LOT, "HOLD")


# ── Round trip & the regression this module exists to prevent ───────────────

def test_short_round_trip_charges_stt_on_the_larger_entry_premium():
    """Sell-to-open at 100, buy-to-close at 50: STT hits the 100."""
    rt = C.option_round_trip(100.0, 50.0, 1, LOT, short=True)
    entry = C.option_costs(100.0, 1, LOT, "SELL")
    exit_ = C.option_costs(50.0, 1, LOT, "BUY")
    assert rt == pytest.approx(entry.total + exit_.total)
    assert entry.stt > 0 and exit_.stt == 0


def test_real_cost_dwarfs_the_legacy_flat_fee_assumption():
    """
    Regression guard for the actual bug: backtester/shadow_runner.py charges
    fee_bps=0.3. On a 1-lot NIFTY option round trip that is ~0.45 INR against a
    real ~63 INR. If this ratio ever collapses, someone has reintroduced a
    flat-bps cost model.
    """
    V = 100.0 * 1 * LOT
    real = C.option_round_trip(100.0, 50.0, 1, LOT, short=True)
    legacy = V * 0.3 / 1e4 * 2
    assert real / legacy > 50


def test_verify_against_contract_note_flags_a_mismatch():
    ok = C.verify_against_contract_note(100.0, 1, LOT, "SELL", broker_total_inr=37.96)
    assert ok["matches"] is True
    bad = C.verify_against_contract_note(100.0, 1, LOT, "SELL", broker_total_inr=5.0)
    assert bad["matches"] is False
    assert bad["difference_inr"] > 0
