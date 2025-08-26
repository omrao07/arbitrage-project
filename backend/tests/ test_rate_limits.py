# tests/test_rate_limits.py
"""
Unit tests for backend/risk/rate_limits.py
Covers:
- Cap on parallel / tenor-specific shocks
- Enforcement of max/min rate boundaries
- Stress scenarios (steepener, flattener)
"""

import pytest # type: ignore
from backend.risk.rate_limits import RateLimits, RateShock # type: ignore


def test_parallel_shock_clamp():
    rl = RateLimits(max_parallel_bp=200, min_parallel_bp=-100)

    s1 = RateShock(parallel_bp=150)
    s2 = RateShock(parallel_bp=250)   # should clamp at +200
    s3 = RateShock(parallel_bp=-120)  # should clamp at -100

    assert rl.apply(s1).parallel_bp == 150
    assert rl.apply(s2).parallel_bp == 200
    assert rl.apply(s3).parallel_bp == -100


def test_tenor_shocks():
    rl = RateLimits(max_per_tenor_bp=300)
    shock = RateShock(rates_by_tenor={"2y": 100, "10y": 350})

    result = rl.apply(shock)

    # 10y should clamp at +300
    assert result.rates_by_tenor["2y"] == 100
    assert result.rates_by_tenor["10y"] == 300


def test_flattener_scenario():
    rl = RateLimits(max_per_tenor_bp=200)
    shock = RateShock(rates_by_tenor={"2y": +150, "10y": +50})

    res = rl.apply(shock)

    # Ensure no overshoot
    assert res.rates_by_tenor["2y"] <= 200
    assert res.rates_by_tenor["10y"] <= 200


def test_steepener_scenario():
    rl = RateLimits(max_per_tenor_bp=200)
    shock = RateShock(rates_by_tenor={"2y": -50, "30y": +250})

    res = rl.apply(shock)

    # 30y clamps
    assert res.rates_by_tenor["2y"] == -50
    assert res.rates_by_tenor["30y"] == 200


def test_combined_shock_and_parallel():
    rl = RateLimits(max_parallel_bp=100, max_per_tenor_bp=200)
    shock = RateShock(parallel_bp=120, rates_by_tenor={"5y": 250})

    res = rl.apply(shock)

    assert res.parallel_bp == 100
    assert res.rates_by_tenor["5y"] == 200