# test_dcf.py
import math
import pytest # type: ignore

# Assumes dcf.py exposes these functions:
# - npv(rate: float, cashflows: list[float]) -> float
# - faded_growth(years: int, g0: float, gN: float | None = None) -> list[float]
# - project_fcf(fcf0: float, growths: list[float]) -> list[float]
# - terminal_value_perpetuity(last_fcf: float, wacc: float, g: float) -> float
# - terminal_value_exit_multiple(ebitda: float, exit_multiple: float) -> float
# - enterprise_value(fcf: list[float], wacc: float, terminal_value: float, terminal_years: int) -> float
# - equity_value_per_share(ev: float, net_debt: float, shares_out: float) -> float

from dcf import ( # type: ignore
    npv,
    faded_growth,
    project_fcf,
    terminal_value_perpetuity,
    terminal_value_exit_multiple,
    enterprise_value,
    equity_value_per_share,
)

# ------------------------------
# NPV
# ------------------------------

def test_npv_known_values():
    # 3 cashflows of 100 at 10% discount
    cfs = [100.0, 100.0, 100.0]
    r = 0.10
    expected = sum(cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cfs))
    assert math.isclose(npv(r, cfs), expected, rel_tol=1e-12, abs_tol=1e-12)

def test_npv_empty_returns_zero():
    assert npv(0.1, []) == 0.0

def test_npv_zero_rate_is_sum():
    cfs = [10, 20, 30]
    assert npv(0.0, cfs) == sum(cfs)

# ------------------------------
# Growth & Projection
# ------------------------------

def test_faded_growth_linear_to_target():
    gs = faded_growth(years=5, g0=0.10, gN=0.02)
    # should linearly interpolate from 10% to 2% over 5 years
    expected = [0.10, 0.08, 0.06, 0.04, 0.02]
    for a, b in zip(gs, expected):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

def test_faded_growth_constant_when_no_target():
    gs = faded_growth(years=4, g0=0.07, gN=None)
    assert gs == [0.07, 0.07, 0.07, 0.07]

def test_project_fcf_compounds_correctly():
    fcf0 = 100.0
    gs = [0.10, 0.10, 0.10]  # 3 years of +10%
    out = project_fcf(fcf0, gs)
    expected = [110.0, 121.0, 133.1]
    for a, b in zip(out, expected):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

# ------------------------------
# Terminal Value
# ------------------------------

def test_terminal_value_perpetuity_gordon():
    last_fcf = 200.0
    wacc = 0.10
    g = 0.02
    tv = terminal_value_perpetuity(last_fcf, wacc, g)
    expected = last_fcf * (1 + g) / (wacc - g)
    assert math.isclose(tv, expected, rel_tol=1e-12, abs_tol=1e-12)

def test_terminal_value_perpetuity_raises_if_g_ge_wacc():
    with pytest.raises(ValueError):
        terminal_value_perpetuity(200.0, wacc=0.05, g=0.05)

def test_terminal_value_exit_multiple_simple():
    ebitda = 150.0
    multiple = 12.0
    tv = terminal_value_exit_multiple(ebitda, multiple)
    assert tv == ebitda * multiple

# ------------------------------
# Enterprise & Equity value
# ------------------------------

def test_enterprise_value_with_discounted_tv():
    # Setup: 5-year FCFs, WACC 9%, terminal via Gordon with g=2%
    fcf = [120, 132, 145.2, 159.72, 175.692]
    wacc = 0.09
    g = 0.02
    last = fcf[-1]
    tv = last * (1 + g) / (wacc - g)   # undiscounted TV at year 5
    # enterprise_value() is expected to discount TV by (1+wacc)^N internally
    ev = enterprise_value(fcf, wacc, tv, terminal_years=len(fcf))
    expected_ev = npv(wacc, fcf) + tv / ((1 + wacc) ** len(fcf))
    assert math.isclose(ev, expected_ev, rel_tol=1e-12, abs_tol=1e-12)

def test_equity_value_per_share_basic():
    ev = 5000.0
    net_debt = 400.0
    shares = 100.0
    px = equity_value_per_share(ev, net_debt, shares)
    expected = (ev - net_debt) / shares
    assert math.isclose(px, expected, rel_tol=1e-12, abs_tol=1e-12)

def test_equity_value_per_share_raises_on_nonpositive_shares():
    with pytest.raises(ValueError):
        equity_value_per_share(1000.0, 0.0, 0.0)

# ------------------------------
# Integrated sanity (mini DCF)
# ------------------------------

def test_mini_dcf_pipeline_consistency():
    # Inputs
    fcf0 = 100.0
    growths = [0.08, 0.07, 0.06, 0.05, 0.04]  # 5-year fade
    fcf = project_fcf(fcf0, growths)
    wacc = 0.09
    g = 0.025
    tv = terminal_value_perpetuity(fcf[-1], wacc, g)
    ev = enterprise_value(fcf, wacc, tv, terminal_years=len(fcf))
    price = equity_value_per_share(ev, net_debt=200.0, shares_out=50.0)

    # Sanity checks
    assert ev > 0
    assert price > 0
    # Higher WACC should reduce EV
    ev_higher_wacc = enterprise_value(fcf, wacc + 0.02, tv * (wacc + 0.02 - g) / (wacc - g), len(fcf))
    assert ev_higher_wacc < ev