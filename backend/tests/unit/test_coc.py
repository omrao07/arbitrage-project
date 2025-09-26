# test_coc.py
import math
import pytest # type: ignore
from coc import compute_wacc, compute_cost_of_equity # type: ignore

def test_compute_wacc_basic():
    equity_value = 1000
    debt_value = 400
    cash = 100
    cost_of_equity = 0.10
    cost_of_debt = 0.05
    tax_rate = 0.25

    wacc = compute_wacc(
        equity_value=equity_value,
        debt_value=debt_value,
        cash=cash,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        tax_rate=tax_rate
    )
    # Manual check
    V = equity_value + debt_value - cash
    expected = (equity_value/V)*cost_of_equity + (debt_value/V)*cost_of_debt*(1-tax_rate)
    assert math.isclose(wacc, expected, rel_tol=1e-6)


def test_compute_cost_of_equity_capm():
    risk_free = 0.03
    beta = 1.2
    market_return = 0.08

    coe = compute_cost_of_equity(risk_free, beta, market_return)
    expected = risk_free + beta*(market_return - risk_free)
    assert math.isclose(coe, expected, rel_tol=1e-6)


def test_negative_debt_value():
    """If debt value is negative, WACC should raise ValueError"""
    with pytest.raises(ValueError):
        compute_wacc(
            equity_value=1000,
            debt_value=-50,
            cash=0,
            cost_of_equity=0.1,
            cost_of_debt=0.05,
            tax_rate=0.25
        )