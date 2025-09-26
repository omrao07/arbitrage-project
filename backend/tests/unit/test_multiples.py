# test_multiples.py
import math
import pytest # type: ignore
from multiples import ( # type: ignore
    ev_to_ebitda,
    pe_ratio,
    ps_ratio,
    pb_ratio,
    fair_price_from_multiple,
)

# ------------------------------
# EV / EBITDA
# ------------------------------

def test_ev_to_ebitda_basic():
    ev = 1200.0
    ebitda = 200.0
    result = ev_to_ebitda(ev, ebitda)
    expected = 6.0
    assert math.isclose(result, expected, rel_tol=1e-12)

def test_ev_to_ebitda_zero_ebitda():
    with pytest.raises(ValueError):
        ev_to_ebitda(1000.0, 0.0)

# ------------------------------
# P/E ratio
# ------------------------------

def test_pe_ratio_basic():
    price = 50.0
    eps = 5.0
    result = pe_ratio(price, eps)
    assert result == 10.0

def test_pe_ratio_zero_eps():
    with pytest.raises(ValueError):
        pe_ratio(50.0, 0.0)

# ------------------------------
# P/S ratio
# ------------------------------

def test_ps_ratio_basic():
    price = 30.0
    sales_per_share = 10.0
    result = ps_ratio(price, sales_per_share)
    assert result == 3.0

def test_ps_ratio_zero_sales():
    with pytest.raises(ValueError):
        ps_ratio(30.0, 0.0)

# ------------------------------
# P/B ratio
# ------------------------------

def test_pb_ratio_basic():
    price = 25.0
    bvps = 5.0
    result = pb_ratio(price, bvps)
    assert result == 5.0

def test_pb_ratio_zero_book():
    with pytest.raises(ValueError):
        pb_ratio(25.0, 0.0)

# ------------------------------
# Fair price from multiple
# ------------------------------

def test_fair_price_from_multiple_basic():
    multiple = 12.0
    metric = 2.5
    fair_price = fair_price_from_multiple(multiple, metric)
    expected = 30.0
    assert math.isclose(fair_price, expected, rel_tol=1e-12)

def test_fair_price_from_multiple_zero_metric():
    fair_price = fair_price_from_multiple(12.0, 0.0)
    assert fair_price == 0.0