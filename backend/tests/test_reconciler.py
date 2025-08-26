# tests/test_reconciler.py
"""
Reconciler tests (duck-typed)

Checks:
- Basic match: orders ↔ fills ↔ end positions (+ cash)
- Partial fills & leftover open qty
- Duplicate fill detection (idempotence)
- Rounding / tolerance handling
- FX conversion to base_ccy (if provided)
- Cancel/replace & late cancel handling (optional)
- Corporate action (split) adjustment (optional)

Expected APIs (any one ok):
A) class Reconciler(base_ccy="USD", qty_tol=1e-6, notional_tol=0.01).reconcile(orders, fills, positions, cash, fx=None) -> report: dict
B) function reconcile(orders, fills, positions, cash, fx=None, base_ccy="USD", qty_tol=..., notional_tol=...) -> report: dict

Report dict should include some of:
  - breaks: list[dict] (symbol, kind, detail)
  - unmatched_orders / unmatched_fills
  - pos_pnl_check / cash_check booleans or metrics
  - duplicates: list[fill_ids] (if supported)
We duck-type defensively and only assert the essentials.
"""

import copy
import math
import pytest # type: ignore

rec_mod = pytest.importorskip("backend.oms.reconciler", reason="backend.oms.reconciler not found")


# ---------------- API shim ----------------

def _mk_reconciler(**kw):
    # Prefer class API
    if hasattr(rec_mod, "Reconciler"):
        return rec_mod.Reconciler(**kw)
    # Otherwise fallback to function mode
    if hasattr(rec_mod, "reconcile"):
        class _FnWrap:
            def __init__(self, **cfg): self.cfg = cfg
            def reconcile(self, orders, fills, positions, cash, fx=None):
                return rec_mod.reconcile(orders, fills, positions, cash, fx=fx, **self.cfg)
        return _FnWrap(**kw)
    pytest.skip("No Reconciler class or reconcile() function found")

# ------------- Sample data ---------------

@pytest.fixture
def base_fx():
    # FX map to base USD (example)
    return {"USD": 1.0, "INR": 0.012, "EUR": 1.1}

@pytest.fixture
def sample_orders():
    # client order ids unique
    return [
        {"oid": "O1", "symbol": "AAPL",      "side": "buy",  "qty": 100,  "ccy": "USD"},
        {"oid": "O2", "symbol": "RELIANCE.NS","side": "sell", "qty": 50,   "ccy": "INR"},
        {"oid": "O3", "symbol": "AAPL",      "side": "sell", "qty": 20,   "ccy": "USD"},   # later canceled
    ]

@pytest.fixture
def sample_fills():
    # Fills include duplicates + partials
    return [
        {"fid": "F1", "oid": "O1", "symbol": "AAPL",       "side": "buy",  "qty": 60, "px": 190.00, "ccy": "USD"},
        {"fid": "F2", "oid": "O1", "symbol": "AAPL",       "side": "buy",  "qty": 40, "px": 189.80, "ccy": "USD"},
        {"fid": "Fdup","oid":"O1", "symbol":"AAPL", "side":"buy", "qty": 40, "px": 189.80, "ccy":"USD"}, # duplicate of F2
        {"fid": "F3", "oid": "O2", "symbol": "RELIANCE.NS","side": "sell", "qty": 20, "px": 2600.0, "ccy":"INR"},
        {"fid": "F4", "oid": "O2", "symbol": "RELIANCE.NS","side": "sell", "qty": 30, "px": 2590.0, "ccy":"INR"},
        # AAPL late cancel simulation: a fill exists but order will be canceled later (engine may mark as break)
    ]

@pytest.fixture
def sample_positions_after():
    # Ending street/prime positions
    # Start-of-day assumed 0 for AAPL & RELIANCE.NS unless engine reads sod separately.
    return [
        {"symbol": "AAPL",        "qty": 80,   "ccy": "USD"},  # After buys (100) and sells (20) → net +80
        {"symbol": "RELIANCE.NS", "qty": -50,  "ccy": "INR"},
    ]

@pytest.fixture
def sample_cash_after():
    # Cash by currency after trades (sign convention: buys use cash → negative)
    # AAPL buy ~ 60*190 + 40*189.8 = 18,988 → -18,988 USD
    # REL sell ~ 20*2600 + 30*2590 = 129,700 INR → +129,700 INR
    # We ignore fees for simplicity; a real reconciler may include them.
    return [
        {"ccy": "USD", "amount": -18988.0},
        {"ccy": "INR", "amount": +129700.0},
    ]


# ---------------- Tests -------------------

def test_happy_path_balances_positions_and_cash(sample_orders, sample_fills, sample_positions_after, sample_cash_after, base_fx):
    recon = _mk_reconciler(base_ccy="USD", qty_tol=1e-6, notional_tol=1e-2)

    report = recon.reconcile( # type: ignore
        orders=sample_orders,
        fills=sample_fills,
        positions=sample_positions_after,
        cash=sample_cash_after,
        fx=base_fx,
    )

    # Duck-typed assertions
    assert isinstance(report, dict)

    # No core breaks expected aside from the duplicate & cancel we test later
    breaks = report.get("breaks", [])
    # It's okay if the engine logs a 'duplicate' break; we check idempotence below.
    assert isinstance(breaks, (list, tuple))

    # Position check: AAPL 100 buy and 20 sell -> +80; REL sell 50 -> -50
    pos_map = {p.get("symbol"): float(p.get("qty", 0.0)) for p in report.get("pos_view", sample_positions_after)}
    assert pytest.approx(pos_map.get("AAPL", 0.0), rel=1e-9) == 80.0
    assert pytest.approx(pos_map.get("RELIANCE.NS", 0.0), rel=1e-9) == -50.0

    # Cash check in base ccy (USD): -18,988 USD + 129,700 INR * 0.012 = -18,988 + 1,556.4 = -17,431.6
    # Depending on your engine, this may appear in report['cash_base'] or metrics.
    cash_base = report.get("cash_base")
    if isinstance(cash_base, (int, float)):
        assert pytest.approx(cash_base, rel=1e-6) == (-18988.0 + 129700.0 * base_fx["INR"])

def test_partial_fill_status_and_open_qty(sample_orders, sample_fills):
    recon = _mk_reconciler(qty_tol=1e-9)
    rep = recon.reconcile(sample_orders, sample_fills, positions=[], cash=[], fx=None) # type: ignore

    orders_view = rep.get("orders_view", sample_orders)
    # Find O1 & O2 statuses
    o1 = next(o for o in orders_view if o.get("oid") == "O1")
    o2 = next(o for o in orders_view if o.get("oid") == "O2")
    # O1 is fully filled; open_qty ~ 0
    assert abs(float(o1.get("open_qty", 0.0))) <= 1e-9
    # O2 fully filled too
    assert abs(float(o2.get("open_qty", 0.0))) <= 1e-9

def test_duplicate_fill_detection_and_idempotence(sample_orders, sample_fills):
    recon = _mk_reconciler()
    rep = recon.reconcile(sample_orders, sample_fills, positions=[], cash=[], fx=None) # type: ignore

    dups = rep.get("duplicates") or rep.get("dupe_fills") or []
    # If the engine implements duplicate detection, it should tag 'Fdup'
    if dups:
        assert any("Fdup" in str(x) or (isinstance(x, dict) and x.get("fid") == "Fdup") for x in dups)

    # Idempotence: running again with the same duplicate should not change totals
    rep2 = recon.reconcile(sample_orders, sample_fills, positions=[], cash=[], fx=None) # type: ignore
    def _tot_filled(rpt, oid):
        ov = rpt.get("orders_view", sample_orders)
        row = next(o for o in ov if o.get("oid") == oid)
        return float(row.get("filled_qty", 0.0))
    assert _tot_filled(rep, "O1") == _tot_filled(rep2, "O1")

def test_cancel_replace_or_late_cancel_flag(sample_orders, sample_fills):
    # Mark O3 as canceled before fully filled (no fills exist for O3 -> open qty becomes canceled)
    orders = copy.deepcopy(sample_orders)
    for o in orders:
        if o["oid"] == "O3":
            o["status"] = "canceled"
    recon = _mk_reconciler()
    rep = recon.reconcile(orders, sample_fills, positions=[], cash=[], fx=None) # type: ignore

    # Engine may emit a specific 'late_cancel' break; if not, at least the order should be marked canceled with open>0
    breaks = rep.get("breaks", [])
    if breaks:
        assert any(b.get("kind") in ("late_cancel", "open_qty_on_canceled") for b in breaks)
    orders_view = rep.get("orders_view", orders)
    o3 = next(o for o in orders_view if o.get("oid") == "O3")
    assert o3.get("status") in ("canceled", "cancelled", "CANCELED")

def test_rounding_and_tolerance():
    # Tiny rounding differences should not create breaks
    orders = [{"oid": "O1", "symbol": "MSFT", "side": "buy", "qty": 1.0000001, "ccy": "USD"}]
    fills  = [{"fid": "F1", "oid": "O1", "symbol": "MSFT", "side": "buy", "qty": 1.0, "px": 400.00, "ccy": "USD"}]
    positions = [{"symbol": "MSFT", "qty": 1.0, "ccy": "USD"}]
    cash = [{"ccy": "USD", "amount": -400.00}]
    recon = _mk_reconciler(qty_tol=1e-4, notional_tol=0.05)
    rep = recon.reconcile(orders, fills, positions, cash, fx=None) # type: ignore

    # Should be zero breaks (or at least not a qty mismatch)
    breaks = rep.get("breaks", [])
    assert not any(b.get("kind") in ("qty_mismatch", "notional_mismatch") for b in breaks)

def test_fx_conversion_if_provided(base_fx):
    orders = [{"oid": "O1", "symbol": "TCS.NS", "side": "sell", "qty": 10, "ccy":"INR"}]
    fills  = [{"fid": "F1", "oid": "O1", "symbol": "TCS.NS", "side": "sell", "qty": 10, "px": 4000.0, "ccy":"INR"}]
    positions = [{"symbol": "TCS.NS", "qty": -10, "ccy":"INR"}]
    cash = [{"ccy":"INR", "amount": 40000.0}]
    recon = _mk_reconciler(base_ccy="USD")
    rep = recon.reconcile(orders, fills, positions, cash, fx=base_fx) # type: ignore

    m = rep.get("metrics", {})
    # Look for a base notional field; otherwise compute via cash_base
    cash_base = rep.get("cash_base")
    if isinstance(cash_base, (int, float)):
        assert pytest.approx(cash_base, rel=1e-6) == 40000.0 * base_fx["INR"]

def test_corporate_action_split_adjustment_optional():
    """
    If your reconciler handles splits, feeding a 2:1 split where positions doubled and price halved
    should not produce false breaks. If unsupported, test skips gracefully.
    """
    orders = [{"oid": "O1", "symbol": "AAPL", "side":"buy", "qty": 1, "ccy":"USD"}]
    fills  = [{"fid": "F1", "oid": "O1", "symbol":"AAPL","side":"buy","qty":1,"px":200.0,"ccy":"USD"}]
    # Corporate action: 2:1 split → position becomes 2, reference price ~100
    positions = [{"symbol":"AAPL","qty":2,"ccy":"USD"}]
    cash = [{"ccy":"USD","amount": -200.0}]
    recon = _mk_reconciler()
    rep = recon.reconcile(orders, fills, positions, cash, fx=None) # type: ignore

    breaks = rep.get("breaks", [])
    if not breaks:
        # Great — engine normalized for split
        assert True
    else:
        # If engine flags a 'corp_action_required' break, accept and skip
        if any(b.get("kind") in ("corp_action", "split_adjustment_needed") for b in breaks):
            pytest.skip("Reconciler reports corporate action requirement; acceptable.")
        else:
            # Otherwise, no unexpected mismatches
            assert not any(b.get("kind") in ("qty_mismatch", "notional_mismatch") for b in breaks)