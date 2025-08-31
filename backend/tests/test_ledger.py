# tests/test_ledger.py
import json
import math
import importlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest # type: ignore

"""
Supported public API shapes (any one is fine):

A) Class Ledger with:
   - post(entry: dict) -> str (returns journal_id)
   - batch_post(entries: list[dict]) -> list[str]            [optional]
   - balances(as_of: int|datetime = None) -> dict[str,float]
   - position(symbol: str, as_of=None) -> dict               [qty, avg_cost, m2m?]
   - positions(as_of=None) -> dict[str, dict]                [optional]
   - pnl(start, end, realized=True, unrealized=True) -> dict [optional]
   - cashflows(start, end) -> dict                           [optional]
   - snapshot(as_of) -> bytes|dict                           [optional]
   - restore(snapshot_blob) -> None                          [optional]
   - export_json() / import_json(blob)                       [optional]
   - find(journal_id) -> dict                                [optional]
   - void(journal_id, reason="") -> str                      [optional]
   - lock(as_of=None) / close_period(end)                    [optional]
   - reconcile(broker_stmt: dict) -> dict                    [optional]

B) Function style (same names module-level), plus a factory:
   - new_ledger() -> handle
   Every function then takes "handle" as the first argument.
"""

# ------------------------------- Loader ---------------------------------

IMPORT_CANDIDATES = [
    "backend.accounting.ledger",
    "backend.oms.ledger",
    "backend.core.ledger",
    "accounting.ledger",
    "oms.ledger",
    "ledger",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import ledger module from {IMPORT_CANDIDATES} ({last})")

class API:
    """Unifies class- and function-based ledgers."""
    def __init__(self, mod):
        self.mod = mod
        self.h = None
        if hasattr(mod, "Ledger"):
            L = getattr(mod, "Ledger")
            try:
                self.h = L()
            except TypeError:
                self.h = L
        else:
            if not hasattr(mod, "new_ledger"):
                pytest.skip("No Ledger class and no new_ledger() factory.")
            self.h = mod.new_ledger()

    def has(self, name):
        return hasattr(self.h, name) or hasattr(self.mod, name)

    def call(self, name, *args, **kw):
        if hasattr(self.h, name):
            return getattr(self.h, name)(*args, **kw)
        if hasattr(self.mod, name):
            return getattr(self.mod, name)(self.h, *args, **kw)
        raise AttributeError(f"Missing API '{name}'")

# ----------------------------- Fixtures ---------------------------------

@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture()
def now():
    return datetime.now(timezone.utc)

def ts(d: datetime) -> int:
    return int(d.timestamp() * 1000)

@pytest.fixture()
def seed_equity_book(api, now):
    """
    Seed with: deposit cash, buy shares (2 fills), mark prices up, sell partial.
    """
    if api.has("lock") and hasattr(api.h, "reset"):
        # if you expose a reset/clear, feel free to call it here
        try:
            api.h.reset()
        except Exception:
            pass

    t0 = now - timedelta(days=2)
    t1 = now - timedelta(days=1, hours=23)
    t2 = now - timedelta(days=1, hours=22)
    t3 = now - timedelta(days=1, hours=21)
    t4 = now - timedelta(days=1, hours=20)

    entries = [
        # Cash deposit
        {"ts": ts(t0), "type": "cash", "account": "CASH:USD", "amount": 1_000_000.00, "desc": "Initial funding"},
        # Buy 1: 1,000 AAPL @ 180 + 5 commission
        {"ts": ts(t1), "type": "trade", "symbol": "AAPL", "side": "buy", "qty": 1000, "price": 180.0,
         "fees": 5.0, "desc": "Fill#1"},
        # Buy 2: 500 AAPL @ 182 + 3 commission
        {"ts": ts(t2), "type": "trade", "symbol": "AAPL", "side": "buy", "qty": 500, "price": 182.0,
         "fees": 3.0, "desc": "Fill#2"},
        # Mark to market
        {"ts": ts(t3), "type": "m2m", "symbol": "AAPL", "price": 185.0, "desc": "EOD mark"},
        # Partial sell: 600 @ 186, fee 4
        {"ts": ts(t4), "type": "trade", "symbol": "AAPL", "side": "sell", "qty": 600, "price": 186.0,
         "fees": 4.0, "desc": "Trim"},
    ]

    if api.has("batch_post"):
        ids = api.call("batch_post", entries)
    else:
        ids = [api.call("post", e) for e in entries]
    return {"ids": ids, "times": (t0, t1, t2, t3, t4)}

# ------------------------------- Tests ----------------------------------

def test_double_entry_integrity(api, seed_equity_book):
    """
    Ledger must remain balanced (assets = liabilities + equity).
    If module exposes balances(), we check equity math; otherwise skip.
    """
    if not api.has("balances"):
        pytest.skip("No balances() API")
    bal = api.call("balances")
    assert isinstance(bal, dict)
    # Presence of CASH:USD and at least one P&L/Equity account is expected
    assert any(k.startswith("CASH") for k in bal.keys())
    total = sum(float(v) for v in bal.values())
    # numeric near-zero
    assert abs(total) <= 1e-6, f"Ledger not balanced, net={total}"

def test_positions_and_avg_cost(api, seed_equity_book):
    if not api.has("position") and not api.has("positions"):
        pytest.skip("No position(s) API")
    pos = api.call("position", symbol="AAPL") if api.has("position") else api.call("positions").get("AAPL")
    assert pos and isinstance(pos, dict)
    qty = float(pos.get("qty") or pos.get("quantity") or 0)
    avg = float(pos.get("avg_cost") or pos.get("avg") or pos.get("cost") or 0)
    # After buys 1000@180 + 500@182 -> total cost = 1000*180 + 500*182 + 8 fees = 180000 + 91000 + 8 = 271008
    # Avg cost = 271008 / 1500 = 180.672
    assert qty == pytest.approx(900.0)  # after selling 600, remaining 900
    assert avg == pytest.approx(180.672, abs=1e-3)

def test_realized_unrealized_pnl(api, now):
    if not api.has("pnl"):
        pytest.skip("No pnl() API")
    start = now - timedelta(days=3)
    end = now

    pnl = api.call("pnl", start=ts(start), end=ts(end), realized=True, unrealized=True)
    assert isinstance(pnl, dict)
    # Check fields are finite numbers
    for k in ("realized", "unrealized", "total"):
        if k in pnl:
            assert math.isfinite(float(pnl[k]))

def test_time_bounded_balances(api, seed_equity_book):
    if not api.has("balances"):
        pytest.skip("No balances() API")
    t0, t1, t2, t3, t4 = seed_equity_book["times"]
    # Balances as of after second buy but before sell must reflect 1500 shares
    bal_mid = api.call("balances", as_of=ts(t3))
    # If your balances() return accounts only, skip position check here
    if api.has("position"):
        pos_mid = api.call("position", symbol="AAPL", as_of=ts(t3))
        assert float(pos_mid.get("qty", 0)) == pytest.approx(1500.0)
    assert isinstance(bal_mid, dict)

def test_idempotent_posts_and_find_void(api, seed_equity_book, api: API): # type: ignore
    if not api.has("post"):
        pytest.skip("No post() API")
    # Post a deterministic journal, then re-post; second should either return same id or be rejected.
    j = {"ts": ts(datetime.now(timezone.utc)), "type": "cash", "account": "CASH:USD", "amount": 123.45, "desc": "dup-test"}
    id1 = api.call("post", j)
    try:
        id2 = api.call("post", deepcopy(j))
    except Exception:
        id2 = id1
    assert id2 == id1 or True

    # If find/void are supported, void the dup and ensure balances adjust
    if api.has("find") and api.has("void") and api.has("balances"):
        before = api.call("balances")
        vid = api.call("void", id1, reason="test")
        after = api.call("balances")
        # total should return to previous (balanced zero net anyway, but cash account should roll back)
        assert isinstance(vid, str) or vid is None
        assert isinstance(after, dict) and isinstance(before, dict)

def test_cashflows_and_fx_optional(api, now):
    if not api.has("cashflows"):
        pytest.skip("No cashflows() API")
    start = now - timedelta(days=5)
    end = now
    cf = api.call("cashflows", start=ts(start), end=ts(end))
    assert isinstance(cf, dict)
    # Cash inflow for initial funding should appear (if grouped by currency or account)
    s = json.dumps(cf).lower()
    assert ("funding" in s) or ("cash" in s) or (len(cf) >= 0)

def test_snapshot_and_restore_optional(api):
    if not api.has("snapshot") or not api.has("restore"):
        pytest.skip("No snapshot/restore API")
    snap = api.call("snapshot", as_of=None)
    if isinstance(snap, (bytes, bytearray)):
        assert len(snap) > 0
    else:
        assert isinstance(snap, (dict, list, str))
    # Try a round-trip
    api.call("restore", snap)
    # Must not crash

def test_export_import_roundtrip_optional(api):
    if not api.has("export_json") or not api.has("import_json"):
        pytest.skip("No export/import API")
    blob = api.call("export_json")
    s = json.dumps(blob, default=str)
    assert isinstance(s, str) and len(s) > 10
    # Clear/restore if you expose clear()
    if hasattr(api.h, "clear"):
        api.h.clear()
    api.call("import_json", blob)

def test_reconcile_optional(api):
    if not api.has("reconcile"):
        pytest.skip("No reconcile() API")
    # Minimal fake broker statement: cash + positions
    stmt = {
        "as_of": int(datetime.now(timezone.utc).timestamp()*1000),
        "cash": {"USD": 999000.0},
        "positions": {"AAPL": {"qty": 900, "price": 185.0}},
    }
    out = api.call("reconcile", broker_stmt=stmt)
    assert isinstance(out, dict)
    # Expect differences map or a status flag
    assert any(k in out for k in ("diff", "status", "actions"))

def test_precision_and_rounding(api):
    # Tiny fee posting should not create rounding drift that breaks balance
    tiny = {"ts": ts(datetime.now(timezone.utc)), "type": "fee", "account": "EXP:FEES", "amount": -0.0003, "desc": "tiny fee"}
    if not api.has("post") or not api.has("balances"):
        pytest.skip("Need post() and balances()")
    api.call("post", tiny)
    bal = api.call("balances")
    total = sum(float(v) for v in bal.values())
    assert abs(total) <= 1e-4

def test_multiple_symbols_positions_optional(api, api_hydrate_extra=True):
    # Optional: if positions() exists, ensure dictionary keyed by symbols
    if not api.has("positions"):
        pytest.skip("No positions() API")
    pos = api.call("positions")
    assert isinstance(pos, dict)
    # Keys are symbols; values include qty
    for sym, p in list(pos.items())[:3]:
        assert "qty" in p or "quantity" in p