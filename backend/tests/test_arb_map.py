# tests/test_arb_map.py
import math
import importlib
import pytest # type: ignore
from typing import Any, Dict, Optional, Tuple, Callable

"""
What this tests
---------------
- Basic cross-venue arbitrage detection (buy low @ ask, sell high @ bid)
- Fee/latency-aware PnL
- Liquidity sizing (capped by available size)
- Staleness window exclusion
- Tie-breaking stability
- (Optional) triangular arb; skipped automatically if your module doesn't expose it

How it adapts to your code
--------------------------
We try a few common export shapes:

1) Class ArbMap with:
   - update(book_dict) or build(...)
   - best_cross(symbol, qty) -> dict with fields:
       { 'buy': {venue, price}, 'sell': {venue, price}, 'qty', 'gross_spread',
         'fees', 'net', 'latency_ms' }

2) Functions:
   - build_arb_graph(prices, fees=None, latencies=None, **kw) -> graph-like
   - find_best_cross(graph, symbol, qty) -> same dict as above

3) One-shot:
   - find_best_cross(prices, symbol, qty, fees=None, latencies=None, **kw)

Adjust IMPORT_PATH below if your file lives elsewhere.
"""

# ---- Adjust this to your module path if needed ----
IMPORT_PATH_CANDIDATES = [
    "backend.analytics.arbmap",
    "backend.research.arbmap",
    "backend.engine.arbmap",
    "analytics.arbmap",
    "arbmap",
]


# ---------- Helpers to adapt to your API ----------

def load_module():
    last_err = None
    for path in IMPORT_PATH_CANDIDATES:
        try:
            return importlib.import_module(path)
        except ModuleNotFoundError as e:
            last_err = e
    pytest.skip(f"Could not import arb map module from candidates {IMPORT_PATH_CANDIDATES}: {last_err}")

def get_api(mod):
    """
    Return a tuple describing how to call your module:
    - mode "class": (mode, cls, None, None)
    - mode "graph": (mode, build_graph_fn, best_cross_fn, None)
    - mode "oneshot": (mode, None, None, oneshot_fn)
    """
    if hasattr(mod, "ArbMap"):
        return ("class", getattr(mod, "ArbMap"), None, None)

    if hasattr(mod, "build_arb_graph") and hasattr(mod, "find_best_cross"):
        return ("graph", getattr(mod, "build_arb_graph"), getattr(mod, "find_best_cross"), None)

    if hasattr(mod, "find_best_cross"):
        # one-shot
        return ("oneshot", None, None, getattr(mod, "find_best_cross"))

    pytest.skip("Cannot find ArbMap class or find_best_cross/build_arb_graph functions in module.")


# ---------- Test Data ----------

@pytest.fixture()
def book_snapshot():
    """
    Minimal, consistent structure of per-symbol orderbooks across venues.
    Use whatever structure your code expects—you can map in the API glue if needed.
    """
    # Prices in USD; sizes in units of base asset
    return {
        "BTCUSD": {
            "BINANCE": {"bid": 40100.0, "bid_size": 0.8,  "ask": 40102.0, "ask_size": 0.9, "ts": 1_700_000_000_000, "fee_bps": 2.0, "latency_ms": 8},
            "COINBASE": {"bid": 40105.5, "bid_size": 0.5, "ask": 40106.0, "ask_size": 0.7, "ts": 1_700_000_000_010, "fee_bps": 2.5, "latency_ms": 10},
            "KRAKEN": {"bid": 40102.5, "bid_size": 0.4,  "ask": 40104.0, "ask_size": 0.6, "ts": 1_700_000_000_020, "fee_bps": 1.8, "latency_ms": 16},
        },
        "ETHUSD": {
            "BINANCE": {"bid": 2420.0, "bid_size": 8,  "ask": 2420.7, "ask_size": 9, "ts": 1_700_000_000_000, "fee_bps": 2.0, "latency_ms": 8},
            "COINBASE": {"bid": 2422.1,"bid_size": 6,  "ask": 2422.6,"ask_size": 7, "ts": 1_700_000_000_010, "fee_bps": 2.5, "latency_ms": 10},
            "KRAKEN": {"bid": 2421.4,"bid_size": 5,  "ask": 2421.8,"ask_size": 6, "ts": 1_700_000_000_020, "fee_bps": 1.8, "latency_ms": 16},
        },
    }


def expected_net_spread(buy_px, sell_px, qty, fee_bps_buy, fee_bps_sell):
    """
    Simple fee model: fees assessed on notional at each venue.
    net = (sell_px - buy_px) * qty - (buy_px * qty * bps/1e4 + sell_px * qty * bps/1e4)
    """
    gross = (sell_px - buy_px) * qty
    fees = (buy_px * qty * fee_bps_buy / 1e4) + (sell_px * qty * fee_bps_sell / 1e4)
    return gross - fees


# ---------- Tests ----------

def test_cross_venue_opportunity_detects_positive_spread(book_snapshot):
    mod = load_module()
    mode, A, B, C = get_api(mod) # type: ignore

    symbol = "BTCUSD"
    qty = 0.4

    if mode == "class":
        # build
        m = A() # type: ignore
        # Accept either m.update(...) or m.build(...)
        if hasattr(m, "update"):
            m.update(book_snapshot)
        elif hasattr(m, "build"):
            m.build(book_snapshot)
        else:
            pytest.skip("ArbMap has neither update() nor build().")
        res = m.best_cross(symbol=symbol, qty=qty)

    elif mode == "graph":
        g = A(book_snapshot) # type: ignore
        res = B(g, symbol, qty) # type: ignore

    else:  # oneshot
        res = C(book_snapshot, symbol, qty) # type: ignore

    assert isinstance(res, dict), "best_cross should return a dict"
    for k in ("buy", "sell", "qty", "net"):
        assert k in res

    # Expect it chose BINANCE ask (40102) and COINBASE bid (40105.5) OR KRAKEN bid (40102.5)
    # but net after fees should be > 0 for qty 0.4 with these defaults.
    buy_px = res["buy"]["price"]
    sell_px = res["sell"]["price"]
    fee_b = res["buy"].get("fee_bps", 0.0)
    fee_s = res["sell"].get("fee_bps", 0.0)
    net_manual = expected_net_spread(buy_px, sell_px, res["qty"], fee_b, fee_s)

    assert res["net"] == pytest.approx(net_manual, rel=1e-9, abs=1e-6)
    assert res["net"] > 0.0, f"Expected positive net, got {res['net']}"


def test_liquidity_caps_position_size(book_snapshot):
    mod = load_module()
    mode, A, B, C = get_api(mod) # type: ignore
    symbol = "BTCUSD"

    # Ask for very large size; should be capped by min(ask_size, bid_size) across chosen venues.
    wanted_qty = 10.0

    if mode == "class":
        m = A() # type: ignore
        (hasattr(m, "update") and m.update(book_snapshot)) or (hasattr(m, "build") and m.build(book_snapshot)) # type: ignore
        res = m.best_cross(symbol, wanted_qty)
    elif mode == "graph":
        g = A(book_snapshot) # type: ignore
        res = B(g, symbol, wanted_qty) # type: ignore
    else:
        res = C(book_snapshot, symbol, wanted_qty) # type: ignore

    assert res["qty"] <= wanted_qty
    # Compute max feasible qty given the best venues picked
    max_qty = min(
        book_snapshot[symbol][res["buy"]["venue"]]["ask_size"],
        book_snapshot[symbol][res["sell"]["venue"]]["bid_size"],
    )
    assert res["qty"] <= max_qty + 1e-9  # small tolerance


def test_fees_can_kill_marginal_arbs(book_snapshot, monkeypatch):
    """
    Inflate fees so spread becomes unprofitable; solver should either return None / net<=0
    or explicitly show negative net and a flag.
    """
    mod = load_module()
    mode, A, B, C = get_api(mod) # type: ignore
    symbol = "ETHUSD"
    qty = 1.0

    # Make fees huge on both legs
    books = {sym: {v: dict(d) for v, d in venues.items()} for sym, venues in book_snapshot.items()}
    for v in books[symbol].values():
        v["fee_bps"] = 30.0  # 0.30%

    if mode == "class":
        m = A() # type: ignore
        (hasattr(m, "update") and m.update(books)) or (hasattr(m, "build") and m.build(books)) # type: ignore
        res = m.best_cross(symbol, qty)
    elif mode == "graph":
        g = A(books) # type: ignore
        res = B(g, symbol, qty) # type: ignore
    else:
        res = C(books, symbol, qty) # type: ignore

    # Accept either "no-opportunity" None/{} OR explicit negative/zero net
    if not res:
        return
    assert res.get("net", -1e9) <= 0.0


def test_staleness_window_excludes_old_quotes(book_snapshot):
    """
    If your solver accepts staleness/window_ms option, verify it ignores outdated venues.
    We simulate one venue being very old, forcing a different pair selection or no opp.
    """
    mod = load_module()

    # If no staleness support, skip gracefully.
    if not any(hasattr(mod, n) for n in ("ArbMap", "find_best_cross")):
        pytest.skip("No arb interface found.")

    symbol = "BTCUSD"
    books = {sym: {v: dict(d) for v, d in venues.items()} for sym, venues in book_snapshot.items()}
    # Make COINBASE (the best bid) very old so it should be excluded
    books[symbol]["COINBASE"]["ts"] = 1_600_000_000_000  # stale

    mode, A, B, C = get_api(mod) # type: ignore

    kwargs = {"window_ms": 5_000}  # common param name; ignored if unsupported

    if mode == "class":
        m = A() # type: ignore
        # Allow set_window / configure / etc.
        if hasattr(m, "configure"):
            m.configure(**kwargs)
        (hasattr(m, "update") and m.update(books)) or (hasattr(m, "build") and m.build(books)) # type: ignore
        res = m.best_cross(symbol, 0.2)
    elif mode == "graph":
        g = A(books, **kwargs) # type: ignore
        res = B(g, symbol, 0.2) # type: ignore
    else:
        try:
            res = C(books, symbol, 0.2, **kwargs) # type: ignore
        except TypeError:
            # if oneshot doesn't accept kwargs, call plain
            res = C(books, symbol, 0.2) # type: ignore

    # If your implementation respects staleness, it should avoid COINBASE bid.
    if res:
        assert res["sell"]["venue"] != "COINBASE", "stale venue should not be used when window_ms is set"


def test_tie_breaking_is_stable(book_snapshot):
    """
    When two venue pairs produce identical net, the solver should pick deterministically,
    e.g., by lowest latency or lexicographic order. We enforce stability by calling twice.
    """
    mod = load_module()
    mode, A, B, C = get_api(mod) # type: ignore
    symbol = "ETHUSD"

    # Create a tie on bids across two venues
    books = {sym: {v: dict(d) for v, d in venues.items()} for sym, venues in book_snapshot.items()}
    books[symbol]["COINBASE"]["bid"] = books[symbol]["KRAKEN"]["bid"]

    def run():
        if mode == "class":
            m = A() # type: ignore
            (hasattr(m, "update") and m.update(books)) or (hasattr(m, "build") and m.build(books)) # type: ignore
            return m.best_cross(symbol, 2.0)
        elif mode == "graph":
            g = A(books) # type: ignore
            return B(g, symbol, 2.0) # type: ignore
        else:
            return C(books, symbol, 2.0) # type: ignore

    r1 = run()
    r2 = run()
    if not r1 or not r2:
        pytest.skip("Solver returned no opportunity in tie scenario.")
    assert (r1["buy"]["venue"], r1["sell"]["venue"]) == (r2["buy"]["venue"], r2["sell"]["venue"]), \
        "Tie-breaking should be deterministic across calls"


def test_triangular_arbitrage_optional(book_snapshot):
    """
    If your module supports triangular arbitrage, expose either:
      - find_best_triangle(prices, base='BTC', quote='USD') -> dict with 'legs', 'net'
      - or ArbMap.best_triangle(...)
    Otherwise this test will skip.
    """
    mod = load_module()
    symbol_sets_supported = hasattr(mod, "find_best_triangle") or hasattr(getattr(mod, "ArbMap", object), "best_triangle")
    if not symbol_sets_supported:
        pytest.skip("No triangular arb API; skipping.")

    # Synthetic triangle in USD/USDT/BTC on one exchange vs another
    # Intent: buy BTC with USD on Venue A, swap BTC->USDT on Venue B, USDT->USD on Venue A, yielding small profit.
    prices = {
        "BTCUSD": {
            "A": {"bid": 40100, "ask": 40102, "fee_bps": 2.0, "ts": 1},
            "B": {"bid": 40101, "ask": 40103, "fee_bps": 2.0, "ts": 1},
        },
        "BTCUSDT": {
            "A": {"bid": 40090, "ask": 40092, "fee_bps": 2.0, "ts": 1},
            "B": {"bid": 40105, "ask": 40106, "fee_bps": 2.0, "ts": 1},
        },
        "USDTUSD": {
            "A": {"bid": 1.0003, "ask": 1.0004, "fee_bps": 1.0, "ts": 1},
            "B": {"bid": 1.0001, "ask": 1.0002, "fee_bps": 1.0, "ts": 1},
        },
    }

    qty = 0.2  # BTC
    if hasattr(mod, "find_best_triangle"):
        res = mod.find_best_triangle(prices, base="BTC", bridge="USDT", quote="USD", qty=qty) # type: ignore
    else:
        arb = getattr(mod, "ArbMap")()
        if hasattr(arb, "update"):
            arb.update(prices)
        elif hasattr(arb, "build"):
            arb.build(prices)
        res = arb.best_triangle(base="BTC", bridge="USDT", quote="USD", qty=qty)

    assert isinstance(res, dict)
    assert "net" in res
    assert "legs" in res
    # We don’t demand positivity (depends on fee model), but at least it should compute a finite number.
    assert math.isfinite(res["net"])