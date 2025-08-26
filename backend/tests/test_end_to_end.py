# tests/test_end_to_end.py
"""
End-to-end integration test (in-memory bus)

Covers:
- publish_stream / consume_stream / hset are monkeypatched to an in-memory bus
- Strategy (ExampleBuyTheDip) consumes ticks and emits orders
- RiskGate validates orders → forwards to 'orders.risk_ok'
- OMS fills orders → writes 'fills' and updates a tiny PnL book
- Assertions: at least one order, one fill, non-empty PnL snapshot

Run: pytest -q tests/test_end_to_end.py
"""

import json
import math
import time
import threading
import queue
import types
import pytest # type: ignore

# --- Try to import your modules; skip cleanly if unavailable ------------------
sb = pytest.importorskip("backend.engine.strategy_base", reason="strategy_base not found")

# Optional pnl_xray; if not present, we use a tiny inline book
try:
    from backend.analytics.pnl_xray import PnLXray  # type: ignore
except Exception:
    class PnLXray:  # minimal stub
        def __init__(self, base_ccy="USD"):
            self.base = base_ccy
            self.pos = {}
            self.realized = 0.0
            self.unreal = 0.0
        def ingest_trade(self, t):
            sym = t["symbol"]
            qty = t["qty"] if t["side"] == "buy" else -t["qty"]
            px  = t["price"]
            p = self.pos.get(sym, {"qty":0.0, "avg":0.0})
            new_qty = p["qty"] + qty
            if new_qty == 0:
                # realize everything
                self.realized += p["qty"] * (px - p["avg"])  # sign handles buy/sell
                p = {"qty":0.0, "avg":0.0}
            elif p["qty"] == 0:
                p = {"qty":new_qty, "avg":px}
            elif math.copysign(1, new_qty) == math.copysign(1, p["qty"]):
                # same direction → update avg
                p["avg"] = (p["avg"] * abs(p["qty"]) + px * abs(qty)) / abs(new_qty)
                p["qty"] = new_qty
            else:
                # partial close
                closed = -qty if abs(qty) < abs(p["qty"]) and qty < 0 else qty if abs(qty) < abs(p["qty"]) else p["qty"]
                self.realized += closed * (px - p["avg"])
                p["qty"] = new_qty
                if p["qty"] == 0:
                    p["avg"] = 0.0
            self.pos[sym] = p
        def update_mark(self, symbol, price, ts=None):
            p = self.pos.get(symbol, {"qty":0.0, "avg":0.0})
            self.unreal = p["qty"] * (price - p["avg"])
        def snapshot(self, group_by=("symbol",)):
            return {"rows":[{"symbol":k, "qty":v["qty"], "avg":v["avg"]} for k,v in self.pos.items()],
                    "realized": self.realized, "unrealized": self.unreal, "ts": int(time.time()*1000)}

# --- In-memory message bus ----------------------------------------------------
class MemBus:
    def __init__(self):
        self.queues = {}         # stream -> Queue
        self.hashes = {}         # key -> dict
        self.stop = False
    def publish(self, stream, payload):
        q = self.queues.setdefault(stream, queue.Queue())
        q.put(payload)
    def consume(self, stream, block_ms=1000, count=200):
        q = self.queues.setdefault(stream, queue.Queue())
        # yield up to 'count' messages as (id, payload)
        items = []
        try:
            item = q.get(timeout=block_ms/1000.0)
            items.append(item)
            for _ in range(count-1):
                try:
                    items.append(q.get_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        for p in items:
            yield ("*", p)
    def hset(self, key, field, value):
        h = self.hashes.setdefault(key, {})
        h[field] = value

BUS = MemBus()

# --- Monkeypatch backend.bus.streams -----------------------------------------
@pytest.fixture(autouse=True)
def patch_streams(monkeypatch):
    # Build a tiny module-like object
    m = types.SimpleNamespace()

    def publish_stream(stream, payload):
        # keep JSON contract similar to prod
        BUS.publish(stream, json.dumps(payload) if not isinstance(payload, str) else payload)

    def consume_stream(stream, start_id="$", block_ms=1000, count=200):
        for msg_id, raw in BUS.consume(stream, block_ms=block_ms, count=count):
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                payload = raw
            yield (msg_id, payload)

    def hset(key, field, value=None):
        # Accept either key, field, value OR key, field, dict
        if isinstance(value, dict):
            BUS.hset(key, field, value)
        else:
            BUS.hset(key, field, value)

    m.publish_stream = publish_stream
    m.consume_stream = consume_stream
    m.hset = hset

    monkeypatch.setitem(sys.modules, "backend.bus.streams", m) # type: ignore
    # also patch inside already-imported strategy_base
    monkeypatch.setattr(sb, "publish_stream", publish_stream, raising=False)
    monkeypatch.setattr(sb, "consume_stream", consume_stream, raising=False)
    monkeypatch.setattr(sb, "hset", hset, raising=False)

    yield

# --- Simple Risk Gate & Paper OMS threads ------------------------------------
class RiskGate(threading.Thread):
    def __init__(self, incoming="orders.incoming", outgoing="orders.risk_ok"):
        super().__init__(daemon=True)
        self.incoming = incoming
        self.outgoing = outgoing
        self.running = True
    def run(self):
        while self.running and not BUS.stop:
            for _, order in BUS.consume(self.incoming, block_ms=200, count=50):
                # trivial checks: qty > 0 and side in {buy,sell}
                if order.get("qty", 0) > 0 and order.get("side") in ("buy","sell"):
                    BUS.publish(self.outgoing, order)

class PaperOMS(threading.Thread):
    def __init__(self, incoming="orders.risk_ok", fills="fills", marks="marks"):
        super().__init__(daemon=True)
        self.incoming = incoming
        self.fills = fills
        self.marks = marks
        self.xr = PnLXray()
        self.running = True
    def run(self):
        last_px = {}
        while self.running and not BUS.stop:
            progressed = False
            for _, mk in BUS.consume(self.marks, block_ms=10, count=100):
                last_px[mk["symbol"]] = mk["price"]
                self.xr.update_mark(mk["symbol"], mk["price"])
                progressed = True
            for _, o in BUS.consume(self.incoming, block_ms=10, count=100):
                # immediate fill at (mark or limit/market)
                px = o.get("limit_price") or last_px.get(o["symbol"], o.get("mark_price") or 100.0)
                fill = {
                    "ts": time.time(),
                    "symbol": o["symbol"], "side": o["side"], "qty": float(o["qty"]),
                    "price": float(px), "strategy": o.get("strategy","unknown")
                }
                self.xr.ingest_trade(fill)
                BUS.publish(self.fills, fill)
                progressed = True
            if not progressed:
                time.sleep(0.01)

# --- The E2E test -------------------------------------------------------------

def test_e2e_strategy_to_pnl(monkeypatch):
    """
    Wires:
      ticks.stream → Strategy.on_tick → orders.incoming → RiskGate → orders.risk_ok
                   → PaperOMS → fills + pnl_xray; also marks stream updates PnL
    Validates at least one order, one fill, and a non-empty PnL snapshot.
    """
    # Start gate & OMS
    gate = RiskGate(); gate.start()
    oms  = PaperOMS(); oms.start()

    # Instantiate example strategy
    strat = sb.ExampleBuyTheDip(name="e2e_buy_dip", default_qty=1.0, bps=5.0)

    # Feed ticks and marks into the bus in a small loop
    sym = "AAPL"
    pxs = [100.00, 99.40, 99.20, 99.80, 100.60, 100.20, 99.90]
    # Send as both ticks (for strategy) and marks (for OMS/PnL)
    orders_seen = 0
    fills_seen  = 0

    # Run strategy in a small thread consuming from our in-memory tick stream
    # but we can simply call on_tick directly since we control the loop.
    for i, px in enumerate(pxs):
        # emit mark for OMS
        BUS.publish("marks", {"symbol": sym, "price": px})
        # call strategy
        strat.on_tick({"symbol": sym, "price": px})
        # drain any orders to count
        for _, o in BUS.consume("orders.incoming", block_ms=10, count=100):
            orders_seen += 1
            BUS.publish("orders.incoming", o)  # put back for RiskGate consumption

        # give gate/oms time to process
        time.sleep(0.02)
        for _, f in BUS.consume("fills", block_ms=10, count=100):
            fills_seen += 1

    # Final small settle
    time.sleep(0.1)

    # Assertions
    assert orders_seen >= 1, "Strategy should emit at least one order"
    assert fills_seen  >= 1, "OMS should produce at least one fill"

    snap = oms.xr.snapshot(group_by=("symbol",))
    assert isinstance(snap, dict) and "rows" in snap
    assert any(r["symbol"] == sym for r in snap["rows"]), "PnL snapshot should include our symbol"

    # stop threads
    gate.running = False
    oms.running = False
    BUS.stop = True