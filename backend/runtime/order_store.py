# backend/execution/order_store.py
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------- helpers -----------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def _j(x: Any) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)

# ----------------------------- schema ------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS orders (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms             INTEGER NOT NULL,
  client_order_id   TEXT,
  broker_order_id   TEXT,
  symbol            TEXT NOT NULL,
  side              TEXT NOT NULL,          -- 'buy' | 'sell'
  qty               REAL NOT NULL,
  order_type        TEXT NOT NULL,          -- 'market' | 'limit'
  limit_price       REAL,
  tif               TEXT,                   -- 'day' | 'ioc' | 'gtc'
  strategy          TEXT,
  status            TEXT NOT NULL,          -- 'new' | 'acked' | 'rejected' | 'filled' | 'partial'
  reason            TEXT,                   -- reject/ack reason
  meta              TEXT                    -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_orders_ts         ON orders(ts_ms);
CREATE INDEX IF NOT EXISTS idx_orders_coid       ON orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol     ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_strategy   ON orders(strategy);

CREATE TABLE IF NOT EXISTS fills (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id    INTEGER,                     -- foreign key to orders.id (nullable if unknown)
  order_coid  TEXT,                        -- for linking when id unknown
  order_oid   TEXT,                        -- broker order id
  ts_ms       INTEGER NOT NULL,
  symbol      TEXT NOT NULL,
  side        TEXT NOT NULL,
  qty         REAL NOT NULL,
  price       REAL NOT NULL,
  notional    REAL NOT NULL,
  strategy    TEXT,
  raw         TEXT                         -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_fills_ts         ON fills(ts_ms);
CREATE INDEX IF NOT EXISTS idx_fills_symbol     ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_strategy   ON fills(strategy);

-- Running positions (avg cost) maintained on each fill (for speed).
CREATE TABLE IF NOT EXISTS positions (
  symbol    TEXT PRIMARY KEY,
  qty       REAL NOT NULL,
  avg_price REAL NOT NULL
);

-- Day PnL snapshot (very lightweight; you can compute more in your analytics).
CREATE TABLE IF NOT EXISTS pnl_day (
  dte            TEXT PRIMARY KEY,  -- YYYY-MM-DD
  realized       REAL NOT NULL DEFAULT 0.0,
  fees           REAL NOT NULL DEFAULT 0.0,
  last_update_ms INTEGER NOT NULL
);
"""

# ----------------------------- API ------------------------------

@dataclass
class OrderIn:
    symbol: str
    side: str                 # 'buy' | 'sell'
    qty: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    tif: Optional[str] = None
    strategy: Optional[str] = None
    client_order_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    ts_ms: int = now_ms()

@dataclass
class AckIn:
    client_order_id: Optional[str]
    broker_order_id: Optional[str]
    ok: bool
    reason: Optional[str] = None
    ts_ms: int = now_ms()

@dataclass
class FillIn:
    order_id: Optional[int]
    order_coid: Optional[str]
    order_oid: Optional[str]
    symbol: str
    side: str
    qty: float
    price: float
    strategy: Optional[str] = None
    ts_ms: int = now_ms()
    raw: Optional[Dict[str, Any]] = None

# ----------------------------- Store ------------------------------

class OrderStore:
    """
    Thread-safe SQLite store for orders, acks, fills, and positions.
    - Append-only tables for orders & fills
    - Maintains running average-cost positions
    - Simple realized PnL (sell vs average cost) and daily turnover
    """

    def __init__(self, db_path: str = "runtime/order_store.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # --------------- orders ----------------

    def record_order(self, o: OrderIn) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO orders (ts_ms, client_order_id, symbol, side, qty, order_type,
                                    limit_price, tif, strategy, status, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)
                """,
                (
                    o.ts_ms, o.client_order_id, o.symbol.upper(), o.side.lower(), float(o.qty),
                    o.order_type.lower(), o.limit_price, (o.tif or None), (o.strategy or None),
                    _j(o.meta or {}),
                ),
            )
            return int(cur.lastrowid) # type: ignore

    def record_ack(self, a: AckIn) -> None:
        with self._lock:
            # find order by client_order_id (preferred)
            if a.client_order_id:
                row = self._conn.execute(
                    "SELECT id FROM orders WHERE client_order_id = ? ORDER BY id DESC LIMIT 1",
                    (a.client_order_id,),
                ).fetchone()
                if row:
                    oid = int(row["id"])
                    self._conn.execute(
                        "UPDATE orders SET broker_order_id=?, status=?, reason=? WHERE id=?",
                        (a.broker_order_id, ("acked" if a.ok else "rejected"), a.reason, oid),
                    )
                    return
            # fallback: nothing to update (still keep a separate audit if you want)

    # --------------- fills & positions ---------------

    def record_fill(self, f: FillIn) -> int:
        notional = float(f.qty) * float(f.price)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO fills (order_id, order_coid, order_oid, ts_ms, symbol, side, qty, price, notional, strategy, raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f.order_id, f.order_coid, f.order_oid, f.ts_ms, f.symbol.upper(), f.side.lower(),
                    float(f.qty), float(f.price), notional, (f.strategy or None), _j(f.raw or {}),
                ),
            )
            fill_id = int(cur.lastrowid) # type: ignore

            # mark order as (partial) filled if we can link it
            if f.order_id:
                # naive: mark as 'filled' (you can track cumulative fills to detect 'partial')
                self._conn.execute("UPDATE orders SET status='filled' WHERE id=?", (f.order_id,))

            # update running positions and realized PnL
            self._apply_fill_to_positions(f.symbol.upper(), f.side.lower(), float(f.qty), float(f.price))

            return fill_id

    def _apply_fill_to_positions(self, symbol: str, side: str, qty: float, price: float) -> None:
        # load current pos
        row = self._conn.execute("SELECT qty, avg_price FROM positions WHERE symbol=?", (symbol,)).fetchone()
        cur_qty = float(row["qty"]) if row else 0.0
        cur_avg = float(row["avg_price"]) if row else 0.0

        if side == "buy":
            new_qty = cur_qty + qty
            new_avg = ((cur_avg * cur_qty) + (qty * price)) / max(new_qty, 1e-9)
            if row:
                self._conn.execute("UPDATE positions SET qty=?, avg_price=? WHERE symbol=?", (new_qty, new_avg, symbol))
            else:
                self._conn.execute("INSERT INTO positions(symbol, qty, avg_price) VALUES (?,?,?)", (symbol, new_qty, new_avg))
        else:  # sell
            new_qty = cur_qty - qty
            # realized PnL = (sell price - avg cost) * qty
            realized = (price - cur_avg) * qty
            self._bump_realized_pnl(realized)

            # avg price stays the same for remaining inventory; reset if flat
            if abs(new_qty) <= 1e-12:
                new_qty = 0.0
                cur_avg = price  # optional: reset avg to last
            if row:
                self._conn.execute("UPDATE positions SET qty=?, avg_price=? WHERE symbol=?", (new_qty, cur_avg, symbol))
            else:
                self._conn.execute("INSERT INTO positions(symbol, qty, avg_price) VALUES (?,?,?)", (new_qty, cur_avg, symbol))

    # --------------- pnl & turnover ---------------

    def _today_key(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _bump_realized_pnl(self, delta: float) -> None:
        d = self._today_key()
        self._conn.execute(
            """
            INSERT INTO pnl_day(dte, realized, last_update_ms)
            VALUES (?, ?, ?)
            ON CONFLICT(dte) DO UPDATE SET
              realized = COALESCE(pnl_day.realized,0) + excluded.realized,
              last_update_ms = excluded.last_update_ms
            """,
            (d, float(delta), now_ms()),
        )

    def bump_fees(self, delta: float) -> None:
        d = self._today_key()
        self._conn.execute(
            """
            INSERT INTO pnl_day(dte, fees, last_update_ms)
            VALUES (?, ?, ?)
            ON CONFLICT(dte) DO UPDATE SET
              fees = COALESCE(pnl_day.fees,0) + excluded.fees,
              last_update_ms = excluded.last_update_ms
            """,
            (d, float(delta), now_ms()),
        )

    # --------------- queries ----------------

    def get_positions(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT symbol, qty, avg_price FROM positions ORDER BY symbol").fetchall()
        return [dict(symbol=r["symbol"], qty=float(r["qty"]), avg_price=float(r["avg_price"])) for r in rows]

    def get_orders(self, limit: int = 200, status: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM orders"
        args: Tuple[Any, ...] = tuple()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY id DESC LIMIT ?"
        args = args + (int(limit),)
        rows = self._conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def get_fills(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM fills ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def get_pnl_day(self) -> Dict[str, float]:
        d = self._today_key()
        row = self._conn.execute("SELECT realized, fees FROM pnl_day WHERE dte=?", (d,)).fetchone()
        realized = float(row["realized"]) if row else 0.0
        fees = float(row["fees"]) if row else 0.0
        return {"realized": realized, "fees": fees, "pnl": realized - fees}

    # --------------- maintenance ----------------

    def rebuild_positions_from_fills(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM positions")
            rows = self._conn.execute("SELECT symbol, side, qty, price FROM fills ORDER BY ts_ms ASC").fetchall()
            for r in rows:
                self._apply_fill_to_positions(
                    symbol=str(r["symbol"]),
                    side=str(r["side"]),
                    qty=float(r["qty"]),
                    price=float(r["price"]),
                )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

# ----------------------------- Optional stream consumers ----------------
# If you want this store to listen directly to your Redis buses, you can use
# these helpers (requires your `backend.bus.streams`).

def run_bus_consumer(db_path: str = "runtime/order_store.db") -> None:
    """
    Consume orders.acks (echo), fills, and (optionally) cost/fees events to update the store.
    """
    from backend.bus.streams import consume_stream  # lazy import

    store = OrderStore(db_path=db_path)
    try:
        while True:
            # You can multiplex; here we poll two streams sequentially for simplicity.
            for _id, ack in consume_stream("orders.acks", start_id="$", block_ms=250, count=200):
                # expected shape: {"ts_ms":..., "symbol":..., "ok": bool, "order_id": "...", "reason": "...", "strategy": "...", "client_order_id": "..."}
                a = AckIn(
                    client_order_id=ack.get("client_order_id"),
                    broker_order_id=ack.get("order_id"),
                    ok=bool(ack.get("ok", False)),
                    reason=ack.get("reason"),
                    ts_ms=int(ack.get("ts_ms") or now_ms()),
                )
                store.record_ack(a)

            for _id, fill in consume_stream("fills", start_id="$", block_ms=250, count=200):
                f = FillIn(
                    order_id=fill.get("order_id_int"),  # populate if you echo it
                    order_coid=fill.get("client_order_id"),
                    order_oid=fill.get("order_id"),
                    symbol=str(fill.get("symbol")),
                    side=str(fill.get("side")).lower(),
                    qty=float(fill.get("qty", 0.0)),
                    price=float(fill.get("price", 0.0)),
                    strategy=fill.get("strategy"),
                    ts_ms=int(fill.get("ts_ms") or now_ms()),
                    raw=fill,
                )
                store.record_fill(f)
    finally:
        store.close()

# ----------------------------- CLI -------------------------------------

if __name__ == "__main__":
    # tiny smoke test
    st = OrderStore("runtime/order_store.db")
    oid = st.record_order(OrderIn(symbol="RELIANCE.NS", side="buy", qty=10, order_type="limit", limit_price=2950.0, client_order_id="coid-1", strategy="mm_core"))
    st.record_ack(AckIn(client_order_id="coid-1", broker_order_id="PB-1", ok=True, reason="FILLED"))
    st.record_fill(FillIn(order_id=oid, order_coid="coid-1", order_oid="PB-1", symbol="RELIANCE.NS", side="buy", qty=10, price=2945.0, strategy="mm_core"))
    st.record_fill(FillIn(order_id=oid, order_coid="coid-1", order_oid="PB-1", symbol="RELIANCE.NS", side="sell", qty=4, price=2960.0, strategy="mm_core"))
    print("positions:", st.get_positions())
    print("pnl_day:", st.get_pnl_day())
    st.close()