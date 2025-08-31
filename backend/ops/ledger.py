# backend/ledger/ledger.py
from __future__ import annotations
"""
SQLite Ledger (append-only events + trades + positions)
-------------------------------------------------------
- Zero non-stdlib dependencies (uses sqlite3 + json + hashlib).
- Hash-chained `events` table for tamper-evident audit.
- Normalized tables: orders, fills, positions, cash_ledger, instruments, accounts.
- Simple P&L math: position avg price & realized PnL on each fill.
- Cash movements on fills (buy consumes cash, sell adds cash) + optional fees.
- Convenience queries (positions, cash, open orders) and integrity verify.
- Small CLI for init/verify/export.

Recommended path: backend/ledger/ledger.py

Quick start (Python):
    ledg = Ledger("data/ledger.db")
    ledg.init_schema()
    ledg.ensure_account("ACC1", base_ccy="USD")
    ledg.record_order({"id":"O1","ts_ms":1724982000000,"account_id":"ACC1","symbol":"AAPL","side":"buy","qty":100,"order_type":"market"})
    ledg.record_fill({"id":"F1","ts_ms":1724982000300,"order_id":"O1","account_id":"ACC1","symbol":"AAPL","price":192.33,"qty":100,"venue":"NASDAQ"}, fee_bps=0.5)
    print(ledg.get_positions())
    print(ledg.get_cash_balance("ACC1"))

CLI:
    python -m backend.ledger.ledger init --db data/ledger.db
    python -m backend.ledger.ledger verify --db data/ledger.db
    python -m backend.ledger.ledger export --db data/ledger.db --out events.jsonl
"""

import os, json, sqlite3, hashlib, time, pathlib, argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------- helpers ----------

def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _now_ms() -> int:
    return int(time.time() * 1000)

# ---------- dataclasses (optional ergonomics) ----------

@dataclass
class Order:
    id: str
    ts_ms: int
    account_id: str
    symbol: str
    side: str          # 'buy' | 'sell'
    qty: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    status: str = "accepted"      # accepted|partially_filled|filled|canceled

@dataclass
class Fill:
    id: str
    ts_ms: int
    order_id: str
    account_id: str
    symbol: str
    price: float
    qty: float
    venue: Optional[str] = None
    liquidity: Optional[str] = None  # maker|taker|None

# ---------- main class ----------

class Ledger:
    def __init__(self, db_path: str = "data/ledger.db"):
        self.db_path = db_path
        pathlib.Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)  # autocommit
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

    # ----- schema -----
    def init_schema(self) -> None:
        c = self.conn.cursor()

        c.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            base_ccy TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS instruments (
            symbol TEXT PRIMARY KEY,
            asset_class TEXT DEFAULT 'Equity',
            tick_size REAL DEFAULT 0.01
        );

        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            order_type TEXT NOT NULL,
            limit_price REAL,
            status TEXT NOT NULL,
            FOREIGN KEY(account_id) REFERENCES accounts(id)
        );

        CREATE TABLE IF NOT EXISTS fills (
            id TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            order_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            qty REAL NOT NULL,
            venue TEXT,
            liquidity TEXT,
            FOREIGN KEY(order_id) REFERENCES orders(id),
            FOREIGN KEY(account_id) REFERENCES accounts(id)
        );

        CREATE TABLE IF NOT EXISTS positions (
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            avg_px REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY(account_id, symbol),
            FOREIGN KEY(account_id) REFERENCES accounts(id)
        );

        CREATE TABLE IF NOT EXISTS cash_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            account_id TEXT NOT NULL,
            ccy TEXT NOT NULL,
            delta REAL NOT NULL,
            reason TEXT NOT NULL,
            ref TEXT,
            FOREIGN KEY(account_id) REFERENCES accounts(id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_ms);
        CREATE INDEX IF NOT EXISTS idx_orders_acct ON orders(account_id);
        CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
        CREATE INDEX IF NOT EXISTS idx_cash_acct ON cash_ledger(account_id, ts_ms);
        """)
        c.close()

    # ----- accounts/instruments -----
    def ensure_account(self, account_id: str, *, base_ccy: str = "USD") -> None:
        self.conn.execute("INSERT OR IGNORE INTO accounts(id, base_ccy) VALUES (?,?)", (account_id, base_ccy))

    def upsert_instrument(self, symbol: str, *, asset_class: str = "Equity", tick_size: float = 0.01) -> None:
        self.conn.execute(
            "INSERT INTO instruments(symbol, asset_class, tick_size) VALUES (?,?,?) "
            "ON CONFLICT(symbol) DO UPDATE SET asset_class=excluded.asset_class, tick_size=excluded.tick_size",
            (symbol, asset_class, tick_size)
        )

    # ----- event append (hash chain) -----
    def append_event(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        ts = payload.get("ts_ms") or _now_ms()
        prev = self._last_event_hash()
        rec = {
            "ts_ms": int(ts),
            "kind": str(kind),
            "payload": payload,
            "prev_hash": prev
        }
        h = _sha256_hex(_canon(rec).encode("utf-8"))
        rec["hash"] = h
        self.conn.execute(
            "INSERT INTO events(ts_ms, kind, payload, prev_hash, hash) VALUES (?,?,?,?,?)",
            (rec["ts_ms"], rec["kind"], _canon(payload), rec["prev_hash"], rec["hash"])
        )
        return rec

    def _last_event_hash(self) -> str:
        row = self.conn.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else "0"*64

    # ----- orders & fills -----
    def record_order(self, order: Dict[str, Any] | Order) -> None:
        if isinstance(order, Order):
            order = asdict(order)
        # normalize
        o = {
            "id": str(order["id"]),
            "ts_ms": int(order.get("ts_ms") or _now_ms()),
            "account_id": str(order["account_id"]),
            "symbol": str(order["symbol"]).upper(),
            "side": str(order["side"]).lower(),
            "qty": float(order["qty"]),
            "order_type": str(order.get("order_type","market")).lower(),
            "limit_price": float(order["limit_price"]) if order.get("limit_price") is not None else None,
            "status": str(order.get("status","accepted"))
        }
        self.ensure_account(o["account_id"])
        self.upsert_instrument(o["symbol"])
        self.conn.execute(
            "INSERT OR REPLACE INTO orders(id, ts_ms, account_id, symbol, side, qty, order_type, limit_price, status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (o["id"], o["ts_ms"], o["account_id"], o["symbol"], o["side"], o["qty"], o["order_type"], o["limit_price"], o["status"])
        )
        self.append_event("order", o)

    def record_fill(self, fill: Dict[str, Any] | Fill, *, fee_bps: float = 0.0, fee_min: float = 0.0) -> None:
        if isinstance(fill, Fill):
            fill = asdict(fill)
        f = {
            "id": str(fill["id"]),
            "ts_ms": int(fill.get("ts_ms") or _now_ms()),
            "order_id": str(fill["order_id"]),
            "account_id": str(fill["account_id"]),
            "symbol": str(fill["symbol"]).upper(),
            "price": float(fill["price"]),
            "qty": float(fill["qty"]),
            "venue": fill.get("venue"),
            "liquidity": fill.get("liquidity")
        }
        self.ensure_account(f["account_id"])
        self.upsert_instrument(f["symbol"])

        # write fill row
        self.conn.execute(
            "INSERT OR REPLACE INTO fills(id, ts_ms, order_id, account_id, symbol, price, qty, venue, liquidity) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (f["id"], f["ts_ms"], f["order_id"], f["account_id"], f["symbol"], f["price"], f["qty"], f["venue"], f["liquidity"])
        )

        # update order status (best-effort)
        row = self.conn.execute("SELECT qty, side, status FROM orders WHERE id=?", (f["order_id"],)).fetchone()
        if row:
            ord_qty, side, status = float(row[0]), row[1], row[2]
            filled_qty = self._order_filled_qty(f["order_id"])
            new_status = "filled" if abs(filled_qty - ord_qty) <= 1e-9 else "partially_filled"
            self.conn.execute("UPDATE orders SET status=? WHERE id=?", (new_status, f["order_id"]))
        else:
            side = "buy"  # default if missing

        # fees (bps of notional, maker/taker neutral)
        notional = f["price"] * abs(f["qty"])
        fee = max(fee_min, (fee_bps / 1e4) * notional) if fee_bps > 0 else 0.0

        # positions math
        self._apply_fill_to_position(
            account_id=f["account_id"],
            symbol=f["symbol"],
            side=side,
            price=f["price"],
            qty=f["qty"],   # positive for buy fills; negative for sells handled via side
            fee=fee
        )

        # cash movement: buys consume cash, sells add cash (exclusive of fees; fees deducted)
        acct_ccy = self._account_ccy(f["account_id"])
        cash_delta = -notional if side == "buy" else notional
        cash_delta -= fee
        self.conn.execute(
            "INSERT INTO cash_ledger(ts_ms, account_id, ccy, delta, reason, ref) VALUES (?,?,?,?,?,?)",
            (f["ts_ms"], f["account_id"], acct_ccy, cash_delta, "fill", f["id"])
        )

        # event
        ev = dict(f)
        ev["fee"] = fee
        ev["cash_delta"] = cash_delta
        self.append_event("fill", ev)

    def _order_filled_qty(self, order_id: str) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(qty),0) FROM fills WHERE order_id=?", (order_id,)).fetchone()
        return float(row[0] or 0.0)

    def _account_ccy(self, account_id: str) -> str:
        row = self.conn.execute("SELECT base_ccy FROM accounts WHERE id=?", (account_id,)).fetchone()
        return row[0] if row else "USD"

    def _apply_fill_to_position(self, *, account_id: str, symbol: str, side: str, price: float, qty: float, fee: float) -> None:
        """
        Update position (qty, avg_px) and realized_pnl.
        Convention:
          - We compute signed fill_qty: +qty for buys, -qty for sells.
          - avg_px is cost basis per share (positive long, negative short allowed if you short).
        Realized PnL when closing (sign switch).
        """
        signed_qty = abs(qty) if side == "buy" else -abs(qty)
        row = self.conn.execute(
            "SELECT qty, avg_px, realized_pnl FROM positions WHERE account_id=? AND symbol=?",
            (account_id, symbol)
        ).fetchone()
        if not row:
            # opening new position
            new_qty = signed_qty
            new_avg = price if new_qty != 0 else 0.0
            realized = -fee
            self.conn.execute(
                "INSERT INTO positions(account_id, symbol, qty, avg_px, realized_pnl) VALUES (?,?,?,?,?)",
                (account_id, symbol, new_qty, new_avg, realized)
            )
            return

        cur_qty, cur_avg, realized = float(row[0]), float(row[1]), float(row[2])

        if cur_qty == 0.0 or (cur_qty > 0 and signed_qty > 0) or (cur_qty < 0 and signed_qty < 0):
            # increasing existing position (same direction)
            tot_shares = cur_qty + signed_qty
            if abs(tot_shares) < 1e-12:
                new_avg = 0.0
            else:
                # weighted avg cost
                new_avg = (cur_qty * cur_avg + signed_qty * price) / tot_shares
            new_qty = tot_shares
            realized += -fee
        else:
            # reducing or flipping position
            closing_qty = min(abs(cur_qty), abs(signed_qty)) * (1 if cur_qty > 0 else -1)  # sign of current
            # realized pnl = (close_px - avg) * closed_shares (respect sign)
            if cur_qty > 0:   # closing long
                realized_pnl_delta = (price - cur_avg) * abs(closing_qty)
            else:             # closing short
                realized_pnl_delta = (cur_avg - price) * abs(closing_qty)
            realized += realized_pnl_delta - fee
            new_qty = cur_qty + signed_qty
            new_avg = cur_avg if abs(new_qty) > 1e-12 else 0.0
            # if flipped (crossed through zero), reset avg to entry px of the residual leg
            if (cur_qty > 0 and new_qty < 0) or (cur_qty < 0 and new_qty > 0):
                # residual shares were opened at `price`
                new_avg = price

        self.conn.execute(
            "UPDATE positions SET qty=?, avg_px=?, realized_pnl=? WHERE account_id=? AND symbol=?",
            (new_qty, new_avg, realized, account_id, symbol)
        )

    # ----- queries -----
    def get_positions(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if account_id:
            rows = self.conn.execute(
                "SELECT account_id, symbol, qty, avg_px, realized_pnl FROM positions WHERE account_id=? ORDER BY symbol",
                (account_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT account_id, symbol, qty, avg_px, realized_pnl FROM positions ORDER BY account_id, symbol"
            ).fetchall()
        return [
            {"account_id": r[0], "symbol": r[1], "qty": float(r[2]), "avg_px": float(r[3]), "realized_pnl": float(r[4])}
            for r in rows
        ]

    def get_cash_balance(self, account_id: str, ccy: Optional[str] = None) -> Dict[str, float]:
        if ccy:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(delta),0) FROM cash_ledger WHERE account_id=? AND ccy=?",
                (account_id, ccy)
            ).fetchone()
            return {ccy: float(row[0] or 0.0)}
        # by ccy
        rows = self.conn.execute(
            "SELECT ccy, COALESCE(SUM(delta),0) FROM cash_ledger WHERE account_id=? GROUP BY ccy",
            (account_id,)
        ).fetchall()
        return {r[0]: float(r[1] or 0.0) for r in rows}

    def get_open_orders(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT id, ts_ms, account_id, symbol, side, qty, order_type, limit_price, status FROM orders WHERE status IN ('accepted','partially_filled')"
        args: Tuple[Any, ...] = ()
        if account_id:
            q += " AND account_id=?"
            args = (account_id,)
        q += " ORDER BY ts_ms"
        rows = self.conn.execute(q, args).fetchall()
        cols = ["id","ts_ms","account_id","symbol","side","qty","order_type","limit_price","status"]
        out = []
        for r in rows:
            d = {cols[i]: r[i] for i in range(len(cols))}
            d["qty"] = float(d["qty"]); d["limit_price"] = None if d["limit_price"] is None else float(d["limit_price"])
            out.append(d)
        return out

    # ----- integrity -----
    def verify_event_chain(self) -> Tuple[bool, str]:
        prev = "0"*64
        cur = self.conn.cursor()
        for (ts_ms, kind, payload, prev_hash, h) in cur.execute("SELECT ts_ms, kind, payload, prev_hash, hash FROM events ORDER BY id ASC"):
            rec = {"ts_ms": ts_ms, "kind": kind, "payload": json.loads(payload), "prev_hash": prev_hash}
            expect = _sha256_hex(_canon(rec).encode("utf-8"))
            if prev_hash != prev:
                return False, "prev_hash mismatch (chain break)"
            if expect != h:
                return False, "hash mismatch (tamper suspected)"
            prev = h
        return True, "ok"

    # ----- export -----
    def export_events_jsonl(self, out_path: str) -> int:
        rows = self.conn.execute("SELECT ts_ms, kind, payload, prev_hash, hash FROM events ORDER BY id ASC").fetchall()
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ts_ms, kind, payload, prev_hash, h in rows:
                obj = {"ts_ms": ts_ms, "kind": kind, "payload": json.loads(payload), "prev_hash": prev_hash, "hash": h}
                f.write(_canon(obj) + "\n")
                n += 1
        return n

# ---------- CLI ----------

def _cmd_init(db: str):
    L = Ledger(db)
    L.init_schema()
    print(f"initialized {db}")

def _cmd_verify(db: str):
    L = Ledger(db)
    ok, msg = L.verify_event_chain()
    print("OK" if ok else "FAIL", "-", msg)

def _cmd_export(db: str, out: str):
    L = Ledger(db)
    n = L.export_events_jsonl(out)
    print(f"wrote {n} events → {out}")

def main():
    ap = argparse.ArgumentParser(description="SQLite ledger for trades/events")
    ap.add_argument("--db", required=True, help="Path to sqlite db (e.g., data/ledger.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("verify")
    ex = sub.add_parser("export")
    ex.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "init":
        _cmd_init(args.db)
    elif args.cmd == "verify":
        _cmd_verify(args.db)
    elif args.cmd == "export":
        _cmd_export(args.db, args.out)
    else:
        ap.print_help()

if __name__ == "__main__":  # pragma: no cover
    main()