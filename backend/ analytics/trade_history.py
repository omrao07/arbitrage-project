# backend/analytics/trade_history.py
"""
Trade History Store + Query
---------------------------
- Listens to OMS streams and persists to SQLite (runtime/trades.db):
    * oms.parent  -> parents(order_id, symbol, side, qty, ts_ms, strategy, route_hint, mark_px, urgency, asset_class)
    * oms.child   -> children(child_id, parent_id, ts_ms, venue, typ, px, qty)
    * oms.fill    -> fills(fill_id, parent_id, child_id, symbol, side, price, qty, ts_ms, venue, fee)

- Public API:
    TradeHistory(db_path="runtime/trades.db").ingest_event(stream, msg)
    TradeHistory.query_fills(...), query_parents(...), timeline(...), stats(...),
    export_csv(...), export_json(...)

- CLI:
    python -m backend.analytics.trade_history --run          # listen to bus & persist
    python -m backend.analytics.trade_history --export fills.csv --kind fills --days 7
    python -m backend.analytics.trade_history --probe        # synthetic demo into DB

Notes:
- Safe to run alongside TCA++. You can join on parent_id for analytics.
- If bus isn’t available, you can still write to the DB via the API from your OMS.
"""

from __future__ import annotations

import os, json, time, sqlite3, contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional bus (graceful if missing)
try:
    from backend.bus.streams import consume_stream
except Exception:
    consume_stream = None  # type: ignore

# Optional pandas for nicer exports (graceful if missing)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# ------------------------------- utils ---------------------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _like(x: Optional[str]) -> str:
    return f"%{x}%" if x else "%"

# ------------------------------- schema --------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS parents (
  order_id     TEXT PRIMARY KEY,
  symbol       TEXT,
  side         TEXT,
  qty          REAL,
  ts_ms        INTEGER,
  strategy     TEXT,
  route_hint   TEXT,
  mark_px      REAL,
  urgency      REAL,
  asset_class  TEXT
);

CREATE TABLE IF NOT EXISTS children (
  child_id     TEXT PRIMARY KEY,
  parent_id    TEXT,
  ts_ms        INTEGER,
  venue        TEXT,
  typ          TEXT,
  px           REAL,
  qty          REAL,
  FOREIGN KEY(parent_id) REFERENCES parents(order_id)
);

CREATE TABLE IF NOT EXISTS fills (
  fill_id      TEXT PRIMARY KEY,
  parent_id    TEXT,
  child_id     TEXT,
  symbol       TEXT,
  side         TEXT,
  price        REAL,
  qty          REAL,
  ts_ms        INTEGER,
  venue        TEXT,
  fee          REAL,
  FOREIGN KEY(parent_id) REFERENCES parents(order_id),
  FOREIGN KEY(child_id)  REFERENCES children(child_id)
);

CREATE INDEX IF NOT EXISTS idx_fills_symbol_ts ON fills(symbol, ts_ms);
CREATE INDEX IF NOT EXISTS idx_fills_parent   ON fills(parent_id);
CREATE INDEX IF NOT EXISTS idx_children_parent ON children(parent_id);
CREATE INDEX IF NOT EXISTS idx_parents_ts ON parents(ts_ms);
"""

# ------------------------------- core ----------------------------------------

@dataclass
class TradeHistory:
    db_path: str = "runtime/trades.db"

    def __post_init__(self):
        _ensure_dir(self.db_path)
        with self._conn() as cx:
            cx.executescript(_SCHEMA)

    # ---- connection helper ----
    def _conn(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path, timeout=30.0)
        cx.row_factory = sqlite3.Row
        return cx

    # ---- ingestion (from OMS or your own calls) ----
    def ingest_event(self, stream: str, msg: Dict[str, Any]) -> None:
        """
        Accepts messages shaped like your OMS topics:
          - oms.parent
          - oms.child
          - oms.fill
        """
        if isinstance(msg, str):
            try: msg = json.loads(msg)
            except Exception: return

        with self._conn() as cx:
            if stream.endswith("oms.parent"):
                self._ingest_parent(cx, msg)
            elif stream.endswith("oms.child"):
                self._ingest_child(cx, msg)
            elif stream.endswith("oms.fill"):
                self._ingest_fill(cx, msg)

    def _ingest_parent(self, cx: sqlite3.Connection, m: Dict[str, Any]) -> None:
        cx.execute("""
            INSERT OR REPLACE INTO parents(order_id, symbol, side, qty, ts_ms, strategy, route_hint, mark_px, urgency, asset_class)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(m.get("order_id") or m.get("parent_id") or ""),
            (m.get("symbol") or "").upper(),
            (m.get("side") or "").lower(),
            float(m.get("qty") or 0.0),
            int(m.get("ts_ms") or m.get("ts") or _utc_ms()),
            m.get("strategy") or m.get("strategy_name") or None,
            m.get("route") or m.get("route_hint") or None,
            float(m.get("mark_px") or m.get("mark_price") or 0.0),
            float(m.get("urgency") or 0.0),
            m.get("asset_class") or "equity"
        ))

    def _ingest_child(self, cx: sqlite3.Connection, m: Dict[str, Any]) -> None:
        cx.execute("""
            INSERT OR REPLACE INTO children(child_id, parent_id, ts_ms, venue, typ, px, qty)
            VALUES(?, ?, ?, ?, ?, ?, ?)
        """, (
            str(m.get("order_id") or m.get("child_id") or ""),
            str(m.get("parent_id") or ""),
            int(m.get("ts_ms") or m.get("ts") or _utc_ms()),
            m.get("venue") or None,
            m.get("typ") or m.get("type") or None,
            float(m.get("px") or m.get("price") or 0.0),
            float(m.get("qty") or 0.0),
        ))

    def _ingest_fill(self, cx: sqlite3.Connection, m: Dict[str, Any]) -> None:
        cx.execute("""
            INSERT OR REPLACE INTO fills(fill_id, parent_id, child_id, symbol, side, price, qty, ts_ms, venue, fee)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(m.get("fill_id") or m.get("order_id") or m.get("child_id") or f"fill_{m.get('ts_ms')}_{m.get('qty')}"),
            str(m.get("parent_id") or ""),
            str(m.get("child_id") or m.get("order_id") or ""),
            (m.get("symbol") or "").upper(),
            (m.get("side") or "").lower(),
            float(m.get("price") or m.get("px") or 0.0),
            float(m.get("qty") or 0.0),
            int(m.get("ts_ms") or m.get("ts") or _utc_ms()),
            m.get("venue") or None,
            float(m.get("fee") or 0.0),
        ))

    # ---- queries ----
    def query_fills(
        self,
        *,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        symbol_like: Optional[str] = None,
        side: Optional[str] = None,
        parent_id: Optional[str] = None,
        venue_like: Optional[str] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        q = """
        SELECT f.*, p.strategy, p.route_hint
        FROM fills f LEFT JOIN parents p ON f.parent_id = p.order_id
        WHERE 1=1
        """
        args: List[Any] = []
        if since_ms is not None:
            q += " AND f.ts_ms >= ?"; args.append(int(since_ms))
        if until_ms is not None:
            q += " AND f.ts_ms <= ?"; args.append(int(until_ms))
        if symbol_like:
            q += " AND f.symbol LIKE ?"; args.append(_like(symbol_like.upper()))
        if side:
            q += " AND f.side = ?"; args.append(side.lower())
        if parent_id:
            q += " AND f.parent_id = ?"; args.append(parent_id)
        if venue_like:
            q += " AND f.venue LIKE ?"; args.append(_like(venue_like))
        q += " ORDER BY f.ts_ms DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as cx:
            rows = cx.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def query_parents(
        self,
        *,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        symbol_like: Optional[str] = None,
        strategy_like: Optional[str] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        q = "SELECT * FROM parents WHERE 1=1"; args: List[Any] = []
        if since_ms is not None: q += " AND ts_ms >= ?"; args.append(int(since_ms))
        if until_ms is not None: q += " AND ts_ms <= ?"; args.append(int(until_ms))
        if symbol_like: q += " AND symbol LIKE ?"; args.append(_like(symbol_like.upper()))
        if strategy_like: q += " AND strategy LIKE ?"; args.append(_like(strategy_like))
        q += " ORDER BY ts_ms DESC"
        if limit: q += f" LIMIT {int(limit)}"
        with self._conn() as cx:
            rows = cx.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def timeline(self, parent_id: str) -> Dict[str, Any]:
        """
        Return a parent order execution timeline with children + fills.
        """
        with self._conn() as cx:
            p = cx.execute("SELECT * FROM parents WHERE order_id=?", (parent_id,)).fetchone()
            if not p: return {}
            ch = cx.execute("SELECT * FROM children WHERE parent_id=? ORDER BY ts_ms", (parent_id,)).fetchall()
            fl = cx.execute("SELECT * FROM fills    WHERE parent_id=? ORDER BY ts_ms", (parent_id,)).fetchall()
        return {
            "parent": dict(p),
            "children": [dict(r) for r in ch],
            "fills": [dict(r) for r in fl],
        }

    def stats(
        self,
        *,
        days: int = 7,
        symbol_like: Optional[str] = None,
        strategy_like: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quick execution stats over the last N days: notional, avg price per symbol/side,
        fill count, venues used.
        """
        since = _utc_ms() - days*86_400_000
        q = """
        SELECT symbol, side,
               COUNT(*) as fills,
               SUM(price*qty) as notional,
               SUM(qty) as qty,
               GROUP_CONCAT(DISTINCT venue) as venues
        FROM fills
        WHERE ts_ms >= ?
        """
        args: List[Any] = [since]
        if symbol_like:
            q += " AND symbol LIKE ?"; args.append(_like(symbol_like.upper()))
        q += " GROUP BY symbol, side ORDER BY notional DESC"
        with self._conn() as cx:
            rows = [dict(r) for r in cx.execute(q, args).fetchall()]
        for r in rows:
            qsum = float(r.get("qty") or 0.0)
            r["avg_px"] = (float(r.get("notional") or 0.0) / qsum) if qsum > 0 else None
        return {
            "asof": _utc_ms(),
            "window_days": days,
            "rows": rows
        }

    # ---- exports ----
    def export_csv(self, out_path: str, kind: str = "fills", **query_kwargs) -> str:
        _ensure_dir(out_path)
        if pd is None:
            # pure sqlite dump
            rows = self._export_rows(kind, **query_kwargs)
            with open(out_path, "w", encoding="utf-8") as f:
                if rows:
                    headers = list(rows[0].keys())
                    f.write(",".join(headers) + "\n")
                    for r in rows:
                        f.write(",".join(str(r.get(h,"")) for h in headers) + "\n")
        else:
            df = self._export_df(kind, **query_kwargs)
            df.to_csv(out_path, index=False) # type: ignore
        return out_path

    def export_json(self, out_path: str, kind: str = "fills", **query_kwargs) -> str:
        _ensure_dir(out_path)
        rows = self._export_rows(kind, **query_kwargs)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        return out_path

    def _export_rows(self, kind: str, **query_kwargs) -> List[Dict[str, Any]]:
        if kind == "fills":
            return self.query_fills(**query_kwargs)
        if kind == "parents":
            return self.query_parents(**query_kwargs)
        raise ValueError("kind must be 'fills' or 'parents'")

    def _export_df(self, kind: str, **query_kwargs):
        rows = self._export_rows(kind, **query_kwargs)
        return pd.DataFrame(rows) if pd is not None else rows


# ------------------------------- bus runner ----------------------------------

def run_bus_listener(db_path: str = "runtime/trades.db"):
    assert consume_stream, "bus streams not available"
    th = TradeHistory(db_path=db_path)
    cursors = {"parent": "$", "child": "$", "fill": "$"}
    streams = {"parent": "oms.parent", "child": "oms.child", "fill": "oms.fill"}
    while True:
        for key, sname in streams.items():
            for _, msg in consume_stream(sname, start_id=cursors[key], block_ms=300, count=500):
                cursors[key] = "$"
                try:
                    if isinstance(msg, str):
                        msg = json.loads(msg)
                except Exception:
                    continue
                th.ingest_event(f"backend.{sname}", msg)
        time.sleep(0.05)


# ------------------------------- CLI -----------------------------------------

def main():
    import argparse, random
    ap = argparse.ArgumentParser(description="Trade History Store/Query")
    ap.add_argument("--run", action="store_true", help="Run bus listener (needs backend.bus.streams)")
    ap.add_argument("--db", type=str, default="runtime/trades.db")
    ap.add_argument("--export", type=str, help="Path to export CSV/JSON")
    ap.add_argument("--json", action="store_true", help="Export JSON instead of CSV")
    ap.add_argument("--kind", type=str, default="fills", choices=["fills","parents"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--symbol", type=str, help="Symbol LIKE filter (e.g., AAPL or .NS)")
    ap.add_argument("--probe", action="store_true", help="Write a few synthetic trades then exit")
    args = ap.parse_args()

    th = TradeHistory(db_path=args.db)

    if args.probe:
        # synthetic demo
        now = _utc_ms()
        pid = "P" + str(now)
        th.ingest_event("oms.parent", {"order_id": pid, "symbol":"AAPL", "side":"buy", "qty":10000, "ts_ms":now, "strategy":"alpha.momo", "mark_px":190.0})
        th.ingest_event("oms.child",  {"child_id":"C1","parent_id":pid,"ts_ms":now+500,"venue":"NASDAQ","typ":"limit","px":190.02,"qty":3000})
        th.ingest_event("oms.fill",   {"fill_id":"F1","parent_id":pid,"child_id":"C1","symbol":"AAPL","side":"buy","price":190.03,"qty":3000,"ts_ms":now+800,"venue":"NASDAQ","fee":3.1})
        th.ingest_event("oms.child",  {"child_id":"C2","parent_id":pid,"ts_ms":now+1500,"venue":"NASDAQ","typ":"market","px":190.05,"qty":7000})
        th.ingest_event("oms.fill",   {"fill_id":"F2","parent_id":pid,"child_id":"C2","symbol":"AAPL","side":"buy","price":190.06,"qty":7000,"ts_ms":now+1700,"venue":"NASDAQ","fee":7.2})
        print(json.dumps(th.timeline(pid), indent=2))
        print(json.dumps(th.stats(days=30), indent=2))
        return

    if args.run:
        try:
            run_bus_listener(db_path=args.db)
        except KeyboardInterrupt:
            pass
        return

    if args.export:
        since = _utc_ms() - args.days*86_400_000
        path = (th.export_json if args.json else th.export_csv)(
            args.export,
            kind=args.kind,
            since_ms=since,
            symbol_like=args.symbol,
        )
        print(f"Wrote {path}")
    else:
        # default: print a small stats summary
        print(json.dumps(th.stats(days=args.days, symbol_like=args.symbol), indent=2))

if __name__ == "__main__":
    main()