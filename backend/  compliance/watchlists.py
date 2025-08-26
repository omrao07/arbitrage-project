# backend/analytics/watchlists.py
"""
Rule-based Watchlists (+ alerts)
--------------------------------
Features
- Multiple watchlists (e.g., "India-LargeCaps", "US-Tech", "Event-Arb")
- CRUD: create/list/rename/delete; add/remove symbols; tags/notes
- Fast persistence in SQLite (runtime/watchlists.db)
- Subscribe to normalized feeds (ticks.quotes / ticks.trades / news.events)
- Flexible rules per watchlist & per symbol:
    * price >= / <= X
    * % change over window >= / <= X
    * spread (ask-bid) bps >= X
    * volume spike vs rolling avg
    * news keyword hit / sentiment score >= X (if you send it in news payload)
- Alert throttling (don’t spam), cool-downs, one-shot vs sticky
- Emits alerts to `alerts.watchlist` and optional `ai.insight`
- CSV/JSON export for dashboards

CLI
  python -m backend.analytics.watchlists --new India-LargeCaps --tags nse,eq
  python -m backend.analytics.watchlists --add India-LargeCaps RELIANCE.NS TCS.NS HDFCBANK.NS
  python -m backend.analytics.watchlists --rule India-LargeCaps 'price>=3000' '%chg_5m<=-1.5'
  python -m backend.analytics.watchlists --run                       # attach to bus
  python -m backend.analytics.watchlists --export India-LargeCaps --csv runtime/ilc.csv
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional bus (graceful if missing)
try:
    from backend.bus.streams import consume_stream, publish_stream, hset
except Exception:
    consume_stream = publish_stream = hset = None  # type: ignore

# ----------- util helpers -----------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

def _like(x: Optional[str]) -> str:
    return f"%{x}%" if x else "%"

_price_rule_re = re.compile(r"^\s*(price|spread_bps|bid|ask|mid|news_sent)\s*(>=|<=|>|<|==)\s*([\-+]?\d+(\.\d+)?)\s*$", re.I)
_pct_rule_re   = re.compile(r"^\s*%chg_(\d+)(s|m|h)\s*(>=|<=|>|<|==)\s*([\-+]?\d+(\.\d+)?)\s*$", re.I)
_vol_rule_re   = re.compile(r"^\s*volx_(\d+)(s|m|h)\s*(>=|<=|>|<|==)\s*([\-+]?\d+(\.\d+)?)\s*$", re.I)
_news_kw_re    = re.compile(r"^\s*news_kw\s*:\s*(.+)$", re.I)

def _cmp(a: float, op: str, b: float) -> bool:
    return {
        ">":  a > b, ">=": a >= b, "<": a < b, "<=": a <= b, "==": a == b
    }[op]

def _to_ms_window(n: int, unit: str) -> int:
    mult = {"s":1, "m":60, "h":3600}[unit.lower()]
    return n * mult * 1000

# ----------- storage (sqlite) -----------

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS lists (
  name TEXT PRIMARY KEY,
  tags TEXT,
  note TEXT,
  created_ms INTEGER
);
CREATE TABLE IF NOT EXISTS entries (
  name TEXT,
  symbol TEXT,
  added_ms INTEGER,
  meta TEXT,
  PRIMARY KEY (name, symbol),
  FOREIGN KEY(name) REFERENCES lists(name) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS rules (
  name TEXT,
  rule TEXT,
  scope TEXT DEFAULT 'list',   -- 'list' or 'symbol:<SYM>'
  PRIMARY KEY (name, rule, scope),
  FOREIGN KEY(name) REFERENCES lists(name) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS alerts_log (
  ts_ms INTEGER,
  name TEXT,
  symbol TEXT,
  rule TEXT,
  value REAL,
  note TEXT
);
"""

@dataclass
class WatchlistStore:
    db_path: str = "runtime/watchlists.db"

    def __post_init__(self):
        _ensure_dir(self.db_path)
        with self._cx() as cx:
            cx.executescript(_SCHEMA)

    def _cx(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path, timeout=30.0)
        cx.row_factory = sqlite3.Row
        return cx

    # ---- CRUD lists ----
    def create(self, name: str, tags: Optional[str] = None, note: Optional[str] = None) -> None:
        with self._cx() as cx:
            cx.execute("INSERT OR IGNORE INTO lists(name,tags,note,created_ms) VALUES(?,?,?,?)",
                       (name, tags, note, _utc_ms()))

    def rename(self, old: str, new: str) -> None:
        with self._cx() as cx:
            cx.execute("UPDATE lists SET name=? WHERE name=?", (new, old))
            cx.execute("UPDATE entries SET name=? WHERE name=?", (new, old))
            cx.execute("UPDATE rules SET name=? WHERE name=?", (new, old))

    def delete(self, name: str) -> None:
        with self._cx() as cx:
            cx.execute("DELETE FROM lists WHERE name=?", (name,))

    def info(self, name: str) -> Dict[str, Any]:
        with self._cx() as cx:
            row = cx.execute("SELECT * FROM lists WHERE name=?", (name,)).fetchone()
            syms = [r["symbol"] for r in cx.execute("SELECT symbol FROM entries WHERE name=?", (name,)).fetchall()]
            rules = [dict(r) for r in cx.execute("SELECT rule,scope FROM rules WHERE name=?", (name,)).fetchall()]
        return {"list": dict(row) if row else None, "symbols": syms, "rules": rules}

    def list_all(self, like: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._cx() as cx:
            rows = cx.execute("SELECT * FROM lists WHERE name LIKE ?", (_like(like),)).fetchall()
        return [dict(r) for r in rows]

    # ---- entries ----
    def add_symbols(self, name: str, symbols: Iterable[str], meta: Optional[Dict[str, Any]] = None) -> None:
        meta_s = json.dumps(meta or {})
        now = _utc_ms()
        with self._cx() as cx:
            for s in symbols:
                cx.execute("INSERT OR IGNORE INTO entries(name,symbol,added_ms,meta) VALUES(?,?,?,?)",
                           (name, s.upper(), now, meta_s))

    def remove_symbols(self, name: str, symbols: Iterable[str]) -> None:
        with self._cx() as cx:
            for s in symbols:
                cx.execute("DELETE FROM entries WHERE name=? AND symbol=?", (name, s.upper()))

    def set_rule(self, name: str, rule: str, scope: str = "list") -> None:
        with self._cx() as cx:
            cx.execute("INSERT OR REPLACE INTO rules(name,rule,scope) VALUES(?,?,?)", (name, rule, scope))

    def delete_rule(self, name: str, rule: str, scope: str = "list") -> None:
        with self._cx() as cx:
            cx.execute("DELETE FROM rules WHERE name=? AND rule=? AND scope=?", (name, rule, scope))

    def rules_for(self, name: str, symbol: Optional[str] = None) -> List[str]:
        sym_scope = f"symbol:{symbol.upper()}" if symbol else None
        with self._cx() as cx:
            rows = cx.execute("SELECT rule FROM rules WHERE name=? AND (scope='list' OR scope=?)",
                              (name, sym_scope)).fetchall()
        return [r["rule"] for r in rows]

    def symbols(self, name: str) -> List[str]:
        with self._cx() as cx:
            rows = cx.execute("SELECT symbol FROM entries WHERE name=?", (name,)).fetchall()
        return [r["symbol"] for r in rows]

    def log_alert(self, name: str, symbol: str, rule: str, value: float, note: str = "") -> None:
        with self._cx() as cx:
            cx.execute("INSERT INTO alerts_log(ts_ms,name,symbol,rule,value,note) VALUES(?,?,?,?,?,?)",
                       (_utc_ms(), name, symbol.upper(), rule, float(value), note))

    def export(self, name: str) -> Dict[str, Any]:
        return self.info(name)

# ----------- live engine -----------

@dataclass
class LiveState:
    # last seen quotes/trades for quick calc
    last: Dict[str, Dict[str, Any]]  = None # type: ignore
    # rolling for returns & volume
    hist: Dict[str, List[Tuple[int, float, float]]] = None  # type: ignore # symbol -> [(ts_ms, price, qty)]
    last_news_sent: Dict[str, float] = None # type: ignore
    last_alert_ms: Dict[Tuple[str, str, str], int] = None   # type: ignore # (name,symbol,rule)->ts_ms

    def __post_init__(self):
        self.last = {}
        self.hist = {}
        self.last_news_sent = {}
        self.last_alert_ms = {}

class Watchlists:
    def __init__(self, store: Optional[WatchlistStore] = None, *, alert_cooldown_ms: int = 60_000):
        self.store = store or WatchlistStore()
        self.state = LiveState()
        self.alert_cooldown_ms = int(alert_cooldown_ms)

    # ---- ingest normalized feeds ----
    def on_quote(self, q: Dict[str, Any]) -> None:
        # expected: {ts_ms, symbol, bid, ask, venue?}
        sym = q.get("symbol")
        if not sym: return
        bid, ask = float(q.get("bid") or 0), float(q.get("ask") or 0)
        if bid <= 0 or ask <= 0: return
        mid = (bid + ask) / 2.0
        row = self.state.last.get(sym, {})
        row.update({"ts": int(q.get("ts_ms") or _utc_ms()), "bid": bid, "ask": ask, "mid": mid})
        self.state.last[sym] = row
        self._append_hist(sym, row["ts"], mid, 0.0)

    def on_trade(self, t: Dict[str, Any]) -> None:
        # expected: {ts_ms, symbol, price, qty}
        sym = t.get("symbol")
        if not sym: return
        px = float(t.get("price") or 0)
        qty = float(t.get("qty") or 0)
        if px <= 0 or qty <= 0: return
        row = self.state.last.get(sym, {})
        row.update({"ts": int(t.get("ts_ms") or _utc_ms()), "price": px})
        if "mid" not in row:
            row["mid"] = px
        self.state.last[sym] = row
        self._append_hist(sym, row["ts"], px, qty)

    def on_news(self, n: Dict[str, Any]) -> None:
        # expected: {ts_ms, symbol?, headline, sentiment? [-1..1]}
        sym = (n.get("symbol") or "").upper()
        if sym:
            s = float(n.get("sentiment") or 0.0)
            self.state.last_news_sent[sym] = s
            # optional quick keyword match alerts (handled in _eval_rules)

    def _append_hist(self, sym: str, ts: int, price: float, qty: float):
        buf = self.state.hist.setdefault(sym, [])
        buf.append((ts, price, qty))
        # trim 1 day
        cutoff = ts - 24*3600*1000
        while buf and buf[0][0] < cutoff:
            buf.pop(0)

    # ---- rule evaluation ----
    def evaluate(self, name: str) -> List[Dict[str, Any]]:
        syms = self.store.symbols(name)
        if not syms: return []
        out: List[Dict[str, Any]] = []
        for sym in syms:
            rules = self.store.rules_for(name, sym)
            if not rules: continue
            ctx = self.state.last.get(sym, {})
            if not ctx: continue
            for rule in rules:
                hit, val, note = self._eval_rule(rule, sym, ctx)
                if hit and self._should_alert(name, sym, rule):
                    alert = {
                        "ts_ms": _utc_ms(),
                        "name": name,
                        "symbol": sym,
                        "rule": rule,
                        "value": val,
                        "note": note
                    }
                    out.append(alert)
                    self.store.log_alert(name, sym, rule, val, note)
                    self._emit(alert)
        return out

    def _should_alert(self, name: str, sym: str, rule: str) -> bool:
        key = (name, sym, rule)
        now = _utc_ms()
        last = self.state.last_alert_ms.get(key, 0)
        if now - last >= self.alert_cooldown_ms:
            self.state.last_alert_ms[key] = now
            return True
        return False

    def _emit(self, alert: Dict[str, Any]) -> None:
        if publish_stream:
            publish_stream("alerts.watchlist", alert)
            publish_stream("ai.insight", {
                "ts_ms": alert["ts_ms"],
                "kind": "watchlist",
                "summary": f"{alert['name']} | {alert['symbol']} | {alert['rule']} hit ({alert['value']:.4g})",
                "details": [alert.get("note","")],
                "tags": ["watchlist","alert", alert["symbol"]]
            })

    def _eval_rule(self, rule: str, sym: str, ctx: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Returns (hit, value, note). Supports:
          - 'price >= 3000', 'bid < 100', 'spread_bps >= 5', 'mid >= X'
          - '%chg_5m <= -1.5'   (percent change vs last price in that window)
          - 'volx_10m >= 3'     (current traded volume vs avg in window)
          - 'news_sent >= 0.5'  (last sentiment seen for the symbol)
          - 'news_kw: earnings|merger'
        """
        r = rule.strip()

        # News keyword rule
        m = _news_kw_re.match(r)
        if m:
            kw_expr = m.group(1).strip()
            # We don’t store news text here; recommend handling keyword match in normalizer,
            # setting ctx['news_kw_hit']=True on last event. But we give a fallback:
            hit = bool(ctx.get("news_kw_hit"))
            return hit, float(1 if hit else 0), "news_kw_hit"

        # Price / spread / sentiment rules
        m = _price_rule_re.match(r)
        if m:
            fld, op, th = m.group(1).lower(), m.group(2), float(m.group(3))
            bid = ctx.get("bid"); ask = ctx.get("ask")
            mid = ctx.get("mid") or ctx.get("price")
            val_map = {
                "price": ctx.get("price", mid),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_bps": ((ask - bid) / mid * 1e4) if (ask and bid and mid and mid > 0) else None,
                "news_sent": float(self.state.last_news_sent.get(sym, 0.0))
            }
            val = val_map.get(fld)
            if val is None: return (False, float("nan"), f"{fld} unavailable")
            return _cmp(float(val), op, th), float(val), f"{fld} {op} {th}"

        # % change rule
        m = _pct_rule_re.match(r)
        if m:
            span_n, span_u, op, th = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
            dt = _to_ms_window(span_n, span_u)
            pct = self._pct_change(sym, dt)
            if pct is None: return (False, float("nan"), "insufficient history")
            return _cmp(pct, op, th), pct, f"%chg_{span_n}{span_u} {op} {th}"

        # volume spike rule
        m = _vol_rule_re.match(r)
        if m:
            span_n, span_u, op, th = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
            dt = _to_ms_window(span_n, span_u)
            volx = self._vol_multiple(sym, dt)
            if volx is None: return (False, float("nan"), "insufficient history")
            return _cmp(volx, op, th), volx, f"volx_{span_n}{span_u} {op} {th}"

        # Unknown rule type
        return (False, float("nan"), "unknown rule")

    def _pct_change(self, sym: str, window_ms: int) -> Optional[float]:
        buf = self.state.hist.get(sym, [])
        if not buf: return None
        now = buf[-1][0]
        cut = now - window_ms
        # find earliest >= cut
        base = None
        for ts, price, _ in buf:
            if ts >= cut:
                base = price
                break
        if base is None or base <= 0:
            return None
        last = buf[-1][1]
        return (last - base) / base * 100.0

    def _vol_multiple(self, sym: str, window_ms: int) -> Optional[float]:
        buf = self.state.hist.get(sym, [])
        if not buf: return None
        now = buf[-1][0]
        cut = now - window_ms
        # current volume in window
        cur = sum(q for ts, _, q in buf if ts >= cut)
        if cur <= 0: return 0.0
        # average volume per equal window over last N windows (N=10)
        N = 10
        buckets = []
        for i in range(1, N+1):
            hi = now - (i-1)*window_ms
            lo = now - i*window_ms
            buckets.append(sum(q for ts, _, q in buf if lo <= ts < hi))
        avg = (sum(buckets) / len(buckets)) if buckets else 0.0
        if avg <= 0: return None
        return cur / avg

# ----------- bus loop -----------

def run_loop(store: Optional[WatchlistStore] = None, lists: Optional[List[str]] = None, cooldown_ms: int = 60_000):
    assert consume_stream is not None, "bus streams not available"
    wl = Watchlists(store, alert_cooldown_ms=cooldown_ms)
    cursors = {"q": "$", "t": "$", "n": "$"}
    watchlists = set(lists or [r["name"] for r in (store or WatchlistStore()).list_all()])

    while True:
        # quotes
        try:
            for _, m in consume_stream("ticks.quotes", start_id=cursors["q"], block_ms=150, count=500):
                cursors["q"] = "$"
                try:
                    msg = json.loads(m) if isinstance(m, str) else m
                except Exception:
                    continue
                wl.on_quote(msg)
        except Exception:
            pass
        # trades
        try:
            for _, m in consume_stream("ticks.trades", start_id=cursors["t"], block_ms=150, count=500):
                cursors["t"] = "$"
                try:
                    msg = json.loads(m) if isinstance(m, str) else m
                except Exception:
                    continue
                wl.on_trade(msg)
        except Exception:
            pass
        # news
        try:
            for _, m in consume_stream("news.events", start_id=cursors["n"], block_ms=150, count=200):
                cursors["n"] = "$"
                try:
                    msg = json.loads(m) if isinstance(m, str) else m
                except Exception:
                    continue
                wl.on_news(msg)
        except Exception:
            pass

        # evaluate each list
        for name in list(watchlists):
            try:
                wl.evaluate(name)
            except Exception:
                pass

        time.sleep(0.05)

# ----------- CLI -----------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Rule-based Watchlists (+ alerts)")
    ap.add_argument("--db", type=str, default="runtime/watchlists.db")
    ap.add_argument("--new", type=str, help="Create a new watchlist")
    ap.add_argument("--tags", type=str, help="Comma tags when creating")
    ap.add_argument("--note", type=str, help="Note when creating")
    ap.add_argument("--add", nargs="+", help="Add symbols to a watchlist: <list> <SYM...>")
    ap.add_argument("--rm", nargs="+", help="Remove symbols: <list> <SYM...>")
    ap.add_argument("--rule", nargs="+", help="Set rules: <list> <rule...>")
    ap.add_argument("--rule-sym", nargs="+", help="Set symbol-scoped rule: <list> <SYM> <rule...>")
    ap.add_argument("--delrule", nargs="+", help="Delete rule: <list> <rule>")
    ap.add_argument("--info", type=str, help="Show info for a list")
    ap.add_argument("--ls", action="store_true", help="List all watchlists")
    ap.add_argument("--export", type=str, help="Export a watchlist to JSON")
    ap.add_argument("--csv", type=str, help="CSV path for exported symbols (name,symbol)")
    ap.add_argument("--run", action="store_true", help="Attach to bus and evaluate continuously")
    ap.add_argument("--cooldown", type=int, default=60, help="Alert cooldown seconds")
    args = ap.parse_args()

    store = WatchlistStore(db_path=args.db)

    if args.new:
        store.create(args.new, tags=(args.tags or ""), note=(args.note or ""))
        print(f"Created {args.new}")

    if args.add:
        name, *syms = args.add
        store.add_symbols(name, syms)
        print(f"Added {len(syms)} syms to {name}")

    if args.rm:
        name, *syms = args.rm
        store.remove_symbols(name, syms)
        print(f"Removed {len(syms)} syms from {name}")

    if args.rule:
        name, *rules = args.rule
        for r in rules:
            store.set_rule(name, r, scope="list")
        print(f"Set {len(rules)} list rules on {name}")

    if args.rule_sym:
        name, sym, *rules = args.rule_sym
        for r in rules:
            store.set_rule(name, r, scope=f"symbol:{sym.upper()}")
        print(f"Set {len(rules)} symbol rules on {name}:{sym}")

    if args.delrule:
        name, rule = args.delrule
        store.delete_rule(name, rule)
        print(f"Deleted rule on {name}: {rule}")

    if args.info:
        print(json.dumps(store.info(args.info), indent=2))

    if args.ls:
        print(json.dumps(store.list_all(), indent=2))

    if args.export:
        data = store.export(args.export)
        print(json.dumps(data, indent=2))
        if args.csv:
            _ensure_dir(args.csv)
            with open(args.csv, "w", encoding="utf-8") as f:
                f.write("name,symbol\n")
                for s in data.get("symbols", []):
                    f.write(f"{args.export},{s}\n")
            print(f"Wrote {args.csv}")

    if args.run:
        if not consume_stream:
            print("Bus not available. Exiting.")
            return
        try:
            run_loop(store, cooldown_ms=args.cooldown*1000)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()