# backend/altdata/card_spend.py
"""
Card Spend Analytics (Alt-Data)
-------------------------------
Purpose
- Ingest raw card transactions (timestamp, amount, merchant, category, location, user_id*)
- Build daily & weekly merchant/category spend indices
- Compute growth metrics (DoD, WoW, MoM, YoY) and rolling trends
- Map merchants/brands -> tickers to produce equity-aligned signals
- Detect anomalies (spikes, drops) with robust z-scores & IQR
- Persist to SQLite for backtests (runtime/altdata.db)
- Optionally publish insights to bus topics: altdata.card_spend, ai.insight

Expected raw columns (best-effort; case-insensitive, aliases supported):
  ts|timestamp|date, amount|amt|value, merchant|brand|m_name, category|cat, country|region|state,
  user_id|uid (optional, for cohort/unique-user calc)

CLI
  python -m backend.altdata.card_spend --load tx.csv --brand-map config/brand_map.yaml --save --publish
  python -m backend.altdata.card_spend --reindex --export runtime/spend_index.csv
  python -m backend.altdata.card_spend --probe
"""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import statistics as stats
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional deps
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore


# -------------------------- utils --------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}

def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def _parse_ts(x) -> int:
    """Accept epoch (s/ms/us/ns) or ISO date/datetime; return epoch ms."""
    if x is None:
        return _utc_ms()
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            if v > 1e16:  # ns
                return int(v / 1e6)
            if v > 1e14:  # us
                return int(v / 1e3)
            if v > 1e12:  # ms
                return int(v)
            return int(v * 1000)  # s
        s = str(x).strip().replace("Z", "+00:00")
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(s)
            return int(dt.timestamp() * 1000)
        except Exception:
            # try date only
            try:
                dt = datetime.strptime(s[:10], "%Y-%m-%d")
                return int(time.mktime(dt.timetuple()) * 1000)
            except Exception:
                return _utc_ms()
    except Exception:
        return _utc_ms()

def _day(ms: int) -> str:
    t = time.gmtime(ms/1000)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"

def _week_tag(ms: int) -> str:
    t = time.gmtime(ms/1000)
    # ISO year-week
    import datetime as _dt
    dt = _dt.datetime.utcfromtimestamp(ms/1000)
    y, w, _ = dt.isocalendar()
    return f"{y:04d}-W{int(w):02d}"


# -------------------------- storage --------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS card_tx (
  ts_ms INTEGER,
  day TEXT,
  week TEXT,
  amount REAL,
  merchant TEXT,
  category TEXT,
  country TEXT,
  user_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_tx_day ON card_tx(day);
CREATE INDEX IF NOT EXISTS ix_tx_merchant ON card_tx(merchant);

CREATE TABLE IF NOT EXISTS spend_index (
  level TEXT,              -- 'merchant' | 'category' | 'country' | 'ticker'
  key TEXT,                -- e.g., 'STARBUCKS' or 'AAPL'
  day TEXT,                -- YYYY-MM-DD
  gross REAL,              -- sum amount
  users INTEGER,           -- unique users if available
  avg_ticket REAL,         -- gross/users (approx when users>0)
  wow REAL, mom REAL, yoy REAL, dod REAL,  -- growth %
  trend REAL,              -- seasonally-adjusted (rough)
  anomaly REAL,            -- z-score like indicator
  meta TEXT,               -- JSON (notes)
  PRIMARY KEY(level, key, day)
);
CREATE INDEX IF NOT EXISTS ix_idx_day ON spend_index(day);

CREATE TABLE IF NOT EXISTS brand_map (
  merchant TEXT PRIMARY KEY,
  ticker TEXT,
  weight REAL DEFAULT 1.0
);
"""

class CardStore:
    def __init__(self, db_path: str = "runtime/altdata.db"):
        self.db_path = db_path
        _ensure_dir(db_path)
        with self._cx() as cx:
            cx.executescript(_SCHEMA)

    def _cx(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path, timeout=60.0)
        cx.row_factory = sqlite3.Row
        return cx

    # ---- ingest raw ----
    def ingest_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        n = 0
        with self._cx() as cx:
            for r in rows:
                rr = _lower_keys(r)
                ts = _parse_ts(_get(rr, "ts","timestamp","time","date"))
                amt = _safe_float(_get(rr, "amount","amt","value"))
                if amt == 0.0:
                    continue
                merchant = str(_get(rr, "merchant","brand","m_name","m"))[:128].upper()
                category = str(_get(rr, "category","cat","segment","vertical") or "")[:64].upper()
                country  = str(_get(rr, "country","region","state") or "")[:32].upper()
                user_id  = str(_get(rr, "user_id","uid","user") or "")
                cx.execute(
                    "INSERT INTO card_tx(ts_ms,day,week,amount,merchant,category,country,user_id) VALUES(?,?,?,?,?,?,?,?)",
                    (ts, _day(ts), _week_tag(ts), amt, merchant, category, country, user_id)
                )
                n += 1
            cx.commit()
        return n

    def load_csv(self, path: str, *, delimiter: str = ",") -> int:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return self.ingest_rows(reader)

    def load_json(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return self.ingest_rows(data)
        elif isinstance(data, dict) and "rows" in data:
            return self.ingest_rows(data["rows"])
        else:
            return 0

    def load_parquet(self, path: str) -> int:
        assert pq is not None, "pyarrow not installed"
        table = pq.read_table(path)
        df = table.to_pandas()
        return self.ingest_rows(df.to_dict(orient="records"))

    # ---- brand map ----
    def upsert_brand_map(self, mapping: Dict[str, Dict[str, Any]]) -> None:
        """
        mapping: {MERCHANT: {"ticker": "AAPL", "weight": 0.6}, ...}
        """
        with self._cx() as cx:
            for m, v in mapping.items():
                cx.execute("INSERT OR REPLACE INTO brand_map(merchant,ticker,weight) VALUES(?,?,?)",
                           (str(m).upper(), str(v.get("ticker","")).upper(), float(v.get("weight",1.0))))
            cx.commit()

    def load_brand_map_yaml(self, path: str) -> None:
        assert yaml is not None, "pyyaml not installed"
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        # allow simple {merchant: ticker} as well
        mapping: Dict[str, Dict[str, Any]] = {}
        for k, v in (doc.items() if isinstance(doc, dict) else []):
            if isinstance(v, str):
                mapping[k] = {"ticker": v, "weight": 1.0}
            else:
                mapping[k] = {"ticker": v.get("ticker",""), "weight": v.get("weight", 1.0)}
        self.upsert_brand_map(mapping)

    # ---- compute index ----
    def compute_daily(self) -> int:
        """
        Aggregates card_tx -> spend_index for levels: merchant/category/country and ticker (via brand_map).
        Returns rows written.
        """
        written = 0
        with self._cx() as cx:
            # Which days need recompute? Simple approach: all distinct days present in tx not in index for merchant level.
            days = [r["day"] for r in cx.execute("SELECT DISTINCT day FROM card_tx ORDER BY day").fetchall()]
            # Preload brand map
            bmap = {r["merchant"]: (r["ticker"], r["weight"]) for r in cx.execute("SELECT merchant,ticker,weight FROM brand_map").fetchall()}

            for day in days:
                # merchant
                rows = cx.execute("""
                    SELECT merchant AS key, SUM(amount) AS gross,
                           COUNT(DISTINCT user_id) AS users
                    FROM card_tx WHERE day=? GROUP BY merchant
                """, (day,)).fetchall()
                for r in rows:
                    meta = {}
                    written += self._write_index_row(cx, "merchant", r["key"], day, r["gross"], r["users"], meta)

                # category
                rows = cx.execute("""
                    SELECT category AS key, SUM(amount) AS gross,
                           COUNT(DISTINCT user_id) AS users
                    FROM card_tx WHERE day=? GROUP BY category
                """, (day,)).fetchall()
                for r in rows:
                    written += self._write_index_row(cx, "category", r["key"], day, r["gross"], r["users"], {})

                # country
                rows = cx.execute("""
                    SELECT country AS key, SUM(amount) AS gross,
                           COUNT(DISTINCT user_id) AS users
                    FROM card_tx WHERE day=? GROUP BY country
                """, (day,)).fetchall()
                for r in rows:
                    written += self._write_index_row(cx, "country", r["key"], day, r["gross"], r["users"], {})

                # ticker (brand roll-up)
                # We multiply merchant gross by weight and sum into ticker
                rows = cx.execute("""
                    SELECT merchant, SUM(amount) AS gross, COUNT(DISTINCT user_id) AS users
                    FROM card_tx WHERE day=? GROUP BY merchant
                """, (day,)).fetchall()
                ticker_gross: Dict[str, Tuple[float, int]] = {}
                for r in rows:
                    m = r["merchant"]
                    if m in bmap and bmap[m][0]:
                        tk, w = bmap[m]
                        g = float(r["gross"]) * float(w or 1.0)
                        u = int(r["users"] or 0)
                        gg, uu = ticker_gross.get(tk, (0.0, 0))
                        ticker_gross[tk] = (gg + g, uu + u)
                for tk, (g, u) in ticker_gross.items():
                    written += self._write_index_row(cx, "ticker", tk, day, g, u, {"brand_rollup": True})

            cx.commit()
        # growth + trend pass
        self._compute_growth_and_trend()
        return written

    def _write_index_row(self, cx: sqlite3.Connection, level: str, key: str, day: str,
                         gross: float, users: int, meta: Dict[str, Any]) -> int:
        if not key:
            return 0
        avg_ticket = (gross / users) if users and users > 0 else None
        cx.execute("""
            INSERT OR REPLACE INTO spend_index(level,key,day,gross,users,avg_ticket,wow,mom,yoy,dod,trend,anomaly,meta)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (level, str(key).upper(), day, float(gross or 0.0), int(users or 0),
              avg_ticket, None, None, None, None, None, None, json.dumps(meta or {})))
        return 1

    def _compute_growth_and_trend(self) -> None:
        with self._cx() as cx:
            keys = cx.execute("SELECT DISTINCT level, key FROM spend_index").fetchall()
            for row in keys:
                level, key = row["level"], row["key"]
                series = cx.execute("""
                    SELECT day, gross FROM spend_index
                    WHERE level=? AND key=? ORDER BY day
                """, (level, key)).fetchall()
                days = [r["day"] for r in series]
                vals = [float(r["gross"] or 0.0) for r in series]
                # compute growth metrics
                growth = _growth_metrics(days, vals)
                # trend + anomaly
                trend, anomaly = _trend_and_anomaly(vals)
                for i, day in enumerate(days):
                    g = growth.get(day, {})
                    tr = trend[i] if i < len(trend) else None
                    an = anomaly[i] if i < len(anomaly) else None
                    cx.execute("""
                        UPDATE spend_index
                        SET dod=?, wow=?, mom=?, yoy=?, trend=?, anomaly=?
                        WHERE level=? AND key=? AND day=?
                    """, (g.get("dod"), g.get("wow"), g.get("mom"), g.get("yoy"), tr, an, level, key, day))
            cx.commit()

    # ---- query/export ----
    def latest(self, level: str, key: str, n: int = 30) -> List[Dict[str, Any]]:
        with self._cx() as cx:
            rows = cx.execute("""
                SELECT * FROM spend_index WHERE level=? AND key=? ORDER BY day DESC LIMIT ?
            """, (level, key.upper(), n)).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, path: str) -> str:
        _ensure_dir(path)
        with self._cx() as cx, open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = [c[1] for c in cx.execute("PRAGMA table_info(spend_index)").fetchall()]
            w.writerow(cols)
            for r in cx.execute("SELECT * FROM spend_index ORDER BY level,key,day").fetchall():
                w.writerow([r[c] for c in cols])
        return path


# -------------------------- math helpers --------------------------

def _growth_metrics(days: List[str], vals: List[float]) -> Dict[str, Dict[str, float]]:
    """
    Return dict[day] -> {dod, wow, mom, yoy} in percent.
    Aligns to calendar by string ops: day 'YYYY-MM-DD'; week math not needed.
    """
    from datetime import datetime, timedelta
    idx = {d: i for i, d in enumerate(days)}
    out: Dict[str, Dict[str, float]] = {}
    for i, d in enumerate(days):
        v = vals[i]
        dt = datetime.strptime(d, "%Y-%m-%d")
        def pct(prev_val):
            if prev_val is None or prev_val <= 0:
                return None
            return (v - prev_val) / prev_val * 100.0
        # DoD
        d_prev = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        dod = pct(vals[idx[d_prev]]) if d_prev in idx else None
        # WoW (same weekday last week)
        d_wow = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        wow = pct(vals[idx[d_wow]]) if d_wow in idx else None
        # MoM (approx 30d)
        d_mom = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
        mom = pct(vals[idx[d_mom]]) if d_mom in idx else None
        # YoY (365d)
        d_yoy = (dt - timedelta(days=365)).strftime("%Y-%m-%d")
        yoy = pct(vals[idx[d_yoy]]) if d_yoy in idx else None
        out[d] = {"dod": dod, "wow": wow, "mom": mom, "yoy": yoy} # type: ignore
    return out

def _rolling(vals: List[float], win: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    s = 0.0
    q: List[float] = []
    for v in vals:
        q.append(v)
        s += v
        if len(q) > win:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def _trend_and_anomaly(vals: List[float]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Very simple: rolling median & MAD z-score-ish anomaly. Trend via 14d EMA proxy using rolling mean if pandas absent.
    """
    if not vals:
        return [], []
    # Trend
    if pd is not None:
        s = pd.Series(vals, dtype="float64")
        trend = s.ewm(span=14, adjust=False).mean().tolist()
    else:
        trend = _rolling(vals, 14)

    # Anomaly via robust z-score (median/MAD)
    med = stats.median(vals)
    mad = stats.median([abs(x - med) for x in vals]) or 1e-6
    z = [ (x - med) / (1.4826 * mad) for x in vals ]
    # Cap to +/- 6
    z = [max(-6.0, min(6.0, float(v))) for v in z]
    return trend, z # type: ignore


# -------------------------- publisher --------------------------

def publish_latest_signals(store: CardStore, *, top_k: int = 20, min_days: int = 14) -> None:
    if not publish_stream:
        return
    # pick most recent day across all rows
    with store._cx() as cx:
        row = cx.execute("SELECT day FROM spend_index ORDER BY day DESC LIMIT 1").fetchone()
        if not row: return
        day = row["day"]
        # merchant movers by yoy/wow
        movers = cx.execute("""
            SELECT key, yoy, wow, gross, users FROM spend_index
            WHERE level='merchant' AND day=?
            ORDER BY COALESCE(yoy,0) DESC
            LIMIT ?
        """, (day, top_k)).fetchall()
        payload = {
            "ts_ms": _utc_ms(),
            "day": day,
            "top_merchants": [{"merchant": r["key"], "yoy": r["yoy"], "wow": r["wow"], "gross": r["gross"], "users": r["users"]} for r in movers]
        }
        publish_stream("altdata.card_spend", payload)
        # tiny insight
        if movers:
            m0 = movers[0]
            publish_stream("ai.insight", {
                "ts_ms": payload["ts_ms"],
                "kind": "card_spend",
                "summary": f"Top YoY merchant: {m0['key']} (+{(m0['yoy'] or 0):.1f}%)",
                "details": [f"WoW={(m0['wow'] or 0):.1f}%", f"Gross={m0['gross']:.0f}"],
                "tags": ["altdata","card-spend", m0["key"]]
            })


# -------------------------- CLI --------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Card Spend Analytics (Alt-Data)")
    ap.add_argument("--db", type=str, default="runtime/altdata.db")
    ap.add_argument("--load", type=str, help="Load transactions from CSV/JSON/Parquet")
    ap.add_argument("--delimiter", type=str, default=",")
    ap.add_argument("--brand-map", type=str, help="YAML mapping MERCHANT->ticker/weight")
    ap.add_argument("--reindex", action="store_true", help="Recompute daily indices/growth/trend")
    ap.add_argument("--save", action="store_true", help="Alias for --reindex (kept for compatibility)")
    ap.add_argument("--export", type=str, help="Export spend_index to CSV")
    ap.add_argument("--publish", action="store_true", help="Publish latest movers to bus")
    ap.add_argument("--probe", action="store_true", help="Load a tiny synthetic sample and run end-to-end")
    args = ap.parse_args()

    store = CardStore(db_path=args.db)

    if args.probe:
        # minimal synthetic
        now = _utc_ms()
        syn = []
        for i in range(40):
            ts = now - (39 - i)*24*3600*1000
            syn += [
                {"ts": ts, "amount": 12.5 + i*0.3, "merchant": "STARBUCKS", "category": "FOOD", "country": "US", "user_id": "u1"},
                {"ts": ts, "amount": 45.0 + (i%7), "merchant": "APPLE STORE", "category": "ELECTRONICS", "country": "US", "user_id": "u2"},
                {"ts": ts, "amount": 20 + (i%5), "merchant": "RELIANCE RETAIL", "category": "RETAIL", "country": "IN", "user_id": "u3"},
            ]
        store.ingest_rows(syn)
        store.upsert_brand_map({
            "STARBUCKS": {"ticker": "SBUX", "weight": 1.0},
            "APPLE STORE": {"ticker": "AAPL", "weight": 1.0},
            "RELIANCE RETAIL": {"ticker": "RELIANCE.NS", "weight": 1.0},
        })
        store.compute_daily()
        if args.publish:
            publish_latest_signals(store)
        if args.export:
            p = store.export_csv(args.export)
            print(f"Wrote {p}")
        return

    if args.load:
        path = args.load
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            n = store.load_csv(path, delimiter=args.delimiter)
        elif ext in (".json", ".ndjson"):
            n = store.load_json(path)
        elif ext in (".parquet", ".pq"):
            n = store.load_parquet(path)
        else:
            raise SystemExit("Unsupported --load file type (use .csv/.json/.parquet)")
        print(f"Ingested {n} transactions from {path}")

    if args.brand_map:
        store.load_brand_map_yaml(args.brand_map)
        print(f"Loaded brand map from {args.brand_map}")

    if args.reindex or args.save:
        rows = store.compute_daily()
        print(f"Computed/updated {rows} index rows")

    if args.publish:
        publish_latest_signals(store)
        print("Published latest movers to bus")

    if args.export:
        p = store.export_csv(args.export)
        print(f"Wrote {p}")

if __name__ == "__main__":
    main()