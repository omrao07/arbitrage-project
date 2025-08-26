# backend/compliance/sebi_otr.py
"""
SEBI / Exchange OTR Monitor (NSE/BSE)
-------------------------------------
Tracks Order-to-Trade Ratio (OTR) and related metrics and alerts on breaches.

What it does
- Computes OTR = order_msgs / trade_msgs (configurable formula)
- Buckets by member_id, user_id, strategy_id, symbol, segment (EQ/FO/CUR), and time windows
- Supports multiple windows (e.g., 1m, 5m, session/day)
- Tracks new/cancel/modify counts + trades (fills) and notional
- Emits alerts when configured slabs/thresholds are crossed
- Persists rolling counters for auditability
- Daily CSV report + optional Redis/bus insight events

Inputs (best-effort; from your normalizer or OMS):
  raw order messages via 'oms.child' or 'oms.parent', and trades via 'oms.fill'.

  - Order message (child or parent submissions/replace/cancel)
    {
      "ts_ms": 1690000000123,
      "member_id": "BRK123",         # your broker/member code
      "user_id": "U1001",            # dealer/user code
      "strategy": "alpha.momo",      # optional
      "symbol": "RELIANCE.NS",
      "segment": "EQ"|"FO"|"CUR",    # optional
      "venue": "NSE"|"BSE",
      "typ": "new"|"modify"|"cancel" # inferred if absent
    }

  - Trade message (fill)
    {
      "ts_ms": 1690000000456,
      "member_id": "BRK123",
      "user_id": "U1001",
      "strategy": "alpha.momo",
      "symbol": "RELIANCE.NS",
      "segment": "EQ",
      "venue": "NSE",
      "price": 2900.25,
      "qty": 100
    }

Outputs
- Realtime alerts to `compliance.otr.alerts` (if bus present)
- Compact insight bullets to `ai.insight` (optional)
- Rolling counters in memory (optionally mirrored to Redis hash)
- Daily CSV at runtime/otr_YYYYMMDD.csv (per bucket + window)

CLI
  python -m backend.compliance.sebi_otr --probe
  python -m backend.compliance.sebi_otr --run

Config (YAML)
  config/sebi_otr.yaml (all fields optional; sane defaults included below)
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict, deque

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Optional bus (graceful if missing)
try:
    from backend.bus.streams import consume_stream, publish_stream, hset
except Exception:
    consume_stream = publish_stream = hset = None  # type: ignore


# ----------------------------- utils -----------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _day_tag(ts_ms: int) -> str:
    t = time.gmtime(ts_ms / 1000)
    return f"{t.tm_year:04d}{t.tm_mon:02d}{t.tm_mday:02d}"

def _open_daily_csv(base_dir: str, prefix: str, columns: List[str]):
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{prefix}_{_day_tag(_utc_ms())}.csv")
    exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    if not exists:
        w.writeheader()
    return f, w, path

def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

# ----------------------------- config -----------------------------

_DEFAULT_CFG = {
    "windows_ms": [60_000, 300_000, 21_600_000],     # 1m, 5m, 6h (session-ish)
    "group_by": ["member_id", "user_id", "strategy", "segment", "symbol", "venue"],
    "otr_formula": "orders/trades",                  # or "messages/trades"
    "count_modify_as_order": True,                   # include modify in 'orders'
    "count_cancel_as_order": True,                   # include cancel in 'orders'
    "alert_thresholds": [50.0, 100.0, 500.0],        # slabs (example). Keep in YAML for live rules.
    "min_trades_for_valid_otr": 1,                   # avoid div-by-zero noise
    "emit_insights": True,
    "persist_to_redis": False,
    "csv_out_dir": "runtime",
    "csv_prefix": "otr",
    "topic_orders": ["oms.child", "oms.parent"],     # order messages
    "topic_trades": ["oms.fill"],                    # fills
}

@dataclass
class OtrConfig:
    windows_ms: List[int] = field(default_factory=lambda: _DEFAULT_CFG["windows_ms"])
    group_by: List[str] = field(default_factory=lambda: _DEFAULT_CFG["group_by"])
    otr_formula: str = _DEFAULT_CFG["otr_formula"]
    count_modify_as_order: bool = True
    count_cancel_as_order: bool = True
    alert_thresholds: List[float] = field(default_factory=lambda: _DEFAULT_CFG["alert_thresholds"])
    min_trades_for_valid_otr: int = 1
    emit_insights: bool = True
    persist_to_redis: bool = False
    csv_out_dir: str = "runtime"
    csv_prefix: str = "otr"
    topic_orders: List[str] = field(default_factory=lambda: _DEFAULT_CFG["topic_orders"])
    topic_trades: List[str] = field(default_factory=lambda: _DEFAULT_CFG["topic_trades"])

def load_config(path: str = "config/sebi_otr.yaml") -> OtrConfig:
    if yaml and os.path.exists(path):
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = {**_DEFAULT_CFG, **raw}
    else:
        cfg = dict(_DEFAULT_CFG)
    return OtrConfig(**cfg)


# ----------------------------- core monitor -----------------------------

def _bucket_key(msg: Dict[str, Any], group_by: List[str]) -> Tuple:
    key = []
    for g in group_by:
        key.append(_get(msg, g, default="*"))
    return tuple(key)

class RollingCounters:
    """
    Maintain rolling counts inside a fixed window using deques.
    """
    def __init__(self, window_ms: int):
        self.window_ms = int(window_ms)
        # store tuples (ts_ms, kind, qty, notional)
        self.events: deque = deque(maxlen=200000)
        # aggregated counters
        self.orders = 0
        self.modifies = 0
        self.cancels = 0
        self.trades = 0
        self.trade_qty = 0.0
        self.trade_notional = 0.0

    def ingest_order(self, ts_ms: int, typ: str):
        self.events.append((ts_ms, "order", 0.0, 0.0))
        self.orders += 1
        if typ == "modify":
            self.modifies += 1
        elif typ == "cancel":
            self.cancels += 1

    def ingest_trade(self, ts_ms: int, qty: float, px: float):
        self.events.append((ts_ms, "trade", qty, qty * px))
        self.trades += 1
        self.trade_qty += float(qty)
        self.trade_notional += float(qty) * float(px)

    def evict(self, now_ms: int):
        w = self.window_ms
        while self.events and (now_ms - self.events[0][0] > w):
            ts, kind, qty, notion = self.events.popleft()
            if kind == "order":
                self.orders -= 1
                # cannot precisely decrement modifies/cancels here; acceptable for rolling signal
            else:
                self.trades -= 1
                self.trade_qty -= float(qty)
                self.trade_notional -= float(notion)

    def snapshot(self, cfg: OtrConfig, now_ms: int) -> Dict[str, Any]:
        self.evict(now_ms)
        # choose numerator based on formula
        if cfg.otr_formula == "messages/trades":
            numerator = len(self.events)  # crude: all messages considered
        else:
            numerator = self.orders + (self.cancels if cfg.count_cancel_as_order else 0) + (self.modifies if cfg.count_modify_as_order else 0)
        denom = max(0, self.trades)
        otr = float("inf") if denom == 0 and numerator > 0 else (0.0 if denom == 0 else numerator / denom)
        return {
            "orders": max(0, self.orders),
            "trades": max(0, self.trades),
            "trade_qty": max(0.0, self.trade_qty),
            "trade_notional": max(0.0, self.trade_notional),
            "otr": float(otr),
        }

class OtrMonitor:
    def __init__(self, cfg: Optional[OtrConfig] = None):
        self.cfg = cfg or load_config()
        # state: per window, per bucket -> counters
        self.state: Dict[int, Dict[Tuple, RollingCounters]] = {
            w: defaultdict(lambda w=w: RollingCounters(w)) for w in self.cfg.windows_ms
        }
        # for CSV flush throttling
        self._last_csv_write_ms = 0

    # ---- ingestion ----
    def on_order(self, msg: Dict[str, Any]) -> None:
        ts = int(_get(msg, "ts_ms", "ts", default=_utc_ms())) # type: ignore
        typ = str(_get(msg, "typ", default="new")).lower()
        bucket = _bucket_key(msg, self.cfg.group_by)
        for w, buckets in self.state.items():
            buckets[bucket].ingest_order(ts, typ)

    def on_trade(self, msg: Dict[str, Any]) -> None:
        ts = int(_get(msg, "ts_ms", "ts", default=_utc_ms())) # type: ignore
        px = float(_get(msg, "price", "px", default=0.0) or 0.0)
        qty = float(_get(msg, "qty", default=0.0) or 0.0)
        if px <= 0 or qty <= 0:
            return
        bucket = _bucket_key(msg, self.cfg.group_by)
        for w, buckets in self.state.items():
            buckets[bucket].ingest_trade(ts, qty, px)

    # ---- compute & alert ----
    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Returns a list of alerts generated on this tick.
        """
        now = _utc_ms()
        alerts: List[Dict[str, Any]] = []
        for w, buckets in self.state.items():
            for bucket_key, rc in list(buckets.items()):
                snap = rc.snapshot(self.cfg, now)
                # alert only if enough trades to be meaningful
                if snap["trades"] < self.cfg.min_trades_for_valid_otr:
                    continue
                ratio = snap["otr"]
                slab = self._breach_slab(ratio)
                if slab is not None:
                    alert = self._build_alert(bucket_key, w, snap, slab, now)
                    alerts.append(alert)
                    self._emit_alert(alert)
                    if self.cfg.emit_insights and publish_stream:
                        publish_stream("ai.insight", {
                            "ts_ms": now,
                            "kind": "otr",
                            "summary": f"OTR breach slab {slab} at {ratio:.1f} ({w//1000}s window)",
                            "details": [f"bucket={self._bucket_dict(bucket_key)}", f"orders={snap['orders']} trades={snap['trades']}"],
                            "tags": ["compliance","otr"]
                        })
                # optionally mirror a compact snapshot to Redis for dashboards
                if self.cfg.persist_to_redis and hset:
                    try:
                        hset(f"otr:last:{w}", str(bucket_key), {
                            "otr": ratio, "orders": snap["orders"], "trades": snap["trades"],
                            "qty": snap["trade_qty"], "notional": snap["trade_notional"]
                        })
                    except Exception:
                        pass

        # write CSV periodically (every ~30s)
        if now - self._last_csv_write_ms >= 30_000:
            try:
                self.write_csv()
                self._last_csv_write_ms = now
            except Exception:
                pass
        return alerts

    def _breach_slab(self, ratio: float) -> Optional[float]:
        for t in sorted(self.cfg.alert_thresholds):
            if ratio >= t:
                slab = t
        return locals().get("slab")

    def _bucket_dict(self, key: Tuple) -> Dict[str, Any]:
        return {g: key[i] for i, g in enumerate(self.cfg.group_by)}

    def _build_alert(self, bucket_key: Tuple, window_ms: int, snap: Dict[str, Any], slab: float, now: int) -> Dict[str, Any]:
        data = {
            "ts_ms": now,
            "window_ms": int(window_ms),
            "bucket": self._bucket_dict(bucket_key),
            "orders": int(snap["orders"]),
            "trades": int(snap["trades"]),
            "otr": float(snap["otr"]),
            "slab": float(slab),
        }
        return data

    def _emit_alert(self, alert: Dict[str, Any]) -> None:
        if publish_stream:
            publish_stream("compliance.otr.alerts", alert)

    # ---- CSV reporting ----
    def write_csv(self) -> Optional[str]:
        """
        Append a snapshot of all buckets/windows to a daily CSV.
        """
        cols = ["day","asof_ms","window_s"] + self.cfg.group_by + ["orders","trades","otr","trade_qty","trade_notional"]
        f, w, path = _open_daily_csv(self.cfg.csv_out_dir, self.cfg.csv_prefix, cols)
        now = _utc_ms()
        day = _day_tag(now)
        try:
            for wms, buckets in self.state.items():
                window_s = wms // 1000
                for key, rc in buckets.items():
                    snap = rc.snapshot(self.cfg, now)
                    row = {
                        "day": day,
                        "asof_ms": now,
                        "window_s": window_s,
                        **self._bucket_dict(key),
                        "orders": snap["orders"],
                        "trades": snap["trades"],
                        "otr": round(snap["otr"], 4) if snap["trades"] else None,
                        "trade_qty": round(snap["trade_qty"], 6),
                        "trade_notional": round(snap["trade_notional"], 2),
                    }
                    w.writerow(row)
        finally:
            f.close()
        return path


# ----------------------------- runner -----------------------------

def run_loop(cfg: Optional[OtrConfig] = None):
    """
    Attach to bus topics defined in config; consume and evaluate continuously.
    """
    assert consume_stream is not None, "bus streams not available"
    mon = OtrMonitor(cfg or load_config())

    cursors = {"orders": "$", "trades": "$"}
    while True:
        # Orders
        for topic in mon.cfg.topic_orders:
            try:
                for _, raw in consume_stream(topic, start_id=cursors["orders"], block_ms=100, count=500):
                    cursors["orders"] = "$"
                    try:
                        msg = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception:
                        continue
                    # Infer 'typ' if missing
                    typ = str(_get(msg, "typ", "type", default="new")).lower()
                    m = {
                        "ts_ms": _get(msg, "ts_ms", "ts", default=_utc_ms()),
                        "member_id": _get(msg, "member_id", default="*"),
                        "user_id": _get(msg, "user_id", default="*"),
                        "strategy": _get(msg, "strategy", "strategy_name", default=None),
                        "symbol": _get(msg, "symbol", default=None),
                        "segment": _get(msg, "segment", default=None),
                        "venue": _get(msg, "venue", default=None),
                        "typ": typ,
                    }
                    mon.on_order(m)
            except Exception:
                pass
        # Trades
        for topic in mon.cfg.topic_trades:
            try:
                for _, raw in consume_stream(topic, start_id=cursors["trades"], block_ms=100, count=500):
                    cursors["trades"] = "$"
                    try:
                        msg = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception:
                        continue
                    m = {
                        "ts_ms": _get(msg, "ts_ms", "ts", default=_utc_ms()),
                        "member_id": _get(msg, "member_id", default="*"),
                        "user_id": _get(msg, "user_id", default="*"),
                        "strategy": _get(msg, "strategy", "strategy_name", default=None),
                        "symbol": _get(msg, "symbol", default=None),
                        "segment": _get(msg, "segment", default=None),
                        "venue": _get(msg, "venue", default=None),
                        "price": _get(msg, "price", "px", default=0.0),
                        "qty": _get(msg, "qty", default=0.0),
                    }
                    mon.on_trade(m)
            except Exception:
                pass

        # Evaluate + (periodically) write CSV
        try:
            mon.evaluate()
        except Exception:
            pass

        time.sleep(0.05)


# ----------------------------- CLI -----------------------------

def _probe():
    cfg = load_config()
    mon = OtrMonitor(cfg)
    now = _utc_ms()
    # synthetic: 99 orders, 1 trade in 1 minute for one user
    for i in range(99):
        mon.on_order({"ts_ms": now - 20_000 + i * 100, "member_id":"BRK123","user_id":"U1","strategy":"alpha","symbol":"RELIANCE.NS","segment":"EQ","venue":"NSE","typ":"new"})
    mon.on_trade({"ts_ms": now - 5_000, "member_id":"BRK123","user_id":"U1","strategy":"alpha","symbol":"RELIANCE.NS","segment":"EQ","venue":"NSE","price":2900.0,"qty":10})
    alerts = mon.evaluate()
    print(json.dumps({"alerts": alerts, "csv": mon.write_csv()}, indent=2))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="SEBI / Exchange OTR Monitor")
    ap.add_argument("--run", action="store_true", help="Attach to bus and monitor OTR")
    ap.add_argument("--probe", action="store_true", help="Run synthetic example and print result")
    args = ap.parse_args()
    if args.probe:
        _probe(); return
    if args.run:
        if not consume_stream:
            print("Bus not available. Exiting.")
            return
        try:
            run_loop()
        except KeyboardInterrupt:
            pass
        return
    ap.print_help()

if __name__ == "__main__":
    main()