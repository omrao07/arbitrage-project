# backend/data/normalizer.py
"""
Unified Market/OMS Message Normalizer
-------------------------------------
Normalizes heterogeneous provider payloads into compact, consistent envelopes:

Emits (topics you can subscribe to elsewhere):
  - ticks.trades   {ts_ms, symbol, venue, price, qty, side?, id?}
  - ticks.quotes   {ts_ms, symbol, venue, bid, ask, bidsz?, asksz?}
  - bars.1s|1m     {ts_ms, symbol, o, h, l, c, v, venue?}
  - news.events    {ts_ms, source, symbol?, headline, link?, id?}
  - oms.parent     {order_id, symbol, side, qty, ts_ms, strategy?, route? ...}
  - oms.child      {child_id, parent_id, ts_ms, venue?, typ?, px?, qty?}
  - oms.fill       {fill_id, parent_id, child_id?, symbol, side, price, qty, ts_ms, venue?, fee?}

It also exposes pure functions so you can use it as a library:
  normalize_trade(msg, source=...), normalize_quote(...), normalize_news(...), etc.

Design goals
- Provider-agnostic adapters (see Provider enum + small heuristics)
- Idempotency: stable event key (`norm_hash`) for de-dupe
- Robust ts normalization to **ms** since epoch (accepts iso8601, ns/us/ms/s)
- Symbol canonicalization (e.g., NSE/BSE suffixes, crypto pairs, futures root+expiry)
- Lightweight validation & value clamping to avoid NaNs/negatives
- Optional re-publish via your `backend.bus.streams` if available

CLI:
  python -m backend.data.normalizer --probe          # prints a few normalized sample rows
  python -m backend.data.normalizer --run            # bridge: reads raw.* -> emits normalized topics

Wire-up tip:
- Aim raw feeds at topics like `raw.trades.binance`, `raw.quotes.nse`, `raw.news.yahoo`, `raw.oms.*`.
- This process listens to `raw.*` and outputs clean topics above.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Optional bus (graceful if missing)
try:
    from backend.bus.streams import consume_stream, publish_stream
except Exception:
    consume_stream = publish_stream = None  # type: ignore

# -------------- provider hints --------------

class Provider(enum.Enum):
    UNKNOWN = "unknown"
    NSE = "nse"
    BSE = "bse"
    YFIN = "yahoo"
    MC = "moneycontrol"
    BINANCE = "binance"
    ALPACA = "alpaca"
    IB = "ib"
    ZERODHA = "zerodha"
    CUSTOM = "custom"

# -------------- helpers --------------

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+\-]\d{2}:\d{2})?$")

def _utc_ms_now() -> int:
    return int(time.time() * 1000)

def _to_ms(ts: Any) -> Optional[int]:
    """
    Coerce many timestamp shapes into epoch ms.
    Accepts: int/float (s, ms, us, ns heuristics), or ISO8601 string.
    """
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            x = float(ts)
            # Heuristics by magnitude
            if x > 1e16:   # ns
                return int(x / 1e6)
            if x > 1e14:   # us
                return int(x / 1e3)
            if x > 1e12:   # already ms
                return int(x)
            if x > 1e9:    # s with decimals (unlikely)
                return int(x * 1000.0)
            # seconds
            return int(x * 1000.0)
        s = str(ts).strip()
        if _ISO_RE.match(s):
            # minimal ISO parser: use time.strptime for Z; offset not fully applied here.
            # In production, prefer dateutil.parser; we keep stdlib-only.
            from datetime import datetime, timezone
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception:
                pass
        # last resort: int cast
        return _to_ms(float(s))
    except Exception:
        return None

def _clamp_pos(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None

def _float_or_none(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _norm_symbol(sym: Any, prov: Provider) -> str:
    if not sym:
        return ""
    s = str(sym).strip().upper()
    # NSE/BSE normalization
    if prov in {Provider.NSE, Provider.ZERODHA}:
        if not s.endswith(".NS") and "." not in s and "-" not in s and ":" not in s:
            return s + ".NS"
    if prov == Provider.BSE:
        if not s.endswith(".BO") and "." not in s:
            return s + ".BO"
    # Binance pairs (ensure base/quote separator)
    if prov == Provider.BINANCE and len(s) >= 6 and "/" not in s and "-" not in s:
        # Try to insert slash between common fiat/USDT
        for q in ("USDT","BUSD","USDC","BTC","ETH"):
            if s.endswith(q):
                return s[:-len(q)] + "/" + q
    return s

def _side_str(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x).lower()
    if s.startswith("b"): return "buy"
    if s.startswith("s"): return "sell"
    return None

def _norm_hash(payload: Dict[str, Any]) -> str:
    b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(b).hexdigest()

# -------------- normalizers --------------

def normalize_trade(msg: Dict[str, Any], *, source: str = "", provider: Provider = Provider.UNKNOWN) -> Optional[Dict[str, Any]]:
    """
    Inputs accepted (best-effort):
      - price: price|p|last|ltp
      - qty: qty|size|q
      - symbol: symbol|s|sym|instrument
      - ts: ts_ms|ts|timestamp|time
      - side (optional): side|taker_side
      - venue/exchange (optional): venue|exchange|exch
      - id (optional): id|trade_id
    """
    sym = _norm_symbol(msg.get("symbol") or msg.get("s") or msg.get("sym") or msg.get("instrument"), provider)
    px = _clamp_pos(msg.get("price") or msg.get("p") or msg.get("last") or msg.get("ltp"))
    qty = _clamp_pos(msg.get("qty") or msg.get("size") or msg.get("q"))
    ts  = _to_ms(msg.get("ts_ms") or msg.get("ts") or msg.get("timestamp") or msg.get("time"))
    if not (sym and px and qty and ts):
        return None
    out = {
        "ts_ms": ts,
        "symbol": sym,
        "venue": (msg.get("venue") or msg.get("exchange") or msg.get("exch") or "").upper() or None,
        "price": float(px),
        "qty": float(qty),
    }
    side = _side_str(msg.get("side") or msg.get("taker_side"))
    if side: out["side"] = side
    if msg.get("id") or msg.get("trade_id"):
        out["id"] = str(msg.get("id") or msg.get("trade_id"))
    out["source"] = source or provider.value
    out["norm_hash"] = _norm_hash(out)
    return out

def normalize_quote(msg: Dict[str, Any], *, source: str = "", provider: Provider = Provider.UNKNOWN) -> Optional[Dict[str, Any]]:
    """
    Inputs accepted (best-effort):
      - bid/b, ask/a (+ optional sizes)
      - symbol, ts, venue/exchange
    """
    sym = _norm_symbol(msg.get("symbol") or msg.get("s") or msg.get("sym") or msg.get("instrument"), provider)
    bid = _clamp_pos(msg.get("bid") if msg.get("bid") is not None else msg.get("b"))
    ask = _clamp_pos(msg.get("ask") if msg.get("ask") is not None else msg.get("a"))
    if not (sym and bid and ask):
        return None
    ts  = _to_ms(msg.get("ts_ms") or msg.get("ts") or msg.get("timestamp") or msg.get("time")) or _utc_ms_now()
    out = {
        "ts_ms": ts,
        "symbol": sym,
        "venue": (msg.get("venue") or msg.get("exchange") or msg.get("exch") or "").upper() or None,
        "bid": float(bid),
        "ask": float(ask),
    }
    bs = _float_or_none(msg.get("bsize") or msg.get("bidsz") or msg.get("bid_size"))
    if bs is not None: out["bidsz"] = bs
    asz = _float_or_none(msg.get("asize") or msg.get("asksz") or msg.get("ask_size"))
    if asz is not None: out["asksz"] = asz
    out["source"] = source or provider.value
    out["norm_hash"] = _norm_hash(out)
    return out

def normalize_bar(msg: Dict[str, Any], *, source: str = "", provider: Provider = Provider.UNKNOWN) -> Optional[Dict[str, Any]]:
    """
    Inputs accepted:
      - o/h/l/c/v (open/high/low/close/volume)
      - symbol, ts (bar end), venue?, timeframe? (e.g., "1s","1m")
    """
    sym = _norm_symbol(msg.get("symbol") or msg.get("s"), provider)
    o = _clamp_pos(msg.get("o") or msg.get("open"))
    h = _clamp_pos(msg.get("h") or msg.get("high"))
    l = _clamp_pos(msg.get("l") or msg.get("low"))
    c = _clamp_pos(msg.get("c") or msg.get("close"))
    v = _clamp_pos(msg.get("v") or msg.get("vol") or msg.get("volume")) or 0.0
    if not (sym and o and h and l and c):
        return None
    ts = _to_ms(msg.get("ts_ms") or msg.get("ts") or msg.get("t")) or _utc_ms_now()
    out = {
        "ts_ms": ts,
        "symbol": sym,
        "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v),
        "venue": (msg.get("venue") or "").upper() or None,
        "tf": (msg.get("timeframe") or msg.get("tf") or "").lower() or None,
        "source": source or provider.value,
    }
    out["norm_hash"] = _norm_hash(out)
    return out

def normalize_news(msg: Dict[str, Any], *, source: str = "", provider: Provider = Provider.UNKNOWN) -> Optional[Dict[str, Any]]:
    """
    Inputs accepted:
      - headline/title, link/url, ts
      - symbols (symbol|tickers|tags) – string or list
    """
    headline = (msg.get("headline") or msg.get("title") or "").strip()
    if not headline:
        return None
    ts = _to_ms(msg.get("ts_ms") or msg.get("ts") or msg.get("published") or msg.get("time")) or _utc_ms_now()
    syms = msg.get("symbols") or msg.get("tickers") or msg.get("tags")
    symbol = None
    if isinstance(syms, (list, tuple)) and syms:
        symbol = _norm_symbol(syms[0], provider)
    elif isinstance(syms, str):
        symbol = _norm_symbol(syms, provider)
    out = {
        "ts_ms": ts,
        "source": source or provider.value,
        "symbol": symbol,
        "headline": headline,
        "link": msg.get("link") or msg.get("url") or None,
        "id": str(msg.get("id") or msg.get("guid") or ""),
    }
    out["norm_hash"] = _norm_hash(out)
    return out

def normalize_parent_order(msg: Dict[str, Any], *, provider: Provider = Provider.UNKNOWN) -> Optional[Dict[str, Any]]:
    oid = str(msg.get("order_id") or msg.get("parent_id") or "")
    sym = _norm_symbol(msg.get("symbol") or "", provider)
    side = _side_str(msg.get("side"))
    qty  = _clamp_pos(msg.get("qty"))
    ts   = _to_ms(msg.get("ts_ms") or msg.get("ts"))
    if not (oid and sym and side and qty and ts):
        return None
    out = {
        "order_id": oid, "symbol": sym, "side": side, "qty": float(qty),
        "ts_ms": ts, "strategy": msg.get("strategy") or msg.get("strategy_name"),
        "route": msg.get("route") or msg.get("route_hint"),
        "mark_px": _float_or_none(msg.get("mark_px") or msg.get("mark_price")),
        "urgency": _float_or_none(msg.get("urgency")),
        "asset_class": msg.get("asset_class") or "equity",
    }
    out["norm_hash"] = _norm_hash(out)
    return out

def normalize_child_order(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cid = str(msg.get("child_id") or msg.get("order_id") or "")
    pid = str(msg.get("parent_id") or "")
    ts  = _to_ms(msg.get("ts_ms") or msg.get("ts"))
    if not (cid and pid and ts):
        return None
    out = {
        "child_id": cid, "parent_id": pid, "ts_ms": ts,
        "venue": (msg.get("venue") or "").upper() or None,
        "typ": msg.get("typ") or msg.get("type"),
        "px": _float_or_none(msg.get("px") or msg.get("price")),
        "qty": _float_or_none(msg.get("qty")),
    }
    out["norm_hash"] = _norm_hash(out)
    return out

def normalize_fill(msg: Dict[str, Any], *, provider: Provider = Provider.UNKNOWN) -> Optional[Dict[str, Any]]:
    fid = str(msg.get("fill_id") or msg.get("order_id") or msg.get("child_id") or "")
    pid = str(msg.get("parent_id") or "")
    sym = _norm_symbol(msg.get("symbol") or "", provider)
    side = _side_str(msg.get("side"))
    px   = _clamp_pos(msg.get("price") or msg.get("px"))
    qty  = _clamp_pos(msg.get("qty"))
    ts   = _to_ms(msg.get("ts_ms") or msg.get("ts"))
    if not (fid and pid and sym and side and px and qty and ts):
        return None
    out = {
        "fill_id": fid, "parent_id": pid, "child_id": (msg.get("child_id") or None),
        "symbol": sym, "side": side, "price": float(px), "qty": float(qty),
        "ts_ms": ts, "venue": (msg.get("venue") or "").upper() or None,
        "fee": _float_or_none(msg.get("fee")),
    }
    out["norm_hash"] = _norm_hash(out)
    return out

# -------------- bridge loop --------------

RAW_TOPICS = {
    "trades":  ["raw.trades", "raw.trades.*"],
    "quotes":  ["raw.quotes", "raw.quotes.*"],
    "bars":    ["raw.bars", "raw.bars.*"],
    "news":    ["raw.news", "raw.news.*"],
    "parent":  ["raw.oms.parent"],
    "child":   ["raw.oms.child"],
    "fill":    ["raw.oms.fill"],
}

def _route_emit(kind: str, payload: Dict[str, Any]) -> None:
    if not publish_stream:
        return
    topic = {
        "trades": "ticks.trades",
        "quotes": "ticks.quotes",
        "bars":   "bars.1m" if (payload.get("tf") == "1m") else "bars.1s",
        "news":   "news.events",
        "parent": "oms.parent",
        "child":  "oms.child",
        "fill":   "oms.fill",
    }.get(kind, kind)
    publish_stream(topic, payload)

def run_bridge() -> None:
    """
    Consume `raw.*` topics, detect provider from topic name, normalize, publish.
    """
    assert consume_stream and publish_stream, "bus streams not available"
    cursors: Dict[str, str] = {k: "$" for k in RAW_TOPICS.keys()}

    def detect_provider(topic: str) -> Provider:
        t = topic.lower()
        if "nse" in t: return Provider.NSE
        if "bse" in t: return Provider.BSE
        if "binance" in t: return Provider.BINANCE
        if "moneycontrol" in t: return Provider.MC
        if "yahoo" in t: return Provider.YFIN
        if "ib" in t: return Provider.IB
        if "zerodha" in t: return Provider.ZERODHA
        return Provider.UNKNOWN

    topics_flat = []
    for v in RAW_TOPICS.values():
        topics_flat.extend(v)

    while True:
        for logical, patterns in RAW_TOPICS.items():
            for pattern in patterns:
                try:
                    for _, raw in consume_stream(pattern, start_id=cursors[logical], block_ms=200, count=500):
                        cursors[logical] = "$"
                        try:
                            msg = json.loads(raw) if isinstance(raw, str) else raw
                        except Exception:
                            continue
                        prov = detect_provider(pattern)
                        kind = logical
                        norm = None
                        if logical == "trades":
                            norm = normalize_trade(msg, provider=prov, source=pattern)
                        elif logical == "quotes":
                            norm = normalize_quote(msg, provider=prov, source=pattern)
                        elif logical == "bars":
                            norm = normalize_bar(msg, provider=prov, source=pattern)
                        elif logical == "news":
                            norm = normalize_news(msg, provider=prov, source=pattern)
                        elif logical == "parent":
                            norm = normalize_parent_order(msg, provider=prov)
                        elif logical == "child":
                            norm = normalize_child_order(msg)
                        elif logical == "fill":
                            norm = normalize_fill(msg, provider=prov)
                        if norm:
                            _route_emit(kind, norm)
                except Exception:
                    # Topic may not exist; keep looping
                    pass
        time.sleep(0.02)

# -------------- CLI / probe --------------

def _probe():
    samples = {
        "trade_binance": normalize_trade({"s":"BTCUSDT","p":"65000.5","q":"0.02","T":_utc_ms_now()}, provider=Provider.BINANCE, source="raw.trades.binance"),
        "quote_nse": normalize_quote({"symbol":"RELIANCE","bid":2900.1,"ask":2900.3,"ts":_utc_ms_now(),"venue":"NSE"}, provider=Provider.NSE, source="raw.quotes.nse"),
        "bar_yf": normalize_bar({"s":"AAPL","open":190.0,"high":191.2,"low":189.8,"close":190.7,"v":5_200_000,"ts":_utc_ms_now(),"timeframe":"1m"}, provider=Provider.YFIN, source="raw.bars.yahoo"),
        "news_mc": normalize_news({"title":"Reliance raises capex","published":_utc_ms_now(),"tickers":["RELIANCE"],"url":"https://moneycontrol.com/a"}, provider=Provider.MC, source="raw.news.moneycontrol"),
        "parent": normalize_parent_order({"order_id":"P1","symbol":"AAPL","side":"buy","qty":10000,"ts_ms":_utc_ms_now(),"strategy":"alpha.momo"}, provider=Provider.CUSTOM),
        "fill": normalize_fill({"fill_id":"F1","parent_id":"P1","symbol":"AAPL","side":"buy","price":190.06,"qty":3000,"ts_ms":_utc_ms_now(),"venue":"XNAS"}, provider=Provider.CUSTOM),
    }
    print(json.dumps(samples, indent=2, default=str))

def main():
    import argparse, time as _t
    ap = argparse.ArgumentParser(description="Unified message normalizer")
    ap.add_argument("--run", action="store_true", help="Run raw.* -> normalized bridge on the bus")
    ap.add_argument("--probe", action="store_true", help="Print example normalized payloads")
    args = ap.parse_args()
    if args.probe:
        _probe(); return
    if args.run:
        if not (consume_stream and publish_stream):
            sys.stderr.write("Bus not available. Exiting.\n")
            return
        try:
            run_bridge()
        except KeyboardInterrupt:
            pass
        return
    ap.print_help()

if __name__ == "__main__":
    main()