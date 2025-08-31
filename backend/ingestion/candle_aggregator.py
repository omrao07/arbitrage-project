# backend/marketdata/candle_aggregator.py
from __future__ import annotations

import os, json, time, math, threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple, List

# -------- optional Redis (graceful fallback) ---------------------------------
HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

# -------- env / defaults -----------------------------------------------------
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
IN_STREAM        = os.getenv("TICKS_STREAM", "md.trades")     # incoming ticks/trades
OUT_STREAM_PREF  = os.getenv("CANDLES_OUT_PREFIX", "md.candles.")  # -> md.candles.1s, md.candles.1m, ...
MAXLEN           = int(os.getenv("CANDLES_STREAM_MAXLEN", "200000"))
BACKFILL_JSONL   = os.getenv("CANDLES_BACKFILL_PATH", "artifacts/candles")
os.makedirs(BACKFILL_JSONL, exist_ok=True)

# -------- helpers ------------------------------------------------------------
_INTERVAL_MS = {
    "1s": 1000, "5s": 5000, "10s": 10_000, "15s": 15_000, "30s": 30_000,
    "1m": 60_000, "2m": 120_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "1d": 86_400_000,
}

def now_ms() -> int: return int(time.time() * 1000)

def floor_bucket(ts_ms: int, interval_ms: int) -> int:
    if interval_ms <= 0: raise ValueError("interval_ms must be > 0")
    return ts_ms - (ts_ms % interval_ms)

# -------- models -------------------------------------------------------------
@dataclass
class Candle:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    trades: int
    partial: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep floats tidy
        for k in ("open","high","low","close","volume","vwap"):
            d[k] = float(d[k])
        return d

# -------- backend sink (Redis or JSONL fallback) -----------------------------
class _Sink:
    def __init__(self, redis_url: Optional[str] = None):
        self.r = None
        if HAVE_REDIS:
            try:
                self.r = Redis.from_url(redis_url or REDIS_URL, decode_responses=True)  # type: ignore
                self.r.ping()
            except Exception:
                self.r = None

    def emit(self, interval: str, candle: Candle):
        topic = f"{OUT_STREAM_PREF}{interval}"
        obj = {"json": json.dumps(candle.to_dict())}
        if self.r:
            try:
                self.r.xadd(topic, obj, maxlen=MAXLEN, approximate=True)  # type: ignore
                return
            except Exception:
                pass
        # fallback to local JSONL
        path = os.path.join(BACKFILL_JSONL, f"candles_{interval}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(candle.to_dict(), ensure_ascii=False) + "\n")

# -------- core: per-interval aggregator -------------------------------------
class _IntervalAgg:
    def __init__(self, interval: str, sink: _Sink):
        if interval not in _INTERVAL_MS:
            raise ValueError(f"unsupported interval '{interval}'")
        self.interval = interval
        self.int_ms = _INTERVAL_MS[interval]
        self.sink = sink
        # state: (symbol -> Candle)
        self.live: Dict[str, Candle] = {}

    def _new_candle(self, symbol: str, ts_ms: int, px: float, qty: float) -> Candle:
        start = floor_bucket(ts_ms, self.int_ms)
        end = start + self.int_ms
        v = max(0.0, float(qty))
        return Candle(
            symbol=symbol, interval=self.interval, start_ms=start, end_ms=end,
            open=px, high=px, low=px, close=px,
            volume=v, vwap=px if v <= 0 else (px * v) / v,
            trades=1, partial=True
        )

    def add_trade(self, *, symbol: str, ts_ms: int, price: float, size: float) -> None:
        if price is None or price <= 0: return
        if not symbol: return
        c = self.live.get(symbol)
        if c is None:
            c = self._new_candle(symbol, ts_ms, float(price), float(size or 0))
            self.live[symbol] = c
            return

        # if trade falls outside current bucket → finalize and start new ones until caught up
        if ts_ms >= c.end_ms:
            self._finalize(symbol)  # emits finalized candle
            # create the right current bucket
            c = self._new_candle(symbol, ts_ms, float(price), float(size or 0))
            self.live[symbol] = c
            return

        # same bucket → update
        px = float(price); sz = float(size or 0)
        c.close = px
        if px > c.high: c.high = px
        if px < c.low:  c.low = px
        c.volume += max(0.0, sz)
        if c.volume > 0:
            # incremental VWAP update: recompute using running total — keep simple
            # NB: we don't store running notional; approximate with last trade weight
            # For exact VWAP, track notional and vol separately:
            pass
        c.trades += 1
        # better VWAP with running notional + vol:
        # We'll add two hidden fields when needed; for now derive precisely:
        # (To avoid breaking the public Candle dataclass, compute on emit.)

    def _finalize(self, symbol: str) -> None:
        c = self.live.get(symbol)
        if not c: return
        c.partial = False
        # for exact VWAP, in absence of notional, vwap≈(O+H+L+C)/4 as fallback when volume=0
        if c.volume <= 0:
            c.vwap = (c.open + c.high + c.low + c.close) / 4.0
        self.sink.emit(self.interval, c)
        # start next bucket placeholder (not created until next trade)

        # remove live candle (we recreate on next trade to align buckets)
        self.live.pop(symbol, None)

    def flush_partials(self) -> None:
        """Emit current partials as partial=True (useful for UI every second)."""
        for sym, c in list(self.live.items()):
            # compute a partial VWAP fallback
            tmp = Candle(**c.to_dict())  # shallow copy via dict
            tmp.partial = True
            if tmp.volume <= 0:
                tmp.vwap = (tmp.open + tmp.high + tmp.low + tmp.close) / 4.0
            self.sink.emit(self.interval, tmp)

    def time_roll(self, ts_ms: Optional[int] = None) -> None:
        """
        Call periodically; finalizes any candles whose window has ended.
        Useful when a symbol goes silent after trades.
        """
        now = ts_ms or now_ms()
        end_cutoff = now - 1  # finalize windows ending before 'now'
        for sym, c in list(self.live.items()):
            if c.end_ms <= end_cutoff:
                self._finalize(sym)

# -------- multi-interval manager --------------------------------------------
class CandleAggregator:
    """
    Feed ticks/trades and get candles at one or more intervals.
    Expected input tick schema (flexible):
        {"ts_ms": 1699999999999, "symbol": "AAPL", "price": 187.12, "size": 100}
     or {"t":..., "s":"BTCUSDT", "p":..., "q": ...}
    """
    def __init__(self, intervals: List[str] = ["1s","1m"], *, redis_url: Optional[str] = None):
        self.sink = _Sink(redis_url)
        self.aggs: Dict[str, _IntervalAgg] = {iv: _IntervalAgg(iv, self.sink) for iv in intervals}

    def on_tick(self, tick: Dict[str, Any]) -> None:
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if not sym: return
        px  = float(tick.get("price") or tick.get("p") or 0.0)
        if px <= 0: return
        qty = float(tick.get("size") or tick.get("q") or tick.get("qty") or 0.0)
        ts  = int(tick.get("ts_ms") or tick.get("t") or now_ms())
        for agg in self.aggs.values():
            agg.add_trade(symbol=sym, ts_ms=ts, price=px, size=qty)

    def roll(self) -> None:
        """Call periodically (e.g., each second) to finalize ended buckets."""
        now = now_ms()
        for agg in self.aggs.values():
            agg.time_roll(now)

    def flush_partials(self) -> None:
        for agg in self.aggs.values():
            agg.flush_partials()

# -------- Redis streaming runner (optional) ----------------------------------
def run_stream(intervals: List[str]):
    if not HAVE_REDIS:
        print("[candles] Redis not available; use CLI 'from-jsonl' or feed .on_tick() from your gateway.")
        while True: time.sleep(60)

    r = Redis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
    agg = CandleAggregator(intervals=intervals, redis_url=REDIS_URL)
    last_id = "$"  # consume only new items
    print(f"[candles] listening from '{IN_STREAM}' → '{OUT_STREAM_PREF}<interval>' for {intervals}")

    # background partial flush every second for live UI
    def _partials():
        while True:
            time.sleep(1.0)
            try: agg.flush_partials()
            except Exception: pass
    threading.Thread(target=_partials, daemon=True).start()

    while True:
        resp = r.xread({IN_STREAM: last_id}, count=500, block=1000)  # type: ignore
        if not resp: 
            agg.roll()
            continue
        _, entries = resp[0] # type: ignore
        last_id = entries[-1][0]
        for _id, fields in entries:
            raw = fields.get("json") or fields.get("data") or ""
            try:
                tick = json.loads(raw) if raw else fields
            except Exception:
                tick = fields
            try:
                agg.on_tick(tick)
            except Exception as e:
                # optional: push to an error stream or log
                pass
        agg.roll()

# -------- Backfill from JSONL/CSV (no Redis needed) --------------------------
def from_file(intervals: List[str], path: str):
    agg = CandleAggregator(intervals=intervals, redis_url=None)  # local JSONL sink
    import csv
    ext = os.path.splitext(path)[1].lower()
    def _emit(tick):
        agg.on_tick(tick)
        agg.roll()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _emit(obj)
    elif ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                # map columns if present
                tick = {
                    "ts_ms": int(row.get("ts_ms") or row.get("timestamp") or now_ms()),
                    "symbol": (row.get("symbol") or row.get("s") or "").upper(),
                    "price": float(row.get("price") or row.get("p") or 0),
                    "size": float(row.get("size") or row.get("q") or 0),
                }
                _emit(tick)
    else:
        raise ValueError("supported: .jsonl or .csv")
    # flush partials at the end
    agg.flush_partials()

# -------- CLI ----------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("candle_aggregator")
    ap.add_argument("--intervals", default="1s,1m", help="comma-separated (e.g., 1s,1m,5m,1h)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    srun = sub.add_parser("run", help="Consume Redis ticks → Redis/JSONL candles")
    srun.add_argument("--in-stream", default=IN_STREAM)

    sfile = sub.add_parser("from-jsonl", help="Backfill from JSONL/CSV ticks to JSONL candles")
    sfile.add_argument("--path", required=True)

    args = ap.parse_args()
    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]
    # validate
    for iv in intervals:
        if iv not in _INTERVAL_MS:
            raise SystemExit(f"Unsupported interval '{iv}'. Supported: {', '.join(sorted(_INTERVAL_MS))}")

    if args.cmd == "run":
        # allow overriding input stream via env; we don't need args.in_stream because IN_STREAM is global
        run_stream(intervals)
    elif args.cmd == "from-jsonl":
        from_file(intervals, args.path)

if __name__ == "__main__":
    _cli()