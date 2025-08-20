# backend/data/adapters/coinbase_ws.py
# Coinbase public market-data adapter (NO API keys required).
# Modes:
#   live   : public WS (ticker + level2 top-of-book)
#   replay : stream from local jsonl/csv
#   mock   : synthetic feed
#
# Normalized output everywhere:
#   quote -> {"type":"quote","sym","bid","bsz","ask","asz","ts"}
#   trade -> {"type":"trade","sym","px","sz","side","ts"}   # side: "buy"|"sell" (best-effort)
#   hb    -> {"type":"hb","ts"}
#
# ENV / CLI:
#   COINBASE_WS_MODE=live|replay|mock
#   COINBASE_WS_URL=wss://...     # overrides endpoint
#   COINBASE_SYMBOLS=BTC-USD,ETH-USD
#   COINBASE_REPLAY_PATH=path.jsonl|.csv
#   REDIS_URL=redis://localhost:6379/0
#
# Examples:
#   python backend/data/adapters/coinbase_ws.py --mode live --symbols BTC-USD,ETH-USD
#   python backend/data/adapters/coinbase_ws.py --mode replay --replay data/cb_ticks.jsonl
#   python backend/data/adapters/coinbase_ws.py --mode mock --symbols BTC-USD

from __future__ import annotations
import argparse, asyncio, contextlib, csv, json, os, random, signal, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- optional deps -----
try:
    import websockets  # type: ignore
except Exception:
    websockets = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis = None
try:
    import redis  # type: ignore
    _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    _redis = None

# ----- utils -----
def _ms() -> int: return int(time.time() * 1000)
def _json(d: Dict[str, Any]) -> str: return json.dumps(d, separators=(",", ":"), ensure_ascii=False)
def _f(x):
    try:
        if x in (None, ""): return None
        return float(x)
    except Exception: return None
def _i(x):
    try:
        if x in (None, ""): return None
        return int(float(x))
    except Exception: return None

@dataclass
class Bus:
    def pub(self, ch: str, payload: Dict[str, Any]) -> None:
        sys.stdout.write(f"[{ch}] {_json(payload)}\n"); sys.stdout.flush()
        if _redis:
            try: _redis.publish(ch, _json(payload))
            except Exception: pass

# ======= Adapter =======
class CoinbaseWS:
    """
    Live mode uses Coinbase public WS.
    - We subscribe to "ticker" for trades and "level2" for book (snapshot + deltas).
    - We maintain top-of-book per symbol and emit normalized quotes.
    """
    def __init__(self, mode: str, symbols: List[str], url: Optional[str], replay: Optional[str], hb_sec: float = 1.5):
        self.mode = mode
        self.symbols = symbols or ["BTC-USD"]
        self.url = url
        self.replay = replay
        self.hb_sec = hb_sec
        self.bus = Bus()
        self._stop = asyncio.Event()
        self._top: Dict[str, Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]] = {}

    # ---- run loop ----
    async def run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._stop.set)

        hb = asyncio.create_task(self._heartbeat())
        try:
            if self.mode == "live":
                await self._run_live()
            elif self.mode == "replay":
                await self._run_replay()
            else:
                await self._run_mock()
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

    async def _heartbeat(self):
        while not self._stop.is_set():
            self.bus.pub("hb:tape", {"type": "hb", "ts": _ms()})
            await asyncio.sleep(self.hb_sec)

    # ---- live mode ----
    def _default_urls(self) -> List[str]:
        # Prefer Advanced Trade WS if available; fall back to legacy Coinbase Exchange feed.
        # You can override with COINBASE_WS_URL.
        return [
            "wss://ws-feed.exchange.coinbase.com",           # legacy public (still widely available)
            "wss://advanced-trade-ws.coinbase.com"           # new endpoint for Advanced Trade
        ]

    async def _run_live(self):
        if not websockets:
            raise RuntimeError("websockets not installed. pip install websockets")

        urls = [self.url] if self.url else self._default_urls()
        topics = {
            "type": "subscribe",
            "product_ids": self.symbols,
            "channels": ["ticker", "level2"]
        }

        backoff = 1.0
        while not self._stop.is_set():
            for url in urls:
                try:
                    async with websockets.connect(url, ping_interval=15) as ws:
                        await ws.send(json.dumps(topics))
                        backoff = 1.0
                        async for raw in ws:
                            if self._stop.is_set(): break
                            try:
                                msg = json.loads(raw)
                            except Exception:
                                continue
                            await self._handle_live_msg(msg)
                except Exception as e:
                    sys.stderr.write(f"[coinbase_ws] WS error {e}; reconnecting in {backoff:.1f}s\n")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 15.0)

    async def _handle_live_msg(self, m: Dict[str, Any]):
        t = m.get("type")
        now = _ms()

        # Ticker → trade prints with side if provided
        if t == "ticker":
            sym = m.get("product_id")
            px = _f(m.get("price"))
            # size sometimes 'last_size'; trade_id not needed
            sz = _f(m.get("last_size") or m.get("size"))
            side = (m.get("side") or "").lower() or None
            ts = _i(m.get("time"))  # ISO timestamp usually provided; fallback to now
            if ts is None:
                ts = now
            if sym and px is not None:
                self.bus.pub("tape:trade", {"type": "trade", "sym": sym, "px": px, "sz": sz, "side": side, "ts": ts})
            return

        # Level2 snapshot/delta → maintain top‑of‑book
        if t == "snapshot":
            sym = m.get("product_id")
            bids = m.get("bids") or []
            asks = m.get("asks") or []
            bid = _f(bids[0][0]) if bids else None
            bsz = _i(bids[0][1]) if bids else None
            ask = _f(asks[0][0]) if asks else None
            asz = _i(asks[0][1]) if asks else None
            self._top[sym] = (bid, bsz, ask, asz)
            if sym and (bid is not None or ask is not None):
                self.bus.pub("tape:quote", {"type":"quote","sym":sym,"bid":bid,"bsz":bsz,"ask":ask,"asz":asz,"ts":now})
            return

        if t == "l2update":
            sym = m.get("product_id")
            changes = m.get("changes") or []
            bid, bsz, ask, asz = self._top.get(sym, (None, None, None, None))

            # changes example: [["buy","10101.10","0.45054140"], ["sell","10102.55","0"]]
            for side, price, size in changes:
                p = _f(price)
                s = _f(size)
                if side == "buy":
                    if s == 0 or s is None:
                        if bid == p:  # invalidate if top removed
                            bid, bsz = None, None
                    else:
                        if bid is None or (p is not None and p > bid):
                            bid, bsz = p, _i(s)
                else:  # sell
                    if s == 0 or s is None:
                        if ask == p:
                            ask, asz = None, None
                    else:
                        if ask is None or (p is not None and p < ask):
                            ask, asz = p, _i(s)

            self._top[sym] = (bid, bsz, ask, asz)
            if sym and (bid is not None or ask is not None):
                self.bus.pub("tape:quote", {"type":"quote","sym":sym,"bid":bid,"bsz":bsz,"ask":ask,"asz":asz,"ts":_ms()})
            return
        # other message types ignored

    # ---- replay ----
    async def _run_replay(self):
        if not self.replay:
            raise ValueError("--replay path required")
        p = Path(self.replay)
        if not p.exists(): raise FileNotFoundError(str(p))

        if p.suffix.lower() in (".jsonl", ".ndjson"):
            async for rec in self._read_jsonl(p):
                self._emit_normalized(rec)
                if self._stop.is_set(): break
                await asyncio.sleep(0.02)
        else:
            async for rec in self._read_csv(p):
                self._emit_normalized(rec)
                if self._stop.is_set(): break
                await asyncio.sleep(0.02)

    def _emit_normalized(self, m: Dict[str, Any]):
        typ = (m.get("type") or "").lower()
        ts = int(m.get("ts") or _ms())
        if typ == "trade":
            out = {"type":"trade","sym":m.get("sym"),"px":_f(m.get("px") or m.get("price")),"sz":_f(m.get("sz") or m.get("size")),"side":(m.get("side") or "").lower() or None,"ts":ts}
            if out["sym"] and out["px"] is not None: self.bus.pub("tape:trade", out)
        elif typ == "quote":
            out = {"type":"quote","sym":m.get("sym"),"bid":_f(m.get("bid")),"bsz":_i(m.get("bsz")),"ask":_f(m.get("ask")),"asz":_i(m.get("asz")),"ts":ts}
            if out["sym"] and (out["bid"] is not None or out["ask"] is not None): self.bus.pub("tape:quote", out)
        else:
            # coerce generic rows
            if "price" in m and "size" in m:
                self.bus.pub("tape:trade", {"type":"trade","sym":m.get("sym"),"px":_f(m["price"]),"sz":_f(m["size"]),"side":(m.get("side") or "").lower() or None,"ts":ts})

    async def _read_jsonl(self, p: Path):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if self._stop.is_set(): break
                s = line.strip()
                if not s: continue
                try: yield json.loads(s)
                except Exception: continue

    async def _read_csv(self, p: Path):
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if self._stop.is_set(): break
                yield row

    # ---- mock ----
    async def _run_mock(self):
        px = {s: 25000.0 + random.random()*1000 for s in self.symbols}
        base = {s: 0.8 for s in self.symbols}
        while not self._stop.is_set():
            now = _ms()
            for s in self.symbols:
                drift = random.gauss(0, 5)  # ~5 bps std
                px[s] = max(1.0, px[s] * (1 + drift/10000.0))
                spr = max(0.01, px[s] * 0.0007)
                bid = round(px[s] - spr/2, 2)
                ask = round(px[s] + spr/2, 2)
                bsz = max(1, int(abs(random.gauss(base[s]*80, 12))))
                asz = max(1, int(abs(random.gauss(base[s]*80, 12))))
                self.bus.pub("tape:quote", {"type":"quote","sym":s,"bid":bid,"bsz":bsz,"ask":ask,"asz":asz,"ts":now})
                if random.random() < 0.33:
                    side = "buy" if random.random() > 0.5 else "sell"
                    px_tr = ask if side == "buy" else bid
                    sz = max(1, int(abs(random.gauss(base[s]*20, 6))))
                    self.bus.pub("tape:trade", {"type":"trade","sym":s,"px":px_tr,"sz":sz,"side":side,"ts":now})
            await asyncio.sleep(0.1)

# ----- CLI -----
def _args(argv=None):
    p = argparse.ArgumentParser(description="Coinbase public WS / replay / mock (no API keys)")
    p.add_argument("--mode", choices=["live","replay","mock"], default=os.getenv("COINBASE_WS_MODE","live"))
    p.add_argument("--symbols", default=os.getenv("COINBASE_SYMBOLS","BTC-USD"))
    p.add_argument("--url", default=os.getenv("COINBASE_WS_URL"))
    p.add_argument("--replay", default=os.getenv("COINBASE_REPLAY_PATH"))
    return p.parse_args(argv)

def main(argv=None):
    ns = _args(argv)
    syms = [s for s in (ns.symbols or "").split(",") if s]
    app = CoinbaseWS(mode=ns.mode, symbols=syms, url=ns.url, replay=ns.replay)
    asyncio.run(app.run())

if __name__ == "__main__":
    main()