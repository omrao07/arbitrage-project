# backend/data/adapters/bybit_ws.py
# Bybit public market-data adapter (NO API keys required).
# Modes:
#   live   : connect to Bybit public WS and emit normalized quotes/trades
#   replay : stream ticks from local file (jsonl/csv)
#   mock   : synthetic feed (random walk)
#
# Output (normalized everywhere):
#   quote -> {"type":"quote","sym","bid","bsz","ask","asz","ts"}
#   trade -> {"type":"trade","sym","px","sz","side","ts"}    # side: "buy"|"sell"
#   hb    -> {"type":"hb","ts"}
#
# Env / CLI:
#   BYBIT_WS_MODE=live|replay|mock
#   BYBIT_WS_MARKET=spot|linear|inverse   (default: spot)
#   BYBIT_WS_URL=wss://...                (override endpoint)
#   BYBIT_SYMBOLS=BTCUSDT,ETHUSDT
#   BYBIT_REPLAY_PATH=path/to/file.jsonl
#   REDIS_URL=redis://localhost:6379/0
#
# Examples:
#   python backend/data/adapters/bybit_ws.py --mode live --market spot --symbols BTCUSDT,ETHUSDT
#   python backend/data/adapters/bybit_ws.py --mode replay --replay data/bybit_ticks.jsonl
#   python backend/data/adapters/bybit_ws.py --mode mock --symbols BTCUSDT
#
from __future__ import annotations
import argparse, asyncio, contextlib, csv, json, os, random, signal, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------- optional deps ----------
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

# -------- utilities ----------
def _ms() -> int: return int(time.time() * 1000)
def _json(x: Dict[str, Any]) -> str: return json.dumps(x, separators=(",", ":"), ensure_ascii=False)
def _f(x): 
    try:
        if x is None or x == "": return None
        return float(x)
    except Exception: return None
def _i(x):
    try:
        if x is None or x == "": return None
        return int(float(x))
    except Exception: return None

@dataclass
class Bus:
    def pub(self, ch: str, payload: Dict[str, Any]) -> None:
        line = f"[{ch}] {_json(payload)}\n"
        sys.stdout.write(line); sys.stdout.flush()
        if _redis:
            try: _redis.publish(ch, _json(payload))
            except Exception: pass

# -------- adapter ----------
class BybitWS:
    def __init__(self, mode: str, market: str, symbols: List[str], url: Optional[str], replay: Optional[str], hb_sec: float = 1.5):
        self.mode = mode
        self.market = market  # spot|linear|inverse
        self.symbols = symbols or ["BTCUSDT"]
        self.url = url
        self.replay = replay
        self.hb_sec = hb_sec
        self.bus = Bus()
        self._stop = asyncio.Event()

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

    # ----- LIVE (public WS, no keys) -----
    def _default_ws_url(self) -> str:
        # Bybit v5 public endpoints (non-auth)
        # refs: https://bybit-exchange.github.io/docs/v5/ws/connect
        if self.market == "spot":
            return "wss://stream.bybit.com/v5/public/spot"
        if self.market == "linear":
            return "wss://stream.bybit.com/v5/public/linear"
        if self.market == "inverse":
            return "wss://stream.bybit.com/v5/public/inverse"
        return "wss://stream.bybit.com/v5/public/spot"

    def _topics(self) -> List[str]:
        # subscribe to best quotes (orderbook.1) and trades
        # topic formats: orderbook.1.<symbol>, publicTrade.<symbol>
        t = []
        for s in self.symbols:
            t.append(f"orderbook.1.{s}")
            t.append(f"publicTrade.{s}")
        return t

    async def _run_live(self):
        if not websockets:
            raise RuntimeError("websockets not installed. pip install websockets")
        url = self.url or self._default_ws_url()
        topics = self._topics()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=15) as ws:
                    # subscribe
                    sub = {"op": "subscribe", "args": topics}
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    async for raw in ws:
                        if self._stop.is_set(): break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        await self._handle_live_msg(msg)
            except Exception as e:
                sys.stderr.write(f"[bybit_ws] reconnect in {backoff:.1f}s (err: {e})\n")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    async def _handle_live_msg(self, msg: Dict[str, Any]):
        # Trade
        if msg.get("topic", "").startswith("publicTrade."):
            symbol = msg.get("topic", "").split(".")[-1]
            for t in msg.get("data", []) or []:
                # sample t: {"T":167230...,"s":"BTCUSDT","S":"Buy","v":"0.001","p":"16850.50"}
                px = _f(t.get("p") or t.get("price"))
                sz = _f(t.get("v") or t.get("q") or t.get("size"))
                side = (t.get("S") or t.get("side") or "").lower() or None
                ts = int(t.get("T") or t.get("ts") or _ms())
                if px is not None and sz is not None:
                    self.bus.pub("tape:trade", {"type": "trade", "sym": symbol, "px": px, "sz": sz, "side": side, "ts": ts})
            return

        # Quote (best bid/ask from orderbook.1)
        if msg.get("topic", "").startswith("orderbook.1."):
            symbol = msg.get("topic", "").split(".")[-1]
            data = msg.get("data") or {}
            # data may be snapshot with "b" (bids) / "a" (asks), or delta; we only need top 1
            bids = data.get("b") or data.get("bid") or []
            asks = data.get("a") or data.get("ask") or []
            def _top(levels):
                # bybit sends arrays like ["price","size"] or dicts
                if not levels: return (None, None)
                lv = levels[0]
                if isinstance(lv, dict):
                    return (_f(lv.get("price")), _i(lv.get("size")))
                if isinstance(lv, list) and len(lv) >= 2:
                    return (_f(lv[0]), _i(lv[1]))
                return (None, None)
            bid, bsz = _top(bids)
            ask, asz = _top(asks)
            ts = int(data.get("ts") or _ms())
            if bid is not None or ask is not None:
                self.bus.pub("tape:quote", {"type":"quote","sym":symbol,"bid":bid,"bsz":bsz,"ask":ask,"asz":asz,"ts":ts})
            return

        # Pings / info messages ignored

    # ----- REPLAY -----
    async def _run_replay(self):
        if not self.replay:
            raise ValueError("--replay path required in replay mode")
        path = Path(self.replay)
        if not path.exists(): raise FileNotFoundError(str(path))

        if path.suffix.lower() in (".jsonl", ".ndjson"):
            async for rec in self._read_jsonl(path):
                self._emit_normalized(rec)
                if self._stop.is_set(): break
                await asyncio.sleep(0.02)
        else:
            async for rec in self._read_csv(path):
                self._emit_normalized(rec)
                if self._stop.is_set(): break
                await asyncio.sleep(0.02)

    def _emit_normalized(self, m: Dict[str, Any]):
        t = (m.get("type") or "").lower()
        ts = int(m.get("ts") or _ms())
        if t == "trade":
            out = {"type":"trade","sym":m.get("sym"),"px":_f(m.get("px") or m.get("price")),"sz":_f(m.get("sz") or m.get("size")),"side":(m.get("side") or "").lower() or None,"ts":ts}
            if out["sym"] and out["px"] is not None: self.bus.pub("tape:trade", out)
        elif t == "quote":
            out = {"type":"quote","sym":m.get("sym"),"bid":_f(m.get("bid")),"bsz":_i(m.get("bsz")),"ask":_f(m.get("ask")),"asz":_i(m.get("asz")),"ts":ts}
            if out["sym"] and (out["bid"] is not None or out["ask"] is not None): self.bus.pub("tape:quote", out)
        else:
            # try coerce
            if "price" in m and "size" in m:
                self.bus.pub("tape:trade", {"type":"trade","sym":m.get("sym"),"px":_f(m["price"]),"sz":_f(m["size"]),"side":(m.get("side") or "").lower() or None,"ts":ts})

    async def _read_jsonl(self, p: Path):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if self._stop.is_set(): break
                line = line.strip()
                if not line: continue
                try: yield json.loads(line)
                except Exception: continue

    async def _read_csv(self, p: Path):
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if self._stop.is_set(): break
                yield row

    # ----- MOCK -----
    async def _run_mock(self):
        prices = {s: 25000.0 + random.random()*1000 for s in self.symbols}
        base = {s: 1.5 for s in self.symbols}  # lot size approx
        while not self._stop.is_set():
            now = _ms()
            for s in self.symbols:
                drift = random.gauss(0, 6)  # ~6 bps std
                prices[s] = max(1.0, prices[s] * (1 + drift/10000.0))
                spread = max(0.01, prices[s] * 0.0006)
                bid = round(prices[s] - spread/2, 2)
                ask = round(prices[s] + spread/2, 2)
                bsz = max(1, int(abs(random.gauss(base[s]*50, 10))))
                asz = max(1, int(abs(random.gauss(base[s]*50, 10))))
                self.bus.pub("tape:quote", {"type":"quote","sym":s,"bid":bid,"bsz":bsz,"ask":ask,"asz":asz,"ts":now})
                if random.random() < 0.35:
                    side = "buy" if random.random() > 0.5 else "sell"
                    px = ask if side == "buy" else bid
                    sz = max(1, int(abs(random.gauss(base[s]*10, 5))))
                    self.bus.pub("tape:trade", {"type":"trade","sym":s,"px":px,"sz":sz,"side":side,"ts":now})
            await asyncio.sleep(0.1)

# -------- CLI ----------
def _args(argv=None):
    p = argparse.ArgumentParser(description="Bybit public WS / replay / mock (no API keys)")
    p.add_argument("--mode", choices=["live","replay","mock"], default=os.getenv("BYBIT_WS_MODE","live"))
    p.add_argument("--market", choices=["spot","linear","inverse"], default=os.getenv("BYBIT_WS_MARKET","spot"))
    p.add_argument("--symbols", default=os.getenv("BYBIT_SYMBOLS","BTCUSDT"))
    p.add_argument("--url", default=os.getenv("BYBIT_WS_URL"))
    p.add_argument("--replay", default=os.getenv("BYBIT_REPLAY_PATH"))
    return p.parse_args(argv)

def main(argv=None):
    ns = _args(argv)
    symbols = [s for s in (ns.symbols or "").split(",") if s]
    app = BybitWS(mode=ns.mode, market=ns.market, symbols=symbols, url=ns.url, replay=ns.replay)
    asyncio.run(app.run())

if __name__ == "__main__":
    main()