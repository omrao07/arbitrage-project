# backend/data/adapters/eurex_ws.py
# Eurex market-data adapter with NO vendor keys required.
# Modes:
#   mock   : synthetic quotes/trades for Eurex futures (default)
#   replay : read local file (.jsonl/.ndjson or .csv)
#   live   : connect to a custom WS gateway (if you have one) and normalize
#
# Output channels (stdout and optional Redis):
#   tape:quote -> {"type":"quote","sym","bid","bsz","ask","asz","ts"}
#   tape:trade -> {"type":"trade","sym","px","sz","side","ts"}
#   hb:tape    -> {"type":"hb","ts"}
#
# ENV / CLI
#   EUREX_WS_MODE=mock|replay|live
#   EUREX_WS_URL=wss://your.gateway/ws/eurex      # live
#   EUREX_WS_TOKEN=...                             # optional header for live
#   EUREX_SYMBOLS=FDAX,FESX,FGBL,FGBM,FGBS
#   EUREX_REPLAY_PATH=data/eurex_ticks.jsonl
#   REDIS_URL=redis://localhost:6379/0
#
# Examples:
#   python backend/data/adapters/eurex_ws.py --mode mock --symbols FDAX,FESX
#   python backend/data/adapters/eurex_ws.py --mode replay --replay data/eurex_ticks.csv
#   EUREX_WS_MODE=live EUREX_WS_URL=wss://host/ws/eurex python backend/data/adapters/eurex_ws.py
from __future__ import annotations
import argparse, asyncio, contextlib, csv, json, os, random, signal, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
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

# ---------- utils ----------
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

# ---------- adapter ----------
class EurexWS:
    """
    Normalizes Eurex‑style futures data into your schema.
    Live mode expects a gateway that sends either:
      {"type":"quote","sym","bid","bsz","ask","asz","ts"}
      {"type":"trade","sym","px","sz","side","ts"}
    or vendor payloads you can map in _handle_live_msg().
    """
    def __init__(self, mode: str, symbols: List[str], url: Optional[str], token: Optional[str], replay: Optional[str], hb_sec: float = 1.5):
        self.mode = mode
        self.symbols = symbols or ["FDAX","FESX","FGBL","FGBM","FGBS"]
        self.url = url
        self.token = token
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

    # ---- live (custom gateway) ----
    async def _run_live(self):
        if not websockets:
            raise RuntimeError("websockets not installed. pip install websockets")
        if not self.url:
            raise ValueError("EUREX_WS_URL required for live mode")

        backoff = 1.0
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            headers["X-API-Key"] = self.token

        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, extra_headers=headers, ping_interval=15) as ws:
                    # Let server auto-subscribe, or send a subscribe message if your gateway expects it:
                    await ws.send(json.dumps({"op":"subscribe","symbols":self.symbols}))
                    backoff = 1.0
                    async for raw in ws:
                        if self._stop.is_set(): break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        await self._handle_live_msg(msg)
            except Exception as e:
                sys.stderr.write(f"[eurex_ws] live error: {e}; retry in {backoff:.1f}s\n")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    async def _handle_live_msg(self, m: Dict[str, Any]):
        # If your gateway already sends normalized messages ("type" == quote/trade), passthrough:
        t = (m.get("type") or "").lower()
        if t == "quote":
            sym = m.get("sym") or m.get("symbol")
            bid = _f(m.get("bid")); bsz = _i(m.get("bsz"))
            ask = _f(m.get("ask")); asz = _i(m.get("asz"))
            ts = int(m.get("ts") or _ms())
            if sym and (bid is not None or ask is not None):
                self.bus.pub("tape:quote", {"type":"quote","sym":sym,"bid":bid,"bsz":bsz,"ask":ask,"asz":asz,"ts":ts})
            return
        if t == "trade":
            sym = m.get("sym") or m.get("symbol")
            px = _f(m.get("px") or m.get("price"))
            sz = _i(m.get("sz") or m.get("size"))
            side = (m.get("side") or "").lower() or None
            ts = int(m.get("ts") or _ms())
            if sym and px is not None:
                self.bus.pub("tape:trade", {"type":"trade","sym":sym,"px":px,"sz":sz,"side":side,"ts":ts})
            return

        # Otherwise map from vendor-style payloads here (add cases as needed)
        # Example hypothetical:
        # if "bids" in m or "asks" in m: ... update top-of-book and emit quote
        # elif "last" in m and "qty" in m: ... emit trade

    # ---- replay ----
    async def _run_replay(self):
        if not self.replay:
            raise ValueError("--replay path required for replay mode")
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
        if typ == "quote":
            out = {"type":"quote","sym":m.get("sym"),"bid":_f(m.get("bid")),"bsz":_i(m.get("bsz")),"ask":_f(m.get("ask")),"asz":_i(m.get("asz")),"ts":ts}
            if out["sym"] and (out["bid"] is not None or out["ask"] is not None): self.bus.pub("tape:quote", out)
        elif typ == "trade":
            out = {"type":"trade","sym":m.get("sym"),"px":_f(m.get("px") or m.get("price")),"sz":_i(m.get("sz") or m.get("size")),"side":(m.get("side") or "").lower() or None,"ts":ts}
            if out["sym"] and out["px"] is not None: self.bus.pub("tape:trade", out)
        else:
            # best-effort coercion
            if "price" in m and "size" in m:
                self.bus.pub("tape:trade", {"type":"trade","sym":m.get("sym"),"px":_f(m["price"]),"sz":_i(m["size"]),"side":(m.get("side") or "").lower() or None,"ts":ts})

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
        # Reasonable starting prices for Eurex futures (purely illustrative)
        seeds = {
            "FDAX": 18000.0,  # DAX
            "FESX": 5000.0,   # Euro Stoxx 50
            "FGBL": 134.00,   # Bund
            "FGBM": 118.50,   # Bobl
            "FGBS": 106.00,   # Schatz
        }
        prices = {s: (seeds.get(s) or 100.0) * (1 + random.random()*0.01) for s in self.symbols}
        base_size = {s: 50 for s in self.symbols}  # arbitrary

        while not self._stop.is_set():
            now = _ms()
            for s in self.symbols:
                # small Gaussian walk ~3 bps
                drift_bps = random.gauss(0, 3.0)
                prices[s] = max(0.01, prices[s] * (1 + drift_bps/10000.0))

                # tighter spreads for rates than equity index
                spr_bps = 0.8 if s.startswith("FG") else 1.2
                spread = max(0.001, prices[s] * (spr_bps/10000.0))
                bid = round(prices[s] - spread/2, 2 if s.startswith("F") and not s.startswith("FG") else 3)
                ask = round(prices[s] + spread/2, 2 if s.startswith("F") and not s.startswith("FG") else 3)
                bsz = max(1, int(abs(random.gauss(base_size[s], base_size[s]*0.25))))
                asz = max(1, int(abs(random.gauss(base_size[s], base_size[s]*0.25))))

                self.bus.pub("tape:quote", {"type":"quote","sym":s,"bid":bid,"bsz":bsz,"ask":ask,"asz":asz,"ts":now})

                # occasional trade at inside
                if random.random() < 0.30:
                    side = "buy" if random.random() > 0.5 else "sell"
                    px = ask if side == "buy" else bid
                    sz = max(1, int(abs(random.gauss(base_size[s]*0.6, base_size[s]*0.2))))
                    self.bus.pub("tape:trade", {"type":"trade","sym":s,"px":px,"sz":sz,"side":side,"ts":now})

            await asyncio.sleep(0.12)

# ---------- CLI ----------
def _args(argv=None):
    p = argparse.ArgumentParser(description="Eurex WS adapter (mock/replay/live gateway; no vendor keys)")
    p.add_argument("--mode", choices=["mock","replay","live"], default=os.getenv("EUREX_WS_MODE","mock"))
    p.add_argument("--symbols", default=os.getenv("EUREX_SYMBOLS","FDAX,FESX,FGBL,FGBM,FGBS"))
    p.add_argument("--url", default=os.getenv("EUREX_WS_URL"))
    p.add_argument("--token", default=os.getenv("EUREX_WS_TOKEN"))
    p.add_argument("--replay", default=os.getenv("EUREX_REPLAY_PATH"))
    return p.parse_args(argv)

def main(argv=None):
    ns = _args(argv)
    syms = [s for s in (ns.symbols or "").split(",") if s]
    app = EurexWS(mode=ns.mode, symbols=syms, url=ns.url, token=ns.token, replay=ns.replay)
    asyncio.run(app.run())

if __name__ == "__main__":
    main()