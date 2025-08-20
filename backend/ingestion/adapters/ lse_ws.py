# backend/data/adapters/lse_ws.py
# Self-contained LSE "feed" with NO API dependency.
# Modes:
#   - mock   : synthetic quotes/trades for given symbols
#   - replay : stream from local .jsonl/.ndjson or .csv (see columns below)
#
# Output (always normalized):
#   - stdout lines (JSON) and, if redis-py is present, publishes to:
#       tape:quote, tape:trade, hb:tape
#
# CSV expected columns (any subset; missing fields are skipped):
#   type (quote|trade), sym, ts, bid, bsz, ask, asz, px, sz, side
#
# Examples
# --------
# Mock (no deps, no keys):
#   python backend/data/adapters/lse_ws.py --mode mock --symbols VOD.L,BARC.L,HSBA.L
#
# Replay from file:
#   python backend/data/adapters/lse_ws.py --mode replay --replay data/lse_ticks.jsonl
#
# With Redis publish (optional, only if redis is installed):
#   REDIS_URL=redis://localhost:6379/0 python backend/data/adapters/lse_ws.py --mode mock

from __future__ import annotations
import argparse
import asyncio
import contextlib
import csv
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Optional Redis support (will be skipped if not installed)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client = None
try:
    import redis  # type: ignore
    _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    _redis_client = None


def _ms() -> int:
    return int(time.time() * 1000)


def _json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


@dataclass
class OutBus:
    """Dual sink: stdout (always) + optional Redis publish if available."""
    redis_channel_prefix: str = ""

    def publish(self, channel: str, payload: Dict[str, Any]) -> None:
        # stdout
        sys.stdout.write(f"[{channel}] {_json(payload)}\n")
        sys.stdout.flush()
        # optional redis
        if _redis_client:
            try:
                _redis_client.publish(channel, _json(payload))
            except Exception:
                pass


class LSEFeed:
    def __init__(self, mode: str, symbols: List[str], replay: Optional[str] = None, hb_interval: float = 1.5):
        self.mode = mode
        self.symbols = symbols or ["VOD.L", "BARC.L", "HSBA.L"]
        self.replay = replay
        self.hb_interval = hb_interval
        self.bus = OutBus()
        self._stop = asyncio.Event()

    async def run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._stop.set)

        hb = asyncio.create_task(self._heartbeat())

        try:
            if self.mode == "replay":
                await self._run_replay()
            else:
                await self._run_mock()
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

    async def _heartbeat(self):
        while not self._stop.is_set():
            self.bus.publish("hb:tape", {"type": "hb", "ts": _ms()})
            await asyncio.sleep(self.hb_interval)

    # ---------------- MOCK MODE ----------------
    async def _run_mock(self):
        # Start prices & sizes
        px = {s: 100.0 + random.random() * 20 for s in self.symbols}
        base_size = {s: 1200 for s in self.symbols}
        quote_every = 0.12  # seconds
        trade_prob = 0.28

        while not self._stop.is_set():
            now = _ms()
            for s in self.symbols:
                # small random walk
                drift_bps = random.gauss(0, 2.0)  # ~2 bps std
                px[s] = max(0.01, px[s] * (1 + drift_bps / 10000.0))

                spread = max(0.001, px[s] * 0.0008)  # ~8 bps spread
                bid = round(px[s] - spread / 2, 4)
                ask = round(px[s] + spread / 2, 4)
                bsz = max(1, base_size[s] + random.randint(-150, 150))
                asz = max(1, base_size[s] + random.randint(-150, 150))

                # quote
                self.bus.publish(
                    "tape:quote",
                    {"type": "quote", "sym": s, "bid": bid, "bsz": bsz, "ask": ask, "asz": asz, "ts": now},
                )

                # occasional trade at bid/ask
                if random.random() < trade_prob:
                    side = "buy" if random.random() > 0.5 else "sell"
                    trade_px = ask if side == "buy" else bid
                    size = max(1, int(abs(random.gauss(300, 140))))
                    self.bus.publish(
                        "tape:trade",
                        {"type": "trade", "sym": s, "px": trade_px, "sz": size, "side": side, "ts": now},
                    )

            await asyncio.sleep(quote_every)

    # ---------------- REPLAY MODE ----------------
    async def _run_replay(self):
        if not self.replay:
            raise ValueError("--replay path is required for replay mode")

        path = Path(self.replay)
        if not path.exists():
            raise FileNotFoundError(str(path))

        # Decide format by extension
        if path.suffix.lower() in (".jsonl", ".ndjson"):
            async for rec in self._read_jsonl(path):
                if self._stop.is_set():
                    break
                self._publish_normalized(rec)
                await asyncio.sleep(0.02)  # ~50 Hz
        else:
            async for rec in self._read_csv(path):
                if self._stop.is_set():
                    break
                self._publish_normalized(rec)
                await asyncio.sleep(0.02)

    def _publish_normalized(self, m: Dict[str, Any]):
        t = (m.get("type") or "").lower()
        now = _ms()
        if t == "quote":
            out = {
                "type": "quote",
                "sym": m.get("sym") or m.get("symbol"),
                "bid": _safe_float(m.get("bid")),
                "bsz": _safe_int(m.get("bsz") or m.get("bidSize")),
                "ask": _safe_float(m.get("ask")),
                "asz": _safe_int(m.get("asz") or m.get("askSize")),
                "ts": int(m.get("ts") or now),
            }
            if out["sym"] and (out["bid"] is not None or out["ask"] is not None):
                self.bus.publish("tape:quote", out)
        elif t == "trade":
            out = {
                "type": "trade",
                "sym": m.get("sym") or m.get("symbol"),
                "px": _safe_float(m.get("px") or m.get("price")),
                "sz": _safe_int(m.get("sz") or m.get("size")),
                "side": (m.get("side") or "").lower() or None,
                "ts": int(m.get("ts") or now),
            }
            if out["sym"] and out["px"] is not None:
                self.bus.publish("tape:trade", out)
        else:
            # Try to coerce unknown rows
            if "price" in m and "size" in m:
                self.bus.publish(
                    "tape:trade",
                    {
                        "type": "trade",
                        "sym": m.get("sym") or m.get("symbol"),
                        "px": _safe_float(m.get("price")),
                        "sz": _safe_int(m.get("size")),
                        "side": (m.get("side") or "").lower() or None,
                        "ts": int(m.get("ts") or now),
                    },
                )

    async def _read_jsonl(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    async def _read_csv(self, path: Path):
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self._stop.is_set():
                    break
                yield row


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="LSE feed (mock/replay, no API)")
    p.add_argument("--mode", choices=["mock", "replay"], default=os.getenv("LSE_MODE", "mock"))
    p.add_argument("--symbols", default=os.getenv("LSE_SYMBOLS", "VOD.L,BARC.L,HSBA.L"))
    p.add_argument("--replay", default=os.getenv("LSE_REPLAY_PATH"))
    return p.parse_args(argv)


def main(argv=None):
    ns = _parse_args(argv)
    symbols = [s for s in (ns.symbols or "").split(",") if s]

    feed = LSEFeed(mode=ns.mode, symbols=symbols, replay=ns.replay)
    asyncio.run(feed.run())


if __name__ == "__main__":
    main()