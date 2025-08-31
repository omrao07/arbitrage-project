# backend/backtest/replay_engine.py
from __future__ import annotations
"""
Replay Engine
-------------
Plays back historical market & news data into your strategy/OMS stack.

Features:
- Input: JSONL, CSV, or list of dicts (ticks, candles, news, events).
- Supports speed control: realtime, accelerated (xN), step-by-step.
- Sends events into bus (publish_stream) or directly into a strategy.on_tick/on_news.
- Can update Ledger with synthetic fills for backtesting.
- Tracks wallclock vs event-time skew.

Example:
    eng = ReplayEngine("data/msft_ticks.jsonl", speed=10.0)
    eng.run(lambda ev: my_strategy.on_tick(ev))
"""

import os, sys, time, json, csv
from typing import Any, Callable, Dict, List, Optional

try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        print(f"[bus:{stream}] {payload}")

try:
    from backend.ledger.ledger import Ledger # type: ignore
except Exception:
    Ledger = None  # type: ignore

class ReplayEngine:
    def __init__(self, path: str, *, fmt: Optional[str] = None, stream: str = "replay.ticks",
                 speed: float = 1.0, loop: bool = False, ledger: Optional[Ledger] = None): # type: ignore
        """
        Args:
            path: JSONL or CSV file with events (must include ts_ms field).
            fmt: 'jsonl'|'csv'|None (autodetect).
            stream: bus stream to publish into.
            speed: 1.0 = realtime; 10.0 = 10x faster; 0 = as-fast-as-possible.
            loop: replay continuously in a loop.
            ledger: optional Ledger instance; if provided, can record synthetic fills.
        """
        self.path = path
        self.fmt = fmt or ("csv" if path.endswith(".csv") else "jsonl")
        self.stream = stream
        self.speed = speed
        self.loop = loop
        self.ledger = ledger
        self._events: List[Dict[str, Any]] = []

    # ----------- load ----------
    def load(self) -> None:
        if self.fmt == "jsonl":
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    ev = json.loads(line)
                    if "ts_ms" not in ev: continue
                    self._events.append(ev)
        elif self.fmt == "csv":
            with open(self.path, newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    try:
                        row["ts_ms"] = int(row.get("ts_ms") or 0)
                        self._events.append(row)
                    except Exception: continue
        else:
            raise ValueError(f"unsupported format: {self.fmt}")
        self._events.sort(key=lambda e: int(e["ts_ms"]))

    # ----------- run ----------
    def run(self, handler: Optional[Callable[[Dict[str,Any]], None]] = None) -> None:
        if not self._events:
            self.load()
        if not self._events:
            print("no events loaded")
            return

        while True:
            prev_ts = None
            start_wall = time.time()
            base_event_ts = int(self._events[0]["ts_ms"])
            for ev in self._events:
                ev_ts = int(ev["ts_ms"])
                if prev_ts is not None and self.speed > 0:
                    dt_event = (ev_ts - prev_ts) / 1000.0
                    dt_wall = dt_event / self.speed
                    if dt_wall > 0:
                        time.sleep(min(dt_wall, 5.0))
                prev_ts = ev_ts

                # emit
                if handler:
                    handler(ev)
                else:
                    publish_stream(self.stream, ev)

                # optional: ledger synthetic fills if event marks trades
                if self.ledger and ev.get("kind") == "fill":
                    try:
                        self.ledger.record_fill(ev)
                    except Exception as e:
                        print("ledger error:", e, file=sys.stderr)

            if not self.loop:
                break

    # ----------- step mode ----------
    def step(self, n: int = 1, handler: Optional[Callable[[Dict[str,Any]], None]] = None) -> List[Dict[str,Any]]:
        if not self._events:
            self.load()
        out = []
        for i in range(min(n, len(self._events))):
            ev = self._events.pop(0)
            if handler:
                handler(ev)
            else:
                publish_stream(self.stream, ev)
            out.append(ev)
        return out

# ---------- CLI ----------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Replay historical events")
    p.add_argument("--in", dest="inp", required=True, help="Input file (jsonl/csv)")
    p.add_argument("--fmt", choices=["jsonl","csv"], default=None)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--stream", default="replay.ticks")
    args = p.parse_args()

    eng = ReplayEngine(args.inp, fmt=args.fmt, stream=args.stream, speed=args.speed, loop=args.loop)
    eng.load()
    eng.run()

if __name__ == "__main__":  # pragma: no cover
    main()