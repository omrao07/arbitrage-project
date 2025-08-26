# backend/engine/engine.py
"""
Unified Engine Orchestrator
---------------------------
Purpose
- Load & manage multiple strategies
- Subscribe to bus streams (ticks/news/others) and dispatch to strategies
- Provide lightweight health + metrics + graceful shutdown

Assumptions
- Redis-backed bus helpers exist in backend.bus.streams:
    consume_stream(stream, start_id="$", block_ms=1000, count=200)
    publish_stream(stream, payload)
    hset(hkey, field, value)  # value may be dict; your helper serializes

- Strategies subclass backend.engine.strategy_base.Strategy and implement:
    on_tick(tick: dict) -> None
    (optional) on_news(event: dict) -> None
    on_start(), on_stop() (optional)
  You can also register any object that simply has those methods.

CLI
  python -m backend.engine.engine --probe
  python -m backend.engine.engine --run --ticks trades.crypto,md.equities --news news.moneycontrol,news.yahoo
  python -m backend.engine.engine --run --registry config/registry.yaml

registry.yaml (example)
  strategies:
    - module: backend.engine.my_strategies.momo
      class:  MomentumUS
      name:   momo_us
      region: US
      params: { lookback: 50, threshold_bps: 15 }
    - module: backend.engine.strategy_base
      class:  ExampleBuyTheDip
      name:   dipper_crypto
      region: CRYPTO
      params: { default_qty: 0.002, bps: 8 }

Env
  REDIS_HOST=localhost REDIS_PORT=6379
  ENGINE_TICK_STREAMS="trades.crypto,md.stocks"
  ENGINE_NEWS_STREAMS="news.moneycontrol,news.yahoo"
"""

from __future__ import annotations

import importlib
import json
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---------- optional deps ----------
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # type: ignore

try:
    from backend.bus.streams import consume_stream, publish_stream, hset
except Exception as e:
    raise RuntimeError("backend.bus.streams not found; ensure your bus helpers are in place") from e

# Strategy base (for typing / defaults; not strictly required)
try:
    from backend.engine.strategy_base import Strategy
except Exception:
    Strategy = object  # type: ignore

# ---------- config & defaults ----------
DEFAULT_TICK_STREAMS = os.getenv("ENGINE_TICK_STREAMS", "trades.crypto").split(",")
DEFAULT_NEWS_STREAMS = os.getenv("ENGINE_NEWS_STREAMS", "news.moneycontrol").split(",")
HEALTH_KEY = "engine:health"
METRIC_KEY = "engine:metrics"

@dataclass
class StratSpec:
    module: str
    class_name: str
    name: str
    region: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EngineConfig:
    tick_streams: Sequence[str] = tuple(s for s in DEFAULT_TICK_STREAMS if s.strip())
    news_streams: Sequence[str] = tuple(s for s in DEFAULT_NEWS_STREAMS if s.strip())
    registry_path: Optional[str] = None
    queue_max: int = 10000
    tick_workers: int = 2
    news_workers: int = 1
    log_every_s: int = 10

# ---------- small helpers ----------
def _utc_ms() -> int:
    return int(time.time() * 1000)

def _safe_json(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}

def _load_registry(path: str) -> List[StratSpec]:
    if not yaml:
        raise RuntimeError("pyyaml not installed; cannot read registry.yaml")
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    specs: List[StratSpec] = []
    for s in (doc.get("strategies") or []):
        specs.append(StratSpec(
            module = s.get("module"),
            class_name = s.get("class"),
            name = s.get("name"),
            region = s.get("region"),
            params = s.get("params") or {}
        ))
    return specs

def _construct_strategy(spec: StratSpec) -> Any:
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.class_name)
    # common ctor signature; if custom, kwargs should handle it
    obj = cls(name=spec.name, region=spec.region, **spec.params)
    return obj

# ---------- Engine ----------
class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.strats: List[Any] = []
        self._running = False

        # event queues (producer=bus threads, consumer=worker pool)
        self.q_ticks: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_max)
        self.q_news: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_max)

        # metrics (best-effort)
        self._m = {
            "ticks_in": 0, "ticks_dropped": 0, "ticks_dispatched": 0,
            "news_in": 0, "news_dropped": 0, "news_dispatched": 0,
            "last_ts": _utc_ms(),
        }

        # threads
        self._threads: List[threading.Thread] = []

    # ---- strategy lifecycle ----
    def register(self, strat: Any) -> None:
        """Register a strategy-like object with on_tick(), optional on_news()."""
        self.strats.append(strat)

    def load_from_registry(self, path: str) -> None:
        for spec in _load_registry(path):
            s = _construct_strategy(spec)
            self.register(s)

    # ---- I/O ----
    def _consume_stream_loop(self, stream: str, q: "queue.Queue[Dict[str, Any]]", tag: str) -> None:
        cursor = "$"
        while self._running:
            try:
                for _, raw in consume_stream(stream, start_id=cursor, block_ms=1000, count=500):
                    data = raw if isinstance(raw, dict) else _safe_json(raw)
                    data["_stream"] = stream
                    if tag == "tick":
                        self._m["ticks_in"] += 1
                    else:
                        self._m["news_in"] += 1
                    try:
                        q.put_nowait(data)
                    except queue.Full:
                        if tag == "tick":
                            self._m["ticks_dropped"] += 1
                        else:
                            self._m["news_dropped"] += 1
                cursor = "$"
            except Exception:
                # stream may not exist; brief nap to prevent spin
                time.sleep(0.25)

    def _tick_worker(self, idx: int) -> None:
        while self._running:
            try:
                ev = self.q_ticks.get(timeout=0.5)
            except queue.Empty:
                continue
            for s in self.strats:
                try:
                    if hasattr(s, "on_tick"):
                        s.on_tick(ev)
                        self._m["ticks_dispatched"] += 1
                except Exception as e:
                    # push to a minimal error log on the bus (optional)
                    try:
                        publish_stream("engine.errors", {
                            "ts_ms": _utc_ms(), "where": "on_tick", "strategy": getattr(s, "ctx", None) and s.ctx.name or str(s),
                            "err": str(e)
                        })
                    except Exception:
                        pass

    def _news_worker(self, idx: int) -> None:
        while self._running:
            try:
                ev = self.q_news.get(timeout=0.5)
            except queue.Empty:
                continue
            for s in self.strats:
                try:
                    if hasattr(s, "on_news"):
                        s.on_news(ev)  # optional hook
                        self._m["news_dispatched"] += 1
                except Exception as e:
                    try:
                        publish_stream("engine.errors", {
                            "ts_ms": _utc_ms(), "where": "on_news", "strategy": getattr(s, "ctx", None) and s.ctx.name or str(s),
                            "err": str(e)
                        })
                    except Exception:
                        pass

    # ---- health / metrics ----
    def _heartbeat(self) -> None:
        """Periodic health ping + metrics dump to Redis (best-effort)."""
        while self._running:
            now = _utc_ms()
            info = {
                "ts_ms": now,
                "alive": True,
                "ticks_in": self._m["ticks_in"],
                "ticks_dropped": self._m["ticks_dropped"],
                "news_in": self._m["news_in"],
                "news_dropped": self._m["news_dropped"],
                "strats": [getattr(s, "ctx", None) and s.ctx.name or str(s) for s in self.strats],
                "tick_streams": list(self.cfg.tick_streams),
                "news_streams": list(self.cfg.news_streams),
                "pid": os.getpid(),
            }
            try:
                hset(HEALTH_KEY, "engine", info)
            except Exception:
                pass
            # tiny log every N seconds
            if (now - self._m["last_ts"]) >= self.cfg.log_every_s * 1000:
                self._m["last_ts"] = now
                try:
                    publish_stream("engine.metrics", {**info, "kind": "summary"})
                except Exception:
                    pass
            time.sleep(1.0)

    # ---- lifecycle ----
    def start(self) -> None:
        if self._running:
            return
        self._running = True

        # call on_start on each strategy
        for s in self.strats:
            try:
                if hasattr(s, "on_start"):
                    s.on_start()
            except Exception:
                pass

        # producers (bus -> queues)
        for stream in self.cfg.tick_streams:
            t = threading.Thread(target=self._consume_stream_loop, args=(stream.strip(), self.q_ticks, "tick"), daemon=True)
            t.start(); self._threads.append(t)
        for stream in self.cfg.news_streams:
            t = threading.Thread(target=self._consume_stream_loop, args=(stream.strip(), self.q_news, "news"), daemon=True)
            t.start(); self._threads.append(t)

        # consumers (queues -> strategies)
        for i in range(self.cfg.tick_workers):
            t = threading.Thread(target=self._tick_worker, args=(i,), daemon=True)
            t.start(); self._threads.append(t)
        for i in range(self.cfg.news_workers):
            t = threading.Thread(target=self._news_worker, args=(i,), daemon=True)
            t.start(); self._threads.append(t)

        # heartbeat
        t = threading.Thread(target=self._heartbeat, daemon=True)
        t.start(); self._threads.append(t)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        # give workers a moment to exit
        time.sleep(0.5)
        # call on_stop on each strategy
        for s in self.strats:
            try:
                if hasattr(s, "on_stop"):
                    s.on_stop()
            except Exception:
                pass
        try:
            hset(HEALTH_KEY, "engine", {"ts_ms": _utc_ms(), "alive": False})
        except Exception:
            pass

# ---------- CLI ----------
def _install_signal_handlers(engine: Engine):
    def _stop(signum, frame):
        try:
            engine.stop()
        finally:
            sys.exit(0)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

def _probe():
    # Minimal smoke test: load the ExampleBuyTheDip if present and pump synthetic events
    print("Engine probe: creating ExampleBuyTheDip and pushing 20 fake ticks...")
    try:
        from backend.engine.strategy_base import ExampleBuyTheDip
        strat = ExampleBuyTheDip(name="probe_dip", region="CRYPTO", default_qty=0.001, bps=10)
    except Exception:
        print("Could not import ExampleBuyTheDip; probe will just run engine without events.")
        strat = None

    cfg = EngineConfig(tick_streams=(), news_streams=())
    eng = Engine(cfg)
    if strat:
        eng.register(strat)
    _install_signal_handlers(eng)
    eng.start()

    # Emit fake ticks directly into queue (no Redis dependency)
    import random
    for i in range(20):
        tick = {"ts_ms": _utc_ms(), "symbol": "BTCUSDT", "price": 60000 + random.uniform(-200, 200)}
        try:
            eng.q_ticks.put_nowait(tick)
        except queue.Full:
            pass
        time.sleep(0.05)

    # brief run
    time.sleep(1.0)
    eng.stop()
    print("Probe complete.")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Engine Orchestrator")
    ap.add_argument("--run", action="store_true", help="Run engine attached to bus")
    ap.add_argument("--ticks", type=str, help="Comma-separated tick streams (override env/registry)")
    ap.add_argument("--news", type=str, help="Comma-separated news streams (override env/registry)")
    ap.add_argument("--registry", type=str, help="Path to registry.yaml to load strategies")
    ap.add_argument("--probe", action="store_true", help="Run a quick local probe (no Redis required)")
    args = ap.parse_args()

    if args.probe:
        _probe()
        return

    # build config
    tick_streams = tuple([s.strip() for s in (args.ticks.split(",") if args.ticks else DEFAULT_TICK_STREAMS) if s.strip()])
    news_streams = tuple([s.strip() for s in (args.news.split(",") if args.news else DEFAULT_NEWS_STREAMS) if s.strip()])
    cfg = EngineConfig(tick_streams=tick_streams, news_streams=news_streams, registry_path=args.registry)

    eng = Engine(cfg)
    if args.registry:
        eng.load_from_registry(args.registry)
    else:
        # Fallback: try to load ExampleBuyTheDip so engine is never empty
        try:
            from backend.engine.strategy_base import ExampleBuyTheDip
            eng.register(ExampleBuyTheDip(name="example_buy_dip", region="CRYPTO", default_qty=0.001, bps=10))
        except Exception:
            pass

    if not eng.strats:
        print("No strategies registered. Provide --registry or ensure at least one strategy import works.")
    _install_signal_handlers(eng)
    eng.start()

    # Keep the main thread alive while workers run
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()

if __name__ == "__main__":
    main()