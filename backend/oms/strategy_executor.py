# backend/engine/strategy_executor.py
"""
Strategy Executor
-----------------
Run one or more strategies with robust lifecycle management:
- Load from registry YAML or programmatically register specs
- Subscribe to tick/news streams; dispatch to on_tick/on_news
- Enable/disable via Redis flag HSET strategy:enabled <name> true/false
- Command channel: strategy.cmd.<name> (pause|resume|restart|set key=val)
- Crash auto-restart with exponential backoff
- Heartbeats & metrics to Redis bus

CLI
  python -m backend.engine.strategy_executor --probe
  python -m backend.engine.strategy_executor --run --registry config/registry.yaml
  python -m backend.engine.strategy_executor --run --module backend.engine.strategy_base --klass ExampleBuyTheDip --name dip --ticks trades.crypto

Registry YAML (example)
  strategies:
    - module: backend.engine.strategy_base
      class: ExampleBuyTheDip
      name:  dipper_crypto
      region: CRYPTO
      params: { default_qty: 0.002, bps: 8 }
      ticks:  [trades.crypto]
      news:   [news.moneycontrol, news.yahoo]

Env
  REDIS_HOST=localhost REDIS_PORT=6379
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

# ---- Optional deps / glue (graceful if missing) ----
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # type: ignore

try:
    from backend.bus.streams import consume_stream, publish_stream, hset
except Exception as e:
    raise RuntimeError("backend.bus.streams not found; add your Redis bus helpers") from e

try:
    import redis  # pip install redis
except Exception:
    redis = None  # type: ignore

# Strategy base is only for typing; any object with on_tick/on_news works.
try:
    from backend.engine.strategy_base import Strategy
except Exception:
    Strategy = object  # type: ignore


# ---------- Config models ----------

@dataclass
class StratSpec:
    module: str
    klass: str
    name: str
    region: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    ticks: Sequence[str] = field(default_factory=tuple)
    news: Sequence[str] = field(default_factory=tuple)

@dataclass
class ExecConfig:
    queue_max: int = 10000
    tick_workers: int = 1
    news_workers: int = 1
    log_every_s: int = 10


# ---------- Helpers ----------

def _utc_ms() -> int: return int(time.time() * 1000)

def _safe_json(d: Any) -> Dict[str, Any]:
    if isinstance(d, dict): return d
    try: return json.loads(d)
    except Exception: return {}

def _load_registry(path: str) -> List[StratSpec]:
    if not yaml:
        raise RuntimeError("pyyaml not installed; cannot read registry.yaml")
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    out: List[StratSpec] = []
    for s in (doc.get("strategies") or []):
        out.append(StratSpec(
            module=s["module"],
            klass=s["class"],
            name=s["name"],
            region=s.get("region"),
            params=s.get("params") or {},
            ticks=tuple(s.get("ticks") or []),
            news=tuple(s.get("news") or []),
        ))
    return out

def _construct_strategy(spec: StratSpec) -> Any:
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.klass)
    return cls(name=spec.name, region=spec.region, **spec.params)


# ---------- Per-strategy runner ----------

class _StratRunner:
    """
    Encapsulates queues, threads, and lifecycle for a single strategy.
    """
    def __init__(self, spec: StratSpec, cfg: ExecConfig, rconn: Optional["redis.Redis"] = None): # type: ignore
        self.spec = spec
        self.cfg = cfg
        self._r = rconn
        self._running = False
        self._paused = False
        self._threads: List[threading.Thread] = []
        self._backoff = 1.0

        self.q_ticks: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_max)
        self.q_news: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_max)

        self._m = {
            "ticks_in": 0, "ticks_drop": 0, "ticks_dispatch": 0,
            "news_in": 0,  "news_drop": 0,  "news_dispatch": 0,
            "last_hb": _utc_ms()
        }

        self._obj: Optional[Any] = None  # strategy instance

    # ---- Redis helpers ----
    def _enabled(self) -> bool:
        # Enabled by default if Redis absent
        if not self._r:
            return True
        try:
            v = self._r.hget("strategy:enabled", self.spec.name)
            return (v is None) or (str(v).lower() == "true")
        except Exception:
            return True

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        msg = {"ts_ms": _utc_ms(), "strategy": self.spec.name, "kind": kind, **payload}
        try:
            publish_stream("engine.strategy", msg)
        except Exception:
            pass

    # ---- lifecycle ----
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._spawn_strategy()
        self._spawn_io()
        self._spawn_cmd_listener()
        self._spawn_heartbeat()

    def stop(self) -> None:
        self._running = False
        # give workers a breath
        time.sleep(0.25)
        # stop strategy
        try:
            if self._obj and hasattr(self._obj, "on_stop"):
                self._obj.on_stop()
        except Exception:
            pass

    def pause(self) -> None:
        self._paused = True
        self._emit("paused", {})

    def resume(self) -> None:
        self._paused = False
        self._emit("resumed", {})

    def restart(self) -> None:
        self.stop()
        self._backoff = 1.0
        self.start()
        self._emit("restarted", {})

    def set_param(self, key: str, value: Any) -> None:
        try:
            if self._obj and hasattr(self._obj, "ctx"):
                setattr(self._obj.ctx, key, value)
            else:
                setattr(self._obj, key, value)  # last resort
            self._emit("param_set", {"key": key, "value": value})
        except Exception as e:
            self._emit("error", {"op": "set_param", "err": str(e)})

    # ---- internals ----
    def _spawn_strategy(self):
        """
        Construct strategy (with exponential backoff on failure).
        """
        while self._running:
            try:
                self._obj = _construct_strategy(self.spec)
                if hasattr(self._obj, "on_start"):
                    self._obj.on_start() # type: ignore
                self._emit("started", {"region": self.spec.region})
                self._backoff = 1.0
                return
            except Exception as e:
                self._emit("error", {"op": "construct", "err": str(e)})
                time.sleep(min(60.0, self._backoff))
                self._backoff *= 2

    def _spawn_io(self):
        # producers -> queues
        for s in self.spec.ticks:
            t = threading.Thread(target=self._consume_loop, args=(s, self.q_ticks, "tick"), daemon=True)
            t.start(); self._threads.append(t)
        for s in self.spec.news:
            t = threading.Thread(target=self._consume_loop, args=(s, self.q_news, "news"), daemon=True)
            t.start(); self._threads.append(t)

        # consumers -> strategy
        for i in range(self.cfg.tick_workers):
            t = threading.Thread(target=self._tick_worker, args=(i,), daemon=True)
            t.start(); self._threads.append(t)
        for i in range(self.cfg.news_workers):
            t = threading.Thread(target=self._news_worker, args=(i,), daemon=True)
            t.start(); self._threads.append(t)

    def _consume_loop(self, stream: str, q: "queue.Queue[Dict[str, Any]]", tag: str):
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
                        if tag == "tick": self._m["ticks_drop"] += 1
                        else: self._m["news_drop"] += 1
                cursor = "$"
            except Exception:
                time.sleep(0.25)

    def _tick_worker(self, idx: int):
        while self._running:
            try:
                ev = self.q_ticks.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._paused or not self._enabled():
                continue
            try:
                if self._obj and hasattr(self._obj, "on_tick"):
                    self._obj.on_tick(ev)
                    self._m["ticks_dispatch"] += 1
            except Exception as e:
                self._emit("error", {"where": "on_tick", "err": str(e)})
                self._crash_restart()

    def _news_worker(self, idx: int):
        while self._running:
            try:
                ev = self.q_news.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._paused or not self._enabled():
                continue
            try:
                if self._obj and hasattr(self._obj, "on_news"):
                    self._obj.on_news(ev)
                    self._m["news_dispatch"] += 1
            except Exception as e:
                self._emit("error", {"where": "on_news", "err": str(e)})
                self._crash_restart()

    def _crash_restart(self):
        # backoff + rebuild strategy
        try:
            if self._obj and hasattr(self._obj, "on_stop"):
                self._obj.on_stop()
        except Exception:
            pass
        delay = min(60.0, max(1.0, getattr(self, "_backoff", 1.0)))
        self._emit("restarting", {"after_s": delay})
        time.sleep(delay)
        self._backoff = min(60.0, delay * 2.0)
        self._spawn_strategy()

    def _spawn_cmd_listener(self):
        """
        Listen for admin commands on stream: strategy.cmd.<name>
        Payloads:
          {"cmd":"pause"} | {"cmd":"resume"} | {"cmd":"restart"}
          {"cmd":"set","key":"default_qty","value":2.0}
        """
        stream = f"strategy.cmd.{self.spec.name}"
        def _loop():
            cursor = "$"
            while self._running:
                try:
                    for _, raw in consume_stream(stream, start_id=cursor, block_ms=1000, count=100):
                        cmd = raw if isinstance(raw, dict) else _safe_json(raw)
                        c = (cmd.get("cmd") or "").lower()
                        if c == "pause": self.pause()
                        elif c == "resume": self.resume()
                        elif c == "restart": self.restart()
                        elif c == "set": self.set_param(str(cmd.get("key")), cmd.get("value"))
                        else: self._emit("unknown_cmd", {"cmd": cmd})
                    cursor = "$"
                except Exception:
                    time.sleep(0.25)
        t = threading.Thread(target=_loop, daemon=True)
        t.start(); self._threads.append(t)

    def _spawn_heartbeat(self):
        def _hb():
            while self._running:
                info = {
                    "ts_ms": _utc_ms(),
                    "name": self.spec.name,
                    "region": self.spec.region,
                    "paused": self._paused,
                    "enabled": self._enabled(),
                    "ticks_in": self._m["ticks_in"], "ticks_drop": self._m["ticks_drop"], "ticks_dispatch": self._m["ticks_dispatch"],
                    "news_in": self._m["news_in"],   "news_drop": self._m["news_drop"],   "news_dispatch": self._m["news_dispatch"],
                    "pid": os.getpid(),
                }
                try:
                    hset("strategy:health", self.spec.name, info)
                    if (_utc_ms() - self._m["last_hb"]) >= (self.cfg.log_every_s * 1000):
                        self._m["last_hb"] = _utc_ms()
                        publish_stream("engine.metrics", {"kind":"strategy", **info})
                except Exception:
                    pass
                time.sleep(1.0)
        t = threading.Thread(target=_hb, daemon=True)
        t.start(); self._threads.append(t)


# ---------- Orchestrator for many strategies ----------

class StrategyExecutor:
    def __init__(self, exec_cfg: Optional[ExecConfig] = None):
        self.cfg = exec_cfg or ExecConfig()
        self.runners: List[_StratRunner] = []
        self._running = False
        self._r = None
        if redis:
            try:
                host = os.getenv("REDIS_HOST", "localhost"); port = int(os.getenv("REDIS_PORT", "6379"))
                self._r = redis.Redis(host=host, port=port, decode_responses=True)
            except Exception:
                self._r = None

    def register(self, spec: StratSpec) -> None:
        self.runners.append(_StratRunner(spec, self.cfg, self._r))

    def load_registry(self, path: str) -> None:
        for s in _load_registry(path):
            self.register(s)

    def start(self) -> None:
        if self._running: return
        self._running = True
        for r in self.runners:
            r.start()

    def stop(self) -> None:
        if not self._running: return
        self._running = False
        for r in self.runners:
            r.stop()


# ---------- CLI ----------

def _install_signals(exe: StrategyExecutor):
    def _stop(signum, frame):
        try:
            exe.stop()
        finally:
            sys.exit(0)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

def _probe():
    """
    No Redis required: instantiate ExampleBuyTheDip and feed synthetic ticks.
    """
    print("Probe: spinning a single ExampleBuyTheDip with fake ticks...")
    try:
        from backend.engine.strategy_base import ExampleBuyTheDip
        # Build a spec that points at no streams (we'll feed directly)
        spec = StratSpec(
            module="backend.engine.strategy_base",
            klass="ExampleBuyTheDip",
            name="probe_dip",
            region="CRYPTO",
            params={"default_qty": 0.001, "bps": 10},
            ticks=(), news=(),
        )
    except Exception:
        print("Could not import ExampleBuyTheDip. Probe will still exercise lifecycle.")
        spec = StratSpec(module="backend.engine.strategy_base", klass="Strategy", name="noop", ticks=(), news=())

    exe = StrategyExecutor()
    exe.register(spec)
    _install_signals(exe)
    exe.start()

    # feed a handful of synthetic ticks to the runner queue if present
    runner = exe.runners[0]
    import random
    for _ in range(30):
        tick = {"ts_ms": _utc_ms(), "symbol": "BTCUSDT", "price": 60000 + random.uniform(-150, 150)}
        try:
            runner.q_ticks.put_nowait(tick)
        except queue.Full:
            pass
        time.sleep(0.05)

    time.sleep(1.0)
    exe.stop()
    print("Probe complete.")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Strategy Executor")
    ap.add_argument("--probe", action="store_true", help="Run a local probe (no Redis)")
    ap.add_argument("--run", action="store_true", help="Run with real streams")
    ap.add_argument("--registry", type=str, help="Path to registry.yaml")
    # quick one-off without registry
    ap.add_argument("--module", type=str, help="Module path to strategy class")
    ap.add_argument("--klass", type=str, help="Class name")
    ap.add_argument("--name", type=str, help="Instance name")
    ap.add_argument("--region", type=str)
    ap.add_argument("--ticks", type=str, help="Comma-separated tick streams")
    ap.add_argument("--news", type=str, help="Comma-separated news streams")
    args = ap.parse_args()

    if args.probe:
        _probe()
        return

    exe = StrategyExecutor()
    if args.registry:
        exe.load_registry(args.registry)
    elif args.module and args.klass and args.name:
        spec = StratSpec(
            module=args.module, klass=args.klass, name=args.name,
            region=args.region, params={},
            ticks=tuple([s.strip() for s in (args.ticks or "").split(",") if s.strip()]),
            news=tuple([s.strip() for s in (args.news or "").split(",") if s.strip()]),
        )
        exe.register(spec)
    else:
        print("Provide --registry or (--module --klass --name). For a quick check, use --probe.")
        return

    _install_signals(exe)
    exe.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        exe.stop()

if __name__ == "__main__":
    main()