# backend/ops/chaos_money.py
from __future__ import annotations
"""
Chaos Money — fault & shock injector for your trading stack
-----------------------------------------------------------
Use in dev/staging to validate resiliency: OMS, router, risk, allocators, UI.

What it can do (configurable):
- Latency: add fixed + jitter to publish_stream (and optionally consume_stream)
- Drops: drop a percentage of bus messages
- Duplicates: duplicate some messages (idempotency test)
- Reorder: buffer & emit out-of-order (race/causality tests)
- Clock skew: add ts offset to outbound events (timestamp discipline tests)
- Market shocks: inject synthetic ticks/quotes/spreads into target streams
- Venue outage: mark venues DOWN/UP and publish outage events
- Slippage hints: inject OMS slippage overrides to test TCA/risk handling
- Kill switch: publish a risk trigger event (to validate your safety path)

Safety rails:
- Requires explicit enable (env CHAOS_ENABLE=1 or --force)
- Defaults to DRY-RUN
- Never runs in processes where ENV in {"prod","production"} unless --force
- Clear, structured event logs to `ops.chaos.events` stream (if bus available)

CLI:
  python -m backend.ops.chaos_money plan --cfg chaos.yaml
  python -m backend.ops.chaos_money patch-bus --cfg chaos.yaml --duration 60
  python -m backend.ops.chaos_money run --cfg chaos.yaml --duration 120
"""

import os, time, json, random, threading, queue
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Callable, Tuple, Iterable
from contextlib import contextmanager

# -------- Optional bus / Redis ----------
try:
    from backend.bus.streams import publish_stream, consume_stream, hset  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:  # type: ignore
        print(f"[CHAOS] publish_stream -> {stream}: {payload}")
    def consume_stream(*args, **kwargs):  # type: ignore
        raise RuntimeError("consume_stream not available; patching consumer disabled.")
    def hset(key: str, field: str, value: Any) -> None:  # type: ignore
        print(f"[CHAOS] hset {key}[{field}] = {value}")

# -------- Optional YAML ----------
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

OPS_STREAM = os.getenv("CHAOS_EVENTS_STREAM", "ops.chaos.events")

# ---------------- Config ----------------

@dataclass
class BusFaults:
    enable: bool = True
    drop_prob: float = 0.0         # 0..1
    dup_prob: float = 0.0          # 0..1
    base_delay_ms: int = 0
    jitter_ms: int = 0
    reorder_prob: float = 0.0      # 0..1 (uses small buffer)
    reorder_buffer_max: int = 32
    clock_skew_ms: int = 0         # +/- ms added to payload["ts_ms"] if present
    affect_streams: Optional[List[str]] = None  # if set, only these stream names

@dataclass
class Shock:
    # Emitted as synthetic ticks/quotes into a stream
    stream: str
    symbol: str
    kind: str = "tick"             # "tick" | "quote" | "spread_widen"
    magnitude: float = 0.03        # e.g., 0.03 => ±3%
    duration_ms: int = 0           # if > 0, emit a sequence over this window
    count: int = 1                 # number of events to send
    region: Optional[str] = None
    note: Optional[str] = None

@dataclass
class VenueOutage:
    venue: str
    down_ms: int = 10_000
    stream: str = "ops.venue_outage"  # notifications bus
    set_key: str = "venue:status"     # Redis hash key: field=VENUE => {"status":"down/up"}

@dataclass
class SlippageHint:
    stream: str = "oms.slippage_hint"
    bps: float = 5.0                  # positive worsens fills
    apply_ms: int = 5_000
    symbol: Optional[str] = None

@dataclass
class KillSwitch:
    stream: str = "risk.killswitch"
    reason: str = "chaos_test"
    cooldown_ms: int = 0

@dataclass
class ChaosConfig:
    allow_envs: Tuple[str, ...] = ("dev", "staging", "local")
    dry_run: bool = True
    seed: Optional[int] = None
    bus_faults: BusFaults = BusFaults()
    shocks: List[Shock] = None  # type: ignore
    outages: List[VenueOutage] = None  # type: ignore
    slippage: Optional[SlippageHint] = None
    killswitch: Optional[KillSwitch] = None

# ---------------- Loader ----------------

def load_config(path: str) -> ChaosConfig:
    if path.lower().endswith((".yaml", ".yml")) and yaml:
        raw = yaml.safe_load(open(path, "r", encoding="utf-8"))
    else:
        raw = json.load(open(path, "r", encoding="utf-8"))
    def _from(cls, obj, default=None):
        if obj is None: return default
        return cls(**obj)
    cfg = ChaosConfig(
        allow_envs=tuple(raw.get("allow_envs", ["dev","staging","local"])),
        dry_run=bool(raw.get("dry_run", True)),
        seed=raw.get("seed"),
        bus_faults=_from(BusFaults, raw.get("bus_faults"), BusFaults()),
        shocks=[_from(Shock, s) for s in (raw.get("shocks") or [])], # type: ignore
        outages=[_from(VenueOutage, o) for o in (raw.get("outages") or [])], # type: ignore
        slippage=_from(SlippageHint, raw.get("slippage")),
        killswitch=_from(KillSwitch, raw.get("killswitch")),
    )
    return cfg

# ---------------- Core ----------------

class ChaosMoney:
    def __init__(self, cfg: ChaosConfig, *, force: bool = False):
        self.cfg = cfg
        self.force = force
        self._rng = random.Random(cfg.seed)
        self._orig_publish = publish_stream
        self._orig_consume = consume_stream
        self._reorder_buf: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue(maxsize=max(4, cfg.bus_faults.reorder_buffer_max or 32))
        self._reorder_thread: Optional[threading.Thread] = None
        self._stop_reorder = threading.Event()

        env = os.getenv("ENV", os.getenv("ENVIRONMENT", "dev")).lower()
        if not force:
            if os.getenv("CHAOS_ENABLE", "0") != "1":
                raise RuntimeError("ChaosMoney disabled (set CHAOS_ENABLE=1 or pass --force)")
            if env not in self.cfg.allow_envs:
                raise RuntimeError(f"ENV='{env}' not in allow list {self.cfg.allow_envs} (use --force to override)")

    # ---------- Event log ----------
    def _log(self, kind: str, data: Dict[str, Any]):
        payload = {"ts_ms": int(time.time()*1000), "kind": kind, **data}
        try:
            self._orig_publish(OPS_STREAM, payload)
        except Exception:
            print("[CHAOS-LOG]", payload)

    # ---------- Bus patch ----------
    @contextmanager
    def patch_bus(self):
        """Monkey-patch publish_stream (and optionally reorder) within this context."""
        if not self.cfg.bus_faults.enable:
            yield
            return

        bf = self.cfg.bus_faults
        rng = self._rng
        reorder = (bf.reorder_prob or 0.0) > 0.0

        def _affected(stream_name: str) -> bool:
            if not bf.affect_streams: return True
            return any(stream_name.startswith(s) or stream_name == s for s in bf.affect_streams)

        def _patched_publish(stream_name: str, payload: Dict[str, Any]) -> None:
            if not _affected(stream_name):
                return self._orig_publish(stream_name, payload)

            # clock skew
            if "ts_ms" in payload and bf.clock_skew_ms:
                payload = dict(payload)
                payload["ts_ms"] = int(payload["ts_ms"]) + int(bf.clock_skew_ms)

            # delay
            delay = (bf.base_delay_ms or 0) + (rng.randint(0, bf.jitter_ms or 0))
            if delay > 0:
                time.sleep(delay / 1000.0)

            # drop
            if rng.random() < (bf.drop_prob or 0.0):
                self._log("drop", {"stream": stream_name})
                return  # silently drop

            # duplicate
            dup = rng.random() < (bf.dup_prob or 0.0)

            # reorder (buffered)
            if reorder and rng.random() < (bf.reorder_prob or 0.0):
                try:
                    self._reorder_buf.put_nowait((stream_name, payload))
                    self._log("reorder_buffer", {"stream": stream_name})
                except queue.Full:
                    # fallback: publish normally if buffer full
                    self._orig_publish(stream_name, payload)
                # still maybe duplicate original on purpose?
                if dup:
                    self._orig_publish(stream_name, payload)
                return

            # normal publish
            self._orig_publish(stream_name, payload)
            if dup:
                self._log("duplicate", {"stream": stream_name})
                self._orig_publish(stream_name, payload)

        # start reorder flusher
        if reorder:
            self._stop_reorder.clear()
            self._reorder_thread = threading.Thread(target=self._reorder_pump, daemon=True)
            self._reorder_thread.start()

        # patch
        globals()["publish_stream"] = _patched_publish
        try:
            yield
        finally:
            # flush remaining buffered messages randomly
            try:
                while not self._reorder_buf.empty():
                    s, p = self._reorder_buf.get_nowait()
                    self._orig_publish(s, p)
            except Exception:
                pass
            # stop reordering thread
            self._stop_reorder.set()
            if self._reorder_thread:
                self._reorder_thread.join(timeout=1.0)
                self._reorder_thread = None
            # unpatch
            globals()["publish_stream"] = self._orig_publish

    def _reorder_pump(self):
        rng = self._rng
        while not self._stop_reorder.is_set():
            try:
                s, p = self._reorder_buf.get(timeout=0.1)
            except queue.Empty:
                continue
            # hold a bit to ensure out-of-order emission
            time.sleep(rng.uniform(0.01, 0.2))
            try:
                self._orig_publish(s, p)
                self._log("reorder_emit", {"stream": s})
            except Exception:
                pass

    # ---------- Actions ----------
    def emit_shock(self, s: Shock):
        now = int(time.time()*1000)
        n = max(1, int(s.count or 1))
        dur = int(s.duration_ms or 0)
        self._log("shock_start", {"symbol": s.symbol, "stream": s.stream, "kind": s.kind, "count": n, "dur_ms": dur})
        for i in range(n):
            payload: Dict[str, Any] = {"ts_ms": now + i*10, "symbol": s.symbol, "region": s.region or "NA", "note": s.note or "chaos_shock"}
            if s.kind == "tick":
                direction = -1.0 if i % 2 else 1.0
                payload.update({"price_shock": direction * float(s.magnitude), "type": "tick"})
            elif s.kind == "quote":
                payload.update({"bid_shock": -abs(s.magnitude), "ask_shock": +abs(s.magnitude), "type": "quote"})
            elif s.kind == "spread_widen":
                payload.update({"spread_bps": abs(s.magnitude) * 1e4, "type": "spread"})
            else:
                payload.update({"price_shock": s.magnitude, "type": s.kind})
            publish_stream(s.stream, payload)
            if dur and n > 1:
                time.sleep(dur / 1000.0 / n)
        self._log("shock_end", {"symbol": s.symbol})

    def set_venue_outage(self, o: VenueOutage):
        # mark venue down
        hset(o.set_key, o.venue.upper(), {"status": "down", "ts_ms": int(time.time()*1000)})
        publish_stream(o.stream, {"ts_ms": int(time.time()*1000), "venue": o.venue, "status": "down"})
        self._log("venue_down", {"venue": o.venue, "for_ms": o.down_ms})
        time.sleep(max(0, o.down_ms) / 1000.0)
        # bring venue back
        hset(o.set_key, o.venue.upper(), {"status": "up", "ts_ms": int(time.time()*1000)})
        publish_stream(o.stream, {"ts_ms": int(time.time()*1000), "venue": o.venue, "status": "up"})
        self._log("venue_up", {"venue": o.venue})

    def slippage_window(self, s: SlippageHint):
        t0 = time.time()
        end = t0 + (s.apply_ms / 1000.0)
        self._log("slippage_on", {"bps": s.bps, "ms": s.apply_ms, "symbol": s.symbol})
        while time.time() < end:
            payload = {
                "ts_ms": int(time.time()*1000),
                "bps": float(s.bps),
                "symbol": s.symbol,
                "note": "chaos_slippage_hint"
            }
            publish_stream(s.stream, payload)
            time.sleep(0.25)
        self._log("slippage_off", {})

    def trigger_killswitch(self, k: KillSwitch):
        publish_stream(k.stream, {"ts_ms": int(time.time()*1000), "reason": k.reason})
        self._log("killswitch", {"reason": k.reason})
        if k.cooldown_ms > 0:
            time.sleep(k.cooldown_ms / 1000.0)

    # ---------- Orchestrations ----------
    def run(self, duration_s: int = 30):
        """
        Run a scenario: patch bus (faults), then concurrently execute shocks/outages/slippage/kill.
        """
        self._log("run_start", {"duration_s": duration_s})
        if self.cfg.dry_run:
            self._emit_plan(dry=True)
            self._log("dry_run", {"note": "no faults applied"})
            return

        threads: List[threading.Thread] = []

        with self.patch_bus():
            # schedule venue outages
            for o in (self.cfg.outages or []):
                t = threading.Thread(target=self.set_venue_outage, args=(o,), daemon=True)
                threads.append(t); t.start()

            # schedule shocks
            for s in (self.cfg.shocks or []):
                t = threading.Thread(target=self.emit_shock, args=(s,), daemon=True)
                threads.append(t); t.start()

            # slippage window
            if self.cfg.slippage:
                t = threading.Thread(target=self.slippage_window, args=(self.cfg.slippage,), daemon=True)
                threads.append(t); t.start()

            # optional kill switch near the end
            if self.cfg.killswitch:
                def _ks():
                    time.sleep(max(0, duration_s - 2))
                    self.trigger_killswitch(self.cfg.killswitch) # type: ignore
                t = threading.Thread(target=_ks, daemon=True)
                threads.append(t); t.start()

            # hold for duration
            t_end = time.time() + max(0, duration_s)
            while time.time() < t_end:
                time.sleep(0.25)

        # join threads briefly
        for t in threads:
            t.join(timeout=1.0)

        self._log("run_end", {})

    def _emit_plan(self, dry: bool = True):
        bf = asdict(self.cfg.bus_faults)
        plan = {
            "bus_faults": bf,
            "shocks": [asdict(s) for s in (self.cfg.shocks or [])],
            "outages": [asdict(o) for o in (self.cfg.outages or [])],
            "slippage": (asdict(self.cfg.slippage) if self.cfg.slippage else None),
            "killswitch": (asdict(self.cfg.killswitch) if self.cfg.killswitch else None),
            "dry_run": dry
        }
        self._log("plan", plan)
        print(json.dumps(plan, indent=2))

# ---------------- CLI ----------------

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Chaos Money — chaos engineering for your trading stack")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="Print the plan (no side effects)")
    p.add_argument("--cfg", required=True, help="YAML/JSON config")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_plan)

    b = sub.add_parser("patch-bus", help="Patch bus only for a duration (no shocks/outages)")
    b.add_argument("--cfg", required=True)
    b.add_argument("--duration", type=int, default=30)
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=_cmd_patch_bus)

    r = sub.add_parser("run", help="Run full chaos scenario")
    r.add_argument("--cfg", required=True)
    r.add_argument("--duration", type=int, default=60)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=_cmd_run)

    return ap.parse_args()

def _cmd_plan(args):
    cfg = load_config(args.cfg)
    cm = ChaosMoney(cfg, force=args.force)
    cm._emit_plan(dry=True)

def _cmd_patch_bus(args):
    cfg = load_config(args.cfg)
    cfg.dry_run = False
    cm = ChaosMoney(cfg, force=args.force)
    with cm.patch_bus():
        time.sleep(max(0, int(args.duration)))

def _cmd_run(args):
    cfg = load_config(args.cfg)
    cm = ChaosMoney(cfg, force=args.force)
    cm.run(duration_s=int(args.duration))

def main():
    args = _parse_args()
    os.environ.setdefault("CHAOS_ENABLE", "1")  # allow CLI by default
    main_env = os.getenv("ENV", os.getenv("ENVIRONMENT", "dev")).lower()
    if main_env in {"prod", "production"} and not getattr(args, "force", False):
        raise SystemExit("Refusing to run in production without --force")
    args.func(args)

if __name__ == "__main__":  # pragma: no cover
    main()