# observability/chaos/kill_switch.py
"""
Chaos kill-switch & fault injector.

Usage (quick):
    from observability.chaos.kill_switch import chaos

    # simple delay gate in API handler
    async def handle_request(req):
        await chaos.maybe_delay("inject_latency_api", scope="service:api")
        ...

    # probabilistic drop in a bus consumer
    if chaos.maybe_drop("drop_bus_messages", scope="topic:orders"):
        return  # skip processing

    # decorators
    @chaos.latency("inject_latency_api", scope="service:api")
    def fetch_quote(...):
        ...

    # context manager
    with chaos.inject("cpu_stress"):
        chaos.cpu_stress(ms=2000)

Config:
    Reads YAML at observability/chaos/fault_injection.yaml (override via env CHAOS_CONFIG).
Master switch:
    - env CHAOS_ENABLED=1 overrides config defaults.enabled
Hot reload:
    chaos.reload()  # re-read YAML (e.g., on SIGHUP)
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Optional YAML (fallback to JSON if yaml not present)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


# ----------------------------- models -----------------------------

@dataclass
class Scenario:
    name: str
    enabled: bool
    probability: float
    duration_ms: int
    scope: str
    action: Optional[str] = None  # e.g., "SIGKILL"


# ----------------------------- core controller -----------------------------

class Chaos:
    def __init__(self, path: Optional[str] = None):
        self.path = (
            path or os.getenv("CHAOS_CONFIG") or "observability/chaos/fault_injection.yaml"
        )
        self._enabled = False
        self._scenarios: Dict[str, Scenario] = {}
        self.reload()

        # allow SIGUSR1 to toggle, SIGUSR2 to reload (optional)
        with _safe_signal(signal.SIGUSR1, self._sig_toggle):
            pass
        with _safe_signal(signal.SIGUSR2, self._sig_reload):
            pass

    # ---------- lifecycle ----------

    def reload(self) -> None:
        cfg = self._load_cfg(self.path)
        defaults = cfg.get("defaults", {})
        self._enabled = bool(
            int(os.getenv("CHAOS_ENABLED", "1" if defaults.get("enabled") else "0"))
        )
        self._scenarios.clear()
        for s in cfg.get("scenarios", []) or []:
            scen = Scenario(
                name=s.get("name"),
                enabled=bool(s.get("enabled", defaults.get("enabled", False))),
                probability=float(s.get("probability", defaults.get("probability", 0.0))),
                duration_ms=int(s.get("duration_ms", defaults.get("duration_ms", 0))),
                scope=str(s.get("scope", defaults.get("scope", "global"))),
                action=s.get("action"),
            )
            self._scenarios[scen.name] = scen

    def toggle(self, on: Optional[bool] = None) -> None:
        self._enabled = (not self._enabled) if on is None else bool(on)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ---------- decision helpers ----------

    def scenario(self, name: str) -> Optional[Scenario]:
        return self._scenarios.get(name)

    def should_inject(self, name: str, scope: Optional[str] = None) -> bool:
        """
        Return True if fault 'name' should be injected for this call.
        Scope filter: if scenario.scope is not 'global', it must match provided scope.
        """
        if not self._enabled:
            return False
        s = self._scenarios.get(name)
        if not s or not s.enabled:
            return False
        if s.scope != "global" and scope and s.scope != scope:
            return False
        # probabilistic decision
        return random.random() < max(0.0, min(1.0, s.probability))

    # ---------- primitives ----------

    def maybe_delay(self, name: str, scope: Optional[str] = None) -> None:
        s = self._scenarios.get(name)
        if self.should_inject(name, scope) and s and s.duration_ms > 0:
            _sleep_ms(s.duration_ms)

    def maybe_drop(self, name: str, scope: Optional[str] = None) -> bool:
        """Returns True if the message/request should be dropped (i.e., caller should early-return)."""
        return self.should_inject(name, scope)

    def maybe_kill(self, name: str, scope: Optional[str] = None) -> None:
        s = self._scenarios.get(name)
        if self.should_inject(name, scope) and s and s.action:
            sig = getattr(signal, s.action, None)
            if isinstance(sig, int):
                os.kill(os.getpid(), sig)

    def cpu_stress(self, ms: int) -> None:
        """Busy-loop CPU for ms milliseconds (single thread)."""
        end = time.perf_counter() + (ms / 1000.0)
        x = 0.0
        while time.perf_counter() < end:
            # tiny math to avoid being optimized away
            x = (x * x + 1.2345) % 3.14159  # noqa: F841

    def memory_leak(self, mb: int, hold_ms: int = 5000) -> None:
        """Allocate ~mb megabytes and hold for a while (then GC)."""
        block = bytearray(mb * 1024 * 1024)
        _sleep_ms(hold_ms)
        # release by scope end; caller controls repetition for leak simulation

    # ---------- decorators / contexts ----------

    def latency(self, name: str, scope: Optional[str] = None):
        """Decorator that injects delay before function if scenario hits."""
        def deco(fn):
            if _is_coroutine(fn):
                async def aw(*a, **kw):
                    self.maybe_delay(name, scope)
                    return await fn(*a, **kw)
                return aw
            else:
                def w(*a, **kw):
                    self.maybe_delay(name, scope)
                    return fn(*a, **kw)
                return w
        return deco

    @contextmanager
    def inject(self, name: str, scope: Optional[str] = None):
        """Generic context manager to gate a block by scenario decision."""
        fired = self.should_inject(name, scope)
        try:
            yield fired
        finally:
            pass

    # ---------- signals ----------

    def _sig_toggle(self, *_):
        self.toggle()

    def _sig_reload(self, *_):
        self.reload()

    # ---------- I/O ----------

    @staticmethod
    def _load_cfg(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {"defaults": {"enabled": False}, "scenarios": []}
        if yaml is not None and path.endswith((".yaml", ".yml")):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        # allow JSON fallback
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {"defaults": {"enabled": False}, "scenarios": []}


# ----------------------------- helpers -----------------------------

def _sleep_ms(ms: int) -> None:
    time.sleep(max(0.0, ms) / 1000.0)

def _is_coroutine(fn) -> bool:
    try:
        import inspect
        return inspect.iscoroutinefunction(fn)
    except Exception:
        return False

class _safe_signal:
    """Context manager to register a signal handler if supported (no-op on Windows)."""
    def __init__(self, sig, handler):
        self.sig = sig
        self.handler = handler
        self.ok = hasattr(signal, "signal")

    def __enter__(self):
        if self.ok:
            try:
                signal.signal(self.sig, self.handler)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# Singleton for convenience
chaos = Chaos()