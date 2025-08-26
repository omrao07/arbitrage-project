# backend/execution_plus/registry.py
"""
Execution/Strategy Registry Hub

What this gives you
-------------------
1) A generic, thread-safe Registry class.
2) A RegistryHub with namespaces:
   - adapters     : execution adapters (e.g., BINANCE, NYSE)
   - strategies   : strategy classes/factories
   - cost_models  : cost/fee/slippage estimators
   - risk_checks  : pre-trade risk rules
   - routers      : route planners (e.g., min_cost_latency)
3) Loader helpers:
   - load_adapters_from_yaml(venues_yaml_path) -> Dict[venue_id, AdapterBase]
   - autoload_plugins() via env HFX_PLUGIN_PATHS="pkg.plugins,more.plugins"

Ties into your existing adapters.py:
  - AdapterBase, VenueConfig, AdapterRegistry (mock defaults), load_from_path
"""

from __future__ import annotations

import os
import threading
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

# Optional YAML (used only if present)
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# Pull in your adapter types
from .adapters import ( # type: ignore
    AdapterBase,
    AdapterRegistry as BuiltinAdapterRegistry,
    VenueConfig,
    load_from_path as load_adapter_from_path,
)

# ---------------------------------------------------------------------
# Generic Registry (thread-safe)
# ---------------------------------------------------------------------

class Registry:
    """
    Minimal, thread-safe registry for named objects/classes/factories.

    - register(name, obj, *, alias=None)
    - get(name) -> object
    - create(name, **kwargs) -> instance (if stored value is class/callable)
    - alias(existing, as_name)
    - all() -> shallow copy of mapping
    """

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.RLock()
        self._items: Dict[str, Any] = {}

    def register(self, name: str, obj: Any, *, alias: Optional[str] = None) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError(f"{self.name}: empty key not allowed")
        with self._lock:
            self._items[key] = obj
            if alias:
                self._items[str(alias).strip()] = obj

    def alias(self, existing: str, as_name: str) -> None:
        with self._lock:
            if existing not in self._items:
                raise KeyError(f"{self.name}: cannot alias missing item '{existing}'")
            self._items[str(as_name).strip()] = self._items[existing]

    def get(self, name: str) -> Any:
        key = str(name).strip()
        with self._lock:
            try:
                return self._items[key]
            except KeyError:
                raise KeyError(f"{self.name}: '{name}' not found")

    def maybe_get(self, name: Optional[str], default: Any = None) -> Any:
        if not name:
            return default
        try:
            return self.get(name)
        except KeyError:
            return default

    def create(self, name: str, **kwargs) -> Any:
        """
        If the stored item is a class or callable factory, instantiate it.
        Otherwise return the object as-is.
        """
        obj = self.get(name)
        if isinstance(obj, type):
            return obj(**kwargs)
        if callable(obj):
            return obj(**kwargs)
        return obj

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._items)

# ---------------------------------------------------------------------
# Registry Hub
# ---------------------------------------------------------------------

@dataclass
class RegistryHub:
    adapters: Registry
    strategies: Registry
    cost_models: Registry
    risk_checks: Registry
    routers: Registry

    @classmethod
    def default(cls) -> "RegistryHub":
        hub = cls(
            adapters=Registry("adapters"),
            strategies=Registry("strategies"),
            cost_models=Registry("cost_models"),
            risk_checks=Registry("risk_checks"),
            routers=Registry("routers"),
        )
        # Preload built-in adapters from AdapterRegistry mocks
        for vid, adapter in BuiltinAdapterRegistry.all().items():
            hub.adapters.register(vid, adapter)
        return hub

# Process-global hub
HUB = RegistryHub.default()

# ---------------------------------------------------------------------
# Dynamic import helpers
# ---------------------------------------------------------------------

def import_object(path: str) -> Any:
    """
    Import an object by dotted path:
      "package.module:ClassName"
      "package.module:function_name"
    If no colon provided, returns the module.
    """
    mod_path, sep, obj_name = path.partition(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, obj_name) if sep and obj_name else mod

# ---------------------------------------------------------------------
# Adapters from YAML
# ---------------------------------------------------------------------

def _mk_venue_config(node: Dict[str, Any]) -> VenueConfig:
    return VenueConfig(
        id=str(node["id"]),
        name=str(node.get("name", node["id"])),
        type=str(node.get("type", "equities")),
        region=str(node.get("region", "US")),
        base_currency=str(node.get("base_currency", "USD")),
        maker_fee_bps=float(node.get("maker_fee_bps", 0.0)),
        taker_fee_bps=float(node.get("taker_fee_bps", 0.0)),
        min_order_size=float(node.get("min_order_size", 0.0)),
        max_order_size=float(node.get("max_order_size", 1e12)),
        avg_latency_ms=int(node.get("avg_latency_ms", 0)),
    )

def load_adapters_from_yaml(yaml_path: str, *, use_yaml_adapters_field: bool = True) -> Dict[str, AdapterBase]:
    """
    Load/instantiate adapters defined in venues.yaml.

    YAML:
      venues:
        - id: "BINANCE"
          ...
          adapter: "plugins.exec.ccxt:BinanceAdapter"   # optional dotted path
    """
    if not _HAVE_YAML:
        raise RuntimeError("pyyaml is not installed; cannot parse venues.yaml")

    with open(yaml_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    venues = doc.get("venues") or []
    out: Dict[str, AdapterBase] = {}

    for v in venues:
        vid = str(v["id"]).upper()
        cfg = _mk_venue_config(v)

        adapter: Optional[AdapterBase] = None

        # 1) Dotted path in YAML (if any)
        dotted = v.get("adapter") if use_yaml_adapters_field else None
        if dotted:
            try:
                mod_path, sep, cls_name = str(dotted).partition(":")
                if sep:
                    cls = import_object(dotted)
                    if not isinstance(cls, type):
                        raise TypeError(f"{dotted} did not resolve to a class")
                    adapter = cls(cfg)  # type: ignore[call-arg]
                else:
                    adapter = load_adapter_from_path(dotted, cfg)
            except Exception:
                adapter = None  # fall back

        # 2) Built-in mocks fallback
        if adapter is None:
            try:
                adapter = BuiltinAdapterRegistry.get(vid)
            except Exception as e:
                raise RuntimeError(f"Could not build adapter for venue '{vid}': {e}")

        HUB.adapters.register(vid, adapter)
        out[vid] = adapter

    return out

# ---------------------------------------------------------------------
# Plugin autoload
# ---------------------------------------------------------------------

def autoload_plugins(env_var: str = "HFX_PLUGIN_PATHS") -> None:
    """
    Import comma-separated dotted modules. Each module can call
    HUB.<namespace>.register(...) at import time.
    """
    paths = (os.getenv(env_var) or "").strip()
    if not paths:
        return
    for mod in [p.strip() for p in paths.split(",") if p.strip()]:
        try:
            importlib.import_module(mod)
        except Exception:
            # Keep startup resilient; plugin errors shouldn't crash the core
            continue

# ---------------------------------------------------------------------
# Sugar decorators (optional)
# ---------------------------------------------------------------------

def register_strategy(name: str) -> Callable[[Any], Any]:
    def _dec(obj: Any) -> Any:
        HUB.strategies.register(name, obj)
        return obj
    return _dec

def register_cost_model(name: str) -> Callable[[Any], Any]:
    def _dec(obj: Any) -> Any:
        HUB.cost_models.register(name, obj)
        return obj
    return _dec

def register_risk_check(name: str) -> Callable[[Any], Any]:
    def _dec(obj: Any) -> Any:
        HUB.risk_checks.register(name, obj)
        return obj
    return _dec

def register_router(name: str) -> Callable[[Any], Any]:
    def _dec(obj: Any) -> Any:
        HUB.routers.register(name, obj)
        return obj
    return _dec

# ---------------------------------------------------------------------
# Tiny smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Try autoload (no-op if env not set)
    autoload_plugins()

    # Show built-ins that were pulled from AdapterRegistry
    print("Adapters in HUB:", sorted(HUB.adapters.all().keys()))

    # If you have a venues.yaml, demonstrate the YAML loader:
    path = os.getenv("VENUES_YAML_DEMO")
    if path and os.path.isfile(path):
        adapters = load_adapters_from_yaml(path)
        print("Loaded from YAML:", sorted(adapters.keys()))