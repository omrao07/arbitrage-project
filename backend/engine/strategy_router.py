# backend/engine/strategy_router.py
"""
Strategy Router
- Dynamically loads strategies from backend/strategies/catalog.yaml
- Fans ticks to all enabled strategies (optionally region-filtered)
- Exposes route_tick(tick) used by aggregator
- Provides reload_strategies() to apply catalog changes at runtime

Catalog schema (backend/strategies/catalog.yaml) example:
---------------------------------------------------------
strategies:
  - name: example_buy_dip
    module: backend.engine.strategy_base
    class: ExampleBuyTheDip
    region: CRYPTO         # optional; if set, only receives ticks from that region
    default_qty: 0.001
    params:
      bps: 10.0
  - name: us_mean_reversion
    module: backend.strategies.alpha.us_mean_reversion
    class: USMeanReversion
    region: US
    default_qty: 10
    params:
      lookback: 50
      z_entry: 2.0
---------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
import yaml

from backend.engine.strategy_base import Strategy
from backend.engine.region_router import infer_region  # for optional region filtering

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
log = logging.getLogger("strategy_router")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_FILE = REPO_ROOT / "backend" / "strategies" / "catalog.yaml"

# Redis (for enable/disable flags)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# In‑memory registry
_STRATS: Dict[str, Strategy] = {}
_META: Dict[str, Dict[str, Any]] = {}  # name -> {region, ...}

def _load_catalog() -> List[Dict[str, Any]]:
    if not CATALOG_FILE.exists():
        log.warning("Strategy catalog file not found: %s", CATALOG_FILE)
        return []
    with open(CATALOG_FILE, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    return list(cfg.get("strategies", []))

def _dyn_import(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls

def _is_enabled(name: str) -> bool:
    # default: enabled unless explicitly disabled
    v = r.hget("strategy:enabled", name)
    return False if v and str(v).lower() in ("0", "false", "no") else True

def _want_region(name: str, strategy_region: Optional[str], tick: Dict[str, Any]) -> bool:
    """Return True if this strategy should receive this tick based on optional region filter."""
    if not strategy_region:
        return True
    # Infer region from tick symbol/venue (fallback to US)
    symbol = str(tick.get("symbol") or tick.get("s") or "")
    venue = tick.get("venue")
    tick_region = infer_region(symbol, venue)
    return tick_region.upper() == str(strategy_region).upper()

def reload_strategies() -> List[str]:
    """
    Load/refresh strategies from catalog.yaml.
    Keeps already-created instances if unchanged; recreates when definition changed.
    Returns list of active strategy names.
    """
    global _STRATS, _META
    defs = _load_catalog()
    if not defs:
        log.warning("No strategies defined in catalog.")
        _STRATS.clear()
        _META.clear()
        return []

    active_names: List[str] = []
    next_registry: Dict[str, Strategy] = {}
    next_meta: Dict[str, Dict[str, Any]] = {}

    for spec in defs:
        try:
            name = str(spec["name"])
            module = str(spec["module"])
            clsname = str(spec["class"])
            region = spec.get("region")  # optional
            default_qty = float(spec.get("default_qty", 1.0))
            params = dict(spec.get("params", {}))

            # Reuse instance if same module/class and we already have it
            reuse = False
            if name in _META:
                old = _META[name]
                if old.get("module") == module and old.get("class") == clsname and old.get("params") == params and old.get("default_qty") == default_qty and old.get("region") == region:
                    next_registry[name] = _STRATS[name]
                    next_meta[name] = old
                    active_names.append(name)
                    reuse = True

            if not reuse:
                cls = _dyn_import(module, clsname)
                if not issubclass(cls, Strategy):
                    log.warning("Strategy %s is not subclass of Strategy; skipping.", name)
                    continue
                inst: Strategy = cls(name=name, region=region, default_qty=default_qty, **params)
                next_registry[name] = inst
                next_meta[name] = {"module": module, "class": clsname, "params": params, "default_qty": default_qty, "region": region}
                active_names.append(name)
        except Exception as e:
            log.exception("Failed to load strategy spec %s: %s", spec, e)

    _STRATS = next_registry
    _META = next_meta
    log.info("Loaded %d strategies: %s", len(_STRATS), ", ".join(sorted(_STRATS.keys())))
    return active_names

# Initial load on import
reload_strategies()

def set_enabled(name: str, enabled: bool) -> None:
    r.hset("strategy:enabled", name, "true" if enabled else "false")

def route_tick(tick: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Fan-out a single tick to all enabled strategies (region-filtered).
    Returns a minimal summary (no need to return orders; strategies submit to risk bus themselves).
    """
    if not _STRATS:
        # attempt lazy reload if catalog was added later
        reload_strategies()
        if not _STRATS:
            return None

    delivered: List[str] = []
    for name, strat in _STRATS.items():
        try:
            if not _is_enabled(name):
                continue
            meta = _META.get(name, {})
            if not _want_region(name, meta.get("region"), tick):
                continue
            strat.on_tick(tick)  # strategies handle their own order() emissions
            delivered.append(name)
        except Exception as e:
            # Log error; keep routing others
            r.lpush(f"strategy:errors:{name}", json.dumps({"ts": tick.get("ts_ms"), "err": str(e)}))
    return [{"delivered_to": delivered, "count": len(delivered)}]

# Optional helper: manual trigger to reload catalog at runtime
def hot_reload() -> Dict[str, Any]:
    names = reload_strategies()
    return {"reloaded": names, "count": len(names)}

if __name__ == "__main__":
    # Simple test: fan a fake crypto tick
    fake = {"ts_ms": 0, "symbol": "BTCUSDT", "price": 50000.0, "venue": "BINANCE"}
    print(hot_reload())
    print(route_tick(fake))