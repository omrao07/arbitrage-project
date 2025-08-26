# backend/config/feature_flags.py
"""
Feature flags for Hedge Fund X.

Priority (highest -> lowest):
1) Environment variables (e.g., FEATURE_ROBO_PM=true)
2) feature_flags.yaml (optional file-based overrides for dev/stage)
3) Hardcoded defaults below

Usage:
    from backend.config.feature_flags import (
        is_enabled, flags, feature_required, reload_flags
    )

    if is_enabled("ROUTER"):
        from backend.execution_plus.arb_router.router import GlobalRouter
        ...

    @feature_required("CHAOS")
    def run_chaos_scenarios():
        ...

YAML example (backend/config/feature_flags.yaml):
-------------------------------------------------
ROBO_PM: true
SWARM: true
NEUROSYM: true
PERSONAS: true
ALTDATA: true
SENTIMENT: true
Biodata: false   # case-insensitive
CLIMATE: true
ROUTER: true
SYNTH_ASSETS: true
CHAOS: true
SELF_HEAL: true
QUANTUM: false
SANDBOX: true
CBANK_AI: true
"""

from __future__ import annotations

import os
import threading
from functools import wraps
from typing import Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML overrides will be skipped if pyyaml not installed

# ------------------------------------------------------------------------------
# Defaults (lowest priority)
# ------------------------------------------------------------------------------

_DEFAULTS: Dict[str, bool] = {
    "ROBO_PM": True,
    "SWARM": True,
    "NEUROSYM": True,
    "PERSONAS": True,
    "ALTDATA": True,
    "SENTIMENT": True,
    "BIODATA": False,
    "CLIMATE": True,
    "ROUTER": True,
    "SYNTH_ASSETS": True,
    "CHAOS": True,
    "SELF_HEAL": True,
    "QUANTUM": False,
    "SANDBOX": True,
    "CBANK_AI": True,
}

# Map env var names: FEATURE_<NAME>=true/false
_ENV_PREFIX = "FEATURE_"

# Optional YAML file path
_YAML_PATH = os.getenv(
    "FEATURE_FLAGS_YAML",
    os.path.join(os.path.dirname(__file__), "feature_flags.yaml"),
)

# ------------------------------------------------------------------------------
# Internal: load & merge
# ------------------------------------------------------------------------------

_flags_lock = threading.RLock()
_flags_cache: Optional[Dict[str, bool]] = None


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _load_yaml_file() -> Dict[str, bool]:
    if not yaml:
        return {}
    try:
        if os.path.isfile(_YAML_PATH):
            with open(_YAML_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Normalize keys to UPPER_SNAKE
            return {str(k).upper(): _to_bool(v) for k, v in data.items()}
    except Exception:
        # Silent fallback to no YAML overrides
        pass
    return {}


def _load_env_overrides() -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for key, val in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            name = key[len(_ENV_PREFIX):].upper()
            out[name] = _to_bool(val)
    return out


def _build_flags() -> Dict[str, bool]:
    merged = dict(_DEFAULTS)

    # YAML (medium priority)
    y = _load_yaml_file()
    for k, v in y.items():
        if k in merged:
            merged[k] = v
        else:
            # allow new keys via YAML too
            merged[k] = v

    # ENV (highest priority)
    e = _load_env_overrides()
    for k, v in e.items():
        merged[k] = v

    return merged


def reload_flags() -> None:
    """Recompute the flag cache (call after changing env/yaml)."""
    global _flags_cache
    with _flags_lock:
        _flags_cache = _build_flags()


def flags() -> Dict[str, bool]:
    """Get a copy of the current flag map."""
    global _flags_cache
    with _flags_lock:
        if _flags_cache is None:
            _flags_cache = _build_flags()
        return dict(_flags_cache)


def is_enabled(name: str) -> bool:
    """Check if a feature is enabled. Case-insensitive."""
    return flags().get(name.upper(), False)


# ------------------------------------------------------------------------------
# Convenience decorator
# ------------------------------------------------------------------------------

class FeatureDisabledError(RuntimeError):
    pass


def feature_required(name: str):
    """
    Decorator to guard functions behind a feature flag.

    Example:
        @feature_required("ROUTER")
        def compute_routes(...):
            ...
    """
    key = name.upper()

    def _decorator(fn):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            if not is_enabled(key):
                raise FeatureDisabledError(f"Feature '{key}' is disabled")
            return fn(*args, **kwargs)

        return _wrapped

    return _decorator