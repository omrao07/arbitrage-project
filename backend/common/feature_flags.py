"""
Feature Flags Registry
======================
Central switchboard for experimental + critical features.
- Default values are OFF (False).
- Override with ENV (FEATURE_GOVERNOR=1) or config YAML.
"""

import os
import yaml
from pathlib import Path
from typing import Dict

# --- Default flags (safe OFF in live) ---
DEFAULT_FLAGS: Dict[str, bool] = {
    # Intelligence / Swarm
    "governor": False,             # Meta-learning governor
    "adversary": False,            # Adversarial market generator
    "evolver": False,              # Self-evolving strategies (GA)
    "competitors": False,          # Rival hedge funds (game-theory)
    "ai_centralbank": False,       # AI central bank duel

    # Macro / Systemic Risk
    "sovereign": False,            # Sovereign default contagion
    "bank_stress": False,          # Bank balance sheet shocks
    "capital_flows": False,        # Cross-border capital flows
    "liquidity_spiral": False,     # Fire-sale liquidity spiral

    # Microstructure
    "latency": False,              # Venue latency model
    "liquidity_surface": False,    # Dynamic spread/depth
    "dark_pool": False,            # Hidden liquidity venues

    # Portfolio Science
    "hrp": False,                  # Hierarchical Risk Parity
    "stress_attribution": False,   # Stress→PnL attribution
    "contagion_graph": False,      # Risk contagion graph

    # Alt-Data
    "geo_spatial": False,          # Satellite/AIS commodity arb
    "corp_exhaust": False,         # Hiring/supply-chain exhaust
    "mood_index": False,           # Multi-social sentiment index

    # Explainers / Dashboards
    "pnl_xray": False,             # Attribution explainer
    "crisis_theatre": False,       # Scenario replay storyteller
}

# --- Global state (mutable at runtime) ---
_flags: Dict[str, bool] = DEFAULT_FLAGS.copy()

# --- Load from YAML config (optional) ---
def load_from_yaml(path: str = "config/feature_flags.yaml") -> None:
    p = Path(path)
    if not p.exists(): return
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    for k,v in data.items():
        if k in _flags: _flags[k] = bool(v)

# --- Load from ENV ---
def load_from_env() -> None:
    for k in _flags.keys():
        env_key = f"FEATURE_{k.upper()}"
        if env_key in os.environ:
            _flags[k] = os.environ[env_key] in ("1","true","True","yes")

# --- Public API ---
def enabled(name: str) -> bool:
    return _flags.get(name, False)

def set_flag(name: str, val: bool) -> None:
    if name not in _flags: raise KeyError(f"Unknown flag {name}")
    _flags[name] = val

def all_flags() -> Dict[str,bool]:
    return dict(_flags)

# --- Init at import ---
load_from_env()
load_from_yaml()