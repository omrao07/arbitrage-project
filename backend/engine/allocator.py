# backend/engine/allocator.py
"""
Allocator for multi-strategy portfolio:
- Computes target weights per strategy using risk parity + vol targeting + signal tilt
- Enforces per-strategy and global caps
- Applies drawdown guard
- Persists targets to Redis (for OMS/risk/dashboard)

Inputs expected via Redis keys:
  hgetall("strategy:vol")            -> {name: json({"vol": 0.12})}   # annualized or daily stdev (consistent scale)
  hgetall("strategy:signal")         -> {name: json({"score": 0.8})}  # -1..+1 typical
  hgetall("strategy:drawdown")       -> {name: json({"dd": 0.08})}    # 0..1 (8% = 0.08)
  hgetall("strategy:enabled")        -> {name: "true"/"false"}        # toggle
  get("portfolio:capital_base")      -> "100000.0"                    # USD notional

Writes:
  hset("allocator:weights", name, {"w": float})
  hset("allocator:notional", name, {"usd": float})
  set("allocator:last_run_ts", {...})

You can call allocate() on a schedule (e.g., every minute) or on demand.
"""

from __future__ import annotations
import json, time, math, os
import redis
from typing import Dict, Tuple

# --- Config (can be moved to settings.py) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Target portfolio risk (volatility targeting) in annualized terms
TARGET_VOL = float(os.getenv("ALLOC_TARGET_VOL", "0.12"))         # 12% annualized target
MIN_VOL_FLOOR = float(os.getenv("ALLOC_MIN_VOL_FLOOR", "0.02"))   # avoid division by tiny vol
MAX_WEIGHT_PER_STRAT = float(os.getenv("ALLOC_MAX_W", "0.15"))    # 15% cap
GLOBAL_GROSS_CAP = float(os.getenv("ALLOC_GROSS_CAP", "1.0"))     # <= 1.0 of capital base
DRAWDOWN_CUTOFF = float(os.getenv("ALLOC_DD_CUTOFF", "0.10"))     # 10% DD -> strong de-weight
DRAWDOWN_MIN_FACTOR = float(os.getenv("ALLOC_DD_MINF", "0.25"))   # min keep 25% of computed weight
SIGNAL_TILT = float(os.getenv("ALLOC_SIGNAL_TILT", "0.5"))        # 0..1; 0 = ignore signal, 1 = full tilt

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ---- Helpers ----
def _hget_json_map(key: str) -> Dict[str, dict]:
    raw = r.hgetall(key) or {}
    out: Dict[str, dict] = {}
    for k, v in raw.items(): # type: ignore
        try:
            out[k] = json.loads(v) if isinstance(v, str) else v
        except Exception:
            out[k] = {"value": v}
    return out

def _get_capital_base() -> float:
    val = r.get("portfolio:capital_base")
    try:
        return float(val) if val is not None else 100_000.0 # type: ignore
    except Exception:
        return 100_000.0

def _safe_vol(x: float) -> float:
    return max(abs(x), MIN_VOL_FLOOR)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---- Core allocation logic ----
def _risk_parity_weights(vol_map: Dict[str, float]) -> Dict[str, float]:
    inv_vol = {k: 1.0 / _safe_vol(v) for k, v in vol_map.items()}
    s = sum(inv_vol.values()) or 1.0
    return {k: v / s for k, v in inv_vol.items()}

def _apply_signal_tilt(w: Dict[str, float], signals: Dict[str, float]) -> Dict[str, float]:
    if SIGNAL_TILT <= 1e-6:
        return w
    # Normalize signal to [0,1] and tilt weights
    # s_norm = (s + 1)/2 for s in [-1,1]
    tilted = {}
    for k, base_w in w.items():
        s = signals.get(k, 0.0)
        s_norm = (s + 1.0) * 0.5
        # convex combo between equal weight (or base_w) and signal emphasis
        tilt = (1 - SIGNAL_TILT) * base_w + SIGNAL_TILT * (base_w * (0.5 + s_norm))
        tilted[k] = tilt
    # renormalize to 1
    s = sum(tilted.values()) or 1.0
    return {k: v / s for k, v in tilted.items()}

def _apply_drawdown_guard(w: Dict[str, float], dd_map: Dict[str, float]) -> Dict[str, float]:
    guarded = {}
    for k, base_w in w.items():
        dd = dd_map.get(k, 0.0)
        if dd <= DRAWDOWN_CUTOFF:
            guarded[k] = base_w
        else:
            # linearly scale down to a minimum factor
            # e.g., 10% dd -> base_w * DRAWDOWN_MIN_FACTOR
            scale = _clamp(1.0 - (dd - DRAWDOWN_CUTOFF) / max(1e-9, 1.0 - DRAWDOWN_CUTOFF), DRAWDOWN_MIN_FACTOR, 1.0)
            guarded[k] = max(base_w * scale, base_w * DRAWDOWN_MIN_FACTOR)
    # renormalize
    s = sum(guarded.values()) or 1.0
    return {k: v / s for k, v in guarded.items()}

def _apply_caps(w: Dict[str, float]) -> Dict[str, float]:
    # Cap each strategy and renormalize if needed
    capped = {k: min(v, MAX_WEIGHT_PER_STRAT) for k, v in w.items()}
    s = sum(capped.values())
    if s == 0:
        return w
    if s > 1.0:
        # renormalize down to 1.0 (pre gross cap)
        capped = {k: v / s for k, v in capped.items()}
    return capped

def _apply_global_gross_cap(w: Dict[str, float]) -> Dict[str, float]:
    # Scale weights so total <= GLOBAL_GROSS_CAP
    s = sum(w.values()) or 1.0
    if s > GLOBAL_GROSS_CAP:
        scale = GLOBAL_GROSS_CAP / s
        return {k: v * scale for k, v in w.items()}
    return w

def _volatility_target_scale(current_port_vol: float) -> float:
    # If you later compute realized portfolio vol, plug here.
    # For now, return 1.0 (placeholder to fit into pipeline).
    return 1.0

# ---- Public API ----
def allocate() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute target weights and implied notionals per strategy.
    Returns: (weights, notionals)
    """
    # Enabled set
    enabled = r.hgetall("strategy:enabled") or {}
    enabled_names = {k for k, v in enabled.items() if str(v).lower() in ("1", "true", "yes")} # type: ignore

    # Load inputs
    vol3 = _hget_json_map("strategy:vol")        # {name: {"vol": x}}
    sig3 = _hget_json_map("strategy:signal")     # {name: {"score": x}}
    dd3  = _hget_json_map("strategy:drawdown")   # {name: {"dd": x}}

    # Build numeric maps for enabled strategies only
    vols = {k: float(v.get("vol", 0.1)) for k, v in vol3.items() if not enabled_names or k in enabled_names}
    sigs = {k: float(v.get("score", 0.0)) for k, v in sig3.items() if not enabled_names or k in enabled_names}
    dds  = {k: float(v.get("dd", 0.0)) for k, v in dd3.items() if not enabled_names or k in enabled_names}

    if not vols:
        return {}, {}

    # 1) Risk parity
    w = _risk_parity_weights(vols)
    # 2) Signal tilt
    w = _apply_signal_tilt(w, sigs)
    # 3) Drawdown guard
    w = _apply_drawdown_guard(w, dds)
    # 4) Per-strategy caps
    w = _apply_caps(w)
    # 5) Global gross cap
    w = _apply_global_gross_cap(w)
    # 6) Vol targeting scale (hook for realized port vol)
    scale = _volatility_target_scale(current_port_vol=TARGET_VOL)  # placeholder
    w = {k: v * scale for k, v in w.items()}

    # Notionals
    capital = _get_capital_base()
    notionals = {k: v * capital for k, v in w.items()}

    # Persist for OMS/dashboard
    for k, v in w.items():
        r.hset("allocator:weights", k, json.dumps({"w": float(v)}))
    for k, usd in notionals.items():
        r.hset("allocator:notional", k, json.dumps({"usd": float(usd)}))
    r.set("allocator:last_run_ts", json.dumps({"ts": int(time.time())}))

    return w, notionals

def get_weights() -> Dict[str, float]:
    raw = r.hgetall("allocator:weights") or {}
    out = {}
    for k, v in raw.items(): # type: ignore
        try:
            out[k] = float(json.loads(v).get("w", 0.0))
        except Exception:
            out[k] = 0.0
    return out

def get_notionals() -> Dict[str, float]:
    raw = r.hgetall("allocator:notional") or {}
    out = {}
    for k, v in raw.items(): # type: ignore
        try:
            out[k] = float(json.loads(v).get("usd", 0.0))
        except Exception:
            out[k] = 0.0
    return out