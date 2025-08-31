# backend/engine/allocators/adaptive_ensemble.py
from __future__ import annotations

import os
import json
import time
import math
import signal
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import redis

# Your existing Redis/stream helpers
from backend.bus.streams import hset, publish_stream

# ---------- Environment / Defaults ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# How often to recompute weights (ms)
ALLOCATOR_INTERVAL_MS = int(os.getenv("ALLOCATOR_INTERVAL_MS", "2000"))

# Weight smoothing (EWMA): new = beta*old + (1-beta)*raw
SMOOTHING_BETA = float(os.getenv("ALLOCATOR_SMOOTHING_BETA", "0.70"))

# Risk/penalties
VOL_FLOOR = float(os.getenv("ALLOCATOR_VOL_FLOOR", "1e-4"))           # avoid div-by-zero
MAX_DRAWDOWN = float(os.getenv("ALLOCATOR_MAX_DRAWDOWN", "0.20"))     # 20% soft cutoff
DRAWDOWN_HARD_CUTOFF = float(os.getenv("ALLOCATOR_DD_HARD", "0.35"))  # >=35% -> hard zero
MIN_SIGNAL_ABS = float(os.getenv("ALLOCATOR_MIN_SIGNAL_ABS", "0.05")) # ignore tiny noise

# Weight caps & options
MIN_WEIGHT = float(os.getenv("ALLOCATOR_MIN_WEIGHT", "0.00"))
MAX_WEIGHT = float(os.getenv("ALLOCATOR_MAX_WEIGHT", "0.35"))
ALLOW_NEGATIVE_WEIGHTS = os.getenv("ALLOCATOR_ALLOW_NEG", "false").lower() == "true"

# Inclusion/exclusion lists (CSV of names); empty => all
INCLUDE = {s for s in os.getenv("ALLOCATOR_INCLUDE", "").split(",") if s.strip()}
EXCLUDE = {s for s in os.getenv("ALLOCATOR_EXCLUDE", "").split(",") if s.strip()}

# Kill switch (if true, allocator sets all weights to 0)
KILL_SWITCH_KEY = os.getenv("KILL_SWITCH_KEY", "policy:kill_switch")

# Redis keys where strategies store metrics (as in your Strategy.emit_* API)
HKEY_SIGNAL   = os.getenv("STRAT_SIGNAL_KEY",   "strategy:signal")   # field=name, value={"score":x}
HKEY_VOL      = os.getenv("STRAT_VOL_KEY",      "strategy:vol")      # field=name, value={"vol":x}
HKEY_DRAWDOWN = os.getenv("STRAT_DD_KEY",       "strategy:drawdown") # field=name, value={"dd":x}

# Where we publish allocator output
HKEY_WEIGHTS  = os.getenv("ALLOCATOR_WEIGHTS_KEY", "strategy:weight")     # hset <name> {"w": x}
STREAM_WEIGHTS = os.getenv("ALLOCATOR_STREAM",     "allocator.weights")   # xadd events for history/UI

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


@dataclass
class StratMetrics:
    name: str
    signal: float = 0.0   # [-1, +1], where sign => direction/conviction
    vol: float = 0.0      # risk proxy (stdev or similar)
    dd: float = 0.0       # drawdown fraction [0,1]
    old_w: float = 0.0    # last weight (for smoothing)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _parse_field_value(v: Any) -> Dict[str, Any]:
    """
    Values in the hashes may be dict or JSON string; be tolerant.
    """
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            # Could be plain number stored as string
            try:
                f = float(v)
                return {"value": f}
            except Exception:
                return {}
    return {}


def _load_hash_as_dict(key: str) -> Dict[str, Dict[str, Any]]:
    d: Dict[str, Dict[str, Any]] = {}
    try:
        raw = r.hgetall(key) or {}
        for k, v in raw.items(): # type: ignore
            d[k] = _parse_field_value(v)
    except Exception:
        pass
    return d


def _kill_switch_enabled() -> bool:
    try:
        # Expect something like HGET policy:kill_switch enabled -> "true"/"false"
        val = r.hget(KILL_SWITCH_KEY, "enabled")
        if val is None:
            return False
        return str(val).lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


class AdaptiveEnsembleAllocator:
    """
    Reads strategy metrics from Redis hashes and computes portfolio weights that:
      - favor higher signal and lower risk (vol)
      - penalize drawdown
      - respect min/max caps
      - smooth over time to avoid churn
      - (optional) allow negative weights if ALLOW_NEGATIVE_WEIGHTS=true

    Writes results to:
      - HSET strategy:weight <name> {"w": weight}
      - XADD allocator.weights {...} for audit/visualization
    """

    def __init__(self) -> None:
        self._running = False
        self._last_weights: Dict[str, float] = {}  # for smoothing

    # ---------- Core ----------
    def _gather(self) -> Dict[str, StratMetrics]:
        sigs = _load_hash_as_dict(HKEY_SIGNAL)
        vols = _load_hash_as_dict(HKEY_VOL)
        dds  = _load_hash_as_dict(HKEY_DRAWDOWN)

        names = set(sigs) | set(vols) | set(dds)
        metrics: Dict[str, StratMetrics] = {}

        for name in names:
            if name in EXCLUDE:
                continue
            if INCLUDE and name not in INCLUDE:
                continue

            s = float(sigs.get(name, {}).get("score", sigs.get(name, {}).get("value", 0.0)) or 0.0)
            v = float(vols.get(name, {}).get("vol", vols.get(name, {}).get("value", 0.0)) or 0.0)
            d = float(dds.get(name,  {}).get("dd",  dds.get(name,  {}).get("value", 0.0)) or 0.0)

            m = StratMetrics(
                name=name,
                signal=float(_clamp(s, -1.0, 1.0)),
                vol=float(max(VOL_FLOOR, v)),
                dd=float(_clamp(d, 0.0, 1.0)),
                old_w=float(self._last_weights.get(name, 0.0)),
            )
            metrics[name] = m

        return metrics

    def _raw_weight(self, m: StratMetrics) -> float:
        """
        Risk-adjusted + drawdown-penalized weight signal.
        - Base: signal / vol
        - Soft penalty as dd approaches MAX_DRAWDOWN
        - Hard zero if dd >= DRAWDOWN_HARD_CUTOFF
        - Ignore tiny signals
        """
        if m.dd >= DRAWDOWN_HARD_CUTOFF:
            return 0.0

        if abs(m.signal) < MIN_SIGNAL_ABS:
            return 0.0

        base = m.signal / m.vol  # risk-aware conviction

        # Soft penalty: linear dropoff from 0..MAX_DRAWDOWN (no penalty below 0)
        if m.dd > 0.0:
            penalty = _clamp(1.0 - (m.dd / max(1e-9, MAX_DRAWDOWN)), 0.0, 1.0)
            base *= penalty

        if not ALLOW_NEGATIVE_WEIGHTS:
            base = max(0.0, base)

        # Cap per-strategy before normalization
        return _clamp(base, -MAX_WEIGHT, MAX_WEIGHT)

    def _normalize(self, raw: Dict[str, float]) -> Dict[str, float]:
        """
        L1 normalization to sum of absolute weights <= 1.
        If all zero, return zeros.
        """
        denom = sum(abs(x) for x in raw.values())
        if denom <= 0.0:
            return {k: 0.0 for k in raw}

        # Target total weight = 1.0
        out = {k: _clamp(x / denom, -MAX_WEIGHT, MAX_WEIGHT) for k, x in raw.items()}
        # Enforce MIN_WEIGHT floor for nonzero entries (only if positive weights allowed)
        if not ALLOW_NEGATIVE_WEIGHTS and MIN_WEIGHT > 0.0:
            # Raise small non-zero weights to MIN_WEIGHT, renormalize
            keys = [k for k, v in out.items() if v > 0.0]
            for k in keys:
                if out[k] < MIN_WEIGHT:
                    out[k] = MIN_WEIGHT
            denom2 = sum(out[k] for k in keys)
            if denom2 > 0:
                for k in keys:
                    out[k] /= denom2
        return out

    def _smooth(self, new: Dict[str, float]) -> Dict[str, float]:
        """
        EWMA smoothing to avoid churn.
        """
        beta = _clamp(SMOOTHING_BETA, 0.0, 0.99)
        sm: Dict[str, float] = {}
        for k, v in new.items():
            prev = self._last_weights.get(k, 0.0)
            sm[k] = beta * prev + (1.0 - beta) * v
        # also decay removed strategies to zero over time
        for k in list(self._last_weights.keys()):
            if k not in sm:
                sm[k] = beta * self._last_weights[k]
        return sm

    def compute_weights(self) -> Dict[str, float]:
        """
        Pull metrics -> raw weights -> normalize -> smooth -> clamp.
        """
        if _kill_switch_enabled():
            # Hard zero across the board
            weights = {k: 0.0 for k in self._last_weights.keys()}
            return weights

        met = self._gather()
        if not met:
            return {k: 0.0 for k in self._last_weights.keys()}

        raw = {name: self._raw_weight(m) for name, m in met.items()}
        norm = self._normalize(raw)
        sm = self._smooth(norm)

        # Final clamp & small-value cleanup
        final = {k: (0.0 if abs(v) < 1e-6 else _clamp(v, -MAX_WEIGHT, MAX_WEIGHT)) for k, v in sm.items()}
        self._last_weights = final.copy()
        return final

    # ---------- IO ----------
    def publish(self, weights: Dict[str, float]) -> None:
        ts = int(time.time() * 1000)
        # Write HSET for current weights (UI & other services read this)
        for name, w in weights.items():
            hset(HKEY_WEIGHTS, name, {"w": float(w), "ts": ts})

        # Append to stream for history/audit
        payload = {"ts_ms": ts, "weights": json.dumps(weights)}
        publish_stream(STREAM_WEIGHTS, payload)

    # ---------- Runner ----------
    def _graceful_stop(self, *_):
        self._running = False

    def run_forever(self) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._graceful_stop)
        signal.signal(signal.SIGTERM, self._graceful_stop)

        interval = max(100, ALLOCATOR_INTERVAL_MS) / 1000.0
        while self._running:
            try:
                w = self.compute_weights()
                self.publish(w)
            except Exception as e:
                # Best-effort alert; avoid crash-loop
                try:
                    publish_stream("alerts", {
                        "ts_ms": int(time.time()*1000),
                        "lvl": "error",
                        "src": "allocator",
                        "msg": "compute_or_publish_failed",
                        "err": str(e)
                    })
                except Exception:
                    pass
            time.sleep(interval)


# ---------- CLI ----------
if __name__ == "__main__":
    """
    Example:
      $ python -m backend.engine.allocators.adaptive_ensemble
    Env you might tweak:
      ALLOCATOR_INTERVAL_MS=1000 ALLOCATOR_SMOOTHING_BETA=0.6 \
      ALLOCATOR_MAX_WEIGHT=0.4 ALLOCATOR_ALLOW_NEG=true \
      ALLOCATOR_INCLUDE="alpha_momo,alpha_meanrev"
    """
    allocator = AdaptiveEnsembleAllocator()
    allocator.run_forever()