# backend/engine/allocators/regime_allocator.py
from __future__ import annotations

import os
import json
import time
import signal
from dataclasses import dataclass
from typing import Dict, Any, Optional

import yaml
import redis

# --- Shared bus helpers (you already have these) -----------------------------
from backend.bus.streams import hset, publish_stream

# ---------- Environment / Defaults ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Where the current regime is written (by your regime detector)
REGIME_STATE_KEY = os.getenv("REGIME_STATE_KEY", "regime:state")     # HGETALL -> {"name": "risk_on", "conf": 0.82}
REGIME_OVERRIDE  = os.getenv("REGIME_OVERRIDE", "").strip()          # force regime name (for testing)

# Optionally provide a policy file that maps regime -> tag weights
REGIME_POLICY_PATH = os.getenv("REGIME_POLICY_PATH", "configs/policy.yaml")

# Strategy metadata / metrics
HKEY_STRAT_META  = os.getenv("STRAT_META_KEY", "strategy:meta")      # field=<name> -> {"tags":["mom","dispersion","options"], "region":"US"}
HKEY_SIGNAL      = os.getenv("STRAT_SIGNAL_KEY", "strategy:signal")  # optional strength gate
HKEY_DRAWDOWN    = os.getenv("STRAT_DD_KEY", "strategy:drawdown")    # dd limit guard

# Output keys/streams
HKEY_WEIGHTS   = os.getenv("ALLOCATOR_WEIGHTS_KEY", "strategy:weight")     # hset <name> {"w": x, "src":"regime", ...}
STREAM_WEIGHTS = os.getenv("ALLOCATOR_STREAM",     "allocator.weights")

# Smoothing / caps
SMOOTHING_BETA = float(os.getenv("REGIME_SMOOTH_BETA", "0.70"))      # EWMA smoothing for stability
MAX_WEIGHT     = float(os.getenv("REGIME_MAX_WEIGHT", "0.35"))
MIN_WEIGHT     = float(os.getenv("REGIME_MIN_WEIGHT", "0.00"))
ALLOW_NEG      = os.getenv("REGIME_ALLOW_NEG", "false").lower() == "true"

# Guards
MAX_DRAWDOWN   = float(os.getenv("REGIME_MAX_DRAWDOWN", "0.35"))     # hard zero above this DD
MIN_SIGNAL_ABS = float(os.getenv("REGIME_MIN_SIGNAL_ABS", "0.00"))   # optional gate
KILL_SWITCH_KEY= os.getenv("KILL_SWITCH_KEY", "policy:kill_switch")

# Inclusion/exclusion filters
INCLUDE = {s for s in os.getenv("REGIME_INCLUDE", "").split(",") if s.strip()}
EXCLUDE = {s for s in os.getenv("REGIME_EXCLUDE", "").split(",") if s.strip()}

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ---------- Policy model -----------------------------------------------------
# Policy YAML structure (example):
# regime_policy:
#   risk_on:
#     base: 1.0          # scales all below
#     tags:
#       momentum: 1.0
#       carry:    0.6
#       value:    0.4
#       dispersion: 0.7
#       options:  0.3
#   risk_off:
#     base: 0.8
#     tags:
#       momentum: 0.1
#       carry:    0.3
#       value:    0.9
#       tail:     1.0
#       options:  0.8
DEFAULT_POLICY = {
    "regime_policy": {
        "risk_on": {
            "base": 1.0,
            "tags": {"momentum": 1.0, "carry": 0.6, "value": 0.4, "dispersion": 0.7, "options": 0.3}
        },
        "risk_off": {
            "base": 0.8,
            "tags": {"momentum": 0.1, "carry": 0.3, "value": 0.9, "tail": 1.0, "options": 0.8}
        },
        "panic": {
            "base": 0.6,
            "tags": {"tail": 1.0, "value": 0.8, "dispersion": 0.2, "momentum": 0.0, "options": 0.9}
        },
        "carry": {
            "base": 1.0,
            "tags": {"carry": 1.0, "value": 0.6, "momentum": 0.3, "options": 0.2}
        },
        "earnings": {
            "base": 1.0,
            "tags": {"dispersion": 0.9, "momentum": 0.6, "options": 0.8, "value": 0.4}
        },
    }
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _kill_switch_on() -> bool:
    try:
        v = r.hget(KILL_SWITCH_KEY, "enabled")
        return str(v).lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _load_yaml_policy(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                y = yaml.safe_load(f) or {}
                if "regime_policy" in y:
                    return y
    except Exception:
        pass
    return DEFAULT_POLICY


def _hgetall_json_tolerant(key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        raw = r.hgetall(key) or {}
        for k, v in raw.items(): # type: ignore
            try:
                out[k] = json.loads(v) if isinstance(v, str) else v
            except Exception:
                out[k] = {"value": v}
    except Exception:
        pass
    return out


@dataclass
class StratRow:
    name: str
    tags: set[str]
    region: Optional[str] = None
    # optional gates:
    signal: Optional[float] = None
    dd: Optional[float] = None
    prev_w: float = 0.0


class RegimeAllocator:
    """
    Map (regime, confidence) × (strategy tags) -> portfolio weights.
    - Pulls regime from Redis (or REGIME_OVERRIDE)
    - Loads strategy tags from HGETALL strategy:meta
    - Uses policy (YAML or default) to derive raw weights
    - Applies guards (drawdown, min signal), caps, smoothing
    - Writes HSET strategy:weight and XADD allocator.weights
    """

    def __init__(self, policy_path: str = REGIME_POLICY_PATH) -> None:
        self.policy = _load_yaml_policy(policy_path)
        self._prev: Dict[str, float] = {}
        self.interval = float(os.getenv("REGIME_ALLOCATOR_INTERVAL", "2.0"))  # seconds

    # --------- data pulls ----------
    def _current_regime(self) -> tuple[str, float]:
        if REGIME_OVERRIDE:
            return REGIME_OVERRIDE, 1.0
        try:
            d = _hgetall_json_tolerant(REGIME_STATE_KEY)
            name = str(d.get("name") or d.get("regime") or "risk_on")
            conf = float(d.get("conf") or d.get("confidence") or 1.0)
            return name, _clamp(conf, 0.0, 1.0)
        except Exception:
            return "risk_on", 1.0

    def _load_strategies(self) -> Dict[str, StratRow]:
        meta = _hgetall_json_tolerant(HKEY_STRAT_META)
        sigs = _hgetall_json_tolerant(HKEY_SIGNAL)
        dds  = _hgetall_json_tolerant(HKEY_DRAWDOWN)

        rows: Dict[str, StratRow] = {}
        for name, m in meta.items():
            if name in EXCLUDE:
                continue
            if INCLUDE and name not in INCLUDE:
                continue
            tags = set(map(str.lower, (m.get("tags") or [])))
            region = m.get("region")
            s = sigs.get(name, {})
            d = dds.get(name, {})
            signal = float(s.get("score", s.get("value", 0.0)) or 0.0)
            dd = float(d.get("dd", d.get("value", 0.0)) or 0.0)
            rows[name] = StratRow(
                name=name,
                tags=tags,
                region=region,
                signal=signal,
                dd=dd,
                prev_w=float(self._prev.get(name, 0.0)),
            )
        return rows

    # --------- policy math ----------
    def _tag_weight(self, regime: str, tag: str) -> float:
        p = self.policy.get("regime_policy", {}).get(regime, {})
        base = float(p.get("base", 1.0))
        tagw = float(p.get("tags", {}).get(tag, 0.0))
        return base * tagw

    def _raw_score(self, regime: str, row: StratRow) -> float:
        if row.dd is not None and row.dd >= MAX_DRAWDOWN:
            return 0.0
        if row.signal is not None and abs(row.signal) < MIN_SIGNAL_ABS:
            # optional: gate out tiny/noisy strats
            return 0.0
        # combine all tags the strategy declares
        score = 0.0
        for tag in row.tags:
            score += self._tag_weight(regime, tag)
        return score

    def _normalize(self, raw: Dict[str, float]) -> Dict[str, float]:
        s = sum(abs(v) for v in raw.values())
        if s <= 0:
            return {k: 0.0 for k in raw}
        out = {k: _clamp(v / s, -MAX_WEIGHT, MAX_WEIGHT) for k, v in raw.items()}
        # ensure a minimum positive floor when negatives are disallowed
        if not ALLOW_NEG and MIN_WEIGHT > 0.0:
            keys = [k for k, v in out.items() if v > 0.0]
            for k in keys:
                if out[k] < MIN_WEIGHT:
                    out[k] = MIN_WEIGHT
            denom = sum(out[k] for k in keys) or 1.0
            for k in keys:
                out[k] = out[k] / denom
            # everything not in keys → zero
            for k, v in out.items():
                if k not in keys:
                    out[k] = 0.0
        return out

    def _smooth(self, new: Dict[str, float]) -> Dict[str, float]:
        beta = _clamp(SMOOTHING_BETA, 0.0, 0.99)
        sm = {k: beta * self._prev.get(k, 0.0) + (1.0 - beta) * v for k, v in new.items()}
        # decay removed names
        for k in list(self._prev.keys()):
            if k not in sm:
                sm[k] = beta * self._prev[k]
        return sm

    # --------- public ----------
    def compute(self) -> Dict[str, float]:
        # kill switch hard zero
        if _kill_switch_on():
            return {k: 0.0 for k in self._prev}

        regime, conf = self._current_regime()
        rows = self._load_strategies()
        if not rows:
            return {k: 0.0 for k in self._prev}

        # 1) raw per-strategy scores from regime policy
        raw = {name: self._raw_score(regime, row) * conf for name, row in rows.items()}

        # 2) normalize to budget, caps, floors
        norm = self._normalize(raw)

        # 3) EWMA smooth to avoid weight churn
        sm = self._smooth(norm)

        # finalize clamp & prune tiny values
        final = {k: (0.0 if abs(v) < 1e-6 else _clamp(v, -MAX_WEIGHT, MAX_WEIGHT)) for k, v in sm.items()}
        self._prev = final.copy()
        return final

    def publish(self, weights: Dict[str, float]) -> None:
        ts = int(time.time() * 1000)
        for name, w in weights.items():
            hset(HKEY_WEIGHTS, name, {"w": float(w), "ts": ts, "src": "regime"})
        publish_stream(STREAM_WEIGHTS, {"ts_ms": ts, "src": "regime", "weights": json.dumps(weights)})

    # --------- runner ----------
    def _stop(self, *_):
        self._running = False

    def run_forever(self) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

        while self._running:
            try:
                w = self.compute()
                self.publish(w)
            except Exception as e:
                try:
                    publish_stream("alerts", {
                        "ts_ms": int(time.time()*1000),
                        "lvl": "error",
                        "src": "regime_allocator",
                        "msg": "compute_or_publish_failed",
                        "err": str(e),
                    })
                except Exception:
                    pass
            time.sleep(self.interval)


# ---------- CLI ----------
if __name__ == "__main__":
    """
    Run:
      $ python -m backend.engine.allocators.regime_allocator
    Env knobs:
      REGIME_OVERRIDE=risk_off REGIME_SMOOTH_BETA=0.6 REGIME_MAX_WEIGHT=0.4
      REGIME_POLICY_PATH=configs/policy.yaml
    """
    RegimeAllocator().run_forever()